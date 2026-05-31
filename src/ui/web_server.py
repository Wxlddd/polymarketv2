import asyncio
import glob
import json
import logging
import os
import time
import datetime
from typing import Dict, Any, Set, Optional, List
from aiohttp import web
import polars as pl
from config.settings import SystemConfig
from src.backtest.runner import BacktestRunner

logger = logging.getLogger("WebServer")

class WebServer:
    """
    High-performance, non-blocking HTTP and WebSocket Server.
    Provides Bloomberg Terminal UI hosting, throttled WS state broadcasting (2-5Hz),
    historical data range discovery, and time-window backtesting.
    """
    def __init__(self, config: SystemConfig, orchestrator: Any):
        self.config = config
        self.orchestrator = orchestrator
        self.host = config.polymarket.__dict__.get("WEB_SERVER_HOST", "localhost")
        self.port = int(config.polymarket.__dict__.get("WEB_SERVER_PORT", 8080))
        
        # UI Broadcast throttling rate (Hz)
        self.throttle_hz = float(config.polymarket.__dict__.get("UI_BROADCAST_THROTTLE_HZ", 4.0))
        self.broadcast_interval = 1.0 / self.throttle_hz
        
        self.app = web.Application()
        self.active_monitors: Set[web.WebSocketResponse] = set()
        
        # State buffering to enable throttling
        self.latest_state: Dict[str, Any] = {}
        self.state_dirty = False
        
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Register routes
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/ws/dashboard", self.handle_ws)
        self.app.router.add_get("/api/data_range", self.handle_api_data_range)
        self.app.router.add_post("/api/backtest", self.handle_api_backtest)

    async def start(self) -> None:
        self._is_running = True
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        # Start the background throttled broadcast loop
        self._broadcast_task = asyncio.create_task(self._throttled_broadcast_loop())
        logger.warning(f"Bloomberg Terminal Dashboard hosted at: http://{self.host}:{self.port}")

    async def stop(self) -> None:
        self._is_running = False
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
                
        # Close all open WebSockets
        for ws in list(self.active_monitors):
            await ws.close(code=1001, message="Server shutting down")
            
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
            
        logger.warning("Web Server successfully stopped.")

    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Pushes the latest system state payload into the server buffer.
        Does NOT broadcast immediately (completely non-blocking for core loop).
        """
        self.latest_state = state
        self.state_dirty = True

    def broadcast_message(self, msg_dict: Dict[str, Any]) -> None:
        """
        Broadcasts an arbitrary message dictionary to all active WebSockets concurrently.
        Safe to call from synchronous code within the active event loop.
        """
        if not self._is_running or not self.active_monitors:
            return
        try:
            loop = asyncio.get_running_loop()
            msg = json.dumps(msg_dict)
            for ws in list(self.active_monitors):
                loop.create_task(ws.send_str(msg))
        except RuntimeError:
            pass  # No running event loop
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

    async def _throttled_broadcast_loop(self) -> None:
        """Throttled loop broadcasting state payloads to browsers at a fixed Hz cap."""
        while self._is_running:
            start_time = time.time()
            try:
                if self.state_dirty and self.active_monitors:
                    msg = json.dumps(self.latest_state)
                    # Broadcast to all active browsers concurrently
                    tasks = [ws.send_str(msg) for ws in list(self.active_monitors)]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    self.state_dirty = False
            except Exception as e:
                logger.error(f"Error in throttled broadcast loop: {e}")
                
            elapsed = time.time() - start_time
            sleep_time = max(0.005, self.broadcast_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serves the static Bloomberg Terminal dashboard HTML file."""
        try:
            filepath = os.path.join(os.path.dirname(__file__), "dashboard.html")
            if not os.path.exists(filepath):
                return web.Response(text="dashboard.html not found.", status=404)
                
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return web.Response(text=content, content_type="text/html")
        except Exception as e:
            return web.Response(text=f"Error loading dashboard: {e}", status=500)

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Manages WebSocket connection lifecycles."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.active_monitors.add(ws)
        
        # Send initial state immediately if available
        if self.latest_state:
            try:
                await ws.send_str(json.dumps(self.latest_state))
            except Exception:
                pass
                
        try:
            async for msg in ws:
                # Do nothing, only server-to-client updates are needed
                pass
        finally:
            self.active_monitors.discard(ws)
        return ws

    def _extract_timestamp_from_path(self, file_path: str) -> Optional[float]:
        """Extracts Unix timestamp from either filename or its parent directory name."""
        basename = os.path.basename(file_path)
        ts = None
        # Try extracting from filename, e.g. ticks_rec_1779988461.parquet or tick_data_1779562304.parquet
        try:
            parts = basename.split(".")[0].split("_")
            ts = float(parts[-1])
        except (ValueError, IndexError):
            pass
            
        if ts is None:
            # Try parent directory fallback, e.g. live_1779988461/
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if "_" in parent_dir:
                try:
                    ts = float(parent_dir.split("_")[-1])
                except ValueError:
                    pass
                    
        if ts is not None:
            if ts > 5e9:
                ts /= 1000.0
            return ts
        return None

    async def handle_api_data_range(self, request: web.Request) -> web.Response:
        """Finds available continuous data sessions across all tick files."""
        import glob
        import polars as pl
        
        patterns = [
            "logs/*/*/*/ticks.parquet",
            "logs/*/*/*/ticks.csv",
            "data/raw/ticks.parquet",
            "data/raw/tick_data_*.parquet",
            "data/raw/ticks_rec_*.parquet",
            "c:/Users/loren/Documents/AntiGravity Projects/polymarket/data/raw/ticks.parquet",
            "c:/Users/loren/Documents/AntiGravity Projects/polymarket/data/raw/tick_data_*.parquet",
            "c:/Users/loren/Documents/AntiGravity Projects/polymarket/data/raw/ticks_rec_*.parquet",
        ]
        
        file_paths = []
        for pattern in patterns:
            for p in glob.glob(pattern):
                file_paths.append(p)
                
        # Deduplicate paths
        file_paths = list(set(file_paths))
        
        sessions = []
        min_ts = float('inf')
        max_ts = float('-inf')
        
        for file_path in file_paths:
            # Skip old backtest run logs to avoid cluttering
            if "backtest_" in file_path.replace("\\", "/"):
                continue
                
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".parquet":
                    lf = pl.scan_parquet(file_path)
                    stats = lf.select([
                        pl.col("timestamp").first().alias("min_t"),
                        pl.col("timestamp").last().alias("max_t"),
                        pl.len().alias("count")
                    ]).collect()
                    
                    min_t = float(stats["min_t"][0])
                    max_t = float(stats["max_t"][0])
                    count = int(stats["count"][0])
                else:
                    df = pl.read_csv(file_path, columns=["timestamp"])
                    min_t = float(df["timestamp"].min())
                    max_t = float(df["timestamp"].max())
                    count = len(df)
                
                # Millisecond checks
                if min_t > 5e9:
                    min_t /= 1000.0
                if max_t > 5e9:
                    max_t /= 1000.0
                    
                duration_sec = max_t - min_t
                min_dt = datetime.datetime.fromtimestamp(min_t)
                max_dt = datetime.datetime.fromtimestamp(max_t)
                
                # Format name beautifully
                parts = file_path.replace("\\", "/").split("/")
                if len(parts) >= 4 and parts[0] == "logs":
                    session_name = f"{parts[1]} - {parts[3]}"
                else:
                    session_name = os.path.basename(file_path)
                    
                # Format counts readable, e.g. 118K
                count_str = f"{count/1000:.0f}K" if count >= 1000 else str(count)
                duration_str = f"{duration_sec/60:.1f} min" if duration_sec < 3600 else f"{duration_sec/3600:.1f} hrs"
                
                session_lbl = f"{session_name} ({duration_str}, {count_str} ticks)"
                
                sessions.append({
                    "name": session_lbl,
                    "file_path": file_path,
                    "min_time": min_t,
                    "max_time": max_t,
                    "min_iso": min_dt.strftime("%Y-%m-%dT%H:%M"),
                    "max_iso": max_dt.strftime("%Y-%m-%dT%H:%M"),
                    "duration_minutes": round(duration_sec / 60.0, 1),
                    "ticks_count": count
                })
                
                if min_t < min_ts:
                    min_ts = min_t
                if max_t > max_ts:
                    max_ts = max_t
            except Exception:
                pass
                
        sessions.sort(key=lambda x: x["min_time"])
        
        if min_ts == float('inf') or max_ts == float('-inf'):
            now = time.time()
            min_ts = now - 86400
            max_ts = now
            
        min_iso = datetime.datetime.fromtimestamp(min_ts).strftime("%Y-%m-%dT%H:%M")
        max_iso = datetime.datetime.fromtimestamp(max_ts + 300).strftime("%Y-%m-%dT%H:%M")
        
        return web.json_response({
            "success": True,
            "min_iso": min_iso,
            "max_iso": max_iso,
            "min_time": min_ts,
            "max_time": max_ts + 300,
            "sessions": sessions
        })

    async def handle_api_backtest(self, request: web.Request) -> web.Response:
        """Asynchronously replays historical logs over a selected time window."""
        try:
            body = await request.json()
            start_time = body.get("start_time")
            end_time = body.get("end_time")
            strategy_name = body.get("strategy_name", "merton")
            
            if not start_time or not end_time:
                return web.json_response({"success": False, "error": "Invalid time window parameters."}, status=400)
                
            start_time = float(start_time)
            end_time = float(end_time)
            
            # Perform alignment and run backtest in background executor thread
            loop = asyncio.get_running_loop()
            
            def progress_callback(pct: int):
                # Emit progress update to all active dashboard WebSockets
                payload = {"type": "backtest_progress", "progress": pct}
                msg = json.dumps(payload)
                for ws in list(self.active_monitors):
                    asyncio.run_coroutine_threadsafe(ws.send_str(msg), loop)
                    
            def run_backtest_thread():
                config = SystemConfig()
                if strategy_name:
                    config.__dict__["STRATEGY_NAME"] = strategy_name
                runner = BacktestRunner(config)
                
                # Scan folders to gather files and check boundaries
                paths = [
                    "logs",
                    os.path.join("data", "raw"),
                    "c:/Users/loren/Documents/AntiGravity Projects/polymarket/data/raw"
                ]
                
                files_with_ts = []
                for base_path in paths:
                    if not os.path.exists(base_path):
                        continue
                    for ext in ("*.parquet", "*.csv"):
                        pattern = os.path.join(base_path, "**", ext)
                        for file_path in glob.glob(pattern, recursive=True):
                            basename = os.path.basename(file_path)
                            if "tick" not in basename.lower():
                                continue
                            ts = self._extract_timestamp_from_path(file_path)
                            if ts is not None:
                                files_with_ts.append((ts, file_path))
                                
                files_with_ts.sort(key=lambda x: x[0])
                
                # Filter files overlapping with [start_time, end_time]
                selected_files = []
                last_ts = None
                for i in range(len(files_with_ts)):
                    ts, file_path = files_with_ts[i]
                    next_ts = files_with_ts[i+1][0] if i + 1 < len(files_with_ts) else float('inf')
                    
                    if ts <= end_time and next_ts >= start_time:
                        # Log if there is a gap > 10 minutes between this file and the last one we included
                        if last_ts is not None and (ts - last_ts > 600):
                            logger.warning(f"Backtest data gap detected (>10m) between files. Continuing...")
                        selected_files.append(file_path)
                        last_ts = ts
                        
                if not selected_files:
                    raise ValueError(f"No tick data files found covering the range {start_time} to {end_time}")
                    
                dfs = []
                for f_path in selected_files:
                    df = runner.load_ticks_file(f_path)
                    
                    # Align V1 historical schemas to V2 context models if loading V1 data
                    if "binance_bid" in df.columns and "binance_ask" in df.columns:
                        df = df.with_columns(
                            (0.5 * (pl.col("binance_bid") + pl.col("binance_ask"))).alias("spot_price")
                        )
                    elif "polymarket_bid" in df.columns and "polymarket_ask" in df.columns:
                        df = df.with_columns(
                            (0.5 * (pl.col("polymarket_bid") + pl.col("polymarket_ask"))).alias("spot_price")
                        )
                        
                    rename_map = {}
                    if "polymarket_bids_l2" in df.columns:
                        rename_map["polymarket_bids_l2"] = "bids_l2"
                    if "polymarket_asks_l2" in df.columns:
                        rename_map["polymarket_asks_l2"] = "asks_l2"
                    if "polymarket_ofi" in df.columns:
                        rename_map["polymarket_ofi"] = "ofi"
                        
                    if rename_map:
                        df = df.rename(rename_map)
                        
                    # Keep only essential columns to save memory
                    req_cols = ["timestamp", "spot_price", "bids_l2", "asks_l2"]
                    if "volatility" in df.columns:
                        req_cols.append("volatility")
                    if "ofi" in df.columns:
                        req_cols.append("ofi")
                        
                    df = df.select([c for c in req_cols if c in df.columns])
                    dfs.append(df)
                    
                if not dfs:
                    raise ValueError("No data was successfully loaded from selected files.")
                    
                combined_df = pl.concat(dfs).sort("timestamp")
                aligned_df = combined_df.filter((pl.col("timestamp") >= start_time) & (pl.col("timestamp") <= end_time))
                
                if aligned_df.is_empty():
                    raise ValueError(f"No ticks available within selected time window.")
                    
                # Run event-driven simulation loop
                async_loop = asyncio.new_event_loop()
                try:
                    return async_loop.run_until_complete(
                        runner.run(aligned_df, strategy_name=strategy_name, progress_callback=progress_callback)
                    )
                finally:
                    async_loop.close()
 
            # Execute backtest thread
            results = await loop.run_in_executor(None, run_backtest_thread)
            
            return web.json_response({"success": True, "results": results})
            
        except Exception as e:
            logger.error(f"Error executing backtest via API: {e}", exc_info=True)
            return web.json_response({"success": False, "error": str(e)}, status=500)
