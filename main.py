import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from config.settings import SystemConfig
from src.ingestion.market_manager import MarketManager
from src.ingestion.live_feeds import ChainlinkSpotFeed, ClobOrderBookFeed
from src.core.strike_manager import StrikeManager
from src.core.market_context import MarketContext
from src.strategies.merton_strategy import MertonStrategy
from src.execution.shadow_book import ShadowOrderBook
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.logging.recorder import DataRecorder
from src.ui.dashboard import run_terminal_dashboard
from src.ui.web_server import WebServer

# Configure core console logging
handlers = [logging.FileHandler("system_run.log", encoding="utf-8")]
if "--no-term" in sys.argv:
    # Print clean info logs to console when Rich dashboard is disabled
    handlers.append(logging.StreamHandler(sys.stdout))
    log_level = logging.INFO
else:
    log_level = logging.WARNING

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=handlers
)
logger = logging.getLogger("LiveOrchestrator")

class LiveOrchestrator:
    """
    Main entry point for Polymarket V2 live/paper trading.
    Orchestrates Websockets, strategy pricing, risk limits, executions, logging, and CLI UI.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.is_running = False
        
        # Initialize Core Data Loggers
        self.recorder = DataRecorder(
            base_log_dir=config.LOG_DIR,
            strategy_name="merton",
            run_id=f"live_{int(time.time())}",
            buffer_size=100  # Flush every 100 ticks for quick logging
        )
        
        # Initialize strategy & execution
        self.market_manager = MarketManager(config)
        self.spot_feed = ChainlinkSpotFeed(config)
        self.shadow_book = ShadowOrderBook()
        self.strategy = MertonStrategy(config)
        self.client = MockExecutionClient(config, self.recorder, self.shadow_book)
        self.engine = ExecutionEngine(config)
        self.strike_manager: Optional[StrikeManager] = None
        self.waiting_for_first_rollover = False
        self.total_trades = 0
        self._last_console_log_time = 0.0
        # Time-based EMA state for p_yes smoothing.
        # Timestamp of the previous p_yes update, needed to compute dt for alpha.
        self._smoothed_p_yes_ts: float = 0.0
        
        # Shared decision pointer for the UI Dashboard
        self.latest_decision_ref: List[Dict[str, Any]] = [
            {"side": "HOLD", "size": 0.0, "vwap": 0.0, "reason": "ENGINE_INITIALIZING"}
        ]
        
        # Optional Web Server
        self.web_server: Optional[WebServer] = None
        if self.config.web_server.ENABLED:
            self.web_server = WebServer(self.config, self)
            
        self.clob_feed: Optional[ClobOrderBookFeed] = None
        self._tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self.is_running = True
        self.log_message("info", "Initializing Live Orchestrator...")
        
        # 1. Start keep-awake loop to prevent Windows from sleeping
        if os.name == 'nt':
            self._tasks.append(asyncio.create_task(self._keep_awake_loop()))
        
        # 1. Start Spot Feed Websocket (runs in background)
        await self.spot_feed.start()
        
        # Wait a brief moment to capture initial spot prices
        self.log_message("info", "Waiting for initial Spot ticks...")
        for _ in range(10):
            if self.spot_feed.price is not None:
                break
            await asyncio.sleep(0.5)
            
        initial_spot = self.spot_feed.price or 67500.0
        self.log_message("info", f"Initial Spot Price captured: ${initial_spot:,.2f}")
        
        # 2. Perform initial Market Discovery
        t_now = time.time()
        await self.market_manager.update_market_cycle(t_now)
        
        # Check if started mid-cycle (rollover start is current_expiry - 300)
        if self.market_manager.current_expiry is not None:
            cycle_start_time = self.market_manager.current_expiry - 300
            if t_now - cycle_start_time > 10.0:
                self.waiting_for_first_rollover = True
                self.strike_manager = None
                self.log_message("info", "Ok, aspetto il prossimo ciclo...")
            else:
                self.waiting_for_first_rollover = False
                self.strike_manager = StrikeManager(
                    presumed_strike=self.market_manager.strike_price,
                    expiration_timestamp=self.market_manager.current_expiry
                )
        else:
            self.waiting_for_first_rollover = True
            self.strike_manager = None
        
        # 3. Start CLOB Order Book WebSocket Feed
        await self._restart_clob_feed()
        
        # 4. Start background loops
        self._tasks.append(asyncio.create_task(self._market_discovery_loop()))
        self._tasks.append(asyncio.create_task(self._strike_resolution_loop()))
        
        # 4.5. Start Web Server
        if self.web_server:
            await self.web_server.start()
            # Send initial state update
            self._update_web_state()
        
        # 5. Start CLI Terminal UI Task
        if "--no-term" not in sys.argv:
            self._tasks.append(asyncio.create_task(
                run_terminal_dashboard(
                    config=self.config,
                    market_manager=self.market_manager,
                    spot_feed=self.spot_feed,
                    shadow_book=self.shadow_book,
                    strategy=self.strategy,
                    client=self.client,
                    engine=self.engine,
                    latest_decision_ref=self.latest_decision_ref,
                    stop_event=self._stop_event
                )
            ))
        else:
            self.log_message("warning", "Terminal Dashboard disabled via --no-term. Running in pure console/web server mode.")
        
        self.log_message("info", "Live Orchestrator initialized successfully.")

    async def stop(self) -> None:
        self.is_running = False
        self._stop_event.set()
        
        # Restore default sleep behavior on Windows
        if os.name == 'nt':
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                logger.info("Windows Thread Execution State reset to default sleep behavior.")
            except Exception:
                pass
        
        # Stop Web Server
        if self.web_server:
            await self.web_server.stop()
            
        # Stop background feeds
        await self.spot_feed.stop()
        if self.clob_feed:
            await self.clob_feed.stop()
            
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Final buffer flushes
        self.recorder.flush()
        logger.info("Orchestrator stopped cleanly.")

    async def _keep_awake_loop(self) -> None:
        """
        Periodically refreshes the Windows thread execution state every 30 seconds
        to prevent the system from sleeping while the bot is running.
        A single SetThreadExecutionState call is not sufficient — Windows resets
        the state if it is not refreshed by the calling thread.
        """
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        except Exception:
            return  # Not on Windows or ctypes unavailable

        while self.is_running:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(flags)
                logger.debug("[KeepAwake] Windows execution state refreshed.")
            except Exception as e:
                logger.warning(f"[KeepAwake] Failed to refresh execution state: {e}")
            await asyncio.sleep(30.0)

    async def _restart_clob_feed(self) -> None:
        """Starts or restarts the CLOB WS feed, subscribing to the newly active token IDs."""
        if self.clob_feed:
            await self.clob_feed.stop()
            
        # Dynamically inject YES/NO tokens resolved by MarketManager into configuration settings
        self.config.polymarket.__dict__["YES_TOKEN_ID"] = self.market_manager.yes_token_id
        self.config.polymarket.__dict__["NO_TOKEN_ID"] = self.market_manager.no_token_id
            
        # Re-initialize CLOB Order book feed only if tokens are valid
        if self.market_manager.yes_token_id and self.market_manager.no_token_id:
            self.clob_feed = ClobOrderBookFeed(self.config, self._clob_callback)
            await self.clob_feed.start()
        else:
            self.clob_feed = None
            self.log_message("warning", "[Orchestrator] CLOB feed not started: YES/NO Token IDs are unresolved or missing.")

    def _update_web_state(
        self, 
        decision: Optional[Dict[str, Any]] = None,
        spot: Optional[float] = None,
        strike: Optional[float] = None,
        ofi: Optional[float] = None,
        vol: Optional[float] = None,
        tau_sec: Optional[float] = None,
        p_yes: Optional[float] = None
    ) -> None:
        if not self.web_server:
            return
            
        t_now = time.time()
        
        # Fallbacks
        if spot is None:
            spot = self.spot_feed.price
        if strike is None:
            strike = self.strike_manager.get_strike(t_now, spot) if self.strike_manager and spot is not None else None
        if ofi is None:
            ofi = self.shadow_book.smoothed_ofi
        if vol is None:
            vol = self.strategy.vol_calibrator.calculate_volatility(self.config.merton.DEFAULT_SIGMA)
        if tau_sec is None:
            tau_sec = max(0.0, self.market_manager.current_expiry - t_now) if self.market_manager.current_expiry else 300.0
            
        if p_yes is None and spot is not None and strike is not None:
            dummy_context = MarketContext(
                timestamp=t_now,
                spot_price=spot,
                strike_price=strike,
                tau_seconds=tau_sec,
                volatility=vol,
                ofi=ofi,
                bids_l2=[],
                asks_l2=[]
            )
            p_yes = self.strategy.get_probability(dummy_context)
            
        top_b, top_a = self.shadow_book.get_market_top_of_book()
        bid_price_yes = top_b[0] if top_b else 0.5
        pos_qty_yes = self.client.get_position_size("YES")
        pos_qty_no = self.client.get_position_size("NO")
        pos_val = self.client.get_portfolio_value(bid_price_yes)
        cash = self.client.cash_balance
        equity = cash + pos_val
        pnl = equity - self.config.arbitrage.INITIAL_CAPITAL
        
        p_mkt = 0.5 * (top_b[0] + top_a[0]) if top_b and top_a else None
        edge = p_yes - p_mkt if p_yes is not None and p_mkt is not None else None
        
        if decision is None:
            decision = self.latest_decision_ref[0]
            
        state = {
            "type": "portfolio_state",
            "ticker": self.config.TICKER,
            "time_left": tau_sec,
            "equity": equity,
            "cash": cash,
            "pos_val": pos_val,
            "pnl": pnl,
            "total_trades": self.total_trades,
            "oracle_live": self.spot_feed.is_connected,
            "spot_price": spot,
            "strike": strike,
            "p_fair": p_yes,
            "p_market": p_mkt,
            "edge": edge,
            "volatility": vol,
            "ofi": ofi,
            "slug": self.market_manager.current_slug,
            "condition_id": self.market_manager.condition_id,
            "bids_l2": self.shadow_book.get_sorted_bids()[:5],
            "asks_l2": self.shadow_book.get_sorted_asks()[:5],
            "latest_decision": decision,
            "pos_qty_yes": pos_qty_yes,
            "pos_qty_no": pos_qty_no
        }
        self.web_server.update_state(state)

    def log_message(self, level: str, message: str) -> None:
        """Logs a message to console loggers and broadcasts to web dashboard log console."""
        # 1. Standard python logger
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)

        # 2. WebSocket broadcast if WebServer is running
        if self.web_server:
            ui_type = "system"
            if level in ("warning", "error"):
                ui_type = "warning"
            elif "signal" in message.lower() or "executing" in message.lower():
                ui_type = "trade"
            elif "settle" in message.lower():
                ui_type = "settle"
                
            self.web_server.broadcast_message({
                "type": "log_message",
                "log_type": ui_type,
                "message": message
            })

    async def _clob_callback(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], is_snapshot: bool) -> None:
        """Callback triggered on each CLOB Order book tick arrival."""
        if not self.is_running:
            return
            
        t_now = time.time()
        
        if self.waiting_for_first_rollover:
            if t_now - getattr(self, "_last_wait_log_time", 0.0) >= 15.0:
                self._last_wait_log_time = t_now
                self.log_message("info", "Ok, aspetto il prossimo ciclo...")
            return
        
        # Periodic diagnostic warnings if we are missing critical feeds to proceed
        if self.spot_feed.price is None or self.strike_manager is None:
            if t_now - getattr(self, "_last_feed_warning_time", 0.0) >= 5.0:
                self._last_feed_warning_time = t_now
                status_msg = (
                    f"Waiting for feeds: Spot Price is {'None' if self.spot_feed.price is None else 'OK'}, "
                    f"Strike Manager is {'None' if self.strike_manager is None else 'OK'}"
                )
                self.log_message("warning", status_msg)
            return
            
        spot = self.spot_feed.price
        
        # Reconcile local shadow order book
        ofi = self.shadow_book.update_book(bids, asks, is_snapshot)
        
        # Resolve active Strike Price only after the cycle has actually started
        if self.strike_manager and (self.strike_manager.presumed_strike is None or self.strike_manager.presumed_strike == 0.0):
            if self.market_manager.current_expiry is not None:
                cycle_start_time = self.market_manager.current_expiry - 300
                if t_now >= cycle_start_time:
                    strike_tick = self.spot_feed.get_first_tick_after(cycle_start_time)
                    if strike_tick is not None:
                        _, strike_price_val = strike_tick
                        self.strike_manager.presumed_strike = strike_price_val
                        self.log_message(
                            "info",
                            f"[StrikeManager] Active Strike resolved via 1st Chainlink tick after cycle start: ${strike_price_val:,.2f}"
                        )
                    
        active_strike = self.strike_manager.get_strike(t_now, spot)
        
        # Calculate time remaining
        tau_sec = max(0.0, self.market_manager.current_expiry - t_now)
        
        # Calibrate volatility
        self.strategy.vol_calibrator.add_tick(spot, t_now)
        vol = self.strategy.vol_calibrator.calculate_volatility(self.config.merton.DEFAULT_SIGMA)
        
        # Construct MarketContext
        context = MarketContext(
            timestamp=t_now,
            spot_price=spot,
            strike_price=active_strike,
            tau_seconds=tau_sec,
            volatility=vol,
            ofi=ofi,
            bids_l2=self.shadow_book.get_sorted_bids(),
            asks_l2=self.shadow_book.get_sorted_asks()
        )
        
        # Calculate Merton probability
        p_yes_raw = self.strategy.get_probability(context)
        
        # Apply time-based EMA smoothing to p_yes.
        # A fixed alpha (e.g. 0.05) was designed for 1 tick/s but CLOB delivers
        # 10-20 ticks/s, compressing the effective smoothing window from ~30s to
        # ~2-5s and letting raw OFI oscillations pass through almost unattenuated.
        # Time-based formula: alpha = 1 - exp(-dt / T_half) ensures consistent
        # smoothing regardless of tick rate.
        if p_yes_raw is not None:
            if not hasattr(self, "_smoothed_p_yes") or self._smoothed_p_yes is None:
                self._smoothed_p_yes = p_yes_raw
                self._smoothed_p_yes_ts = t_now
            else:
                halflife = self.config.merton.EMA_HALFLIFE_SEC
                dt_ema = t_now - self._smoothed_p_yes_ts
                alpha = 1.0 - (2.718281828 ** (-dt_ema / halflife)) if halflife > 0.0 else 1.0
                self._smoothed_p_yes = alpha * p_yes_raw + (1.0 - alpha) * self._smoothed_p_yes
                self._smoothed_p_yes_ts = t_now
            p_yes = self._smoothed_p_yes
        else:
            p_yes = None
        
        # Record tick in Parquet database
        self.recorder.record_tick(t_now, spot, ofi, vol, bids, asks)
        
        # Evaluate Trade Sizing and Execution
        decision = self.engine.evaluate_and_trade(p_yes, context, self.client)
        self.latest_decision_ref[0] = decision
        
        # Get Implied Market price and Edge — use REAL market book (q_real), never depleted
        top_b, top_a = self.shadow_book.get_market_top_of_book()
        p_mkt = 0.5 * (top_b[0] + top_a[0]) if top_b and top_a else None
        edge = p_yes - p_mkt if p_yes is not None and p_mkt is not None else None
        
        # Throttled console logger for pure console mode
        if t_now - getattr(self, "_last_console_log_time", 0.0) >= 2.0:
            self._last_console_log_time = t_now
            p_yes_str = f"{p_yes * 100:.2f}%" if p_yes is not None else "—"
            p_mkt_str = f"{p_mkt * 100:.2f}%" if p_mkt is not None else "—"
            edge_str = f"{edge * 100:+.2f}%" if edge is not None else "—"
            msg = (
                f"[TICK] Spot: ${spot:,.2f} | Strike: ${active_strike or 0.0:,.2f} | "
                f"YES: {p_yes_str} | Market YES: {p_mkt_str} | Edge: {edge_str} | OFI: {ofi:+.1f}"
            )
            self.log_message("info", msg)
            
        # Update Web Server state payload (non-blocking)
        if self.web_server:
            self._update_web_state(decision, spot, active_strike, ofi, vol, tau_sec, p_yes)
            
        if decision["side"] != "HOLD":
            self.total_trades += 1
            # Record strategy signal
            top_b, top_a = self.shadow_book.get_market_top_of_book()
            p_mkt = 0.5 * (top_b[0] + top_a[0]) if top_b and top_a else p_yes
            
            self.recorder.record_signal(
                timestamp=t_now,
                spot_price=spot,
                strike=active_strike,
                model_prob=p_yes,
                implied_prob=p_mkt,
                kelly_size=decision["size"],
                status=decision["side"]
            )
            
            # Broadcast signal immediately to UI
            if self.web_server:
                self.web_server.broadcast_message({
                    "type": "trade_signal",
                    "signal": {
                        "side": decision["side"],
                        "size": decision["size"],
                        "vwap": decision["vwap"],
                        "ev": decision["ev"]
                    }
                })
            self.log_message("info", f"Executing trade: {decision['side']} | Size: {decision['size']:.2f} | VWAP: {decision['vwap']:.4f} | EV: {decision['ev']:.4f}")
            
            # Execute trade synchronously (awaited) so shadow book is depleted
            # BEFORE the next CLOB tick arrives — prevents duplicate signals.
            await self.client.execute_trade(
                side=decision["side"],
                qty=decision["size"],
                price=decision["vwap"],
                ev=decision["ev"],
                expected_slippage_bps=decision["expected_slippage_bps"],
                context_state={"timestamp": t_now, "strike_price": active_strike}
            )
            
            # Immediately update the web state so that the UI shows the depleted shadow book
            if self.web_server:
                self._update_web_state(decision, spot, active_strike, ofi, vol, tau_sec, p_yes)

    async def _market_discovery_loop(self) -> None:
        """Periodically evaluates clock to trigger dynamic market Discovery and Rollovers."""
        while self.is_running:
            try:
                t_now = time.time()
                # If a rollover boundary is crossed
                rollover = await self.market_manager.update_market_cycle(t_now)
                if rollover:
                    self.waiting_for_first_rollover = False
                    self.log_message(
                        "info",
                        f"Market Rollover detected. New active cycle expiry: {self.market_manager.current_expiry} | "
                        f"Slug: {self.market_manager.current_slug}"
                    )
                    
                    # 1. Settle old positions at current spot price
                    if self.strike_manager and self.spot_feed.price is not None:
                        settle_spot = self.spot_feed.price
                        settlement_strike = self.strike_manager.get_strike(t_now, settle_spot)
                        self.client.settle_positions(
                            settlement_price=settle_spot,
                            strike_price=settlement_strike,
                            timestamp=t_now
                        )
                        resolved_yes = settle_spot >= settlement_strike
                        if self.web_server:
                            self.web_server.broadcast_message({
                                "type": "settlement_update",
                                "settlement": {
                                    "settle_spot": settle_spot,
                                    "strike": settlement_strike,
                                    "resolved_yes": resolved_yes
                                }
                            })
                        self.log_message(
                            "info",
                            f"Settling old positions at Expiry. Spot: ${settle_spot:,.2f} | "
                            f"Strike: ${settlement_strike:,.2f} | YES resolved as {'WON' if resolved_yes else 'LOST'}"
                        )
                        
                    # 2. Reset StrikeManager for the new cycle
                    self.strike_manager = StrikeManager(
                        presumed_strike=self.market_manager.strike_price,
                        expiration_timestamp=self.market_manager.current_expiry
                    )
                    self._smoothed_p_yes = None
                    self._smoothed_p_yes_ts = 0.0
                    self.strategy.reset()

                    # 3. Restart CLOB feed to subscribe to new tokens
                    await self._restart_clob_feed()
                    
                    # 4. Update web server immediately
                    self._update_web_state()
                else:
                    # If we are in an active cycle but token IDs are missing, try to resolve them mid-cycle!
                    if self.market_manager.yes_token_id is None or self.market_manager.no_token_id is None:
                        if t_now - getattr(self, "_last_api_retry_time", 0.0) >= 10.0:
                            self._last_api_retry_time = t_now
                            self.log_message("info", "[Orchestrator] Token IDs are missing. Retrying Gamma API discovery...")
                            await self.market_manager.fetch_market_context()
                            if self.market_manager.yes_token_id and self.market_manager.no_token_id:
                                if not self.strike_manager:
                                    self.strike_manager = StrikeManager(
                                        presumed_strike=self.market_manager.strike_price,
                                        expiration_timestamp=self.market_manager.current_expiry
                                    )
                                await self._restart_clob_feed()
                                self._update_web_state()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_message("error", f"Error in market discovery loop: {e}")
            await asyncio.sleep(1.0)

    async def _strike_resolution_loop(self) -> None:
        """Periodically checks if expiration is crossed to lock the Strike Price via spot tick."""
        while self.is_running:
            try:
                if self.waiting_for_first_rollover:
                    await asyncio.sleep(0.5)
                    continue
                    
                if self.strike_manager and self.spot_feed.price is not None:
                    t_now = time.time()
                    spot = self.spot_feed.price
                    
                    # Try to resolve presumed strike during active cycle if not already resolved
                    if self.strike_manager.presumed_strike is None or self.strike_manager.presumed_strike == 0.0:
                        if self.market_manager.current_expiry is not None:
                            cycle_start_time = self.market_manager.current_expiry - 300
                            strike_tick = self.spot_feed.get_first_tick_after(cycle_start_time)
                            if strike_tick is not None:
                                _, strike_price_val = strike_tick
                                self.strike_manager.presumed_strike = strike_price_val
                                self.log_message(
                                    "info",
                                    f"[StrikeManager] Active Strike resolved via 2nd Chainlink tick after cycle start: ${strike_price_val:,.2f}"
                                )
                                
                    # Updates resolved strike if t_now >= expiration_timestamp
                    resolved_strike = self.strike_manager.get_strike(t_now, spot)
                    
                    # Update web state immediately on strike resolution
                    self._update_web_state(spot=spot, strike=resolved_strike)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in strike resolution loop: {e}")
            await asyncio.sleep(0.5)


async def main_async():
    config = SystemConfig()
    orchestrator = LiveOrchestrator(config)
    
    try:
        await orchestrator.start()
        # Keep orchestrator running until user interrupts or stops
        while orchestrator.is_running:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        sys.exit(0)
