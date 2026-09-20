import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from config.settings import SystemConfig
from src.ingestion.market_manager import MarketManager
from src.ingestion.live_feeds import ChainlinkSpotFeed, ClobOrderBookFeed
from src.ingestion.binance_feed import BinanceSpotFeed
from src.core.strike_manager import StrikeManager
from src.core.market_context import MarketContext
from src.strategies.factory import StrategyFactory
from src.execution.shadow_book import ShadowOrderBook
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.execution.divergence_filter import DivergenceVelocityFilter
from src.core.keep_awake import keep_awake
from src.logging.recorder import DataRecorder
from src.ui.dashboard import run_terminal_dashboard
from src.ui.web_server import WebServer

# Configure core console logging.
# The run log goes under LOG_DIR and rotates: an unrotated handler in the repo root
# grew past 8 MB and never stopped. 5 files x 10 MB caps it at 50 MB.
_LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
handlers = [
    RotatingFileHandler(
        os.path.join(_LOG_DIR, "system_run.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=4,
        encoding="utf-8",
    )
]
if "--term" in sys.argv:
    log_level = logging.WARNING
else:
    # Print clean info logs to console by default (Web UI is primary)
    handlers.append(logging.StreamHandler(sys.stdout))
    log_level = logging.INFO

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
            buffer_size=1000  # Flush every 1000 ticks to reduce IO pressure
        )
        
        # Initialize strategy & execution
        self.market_manager = MarketManager(config)
        self.spot_feed = ChainlinkSpotFeed(config)
        # External spot, used only to nowcast the oracle: the oracle still settles.
        self.binance_feed = BinanceSpotFeed(config) if config.binance.ENABLED else None
        self.shadow_book = ShadowOrderBook()
        self.strategy = StrategyFactory.get_strategy(config.STRATEGY_NAME, config)
        self.client = MockExecutionClient(config, self.recorder, self.shadow_book)
        self.divergence_filter = DivergenceVelocityFilter(config)
        self.engine = ExecutionEngine(self.strategy, self.client, config)
        self.engine.divergence_filter = self.divergence_filter
        self.strike_manager: Optional[StrikeManager] = None
        self.waiting_for_first_rollover = False
        self.total_trades = 0
        self._last_console_log_time = 0.0
        
        # Initialize HFT performance, toxicity and queue priority metrics
        self.hft_metrics = {
            "volume_sent_touched": 0.0,
            "volume_executed": 0.0,
            "pending_mtm_trades": [],   # List[dict]: {"timestamp", "side", "exec_price", "mtm_1s", "mtm_5s", "mtm_15s"}
            "completed_mtm_trades": [], # List[dict]: historical resolved MTM trades
            "total_taker_edge": 0.0,
            "total_taker_slippage": 0.0,
            "total_taker_trades": 0,
            "total_orders_sent": 0,
            "total_orders_cancelled": 0,
            "tps": 0.0
        }
        self._tick_count_tps = 0
        self._last_tps_time = time.time()
        self._current_active_orders = {
            "bid": None, # dict or None: {"price": P, "qty": Q, "queue_ahead": qa, "prev_depth": pd}
            "ask": None  # dict or None: {"price": P, "qty": Q, "queue_ahead": qa, "prev_depth": pd}
        }
        # Minimum interval (seconds) between full shadow-book resets to prevent
        # CLOB reconnect bursts from triggering multiple independent snapshot fills.
        self._last_snapshot_ts: float = 0.0
        # Time-based EMA state for p_yes smoothing.
        # Timestamp of the previous p_yes update, needed to compute dt for alpha.
        self._smoothed_p_yes_ts: float = 0.0
        self._last_processed_expiry: Optional[int] = None

        # Safety cooldown tracking to prevent infinite rapid-fire taker order spamming on rejections
        self._last_rejection_time: Dict[str, float] = {}

        # ── Strict Trade Queue ────────────────────────────────────────────────
        # Tracks the currently inflight trade execution.
        # No new signals will be evaluated until this task completes.
        self._pending_trade_task: Optional[asyncio.Task] = None
        # ─────────────────────────────────────────────────────────────────────
        
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
        
        # 1. Prevent Windows from sleeping (the display is left to time out normally)
        keep_awake()
        
        # 1. Start Spot Feed Websocket (runs in background)
        await self.spot_feed.start()
        if self.binance_feed:
            await self.binance_feed.start()
        
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
        
        # CRITICAL USER REQUIREMENT: Always wait for the next cycle when started
        self.waiting_for_first_rollover = True
        self.strike_manager = None
        self.log_message("info", "[Special Startup Procedure] Avvio speciale attivo: attendo che sorga il prossimo ciclo prima di procedere...")
        
        # 3. Start CLOB Order Book WebSocket Feed
        await self._restart_clob_feed()
        
        # 4. Start background loops
        self._tasks.append(asyncio.create_task(self._market_discovery_loop()))
        self._tasks.append(asyncio.create_task(self._strike_resolution_loop()))
        self._tasks.append(asyncio.create_task(self._ui_heartbeat_loop()))
        
        # 4.5. Start Web Server
        if self.web_server:
            await self.web_server.start()
            # Send initial state update
            self._update_web_state()
        
        # 5. Start CLI Terminal UI Task
        if "--term" in sys.argv:
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
                    stop_event=self._stop_event,
                    orchestrator=self
                )
            ))
        else:
            self.log_message("warning", "Terminal Dashboard disabled by default (Web UI active). Run with --term to enable.")
        
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
        if self.binance_feed:
            await self.binance_feed.stop()
        if self.clob_feed:
            await self.clob_feed.stop()
            
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Final buffer flushes
        self.recorder.flush()
        logger.info("Orchestrator stopped cleanly.")

    def _pricing_spot(self, oracle_spot: Optional[float]) -> Optional[float]:
        """Spot to price with: the oracle carried forward by the Binance move since the
        oracle last printed.

        The oracle lags Binance by ~4s and the Polymarket book by ~2s, so quoting on the
        raw oracle means quoting on a level the book has already left. Settlement is still
        decided by the oracle, which is why this is a nowcast OF the oracle and not a
        different price: with no Binance move since the print it returns the oracle
        unchanged, and it does the same whenever the feed is stale or disabled.
        """
        if oracle_spot is None or not self.binance_feed:
            return oracle_spot
        if not self.config.binance.USE_FOR_PRICING:
            return oracle_spot
        return self.binance_feed.nowcast(oracle_spot, self.spot_feed.last_updated)

    async def _restart_clob_feed(self) -> None:
        """Starts or restarts the CLOB WS feed, subscribing to the newly active token IDs."""
        if self.clob_feed:
            await self.clob_feed.stop()
            
        # Dynamically inject YES/NO tokens resolved by MarketManager into configuration settings
        self.config.polymarket.__dict__["YES_TOKEN_ID"] = self.market_manager.yes_token_id
        self.config.polymarket.__dict__["NO_TOKEN_ID"] = self.market_manager.no_token_id
            
        # Re-initialize CLOB Order book feed only if tokens are valid
        if self.market_manager.yes_token_id and self.market_manager.no_token_id:
            self.clob_feed = ClobOrderBookFeed(self.config, self._clob_callback, trade_callback=self._print_callback)
            await self.clob_feed.start()
        else:
            self.clob_feed = None
            self.log_message("warning", "[Orchestrator] CLOB feed not started: YES/NO Token IDs are unresolved or missing.")

    def _get_market_implied_price(self, p_fair: Optional[float] = None) -> Optional[float]:
        """
        Computes the market-implied probability robustly from the top of the book.
        Handles single-sided or empty books cleanly by falling back to the single available side,
        and finally to p_fair or None if no inputs exist.
        """
        top_b, top_a = self.shadow_book.get_market_top_of_book()
        if top_b and top_a:
            return 0.5 * (top_b[0] + top_a[0])
        elif top_b:
            return top_b[0]
        elif top_a:
            return top_a[0]
        return p_fair

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
            
        if p_yes is None:
            p_yes = getattr(self, "_smoothed_p_yes", None)
            
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
        
        p_mkt = self._get_market_implied_price(None)
        edge = p_yes - p_mkt if p_yes is not None and p_mkt is not None else None
        
        if decision is None:
            decision = self.latest_decision_ref[0]
            
        # Update / Compute live HFT metrics
        
        # 1. Limit Order Fill Rate
        vol_touched = self.hft_metrics["volume_sent_touched"]
        vol_executed = self.hft_metrics["volume_executed"]
        fill_rate = vol_executed / vol_touched if vol_touched > 0.0 else 1.0
        
        # 2. MTM Post-Fill (1s, 5s, 15s)
        all_mtm_trades = self.hft_metrics["completed_mtm_trades"] + self.hft_metrics["pending_mtm_trades"]
        mtms_1s = [t["mtm_1s"] for t in all_mtm_trades if t["mtm_1s"] is not None]
        mtms_5s = [t["mtm_5s"] for t in all_mtm_trades if t["mtm_5s"] is not None]
        mtms_15s = [t["mtm_15s"] for t in all_mtm_trades if t["mtm_15s"] is not None]
        
        avg_mtm_1s = sum(mtms_1s) / len(mtms_1s) if mtms_1s else 0.0
        avg_mtm_5s = sum(mtms_5s) / len(mtms_5s) if mtms_5s else 0.0
        avg_mtm_15s = sum(mtms_15s) / len(mtms_15s) if mtms_15s else 0.0
        
        # 3. Edge-to-Slippage Ratio
        tot_edge = self.hft_metrics["total_taker_edge"]
        tot_slip = self.hft_metrics["total_taker_slippage"]
        es_ratio = tot_edge / tot_slip if tot_slip > 0.0 else (99.9 if tot_edge > 0.0 else 0.0)
        
        # 4. PnL Divergence (walk the L2 order book bids/asks for precise liquidation)
        liq_val = 0.0
        if pos_qty_yes > 0.0:
            bids = self.shadow_book.get_sorted_bids()
            remaining = pos_qty_yes
            for p, q in bids:
                fill = min(remaining, q)
                liq_val += fill * p
                remaining -= fill
                if remaining <= 0.0:
                    break
            if remaining > 0.0:
                last_p = bids[-1][0] if bids else 0.0
                liq_val += remaining * last_p
                
        if pos_qty_no > 0.0:
            asks = self.shadow_book.get_sorted_asks()
            remaining = pos_qty_no
            for p, q in asks:
                fill = min(remaining, q)
                liq_val += fill * (1.0 - p) # Complementary NO price walk
                remaining -= fill
                if remaining <= 0.0:
                    break
            if remaining > 0.0:
                last_p = asks[-1][0] if asks else 1.0
                liq_val += remaining * (1.0 - last_p)
                
        pnl_liq = cash + liq_val - self.config.arbitrage.INITIAL_CAPITAL
        pnl_mid = pnl
        pnl_divergence = pnl_mid - pnl_liq
        
        # 5. Order-to-Trade Ratio (OTR)
        total_msgs = self.hft_metrics["total_orders_sent"] + self.hft_metrics["total_orders_cancelled"]
        otr = total_msgs / self.total_trades if self.total_trades > 0 else 0.0
        
        router = self.engine.execution_router

        hft_payload = {
            "fill_rate": fill_rate,
            "avg_mtm_1s": avg_mtm_1s,
            "avg_mtm_5s": avg_mtm_5s,
            "avg_mtm_15s": avg_mtm_15s,
            "es_ratio": es_ratio,
            "pnl_mid": pnl_mid,
            "pnl_liq": pnl_liq,
            "pnl_divergence": pnl_divergence,
            "otr": otr,
            "total_orders_sent": self.hft_metrics["total_orders_sent"],
            "total_orders_cancelled": self.hft_metrics["total_orders_cancelled"],
            "tps": self.hft_metrics.get("tps", 0.0)
        }
            
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
            "pos_qty_no": pos_qty_no,
            "divergence": self.divergence_filter.last_divergence,
            "divergence_velocity": self.divergence_filter.last_velocity,
            "divergence_acceleration": self.divergence_filter.last_acceleration,
            "divergence_scale": self.divergence_filter.last_scale,
            # The bot's own resting quotes: what a maker actually needs to watch.
            # Their midpoint is the reservation price and their half-width is delta.
            "quote_bid": router.active_bid_price or None,
            "quote_bid_qty": router.active_bid_qty or None,
            "quote_ask": router.active_ask_price or None,
            "quote_ask_qty": router.active_ask_qty or None,
            # Effective cap for this book, not the configured ceiling.
            "max_inventory": router.effective_max_inventory(),
            "exit_depth": router.exit_depth_ewma,
            "hft_metrics": hft_payload
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
        elif level == "settle":
            # Settlement messages are info-level in the file log but get their
            # own UI type so the dashboard can render them with a distinct colour.
            logger.info(message)

        # 2. WebSocket broadcast if WebServer is running
        if self.web_server:
            if level == "settle":
                ui_type = "settle"
            elif level in ("warning", "error"):
                ui_type = "warning"
            elif any(x in message.lower() for x in ("signal", "execut", "trade", "fill", "maker")):
                ui_type = "trade"
            elif "settle" in message.lower():
                ui_type = "settle"
            else:
                ui_type = "system"

            self.web_server.broadcast_message({
                "type": "log_message",
                "log_type": ui_type,
                "message": message
            })

    def _print_callback(self, price: float, size: float, side: str, exchange_ts: Optional[float]) -> None:
        """Persists every YES trade print; the only ground truth for calibrating the maker fill model."""
        if self.is_running:
            self.recorder.record_print(time.time(), price, size, side, exchange_ts)

    async def _clob_callback(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], is_snapshot: bool) -> None:
        """Callback triggered on each CLOB Order book tick arrival."""
        if not self.is_running:
            return

        t_now = time.time()
        self._tick_count_tps += 1

        # Check if the active expiry has changed. If so, immediately reset the smoothed probability and strategy state.
        current_expiry = self.strike_manager.expiration_timestamp if self.strike_manager else None
        if current_expiry is not None:
            if not hasattr(self, "_last_processed_expiry") or self._last_processed_expiry != current_expiry:
                self._last_processed_expiry = current_expiry
                self._smoothed_p_yes = None
                self._smoothed_p_yes_ts = 0.0
                self.strategy.reset()
                self.log_message("info", f"[Reset] New expiry cycle detected in CLOB callback: {current_expiry}. Resetting smoothed probability and strategy state.")

        if self.waiting_for_first_rollover:
            # Keep recording while waiting for the first cycle boundary: with 4h cycles
            # that wait is up to 4 hours of book and spot data that used to be dropped.
            # Book and vol calibrator are updated too, so trading starts warm.
            # No pricing here — get_probability must run exactly once per tick.
            spot_wait = self.spot_feed.price
            if spot_wait is not None:
                if is_snapshot:
                    if t_now - self._last_snapshot_ts < 1.0:
                        is_snapshot = False
                    else:
                        self._last_snapshot_ts = t_now
                ofi_wait = self.shadow_book.update_book(bids, asks, is_snapshot, timestamp=t_now)
                self.strategy.vol_calibrator.add_tick(spot_wait, t_now)
                vol_wait = self.strategy.vol_calibrator.calculate_volatility(self.config.merton.DEFAULT_SIGMA)
                top_b_wait, top_a_wait = self.shadow_book.get_market_top_of_book()
                self.recorder.record_tick(t_now, spot_wait, ofi_wait, vol_wait, bids, asks,
                                          top_bid=top_b_wait, top_ask=top_a_wait, is_snapshot=is_snapshot,
                                          oracle_price=self.spot_feed.price,
                                          ext_price=self.binance_feed.price if self.binance_feed else None)
            if t_now - getattr(self, "_last_wait_log_time", 0.0) >= 15.0:
                self._last_wait_log_time = t_now
                self.log_message("info", "Ok, aspetto il prossimo ciclo...")
            return

        # Guard: If the current time is at or past the expiration of the current cycle,
        # we have expired. Do not simulate fills, do not quote, do not execute anything.
        if self.strike_manager and t_now >= self.strike_manager.expiration_timestamp:
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
            
        oracle_spot = self.spot_feed.price
        spot = self._pricing_spot(oracle_spot)
        
        # Deduplicate rapid-fire CLOB snapshot bursts (reconnect storms).
        # Multiple book_snapshot events within 1s each reset q_shadow to full
        # liquidity, letting the engine fire identical trades 6× in 85ms.
        # Downgrade subsequent snapshots to delta updates within the 1s window.
        if is_snapshot:
            if t_now - self._last_snapshot_ts < 1.0:
                is_snapshot = False
            else:
                self._last_snapshot_ts = t_now

        # Reconcile local shadow order book
        ofi = self.shadow_book.update_book(bids, asks, is_snapshot, timestamp=t_now)
        
        # Update MTM Post-Fill calculations for both Maker and Taker trades
        top_b_mtm, top_a_mtm = self.shadow_book.get_market_top_of_book()
        if top_b_mtm and top_a_mtm:
            p_mid_mtm = 0.5 * (top_b_mtm[0] + top_a_mtm[0])
            for trade in self.hft_metrics["pending_mtm_trades"][:]:
                dt = t_now - trade["timestamp"]
                
                # Determine direction in YES terms
                side = trade["side"]
                exec_p = trade["exec_price"]
                if side == "BUY_YES":
                    dir_val = 1.0
                    ref_p = exec_p
                elif side == "BUY_NO":
                    dir_val = -1.0
                    ref_p = 1.0 - exec_p
                elif side == "SELL_YES":
                    dir_val = -1.0
                    ref_p = exec_p
                elif side == "SELL_NO":
                    dir_val = 1.0
                    ref_p = 1.0 - exec_p
                else:
                    dir_val = 1.0
                    ref_p = exec_p
                    
                if trade["mtm_1s"] is None and dt >= 1.0:
                    trade["mtm_1s"] = dir_val * (p_mid_mtm - ref_p)
                if trade["mtm_5s"] is None and dt >= 5.0:
                    trade["mtm_5s"] = dir_val * (p_mid_mtm - ref_p)
                if trade["mtm_15s"] is None and dt >= 15.0:
                    trade["mtm_15s"] = dir_val * (p_mid_mtm - ref_p)
                    # Fully resolved, move to completed list
                    try:
                        self.hft_metrics["pending_mtm_trades"].remove(trade)
                        self.hft_metrics["completed_mtm_trades"].append(trade)
                        if len(self.hft_metrics["completed_mtm_trades"]) > 200:
                            self.hft_metrics["completed_mtm_trades"].pop(0)
                    except ValueError:
                        pass
        
        # Resolve active Strike Price only after the cycle has actually started
        if self.strike_manager and (self.strike_manager.presumed_strike is None or self.strike_manager.presumed_strike == 0.0):
            if self.market_manager.current_expiry is not None:
                cycle_start_time = self.market_manager.current_expiry - self.market_manager.cycle_duration_sec
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
        if active_strike is None or active_strike <= 0.0:
            if t_now - getattr(self, "_last_strike_wait_log_time", 0.0) >= 10.0:
                self._last_strike_wait_log_time = t_now
                self.log_message("warning", "[StrikeManager] Lo strike non è ancora conosciuto. Il bot non trada, aspetta...")
            return
        
        # Calculate time remaining
        tau_sec = max(0.0, self.market_manager.current_expiry - t_now)
        
        # Log position in the final 60 seconds before settlement
        if tau_sec <= 60.0 and not getattr(self, "logged_60s_position", False):
            q_yes = self.client.positions.get("YES", 0.0)
            q_no = self.client.positions.get("NO", 0.0)
            net_q = q_yes - q_no
            self.log_message("warning", f"POSITION_60S_BEFORE_SETTLE: Net Position: {net_q} (YES: {q_yes}, NO: {q_no})")
            self.logged_60s_position = True
        
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
                base_halflife = self.config.merton.EMA_HALFLIFE_SEC
                # Dynamically compress halflife as expiration approaches to prevent lag
                # from keeping the probability stuck at ~0.50 when market collapses to 0/1.
                # e.g., at tau=30s, halflife is at most 3.0s; at tau=5s, it is 0.5s.
                halflife = min(base_halflife, max(0.1, tau_sec / 10.0))
                
                dt_ema = t_now - self._smoothed_p_yes_ts
                alpha = 1.0 - (2.718281828 ** (-dt_ema / halflife)) if halflife > 0.0 else 1.0
                self._smoothed_p_yes = alpha * p_yes_raw + (1.0 - alpha) * self._smoothed_p_yes
                self._smoothed_p_yes_ts = t_now
            p_yes = self._smoothed_p_yes
        else:
            p_yes = None
        
        # Record tick: raw update (for replay) + reconciled top of book + snapshot flag
        self.recorder.record_tick(t_now, spot, ofi, vol, bids, asks,
                                  top_bid=top_b_mtm, top_ask=top_a_mtm, is_snapshot=is_snapshot,
                                  oracle_price=oracle_spot,
                                  ext_price=self.binance_feed.price if self.binance_feed else None)
        
        p_mkt = self._get_market_implied_price(p_yes)
        
        # Update divergence filter on every tick
        if p_yes is not None and p_mkt is not None:
            self.divergence_filter.get_scale(p_yes, p_mkt, t_now)

        # Evaluate Trade Sizing and Execution
        
        # 1. Process market data through client to detect limit order fills
        context_state = {"timestamp": t_now, "strike_price": active_strike}
        fills = await self.client.process_market_data(context.bids_l2, context.asks_l2, context_state)
        
        for result in fills:
            if result.get("success"):
                instr = result.get("instruction")
                if instr:
                    self.total_trades += 1
                    self.hft_metrics["volume_executed"] += result["qty"]
                    
                    self.hft_metrics["pending_mtm_trades"].append({
                        "timestamp": t_now,
                        "side": result["side"],
                        "exec_price": result["price"],
                        "mtm_1s": None,
                        "mtm_5s": None,
                        "mtm_15s": None
                    })
                    
                    p_mkt_fill = self._get_market_implied_price(p_yes)
                    status_str = "MAKER_FILL_BUY" if "BUY" in result["side"] else "MAKER_FILL_SELL"
                    self.recorder.record_signal(
                        timestamp=t_now,
                        spot_price=spot,
                        strike=active_strike,
                        model_prob=p_yes,
                        implied_prob=p_mkt_fill if p_mkt_fill is not None else p_yes,
                        kelly_size=result["qty"],
                        status=status_str
                    )
                    
                    self.log_message("info", f"[MAKER FILL] {result['side']} filled at {result['price']:.4f} | Size: {result['qty']:.2f}")
                    if self.web_server:
                        self.web_server.broadcast_message({
                            "type": "trade_signal",
                            "signal": {"side": result["side"], "size": result["qty"], "vwap": result["price"], "ev": 0.0}
                        })
        
        # 2. Generate and process new quoting instructions. The router gets the raw
        #    probability already computed above (one pricing per tick); the EMA is
        #    for logging, the UI and the divergence filter.
        instructions = self.engine.evaluate_and_route(context, p_hat=p_yes_raw)
        
        # Format diagnostic decision for the dashboard
        bid_p = self.engine.execution_router.active_bid_price
        ask_p = self.engine.execution_router.active_ask_price
        # Regime of this tick's instructions (A maker / B taker / C unwind / REDUCE / PANIC).
        # Nothing to do means the router left the quotes as they were.
        regimes = [i.regime for i in instructions if getattr(i, "regime", None)]
        decision = {
            "side": "QUOTING",
            "size": self.engine.execution_router.active_bid_qty,
            "vwap": bid_p if bid_p > 0.0 else ask_p,
            "reason": f"Bid: {bid_p:.2f} Ask: {ask_p:.2f}",
            "regime": regimes[0] if regimes else None,
            "expected_slippage_bps": 0.0,
            "ev": 0.0
        }
        self.latest_decision_ref[0] = decision
        
        for instr in instructions:
            if instr.action == "NEW":
                self.hft_metrics["total_orders_sent"] += 1
            elif instr.action == "CANCEL":
                self.hft_metrics["total_orders_cancelled"] += 1
            elif instr.action == "REPLACE":
                self.hft_metrics["total_orders_sent"] += 1
                self.hft_metrics["total_orders_cancelled"] += 1
                
            res = await self.client.process_instruction(instr, context_state)
            
            # Taker executions are returned synchronously from process_instruction
            if instr.action == "NEW" and instr.regime in ("B", "PANIC") and res.get("success"):
                self.total_trades += 1
                p_mkt_exec = self._get_market_implied_price(p_yes)
                status_str = f"TAKER_{instr.side}" if instr.regime == "B" else f"PANIC_{instr.side}"
                self.recorder.record_signal(
                    timestamp=t_now,
                    spot_price=spot,
                    strike=active_strike,
                    model_prob=p_yes,
                    implied_prob=p_mkt_exec if p_mkt_exec is not None else p_yes,
                    kelly_size=res["qty"],
                    status=status_str
                )
                type_str = 'TAKER' if instr.regime == 'B' else 'PANIC'
                self.log_message("info", f"[{type_str} ORDER] Executed {instr.side} | Size: {res['qty']:.2f} | Price: {res['price']:.4f}")
                
                self.hft_metrics["total_taker_edge"] += abs(p_yes - res["price"] if p_yes is not None else 0.0)
                self.hft_metrics["total_taker_trades"] += 1
                
                self.hft_metrics["pending_mtm_trades"].append({
                    "timestamp": t_now,
                    "side": res["side"],
                    "exec_price": res["price"],
                    "mtm_1s": None,
                    "mtm_5s": None,
                    "mtm_15s": None
                })

        # Get Implied Market price and Edge — use REAL market book (q_real), never depleted
        p_mkt = self._get_market_implied_price(p_yes)
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
            


    async def _market_discovery_loop(self) -> None:
        """Periodically evaluates clock to trigger dynamic market Discovery and Rollovers."""
        while self.is_running:
            try:
                t_now = time.time()
                # If a rollover boundary is crossed
                rollover = await self.market_manager.update_market_cycle(t_now)
                if rollover:
                    self.waiting_for_first_rollover = False
                    self.logged_60s_position = False
                    self.log_message(
                        "info",
                        f"Market Rollover detected. New active cycle expiry: {self.market_manager.current_expiry} | "
                        f"Slug: {self.market_manager.current_slug}"
                    )
                    
                    # 1. Settle old positions at the actual expiration time (e.g. 01:30:00) instead of 15s early
                    if self.strike_manager and self.spot_feed.price is not None:
                        old_strike_manager = self.strike_manager
                        old_expiry = old_strike_manager.expiration_timestamp
                        
                        async def settle_at_expiry(sm: StrikeManager, expiry_time: float):
                            # Sleep until actual expiration time is reached
                            delay = max(0.0, expiry_time - time.time())
                            if delay > 0.0:
                                await asyncio.sleep(delay)

                            # Wait a brief moment to ensure all Chainlink ticks
                            # up to expiry_time have been received and stored.
                            await asyncio.sleep(1.0)

                            t_settle = time.time()

                            # ── Settlement price: LAST oracle tick BEFORE expiry ──────────
                            # get_last_tick_before() gives the final confirmed Chainlink
                            # price of the expiring cycle, uncontaminated by the first tick
                            # of the new cycle (which may already reflect the new round).
                            settle_tick = self.spot_feed.get_last_tick_before(expiry_time)
                            if settle_tick is not None:
                                settle_ts, settle_spot = settle_tick
                            else:
                                # Fallback: first tick at/after expiry if nothing before is cached
                                fallback = self.spot_feed.get_first_tick_after(expiry_time)
                                if fallback is not None:
                                    settle_ts, settle_spot = fallback
                                else:
                                    settle_spot = self.spot_feed.price or 0.0
                                    settle_ts = t_settle

                            settlement_strike = sm.get_strike(t_settle, settle_spot)

                            # ── Snapshot open positions BEFORE settle_positions() zeroes them ──
                            qty_yes_open = self.client.positions.get("YES", 0.0)
                            qty_no_open  = self.client.positions.get("NO",  0.0)
                            entry_yes    = self.client.entry_prices.get("YES", 0.0)
                            entry_no     = self.client.entry_prices.get("NO",  0.0)
                            has_position = qty_yes_open > 0.0 or qty_no_open > 0.0

                            # ── Execute settlement ────────────────────────────────────────────
                            result = self.client.settle_positions(
                                settlement_price=settle_spot,
                                strike_price=settlement_strike,
                                timestamp=t_settle
                            )

                            resolved_yes = settle_spot >= settlement_strike

                            # ── Build rich log message ────────────────────────────────────────
                            verdict = "YES WON ✓" if resolved_yes else "YES LOST ✗"
                            sep = "─" * 56

                            if has_position:
                                lines = [
                                    f"[SETTLEMENT] {sep}",
                                    f"  Expiry Tick : ${settle_spot:>10,.2f}  (ts: {settle_ts:.0f})",
                                    f"  Strike      : ${settlement_strike:>10,.2f}",
                                    f"  Outcome     : {verdict}",
                                ]
                                if qty_yes_open > 0.0:
                                    yes = result["yes"]
                                    lines.append(
                                        f"  YES pos     : {qty_yes_open:.2f} contracts @ entry ${entry_yes:.4f}"
                                        f"  →  payoff ${yes['gross_payoff']:.2f}"
                                        f"  |  PnL {'+' if yes['pnl']>=0 else ''}{yes['pnl']:.2f} USD"
                                        f"  ({'WON ✓' if yes['won'] else 'LOST ✗'})"
                                    )
                                if qty_no_open > 0.0:
                                    no = result["no"]
                                    lines.append(
                                        f"  NO  pos     : {qty_no_open:.2f} contracts @ entry ${entry_no:.4f}"
                                        f"  →  payoff ${no['gross_payoff']:.2f}"
                                        f"  |  PnL {'+' if no['pnl']>=0 else ''}{no['pnl']:.2f} USD"
                                        f"  ({'WON ✓' if no['won'] else 'LOST ✗'})"
                                    )
                                lines += [
                                    f"  Net PnL     : ${result['net_pnl']:>+10.2f} USD",
                                    f"  Cash after  : ${self.client.cash_balance:>10.2f} USD",
                                    f"[SETTLEMENT] {sep}",
                                ]
                                self.log_message("settle", "\n".join(lines))
                            else:
                                # No open positions — log a brief notice
                                self.log_message(
                                    "info",
                                    f"[SETTLEMENT] Expiry tick ${settle_spot:,.2f} | Strike ${settlement_strike:,.2f}"
                                    f" | {verdict} | No open positions."
                                )

                            if self.web_server:
                                self.web_server.broadcast_message({
                                    "type": "settlement_update",
                                    "settlement": {
                                        "settle_spot": settle_spot,
                                        "strike": settlement_strike,
                                        "resolved_yes": resolved_yes,
                                        "net_pnl": result["net_pnl"],
                                        "yes_qty": qty_yes_open,
                                        "no_qty": qty_no_open,
                                    }
                                })

                            
                        # Drop finished tasks first: one settle task is spawned per cycle and
                        # a long run would otherwise keep every completed one alive.
                        self._tasks = [t for t in self._tasks if not t.done()]
                        self._tasks.append(asyncio.create_task(settle_at_expiry(old_strike_manager, old_expiry)))
                        
                    # 2. Reset StrikeManager for the new cycle (force presumed_strike to 0.0 to resolve it only via the first spot tick)
                    self.strike_manager = StrikeManager(
                        presumed_strike=0.0,
                        expiration_timestamp=self.market_manager.current_expiry
                    )
                    self._smoothed_p_yes = None
                    self._smoothed_p_yes_ts = 0.0
                    self.strategy.reset()
                    if hasattr(self.engine, "reset"):
                        self.engine.reset()
                    self._current_active_orders = {"bid": None, "ask": None}
                    if hasattr(self.client, "cancel_all_orders"):
                        self.client.cancel_all_orders()


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
                                        presumed_strike=0.0,
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
                            cycle_start_time = self.market_manager.current_expiry - self.market_manager.cycle_duration_sec
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


    async def _ui_heartbeat_loop(self) -> None:
        """Periodically pushes state to the Web UI even if no market ticks arrive, preventing UI freeze."""
        while self.is_running:
            try:
                t_now = time.time()
                dt = t_now - self._last_tps_time
                if dt >= 1.0:
                    self.hft_metrics["tps"] = self._tick_count_tps / dt
                    self._tick_count_tps = 0
                    self._last_tps_time = t_now
                
                if self.web_server:
                    self._update_web_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"UI heartbeat error: {e}")
            await asyncio.sleep(0.33)

def select_strategy_interactively(default_strategy: str) -> str:
    """
    Prompts the user in the console to choose the active pricing strategy on startup.
    Supports default fallback if input is empty or if stdin is not a TTY.
    """
    import sys
    if not sys.stdin.isatty():
        return default_strategy
        
    print("\n" + "="*56)
    print("      POLYMARKET V2: ACTIVE STRATEGY SELECTION      ")
    print("="*56)
    print("  [1] Hawkes-Driven Merton (New Stochastic Intensity) [DEFAULT]")
    print("  [2] Legacy Merton (Posterior OFI Logit-Shift)")
    print("-"*56)
    
    try:
        choice = input(f"Select strategy [1 or 2, default '1']: ").strip()
        if not choice:
            return "merton"
        if choice == "1":
            return "merton"
        elif choice == "2":
            return "legacy_merton"
        elif choice.lower() in ["merton", "legacy_merton"]:
            return choice.lower()
        else:
            print(f"Invalid choice. Falling back to default: '{default_strategy}'\n")
            return default_strategy
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return default_strategy

async def main_async():
    config = SystemConfig()
    
    # ── Interactive Strategy Selection ──
    cli_strategy = None
    for arg in sys.argv:
        if arg.startswith("--strategy="):
            cli_strategy = arg.split("=")[1]
        elif arg == "--strategy" and sys.argv.index(arg) + 1 < len(sys.argv):
            cli_strategy = sys.argv[sys.argv.index(arg) + 1]
            
    if cli_strategy:
        strategy_name = cli_strategy
    else:
        strategy_name = select_strategy_interactively(config.STRATEGY_NAME)
        
    config.__dict__["STRATEGY_NAME"] = strategy_name
    print(f"\n>>> Starting Polymarket V2 with strategy: {strategy_name.upper()} <<<\n")
    
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
