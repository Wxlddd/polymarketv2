import logging
import time
import os
import json
import numpy as np
import polars as pl
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.core.strike_manager import StrikeManager
from src.ingestion.market_manager import MarketManager
from src.strategies.factory import StrategyFactory
from src.execution.shadow_book import ShadowOrderBook
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.execution.divergence_filter import DivergenceVelocityFilter
from src.logging.recorder import DataRecorder

logger = logging.getLogger("BacktestRunner")

class BacktestRunner:
    """
    Event-driven Historical Backtester.
    Replays ticks Level 2 + Spot price feeds through the live options pricing,
    shadow book proxy, and execution engine pipeline.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config

    def load_ticks_file(self, file_path: str) -> pl.DataFrame:
        """Loads historical ticks from a CSV or Parquet file (or directory of parquets) using Polars."""
        logger.info(f"[BacktestRunner] Loading tick logs from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ticks database not found: {file_path}")
            
        if os.path.isdir(file_path):
            # Load all parquet files in directory (handles chunked/recovery files)
            import glob
            # Only tick segments (ticks.parquet, ticks_002.parquet, ...). A session directory
            # also holds prints.parquet, whose schema would break the concat.
            files = glob.glob(os.path.join(file_path, "ticks*.parquet"))
            if not files:
                files = [f for f in glob.glob(os.path.join(file_path, "*.parquet"))
                         if os.path.basename(f) != "prints.parquet"]
            if not files:
                raise FileNotFoundError(f"No parquet files found in directory: {file_path}")
            files.sort()
            logger.info(f"[BacktestRunner] Found {len(files)} parquet files in directory. Merging...")
            dfs = []
            for f in files:
                try:
                    dfs.append(pl.read_parquet(f))
                except Exception as e:
                    logger.warning(f"[BacktestRunner] Failed to read {f}: {e}")
            if not dfs:
                 raise ValueError(f"Failed to load any valid parquet files from {file_path}")
            df = pl.concat(dfs)
        else:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".parquet":
                df = pl.read_parquet(file_path)
            else:
                df = pl.read_csv(file_path)
            
        df = df.sort("timestamp")
        logger.info(f"[BacktestRunner] Successfully loaded {len(df)} historical ticks.")
        return df

    async def run(self, df: pl.DataFrame, strategy_name: str = "merton", progress_callback=None) -> Dict[str, Any]:
        """
        Runs the event-driven backtest simulation.
        Streams historical ticks level by level, matching production execution pipelines.
        """
        logger.info(f"[BacktestRunner] Commencing historical replay simulation on {len(df)} ticks...")
        
        # 1. Initialize core system components
        recorder = DataRecorder(
            base_log_dir=self.config.LOG_DIR,
            strategy_name=strategy_name,
            run_id=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            buffer_size=1000
        )
        
        shadow_book = ShadowOrderBook()
        shadow_book.is_backtest = True
        strategy = StrategyFactory.get_strategy(self.config.STRATEGY_NAME, self.config)
        client = MockExecutionClient(self.config, recorder, shadow_book)
        client.is_backtest = True
        # Same unified maker/taker engine and divergence filter as the live orchestrator,
        # so a backtest exercises exactly the code path that trades live.
        engine = ExecutionEngine(strategy, client, self.config)
        divergence_filter = DivergenceVelocityFilter(self.config)
        engine.divergence_filter = divergence_filter

        # Expiry tracker variables
        current_expiry: Optional[int] = None
        strike_manager: Optional[StrikeManager] = None

        # Shadow book: only the first tick of each 5-min cycle is a full snapshot.
        # Subsequent ticks are delta updates so that paper_execute depletions persist
        # across ticks.  Resetting to is_snapshot=True on every tick was the primary
        # cause of infinite-liquidity fills in the old backtest.
        cycle_snapshot_sent: bool = False

        # Time-based EMA for p_yes — mirrors the live orchestrator but uses actual
        # tick timestamps so the smoothing is rate-independent.
        p_yes_ema: Optional[float] = None
        p_yes_ema_ts: float = 0.0

        # Parse data column arrays for fast iteration
        timestamps = df["timestamp"].to_numpy()
        spot_prices = df["spot_price"].to_numpy()
        ofis = df["ofi"].to_numpy() if "ofi" in df.columns else np.zeros(len(df))
        vols = df["volatility"].to_numpy() if "volatility" in df.columns else np.full(len(df), self.config.merton.DEFAULT_SIGMA)

        bids_l2_raw = df["bids_l2"].to_list()
        asks_l2_raw = df["asks_l2"].to_list()

        # Pre-parse and pre-convert L2 books to avoid interpreter overhead inside the loop
        bids_l2_parsed = []
        for x in bids_l2_raw:
            item = []
            if isinstance(x, str):
                try:
                    item = json.loads(x)
                except Exception:
                    pass
            elif x is not None:
                item = x
            bids_l2_parsed.append([(float(p), float(q)) for p, q in item])
            
        asks_l2_parsed = []
        for x in asks_l2_raw:
            item = []
            if isinstance(x, str):
                try:
                    item = json.loads(x)
                except Exception:
                    pass
            elif x is not None:
                item = x
            asks_l2_parsed.append([(float(p), float(q)) for p, q in item])

        # Snapshot flags: recorded explicitly by newer recorders. Older files lack the
        # column, and their first tick of a cycle is often a delta from the previous
        # market's feed rather than the new book; a raw update carrying >= 5 levels on
        # both sides is a book snapshot, price_change deltas carry a handful at most.
        if "is_snapshot" in df.columns:
            snapshot_flags = [bool(x) for x in df["is_snapshot"].fill_null(False).to_list()]
        else:
            snapshot_flags = [len(b) >= 5 and len(a) >= 5 for b, a in zip(bids_l2_parsed, asks_l2_parsed)]

        total_ticks = len(df)
        capital_history = []
        trade_count = 0
        pending_orders = []
        
        # Caching state to avoid redundant numerical integration
        last_eval_spot = None
        last_eval_time = 0.0
        last_p_yes_raw = None
        
        last_evaluated_p_yes = None
        last_evaluated_best_bid = None
        last_evaluated_best_ask = None
        
        # Simulation Loop (Chronological ticks stream)
        for i in range(total_ticks):
            if progress_callback and i % max(1, total_ticks // 100) == 0:
                progress_callback(int((i / total_ticks) * 100))
            t = float(timestamps[i])
            spot = float(spot_prices[i])
            ofi = float(ofis[i])
            vol = float(vols[i])

            bids_l2 = bids_l2_parsed[i]
            asks_l2 = asks_l2_parsed[i]

            # 2. Rollover boundary checks
            if current_expiry is None or t >= current_expiry:
                if current_expiry is not None and strike_manager is not None:
                    # Settle active positions using the rollover spot price
                    settlement_strike = strike_manager.get_strike(t, spot)
                    client.settle_positions(settlement_price=spot, strike_price=settlement_strike, timestamp=t)

                # Roll to next cycle expiration (CYCLE_DURATION_SEC-aligned, e.g. 300s for 5m or 14400s for 4h)
                cycle = self.config.polymarket.CYCLE_DURATION_SEC
                current_expiry = int(t) - (int(t) % cycle) + cycle
                # First tick price serves as the new cycle's strike K (ATM)
                strike_manager = StrikeManager(presumed_strike=spot, expiration_timestamp=current_expiry)
                strike_manager.get_strike(t, spot)
                logger.info(f"[BacktestRunner] Rollover to cycle expiration: {current_expiry} | Strike K: ${spot:,.2f}")
                
                # New cycle → force a full snapshot for the shadow book and reset EMA
                cycle_snapshot_sent = False
                p_yes_ema = None
                p_yes_ema_ts = 0.0
                strategy.reset()
                engine.reset()
                client.cancel_all_orders()
            
            # 3. Update shadow order book proxy.
            # First tick of each cycle and every recorded/detected snapshot replace the book
            # (clearing stale levels from the previous market); everything else is a delta,
            # so paper_execute depletions are preserved between snapshots.
            is_snap = (not cycle_snapshot_sent) or snapshot_flags[i]
            shadow_book.update_book(bids_l2, asks_l2, is_snapshot=is_snap, timestamp=t)
            cycle_snapshot_sent = True
            
            # 3.5 Execute queued taker orders whose simulated network latency has elapsed.
            # Taker instructions are delayed in *event time* (not wall-clock) so the
            # replay stays instantaneous while still modelling the 150-300ms round trip.
            ready_orders = [o for o in pending_orders if o["exec_time"] <= t]
            pending_orders = [o for o in pending_orders if o["exec_time"] > t]

            for order in sorted(ready_orders, key=lambda x: x["exec_time"]):
                res = await client.process_instruction(order["instruction"], order["context_state"])
                if res.get("success"):
                    trade_count += 1
            
            # 4. Construct lightweight Context (without costly book sorting)
            active_strike = strike_manager.get_strike(t, spot)
            tau_sec = max(0.0, current_expiry - t)

            context = MarketContext(
                timestamp=t,
                spot_price=spot,
                strike_price=active_strike,
                tau_seconds=tau_sec,
                volatility=vol,
                ofi=ofi,
                bids_l2=[],
                asks_l2=[]
            )

            # Merton pricing solver (optimized to skip redundant adaptive integrations)
            should_eval = (
                last_p_yes_raw is None
                or last_eval_spot is None
                or abs(spot - last_eval_spot) > 1e-6
                or t - last_eval_time >= 5.0
                or is_snap
            )
            
            if should_eval:
                p_yes_raw = strategy.get_probability(context)
                last_p_yes_raw = p_yes_raw
                last_eval_spot = spot
                last_eval_time = t
            else:
                p_yes_raw = last_p_yes_raw
                # Keep the calibrator's tick buffer perfectly updated if supported
                if hasattr(strategy, "vol_calibrator"):
                    strategy.vol_calibrator.add_tick(spot, t)

            # Time-based EMA on p_yes — same logic as the live orchestrator.
            # Prevents raw OFI oscillations from generating a trade signal every tick.
            if p_yes_raw is not None:
                if p_yes_ema is None:
                    p_yes_ema = p_yes_raw
                    p_yes_ema_ts = t
                else:
                    base_halflife = self.config.merton.EMA_HALFLIFE_SEC
                    halflife = min(base_halflife, max(0.1, context.tau_seconds / 10.0))
                    
                    dt_ema = t - p_yes_ema_ts
                    alpha = 1.0 - (2.718281828 ** (-dt_ema / halflife)) if halflife > 0.0 else 1.0
                    p_yes_ema = alpha * p_yes_raw + (1.0 - alpha) * p_yes_ema
                    p_yes_ema_ts = t
                p_yes = p_yes_ema
            else:
                p_yes = None

            # 5. Record Tick update in Parquet buffer (bypassed in backtest to eliminate I/O bottleneck)
            # recorder.record_tick(t, spot, ofi, vol, bids_l2, asks_l2)

            # 6. Engine decisions & routing — mirrors LiveOrchestrator._clob_callback exactly.
            context_state = {"timestamp": t, "strike_price": active_strike, "volatility": vol}

            # A. Evaluate resting maker orders against the updated book (queue depth + fills).
            #    This is the same fill simulator the live paper client uses.
            fills = await client.process_market_data(
                shadow_book.get_sorted_bids(), shadow_book.get_sorted_asks(), context_state
            )
            fill_occurred = False
            for result in fills:
                if result.get("success"):
                    trade_count += 1
                    fill_occurred = True
                    recorder.record_signal(
                        timestamp=t,
                        spot_price=spot,
                        strike=active_strike,
                        model_prob=p_yes,
                        implied_prob=result["price"],
                        kelly_size=result["qty"],
                        status="MAKER_FILL_BUY" if "BUY" in result["side"] else "MAKER_FILL_SELL"
                    )

            # B. Divergence velocity filter (toxic-flow guard) on every tick, as in live.
            top_b_sh, top_a_sh = shadow_book.get_market_top_of_book()
            cur_best_bid = top_b_sh[0] if top_b_sh else None
            cur_best_ask = top_a_sh[0] if top_a_sh else None
            if p_yes is not None and (cur_best_bid is not None or cur_best_ask is not None):
                if cur_best_bid is not None and cur_best_ask is not None:
                    p_mkt_div = 0.5 * (cur_best_bid + cur_best_ask)
                else:
                    p_mkt_div = cur_best_bid if cur_best_bid is not None else cur_best_ask
                divergence_filter.get_scale(p_yes, p_mkt_div, t)

            # C. Quoting evaluation bypass: skip the router when nothing it depends on changed.
            should_eval_quote = (
                last_evaluated_p_yes is None
                or p_yes is None
                or should_eval
                or abs(p_yes - last_evaluated_p_yes) > 1e-5
                or cur_best_bid != last_evaluated_best_bid
                or cur_best_ask != last_evaluated_best_ask
                or fill_occurred
                or is_snap
            )

            if should_eval_quote and not pending_orders:
                context = MarketContext(
                    timestamp=t,
                    spot_price=spot,
                    strike_price=active_strike,
                    tau_seconds=tau_sec,
                    volatility=vol,
                    ofi=ofi,
                    bids_l2=shadow_book.get_sorted_bids(),
                    asks_l2=shadow_book.get_sorted_asks()
                )
                instructions = engine.evaluate_and_route(context, p_hat=p_yes_raw)

                last_evaluated_p_yes = p_yes
                last_evaluated_best_bid = cur_best_bid
                last_evaluated_best_ask = cur_best_ask

                for instr in instructions:
                    if instr.action == "NEW" and instr.regime in ("B", "PANIC"):
                        # Taker / panic sweep: log the signal now, execute after simulated latency.
                        p_mkt = 0.5 * (cur_best_bid + cur_best_ask) if (cur_best_bid is not None and cur_best_ask is not None) else p_yes
                        recorder.record_signal(
                            timestamp=t,
                            spot_price=spot,
                            strike=active_strike,
                            model_prob=p_yes,
                            implied_prob=p_mkt,
                            kelly_size=instr.qty,
                            status=f"TAKER_{instr.side}" if instr.regime == "B" else f"PANIC_{instr.side}"
                        )
                        delay = np.random.uniform(0.150, 0.300)
                        pending_orders.append({
                            "exec_time": t + delay,
                            "instruction": instr,
                            "context_state": {
                                "timestamp": t,
                                "strike_price": active_strike,
                                "volatility": vol,
                                "limit_price": instr.price
                            }
                        })
                    else:
                        # Maker NEW/REPLACE rest in the client book; CANCEL removes them.
                        await client.process_instruction(instr, context_state)

            # Track current equity using mid-price from REAL book for consistent MTM
            top_b, top_a = shadow_book.get_market_top_of_book()
            if top_b and top_a:
                ref_yes_price = 0.5 * (top_b[0] + top_a[0])
            elif top_b:
                ref_yes_price = top_b[0]
            else:
                ref_yes_price = 0.5
            equity = client.cash_balance + client.get_portfolio_value(ref_yes_price)
            capital_history.append(equity)
            
        # Force flush log files at completion
        recorder.flush()
        
        # Settle any remaining positions at simulation end
        if current_expiry is not None and strike_manager is not None:
            last_spot = float(spot_prices[-1])
            last_time = float(timestamps[-1])
            settlement_strike = strike_manager.get_strike(last_time, last_spot)
            client.settle_positions(settlement_price=last_spot, strike_price=settlement_strike, timestamp=last_time)
            capital_history.append(client.cash_balance)
            recorder.flush()
 
        if progress_callback:
            progress_callback(100)
 
        # Compile final stats summary
        total_pnl = client.cash_balance - self.config.arbitrage.INITIAL_CAPITAL
        total_return_pct = total_pnl / self.config.arbitrage.INITIAL_CAPITAL * 100.0
        
        # Calculate max drawdown
        peak = self.config.arbitrage.INITIAL_CAPITAL
        max_dd = 0.0
        max_dd_usd = 0.0
        for eq in capital_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
            dd_usd = peak - eq
            if dd > max_dd:
                max_dd = dd
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
                
        # Downsample capital history for chart rendering performance (max 300 points)
        step = max(1, len(capital_history) // 300)
        downsampled_cap = [float(c) for c in capital_history[::step]]
        if capital_history and (len(capital_history) - 1) % step != 0:
            downsampled_cap.append(float(capital_history[-1]))

        # Calculate advanced performance metrics from client.realized_trades
        realized = client.realized_trades
        wins = [t for t in realized if t["pnl"] > 0.0]
        losses = [t for t in realized if t["pnl"] < 0.0]
        flats = [t for t in realized if t["pnl"] == 0.0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        flat_trades = len(flats)
        total_realized_trades = len(realized)

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = sum(abs(t["pnl"]) for t in losses)

        import math
        def safe_json_float(val: float) -> Any:
            if math.isinf(val) or math.isnan(val):
                return "N/A"
            return float(val)

        profit_factor = safe_json_float(gross_profit / gross_loss) if gross_loss > 0.0 else ("N/A" if gross_profit == 0.0 else "∞")
        win_rate_pct = (winning_trades / total_realized_trades * 100.0) if total_realized_trades > 0 else 0.0
        
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0.0
        
        win_loss_ratio = safe_json_float(avg_win / avg_loss) if avg_loss > 0.0 else ("N/A" if avg_win == 0.0 else "∞")

        serialized_trades = []
        for t in realized:
            serialized_trades.append({
                "timestamp": float(t["timestamp"]),
                "side": str(t["side"]),
                "qty": float(t["qty"]),
                "entry_price": float(t["entry_price"]),
                "exit_price": float(t["exit_price"]),
                "pnl": float(t["pnl"]),
                "won": bool(t["won"])
            })

        results = {
            "final_cash": client.cash_balance,
            "net_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "max_drawdown_usd": max_dd_usd,
            "total_trades": total_realized_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "flat_trades": flat_trades,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "win_rate_pct": win_rate_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "capital_history": downsampled_cap,
            "realized_trades": serialized_trades,
            "log_dir": recorder.log_dir
        }

        # Write summary.json to the backtest log directory
        summary_path = os.path.join(recorder.log_dir, "summary.json")
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "ticks_processed": total_ticks,
            "initial_capital": self.config.arbitrage.INITIAL_CAPITAL,
            "final_cash": client.cash_balance,
            "net_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "max_drawdown_usd": max_dd_usd,
            "total_trades": total_realized_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "flat_trades": flat_trades,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "win_rate_pct": win_rate_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "realized_trades": serialized_trades
        }
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"[BacktestRunner] Summary saved to: {summary_path}")
        except Exception as e:
            logger.error(f"[BacktestRunner] Failed to write summary.json: {e}")

        return results
