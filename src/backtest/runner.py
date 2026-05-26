import logging
import time
import os
import json
import numpy as np
import polars as pl
from typing import Dict, Any, Optional, List, Tuple
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.core.strike_manager import StrikeManager
from src.ingestion.market_manager import MarketManager
from src.strategies.merton_strategy import MertonStrategy
from src.execution.shadow_book import ShadowOrderBook
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
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
        """Loads historical ticks from a CSV or Parquet file using Polars."""
        logger.info(f"[BacktestRunner] Loading tick logs from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ticks database not found: {file_path}")
            
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
            run_id=f"backtest_{int(time.time())}",
            buffer_size=1000
        )
        
        shadow_book = ShadowOrderBook()
        strategy = MertonStrategy(self.config)
        client = MockExecutionClient(self.config, recorder, shadow_book)
        engine = ExecutionEngine(self.config)
        
        # Expiry tracker variables
        current_expiry: Optional[int] = None
        strike_manager: Optional[StrikeManager] = None
        
        # Parse data column arrays for fast iteration
        timestamps = df["timestamp"].to_numpy()
        spot_prices = df["spot_price"].to_numpy()
        ofis = df["ofi"].to_numpy() if "ofi" in df.columns else np.zeros(len(df))
        vols = df["volatility"].to_numpy() if "volatility" in df.columns else np.full(len(df), self.config.merton.DEFAULT_SIGMA)
        
        bids_l2_raw = df["bids_l2"].to_list()
        asks_l2_raw = df["asks_l2"].to_list()
        
        total_ticks = len(df)
        capital_history = []
        trade_count = 0
        
        # Simulation Loop (Chronological ticks stream)
        for i in range(total_ticks):
            if progress_callback and i % max(1, total_ticks // 10) == 0:
                progress_callback(int((i / total_ticks) * 100))
            t = float(timestamps[i])
            spot = float(spot_prices[i])
            ofi = float(ofis[i])
            vol = float(vols[i])
            
            # Parse L2 book updates (JSON string check)
            try:
                bids_l2 = json.loads(bids_l2_raw[i]) if isinstance(bids_l2_raw[i], str) else bids_l2_raw[i]
                asks_l2 = json.loads(asks_l2_raw[i]) if isinstance(asks_l2_raw[i], str) else asks_l2_raw[i]
            except Exception:
                bids_l2 = []
                asks_l2 = []
                
            bids_l2 = [(float(p), float(q)) for p, q in bids_l2]
            asks_l2 = [(float(p), float(q)) for p, q in asks_l2]
            
            # 2. Rollover boundary checks
            if current_expiry is None or t >= current_expiry:
                if current_expiry is not None and strike_manager is not None:
                    # Settle active positions using the rollover spot price
                    settlement_strike = strike_manager.get_strike(t, spot)
                    client.settle_positions(settlement_price=spot, strike_price=settlement_strike, timestamp=t)
                
                # Roll to next 5-minute cycle expiration
                current_expiry = int(t) - (int(t) % 300) + 300
                # First tick price serves as the new cycle's strike K (ATM)
                strike_manager = StrikeManager(presumed_strike=spot, expiration_timestamp=current_expiry)
                # Capture the tick to resolve the strike immediately
                strike_manager.get_strike(t, spot)
                logger.info(f"[BacktestRunner] Rollover to cycle expiration: {current_expiry} | Strike K: ${spot:,.2f}")
            
            # 3. Update shadow order book proxy (reconciliation)
            shadow_book.update_book(bids_l2, asks_l2, is_snapshot=True)
            
            # 4. Construct Context & evaluate Option Fair Value
            active_strike = strike_manager.get_strike(t, spot)
            tau_sec = max(0.0, current_expiry - t)
            
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
            
            # Merton pricing solver
            p_yes = strategy.get_probability(context)
            
            # 5. Record Tick update in Parquet buffer
            recorder.record_tick(t, spot, ofi, vol, bids_l2, asks_l2)
            
            # 6. Engine Decisions & routing execution
            decision = engine.evaluate_and_trade(p_yes, context, client)
            
            if decision["side"] != "HOLD":
                trade_count += 1
                # Compute implied market price
                top_b, top_a = shadow_book.get_top_of_book()
                p_mkt = 0.5 * (top_b[0] + top_a[0]) if top_b and top_a else p_yes
                
                # Log Signal event
                recorder.record_signal(
                    timestamp=t,
                    spot_price=spot,
                    strike=active_strike,
                    model_prob=p_yes,
                    implied_prob=p_mkt,
                    kelly_size=decision["size"],
                    status=decision["side"]
                )
                
                # Execute simulated trade
                await client.execute_trade(
                    side=decision["side"],
                    qty=decision["size"],
                    price=decision["vwap"],
                    ev=decision["ev"],
                    expected_slippage_bps=decision["expected_slippage_bps"],
                    context_state={"timestamp": t, "strike_price": active_strike}
                )
                
            # Track current equity
            top_b, top_a = shadow_book.get_top_of_book()
            ref_yes_price = top_b[0] if top_b else 0.5
            equity = client.cash_balance + client.get_portfolio_value(ref_yes_price)
            capital_history.append(equity)
            
        # Force flush log files at completion
        recorder.flush()
        
        # Settle any remaining positions at simulation end
        if current_expiry is not None and strike_manager is not None:
            last_spot = spot_prices[-1]
            last_time = timestamps[-1]
            settlement_strike = strike_manager.get_strike(last_time, last_spot)
            client.settle_positions(settlement_price=last_spot, strike_price=settlement_strike, timestamp=last_time)
            # Record final equity
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
        for eq in capital_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                
        # Downsample capital history for chart rendering performance (max 300 points)
        step = max(1, len(capital_history) // 300)
        downsampled_cap = [float(c) for c in capital_history[::step]]
        if capital_history and (len(capital_history) - 1) % step != 0:
            downsampled_cap.append(float(capital_history[-1]))
            
        return {
            "final_cash": client.cash_balance,
            "net_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "total_trades": trade_count,
            "capital_history": downsampled_cap,
            "log_dir": recorder.log_dir
        }
