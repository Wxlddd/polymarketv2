import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import polars as pl
from src.core.interfaces import IDataRecorder

class DataRecorder(IDataRecorder):
    """
    Decoupled Data Logger. 
    Saves high-frequency orderbook and spot ticks in Parquet format,
    and trades/signals in immediately readable CSV format.
    """
    
    def __init__(
        self, 
        base_log_dir: str = "logs", 
        strategy_name: str = "merton", 
        run_id: Optional[str] = None, 
        buffer_size: int = 1000
    ):
        self.base_log_dir = base_log_dir
        self.strategy_name = strategy_name
        self.buffer_size = buffer_size
        
        # Resolve today's date (YYYY-MM-DD)
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Resolve run ID timestamp (YYYYMMDD_HHMMSS)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Structure: logs/YYYY-MM-DD/strategy_name/run_id_timestamp/
        self.log_dir = os.path.join(base_log_dir, today_str, strategy_name, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.ticks_path = os.path.join(self.log_dir, "ticks.parquet")
        self.signals_path = os.path.join(self.log_dir, "signals.csv")
        self.trades_path = os.path.join(self.log_dir, "trades.csv")
        
        # In-memory buffer for ticks
        self.tick_buffer: List[Dict[str, Any]] = []
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
    def _init_csv_files(self) -> None:
        """Initializes CSV log files with column headers if they do not exist."""
        # Signals header
        if not os.path.exists(self.signals_path):
            with open(self.signals_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "spot_price", 
                    "strike", 
                    "model_prob", 
                    "implied_prob", 
                    "kelly_size", 
                    "status"
                ])
                
        # Trades header
        if not os.path.exists(self.trades_path):
            with open(self.trades_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "side", 
                    "qty", 
                    "vwap", 
                    "p_market", 
                    "expected_slippage_bps", 
                    "realized_slippage_bps", 
                    "ev", 
                    "strike", 
                    "resolved_won", 
                    "pnl", 
                    "capital"
                ])

    def record_tick(
        self, 
        timestamp: float, 
        spot_price: float, 
        ofi: float, 
        volatility: float, 
        bids_l2: List[Tuple[float, float]], 
        asks_l2: List[Tuple[float, float]]
    ) -> None:
        """Logs a single market tick. Buffers tick in-memory and flushes periodically."""
        best_bid = bids_l2[0][0] if bids_l2 else None
        best_bid_qty = bids_l2[0][1] if bids_l2 else None
        best_ask = asks_l2[0][0] if asks_l2 else None
        best_ask_qty = asks_l2[0][1] if asks_l2 else None
        
        tick = {
            "timestamp": float(timestamp),
            "spot_price": float(spot_price),
            "best_bid": float(best_bid) if best_bid is not None else None,
            "best_bid_qty": float(best_bid_qty) if best_bid_qty is not None else None,
            "best_ask": float(best_ask) if best_ask is not None else None,
            "best_ask_qty": float(best_ask_qty) if best_ask_qty is not None else None,
            "ofi": float(ofi),
            "volatility": float(volatility),
            "bids_l2": json.dumps(bids_l2),
            "asks_l2": json.dumps(asks_l2)
        }
        
        self.tick_buffer.append(tick)
        
        if len(self.tick_buffer) >= self.buffer_size:
            self._flush_ticks_to_parquet()

    def record_signal(
        self, 
        timestamp: float, 
        spot_price: float, 
        strike: float, 
        model_prob: float, 
        implied_prob: float, 
        kelly_size: float, 
        status: str
    ) -> None:
        """Immediately appends strategy signals to CSV for human-readable inspection."""
        try:
            with open(self.signals_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    float(timestamp), 
                    float(spot_price), 
                    float(strike), 
                    float(model_prob), 
                    float(implied_prob), 
                    float(kelly_size), 
                    str(status)
                ])
        except Exception as e:
            print(f"[DataRecorder Error] Failed to write signal to CSV: {e}")

    def record_trade(
        self, 
        timestamp: float, 
        side: str, 
        qty: float, 
        vwap: float, 
        p_market: float, 
        expected_slippage_bps: float, 
        realized_slippage_bps: float, 
        ev: float, 
        strike: float, 
        resolved_won: bool, 
        pnl: float, 
        capital: float
    ) -> None:
        """Immediately appends trade outcomes to CSV for human-readable inspection."""
        try:
            with open(self.trades_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    float(timestamp), 
                    str(side), 
                    float(qty), 
                    float(vwap), 
                    float(p_market), 
                    float(expected_slippage_bps), 
                    float(realized_slippage_bps), 
                    float(ev), 
                    float(strike), 
                    bool(resolved_won), 
                    float(pnl), 
                    float(capital)
                ])
        except Exception as e:
            print(f"[DataRecorder Error] Failed to write trade to CSV: {e}")

    def _flush_ticks_to_parquet(self) -> None:
        """Writes buffered ticks into a compressed Parquet database file using Polars."""
        if not self.tick_buffer:
            return
            
        try:
            new_df = pl.DataFrame(self.tick_buffer)
            
            # Cast L2 JSON string columns to prevent schema mismatches
            new_df = new_df.cast({
                "bids_l2": pl.String,
                "asks_l2": pl.String
            })
            
            if os.path.exists(self.ticks_path):
                try:
                    existing_df = pl.read_parquet(self.ticks_path)
                    combined_df = pl.concat([existing_df, new_df])
                    combined_df.write_parquet(self.ticks_path, compression="zstd")
                except Exception as e:
                    # If reading or writing the main file fails (e.g. file lock on Windows),
                    # write to a new 'recovery' chunk to avoid losing data and prevent 
                    # the buffer from growing indefinitely.
                    ts_suffix = int(time.time() * 1000)
                    recovery_path = self.ticks_path.replace(".parquet", f"_rec_{ts_suffix}.parquet")
                    print(f"[DataRecorder Warning] Main Parquet lock/error, using recovery: {recovery_path} ({e})")
                    new_df.write_parquet(recovery_path, compression="zstd")
            else:
                new_df.write_parquet(self.ticks_path, compression="zstd")
                
            self.tick_buffer.clear()
        except Exception as e:
            # Critical failure (e.g. OOM or Schema error in DataFrame creation)
            print(f"[DataRecorder Error] CRITICAL failure to write Parquet log: {e}")
            # Clear buffer if it gets too large to prevent memory leak
            if len(self.tick_buffer) > self.buffer_size * 5:
                print(f"[DataRecorder Error] Tick buffer cleared due to persistent failures to save memory.")
                self.tick_buffer.clear()

    def flush(self) -> None:
        """Forces all buffered records to disk."""
        self._flush_ticks_to_parquet()
