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
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Resolve run ID timestamp (YYYYMMDD_HHMMSS)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
            
        # Structure: logs/YYYY-MM-DD/strategy_name/run_id_timestamp/
        self.log_dir = os.path.join(base_log_dir, self.current_date, strategy_name, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.ticks_path = os.path.join(self.log_dir, "ticks.parquet")
        self.signals_path = os.path.join(self.log_dir, "signals.csv")
        self.trades_path = os.path.join(self.log_dir, "trades.csv")
        
        # In-memory buffer for ticks
        self.tick_buffer: List[Dict[str, Any]] = []
        
        import pyarrow as pa
        self._arrow_schema = pa.schema([
            ('timestamp', pa.float64()),
            ('spot_price', pa.float64()),
            ('best_bid', pa.float64()),
            ('best_bid_qty', pa.float64()),
            ('best_ask', pa.float64()),
            ('best_ask_qty', pa.float64()),
            ('ofi', pa.float64()),
            ('volatility', pa.float64()),
            ('bids_l2', pa.string()),
            ('asks_l2', pa.string())
        ])
        self.writer = None
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
    def _check_and_rollover_date(self) -> None:
        """Midnight rollover: Dynamically switches the log directory if the date changes."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        if today_str != self.current_date:
            # 1. Flush any buffered ticks first
            self._flush_ticks_to_parquet()
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            
            # 2. Update current date and logging directory
            print(f"[DataRecorder] Midnight rollover detected. Moving from {self.current_date} to {today_str}")
            self.current_date = today_str
            self.log_dir = os.path.join(self.base_log_dir, today_str, self.strategy_name, self.run_id)
            os.makedirs(self.log_dir, exist_ok=True)
            
            self.ticks_path = os.path.join(self.log_dir, "ticks.parquet")
            self.signals_path = os.path.join(self.log_dir, "signals.csv")
            self.trades_path = os.path.join(self.log_dir, "trades.csv")
            
            # 3. Re-initialize CSV files in the new directory
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
        self._check_and_rollover_date()
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
        self._check_and_rollover_date()
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
        self._check_and_rollover_date()
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
        """Writes buffered ticks into a compressed Parquet database file using PyArrow ParquetWriter."""
        if not self.tick_buffer:
            return
            
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Map tick dictionary buffer to Arrow format
            data_dict = {col: [] for col in self._arrow_schema.names}
            for tick in self.tick_buffer:
                for col in self._arrow_schema.names:
                    data_dict[col].append(tick.get(col, None))
            
            table = pa.Table.from_pydict(data_dict, schema=self._arrow_schema)
            
            if self.writer is None:
                # If file exists at startup, load existing data to initialize writer cleanly
                if os.path.exists(self.ticks_path) and os.path.getsize(self.ticks_path) > 0:
                    try:
                        existing_table = pq.read_table(self.ticks_path)
                        self.writer = pq.ParquetWriter(self.ticks_path, self._arrow_schema, compression="zstd")
                        self.writer.write_table(existing_table)
                    except Exception as e:
                        # Fail-safe fallback if the file is corrupted
                        self.writer = pq.ParquetWriter(self.ticks_path, self._arrow_schema, compression="zstd")
                else:
                    self.writer = pq.ParquetWriter(self.ticks_path, self._arrow_schema, compression="zstd")
            
            self.writer.write_table(table)
            self.tick_buffer.clear()
        except Exception as e:
            print(f"[DataRecorder Error] CRITICAL failure to write Parquet log: {e}")
            if len(self.tick_buffer) > self.buffer_size * 5:
                print(f"[DataRecorder Error] Tick buffer cleared due to persistent failures to save memory.")
                self.tick_buffer.clear()

    def flush(self) -> None:
        """Forces all buffered records to disk and closes the active file writer."""
        self._flush_ticks_to_parquet()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
