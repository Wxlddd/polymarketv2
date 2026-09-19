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
        buffer_size: int = 1000,
        tick_segment_sec: float = 600.0
    ):
        self.base_log_dir = base_log_dir
        self.strategy_name = strategy_name
        self.buffer_size = buffer_size
        # A ParquetWriter only writes its footer on close(), so a file still being written
        # is unreadable: a kill or a power cut loses the whole session. Rotating to a new
        # segment every tick_segment_sec caps that loss at one segment. 0 disables rotation.
        self.tick_segment_sec = tick_segment_sec
        self._segment_index = 1
        self._segment_opened_ts = 0.0
        
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
            ('asks_l2', pa.string()),
            ('is_snapshot', pa.bool_())
        ])
        self.writer = None

        # Exchange trade prints (last_trade_price) — separate file, same buffering
        self.prints_path = os.path.join(self.log_dir, "prints.parquet")
        self.print_buffer: List[Dict[str, Any]] = []
        self._prints_schema = pa.schema([
            ('timestamp', pa.float64()),      # local receive time
            ('exchange_ts', pa.float64()),    # exchange timestamp (s), null if absent
            ('price', pa.float64()),
            ('size', pa.float64()),
            ('side', pa.string())             # aggressor side as reported: BUY / SELL
        ])
        self.prints_writer = None
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
    def _check_and_rollover_date(self) -> None:
        """Midnight rollover: Dynamically switches the log directory if the date changes."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        if today_str != self.current_date:
            # 1. Flush any buffered ticks first
            self._flush_ticks_to_parquet()
            self._flush_prints_to_parquet()
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            if self.prints_writer is not None:
                self.prints_writer.close()
                self.prints_writer = None

            # 2. Update current date and logging directory
            print(f"[DataRecorder] Midnight rollover detected. Moving from {self.current_date} to {today_str}")
            self.current_date = today_str
            self.log_dir = os.path.join(self.base_log_dir, today_str, self.strategy_name, self.run_id)
            os.makedirs(self.log_dir, exist_ok=True)

            self.ticks_path = os.path.join(self.log_dir, "ticks.parquet")
            self.prints_path = os.path.join(self.log_dir, "prints.parquet")
            self.signals_path = os.path.join(self.log_dir, "signals.csv")
            self.trades_path = os.path.join(self.log_dir, "trades.csv")
            # New day, new directory: segment numbering restarts at ticks.parquet
            self._segment_index = 1
            
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
        asks_l2: List[Tuple[float, float]],
        top_bid: Optional[Tuple[float, float]] = None,
        top_ask: Optional[Tuple[float, float]] = None,
        is_snapshot: bool = False
    ) -> None:
        """
        Logs a single market tick. bids_l2/asks_l2 are the raw feed update (needed to replay
        the book); best_* come from the reconciled top of book passed in by the caller.
        Before top_bid/top_ask existed the best_* columns held the first level of the
        delta, which is not the top of book — treat those columns in old files as unreliable.
        """
        self._check_and_rollover_date()

        tick = {
            "timestamp": float(timestamp),
            "spot_price": float(spot_price),
            "best_bid": float(top_bid[0]) if top_bid else None,
            "best_bid_qty": float(top_bid[1]) if top_bid else None,
            "best_ask": float(top_ask[0]) if top_ask else None,
            "best_ask_qty": float(top_ask[1]) if top_ask else None,
            "ofi": float(ofi),
            "volatility": float(volatility),
            "bids_l2": json.dumps(bids_l2),
            "asks_l2": json.dumps(asks_l2),
            "is_snapshot": bool(is_snapshot)
        }

        self.tick_buffer.append(tick)

        if len(self.tick_buffer) >= self.buffer_size:
            self._flush_ticks_to_parquet()

    def record_print(self, timestamp: float, price: float, size: float, side: str, exchange_ts: Optional[float] = None) -> None:
        """Logs an exchange trade print for the YES token. Buffered like ticks, written to prints.parquet."""
        self._check_and_rollover_date()
        self.print_buffer.append({
            "timestamp": float(timestamp),
            "exchange_ts": float(exchange_ts) if exchange_ts is not None else None,
            "price": float(price),
            "size": float(size),
            "side": str(side)
        })
        if len(self.print_buffer) >= self.buffer_size:
            self._flush_prints_to_parquet()

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

    def _tick_segment_path(self) -> str:
        """First segment keeps the plain ticks.parquet name; later ones get a suffix.
        A short run is one file exactly as before; the backtester reads the whole directory."""
        if self._segment_index <= 1:
            return self.ticks_path
        base, ext = os.path.splitext(self.ticks_path)
        return f"{base}_{self._segment_index:03d}{ext}"

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

            # Close the current segment once it is old enough, so it gets its footer and
            # becomes readable. The next flush starts a new segment file.
            if (self.writer is not None and self.tick_segment_sec > 0
                    and time.time() - self._segment_opened_ts >= self.tick_segment_sec):
                self.writer.close()
                self.writer = None
                self._segment_index += 1

            if self.writer is None:
                path = self._tick_segment_path()
                # If file exists at startup, load existing data to initialize writer cleanly
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    try:
                        existing_table = pq.read_table(path)
                        self.writer = pq.ParquetWriter(path, self._arrow_schema, compression="zstd")
                        self.writer.write_table(existing_table)
                    except Exception as e:
                        # Fail-safe fallback if the file is corrupted
                        self.writer = pq.ParquetWriter(path, self._arrow_schema, compression="zstd")
                else:
                    self.writer = pq.ParquetWriter(path, self._arrow_schema, compression="zstd")
                self._segment_opened_ts = time.time()

            self.writer.write_table(table)
            self.tick_buffer.clear()
        except Exception as e:
            print(f"[DataRecorder Error] CRITICAL failure to write Parquet log: {e}")
            if len(self.tick_buffer) > self.buffer_size * 5:
                print(f"[DataRecorder Error] Tick buffer cleared due to persistent failures to save memory.")
                self.tick_buffer.clear()

    def _flush_prints_to_parquet(self) -> None:
        if not self.print_buffer:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            data_dict = {col: [row.get(col) for row in self.print_buffer] for col in self._prints_schema.names}
            table = pa.Table.from_pydict(data_dict, schema=self._prints_schema)
            if self.prints_writer is None:
                self.prints_writer = pq.ParquetWriter(self.prints_path, self._prints_schema, compression="zstd")
                if os.path.exists(self.prints_path) and os.path.getsize(self.prints_path) > 0:
                    try:
                        self.prints_writer.write_table(pq.read_table(self.prints_path))
                    except Exception:
                        pass
            self.prints_writer.write_table(table)
            self.print_buffer.clear()
        except Exception as e:
            print(f"[DataRecorder Error] Failed to write prints Parquet log: {e}")
            if len(self.print_buffer) > self.buffer_size * 5:
                self.print_buffer.clear()

    def flush(self) -> None:
        """Forces all buffered records to disk and closes the active file writers."""
        self._flush_ticks_to_parquet()
        self._flush_prints_to_parquet()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.prints_writer is not None:
            self.prints_writer.close()
            self.prints_writer = None
