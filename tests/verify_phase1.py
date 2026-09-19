import os
import time
import csv
import polars as pl
from config.settings import SystemConfig
from src.core.strike_manager import StrikeManager
from src.logging.recorder import DataRecorder

def main():
    print("=== Phase 1 Core Verification ===")
    
    # 1. Load configuration
    config = SystemConfig()
    print(f"Loaded config. Ticker: {config.TICKER}")

    # The live orchestrator always starts a cycle with an unresolved strike and locks it
    # from the first Chainlink tick; this literal stands in for that pre-resolution value.
    presumed_strike = 67500.0
    print(f"Presumed Strike: {presumed_strike}")
    
    # 2. Strike Resolution Logic Test
    # Set expiration time to 3 seconds from now
    now = time.time()
    expiration_time = now + 3.0
    strike_manager = StrikeManager(
        presumed_strike=presumed_strike,
        expiration_timestamp=expiration_time
    )
    
    print("\n--- Testing Strike Resolution ---")
    # Spot tick before expiration
    s1 = strike_manager.get_strike(time.time(), 67200.0)
    print(f"Current strike (before expiration): ${s1} (Expected presumed: {presumed_strike})")
    
    # Wait for expiration
    print("Waiting 3.5 seconds for expiration...")
    time.sleep(3.5)
    
    # Spot tick at/after expiration
    s2 = strike_manager.get_strike(time.time(), 67450.0)
    print(f"Current strike (after expiration): ${s2} (Expected resolved: 67450.0)")
    
    # Subsequent tick should stay locked
    s3 = strike_manager.get_strike(time.time(), 67800.0)
    print(f"Current strike (subsequent tick): ${s3} (Expected locked: 67450.0)")
    
    # 3. DataRecorder Verification
    print("\n--- Testing DataRecorder ---")
    recorder = DataRecorder(
        base_log_dir=config.LOG_DIR,
        strategy_name="merton_test",
        run_id="test_run",
        buffer_size=3  # Low buffer to trigger flush quickly
    )
    
    # Record ticks (should trigger flush since we record 4 ticks and buffer_size is 3)
    bids = [(67440.0, 100.0), (67439.0, 200.0)]
    asks = [(67441.0, 150.0), (67442.0, 250.0)]
    
    print("Recording ticks...")
    recorder.record_tick(time.time(), 67440.5, 12.5, 0.22, bids, asks)
    recorder.record_tick(time.time(), 67440.8, 14.2, 0.22, bids, asks)
    recorder.record_tick(time.time(), 67441.2, -5.6, 0.23, bids, asks) # This should trigger auto-flush
    recorder.record_tick(time.time(), 67441.5, 2.0, 0.23, bids, asks)
    
    # Record signals (written immediately to CSV)
    print("Recording signals...")
    recorder.record_signal(time.time(), 67440.5, 67500.0, 0.45, 0.47, 0.05, "BUY_YES")
    recorder.record_signal(time.time(), 67441.5, 67500.0, 0.42, 0.48, 0.0, "HOLD")
    
    # Record trades (written immediately to CSV)
    print("Recording trades...")
    recorder.record_trade(
        timestamp=time.time(),
        side="BUY_YES",
        qty=100.0,
        vwap=0.48,
        p_market=0.48,
        expected_slippage_bps=5.0,
        realized_slippage_bps=4.5,
        ev=0.02,
        strike=67500.0,
        resolved_won=False,
        pnl=-48.0,
        capital=9952.0
    )
    
    # Force flush remaining ticks
    print("Flushing recorder...")
    recorder.flush()
    
    # 4. Check generated file contents
    print("\n--- Validating Log Files ---")
    log_dir = recorder.log_dir
    print(f"Log directory: {log_dir}")
    
    # Read Parquet ticks
    if os.path.exists(recorder.ticks_path):
        print(f"\n[OK] Found Parquet file: {recorder.ticks_path}")
        df = pl.read_parquet(recorder.ticks_path)
        print("Ticks Data Schema:")
        print(df.schema)
        print(f"Number of rows: {len(df)}")
        print("First 2 rows:")
        print(df.head(2).to_dicts())
    else:
        print("[FAIL] Parquet file ticks.parquet NOT found!")
        
    # Read CSV signals
    if os.path.exists(recorder.signals_path):
        print(f"\n[OK] Found Signals CSV: {recorder.signals_path}")
        with open(recorder.signals_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            print(f"Headers: {rows[0]}")
            print(f"Data rows: {len(rows) - 1}")
            for r in rows[1:]:
                print(r)
    else:
        print("[FAIL] Signals CSV NOT found!")
        
    # Read CSV trades
    if os.path.exists(recorder.trades_path):
        print(f"\n[OK] Found Trades CSV: {recorder.trades_path}")
        with open(recorder.trades_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            print(f"Headers: {rows[0]}")
            print(f"Data rows: {len(rows) - 1}")
            for r in rows[1:]:
                print(r)
    else:
        print("[FAIL] Trades CSV NOT found!")

if __name__ == "__main__":
    main()
