import asyncio
import argparse
import os
import polars as pl
from config.settings import SystemConfig
from src.backtest.runner import BacktestRunner

def parse_datetime(s: str) -> float:
    """Parses a datetime string or timestamp into a Unix timestamp."""
    try:
        return float(s)
    except ValueError:
        pass
        
    from datetime import datetime
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d"
    ):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Could not parse datetime string: '{s}'. Use YYYY-MM-DD HH:MM:SS format.")

def get_overlapping_files(data_dir: str, start_time: float, end_time: float):
    """Scans directory recursively and returns parquet/csv files overlapping with [start_time, end_time]."""
    import glob
    files_with_ts = []
    for ext in ("*.parquet", "*.csv"):
        pattern = os.path.join(data_dir, "**", ext)
        for file_path in glob.glob(pattern, recursive=True):
            basename = os.path.basename(file_path)
            try:
                ts = float(basename.split("_")[-1].split(".")[0])
                files_with_ts.append((ts, file_path))
            except Exception:
                pass
    files_with_ts.sort(key=lambda x: x[0])
    
    selected_files = []
    for i in range(len(files_with_ts)):
        ts, file_path = files_with_ts[i]
        next_ts = files_with_ts[i+1][0] if i + 1 < len(files_with_ts) else float('inf')
        if ts <= end_time and next_ts >= start_time:
            selected_files.append(file_path)
    return selected_files

def parse_args():
    parser = argparse.ArgumentParser(description="Polymarket V2 Backtest Simulator")
    parser.add_argument(
        "--file", 
        type=str, 
        default="c:/Users/loren/Documents/AntiGravity Projects/polymarket/data/raw/tick_data_1779562304.parquet",
        help="Path to a single historical tick file (ignored if --start and --end are used)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start datetime or timestamp (e.g. '2026-05-23 20:51')"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End datetime or timestamp (e.g. '2026-05-26 01:43')"
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    print("=== Polymarket V2 Backtest Replay ===")
    
    config = SystemConfig()
    runner = BacktestRunner(config)
    
    # 1. Load data based on arguments
    if args.start or args.end:
        if not args.start or not args.end:
            print("[Error] Please specify both --start and --end for a time window backtest.")
            return
            
        try:
            start_ts = parse_datetime(args.start)
            end_ts = parse_datetime(args.end)
        except ValueError as e:
            print(f"[Error] {e}")
            return
            
        if start_ts >= end_ts:
            print("[Error] Start time must be strictly before end time.")
            return
            
        data_dirs = [
            os.path.join("data", "raw"),
            "c:/Users/loren/Documents/AntiGravity Projects/polymarket/data/raw"
        ]
        
        selected_files = []
        for d_dir in data_dirs:
            if os.path.exists(d_dir):
                selected_files.extend(get_overlapping_files(d_dir, start_ts, end_ts))
                
        # Deduplicate paths
        seen = set()
        selected_files = [f for f in selected_files if not (f in seen or seen.add(f))]
        
        if not selected_files:
            print(f"[Error] No tick logs found covering the window: {args.start} to {args.end}")
            return
            
        print(f"Loading {len(selected_files)} files covering time window {args.start} to {args.end}...")
        dfs = []
        for f_path in selected_files:
            try:
                df = runner.load_ticks_file(f_path)
                dfs.append(df)
            except Exception as e:
                print(f"[Warning] Failed to load file {f_path}: {e}")
                
        if not dfs:
            print("[Error] Failed to load any tick log files.")
            return
            
        raw_df = pl.concat(dfs).sort("timestamp")
        raw_df = raw_df.filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") <= end_ts))
        
        if raw_df.is_empty():
            print(f"[Error] No ticks available within selected time window: {args.start} to {args.end}")
            return
    else:
        file_path = args.file
        if not os.path.exists(file_path):
            local_fallback = os.path.join("data", "raw", os.path.basename(file_path))
            if os.path.exists(local_fallback):
                file_path = local_fallback
            else:
                print(f"[Error] Ticks file not found at {file_path} or {local_fallback}")
                print("Please provide a valid ticks file using: --file <path> or use --start and --end")
                return
                
        try:
            raw_df = runner.load_ticks_file(file_path)
        except Exception as e:
            print(f"[Error] Failed to load ticks file: {e}")
            return
            
    # 2. Align V1 log schemas to V2 MarketContext structures
    print("Aligning historical schemas to V2 context models...")
    
    # Spot price mapping
    if "binance_bid" in raw_df.columns and "binance_ask" in raw_df.columns:
        aligned_df = raw_df.with_columns(
            (0.5 * (pl.col("binance_bid") + pl.col("binance_ask"))).alias("spot_price")
        )
    elif "polymarket_bid" in raw_df.columns and "polymarket_ask" in raw_df.columns:
        aligned_df = raw_df.with_columns(
            (0.5 * (pl.col("polymarket_bid") + pl.col("polymarket_ask"))).alias("spot_price")
        )
    else:
        aligned_df = raw_df
        
    # Rename columns to V2 standards
    rename_map = {}
    if "polymarket_bids_l2" in aligned_df.columns:
        rename_map["polymarket_bids_l2"] = "bids_l2"
    if "polymarket_asks_l2" in aligned_df.columns:
        rename_map["polymarket_asks_l2"] = "asks_l2"
    if "polymarket_ofi" in aligned_df.columns:
        rename_map["polymarket_ofi"] = "ofi"
        
    if rename_map:
        aligned_df = aligned_df.rename(rename_map)
        
    # Check if necessary columns are present
    req_cols = ["timestamp", "spot_price", "bids_l2", "asks_l2"]
    for col in req_cols:
        if col not in aligned_df.columns:
            print(f"[Error] Schema alignment failed. Missing column: {col}")
            print(f"Columns found: {aligned_df.columns}")
            return
            
    # 3. Run event-driven simulation
    print(f"Starting execution simulation on {len(aligned_df)} ticks...")
    results = await runner.run(aligned_df)
    
    # 4. Output statistics
    print("\n==============================================")
    print("             BACKTEST SIMULATION RESULT       ")
    print("==============================================")
    print(f"Initial Wealth:     ${config.arbitrage.INITIAL_CAPITAL:,.2f}")
    print(f"Ending Cash:        ${results['final_cash']:,.2f}")
    print(f"Net Profit/Loss:    {results['net_pnl']:+,.2f} USD")
    print(f"Return Rate:        {results['total_return_pct']:+.2f}%")
    print(f"Max Drawdown:       -{results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:       {results['total_trades']}")
    print(f"Logs Saved to:      {results['log_dir']}")
    print("==============================================")
    
if __name__ == "__main__":
    asyncio.run(main())
