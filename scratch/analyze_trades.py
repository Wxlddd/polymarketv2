"""Break a trades.csv down by side, settlement win rate and unreconciled quantity.

    uv run python scratch/analyze_trades.py logs/2026-05-28/merton/live_1779986166/trades.csv
    uv run python scratch/analyze_trades.py            # newest live session under logs/
"""
import glob
import os
import sys

import polars as pl


def newest_trades_file() -> str:
    candidates = glob.glob(os.path.join("logs", "*", "*", "live_*", "trades.csv"))
    if not candidates:
        raise SystemExit("No logs/*/*/live_*/trades.csv found. Pass a path explicitly.")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    file_path = sys.argv[1] if len(sys.argv) > 1 else newest_trades_file()
    print(f"Reading: {file_path}")
    df = pl.read_csv(file_path, infer_schema_length=10000)

    print("\n=== TOTAL TRADE ROW COUNT ===")
    print("Total rows:", len(df))

    print("\n=== TRADE SIDE BREAKDOWN ===")
    for row in df["side"].value_counts().iter_rows(named=True):
        print(f"{row['side']}: {row['count']}")

    settle_df = df.filter(pl.col("side").str.starts_with("SETTLE"))
    print("\n=== SETTLEMENT BREAKDOWN ===")
    print(f"Total Settlements: {len(settle_df)}")
    if len(settle_df) > 0:
        for row in settle_df["side"].value_counts().iter_rows(named=True):
            print(f"{row['side']}: {row['count']}")
        print(f"Total Settlement PnL: ${settle_df['pnl'].sum():.2f}")

        won = settle_df.filter(pl.col("pnl") > 0)
        lost = settle_df.filter(pl.col("pnl") < 0)
        print(f"Won: {len(won)} | Lost: {len(lost)} | Win Rate: {len(won) / len(settle_df) * 100:.2f}%")

        print("\n=== SETTLEMENT QUANTITY STATS ===")
        print(f"Mean Qty: {settle_df['qty'].mean():.2f}")
        print(f"Max Qty: {settle_df['qty'].max():.2f}")
        print(f"Min Qty: {settle_df['qty'].min():.2f}")

        print("\n=== LAST 10 SETTLEMENT DETAILS ===")
        for row in settle_df.tail(10).iter_rows(named=True):
            print(
                f"Time: {row['timestamp']} | Side: {row['side']} | Qty: {row['qty']:.2f} | "
                f"VWAP: {row['vwap']:.4f} | PnL: ${row['pnl']:.2f} | Capital: ${row['capital']:.2f}"
            )

    non_settle = df.filter(~pl.col("side").str.starts_with("SETTLE"))
    print("\n=== ACTIVE TRADES BREAKDOWN (EXCLUDING SETTLEMENT) ===")
    for row in non_settle["side"].value_counts().iter_rows(named=True):
        print(f"{row['side']}: {row['count']}")
    print(f"Total non-settle PnL: ${non_settle['pnl'].sum():.2f}")

    print("\n=== NET QUANTITIES BOUGHT VS SOLD ===")
    for token in ("YES", "NO"):
        bought = df.filter(pl.col("side") == f"BUY_{token}")["qty"].sum()
        sold = df.filter(pl.col("side") == f"SELL_{token}")["qty"].sum()
        settled = df.filter(pl.col("side") == f"SETTLE_{token}")["qty"].sum()
        print(
            f"{token}: Bought {bought:.2f} | Sold {sold:.2f} | Settled {settled:.2f} | "
            f"Unaccounted: {bought - sold - settled:.2f}"
        )


if __name__ == "__main__":
    main()
