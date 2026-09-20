"""Backfill the external spot into a recorded session so it can be replayed.

Sessions recorded before the Binance feed existed have no ext_price, and sessions
recorded after it still only carry the values seen live. Binance publishes aggregated
trades with millisecond timestamps for free, so the column can be reconstructed for any
past window and the nowcast can be backtested instead of only run forward.

    uv run python scratch/backfill_binance.py logs5m/2026-09-20/merton/live_1789863462

Writes ext_spot.parquet next to the tick segments. The originals are never modified;
BacktestRunner picks the file up automatically.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import polars as pl

API = "https://api.binance.com/api/v3/aggTrades"


def fetch(symbol, start_ms, end_ms):
    """Aggregated trades in [start_ms, end_ms), paginated at 1000 per call."""
    out = []
    ms = start_ms
    while ms < end_ms:
        url = f"{API}?symbol={symbol}&startTime={ms}&endTime={min(ms + 3_600_000, end_ms)}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(5):
            try:
                batch = json.load(urllib.request.urlopen(req, timeout=30))
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if not batch:
            ms += 3_600_000          # quiet hour, skip forward
            continue
        out += [(t["T"] / 1000.0, float(t["p"])) for t in batch]
        last = batch[-1]["T"]
        ms = last + 1 if len(batch) == 1000 else min(ms + 3_600_000, end_ms)
        print(f"\r  {len(out):,} trades up to {time.strftime('%H:%M:%S', time.localtime(last / 1000))}",
              end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session directory containing ticks*.parquet")
    ap.add_argument("--symbol", default="BTCUSDT")
    a = ap.parse_args()

    segs = sorted(f for f in os.listdir(a.session) if f.startswith("ticks") and f.endswith(".parquet"))
    parts = []
    for f in segs:
        try:
            parts.append(pl.read_parquet(os.path.join(a.session, f)).select("timestamp"))
        except Exception as e:
            print(f"  skipping unreadable {f}: {e}")
    if not parts:
        sys.exit("no readable tick segments")
    ts = pl.concat(parts).sort("timestamp")["timestamp"]
    t0, t1 = ts.min(), ts.max()
    print(f"session spans {(t1 - t0) / 3600:.2f}h, {len(ts):,} ticks")

    # a little margin before the start so the first ticks have a reference
    trades = fetch(a.symbol, int((t0 - 120) * 1000), int((t1 + 1) * 1000))
    if not trades:
        sys.exit("Binance returned nothing for this window")

    df = pl.DataFrame({"timestamp": [x[0] for x in trades], "ext_price": [x[1] for x in trades]}).sort("timestamp")
    out = os.path.join(a.session, "ext_spot.parquet")
    df.write_parquet(out, compression="zstd")
    print(f"wrote {out}: {len(df):,} rows, {df['ext_price'].min():,.0f}-{df['ext_price'].max():,.0f}")


if __name__ == "__main__":
    main()
