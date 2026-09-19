"""Data collection must not lose ticks — the whole point of a long unattended run.

1. Ticks are recorded while waiting for the first cycle boundary (up to 4h with 4h cycles),
   without pricing (get_probability is once-per-tick by contract).
2. Tick parquet segments are rotated and closed, so a kill or power cut costs one segment
   instead of the whole session, and the backtester reads the segments back as one series.

    PYTHONPATH=. uv run python tests/verify_data_collection.py
"""
import asyncio
import os
import shutil
import tempfile
import time
from types import SimpleNamespace

import polars as pl

from config.settings import SystemConfig
from main import LiveOrchestrator
from src.backtest.runner import BacktestRunner
from src.execution.shadow_book import ShadowOrderBook
from src.logging.recorder import DataRecorder
from src.strategies.merton_strategy import HighFrequencyVolatilityCalibrator

BIDS = [(0.47, 500.0), (0.46, 800.0)]
ASKS = [(0.49, 600.0), (0.50, 900.0)]


class CountingRecorder:
    def __init__(self):
        self.ticks = []

    def record_tick(self, timestamp, spot_price, ofi, volatility, bids_l2, asks_l2,
                    top_bid=None, top_ask=None, is_snapshot=False):
        self.ticks.append({"spot": spot_price, "top_bid": top_bid, "top_ask": top_ask,
                           "is_snapshot": is_snapshot, "vol": volatility})


def make_waiting_orchestrator(recorder):
    """Minimal stand-in carrying only what the waiting branch touches."""
    config = SystemConfig()
    priced = []
    strategy = SimpleNamespace(
        vol_calibrator=HighFrequencyVolatilityCalibrator(window_size=config.merton.VOL_ROLLING_WINDOW_SEC),
        get_probability=lambda ctx: priced.append(ctx),
        reset=lambda: None,
    )
    return SimpleNamespace(
        is_running=True,
        _tick_count_tps=0,
        strike_manager=None,
        waiting_for_first_rollover=True,
        _last_snapshot_ts=0.0,
        spot_feed=SimpleNamespace(price=81199.98),
        shadow_book=ShadowOrderBook(),
        strategy=strategy,
        config=config,
        recorder=recorder,
        log_message=lambda *a, **k: None,
    ), priced


def check_tick_segments():
    """A killed session must keep everything except the open segment."""
    tmp = tempfile.mkdtemp(prefix="seg_check_")
    try:
        # Rotate on every flush so the test does not depend on wall-clock timing.
        rec = DataRecorder(base_log_dir=tmp, strategy_name="merton", run_id="seg",
                           buffer_size=2, tick_segment_sec=1e-9)
        book = [(0.47, 500.0)], [(0.49, 600.0)]
        for i in range(6):
            rec.record_tick(1_000_000.0 + i, 81_000.0 + i, 0.0, 0.25, *book,
                            top_bid=(0.47, 500.0), top_ask=(0.49, 600.0), is_snapshot=(i == 0))
        # Simulate a hard kill: the open writer is never closed.
        session = rec.log_dir
        rec.writer = None

        segments = sorted(f for f in os.listdir(session) if f.startswith("ticks"))
        assert len(segments) >= 2, f"no rotation happened: {segments}"
        assert segments[0] == "ticks.parquet", f"first segment renamed: {segments}"
        readable = [f for f in segments if _readable(os.path.join(session, f))]
        assert len(readable) >= 2, f"segments unreadable after kill: {segments}"
        print(f"[OK] {len(segments)} segments, {len(readable)} readable after a kill: {segments}")

        merged = BacktestRunner(SystemConfig()).load_ticks_file(session)
        assert len(merged) >= 4, f"merged only {len(merged)} ticks from {segments}"
        assert merged["timestamp"].is_sorted(), "merged segments not in time order"
        print(f"[OK] backtester merged the session directory: {len(merged)} ticks, in order")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_age_flush():
    """A slow market must not keep hours of ticks (or rare trade prints) in RAM."""
    tmp = tempfile.mkdtemp(prefix="flush_check_")
    try:
        rec = DataRecorder(base_log_dir=tmp, strategy_name="merton", run_id="age",
                           buffer_size=100_000, flush_interval_sec=0.2, tick_segment_sec=0)
        book = [(0.47, 500.0)], [(0.49, 600.0)]
        rec.record_tick(1.0, 81_000.0, 0.0, 0.25, *book, top_bid=(0.47, 500.0), top_ask=(0.49, 600.0))
        rec.record_print(1.0, 0.47, 10.0, "BUY")
        assert rec.tick_buffer, "flushed before the interval elapsed"

        time.sleep(0.25)
        rec.record_tick(2.0, 81_001.0, 0.0, 0.25, *book, top_bid=(0.47, 500.0), top_ask=(0.49, 600.0))
        rec.record_print(2.0, 0.48, 5.0, "SELL")
        assert not rec.tick_buffer, f"ticks held in RAM past the interval: {len(rec.tick_buffer)}"
        assert not rec.print_buffer, f"prints held in RAM past the interval: {len(rec.print_buffer)}"

        rec.flush()
        n_ticks = len(pl.read_parquet(os.path.join(rec.log_dir, "ticks.parquet")))
        n_prints = len(pl.read_parquet(os.path.join(rec.log_dir, "prints.parquet")))
        assert n_ticks == 2 and n_prints == 2, f"ticks={n_ticks} prints={n_prints}"
        print(f"[OK] age-based flush: {n_ticks} ticks and {n_prints} prints on disk without a full buffer")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _readable(path):
    try:
        pl.read_parquet(path)
        return True
    except Exception:
        return False


def main():
    recorder = CountingRecorder()
    orch, priced = make_waiting_orchestrator(recorder)

    asyncio.run(LiveOrchestrator._clob_callback(orch, BIDS, ASKS, True))
    assert len(recorder.ticks) == 1, f"waiting tick not recorded: {recorder.ticks}"
    assert not priced, "pricing ran while waiting for the first cycle"

    first = recorder.ticks[0]
    assert first["spot"] == 81199.98, first
    assert first["top_bid"] == (0.47, 500.0), f"top of book not reconciled: {first['top_bid']}"
    assert first["top_ask"] == (0.49, 600.0), f"top of book not reconciled: {first['top_ask']}"
    assert first["is_snapshot"] is True, first
    print(f"[OK] tick recorded while waiting: top {first['top_bid'][0]}/{first['top_ask'][0]}, no pricing")

    # A second snapshot inside 1s is downgraded to a delta (reconnect-storm dedup), still recorded.
    asyncio.run(LiveOrchestrator._clob_callback(orch, BIDS, ASKS, True))
    assert len(recorder.ticks) == 2, "second waiting tick dropped"
    assert recorder.ticks[1]["is_snapshot"] is False, "snapshot dedup did not apply while waiting"
    print("[OK] snapshot dedup still applies while waiting")

    # No spot yet -> nothing to record, and no crash.
    recorder2 = CountingRecorder()
    orch2, _ = make_waiting_orchestrator(recorder2)
    orch2.spot_feed.price = None
    asyncio.run(LiveOrchestrator._clob_callback(orch2, BIDS, ASKS, True))
    assert not recorder2.ticks, "recorded a tick with no spot price"
    print("[OK] no spot price -> no tick, no crash")

    check_tick_segments()
    check_age_flush()

    print("\n=== Data collection verified ===")


if __name__ == "__main__":
    main()
