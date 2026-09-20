"""The external spot feed must sharpen the oracle, never replace or invent it.

Chainlink settles the contract, so the nowcast is a forecast OF the oracle: it carries the
last oracle print forward by the Binance move since that print, and degrades to the raw
oracle whenever the external feed cannot be trusted.

    PYTHONPATH=. uv run python tests/verify_binance_feed.py          # offline checks
    PYTHONPATH=. uv run python tests/verify_binance_feed.py --live   # also hits Binance
"""
import asyncio
import sys
import time

from config.settings import SystemConfig
from src.ingestion.binance_feed import BinanceSpotFeed

ORACLE = 81_000.0


def make_feed(ticks, now=None):
    feed = BinanceSpotFeed(SystemConfig())
    now = now or time.time()
    for dt, px in ticks:
        feed._ticks.append((now + dt, px))
    feed._price = ticks[-1][1]
    feed._last_updated = now + ticks[-1][0]
    feed._is_connected = True
    return feed, now


def main():
    t0 = time.time()

    # Binance up 0.1% since the oracle printed -> nowcast is the oracle up 0.1%
    feed, now = make_feed([(-10.0, 80_000.0), (0.0, 80_080.0)], t0)
    nc = feed.nowcast(ORACLE, now - 10.0)
    assert abs(nc - ORACLE * 1.001) < 1e-6, nc
    print(f"[OK] oracle {ORACLE:,.0f} + 0.10% of Binance move -> nowcast {nc:,.2f}")

    # No move since the print -> exactly the oracle, never a level of its own
    feed, now = make_feed([(-10.0, 80_000.0), (0.0, 80_000.0)], t0)
    assert feed.nowcast(ORACLE, now - 10.0) == ORACLE
    print("[OK] no external move -> nowcast equals the oracle exactly")

    # Stale feed -> fall back to the oracle rather than trade on a guess
    feed, now = make_feed([(-10.0, 80_000.0), (0.0, 80_500.0)], t0)
    feed._last_updated = t0 - 3600.0
    assert not feed.is_fresh
    assert feed.nowcast(ORACLE, now - 10.0) == ORACLE
    print("[OK] stale feed -> falls back to the oracle")

    # Disconnected -> same fallback
    feed, now = make_feed([(-10.0, 80_000.0), (0.0, 80_500.0)], t0)
    feed._is_connected = False
    assert feed.nowcast(ORACLE, now - 10.0) == ORACLE
    print("[OK] disconnected -> falls back to the oracle")

    # No observation as old as the print -> fallback (cannot measure the move)
    feed, now = make_feed([(0.0, 80_500.0)], t0)
    assert feed.nowcast(ORACLE, now - 3600.0) == ORACLE
    print("[OK] no reference observation -> falls back to the oracle")

    # Direction: Binance down means the nowcast is below the oracle
    feed, now = make_feed([(-5.0, 80_000.0), (0.0, 79_600.0)], t0)
    assert feed.nowcast(ORACLE, now - 5.0) < ORACLE
    print("[OK] external move down -> nowcast below the oracle")

    # The orchestrator must not use it when pricing is switched off
    cfg = SystemConfig()
    cfg.binance.__dict__["USE_FOR_PRICING"] = False
    from main import LiveOrchestrator
    from types import SimpleNamespace
    stub = SimpleNamespace(config=cfg, binance_feed=feed,
                           spot_feed=SimpleNamespace(last_updated=t0 - 5.0))
    assert LiveOrchestrator._pricing_spot(stub, ORACLE) == ORACLE
    cfg.binance.__dict__["USE_FOR_PRICING"] = True
    assert LiveOrchestrator._pricing_spot(stub, ORACLE) != ORACLE
    assert LiveOrchestrator._pricing_spot(stub, None) is None
    print("[OK] BINANCE_USE_FOR_PRICING=False prices off the raw oracle")

    if "--live" in sys.argv:
        async def live():
            f = BinanceSpotFeed(SystemConfig())
            await f.start()
            for _ in range(30):
                await asyncio.sleep(0.5)
                if f.price:
                    break
            assert f.is_fresh, "no data from Binance"
            print(f"[OK] live stream: {f.price:,.2f} after {len(f._ticks)} updates")
            await f.stop()
        asyncio.run(live())

    print("\n=== External spot feed verified ===")


if __name__ == "__main__":
    main()
