"""A maker fill can never exceed the volume that actually traded.

The old simulator filled the whole resting order the instant the book crossed our price,
with no volume limit. Checked against the exchange's own prints on a live session: 46% of
those fills happened at a price where nothing traded at all, and fills of 100+ contracts
claimed a median of 240 contracts against 5 actually printed. That is where a +12% session
P&L came from.

    PYTHONPATH=. uv run python tests/verify_fill_realism.py
"""
import asyncio

from config.settings import SystemConfig
from src.core.interfaces import OrderInstruction
from src.execution.clients import MockExecutionClient
from src.execution.shadow_book import ShadowOrderBook

CTX = {"timestamp": 1_000_000.0, "strike_price": 80_000.0, "volatility": 0.5}
BIDS = [(0.47, 500.0), (0.46, 800.0)]
ASKS = [(0.49, 600.0), (0.50, 900.0)]


class _NullRecorder:
    """The client logs every fill; this test only cares about sizes."""

    def record_trade(self, *a, **k):
        pass

    def record_signal(self, *a, **k):
        pass


def make_client():
    cfg = SystemConfig()
    book = ShadowOrderBook()
    book.is_backtest = True
    book.update_book(BIDS, ASKS, is_snapshot=True, timestamp=CTX["timestamp"])
    c = MockExecutionClient(cfg, recorder=_NullRecorder(), shadow_book=book)
    c.is_backtest = True
    return c


async def rest_bid(client, price=0.47, qty=400.0, queue_ahead=0.0):
    instr = OrderInstruction("NEW", "BUY_YES", price, qty, "bid_1", "A")
    await client.process_instruction(instr, CTX)
    client.active_maker_orders["bid"]["queue_ahead"] = queue_ahead
    return instr


async def main():
    # 1. A print smaller than our order fills only that much.
    c = make_client()
    await rest_bid(c, qty=400.0)
    fills = await c.process_print(0.47, 25.0, "SELL", CTX)
    got = sum(f["qty"] for f in fills if f.get("success"))
    assert abs(got - 25.0) < 1e-6, f"filled {got} on a 25-contract print"
    assert c.active_maker_orders["bid"] is not None, "order vanished after a partial fill"
    print(f"[OK] 25-contract print fills 25 of a 400 order, rest stays resting")

    # 2. The queue ahead of us eats the print first.
    c = make_client()
    await rest_bid(c, qty=400.0, queue_ahead=100.0)
    fills = await c.process_print(0.47, 60.0, "SELL", CTX)
    assert not [f for f in fills if f.get("success")], "filled while 100 contracts were ahead"
    assert abs(c.active_maker_orders["bid"]["queue_ahead"] - 40.0) < 1e-6
    print("[OK] queue ahead consumes the print before we do")

    # 3. A print at a price that cannot reach our quote does nothing.
    c = make_client()
    await rest_bid(c, price=0.47, qty=400.0)
    fills = await c.process_print(0.48, 500.0, "SELL", CTX)
    assert not [f for f in fills if f.get("success")], "filled by a print above our bid"
    print("[OK] a print that never reached our price does not fill us")

    # 4. The book crossing our price, with no print, fills nothing. This is the bug.
    c = make_client()
    await rest_bid(c, price=0.47, qty=400.0)
    crossed_asks = [(0.45, 900.0)]
    fills = await c.process_market_data(BIDS, crossed_asks, CTX)
    assert not [f for f in fills if f.get("success")], (
        "book crossing our quote filled us with no trade behind it")
    print("[OK] book crossing our price fills nothing without a print")

    # 5. Total filled over many prints never exceeds the order size.
    c = make_client()
    await rest_bid(c, qty=100.0)
    total = 0.0
    for _ in range(20):
        for f in await c.process_print(0.47, 30.0, "SELL", CTX):
            if f.get("success"):
                total += f["qty"]
    assert total <= 100.0 + 1e-6, f"filled {total} on a 100-contract order"
    print(f"[OK] 20 prints of 30 fill at most the {100:.0f} we quoted (got {total:.0f})")

    print("\n=== Fill realism verified ===")


if __name__ == "__main__":
    asyncio.run(main())
