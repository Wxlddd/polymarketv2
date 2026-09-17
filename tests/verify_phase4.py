import asyncio
import random
import time
from typing import Optional
from config.settings import SystemConfig
from src.core.base_strategy import BaseStrategy
from src.core.market_context import MarketContext
from src.execution.shadow_book import ShadowOrderBook
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.logging.recorder import DataRecorder


class FixedProbabilityStrategy(BaseStrategy):
    """Stub pricer returning a constant p_hat so the router's regimes can be exercised deterministically."""

    def __init__(self, p_hat: Optional[float]):
        self.p_hat = p_hat

    def get_probability(self, context: MarketContext) -> Optional[float]:
        return self.p_hat


async def main():
    print("=== Phase 4 Execution Engine Verification ===")
    random.seed(0)  # MockExecutionClient applies stochastic taker rejection; pin it for a reproducible run

    config = SystemConfig()
    recorder = DataRecorder(
        base_log_dir=config.LOG_DIR,
        strategy_name="merton_exec_test",
        run_id="test_run",
        buffer_size=10
    )

    # 1. Shadow order book reconciliation
    shadow_book = ShadowOrderBook()
    bids_snap = [(0.45, 1000.0), (0.44, 2000.0), (0.43, 3000.0)]
    asks_snap = [(0.47, 1000.0), (0.48, 1500.0), (0.49, 2000.0)]

    print("\n--- Testing ShadowOrderBook Reconciliation ---")
    ofi = shadow_book.update_book(bids_snap, asks_snap, is_snapshot=True)
    print(f"Shadow book snapshot set. OFI: {ofi}")
    print(f"Top bid: {shadow_book.get_sorted_bids()[0]}")
    print(f"Top ask: {shadow_book.get_sorted_asks()[0]}")

    shadow_book.update_book(
        bid_updates=[(0.45, 0.0)],     # best bid removed
        ask_updates=[(0.47, 1200.0)],  # best ask resized
        is_snapshot=False
    )
    sorted_bids = shadow_book.get_sorted_bids()
    sorted_asks = shadow_book.get_sorted_asks()
    print("After incremental L2 updates:")
    print(f"  - New top bid (expected 0.44): {sorted_bids[0]}")
    print(f"  - New top ask size (expected 1200): {sorted_asks[0]}")
    if sorted_bids[0][0] == 0.44 and sorted_asks[0][1] == 1200.0:
        print("[OK] L2 Delta reconciliation operates correctly.")
    else:
        print("[FAIL] L2 reconciliation error!")

    # 2. Engine + paper client wired exactly as in LiveOrchestrator
    client = MockExecutionClient(config, recorder, shadow_book)
    strategy = FixedProbabilityStrategy(p_hat=0.60)
    engine = ExecutionEngine(strategy, client, config)

    # 3. Regime B taker crossing: p_hat = 60% vs best ask 47% -> router must cross the spread
    print("\n--- Testing Taker Crossing (Regime B) & Shadow Book Depletion ---")
    strike = 60000.0
    spot = 59980.0
    context = MarketContext(
        timestamp=time.time(),
        spot_price=spot,
        strike_price=strike,
        tau_seconds=200.0,
        volatility=0.25,
        ofi=0.0,
        bids_l2=shadow_book.get_sorted_bids(),
        asks_l2=shadow_book.get_sorted_asks()
    )
    print(f"Model Probability: 60.0% | Top Ask YES: {sorted_asks[0][0]:.2f}")
    instructions = engine.evaluate_and_route(context)
    print(f"Router instructions: {instructions}")

    taker = next((i for i in instructions if i.action == "NEW" and i.regime == "B" and i.side == "BUY_YES"), None)
    if taker is None:
        print("[FAIL] Engine failed to generate expected Regime B BUY_YES instruction!")
        recorder.flush()
        return

    context_state = {"timestamp": context.timestamp, "strike_price": strike, "volatility": context.volatility}
    res = await client.process_instruction(taker, context_state)
    print(f"Trade Execution Result: {res}")
    if not res.get("success"):
        print("[FAIL] Taker instruction did not execute!")
        recorder.flush()
        return

    qty = res["qty"]
    price = res["price"]
    print(f"Client Cash Balance: ${client.cash_balance:.2f}")
    print(f"YES position size (qty): {client.get_position_size('YES'):.2f}")
    print(f"YES average entry price: ${client.entry_prices['YES']:.4f}")

    new_sorted_asks = shadow_book.get_sorted_asks()
    print(f"Shadow book top ask size after execute (was 1200): {new_sorted_asks[0]}")
    if abs(client.get_position_size("YES") - qty) < 1e-9 and new_sorted_asks[0][1] < 1200.0:
        print("[OK] Taker crossing, IOC book walk, and shadow book depletion work together.")
    else:
        print("[FAIL] Execution/sweeping integration error!")

    # 4. Settlement: spot above strike -> YES pays 1.0
    print("\n--- Testing Position Settlement at Expiration ---")
    settlement_spot = 60100.0
    print(f"Settling option. Spot at expiration: ${settlement_spot:.2f} > Strike K: ${strike:.2f}")
    payout = client.settle_positions(settlement_price=settlement_spot, strike_price=strike, timestamp=time.time())
    print(f"Settlement Net Payout: ${payout['net_pnl']:.2f}")
    print(f"Final Client Cash Balance: ${client.cash_balance:.2f}")
    print(f"YES position size (expected 0): {client.get_position_size('YES')}")

    # Cash = initial + qty*(1 - vwap) - gas(buy) - taker fee (per-level p*(1-p) fee, single level here).
    # Settlement gas is charged in the reported settlement pnl only, not against the cash balance.
    taker_fee = qty * config.arbitrage.TAKER_FEE_MULTIPLIER * price * (1.0 - price)
    expected_cash = (
        config.arbitrage.INITIAL_CAPITAL
        + qty * (1.0 - price)
        - config.arbitrage.GAS_FEE_USD
        - taker_fee
    )
    print(f"Expected final cash: ${expected_cash:.2f}")
    if abs(client.cash_balance - expected_cash) < 1e-6 and client.get_position_size("YES") == 0.0:
        print("[OK] Position settlement and cash balances are mathematically exact.")
    else:
        print(f"[FAIL] Settlement mismatch! Actual: ${client.cash_balance:.6f}, Expected: ${expected_cash:.6f}")

    recorder.flush()


if __name__ == "__main__":
    asyncio.run(main())
