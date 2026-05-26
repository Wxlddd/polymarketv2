import asyncio
import time
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.execution.shadow_book import ShadowOrderBook
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.logging.recorder import DataRecorder

async def main():
    print("=== Phase 4 Execution Engine Verification ===")
    
    config = SystemConfig()
    recorder = DataRecorder(
        base_log_dir=config.LOG_DIR,
        strategy_name="merton_exec_test",
        run_id="test_run",
        buffer_size=10
    )
    
    # 1. Initialize shadow order book
    shadow_book = ShadowOrderBook()
    
    # Initial snapshot updates
    bids_snap = [(0.45, 1000.0), (0.44, 2000.0), (0.43, 3000.0)]
    asks_snap = [(0.47, 1000.0), (0.48, 1500.0), (0.49, 2000.0)]
    
    print("\n--- Testing ShadowOrderBook Reconciliation ---")
    ofi = shadow_book.update_book(bids_snap, asks_snap, is_snapshot=True)
    print(f"Shadow book snapshot set. OFI: {ofi}")
    print(f"Top bid: {shadow_book.get_sorted_bids()[0]}")
    print(f"Top ask: {shadow_book.get_sorted_asks()[0]}")
    
    # Send incremental update (best ask size increases, best bid drops to 0)
    shadow_book.update_book(
        bid_updates=[(0.45, 0.0)], # Bid removed
        ask_updates=[(0.47, 1200.0)], # Ask updated
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

    # 2. Initialize execution client and engine
    client = MockExecutionClient(config, recorder, shadow_book)
    engine = ExecutionEngine(config)
    
    # 3. Test Kelly Sizing & Sweeping for BUY YES
    print("\n--- Testing Sizing & Sweeping ---")
    # Model probability YES = 60%, ask is 47%, edge = 13% -> should buy YES
    vol = 0.25
    tau = 200.0
    strike = 60000.0
    spot = 59980.0
    
    context = MarketContext(
        timestamp=time.time(),
        spot_price=spot,
        strike_price=strike,
        tau_seconds=tau,
        volatility=vol,
        ofi=0.0,
        bids_l2=shadow_book.get_sorted_bids(),
        asks_l2=shadow_book.get_sorted_asks()
    )
    
    print(f"Model Probability: 60.0% | Top Ask YES: {sorted_asks[0][0]:.2f}")
    decision = engine.evaluate_and_trade(p_yes=0.60, context=context, client=client)
    print(f"Decision: {decision}")
    
    if decision["side"] == "BUY_YES":
        qty = decision["size"]
        price = decision["vwap"]
        expected_slip = decision["expected_slippage_bps"]
        
        print(f"Placing simulated trade: BUY_YES | Qty: {qty:.2f} | Price: {price:.4f} | Slip: {expected_slip:.1f} bps")
        
        # Execute the trade
        res = await client.execute_trade(
            side="BUY_YES",
            qty=qty,
            price=price,
            ev=decision["ev"],
            expected_slippage_bps=expected_slip,
            context_state={"timestamp": context.timestamp, "strike_price": context.strike_price}
        )
        print(f"Trade Execution Result: {res}")
        print(f"Client Cash Balance: ${client.cash_balance:.2f}")
        print(f"YES position size (qty): {client.get_position_size('YES')}")
        print(f"YES average entry price: ${client.entry_prices['YES']:.4f}")
        
        # Check shadow book asks: first level (0.47) size should be depleted by quantity bought
        new_sorted_asks = shadow_book.get_sorted_asks()
        print(f"Shadow book top ask size after execute (was 1200): {new_sorted_asks[0]}")
        
        if client.get_position_size("YES") == qty and new_sorted_asks[0][1] < 1200.0:
            print("[OK] Sizing, sweeping, and shadow book depletion work together.")
        else:
            print("[FAIL] Execution/sweeping integration error!")
    else:
        print("[FAIL] Engine failed to generate expected BUY_YES signal!")

    # 4. Test Option Expiration Settlement
    print("\n--- Testing Position Settlement at Expiration ---")
    settlement_spot = 60100.0  # Spot is above strike K = 60000.0 -> YES wins
    print(f"Settling option. Spot at expiration: ${settlement_spot:.2f} > Strike K: ${strike:.2f}")
    payout = client.settle_positions(
        settlement_price=settlement_spot,
        strike_price=strike,
        timestamp=time.time()
    )
    print(f"Settlement Net Payout: ${payout:.2f}")
    print(f"Final Client Cash Balance: ${client.cash_balance:.2f}")
    print(f"YES position size (expected 0): {client.get_position_size('YES')}")
    
    # Cash should be initial capital + qty * (1.0 - entry_price) - gas - taker_fee
    taker_fee = qty * config.arbitrage.TAKER_FEE_MULTIPLIER * price * (1.0 - price)
    expected_cash = config.arbitrage.INITIAL_CAPITAL + qty * (1.0 - price) - config.arbitrage.GAS_FEE_USD - taker_fee
    print(f"Expected final cash: ${expected_cash:.2f}")
    
    if abs(client.cash_balance - expected_cash) < 1e-9 and client.get_position_size("YES") == 0.0:
        print("[OK] Position settlement and cash balances are mathematically exact.")
    else:
        print(f"[FAIL] Settlement calculations mismatch! Actual cash: ${client.cash_balance:.6f}, Expected cash: ${expected_cash:.6f}")

    # Flush recorder
    recorder.flush()

if __name__ == "__main__":
    asyncio.run(main())
