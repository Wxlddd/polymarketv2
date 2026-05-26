import asyncio
import time
from config.settings import SystemConfig
from src.ingestion.market_manager import MarketManager
from src.core.strike_manager import StrikeManager

async def test_slug_generation(manager: MarketManager):
    print("\n--- Testing Mathematical Slug Generation ---")
    # Fixed timestamp: 1779790234.0 (2026-05-26... something)
    # 1779790234 % 300 = 34
    # next_expiry should be 1779790234 - 34 + 300 = 1779790500
    test_time = 1779790234.0
    next_expiry = manager.get_next_expiry(test_time)
    slug = manager.get_slug_for_expiry(next_expiry)
    
    print(f"Input timestamp: {test_time}")
    print(f"Calculated Next Expiry: {next_expiry} (Expected: 1779790500)")
    print(f"Generated Slug: {slug} (Expected: btc-updown-5m-1779790500)")
    
    if next_expiry == 1779790500 and slug == "btc-updown-5m-1779790500":
        print("[OK] Slug generation is mathematically correct.")
    else:
        print("[FAIL] Slug generation mismatch!")

async def test_rollover_logic(manager: MarketManager):
    print("\n--- Testing Rollover Evaluation ---")
    current_time = time.time()
    
    # Initialize cycle
    updated = await manager.update_market_cycle(current_time)
    first_expiry = manager.current_expiry
    print(f"Initial update triggered: {updated} | Expiry: {first_expiry} | Slug: {manager.current_slug}")
    
    # Check immediate duplicate check (should not update)
    updated_dup = await manager.update_market_cycle(current_time + 10.0)
    print(f"Subsequent update (+10s) triggered: {updated_dup} (Expected: False)")
    
    # Simulate time passing crossing the expiration threshold
    future_time = first_expiry + 1.0
    updated_rollover = await manager.update_market_cycle(future_time)
    new_expiry = manager.current_expiry
    print(f"Rollover update (+{future_time - current_time:.1f}s) triggered: {updated_rollover} (Expected: True)")
    print(f"New Active Expiry: {new_expiry} | New Slug: {manager.current_slug}")
    
    if updated_rollover and new_expiry == first_expiry + 300:
        print("[OK] Rollover logic is functioning correctly.")
    else:
        print("[FAIL] Rollover logic failed!")

async def test_strike_resolution_with_rest_failure(manager: MarketManager):
    print("\n--- Testing Strike Resolution with REST Failure ---")
    # Simulate a failed REST fetch where strike_price is None
    manager.strike_price = None
    manager.current_expiry = int(time.time()) + 10  # 10s from now
    
    # Enforce NO assumptions: StrikeManager receives None as presumed strike
    strike_manager = StrikeManager(
        presumed_strike=manager.strike_price,  # This is None
        expiration_timestamp=manager.current_expiry
    )
    
    # 1. During active trade, strike must remain None (meaning we can't trade)
    current_strike = strike_manager.get_strike(time.time(), 67200.0)
    print(f"Active trade strike (when REST failed): {current_strike} (Expected: None)")
    
    # 2. Wait for expiration to cross
    print("Waiting 11 seconds for rollover...")
    await asyncio.sleep(11.0)
    
    # 3. Rollover spot tick resolves the strike K
    resolved_strike = strike_manager.get_strike(time.time(), 67450.0)
    print(f"Resolved strike K at rollover: {resolved_strike} (Expected: 67450.0)")
    
    if current_strike is None and resolved_strike == 67450.0:
        print("[OK] Strike resolution with REST failure handles None cleanly and waits for rollover.")
    else:
        print("[FAIL] Strike resolution failed under REST failure scenario!")

async def test_gamma_live_call(manager: MarketManager):
    print("\n--- Testing Live Gamma API Call (Discovery) ---")
    current_time = time.time()
    next_expiry = manager.get_next_expiry(current_time)
    manager.current_expiry = next_expiry
    manager.current_slug = manager.get_slug_for_expiry(next_expiry)
    
    # Run the live API call
    await manager.fetch_market_context()
    
    print("API Query Results:")
    print(f"  - YES Token ID: {manager.yes_token_id}")
    print(f"  - NO Token ID: {manager.no_token_id}")
    print(f"  - Strike Price: {manager.strike_price}")
    print(f"  - Condition ID: {manager.condition_id}")
    print("[OK] Live API check completed.")

async def main():
    config = SystemConfig()
    manager = MarketManager(config)
    
    await test_slug_generation(manager)
    await test_rollover_logic(manager)
    await test_strike_resolution_with_rest_failure(manager)
    await test_gamma_live_call(manager)

if __name__ == "__main__":
    asyncio.run(main())
