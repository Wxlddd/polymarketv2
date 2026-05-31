import time
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.strategies.factory import StrategyFactory

def main():
    print("=== Strategy Factory & Dynamic Loading Verification ===")
    
    config = SystemConfig()
    
    # 1. Test Factory Strategy Retrieval
    print("\n1. Testing strategy loading from factory...")
    try:
        merton = StrategyFactory.get_strategy("merton", config)
        print(f"[OK] Successfully loaded 'merton' strategy: {merton.__class__.__name__}")
        
        legacy_merton = StrategyFactory.get_strategy("legacy_merton", config)
        print(f"[OK] Successfully loaded 'legacy_merton' strategy: {legacy_merton.__class__.__name__}")
        
    except Exception as e:
        print(f"[FAIL] Error loading strategies: {e}")
        return
        
    # 2. Test invalid strategy handling
    print("\n2. Testing invalid strategy exception...")
    try:
        StrategyFactory.get_strategy("non_existent_strategy", config)
        print("[FAIL] Factory accepted an invalid strategy name without raising ValueError!")
    except ValueError as e:
        print(f"[OK] Factory correctly raised exception on invalid strategy: {e}")
    except Exception as e:
        print(f"[FAIL] Unexpected exception raised: {e}")

    # 3. Feed spot price ticks to volatility calibrators
    print("\n3. Feeding ticks to calibrate volatility for both...")
    start_time = time.time()
    for i in range(15):
        price = 60000.0 + (i * 1.5)
        t = start_time + (i * 5.0)
        merton.vol_calibrator.add_tick(price, t)
        legacy_merton.vol_calibrator.add_tick(price, t)
        
    # 4. Compare probability outputs under the same context with an OFI imbalance
    print("\n4. Pricing the same context under both models...")
    context = MarketContext(
        timestamp=time.time(),
        spot_price=60020.0,
        strike_price=60000.0,
        tau_seconds=120.0,
        volatility=0.25,
        ofi=250.0,  # strong positive OFI
        bids_l2=[(0.48, 1000.0)],
        asks_l2=[(0.50, 1000.0)]
    )
    
    prob_new = merton.get_probability(context)
    prob_legacy = legacy_merton.get_probability(context)
    
    print(f"Hawkes-Driven Merton ('merton') Prob        : {prob_new:.2%}")
    print(f"Legacy Logit-Shift Merton ('legacy_merton') Prob: {prob_legacy:.2%}")
    
    if prob_new is not None and prob_legacy is not None:
        print("[OK] Both models computed probabilities cleanly in context.")
        if prob_new != prob_legacy:
            print("[OK] Models produced different pricing signals as expected due to different OFI impact channels!")
        else:
            print("[FAIL] Models returned identical probabilities despite radically different mathematical formulations.")
    else:
        print("[FAIL] One or both models returned None.")
        
    print("\n=== Factory Verification Complete ===")

if __name__ == "__main__":
    main()
