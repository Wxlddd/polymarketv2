import time
import numpy as np
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.strategies.merton_strategy import MertonStrategy

def main():
    print("=== Phase 3 Strategy Mathematical Verification ===")
    
    config = SystemConfig()
    strategy = MertonStrategy(config)
    
    # 1. Volatility Calibrator Test
    print("\n--- Testing Volatility Calibrator ---")
    start_time = time.time()
    
    # Feed 20 ticks at 5-second intervals with standard random noise
    np.random.seed(42)
    base_price = 60000.0
    for i in range(20):
        t = start_time + (i * 5.0)
        # Small changes (+/- 0.05%)
        pct_change = np.random.normal(0, 0.0005)
        price = base_price * (1.0 + pct_change)
        strategy.vol_calibrator.add_tick(price, t)
        base_price = price
        
    calibrated_vol = strategy.vol_calibrator.calculate_volatility(config.merton.DEFAULT_SIGMA)
    print(f"Calibrated Volatility: {calibrated_vol:.2%} (Default was: {config.merton.DEFAULT_SIGMA:.2%})")
    
    if 0.15 <= calibrated_vol <= 3.0:
        print("[OK] Volatility calibrated within sanity bounds [15%, 300%].")
    else:
        print("[FAIL] Volatility out of bounds!")

    # 2. Probability calculations for YES option (Strike = 60000)
    print("\n--- Testing Gil-Pelaez Fourier Integrator ---")
    strike = 60000.0
    tau = 150.0  # 150 seconds to expiration
    
    # Case A: Spot is far below Strike (Probability of YES should be low)
    context_below = MarketContext(
        timestamp=time.time(),
        spot_price=59800.0,
        strike_price=strike,
        tau_seconds=tau,
        volatility=calibrated_vol,
        ofi=0.0
    )
    p_below = strategy.get_probability(context_below)
    print(f"Spot: ${context_below.spot_price:,.2f} | K: ${strike:,.2f} | YES Probability: {p_below:.2%}")
    
    # Case B: Spot is far above Strike (Probability of YES should be high)
    context_above = MarketContext(
        timestamp=time.time(),
        spot_price=60200.0,
        strike_price=strike,
        tau_seconds=tau,
        volatility=calibrated_vol,
        ofi=0.0
    )
    p_above = strategy.get_probability(context_above)
    print(f"Spot: ${context_above.spot_price:,.2f} | K: ${strike:,.2f} | YES Probability: {p_above:.2%}")
    
    if p_below is not None and p_above is not None and p_above > p_below:
        print("[OK] Options pricing direction is mathematically correct (P_above > P_below).")
    else:
        print("[FAIL] Options pricing direction error!")

    # 3. Microsecond boundary condition (tau <= 1.0s)
    print("\n--- Testing Microsecond Boundary ---")
    context_bound_yes = MarketContext(
        timestamp=time.time(),
        spot_price=60001.0,
        strike_price=60000.0,
        tau_seconds=0.5,
        volatility=calibrated_vol,
        ofi=0.0
    )
    p_bound_yes = strategy.get_probability(context_bound_yes)
    print(f"Spot: ${context_bound_yes.spot_price:,.2f} > K: ${context_bound_yes.strike_price:,.2f} | Tau: {context_bound_yes.tau_seconds}s | Prob: {p_bound_yes:.2%}")
    
    context_bound_no = MarketContext(
        timestamp=time.time(),
        spot_price=59999.0,
        strike_price=60000.0,
        tau_seconds=0.5,
        volatility=calibrated_vol,
        ofi=0.0
    )
    p_bound_no = strategy.get_probability(context_bound_no)
    print(f"Spot: ${context_bound_no.spot_price:,.2f} < K: ${context_bound_no.strike_price:,.2f} | Tau: {context_bound_no.tau_seconds}s | Prob: {p_bound_no:.2%}")
    
    if p_bound_yes == 0.99 and p_bound_no == 0.01:
        # Note: get_probability clips to [0.01, 0.99]
        print("[OK] Boundary condition correctly clips to option payoffs at expiration.")
    else:
        print("[FAIL] Boundary condition clipping failure!")

    # 4. Zero-Assumptions REST Failure Test (strike is None)
    print("\n--- Testing Zero-Assumptions REST Failure ---")
    context_none = MarketContext(
        timestamp=time.time(),
        spot_price=60000.0,
        strike_price=None,  # REST failed, K not resolved yet
        tau_seconds=150.0,
        volatility=calibrated_vol,
        ofi=0.0
    )
    p_none = strategy.get_probability(context_none)
    print(f"Strike is None | YES Probability: {p_none}")
    
    if p_none is None:
        print("[OK] Returned None probability when strike price was unresolved (blocks trading).")
    else:
        print("[FAIL] Expected None probability for unresolved strike!")

if __name__ == "__main__":
    main()
