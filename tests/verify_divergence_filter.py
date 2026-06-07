import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.execution.divergence_filter import DivergenceVelocityFilter
from src.execution.engine import ExecutionEngine
from src.execution.clients import MockExecutionClient
from src.execution.shadow_book import ShadowOrderBook
from src.logging.recorder import DataRecorder

async def test_scenarios():
    print("=== Testing DivergenceVelocityFilter Scenarios ===")
    
    config = SystemConfig()
    
    # Check that settings load correctly and default to expected values
    assert config.risk.V_MAX == 0.005, f"V_MAX should default to 0.005, got {config.risk.V_MAX}"
    assert config.risk.MIN_KELLY_THRESHOLD == 0.01, f"MIN_KELLY_THRESHOLD should default to 0.01, got {config.risk.MIN_KELLY_THRESHOLD}"
    
    # 1. Initialize Filter directly
    filt = DivergenceVelocityFilter(config)
    
    # Scenario A: Slow Drift
    # Over 10 seconds, divergence goes from 0.0 to 0.02 (drift rate = 0.02/10 = 0.002 points/sec)
    # 0.002 < V_MAX (0.005). Scale should be max(0, 1 - (0.002/0.005)^2) = 1 - 0.16 = 0.84.
    print("\n--- Scenario A: Slow Drift ---")
    t0 = 1000.0
    scale, v, a = filt.get_scale(p_yes_smoothed=0.5, p_mkt=0.5, timestamp=t0)
    print(f"t={t0:.1f} | p_yes=0.50, p_mkt=0.50 | scale={scale:.4f}, v_t={v:.6f}, a_t={a:.6f}")
    
    for step in range(1, 11):
        t = t0 + step
        p_mkt = 0.5 - (0.02 * (step / 10.0))  # p_mkt moves from 0.5 to 0.48, making divergence 0.02
        scale, v, a = filt.get_scale(p_yes_smoothed=0.5, p_mkt=p_mkt, timestamp=t)
        print(f"t={t:.1f} | p_yes=0.50, p_mkt={p_mkt:.4f} | scale={scale:.4f}, v_t={v:.6f}, a_t={a:.6f}")
        
    print(f"Final scale: {scale:.4f} (expected around 0.84), velocity: {v:.6f} (expected 0.002), acceleration: {a:.6f}")
    if abs(scale - 0.84) < 0.01 and abs(v - 0.002) < 1e-5:
        print("[OK] Scenario A: Slow Drift calculation correct!")
    else:
        print("[FAIL] Scenario A: Drift calculation mismatch!")

    # Scenario B: Fast Spike
    filt = DivergenceVelocityFilter(config)
    print("\n--- Scenario B: Fast Spike ---")
    # Divergence jumps by 0.10 in 10 seconds (velocity = 0.10/10 = 0.010 points/sec > V_MAX)
    # Expect scale = 0.0
    t0 = 2000.0
    filt.get_scale(p_yes_smoothed=0.5, p_mkt=0.5, timestamp=t0)
    for step in range(1, 11):
        t = t0 + step
        p_mkt = 0.5 - (0.10 * (step / 10.0))  # p_mkt moves from 0.5 to 0.40, divergence 0.10
        scale, v, a = filt.get_scale(p_yes_smoothed=0.5, p_mkt=p_mkt, timestamp=t)
        print(f"t={t:.1f} | p_yes=0.50, p_mkt={p_mkt:.4f} | scale={scale:.4f}, v_t={v:.6f}, a_t={a:.6f}")
        
    print(f"Final scale: {scale:.4f} (expected 0.0), velocity: {v:.6f} (expected 0.010)")
    if scale == 0.0 and abs(v - 0.010) < 1e-5:
        print("[OK] Scenario B: Fast Spike blocked correctly!")
    else:
        print("[FAIL] Scenario B: Fast Spike failure!")

    # Scenario C: Accelerating Divergence
    filt = DivergenceVelocityFilter(config)
    print("\n--- Scenario C: Accelerating Divergence (Override) ---")
    # divergence = 0.0003 * dt^2
    # At t=10, divergence = 0.03. Velocity = 0.006 (above half threshold 0.0025).
    # Since divergence is accelerating away, expect override to force scale = 0.0.
    t0 = 3000.0
    filt.get_scale(p_yes_smoothed=0.5, p_mkt=0.5, timestamp=t0)
    for step in range(1, 11):
        t = t0 + step
        dt = step
        divergence = 0.0003 * (dt ** 2)
        p_mkt = 0.5 - divergence
        scale, v, a = filt.get_scale(p_yes_smoothed=0.5, p_mkt=p_mkt, timestamp=t)
        print(f"t={t:.1f} | p_yes=0.50, p_mkt={p_mkt:.4f} | scale={scale:.4f}, v_t={v:.6f}, a_t={a:.6f}")
        
    print(f"Final scale: {scale:.4f} (expected 0.0 due to acceleration override), velocity: {v:.6f}, acceleration: {a:.6f}")
    if scale == 0.0 and a > 0.0 and abs(v) > 0.0025:
        print("[OK] Scenario C: Accelerating override blocked correctly!")
    else:
        print("[FAIL] Scenario C: Accelerating override failure!")

    # 2. Integration check with ExecutionEngine
    print("\n--- Testing Integration inside ExecutionEngine ---")
    recorder = DataRecorder(config.LOG_DIR, "merton_test", "div_test_run", buffer_size=10)
    shadow_book = ShadowOrderBook()
    
    # Set initial bids/asks snapshot: p_mkt = 0.45
    shadow_book.update_book(
        [(0.45, 1000.0)],
        [(0.46, 1000.0)],
        is_snapshot=True
    )
    client = MockExecutionClient(config, recorder, shadow_book)
    engine = ExecutionEngine(config)
    
    # First tick: no history. Trade should not be blocked.
    t = 4000.0
    context = MarketContext(
        timestamp=t,
        spot_price=60000.0,
        strike_price=60000.0,
        tau_seconds=100.0,
        volatility=0.25,
        ofi=0.0,
        bids_l2=shadow_book.get_sorted_bids(),
        asks_l2=shadow_book.get_sorted_asks()
    )
    decision = engine.evaluate_and_trade(p_yes=0.55, context=context, client=client)
    print(f"t={t:.1f} | Initial Decision: {decision['side']} (Expected: BUY_YES)")
    assert decision["side"] == "BUY_YES", "Initial trade should be allowed"

    # Fast jump: over the next 10 seconds, p_mkt drops to 0.35 (severe divergence velocity of 0.010/s)
    # Trade should be blocked!
    for step in range(1, 11):
        t_tick = t + step
        p_bid = 0.45 - (0.10 * (step / 10.0))
        p_ask = p_bid + 0.01
        
        shadow_book.update_book(
            [(p_bid, 1000.0)],
            [(p_ask, 1000.0)],
            is_snapshot=True,
            timestamp=t_tick
        )
        
        context = MarketContext(
            timestamp=t_tick,
            spot_price=60000.0,
            strike_price=60000.0,
            tau_seconds=100.0,
            volatility=0.25,
            ofi=0.0,
            bids_l2=shadow_book.get_sorted_bids(),
            asks_l2=shadow_book.get_sorted_asks()
        )
        
        decision = engine.evaluate_and_trade(p_yes=0.55, context=context, client=client)
        print(f"t={t_tick:.1f} | Decision: {decision['side']} | Reason: {decision.get('reason', 'N/A')}")
        
    # The last decision must be HOLD and blocked by the divergence filter
    if decision["side"] == "HOLD" and decision.get("reason") == "REJECT_DIVERGENCE_VELOCITY":
        print("[OK] Integration works correctly! High velocity divergence blocked trade execution.")
    else:
        print("[FAIL] Integration failed! High velocity divergence was not blocked.")
        
    recorder.flush()

if __name__ == "__main__":
    asyncio.run(test_scenarios())
