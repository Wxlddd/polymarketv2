import os
import sys
import numpy as np
import polars as pl
import argparse
import collections

# Add workspace to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from src.strategies.factory import StrategyFactory
from src.core.market_context import MarketContext
from src.core.strike_manager import StrikeManager

def main():
    parser = argparse.ArgumentParser(description="Calibrate V_MAX for DivergenceVelocityFilter")
    parser.add_argument("--file", type=str, required=True, help="Path to ticks.parquet or signals.csv")
    args = parser.parse_args()
    
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    config = SystemConfig()
    
    # Check if it's signals.csv or ticks.parquet
    if file_path.endswith(".csv"):
        print(f"Analyzing signals CSV file: {file_path}")
        df = pl.read_csv(file_path)
        # Columns: timestamp, spot_price, strike, model_prob, implied_prob, kelly_size, status
        timestamps = df["timestamp"].to_numpy()
        p_yes_arr = df["model_prob"].to_numpy()
        p_mkt_arr = df["implied_prob"].to_numpy()
    else:
        print(f"Analyzing ticks Parquet file (recalculating pricing): {file_path}")
        df = pl.read_parquet(file_path)
        df = df.sort("timestamp")
        
        strategy = StrategyFactory.get_strategy(config.STRATEGY_NAME, config)
        
        timestamps = df["timestamp"].to_numpy()
        spot_prices = df["spot_price"].to_numpy()
        ofis = df["ofi"].to_numpy() if "ofi" in df.columns else np.zeros(len(df))
        vols = df["volatility"].to_numpy() if "volatility" in df.columns else np.full(len(df), config.merton.DEFAULT_SIGMA)
        
        best_bids = df["best_bid"].to_numpy()
        best_asks = df["best_ask"].to_numpy()
        
        p_yes_arr = []
        p_mkt_arr = []
        
        current_expiry = None
        strike_manager = None
        p_yes_ema = None
        p_yes_ema_ts = 0.0
        
        for i in range(len(df)):
            t = float(timestamps[i])
            spot = float(spot_prices[i])
            ofi = float(ofis[i])
            vol = float(vols[i])
            bid = best_bids[i]
            ask = best_asks[i]
            
            # Resolve expiry & strike
            if current_expiry is None or t >= current_expiry:
                current_expiry = int(t) - (int(t) % 300) + 300
                strike_manager = StrikeManager(presumed_strike=spot, expiration_timestamp=current_expiry)
                p_yes_ema = None
                p_yes_ema_ts = 0.0
                strategy.reset()
                
            active_strike = strike_manager.get_strike(t, spot)
            tau_sec = max(0.0, current_expiry - t)
            
            context = MarketContext(
                timestamp=t,
                spot_price=spot,
                strike_price=active_strike,
                tau_seconds=tau_sec,
                volatility=vol,
                ofi=ofi,
                bids_l2=[],
                asks_l2=[]
            )
            
            p_yes_raw = strategy.get_probability(context)
            if p_yes_raw is not None:
                if p_yes_ema is None:
                    p_yes_ema = p_yes_raw
                    p_yes_ema_ts = t
                else:
                    base_halflife = config.merton.EMA_HALFLIFE_SEC
                    halflife = min(base_halflife, max(0.1, tau_sec / 10.0))
                    dt_ema = t - p_yes_ema_ts
                    alpha = 1.0 - (2.718281828 ** (-dt_ema / halflife)) if halflife > 0.0 else 1.0
                    p_yes_ema = alpha * p_yes_raw + (1.0 - alpha) * p_yes_ema
                    p_yes_ema_ts = t
            
            p_yes_arr.append(p_yes_ema)
            
            # Compute p_mkt
            if bid is not None and ask is not None:
                p_mkt = 0.5 * (bid + ask)
            elif bid is not None:
                p_mkt = bid
            elif ask is not None:
                p_mkt = ask
            else:
                p_mkt = p_yes_ema
            p_mkt_arr.append(p_mkt)
            
        p_yes_arr = np.array(p_yes_arr)
        p_mkt_arr = np.array(p_mkt_arr)
        
    # Now compute divergence, velocity, and acceleration
    velocities = []
    accelerations = []
    
    # Track historical (t, div, v) buffer
    buffer = collections.deque()
    
    lookback = config.risk.VELOCITY_LOOKBACK_SECONDS
    
    for i in range(len(timestamps)):
        t = float(timestamps[i])
        py = p_yes_arr[i]
        pm = p_mkt_arr[i]
        
        if py is None or pm is None or np.isnan(py) or np.isnan(pm):
            velocities.append(0.0)
            accelerations.append(0.0)
            continue
            
        div = py - pm
        
        # Find closest reference in buffer
        target_t = t - lookback
        closest_obs = None
        min_diff = float("inf")
        for obs in buffer:
            diff = abs(obs[0] - target_t)
            if diff < min_diff:
                min_diff = diff
                closest_obs = obs
                
        if closest_obs is None:
            v_t = 0.0
            a_t = 0.0
        else:
            dt = t - closest_obs[0]
            if dt < 1.0:
                v_t = 0.0
                a_t = 0.0
            else:
                v_t = (div - closest_obs[1]) / dt
                # acceleration is derivative of velocity
                a_t = (v_t - closest_obs[2]) / dt
                
        velocities.append(v_t)
        accelerations.append(a_t)
        buffer.append((t, div, v_t))
        
        # Pruning
        while buffer and buffer[0][0] < t - config.risk.DIVERGENCE_WINDOW_SECONDS:
            buffer.popleft()
            
    abs_velocities = np.abs(velocities)
    abs_accelerations = np.abs(accelerations)
    
    print("\n=== Divergence Velocity Filter Calibration Summary ===")
    print(f"Total Ticks Analyzed: {len(timestamps)}")
    print(f"Mean Absolute Velocity: {np.mean(abs_velocities):.6f} points/sec")
    print(f"Median Absolute Velocity: {np.median(abs_velocities):.6f} points/sec")
    print(f"90th Percentile Absolute Velocity: {np.percentile(abs_velocities, 90):.6f} points/sec")
    print(f"95th Percentile Absolute Velocity: {np.percentile(abs_velocities, 95):.6f} points/sec")
    print(f"99th Percentile Absolute Velocity: {np.percentile(abs_velocities, 99):.6f} points/sec")
    print(f"Maximum Absolute Velocity observed: {np.max(abs_velocities):.6f} points/sec")
    
    print(f"\nMean Absolute Acceleration: {np.mean(abs_accelerations):.6f} points/sec^2")
    print(f"95th Percentile Absolute Acceleration: {np.percentile(abs_accelerations, 95):.6f} points/sec^2")
    print(f"99th Percentile Absolute Acceleration: {np.percentile(abs_accelerations, 99):.6f} points/sec^2")
    
    # Recommendation calculation
    p95_v = np.percentile(abs_velocities, 95)
    p99_v = np.percentile(abs_velocities, 99)
    suggested_v_max_lower = max(0.001, p95_v * 1.5)
    suggested_v_max_upper = max(0.005, p99_v * 2.0)
    
    print("\n--- RECOMMENDATIONS ---")
    print(f"Suggested V_MAX Range: {suggested_v_max_lower:.5f} to {suggested_v_max_upper:.5f} points/sec")
    print(f"Current Config V_MAX: {config.risk.V_MAX:.5f} points/sec")
    print(f"  - If this is a normal/quiet market period, set V_MAX > 99th percentile ({p99_v:.5f}).")
    print(f"  - If this period contains a known toxic spike, V_MAX should be set below the maximum spike velocity.")

if __name__ == "__main__":
    main()
