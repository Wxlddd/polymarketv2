import collections
import logging
from typing import Tuple
from config.settings import SystemConfig

logger = logging.getLogger("DivergenceVelocityFilter")

class DivergenceVelocityFilter:
    """
    DivergenceVelocityFilter
    Tracks the rate of change and acceleration of divergence between smoothed fair probability and market price.
    Calculates a continuous Kelly scaling factor to block or damp execution during periods of rapid toxic divergence.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        # Buffer tracks: {"timestamp": t, "divergence": div, "velocity": v}
        self.buffer = collections.deque()
        self.last_divergence = 0.0
        self.last_velocity = 0.0
        self.last_acceleration = 0.0
        self.last_scale = 1.0
        self._last_ts = 0.0
        
    def get_scale(self, p_yes_smoothed: float, p_mkt: float, timestamp: float) -> Tuple[float, float, float]:
        """
        Processes a single tick observation, updates the historical buffer,
        calculates velocity and acceleration, and returns the Kelly scaling factor.
        
        Args:
            p_yes_smoothed: The EMA-smoothed model probability of YES.
            p_mkt: The market-implied probability from top of the book.
            timestamp: The current epoch timestamp in seconds.
            
        Returns:
            Tuple of (scale, velocity, acceleration)
            - scale: A multiplier in [0.0, 1.0] to apply to the Kelly fraction.
            - velocity: Rate of change of divergence (probability points per second).
            - acceleration: Second derivative of divergence (probability points per second squared).
        """
        # Guard against unresolved inputs
        if p_yes_smoothed is None or p_mkt is None:
            return 1.0, 0.0, 0.0

        # Cache check for duplicate calls with the same timestamp
        if self._last_ts > 0.0 and timestamp == self._last_ts:
            return self.last_scale, self.last_velocity, self.last_acceleration

        # 1. Compute current divergence
        divergence_t = p_yes_smoothed - p_mkt
        
        # 2. Extract configuration parameters
        window_sec = self.config.risk.DIVERGENCE_WINDOW_SECONDS
        lookback_sec = self.config.risk.VELOCITY_LOOKBACK_SECONDS
        v_max = self.config.risk.V_MAX
        gamma = self.config.risk.GAMMA
        
        # 3. Find historical reference observation closest to timestamp - VELOCITY_LOOKBACK_SECONDS
        target_t = timestamp - lookback_sec
        closest_obs = None
        min_diff = float("inf")
        
        for obs in self.buffer:
            diff = abs(obs["timestamp"] - target_t)
            if diff < min_diff:
                min_diff = diff
                closest_obs = obs
                
        # 4. Compute velocity and acceleration
        if closest_obs is None:
            # First observation or no history: default to 0
            v_t = 0.0
            a_t = 0.0
        else:
            dt = timestamp - closest_obs["timestamp"]
            if dt < 1.0:
                # If there's less than 1.0s difference between the current tick and the reference tick,
                # we don't have enough history to make a meaningful/non-noisy calculation.
                v_t = 0.0
                a_t = 0.0
            else:
                # v_t in probability points per second
                v_t = (divergence_t - closest_obs["divergence"]) / dt
                # a_t (second derivative of divergence over time)
                a_t = (v_t - closest_obs["velocity"]) / dt
                
        # 5. Append current observation to rolling buffer
        self.buffer.append({
            "timestamp": timestamp,
            "divergence": divergence_t,
            "velocity": v_t
        })
        
        # 6. Prune buffer of old observations
        cutoff_t = timestamp - window_sec
        while self.buffer and self.buffer[0]["timestamp"] < cutoff_t:
            self.buffer.popleft()
            
        # 7. Compute continuous Kelly scale factor
        # scale(v_t) = max(0, 1 - (|v_t| / V_max)^gamma)
        if v_max > 0.0:
            ratio = abs(v_t) / v_max
            scale_val = 1.0 - (ratio ** gamma)
            scale_val = max(0.0, min(1.0, scale_val))
        else:
            scale_val = 1.0
            
        # 8. Apply acceleration override
        # Symmetrically handles both positive and negative divergence directions.
        # Divergence is accelerating away from fair value if:
        # - divergence > 0, v_t > 0, and a_t > 0 (accelerating positive divergence)
        # - divergence < 0, v_t < 0, and a_t < 0 (accelerating negative divergence)
        # Thus, if a_t * divergence_t > 0 and |v_t| > V_max * 0.5, we force scale = 0.0.
        is_accelerating_away = (a_t * divergence_t > 0.0)
        half_threshold = v_max * 0.5
        
        if is_accelerating_away and abs(v_t) > half_threshold:
            scale_val = 0.0
            
        self.last_divergence = divergence_t
        self.last_velocity = v_t
        self.last_acceleration = a_t
        self.last_scale = scale_val
        self._last_ts = timestamp
            
        return scale_val, v_t, a_t
