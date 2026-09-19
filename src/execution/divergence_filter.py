import bisect
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
        # Rolling history as parallel, time-ordered lists so the lookback reference
        # can be found by bisection. Ticks arrive in order, so appending keeps them sorted.
        self._ts: list = []
        self._div: list = []
        self._vel: list = []
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
        
        # 3. Find the historical observation closest to timestamp - VELOCITY_LOOKBACK_SECONDS.
        #    Bisection on the time-ordered history (earliest wins on a tie, matching a
        #    left-to-right scan). A linear scan here was ~85% of backtest runtime.
        target_t = timestamp - lookback_sec
        ref_idx = -1
        if self._ts:
            i = bisect.bisect_left(self._ts, target_t)
            if i == 0:
                ref_idx = 0
            elif i == len(self._ts):
                ref_idx = i - 1
            else:
                ref_idx = i - 1 if (target_t - self._ts[i - 1]) <= (self._ts[i] - target_t) else i

        # 4. Compute velocity and acceleration
        if ref_idx < 0:
            # First observation or no history: default to 0
            v_t = 0.0
            a_t = 0.0
        else:
            dt = timestamp - self._ts[ref_idx]
            if dt < 1.0:
                # If there's less than 1.0s difference between the current tick and the reference tick,
                # we don't have enough history to make a meaningful/non-noisy calculation.
                v_t = 0.0
                a_t = 0.0
            else:
                # v_t in probability points per second
                v_t = (divergence_t - self._div[ref_idx]) / dt
                # a_t (second derivative of divergence over time)
                a_t = (v_t - self._vel[ref_idx]) / dt

        # 5. Append current observation to rolling history
        self._ts.append(timestamp)
        self._div.append(divergence_t)
        self._vel.append(v_t)

        # 6. Prune observations older than the window
        cutoff_t = timestamp - window_sec
        k = bisect.bisect_left(self._ts, cutoff_t)
        if k:
            del self._ts[:k]
            del self._div[:k]
            del self._vel[:k]
            
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
