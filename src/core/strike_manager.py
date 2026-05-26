import logging
from typing import Optional

logger = logging.getLogger("StrikeManager")

class StrikeManager:
    """
    Manages the Strike Price (K / Price to Beat) for the market cycle.
    
    1. During active trading (current_time < expiration_time), uses a presumed strike
       configured manually via .env (presumed_strike).
    2. Once the expiration time is reached or crossed (current_time >= expiration_time),
       the final strike price K is locked dynamically as the first spot tick at/after expiration.
    """
    
    def __init__(self, presumed_strike: float, expiration_timestamp: float):
        self.presumed_strike = presumed_strike
        self.expiration_timestamp = expiration_timestamp
        self.resolved_strike: Optional[float] = None

    def get_strike(self, current_time: float, current_spot: float) -> float:
        """
        Retrieves the strike price based on time and spot price ticks.
        """
        # If already resolved, return the resolved strike
        if self.resolved_strike is not None:
            return self.resolved_strike
            
        # Check if expiration has been reached or crossed
        if current_time >= self.expiration_timestamp and self.expiration_timestamp > 0.0:
            self.resolved_strike = current_spot
            logger.info(
                f"[StrikeManager] Rollover occurred. Locked final Price to Beat (Strike K) "
                f"at spot tick: ${self.resolved_strike:,.2f} (Timestamp: {current_time})"
            )
            return self.resolved_strike
            
        # Default fallback during active option trading
        return self.presumed_strike

    def reset(self, new_presumed_strike: float, new_expiration_timestamp: float) -> None:
        """Resets the state for a new option/market cycle."""
        self.presumed_strike = new_presumed_strike
        self.expiration_timestamp = new_expiration_timestamp
        self.resolved_strike = None
        logger.info(
            f"[StrikeManager] Reset. Presumed Strike: ${new_presumed_strike:,.2f}, "
            f"Expiration: {new_expiration_timestamp}"
        )
