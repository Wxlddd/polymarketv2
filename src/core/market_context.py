from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass(frozen=True)
class MarketContext:
    """
    Standard state structure carrying market data, parameters, and indicators
    for options evaluation.
    """
    timestamp: float
    spot_price: float
    strike_price: float
    tau_seconds: float
    volatility: float
    ofi: float
    bids_l2: List[Tuple[float, float]] = field(default_factory=list)
    asks_l2: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        """Returns the simple mid price from the orderbook depth."""
        if self.bids_l2 and self.asks_l2:
            return 0.5 * (self.bids_l2[0][0] + self.asks_l2[0][0])
        return self.spot_price

    @property
    def spread(self) -> float:
        """Returns the top of book spread in absolute price units."""
        if self.bids_l2 and self.asks_l2:
            return self.asks_l2[0][0] - self.bids_l2[0][0]
        return 0.0
