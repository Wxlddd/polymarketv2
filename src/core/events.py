from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass(frozen=True)
class TickEvent:
    """Event triggered on spot feed or orderbook tick arrival."""
    timestamp: float
    spot_price: float
    ofi: float
    volatility: float
    bids_l2: List[Tuple[float, float]]
    asks_l2: List[Tuple[float, float]]

@dataclass(frozen=True)
class SignalEvent:
    """Event representing a strategy pricing output."""
    timestamp: float
    spot_price: float
    strike: float
    model_prob: float
    implied_prob: float
    kelly_size: float
    status: str  # e.g., 'BUY_YES', 'BUY_NO', 'HOLD', 'SELL_YES', 'SELL_NO', 'REJECT_PIN_RISK', etc.

@dataclass(frozen=True)
class TradeExecutedEvent:
    """Event emitted when an order gets successfully filled (live or simulated)."""
    timestamp: float
    side: str  # BUY_YES, BUY_NO, SELL_YES, SELL_NO
    qty: float
    vwap: float
    p_market: float
    expected_slippage_bps: float
    realized_slippage_bps: float
    ev: float
    strike: float
    resolved_won: bool
    pnl: float
    capital: float
