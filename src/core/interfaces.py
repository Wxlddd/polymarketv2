from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

class ISpotFeed(ABC):
    """Abstract interface for receiving spot price updates."""
    
    @property
    @abstractmethod
    def price(self) -> Optional[float]:
        """Returns the current spot price."""
        pass

    @property
    @abstractmethod
    def last_updated(self) -> float:
        """Returns the epoch timestamp of the last spot update."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Returns connection status to feed source."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Starts background ingestion loop."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stops background ingestion loop."""
        pass


class IOrderBook(ABC):
    """Abstract interface for managing order book state (real or shadow)."""

    @abstractmethod
    def update_book(self, bid_updates: List[Tuple[float, float]], ask_updates: List[Tuple[float, float]], is_snapshot: bool = False, timestamp: float = 0.0) -> float:
        """Updates internal levels and returns the computed OFI."""
        pass

    @abstractmethod
    def get_sorted_bids(self) -> List[Tuple[float, float]]:
        """Returns descending bids: (price, size)."""
        pass

    @abstractmethod
    def get_sorted_asks(self) -> List[Tuple[float, float]]:
        """Returns ascending asks: (price, size)."""
        pass

    @abstractmethod
    def get_top_of_book(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Returns (best_bid, best_ask) as (price, size) tuples."""
        pass


class IExecutionClient(ABC):
    """Abstract interface for risk management, trades routing, and account tracking."""

    @property
    @abstractmethod
    def cash_balance(self) -> float:
        """Returns current available USD capital."""
        pass

    @abstractmethod
    def get_position_size(self, side: str) -> float:
        """Returns active position size for YES or NO contracts."""
        pass

    @abstractmethod
    async def execute_trade(self, side: str, qty: float, price: float, ev: float, expected_slippage_bps: float, context_state: Dict[str, Any], is_maker: bool = False) -> Dict[str, Any]:
        """
        Executes a paper or live trade.
        Returns a dictionary summarizing execution details (fill price, real slippage, pnl).
        """
        pass


class IDataRecorder(ABC):
    """Abstract interface for data logging and backtest tracking."""

    @abstractmethod
    def record_tick(self, timestamp: float, spot_price: float, ofi: float, volatility: float, bids_l2: List[Tuple[float, float]], asks_l2: List[Tuple[float, float]]) -> None:
        """Logs a single market tick update with L2 book depth."""
        pass

    @abstractmethod
    def record_signal(self, timestamp: float, spot_price: float, strike: float, model_prob: float, implied_prob: float, kelly_size: float, status: str) -> None:
        """Logs strategy-generated model signals."""
        pass

    @abstractmethod
    def record_trade(self, timestamp: float, side: str, qty: float, vwap: float, p_market: float, expected_slippage_bps: float, realized_slippage_bps: float, ev: float, strike: float, resolved_won: bool, pnl: float, capital: float) -> None:
        """Logs an executed trade event."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Forces buffered log elements to be written to disk."""
        pass
