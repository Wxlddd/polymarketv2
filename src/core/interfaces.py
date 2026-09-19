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


class OrderInstruction:
    """
    Lightweight, pre-allocated-friendly routing instruction for low-latency dispatch.
    Uses __slots__ to eliminate dictionary overhead and memory allocations.
    """
    __slots__ = ('action', 'side', 'price', 'qty', 'order_id', 'regime')
    
    def __init__(self, action: str, side: str, price: float, qty: float, order_id: str = "", regime: str = "A"):
        self.action = action      # "NEW", "CANCEL", "REPLACE"
        self.side = side          # "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO", "HOLD"
        self.price = price        # Rounded target execution price
        self.qty = qty            # Sized target execution quantity
        self.order_id = order_id  # Target order identifier for cancels/replaces
        self.regime = regime      # Regime label: "A" (Maker), "B" (Taker), "C" (Unwind), "SAFE" (Safe Mode)

    def __repr__(self) -> str:
        return f"OrderInstruction(action={self.action}, side={self.side}, price={self.price:.4f}, qty={self.qty:.2f}, order_id={self.order_id}, regime={self.regime})"

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
    async def process_instruction(self, instruction: OrderInstruction, context_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a routing instruction (NEW, CANCEL, REPLACE).
        For Taker (Regime B/PANIC), executes immediately (IOC).
        For Maker (Regime A/C), places a limit order resting in the mock client book.
        """
        pass

    @abstractmethod
    def process_market_data(self, bids_l2: List[Tuple[float, float]], asks_l2: List[Tuple[float, float]], context_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Updates the L2 book and evaluates resting Maker limit orders for queue depth and simulated fills.
        Returns a list of fill results (if any).
        """
        pass


class IDataRecorder(ABC):
    """Abstract interface for data logging and backtest tracking."""

    @abstractmethod
    def record_tick(
        self, timestamp: float, spot_price: float, ofi: float, volatility: float,
        bids_l2: List[Tuple[float, float]], asks_l2: List[Tuple[float, float]],
        top_bid: Optional[Tuple[float, float]] = None, top_ask: Optional[Tuple[float, float]] = None,
        is_snapshot: bool = False
    ) -> None:
        """
        Logs a single market tick. bids_l2/asks_l2 are the raw feed update (snapshot or delta,
        as flagged by is_snapshot) so the book can be replayed; top_bid/top_ask are the
        reconciled top of book after applying it.
        """
        pass

    def record_print(self, timestamp: float, price: float, size: float, side: str, exchange_ts: Optional[float] = None) -> None:
        """Logs an exchange trade print (last_trade_price) for the YES token. Optional for recorders that do not persist prints."""
        return None

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
