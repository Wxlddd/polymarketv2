import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from src.core.interfaces import IOrderBook

logger = logging.getLogger("ShadowOrderBook")

class ShadowOrderBook(IOrderBook):
    """
    Shadow Order Book Proxy.
    Maintains q_real (actual live liquidity from WebSocket) and q_shadow 
    (residual liquidity after our simulated/paper fills) to prevent double-dipping.
    """
    
    def __init__(self):
        # bids: price -> qty
        self.q_real_bids: Dict[float, float] = {}
        self.q_shadow_bids: Dict[float, float] = {}
        
        # asks: price -> qty
        self.q_real_asks: Dict[float, float] = {}
        self.q_shadow_asks: Dict[float, float] = {}
        
        # Historical top-of-book memory for OFI calculations
        self.prev_best_bid_price: float = 0.0
        self.prev_best_bid_qty: float = 0.0
        self.prev_best_ask_price: float = 0.0
        self.prev_best_ask_qty: float = 0.0
        self.smoothed_ofi: float = 0.0

    def update_book(
        self, 
        bid_updates: List[Tuple[float, float]], 
        ask_updates: List[Tuple[float, float]], 
        is_snapshot: bool = False
    ) -> float:
        """
        WebSocket Reconciliation Algorithm.
        Reconciles live WebSocket updates (absolute values) with outstanding shadow fills,
        returning the computed smoothed Order Flow Imbalance (OFI).
        """
        if is_snapshot:
            self.q_real_bids.clear()
            self.q_shadow_bids.clear()
            self.q_real_asks.clear()
            self.q_shadow_asks.clear()

        # Reconcile bids L2 delta
        for price, qty in bid_updates:
            price_key = round(price, 6)
            if qty == 0.0:
                self.q_real_bids.pop(price_key, None)
                self.q_shadow_bids.pop(price_key, None)
            else:
                delta = qty - self.q_real_bids.get(price_key, 0.0)
                self.q_shadow_bids[price_key] = max(0.0, self.q_shadow_bids.get(price_key, 0.0) + delta)
                self.q_real_bids[price_key] = qty

        # Reconcile asks L2 delta
        for price, qty in ask_updates:
            price_key = round(price, 6)
            if qty == 0.0:
                self.q_real_asks.pop(price_key, None)
                self.q_shadow_asks.pop(price_key, None)
            else:
                delta = qty - self.q_real_asks.get(price_key, 0.0)
                self.q_shadow_asks[price_key] = max(0.0, self.q_shadow_asks.get(price_key, 0.0) + delta)
                self.q_real_asks[price_key] = qty

        # Prune dead levels from shadow book
        for p in list(self.q_shadow_bids.keys()):
            if self.q_shadow_bids[p] <= 1e-9:
                self.q_shadow_bids.pop(p, None)

        for p in list(self.q_shadow_asks.keys()):
            if self.q_shadow_asks[p] <= 1e-9:
                self.q_shadow_asks.pop(p, None)

        # Recompute Order Flow Imbalance based on q_shadow state
        raw_ofi = self._calculate_ofi()
        self.smoothed_ofi = 0.1 * raw_ofi + 0.9 * self.smoothed_ofi
        return self.smoothed_ofi

    def paper_execute(self, price: float, qty_executed: float, is_bid: bool) -> None:
        """
        Deducts filled quantity from the shadow state to simulate book depletion.
        If is_bid is True, we deduct from bids (selling to bid); else we deduct from asks (buying from ask).
        """
        price_key = round(price, 6)
        target_dict = self.q_shadow_bids if is_bid else self.q_shadow_asks
        
        if price_key in target_dict:
            target_dict[price_key] = max(0.0, target_dict[price_key] - qty_executed)
            if target_dict[price_key] <= 1e-9:
                target_dict.pop(price_key, None)
        else:
            # Fallback if executing at a price that isn't currently at top level (HFT latency gap)
            logger.debug(f"[ShadowOrderBook] Paper execute failed to locate level: {price_key}")

    def get_sorted_bids(self) -> List[Tuple[float, float]]:
        """Returns shadow bids sorted descending (highest bid first)."""
        return sorted(
            [(p, q) for p, q in self.q_shadow_bids.items() if q > 1e-9],
            key=lambda x: x[0],
            reverse=True
        )

    def get_sorted_asks(self) -> List[Tuple[float, float]]:
        """Returns shadow asks sorted ascending (lowest ask first)."""
        return sorted(
            [(p, q) for p, q in self.q_shadow_asks.items() if q > 1e-9],
            key=lambda x: x[0]
        )

    def get_top_of_book(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Returns best (bid, ask) as (price, quantity) tuples from shadow book."""
        sorted_bids = self.get_sorted_bids()
        sorted_asks = self.get_sorted_asks()
        
        best_bid = sorted_bids[0] if sorted_bids else None
        best_ask = sorted_asks[0] if sorted_asks else None
        return best_bid, best_ask

    def get_market_top_of_book(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Returns best (bid, ask) from the REAL market book (q_real), which is never
        depleted by paper_execute fills. Use this for p_mkt display so that simulated
        trades do not corrupt the displayed market-implied probability.
        """
        sorted_bids = sorted(
            [(p, q) for p, q in self.q_real_bids.items() if q > 1e-9],
            key=lambda x: x[0],
            reverse=True
        )
        sorted_asks = sorted(
            [(p, q) for p, q in self.q_real_asks.items() if q > 1e-9],
            key=lambda x: x[0]
        )
        best_bid = sorted_bids[0] if sorted_bids else None
        best_ask = sorted_asks[0] if sorted_asks else None
        return best_bid, best_ask


    def _calculate_ofi(self) -> float:
        """Calculates Order Flow Imbalance (OFI) for the top level changes."""
        best_bid, best_ask = self.get_top_of_book()
        if not best_bid or not best_ask:
            return 0.0
            
        cur_bid_p, cur_bid_q = best_bid
        cur_ask_p, cur_ask_q = best_ask
        
        if self.prev_best_bid_price == 0.0:
            self.prev_best_bid_price = cur_bid_p
            self.prev_best_bid_qty = cur_bid_q
            self.prev_best_ask_price = cur_ask_p
            self.prev_best_ask_qty = cur_ask_q
            return 0.0
            
        # Bid Delta
        if cur_bid_p > self.prev_best_bid_price:
            d_bid = cur_bid_q
        elif cur_bid_p == self.prev_best_bid_price:
            d_bid = cur_bid_q - self.prev_best_bid_qty
        else:
            d_bid = -self.prev_best_bid_qty
            
        # Ask Delta
        if cur_ask_p < self.prev_best_ask_price:
            d_ask = cur_ask_q
        elif cur_ask_p == self.prev_best_ask_price:
            d_ask = cur_ask_q - self.prev_best_ask_qty
        else:
            d_ask = -self.prev_best_ask_qty
            
        # Save state memory
        self.prev_best_bid_price = cur_bid_p
        self.prev_best_bid_qty = cur_bid_q
        self.prev_best_ask_price = cur_ask_p
        self.prev_best_ask_qty = cur_ask_q
        
        return d_bid - d_ask
