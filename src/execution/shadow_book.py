import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from src.core.interfaces import IOrderBook

logger = logging.getLogger("ShadowOrderBook")

# ---------------------------------------------------------------------------
# ConsumptionTracker — Shadow Ledger
# ---------------------------------------------------------------------------
# Tracks V_cons(p, t): the volume consumed by our bot at each price level,
# with an exponential half-life decay that models Market Maker replenishment.
#
#   V_cons(p, t) = V_cons(p, t_last) * exp(-λ * (t - t_last))
#   where λ = ln(2) / half_life
#
# This tracker is kept entirely separate from the historical feed state
# (V_hist) so that feed updates never mutate the consumption ledger directly.
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)

class _ConsumptionTracker:
    """
    Parallel shadow ledger that tracks the bot's consumed liquidity per price
    level with lazy exponential-decay replenishment.

    Key invariant: V_cons(p) is ALWAYS ≤ V_hist(p) after reconciliation.
    """

    def __init__(self, half_life: float = 0.2):
        """
        Args:
            half_life: Half-life (seconds) for exponential replenishment decay.
                       Default 0.2 s ≈ typical HFT MM re-quoting latency.
        """
        self._half_life: float = half_life
        self._lambda: float = _LN2 / half_life if half_life > 0.0 else 0.0

        # V_cons(p) — current consumed volume at price level p
        self._v_cons: Dict[float, float] = {}
        # t_last(p) — timestamp of last decay application at price level p
        self._t_last: Dict[float, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decay(self, price: float, t_now: float) -> float:
        """
        Applies lazy exponential decay to V_cons(price) and returns the
        decayed value WITHOUT writing it back.  Caller decides whether to
        persist the result.
        """
        v = self._v_cons.get(price, 0.0)
        if v <= 0.0:
            return 0.0
        t_last = self._t_last.get(price, t_now)
        dt = t_now - t_last
        if dt <= 0.0 or self._lambda <= 0.0:
            return v
        return v * math.exp(-self._lambda * dt)

    def _apply_decay(self, price: float, t_now: float) -> float:
        """Applies lazy decay in-place and returns the new V_cons value."""
        v_new = self._decay(price, t_now)
        if v_new <= 1e-12:
            self._v_cons.pop(price, None)
            self._t_last.pop(price, None)
            return 0.0
        self._v_cons[price] = v_new
        self._t_last[price] = t_now
        return v_new

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_v_eff(self, price: float, v_hist: float, t_now: float) -> float:
        """
        Returns the effective available volume at `price` at time `t_now`:

            V_eff(p, t) = max(0, V_hist(p, t) - V_cons(p, t))

        Decay is applied lazily before the subtraction.
        """
        v_cons = self._decay(price, t_now)
        return max(0.0, v_hist - v_cons)

    def consume(self, price: float, size: float, t_now: float) -> None:
        """
        Records a bot fill of `size` contracts at `price` at time `t_now`.

            V_cons(p, t) ← V_cons(p, t) + S

        Decay is applied lazily before accumulating.
        """
        # Decay first so we add S to the already-decayed baseline
        v_cons = self._apply_decay(price, t_now)
        self._v_cons[price] = v_cons + size
        self._t_last[price] = t_now

    def reconcile(self, price: float, v_old: float, v_new: float, t_now: float) -> None:
        """
        Event-driven reconciliation when a feed tick arrives at `price`.

        Case A — Replenishment (ΔV > 0):
            V_cons(p) ← max(0, V_cons(p) − ΔV)
            The MM added fresh liquidity; absorb it against our debt first.

        Case B — Drop (ΔV < 0) or level gone (V_new == 0):
            V_cons(p) ← min(V_cons(p), V_new)
            Our debt can never exceed the physical liquidity remaining.

        Decay is applied lazily before reconciliation.
        """
        # Decay first
        v_cons = self._apply_decay(price, t_now)

        if v_new == 0.0:
            # Level fully cleared — zero out debt automatically
            self._v_cons.pop(price, None)
            self._t_last.pop(price, None)
            return

        delta = v_new - v_old

        if delta > 0.0:
            # Case A: genuine replenishment from MM — reduce our debt
            v_cons = max(0.0, v_cons - delta)
        else:
            # Case B: level shrank (third-party consumption or MM pull) —
            # cap our debt so it never exceeds the remaining physical qty.
            v_cons = min(v_cons, v_new)

        if v_cons <= 1e-12:
            self._v_cons.pop(price, None)
            self._t_last.pop(price, None)
        else:
            self._v_cons[price] = v_cons
            self._t_last[price] = t_now

    def reset(self) -> None:
        """Clears all consumption state (called on snapshot/rollover)."""
        self._v_cons.clear()
        self._t_last.clear()

    @property
    def half_life(self) -> float:
        return self._half_life

    @half_life.setter
    def half_life(self, value: float) -> None:
        self._half_life = value
        self._lambda = _LN2 / value if value > 0.0 else 0.0


# ---------------------------------------------------------------------------
# ShadowOrderBook — Refactored with ConsumptionTracker
# ---------------------------------------------------------------------------

class ShadowOrderBook(IOrderBook):
    """
    Shadow Order Book Proxy — Refactored with Rigorous Consumption Ledger.

    Architecture:
    ─────────────
    • q_real_{bids,asks}  — Historical book (V_hist): pure feed state, never
                            mutated by bot fills.  Updated on every WebSocket
                            delta via update_book().

    • _tracker_{bids,asks} — ConsumptionTracker (V_cons): parallel ledger that
                             records ONLY what our bot consumed.  Decays
                             exponentially to model MM replenishment latency.

    • get_sorted_{bids,asks}() — Returns V_eff = max(0, V_hist − V_cons) levels
                                 for the execution engine to walk.

    • paper_execute()     — Registers bot fills into the tracker ledger
                            (consume()) for EXACT levels walked by the IOC loop
                            in MockExecutionClient.  Does NOT mutate q_real.

    • update_book()       — Applies feed deltas to q_real and calls
                            tracker.reconcile() for event-driven correction.
    """

    def __init__(self, half_life: float = 0.2):
        """
        Args:
            half_life: MM replenishment half-life in seconds for the exponential
                       decay in ConsumptionTracker.  Default: 0.2 s.
        """
        # ── Historical Feed State (V_hist) ──────────────────────────────────
        # These dicts are ONLY written by update_book() and NEVER by paper_execute.
        self.q_real_bids: Dict[float, float] = {}
        self.q_real_asks: Dict[float, float] = {}

        # ── Consumption Ledgers (V_cons) ────────────────────────────────────
        self._tracker_bids = _ConsumptionTracker(half_life=half_life)
        self._tracker_asks = _ConsumptionTracker(half_life=half_life)

        # Current simulation clock — updated on every update_book() call
        self._t_now: float = 0.0

        # ── OFI State ────────────────────────────────────────────────────────
        self.prev_best_bid_price: float = 0.0
        self.prev_best_bid_qty: float = 0.0
        self.prev_best_ask_price: float = 0.0
        self.prev_best_ask_qty: float = 0.0
        self.smoothed_ofi: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Core Feed Integration
    # ──────────────────────────────────────────────────────────────────────────

    def update_book(
        self,
        bid_updates: List[Tuple[float, float]],
        ask_updates: List[Tuple[float, float]],
        is_snapshot: bool = False,
        timestamp: float = 0.0,
    ) -> float:
        """
        WebSocket Reconciliation — processes L2 deltas from the historical feed.

        Separates concerns:
          1. Updates V_hist (q_real) with the authoritative feed data.
          2. Calls tracker.reconcile() for event-driven correction of V_cons.
          3. Computes and returns smoothed OFI based on effective (V_eff) state.

        Args:
            bid_updates: List of (price, qty) from feed. qty=0 means level gone.
            ask_updates: List of (price, qty) from feed. qty=0 means level gone.
            is_snapshot: If True, clears all state first (cycle rollover).
            timestamp:   Current simulation time in epoch seconds.

        Returns:
            Exponentially smoothed Order Flow Imbalance (OFI).
        """
        if timestamp > 0.0:
            self._t_now = timestamp
        t = self._t_now

        if is_snapshot:
            self.q_real_bids.clear()
            self.q_real_asks.clear()
            self._tracker_bids.reset()
            self._tracker_asks.reset()

        # ── Process Bid Updates ─────────────────────────────────────────────
        for price, qty in bid_updates:
            p = round(price, 6)
            v_old = self.q_real_bids.get(p, 0.0)

            if qty == 0.0:
                # Level removed from feed — auto-zero V_cons via reconcile
                self._tracker_bids.reconcile(p, v_old, 0.0, t)
                self.q_real_bids.pop(p, None)
            else:
                v_new = float(qty)
                self._tracker_bids.reconcile(p, v_old, v_new, t)
                self.q_real_bids[p] = v_new

        # ── Process Ask Updates ─────────────────────────────────────────────
        for price, qty in ask_updates:
            p = round(price, 6)
            v_old = self.q_real_asks.get(p, 0.0)

            if qty == 0.0:
                self._tracker_asks.reconcile(p, v_old, 0.0, t)
                self.q_real_asks.pop(p, None)
            else:
                v_new = float(qty)
                self._tracker_asks.reconcile(p, v_old, v_new, t)
                self.q_real_asks[p] = v_new

        # ── OFI Calculation (on effective book) ─────────────────────────────
        raw_ofi = self._calculate_ofi()
        self.smoothed_ofi = 0.1 * raw_ofi + 0.9 * self.smoothed_ofi
        return self.smoothed_ofi

    # ──────────────────────────────────────────────────────────────────────────
    # Bot Fill Recording (Taker Execution)
    # ──────────────────────────────────────────────────────────────────────────

    def paper_execute(
        self,
        price: float,
        qty_executed: float,
        is_bid: bool,
        fills: Optional[List[Tuple[float, float]]] = None,
        timestamp: float = 0.0,
    ) -> None:
        """
        Records bot fills into the consumption ledger (V_cons).

        PREFERRED USAGE — pass `fills` (list of (price, qty) pairs walked by
        the IOC loop in MockExecutionClient).  This gives exact per-level
        accounting and avoids any ambiguity in level ordering.

        FALLBACK — if `fills` is None (legacy callers), deducts `qty_executed`
        walking levels by priority exactly as get_sorted_{bids,asks}() would.

        This method NEVER touches q_real_{bids,asks}.

        Args:
            price:        Not used when `fills` is provided; kept for interface
                          compatibility with legacy callers.
            qty_executed: Total quantity executed (used only in fallback mode).
            is_bid:       True → consumed bid side (SELL_YES / BUY_NO);
                          False → consumed ask side (BUY_YES / SELL_NO).
            fills:        Exact (price, qty) pairs consumed during IOC walk.
            timestamp:    Execution timestamp (defaults to last known t_now).
        """
        t = timestamp if timestamp > 0.0 else self._t_now

        if fills is not None:
            # ── Exact per-level accounting (preferred) ───────────────────────
            tracker = self._tracker_bids if is_bid else self._tracker_asks
            for p, q in fills:
                p_key = round(p, 6)
                if q > 1e-12:
                    tracker.consume(p_key, q, t)
        else:
            # ── Fallback: sweep levels in priority order ──────────────────────
            # Replicates what execute_trade() does in MockExecutionClient so
            # the same levels get debited even when exact fills are not passed.
            remaining = qty_executed
            if is_bid:
                levels = self.get_sorted_bids()
                tracker = self._tracker_bids
            else:
                levels = self.get_sorted_asks()
                tracker = self._tracker_asks

            for p, q_eff in levels:
                if remaining <= 1e-12:
                    break
                p_key = round(p, 6)
                fill = min(remaining, q_eff)
                tracker.consume(p_key, fill, t)
                remaining -= fill

    # ──────────────────────────────────────────────────────────────────────────
    # Effective Book Views (V_eff = V_hist − V_cons)
    # ──────────────────────────────────────────────────────────────────────────

    def get_sorted_bids(self) -> List[Tuple[float, float]]:
        """
        Returns shadow bids (V_eff) sorted descending (highest bid first).
        Only levels with V_eff > 0 are included.
        """
        t = self._t_now
        result = []
        for p, v_hist in self.q_real_bids.items():
            v_eff = self._tracker_bids.get_v_eff(p, v_hist, t)
            if v_eff > 1e-9:
                result.append((p, v_eff))
        return sorted(result, key=lambda x: x[0], reverse=True)

    def get_sorted_asks(self) -> List[Tuple[float, float]]:
        """
        Returns shadow asks (V_eff) sorted ascending (lowest ask first).
        Only levels with V_eff > 0 are included.
        """
        t = self._t_now
        result = []
        for p, v_hist in self.q_real_asks.items():
            v_eff = self._tracker_asks.get_v_eff(p, v_hist, t)
            if v_eff > 1e-9:
                result.append((p, v_eff))
        return sorted(result, key=lambda x: x[0])

    def get_top_of_book(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Returns best (bid, ask) as (price, V_eff) tuples from the shadow book."""
        sorted_bids = self.get_sorted_bids()
        sorted_asks = self.get_sorted_asks()
        best_bid = sorted_bids[0] if sorted_bids else None
        best_ask = sorted_asks[0] if sorted_asks else None
        return best_bid, best_ask

    def get_market_top_of_book(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Returns best (bid, ask) from the REAL (V_hist) book.
        Never depleted by paper fills — used for p_mkt display and MTM so that
        simulated trades do not corrupt the displayed market-implied probability.
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

    # ──────────────────────────────────────────────────────────────────────────
    # OFI
    # ──────────────────────────────────────────────────────────────────────────

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
