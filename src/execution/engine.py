import logging
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.core.interfaces import IExecutionClient

logger = logging.getLogger("ExecutionEngine")

class ExecutionEngine:
    """
    Execution and Sizing Engine for Polymarket V2.
    Translates model probabilities into trade sizing using Fractional Kelly,
    sweeps L2 order books for optimal partial fills, and enforces multi-layered HFT risk filters.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.spot_history: List[Tuple[float, float]] = []  # (timestamp, price)

        # Staleness track memory
        self.last_bid_price: Optional[float] = None
        self.last_bid_qty: Optional[float] = None
        self.last_bid_updated_at: float = 0.0

        self.last_ask_price: Optional[float] = None
        self.last_ask_qty: Optional[float] = None
        self.last_ask_updated_at: float = 0.0

    def walk_order_book(
        self, 
        levels: List[Tuple[float, float]], 
        fair_price: float, 
        max_kelly_qty: float, 
        max_slippage_bps: float, 
        is_buy: bool
    ) -> Tuple[float, float, float]:
        """
        Walks the L2 order book levels and consumes them marginally.
        Stops when marginal EV becomes zero/negative, slippage limits are violated, 
        or the target Kelly quantity is filled.
        
        Returns:
            Tuple of (optimal_qty, vwap_price, final_slippage_bps)
        """
        if not levels or max_kelly_qty <= 0.0:
            return 0.0, 0.0, 0.0
            
        best_price = float(levels[0][0])
        if best_price <= 0.0 or best_price >= 1.0:
            return 0.0, 0.0, 0.0
            
        filled_qty = 0.0
        total_cost_or_revenue = 0.0
        
        for price, qty in levels:
            price = float(price)
            qty = float(qty)
            
            # 1. Compute marginal slippage and edge
            if is_buy:
                marginal_slippage_bps = ((price - best_price) / best_price) * 10000.0
                marginal_edge = fair_price - price
            else:
                marginal_slippage_bps = ((best_price - price) / best_price) * 10000.0
                marginal_edge = price - fair_price
                
            # 2. Stop condition: no marginal edge or slippage exceeds dynamic threshold
            if marginal_edge <= 0.0 or marginal_slippage_bps > max_slippage_bps:
                break
                
            # 3. Stop if we filled our target allocation
            remaining = max_kelly_qty - filled_qty
            if remaining <= 1e-9:
                break
                
            fill = min(qty, remaining)
            filled_qty += fill
            total_cost_or_revenue += fill * price
            
            if filled_qty >= max_kelly_qty - 1e-9:
                break
                
        if filled_qty <= 0.0:
            return 0.0, 0.0, 0.0
            
        vwap_price = total_cost_or_revenue / filled_qty
        
        if is_buy:
            final_slippage_bps = ((vwap_price - best_price) / best_price) * 10000.0
        else:
            final_slippage_bps = ((best_price - vwap_price) / best_price) * 10000.0
            
        return float(filled_qty), float(vwap_price), float(final_slippage_bps)

    def evaluate_and_trade(
        self, 
        p_yes: Optional[float], 
        context: MarketContext, 
        client: IExecutionClient
    ) -> Dict[str, Any]:
        """
        Runs portfolio-aware Kelly sizing, partial fill book walking, and HFT risk checks.
        """
        # 1. Check if model resolved probability cleanly (Zero-Assumptions REST check)
        if p_yes is None:
            return {
                "side": "HOLD", 
                "reason": "STRATEGY_UNRESOLVED_PROBABILITY", 
                "size": 0.0,
                "kelly_alloc": 0.0,
                "vwap": 0.0,
                "theoretical_edge_bps": 0.0
            }
            
        # 1.5. Block trading if strike price is unresolved or zero
        if context.strike_price is None or context.strike_price <= 0.0:
            return {
                "side": "HOLD",
                "reason": "WAITING_FOR_STRIKE_RESOLUTION",
                "size": 0.0,
                "kelly_alloc": 0.0,
                "vwap": 0.0,
                "theoretical_edge_bps": 0.0
            }

        # 1.6. Block trading if the order book has no executable levels on either side
        if not context.bids_l2 or not context.asks_l2:
            return {"side": "HOLD", "reason": "NO_BOOK_DATA", "size": 0.0}
            
        t_now = context.timestamp

        self.spot_history.append((t_now, context.spot_price))
        # Keep spot history pruned to 60s
        self.spot_history = [x for x in self.spot_history if x[0] >= t_now - 60.0]
        
        # Track L2 update stale ages
        best_bid, best_ask = context.bids_l2[0] if context.bids_l2 else (None, None), context.asks_l2[0] if context.asks_l2 else (None, None)
        
        if best_bid[0] is not None and (best_bid[0] != self.last_bid_price or best_bid[1] != self.last_bid_qty):
            self.last_bid_price = best_bid[0]
            self.last_bid_qty = best_bid[1]
            self.last_bid_updated_at = t_now
            
        if best_ask[0] is not None and (best_ask[0] != self.last_ask_price or best_ask[1] != self.last_ask_qty):
            self.last_ask_price = best_ask[0]
            self.last_ask_qty = best_ask[1]
            self.last_ask_updated_at = t_now

        # 4. Sizing Calculations via Fractional Kelly (Pre-computed for UI reporting on HOLD)
        gamma = self.config.arbitrage.KELLY_FRACTION
        W = client.cash_balance
        gas = self.config.arbitrage.GAS_FEE_USD
        min_order_usd = 50.0  # Polymarket CLOB minimum size
        
        # Best prices on shadow book
        p_bid_yes = best_bid[0] if best_bid[0] else 0.5
        p_ask_yes = best_ask[0] if best_ask[0] else 0.5
        p_ask_no = 1.0 - p_bid_yes
        
        # Retrieve current position counts
        qty_yes = client.get_position_size("YES")
        qty_no = client.get_position_size("NO")
        
        # Calculate current total portfolio wealth/equity
        wealth = W + (qty_yes * p_ask_yes) + (qty_no * p_ask_no)
        
        # Current weights as fraction of wealth
        w_current_yes = (qty_yes * p_ask_yes) / wealth if wealth > 0.0 else 0.0
        w_current_no = (qty_no * p_ask_no) / wealth if wealth > 0.0 else 0.0
        
        # Raw unconstrained Kelly targets (can be negative if overvalued)
        f_star_yes = gamma * (p_yes - p_ask_yes) / (1.0 - p_ask_yes) if p_ask_yes < 1.0 else 0.0
        
        p_no = 1.0 - p_yes
        f_star_no = gamma * (p_no - p_ask_no) / (1.0 - p_ask_no) if p_ask_no < 1.0 else 0.0
        
        # Apply 2026 Polymarket taker fee regularization buffer delta = taker_fee_multiplier * gamma
        taker_fee_multiplier = self.config.arbitrage.TAKER_FEE_MULTIPLIER
        delta_buffer = taker_fee_multiplier * gamma
        
        # YES regularized target weight
        if f_star_yes > w_current_yes + delta_buffer:
            target_w_yes = f_star_yes - delta_buffer
        elif f_star_yes < w_current_yes - delta_buffer:
            target_w_yes = f_star_yes + delta_buffer
        else:
            target_w_yes = w_current_yes
            
        # NO regularized target weight
        if f_star_no > w_current_no + delta_buffer:
            target_w_no = f_star_no - delta_buffer
        elif f_star_no < w_current_no - delta_buffer:
            target_w_no = f_star_no + delta_buffer
        else:
            target_w_no = w_current_no
            
        # Clip regularized target weights to standard safety bounds [0, 50%]
        target_w_yes = float(np.clip(target_w_yes, 0.0, 0.50))
        target_w_no = float(np.clip(target_w_no, 0.0, 0.50))
        
        # Determine active sizing metrics for UI reporting even on HOLD
        edge_yes = p_yes - p_ask_yes
        edge_no = p_no - p_ask_no
        
        if edge_yes > edge_no and edge_yes > 0.0:
            theoretical_edge_bps = edge_yes * 10000.0
            kelly_alloc = target_w_yes
            target_price = p_yes
        elif edge_no > edge_yes and edge_no > 0.0:
            theoretical_edge_bps = edge_no * 10000.0
            kelly_alloc = target_w_no
            target_price = p_no
        else:
            if edge_yes > edge_no:
                theoretical_edge_bps = max(0.0, edge_yes) * 10000.0
                kelly_alloc = target_w_yes
                target_price = p_yes
            else:
                theoretical_edge_bps = max(0.0, edge_no) * 10000.0
                kelly_alloc = target_w_no
                target_price = p_no

        # 2. Pin Risk (Oracle Jitter) Quantitative Protection
        pin_risk_window = self.config.risk.PIN_RISK_SECONDS
        if context.tau_seconds <= pin_risk_window and context.tau_seconds > 0.0:
            # Noise margin: either standard BPS or empirical standard deviation of spot
            noise_bps_usd = context.spot_price * (self.config.risk.ORACLE_NOISE_BPS / 10000.0)
            recent_spots = [p for t, p in self.spot_history if t >= t_now - 10.0]
            empirical_std = np.std(recent_spots) if len(recent_spots) > 1 else 0.0
            
            oracle_noise = max(noise_bps_usd, empirical_std)
            distance_to_strike = abs(context.spot_price - context.strike_price)
            
            if distance_to_strike < oracle_noise:
                logger.warning(
                    f"[PIN RISK] Trade blocked: |S - K| = {distance_to_strike:.4f} < "
                    f"noise = {oracle_noise:.4f} (tau: {context.tau_seconds:.1f}s)"
                )
                return {
                    "side": "HOLD",
                    "reason": "REJECT_PIN_RISK",
                    "size": 0.0,
                    "kelly_alloc": kelly_alloc,
                    "vwap": target_price,
                    "theoretical_edge_bps": theoretical_edge_bps
                }

        # 3. Market Desync / Staleness Check
        if self.last_ask_updated_at > 0.0 and self.last_bid_updated_at > 0.0:
            # Let quote age be the age of the side we are interacting with
            quote_age = t_now - min(self.last_ask_updated_at, self.last_bid_updated_at)
            
            # If quote is stale (older than 100ms) and spot has walked too far
            if quote_age > 0.100 and self.spot_history:
                t_lookup = t_now - quote_age
                closest_spot = min(self.spot_history, key=lambda x: abs(x[0] - t_lookup))[1]
                delta_S = abs(context.spot_price - closest_spot)
                
                # Check normal diffusion expected moves
                dt_years = quote_age / (365.25 * 24 * 3600.0)
                z_score = self.config.risk.DESYNC_Z_SCORE
                threshold_desync = max(z_score * context.spot_price * context.volatility * np.sqrt(dt_years), context.spot_price * 0.00005)
                
                if delta_S > threshold_desync:
                    logger.warning(f"[STALENESS] Quote Stale (Age: {quote_age:.2f}s, dS: {delta_S:.2f} > th: {threshold_desync:.2f})")
                    return {
                        "side": "HOLD",
                        "reason": "REJECT_DESYNC_STALENESS",
                        "size": 0.0,
                        "kelly_alloc": kelly_alloc,
                        "vwap": target_price,
                        "theoretical_edge_bps": theoretical_edge_bps
                    }


        # Calculate age of each side of the book for Probability of Fill (PoF) scaling
        age_ask = t_now - self.last_ask_updated_at if self.last_ask_updated_at > 0.0 else 0.0
        age_bid = t_now - self.last_bid_updated_at if self.last_bid_updated_at > 0.0 else 0.0

        # Convert Kelly weights to contract target holdings
        target_qty_yes = (target_w_yes * wealth) / p_ask_yes if p_ask_yes > 0.0 else 0.0
        target_qty_no = (target_w_no * wealth) / p_ask_no if p_ask_no > 0.0 else 0.0
        
        # Calculate dynamic transaction costs in bps
        costi_rete_bps = (gas / min_order_usd) * 10000.0
        min_margin = self.config.arbitrage.MIN_ACCEPTABLE_MARGIN_BPS
        abs_max_slippage = self.config.arbitrage.ABSOLUTE_MAX_SLIPPAGE_BPS
        
        # Check decisions:
        # A. SELL YES (Exiting excess YES positions)
        if qty_yes > target_qty_yes and qty_yes > 0.0:
            excess_yes = qty_yes - target_qty_yes
            
            # Selling YES means walking YES bids
            edge_bps = (p_bid_yes - p_yes) * 10000.0
            max_slippage_bps = min(edge_bps - costi_rete_bps - min_margin, abs_max_slippage)
            
            qty_exec, vwap, slippage_bps = self.walk_order_book(
                levels=context.bids_l2,
                fair_price=p_yes,
                max_kelly_qty=excess_yes,
                max_slippage_bps=max_slippage_bps,
                is_buy=False
            )
            
            if qty_exec > 0.0 and (qty_exec * vwap) >= min_order_usd:
                ev = vwap - p_yes
                per_unit_fee = self.config.arbitrage.TAKER_FEE_MULTIPLIER * vwap * (1.0 - vwap)
                ev_net = ev - per_unit_fee
                pof = self._calculate_pof(context.tau_seconds, age_bid)
                ev_adjusted = ev_net * pof
                
                if ev_adjusted >= self.config.arbitrage.MIN_EXPECTED_VALUE:
                    limit_price = max(0.0, p_bid_yes * (1.0 - max_slippage_bps / 10000.0))
                    return {
                        "side": "SELL_YES",
                        "size": qty_exec,
                        "limit_price": limit_price,
                        "vwap": vwap,
                        "ev": ev,
                        "expected_slippage_bps": slippage_bps,
                        "kelly_alloc": kelly_alloc,
                        "theoretical_edge_bps": theoretical_edge_bps
                    }

        # B. SELL NO (Exiting excess NO positions)
        if qty_no > target_qty_no and qty_no > 0.0:
            excess_no = qty_no - target_qty_no
            
            # Selling NO corresponds to matching asks YES in reverse pricing: (1.0 - p, q) sorted descending
            derived_bids_no = sorted([(1.0 - ask_p, ask_q) for ask_p, ask_q in context.asks_l2], key=lambda x: x[0], reverse=True)
            
            edge_bps = ((1.0 - p_ask_yes) - p_no) * 10000.0
            max_slippage_bps = min(edge_bps - costi_rete_bps - min_margin, abs_max_slippage)
            
            qty_exec, vwap, slippage_bps = self.walk_order_book(
                levels=derived_bids_no,
                fair_price=p_no,
                max_kelly_qty=excess_no,
                max_slippage_bps=max_slippage_bps,
                is_buy=False
            )
            
            if qty_exec > 0.0 and (qty_exec * vwap) >= min_order_usd:
                ev = vwap - p_no
                per_unit_fee = self.config.arbitrage.TAKER_FEE_MULTIPLIER * vwap * (1.0 - vwap)
                ev_net = ev - per_unit_fee
                pof = self._calculate_pof(context.tau_seconds, age_ask)
                ev_adjusted = ev_net * pof
                
                if ev_adjusted >= self.config.arbitrage.MIN_EXPECTED_VALUE:
                    best_bid_no = 1.0 - p_ask_yes
                    limit_price_no = max(0.0, best_bid_no * (1.0 - max_slippage_bps / 10000.0))
                    limit_price = 1.0 - limit_price_no # In YES terms
                    return {
                        "side": "SELL_NO",
                        "size": qty_exec,
                        "limit_price": limit_price,
                        "vwap": vwap,
                        "ev": ev,
                        "expected_slippage_bps": slippage_bps,
                        "kelly_alloc": kelly_alloc,
                        "theoretical_edge_bps": theoretical_edge_bps
                    }

        # C. BUY YES INCREMENTAL
        if target_qty_yes > qty_yes and qty_no <= 0.001:
            dq_yes = target_qty_yes - qty_yes
            
            # Sizing limit caps
            q_max_by_cap = max(0.0, (W - gas) / p_ask_yes) if p_ask_yes > 0.0 else 0.0
            q_max_by_risk = self.config.arbitrage.MAX_POSITION_SIZE_USD / p_ask_yes if p_ask_yes > 0.0 else 0.0
            dq_yes = min(dq_yes, q_max_by_cap, q_max_by_risk)
            
            # Sweeping order book to calculate fill price & actual slippage
            edge_bps = (p_yes - p_ask_yes) * 10000.0
            max_slippage_bps = min(edge_bps - costi_rete_bps - min_margin, abs_max_slippage)
            
            qty_exec, vwap, slippage_bps = self.walk_order_book(
                levels=context.asks_l2,
                fair_price=p_yes,
                max_kelly_qty=dq_yes,
                max_slippage_bps=max_slippage_bps,
                is_buy=True
            )
            
            if qty_exec > 0.0 and (qty_exec * vwap) >= min_order_usd:
                ev = p_yes - vwap
                # Subtract the dynamic taker fee per unit from the EV gate.
                per_unit_fee = self.config.arbitrage.TAKER_FEE_MULTIPLIER * vwap * (1.0 - vwap)
                ev_net = ev - per_unit_fee
                # Apply fill probability EV scaling (PoF)
                pof = self._calculate_pof(context.tau_seconds, age_ask)
                ev_adjusted = ev_net * pof
                
                if ev_adjusted >= self.config.arbitrage.MIN_EXPECTED_VALUE:
                    limit_price = min(1.0, p_ask_yes * (1.0 + max_slippage_bps / 10000.0))
                    return {
                        "side": "BUY_YES",
                        "size": qty_exec,
                        "limit_price": limit_price,
                        "vwap": vwap,
                        "ev": ev,
                        "expected_slippage_bps": slippage_bps,
                        "kelly_alloc": kelly_alloc,
                        "theoretical_edge_bps": theoretical_edge_bps
                    }

        # D. BUY NO INCREMENTAL
        if target_qty_no > qty_no and qty_yes <= 0.001:
            dq_no = target_qty_no - qty_no
            
            q_max_by_cap = max(0.0, (W - gas) / p_ask_no) if p_ask_no > 0.0 else 0.0
            q_max_by_risk = self.config.arbitrage.MAX_POSITION_SIZE_USD / p_ask_no if p_ask_no > 0.0 else 0.0
            dq_no = min(dq_no, q_max_by_cap, q_max_by_risk)
            
            edge_bps = (p_no - p_ask_no) * 10000.0
            max_slippage_bps = min(edge_bps - costi_rete_bps - min_margin, abs_max_slippage)
            
            # Buying NO corresponds to walking YES bids in reverse pricing: (1.0 - p, q)
            derived_asks_no = sorted([(1.0 - bid_p, bid_q) for bid_p, bid_q in context.bids_l2], key=lambda x: x[0])
            
            qty_exec, vwap, slippage_bps = self.walk_order_book(
                levels=derived_asks_no,
                fair_price=p_no,
                max_kelly_qty=dq_no,
                max_slippage_bps=max_slippage_bps,
                is_buy=True
            )
            
            if qty_exec > 0.0 and (qty_exec * vwap) >= min_order_usd:
                ev = p_no - vwap
                per_unit_fee = self.config.arbitrage.TAKER_FEE_MULTIPLIER * vwap * (1.0 - vwap)
                ev_net = ev - per_unit_fee
                pof = self._calculate_pof(context.tau_seconds, age_bid)
                ev_adjusted = ev_net * pof
                
                if ev_adjusted >= self.config.arbitrage.MIN_EXPECTED_VALUE:
                    limit_price_no = min(1.0, p_ask_no * (1.0 + max_slippage_bps / 10000.0))
                    limit_price = 1.0 - limit_price_no # In YES terms
                    return {
                        "side": "BUY_NO",
                        "size": qty_exec,
                        "limit_price": limit_price,
                        "vwap": vwap,
                        "ev": ev,
                        "expected_slippage_bps": slippage_bps,
                        "kelly_alloc": kelly_alloc,
                        "theoretical_edge_bps": theoretical_edge_bps
                    }

        return {
            "side": "HOLD",
            "reason": "NO_EV_OR_SIZE_OPPORTUNITY",
            "size": 0.0,
            "kelly_alloc": kelly_alloc,
            "vwap": target_price,
            "theoretical_edge_bps": theoretical_edge_bps
        }

    def _calculate_pof(self, tau_seconds: float, quote_age: float = 0.0) -> float:
        """Calculates fill probability based on remaining lifetime and quote age."""
        tau_lim = self.config.risk.POF_LATENCY_TAU
        k_decay = self.config.risk.POF_DECAY_K
        pof_tau = 1.0 / (1.0 + np.exp(-k_decay * (tau_seconds - tau_lim)))
        
        # Penalize staleness: if a quote is old, it's less likely to be filled.
        # Uses an exponential decay (e.g. e^(-1.0 * age))
        pof_age = np.exp(-quote_age * 1.0)
        
        return float(pof_tau * pof_age)
