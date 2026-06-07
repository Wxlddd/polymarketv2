import logging
from typing import Dict, Any, Tuple, Optional, List
from src.core.market_context import MarketContext
from src.core.base_strategy import BaseStrategy
from src.core.interfaces import IExecutionClient
from config.settings import SystemConfig

logger = logging.getLogger("MakerExecution")

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


class MidPriceVolCalibrator:
    """
    Time-Sampled Exponentially Weighted Moving Average (EWMA) Volatility Calibrator.
    Samples the contract's mid-price at fixed time intervals (e.g. 10s) and computes
    log-return variance. Bypasses tick-by-tick bid-ask bounce noise.
    """
    __slots__ = ('sampling_interval', 'alpha', 'last_sample_time', 'last_sample_price', 'variance')
    
    def __init__(self, sampling_interval: float = 10.0, alpha: float = 0.05, initial_vol: float = 0.25):
        self.sampling_interval = sampling_interval
        self.alpha = alpha
        self.last_sample_time = 0.0
        self.last_sample_price = None
        self.variance = initial_vol * initial_vol
        
    def update(self, current_price: float, t_now: float) -> float:
        """
        Conditionally updates the variance using EWMA log-returns if interval elapsed.
        Returns the current variance (sigma^2).
        """
        if current_price is None or current_price <= 0.0:
            return self.variance
            
        if self.last_sample_price is None:
            self.last_sample_price = current_price
            self.last_sample_time = t_now
            return self.variance
            
        dt = t_now - self.last_sample_time
        if dt >= self.sampling_interval:
            import math
            log_ret = math.log(current_price / self.last_sample_price)
            self.variance = self.alpha * (log_ret * log_ret) + (1.0 - self.alpha) * self.variance
            self.last_sample_price = current_price
            self.last_sample_time = t_now
            
        return self.variance


class InventoryManager:
    r"""
    Tracks net inventory risk and calculates the skewed Avellaneda-Stoikov Reservation Price.
    
    $q_{norm} = \text{clamp}(q / Q_{max}, -1.0, 1.0)$
    $P_{res} = \hat{P} - \gamma \cdot q_{norm} \cdot \sigma^2$
    """
    __slots__ = ('gamma', 'fixed_horizon_sec', 'max_inventory')

    def __init__(self, gamma: float = 0.1, fixed_horizon_sec: float = 300.0, max_inventory: float = 5000.0):
        self.gamma = gamma
        self.fixed_horizon_sec = fixed_horizon_sec
        self.max_inventory = max_inventory

    def get_inventory(self, yes_shares: float, no_shares: float) -> float:
        """Returns the net inventory risk q."""
        return yes_shares - no_shares

    def calculate_reservation_price(
        self, p_hat: float, q: float, sigma_sq: float, tau_seconds: float = 0.0
    ) -> float:
        """
        Calculates the Reservation Price P_res skewed by inventory.
        Implements asymptotic market making by completely omitting tau.
        
        Args:
            p_hat: Model's internal fair probability of YES [0.0, 1.0].
            q: Net inventory count (YES - NO).
            sigma_sq: Mid-price EWMA variance.
            tau_seconds: Deprecated / unused for asymptotic MM.
            
        Returns:
            Reservation price, clipped to [0.01, 0.99] to prevent illegal probability values.
        """
        # Strictly normalize inventory skew with Max Inventory limits and division guard
        if self.max_inventory > 0.0:
            q_norm = max(-1.0, min(1.0, q / self.max_inventory))
        else:
            q_norm = 0.0

        # Asymptotic MM Reservation Price formula (tau completely removed)
        skew = self.gamma * q_norm * sigma_sq
        p_res = p_hat - skew
        
        # Clip to valid probability bounds
        if p_res < 0.01:
            return 0.01
        elif p_res > 0.99:
            return 0.99
        return p_res



class ExecutionRouter:
    r"""
    Handles quoting calculations, evaluates execution regimes, and issues cancel/replace/new order routing commands.
    
    Regime A (Maker Mode): Maintain POST_ONLY limit orders at P_res +/- delta.
    Regime B (Taker Mode): Aggressively cross spread when edge exceeds epsilon, using worst-case Kelly sizing.
    Regime C (Unwind): Cancel quotes on expanding side and quote exclusively to reduce inventory.
    """
    __slots__ = (
        'gamma', 'min_fee_buffer', 'toxicity_buffer', 'maker_size', 'max_inventory',
        'unwind_threshold', 'taker_edge_epsilon', 'tick_size', 'requote_threshold',
        'kelly_fraction', 'fixed_horizon_sec', 'gas_fee_usd', 'taker_fee_multiplier',
        # Active order tracking (single bid/ask for YES contract)
        'active_bid_id', 'active_bid_price', 'active_bid_qty',
        'active_ask_id', 'active_ask_price', 'active_ask_qty',
        # Order counter for mock ID generation
        '_order_counter', 'min_order_usd', 'locked', 'panic_concession'
    )

    def __init__(
        self,
        gamma: float = 0.1,
        min_fee_buffer: float = 0.005,
        toxicity_buffer: float = 0.005,
        maker_size: float = 100.0,
        max_inventory: float = 5000.0,
        unwind_threshold: float = 0.01,
        taker_edge_epsilon: float = 0.015,
        tick_size: float = 0.01,
        requote_threshold: float = 0.01,
        kelly_fraction: float = 0.15,
        fixed_horizon_sec: float = 300.0,
        gas_fee_usd: float = 0.03,
        taker_fee_multiplier: float = 0.072,
        min_order_usd: float = 1.0,
        panic_concession: float = 0.15
    ):
        self.gamma = gamma
        self.min_fee_buffer = min_fee_buffer
        self.toxicity_buffer = toxicity_buffer
        self.maker_size = maker_size
        self.max_inventory = max_inventory
        self.unwind_threshold = unwind_threshold
        self.taker_edge_epsilon = taker_edge_epsilon
        self.tick_size = tick_size
        self.requote_threshold = requote_threshold
        self.kelly_fraction = kelly_fraction
        self.fixed_horizon_sec = fixed_horizon_sec
        self.gas_fee_usd = gas_fee_usd
        self.taker_fee_multiplier = taker_fee_multiplier
        self.min_order_usd = min_order_usd
        self.locked = False
        self.panic_concession = panic_concession

        # State memory
        self.active_bid_id: str = ""
        self.active_bid_price: float = 0.0
        self.active_bid_qty: float = 0.0

        self.active_ask_id: str = ""
        self.active_ask_price: float = 0.0
        self.active_ask_qty: float = 0.0
        
        self._order_counter: int = 0

    def reset_active_orders(self) -> None:
        """Resets active tracking memory (e.g. on market rollover)."""
        self.active_bid_id = ""
        self.active_bid_price = 0.0
        self.active_bid_qty = 0.0
        self.active_ask_id = ""
        self.active_ask_price = 0.0
        self.active_ask_qty = 0.0
        self.locked = False

    def calculate_spread(self, sigma_sq: float, tau_seconds: float) -> float:
        r"""
        Calculates optimal half-spread (delta):
        $\delta = \text{min\_fee\_buffer} + 0.5 \cdot \gamma \cdot \sigma^2 \cdot \tau + \text{toxicity\_buffer}$
        """
        tau_years = tau_seconds / (365.25 * 24.0 * 3600.0) if tau_seconds > 0.0 else (self.fixed_horizon_sec / (365.25 * 24.0 * 3600.0))
        return self.min_fee_buffer + 0.5 * self.gamma * sigma_sq * tau_years + self.toxicity_buffer

    def evaluate_regimes(
        self,
        p_hat: float,
        sigma_sq: float,
        context: MarketContext,
        yes_shares: float,
        no_shares: float,
        cash_balance: float,
        divergence_scale: float = 1.0
    ) -> List[OrderInstruction]:
        """
        Evaluates execution regimes tick-by-tick and yields a list of OrderInstructions.
        
        Args:
            p_hat: Internal fair model probability.
            sigma_sq: Realized variance.
            context: MarketContext carrying L2 order book (bids_l2, asks_l2) and tau.
            yes_shares: Current YES shares held.
            no_shares: Current NO shares held.
            cash_balance: Available USD cash.
            divergence_scale: Sizing scale factor [0, 1] from the divergence velocity filter.
            
        Returns:
            List of OrderInstruction objects.
        """
        instructions: List[OrderInstruction] = []
        
        # Guard: Ensure we have order book data
        if not context.bids_l2 or not context.asks_l2:
            return instructions

        # Toxic Flow Guard: if divergence velocity is high, cancel quotes and do not trade
        if divergence_scale <= 0.01:
            self._cancel_bid(instructions, "SAFE")
            self._cancel_ask(instructions, "SAFE")
            return instructions

        # Lock check
        if self.locked:
            self._cancel_bid(instructions, "LOCKED")
            self._cancel_ask(instructions, "LOCKED")
            return instructions
            
        best_bid = context.bids_l2[0][0]
        best_ask = context.asks_l2[0][0]
        p_mid = 0.5 * (best_bid + best_ask)
        current_spread = best_ask - best_bid
        
        # Get net position q
        q = yes_shares - no_shares
        tau_sec = context.tau_seconds

        # Phase 2: Hard Liquidation Sweep (Panic Sweep)
        is_panic = (tau_sec <= 15.0) or (tau_sec <= 45.0 and current_spread > 0.10)
        if is_panic:
            self._cancel_bid(instructions, "PANIC")
            self._cancel_ask(instructions, "PANIC")
            self.locked = True
            
            if q > 0:
                # Aggressively sell all YES contracts
                limit_p = max(0.01, p_hat - self.panic_concession)
                instructions.append(OrderInstruction("NEW", "SELL_YES", limit_p, yes_shares, "panic_sell_yes", regime="PANIC"))
            elif q < 0:
                # Aggressively buy YES contracts to cover NO position
                limit_p = min(0.99, p_hat + self.panic_concession)
                instructions.append(OrderInstruction("NEW", "BUY_YES", limit_p, no_shares, "panic_buy_yes", regime="PANIC"))
            return instructions

        # Phase 1: Soft Unwind (Reduce-Only Regime)
        is_reduce_only = (tau_sec <= 45.0)
        if is_reduce_only:
            # 1. reservation price calculation
            # Skew target reservation price by inventory
            # Normalize q
            q_norm = 0.0
            if self.max_inventory > 0.0:
                q_norm = max(-1.0, min(1.0, q / self.max_inventory))
                
            # P_res calculation using the pure HFT instant risk formula recommended by the user
            p_res = p_hat - self.gamma * q_norm * sigma_sq
            if p_res < 0.01:
                p_res = 0.01
            elif p_res > 0.99:
                p_res = 0.99
            
            # 2. Calculate optimal spread and target quotes
            delta = self.calculate_spread(sigma_sq, tau_sec)
            inv_tick = 1.0 / self.tick_size
            p_bid_target = round((p_res - delta) * inv_tick) / inv_tick
            p_ask_target = round((p_res + delta) * inv_tick) / inv_tick
            
            if p_bid_target < 0.01:
                p_bid_target = 0.01
            if p_ask_target > 0.99:
                p_ask_target = 0.99
            if p_bid_target >= p_ask_target:
                p_ask_target = p_bid_target + self.tick_size
                
            if q > 0:
                # YES position -> Reduce-only means we only SELL YES. Cancel Bid.
                self._cancel_bid(instructions, "REDUCE")
                p_ask_post = max(p_ask_target, best_bid + self.tick_size)
                p_ask_post = max(0.01, min(0.99, p_ask_post))
                self._route_maker_ask(p_ask_post, instructions, "REDUCE", yes_shares, cash_balance, 1.0)
            elif q < 0:
                # NO position -> Reduce-only means we only BUY YES to cover. Cancel Ask.
                self._cancel_ask(instructions, "REDUCE")
                p_bid_post = min(p_bid_target, best_ask - self.tick_size)
                p_bid_post = max(0.01, min(0.99, p_bid_post))
                self._route_maker_bid(p_bid_post, instructions, "REDUCE", yes_shares, no_shares, cash_balance, 1.0)
            else:
                # q == 0 -> We are perfectly flat! Cancel quotes on both sides to prevent any inventory.
                self._cancel_bid(instructions, "REDUCE")
                self._cancel_ask(instructions, "REDUCE")
            return instructions
        
        # 1. Calculate net inventory and reservation price
        q = yes_shares - no_shares
        q_norm = q / self.max_inventory if self.max_inventory > 0.0 else q
        tau_sec = context.tau_seconds
        
        # P_res calculation using the pure HFT instant risk formula recommended by the user
        p_res = p_hat - self.gamma * q_norm * sigma_sq
        if p_res < 0.01:
            p_res = 0.01
        elif p_res > 0.99:
            p_res = 0.99
            
        # 2. Calculate optimal half spread and target prices
        delta = self.calculate_spread(sigma_sq, tau_sec)
        
        # Round target quotes to tick size
        inv_tick = 1.0 / self.tick_size
        p_bid_target = round((p_res - delta) * inv_tick) / inv_tick
        p_ask_target = round((p_res + delta) * inv_tick) / inv_tick
        
        # Keep quotes within bounds
        if p_bid_target < 0.01:
            p_bid_target = 0.01
        if p_ask_target > 0.99:
            p_ask_target = 0.99
        if p_bid_target >= p_ask_target:
            p_ask_target = p_bid_target + self.tick_size
            
        # Portfolio wealth for Kelly sizing
        wealth = cash_balance + yes_shares * p_mid + no_shares * (1.0 - p_mid)
        
        # REGIME DETERMINATION
        abs_q = abs(q)
        realigned = abs(p_mid - p_hat) < self.unwind_threshold
        should_unwind = (abs_q > self.max_inventory) or (realigned and abs_q > 0.0)
        
        if should_unwind:
            # REGIME C: Unwind / Take Profit
            regime = "C"
            if q > 0:
                # We have YES inventory. Reducing side is Ask (selling YES). Cancel Bid.
                self._cancel_bid(instructions, regime)
                p_ask_post = max(p_ask_target, best_bid + self.tick_size)
                p_ask_post = max(0.01, min(0.99, p_ask_post))
                self._route_maker_ask(p_ask_post, instructions, regime, yes_shares, cash_balance, 1.0)
            elif q < 0:
                # We have NO inventory. Reducing side is Bid (buying YES). Cancel Ask.
                self._cancel_ask(instructions, regime)
                p_bid_post = min(p_bid_target, best_ask - self.tick_size)
                p_bid_post = max(0.01, min(0.99, p_bid_post))
                self._route_maker_bid(p_bid_post, instructions, regime, yes_shares, no_shares, cash_balance, 1.0)
                
        else:
            # Check for REGIME B: Taker crossing conditions
            taker_buy_yes = p_bid_target > best_ask + self.taker_edge_epsilon
            taker_buy_no = p_ask_target < best_bid - self.taker_edge_epsilon
            
            if taker_buy_yes:
                # Massive edge buying YES shares
                regime = "B"
                edge = p_hat - best_ask
                denom = 1.0 - best_ask
                if edge > 0.0 and denom > 0.0:
                    f_star = self.kelly_fraction * (edge / denom)
                    f_star = min(0.50, max(0.0, f_star)) # Cap Kelly at 50%
                    
                    target_qty = (f_star * wealth) / best_ask
                    
                    # Clip target quantity by available cash balance (taking fee into account)
                    cost_per_share = best_ask * (1.0 + self.taker_fee_multiplier * (1.0 - best_ask))
                    max_qty_by_cash = max(0.0, (cash_balance - self.gas_fee_usd) / cost_per_share)
                    
                    max_add_by_inventory = max(0.0, self.max_inventory - q)
                    if max_qty_by_cash < self.maker_size:
                        target_qty = 0.0
                    else:
                        target_qty = max(self.maker_size, min(target_qty, max_qty_by_cash))
                        target_qty = min(target_qty, max_add_by_inventory)
                    
                    # Cancel all maker quotes to prevent fills during taker executions
                    self._cancel_bid(instructions, regime)
                    self._cancel_ask(instructions, regime)
                    
                    # Route taker execution instruction (min 5 USD size check)
                    if target_qty > 0.0 and (target_qty * best_ask) >= 5.0:
                        instructions.append(OrderInstruction("NEW", "BUY_YES", best_ask, target_qty, "", regime))
                    
            elif taker_buy_no:
                # Massive edge selling YES / buying NO
                regime = "B"
                edge = best_bid - p_hat
                denom = best_bid
                if edge > 0.0 and denom > 0.0:
                    f_star = self.kelly_fraction * (edge / denom)
                    f_star = min(0.50, max(0.0, f_star))
                    
                    target_qty = (f_star * wealth) / (1.0 - best_bid)
                    
                    # Clip target quantity by available cash balance (if buying NO)
                    if yes_shares < target_qty:
                        rem_qty = target_qty - yes_shares
                        max_rem_by_inventory = max(0.0, self.max_inventory)
                        rem_qty = min(rem_qty, max_rem_by_inventory)
                        
                        no_price = 1.0 - best_bid
                        cost_per_share_no = no_price * (1.0 + self.taker_fee_multiplier * best_bid)
                        max_rem_by_cash = max(0.0, (cash_balance - self.gas_fee_usd) / cost_per_share_no)
                        
                        if max_rem_by_cash < (self.maker_size - yes_shares):
                            target_qty = yes_shares
                        else:
                            target_qty = yes_shares + min(rem_qty, max_rem_by_cash)
                            
                    if yes_shares == 0.0:
                        max_add_by_inventory = max(0.0, self.max_inventory + q)
                        target_qty = min(target_qty, max_add_by_inventory)
                        
                    if target_qty < self.maker_size and yes_shares == 0.0:
                        target_qty = 0.0
                    else:
                        target_qty = max(self.maker_size, target_qty)
                    
                    # Cancel all maker quotes
                    self._cancel_bid(instructions, regime)
                    self._cancel_ask(instructions, regime)
                    
                    # If we hold YES, we should sell YES at best_bid.
                    # Otherwise, buy NO at (1.0 - best_bid).
                    if yes_shares > 0.0:
                        sell_qty = min(yes_shares, target_qty)
                        if sell_qty > 0.0 and (sell_qty * best_bid) >= 5.0:
                            instructions.append(OrderInstruction("NEW", "SELL_YES", best_bid, sell_qty, "", regime))
                    else:
                        if target_qty > 0.0 and (target_qty * (1.0 - best_bid)) >= 5.0:
                            instructions.append(OrderInstruction("NEW", "BUY_NO", 1.0 - best_bid, target_qty, "", regime))
                        
            else:
                # REGIME A: Maker Mode
                regime = "A"
                # Clip quotes to be strictly post-only (at least 1 tick inside the spread)
                p_bid_post = min(p_bid_target, best_ask - self.tick_size)
                p_ask_post = max(p_ask_target, best_bid + self.tick_size)
                
                # Keep quotes within valid bounds [0.01, 0.99]
                p_bid_post = max(0.01, min(0.99, p_bid_post))
                p_ask_post = max(0.01, min(0.99, p_ask_post))
                if p_bid_post >= p_ask_post:
                    p_ask_post = p_bid_post + self.tick_size
                    
                self._route_maker_bid(p_bid_post, instructions, regime, yes_shares, no_shares, cash_balance, divergence_scale)
                self._route_maker_ask(p_ask_post, instructions, regime, yes_shares, cash_balance, divergence_scale)
                
        return instructions

    # Private Helpers for clean routing logic

    def _cancel_bid(self, instructions: List[OrderInstruction], regime: str) -> None:
        if self.active_bid_id:
            instructions.append(OrderInstruction("CANCEL", "BUY_YES", 0.0, 0.0, self.active_bid_id, regime))
            self.active_bid_id = ""
            self.active_bid_price = 0.0
            self.active_bid_qty = 0.0

    def _cancel_ask(self, instructions: List[OrderInstruction], regime: str) -> None:
        if self.active_ask_id:
            instructions.append(OrderInstruction("CANCEL", "SELL_YES", 0.0, 0.0, self.active_ask_id, regime))
            self.active_ask_id = ""
            self.active_ask_price = 0.0
            self.active_ask_qty = 0.0

    def _route_maker_bid(
        self,
        p_bid_target: float,
        instructions: List[OrderInstruction],
        regime: str,
        yes_shares: float,
        no_shares: float,
        cash_balance: float,
        divergence_scale: float = 1.0
    ) -> None:
        # Enforce maximum total inventory capacity constraint (max_inventory)
        q = yes_shares - no_shares
        target_qty = max(0.0, self.max_inventory - q)
        
        # Scale target quantity by divergence scale for toxic flow protection
        scaled_target = target_qty * divergence_scale
        
        # Enforce minimum trade size of min_order_usd (qty * price >= min_order_usd)
        min_qty = self.min_order_usd / p_bid_target if p_bid_target > 0.0 else 0.0
        if scaled_target < min_qty:
            self._cancel_bid(instructions, regime)
            return
            
        qty = scaled_target
        
        # Check cash constraints
        cost = qty * p_bid_target
        if cost + self.gas_fee_usd > cash_balance:
            max_qty = max(0.0, (cash_balance - self.gas_fee_usd) / p_bid_target)
            if max_qty * p_bid_target < self.min_order_usd:
                self._cancel_bid(instructions, regime)
                return
            qty = max_qty
            
        if not self.active_bid_id:
            self._order_counter += 1
            mock_id = f"mock_bid_{self._order_counter}"
            instructions.append(OrderInstruction("NEW", "BUY_YES", p_bid_target, qty, mock_id, regime))
            self.active_bid_id = mock_id
            self.active_bid_price = p_bid_target
            self.active_bid_qty = qty
        else:
            if abs(self.active_bid_price - p_bid_target) >= self.requote_threshold or abs(self.active_bid_qty - qty) > 1e-5:
                self._order_counter += 1
                new_id = f"mock_bid_{self._order_counter}"
                instructions.append(OrderInstruction("REPLACE", "BUY_YES", p_bid_target, qty, self.active_bid_id, regime))
                self.active_bid_id = new_id
                self.active_bid_price = p_bid_target
                self.active_bid_qty = qty

    def _route_maker_ask(
        self,
        p_ask_target: float,
        instructions: List[OrderInstruction],
        regime: str,
        yes_shares: float,
        cash_balance: float,
        divergence_scale: float = 1.0
    ) -> None:
        # We can ONLY sell YES tokens we actually hold
        if yes_shares <= 0.0:
            self._cancel_ask(instructions, regime)
            return
            
        # Scale quoting ask size by the divergence scale
        target_qty = yes_shares * divergence_scale
        
        # Enforce minimum trade size
        min_qty = self.min_order_usd / p_ask_target if p_ask_target > 0.0 else 0.0
        if target_qty < min_qty:
            self._cancel_ask(instructions, regime)
            return
            
        qty = target_qty
        
        if not self.active_ask_id:
            self._order_counter += 1
            mock_id = f"mock_ask_{self._order_counter}"
            instructions.append(OrderInstruction("NEW", "SELL_YES", p_ask_target, qty, mock_id, regime))
            self.active_ask_id = mock_id
            self.active_ask_price = p_ask_target
            self.active_ask_qty = qty
        else:
            if abs(self.active_ask_price - p_ask_target) >= self.requote_threshold or abs(self.active_ask_qty - qty) > 1e-5:
                self._order_counter += 1
                new_id = f"mock_ask_{self._order_counter}"
                instructions.append(OrderInstruction("REPLACE", "SELL_YES", p_ask_target, qty, self.active_ask_id, regime))
                self.active_ask_id = new_id
                self.active_ask_price = p_ask_target
                self.active_ask_qty = qty


class MakerExecutionEngine:
    """
    High-Frequency Trading Execution Engine wrapping InventoryManager and ExecutionRouter.
    Interfaces with a generic BaseStrategy to extract probability predictions (P_hat)
    and uses the market context to fetch continuous volatility/variance.
    """
    __slots__ = ('strategy', 'inventory_manager', 'execution_router', 'client', 'mid_price_calibrator')

    def __init__(
        self,
        strategy: BaseStrategy,
        client: IExecutionClient,
        config: SystemConfig
    ):
        self.strategy = strategy
        self.client = client
        
        self.inventory_manager = InventoryManager(
            gamma=config.maker.RISK_AVERSION,
            fixed_horizon_sec=config.maker.FIXED_HORIZON_SEC,
            max_inventory=config.maker.MAX_INVENTORY
        )
        
        self.execution_router = ExecutionRouter(
            gamma=config.maker.RISK_AVERSION,
            min_fee_buffer=config.maker.MIN_FEE_BUFFER,
            toxicity_buffer=config.maker.TOXICITY_BUFFER,
            maker_size=config.maker.MAKER_SIZE,
            max_inventory=config.maker.MAX_INVENTORY,
            unwind_threshold=config.maker.UNWIND_THRESHOLD,
            taker_edge_epsilon=config.maker.TAKER_EDGE_EPSILON,
            tick_size=config.maker.TICK_SIZE,
            requote_threshold=config.maker.REQUOTE_THRESHOLD,
            kelly_fraction=config.arbitrage.KELLY_FRACTION,
            fixed_horizon_sec=config.maker.FIXED_HORIZON_SEC,
            gas_fee_usd=config.arbitrage.GAS_FEE_USD,
            taker_fee_multiplier=config.arbitrage.TAKER_FEE_MULTIPLIER,
            min_order_usd=config.arbitrage.MIN_ORDER_USD,
            panic_concession=config.arbitrage.PANIC_CONCESSION
        )
        # Time-sampled EWMA mid-price variance calibrator (10s sampling, alpha=0.05)
        self.mid_price_calibrator = MidPriceVolCalibrator(sampling_interval=10.0, alpha=0.05)

    def reset(self) -> None:
        """Resets the execution router's active orders state (e.g. on market rollover)."""
        self.execution_router.reset_active_orders()

    def evaluate_and_route(self, context: MarketContext) -> List[OrderInstruction]:
        """
        Receives raw context ticks, requests strategy predictions, coordinates 
        inventory skews, and returns the low-overhead list of quoting operations.
        """
        # 1. Strategy pricing interface
        p_hat = self.strategy.get_probability(context)
        if p_hat is None:
            # If strategy fails to resolve price, immediately cancel active quotes to remain flat and safe
            instructions: List[OrderInstruction] = []
            self.execution_router._cancel_bid(instructions, "SAFE")
            self.execution_router._cancel_ask(instructions, "SAFE")
            return instructions

        # 2. Compute contract mid-price and update time-sampled EWMA variance
        mid_price = 0.5
        if context.bids_l2 and context.asks_l2:
            mid_price = 0.5 * (context.bids_l2[0][0] + context.asks_l2[0][0])
        elif context.bids_l2:
            mid_price = context.bids_l2[0][0]
        elif context.asks_l2:
            mid_price = context.asks_l2[0][0]
            
        sigma_sq = self.mid_price_calibrator.update(mid_price, context.timestamp)

        # 3. Position query from exchange client
        yes_shares = self.client.get_position_size("YES")
        no_shares = self.client.get_position_size("NO")
        cash = self.client.cash_balance

        # 4. Get divergence scale from the shared/assigned divergence filter
        divergence_scale = 1.0
        if hasattr(self, "divergence_filter") and self.divergence_filter is not None:
            divergence_scale = self.divergence_filter.last_scale

        # 5. Delegate to state-machine router
        return self.execution_router.evaluate_regimes(
            p_hat=p_hat,
            sigma_sq=sigma_sq,
            context=context,
            yes_shares=yes_shares,
            no_shares=no_shares,
            cash_balance=cash,
            divergence_scale=divergence_scale
        )

