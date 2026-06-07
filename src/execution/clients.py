import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from src.core.interfaces import IExecutionClient, IDataRecorder, OrderInstruction
from src.execution.shadow_book import ShadowOrderBook
from config.settings import SystemConfig

logger = logging.getLogger("MockExecutionClient")

class MockExecutionClient(IExecutionClient):
    """
    Mock/Paper Trading Execution Client.
    Manages a simulated balance and contract holdings, routes simulated executions,
    deducts liquidity from the ShadowOrderBook proxy, and logs events via the DataRecorder.
    """
    
    def __init__(self, config: SystemConfig, recorder: IDataRecorder, shadow_book: ShadowOrderBook):
        self.config = config
        self.recorder = recorder
        self.shadow_book = shadow_book
        
        self._cash_balance = config.arbitrage.INITIAL_CAPITAL
        
        # Position sizes in contract units (quantities)
        self.positions: Dict[str, float] = {"YES": 0.0, "NO": 0.0}
        # Average entry price per contract type
        self.entry_prices: Dict[str, float] = {"YES": 0.0, "NO": 0.0}
        
        # Realized exits/trades (SELL/SETTLE) tracking
        self.realized_trades: List[Dict[str, Any]] = []
        
        # Resting maker orders: {"bid": dict, "ask": dict}
        # Dict format: {"instruction": OrderInstruction, "queue_ahead": float, "prev_depth": float, "touched": bool}
        self.active_maker_orders: Dict[str, Optional[Dict[str, Any]]] = {"bid": None, "ask": None}
        
    def _merge_positions(self) -> None:
        """Symmetric position merging: 1 YES + 1 NO = $1.00 cash."""
        qty_to_merge = min(self.positions["YES"], self.positions["NO"])
        if qty_to_merge > 0.0:
            self._cash_balance += qty_to_merge
            self.positions["YES"] -= qty_to_merge
            self.positions["NO"] -= qty_to_merge
            
            # Reset entry prices if positions are fully closed
            if self.positions["YES"] <= 1e-9:
                self.positions["YES"] = 0.0
                self.entry_prices["YES"] = 0.0
            if self.positions["NO"] <= 1e-9:
                self.positions["NO"] = 0.0
                self.entry_prices["NO"] = 0.0
                
            logger.info(f"[MockClient] Merged {qty_to_merge:.2f} YES and NO positions. Cash received: ${qty_to_merge:.2f}")
        
    @property
    def cash_balance(self) -> float:
        return self._cash_balance

    def get_position_size(self, side: str) -> float:
        """Returns the holding quantity of YES or NO contracts."""
        clean_side = side.upper().replace("BUY_", "").replace("SELL_", "")
        return self.positions.get(clean_side, 0.0)

    async def _execute_trade_internal(
        self, 
        side: str, 
        qty: float, 
        price: float, 
        ev: float, 
        expected_slippage_bps: float, 
        context_state: Dict[str, Any],
        is_maker: bool = False
    ) -> Dict[str, Any]:
        """
        Simulates HFT execution, performs bookkeeping, deducts shadow liquidity,
        and logs outcomes immediately. Now implements True L2 walking, IOC behavior, 
        stochastic rejection, and simulated network latency. Supports maker (limit) executions
        which execute exactly at the limit price with zero taker fees and no book walking.
        """
        import random
        import math
        import asyncio

        # Simulate network round-trip time and exchange processing latency (150ms - 300ms)
        # This is critical for the queue logic in main.py to correctly block
        # new signals from being evaluated while an order is in flight.
        if not getattr(self, "is_backtest", False) and not is_maker:
            latency = random.uniform(0.150, 0.300)
            await asyncio.sleep(latency)

        gas = self.config.arbitrage.GAS_FEE_USD
        top_bid, top_ask = self.shadow_book.get_top_of_book()
        
        if side == "SELL_YES" and qty > self.positions["YES"]:
            qty = self.positions["YES"]
        elif side == "SELL_NO" and qty > self.positions["NO"]:
            qty = self.positions["NO"]
            
        if qty <= 1e-9:
            return {"success": False, "reason": "INSUFFICIENT_POSITION"}

        if is_maker:
            # Maker (limit order) fills execute exactly at the limit price
            # with zero taker fee, bypassing book walking and stochastic rejection
            limit_price = price
            filled_qty = qty
            total_usd = qty * price
            total_taker_fee = 0.0
            vwap_exec = price
            realized_slippage_bps = 0.0
            exec_is_bid = False
            level_fills = None
        else:
            # 1. Stochastic Rejection
            vol = context_state.get("volatility", self.config.merton.DEFAULT_SIGMA)
            # Beta coefficients for logistic rejection:
            # Base failure rate (approx 5% base -> logit -2.94)
            beta_0 = -3.0
            # Impact of size (e.g. 1000 contracts adds +1.0)
            beta_1 = 0.001
            # Impact of vol (e.g. vol of 1.0 adds +2.0)
            beta_2 = 2.0
            
            logit = beta_0 + beta_1 * qty + beta_2 * vol
            p_reject = 1.0 / (1.0 + math.exp(-logit))
            
            if random.random() < p_reject:
                logger.warning(f"[MockClient] Trade rejected stochastically (P_reject={p_reject:.2f})")
                return {"success": False, "reason": "STOCHASTIC_REJECTION"}

            limit_price = context_state.get("limit_price", price)
            
            # 2. L2 Book Walking for Fills (IOC)
            # level_fills records exact (price, qty) consumed at each level so that
            # paper_execute can register them precisely in the ConsumptionTracker.
            taker_fee_multiplier = self.config.arbitrage.TAKER_FEE_MULTIPLIER
            filled_qty = 0.0
            total_usd = 0.0
            total_taker_fee = 0.0
            remaining_qty = qty
            level_fills: list = []   # List[Tuple[float, float]] — (price, qty)
            exec_is_bid: bool = False  # True → consumed bid side

            if side == "BUY_YES":
                exec_is_bid = False  # consuming asks
                levels = self.shadow_book.get_sorted_asks()
                for p, q in levels:
                    if p > limit_price:
                        break
                    fill = min(remaining_qty, q)
                    level_fills.append((p, fill))
                    filled_qty += fill
                    total_usd += fill * p
                    total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                    remaining_qty -= fill
                    if remaining_qty <= 1e-9:
                        break

            elif side == "BUY_NO":
                exec_is_bid = True  # consuming bids
                levels = self.shadow_book.get_sorted_bids()
                for p, q in levels:
                    if p < limit_price:
                        break
                    fill = min(remaining_qty, q)
                    level_fills.append((p, fill))
                    filled_qty += fill
                    total_usd += fill * (1.0 - p)
                    total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                    remaining_qty -= fill
                    if remaining_qty <= 1e-9:
                        break

            elif side == "SELL_YES":
                exec_is_bid = True  # consuming bids
                levels = self.shadow_book.get_sorted_bids()
                for p, q in levels:
                    if p < limit_price:
                        break
                    fill = min(remaining_qty, q)
                    level_fills.append((p, fill))
                    filled_qty += fill
                    total_usd += fill * p
                    total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                    remaining_qty -= fill
                    if remaining_qty <= 1e-9:
                        break

            elif side == "SELL_NO":
                exec_is_bid = False  # consuming asks
                levels = self.shadow_book.get_sorted_asks()
                for p, q in levels:
                    if p > limit_price:
                        break
                    fill = min(remaining_qty, q)
                    level_fills.append((p, fill))
                    filled_qty += fill
                    total_usd += fill * (1.0 - p)
                    total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                    remaining_qty -= fill
                    if remaining_qty <= 1e-9:
                        break

            vwap_exec = total_usd / filled_qty if filled_qty > 0 else 0.0
            
            if (filled_qty * vwap_exec) < self.config.arbitrage.MIN_ORDER_USD:
                logger.warning(f"[MockClient] IOC fill too small: {filled_qty:.2f} at ${vwap_exec:.4f}")
                return {"success": False, "reason": "IOC_FILL_TOO_SMALL"}

        # Get market reference price
        p_market = limit_price
        if side == "BUY_YES":
            p_market = top_ask[0] if top_ask else limit_price
        elif side == "BUY_NO":
            p_market = (1.0 - top_bid[0]) if top_bid else limit_price
        elif side == "SELL_YES":
            p_market = top_bid[0] if top_bid else limit_price
        elif side == "SELL_NO":
            p_market = (1.0 - top_ask[0]) if top_ask else limit_price
            
        realized_slippage_bps = 0.0
        if not is_maker:
            if side in ["BUY_YES", "BUY_NO"]:
                realized_slippage_bps = ((vwap_exec - p_market) / p_market) * 10000.0 if p_market > 0.0 else 0.0
            else:
                realized_slippage_bps = ((p_market - vwap_exec) / p_market) * 10000.0 if p_market > 0.0 else 0.0
            
        realized_pnl = 0.0
        
        # Bookkeeping based on trade side
        exec_ts = context_state.get("timestamp", 0.0)

        if side == "BUY_YES":
            cost = total_usd + gas + total_taker_fee
            if cost > self._cash_balance:
                logger.warning(f"[MockClient] BUY_YES rejected: Insufficient cash balance. Needed ${cost:.2f}, Balance: ${self._cash_balance:.2f}")
                return {"success": False, "reason": "INSUFFICIENT_FUNDS"}

            self._cash_balance -= cost
            cur_qty = self.positions["YES"]
            new_qty = cur_qty + filled_qty
            if new_qty > 0.0:
                self.entry_prices["YES"] = (self.entry_prices["YES"] * cur_qty + total_usd) / new_qty
            self.positions["YES"] = new_qty
            # Pass exact per-level fills to the ConsumptionTracker
            if not is_maker:
                self.shadow_book.paper_execute(0.0, filled_qty, is_bid=exec_is_bid, fills=level_fills, timestamp=exec_ts)

        elif side == "BUY_NO":
            cost = total_usd + gas + total_taker_fee
            if cost > self._cash_balance:
                logger.warning(f"[MockClient] BUY_NO rejected: Insufficient cash. Needed ${cost:.2f}")
                return {"success": False, "reason": "INSUFFICIENT_FUNDS"}

            self._cash_balance -= cost
            cur_qty = self.positions["NO"]
            new_qty = cur_qty + filled_qty
            if new_qty > 0.0:
                self.entry_prices["NO"] = (self.entry_prices["NO"] * cur_qty + total_usd) / new_qty
            self.positions["NO"] = new_qty
            if not is_maker:
                self.shadow_book.paper_execute(0.0, filled_qty, is_bid=exec_is_bid, fills=level_fills, timestamp=exec_ts)

        elif side == "SELL_YES":
            revenue = total_usd - gas - total_taker_fee
            self._cash_balance += revenue
            purchase_cost = filled_qty * self.entry_prices["YES"]
            realized_pnl = revenue - purchase_cost
            avg_entry_price = self.entry_prices["YES"]

            self.positions["YES"] -= filled_qty
            if self.positions["YES"] <= 1e-9:
                self.positions["YES"] = 0.0
                self.entry_prices["YES"] = 0.0
            if not is_maker:
                self.shadow_book.paper_execute(0.0, filled_qty, is_bid=exec_is_bid, fills=level_fills, timestamp=exec_ts)
            
            self.realized_trades.append({
                "timestamp": exec_ts,
                "side": side,
                "qty": filled_qty,
                "entry_price": avg_entry_price,
                "exit_price": vwap_exec,
                "pnl": realized_pnl,
                "won": realized_pnl > 0.0
            })

        elif side == "SELL_NO":
            revenue = total_usd - gas - total_taker_fee
            self._cash_balance += revenue
            purchase_cost = filled_qty * self.entry_prices["NO"]
            realized_pnl = revenue - purchase_cost
            avg_entry_price = self.entry_prices["NO"]

            self.positions["NO"] -= filled_qty
            if self.positions["NO"] <= 1e-9:
                self.positions["NO"] = 0.0
                self.entry_prices["NO"] = 0.0
            if not is_maker:
                self.shadow_book.paper_execute(0.0, filled_qty, is_bid=exec_is_bid, fills=level_fills, timestamp=exec_ts)
            
            self.realized_trades.append({
                "timestamp": exec_ts,
                "side": side,
                "qty": filled_qty,
                "entry_price": avg_entry_price,
                "exit_price": vwap_exec,
                "pnl": realized_pnl,
                "won": realized_pnl > 0.0
            })
            
        # Auto-merge positions to align with Polymarket blockchain cash settlements
        self._merge_positions()

        # Compute mid-price for consistent portfolio MTM, independent of which
        # side triggered the call.  Using the execution-side price (p_market) as the
        # YES reference caused the MTM to flip between bid and ask depending on
        # whether the trade was on YES or NO, producing artificial capital swings.
        top_bid_mtm, top_ask_mtm = self.shadow_book.get_market_top_of_book()
        if top_bid_mtm and top_ask_mtm:
            current_yes_price = 0.5 * (top_bid_mtm[0] + top_ask_mtm[0])
        elif top_bid_mtm:
            current_yes_price = top_bid_mtm[0]
        elif top_ask_mtm:
            current_yes_price = top_ask_mtm[0]
        else:
            current_yes_price = p_market if "YES" in side else (1.0 - p_market)
        self.recorder.record_trade(
            timestamp=context_state.get("timestamp", time.time()),
            side=side,
            qty=filled_qty,
            vwap=vwap_exec,
            p_market=p_market,
            expected_slippage_bps=expected_slippage_bps,
            realized_slippage_bps=realized_slippage_bps,
            ev=ev,
            strike=context_state.get("strike_price", 0.0),
            resolved_won=bool(realized_pnl > 0.0),
            pnl=realized_pnl,
            capital=self._cash_balance + self.get_portfolio_value(current_yes_price)
        )
        
        logger.info(
            f"[MockClient] Executed {side} | Qty: {filled_qty:.2f} | Avg Price: ${vwap_exec:.4f} | "
            f"Slippage: {realized_slippage_bps:.2f} bps | realized PnL: ${realized_pnl:+.2f} | "
            f"Cash: ${self._cash_balance:.2f}"
        )
        
        return {
            "success": True,
            "side": side,
            "qty": filled_qty,
            "price": vwap_exec,
            "realized_slippage_bps": realized_slippage_bps,
            "pnl": realized_pnl
        }

    def get_portfolio_value(self, current_yes_price: float) -> float:
        """Returns the mark-to-market value of open positions."""
        val = 0.0
        # MTM of YES contracts
        val += self.positions["YES"] * current_yes_price
        # MTM of NO contracts (complementary pricing)
        val += self.positions["NO"] * (1.0 - current_yes_price)
        return val

    def settle_positions(self, settlement_price: float, strike_price: float, timestamp: float) -> dict:
        """
        Settles all remaining holdings at expiration and resets contracts to zero.

        Returns a dict with a full breakdown:
        {
            "total_payout": float,       # gross cash received
            "net_pnl":      float,       # payout minus entry cost minus gas fees
            "yes": {                     # None if no YES position was open
                "qty": float,
                "entry_price": float,
                "payoff_per_contract": float,   # 1.0 or 0.0
                "gross_payoff": float,
                "cost_basis": float,
                "pnl": float,
                "won": bool
            } | None,
            "no": {                      # None if no NO position was open
                ...same fields...
            } | None,
        }
        """
        gas = self.config.arbitrage.GAS_FEE_USD
        total_payout = 0.0
        net_pnl = 0.0
        yes_summary = None
        no_summary = None

        # 1. Settle YES positions
        if self.positions["YES"] > 0.0:
            won = settlement_price > strike_price
            payoff_per_contract = 1.0 if won else 0.0

            qty = self.positions["YES"]
            gross_payoff = qty * payoff_per_contract
            cost_basis = qty * self.entry_prices["YES"]
            pnl = gross_payoff - cost_basis - gas

            self._cash_balance += gross_payoff
            total_payout += gross_payoff
            net_pnl += pnl

            yes_summary = {
                "qty": qty,
                "entry_price": self.entry_prices["YES"],
                "payoff_per_contract": payoff_per_contract,
                "gross_payoff": gross_payoff,
                "cost_basis": cost_basis,
                "pnl": pnl,
                "won": won,
            }

            self.recorder.record_trade(
                timestamp=timestamp,
                side="SETTLE_YES",
                qty=qty,
                vwap=payoff_per_contract,
                p_market=payoff_per_contract,
                expected_slippage_bps=0.0,
                realized_slippage_bps=0.0,
                ev=0.0,
                strike=strike_price,
                resolved_won=won,
                pnl=pnl,
                capital=self._cash_balance
            )
            
            self.realized_trades.append({
                "timestamp": timestamp,
                "side": "SETTLE_YES",
                "qty": qty,
                "entry_price": self.entry_prices["YES"],
                "exit_price": payoff_per_contract,
                "pnl": pnl,
                "won": won
            })
            
            self.positions["YES"] = 0.0
            self.entry_prices["YES"] = 0.0

        # 2. Settle NO positions
        if self.positions["NO"] > 0.0:
            won = settlement_price <= strike_price
            payoff_per_contract = 1.0 if won else 0.0

            qty = self.positions["NO"]
            gross_payoff = qty * payoff_per_contract
            cost_basis = qty * self.entry_prices["NO"]
            pnl = gross_payoff - cost_basis - gas

            self._cash_balance += gross_payoff
            total_payout += gross_payoff
            net_pnl += pnl

            no_summary = {
                "qty": qty,
                "entry_price": self.entry_prices["NO"],
                "payoff_per_contract": payoff_per_contract,
                "gross_payoff": gross_payoff,
                "cost_basis": cost_basis,
                "pnl": pnl,
                "won": won,
            }

            self.recorder.record_trade(
                timestamp=timestamp,
                side="SETTLE_NO",
                qty=qty,
                vwap=payoff_per_contract,
                p_market=payoff_per_contract,
                expected_slippage_bps=0.0,
                realized_slippage_bps=0.0,
                ev=0.0,
                strike=strike_price,
                resolved_won=won,
                pnl=pnl,
                capital=self._cash_balance
            )
            
            self.realized_trades.append({
                "timestamp": timestamp,
                "side": "SETTLE_NO",
                "qty": qty,
                "entry_price": self.entry_prices["NO"],
                "exit_price": payoff_per_contract,
                "pnl": pnl,
                "won": won
            })
            
            self.positions["NO"] = 0.0
            self.entry_prices["NO"] = 0.0

        return {
            "total_payout": total_payout,
            "net_pnl": net_pnl,
            "yes": yes_summary,
            "no": no_summary,
        }


    async def process_instruction(self, instruction: OrderInstruction, context_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a routing instruction (NEW, CANCEL, REPLACE).
        For Taker (Regime B/PANIC), executes immediately (IOC).
        For Maker (Regime A/C), places a limit order resting in the mock client book.
        """
        side_key = "bid" if "BUY" in instruction.side else "ask"
        
        if instruction.action == "CANCEL":
            if self.active_maker_orders[side_key] and self.active_maker_orders[side_key]["instruction"].order_id == instruction.order_id:
                self.active_maker_orders[side_key] = None
            return {"success": True, "action": "CANCEL", "order_id": instruction.order_id}

        # For NEW or REPLACE
        if instruction.regime in ("B", "PANIC"):
            # Taker execution (IOC)
            ev = 0.0 # Taker ev should ideally be passed in context_state or calculated
            return await self._execute_trade_internal(
                side=instruction.side,
                qty=instruction.qty,
                price=instruction.price,
                ev=ev,
                expected_slippage_bps=0.0,
                context_state=context_state,
                is_maker=False
            )
        else:
            # Maker execution (RESTING)
            cur_depth = 0.0
            if side_key == "bid":
                for p, q in self.shadow_book.get_sorted_bids():
                    if p >= instruction.price:
                        cur_depth += q
                    else:
                        break
            else:
                for p, q in self.shadow_book.get_sorted_asks():
                    if p <= instruction.price:
                        cur_depth += q
                    else:
                        break

            self.active_maker_orders[side_key] = {
                "instruction": instruction,
                "queue_ahead": cur_depth,
                "prev_depth": cur_depth,
                "touched": False,
                "recorded_touch": False
            }
            return {"success": True, "action": instruction.action, "order_id": instruction.order_id}

    async def process_market_data(self, bids_l2: List[Tuple[float, float]], asks_l2: List[Tuple[float, float]], context_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Updates the L2 book and evaluates resting Maker limit orders for queue depth and simulated fills.
        Returns a list of fill results (if any).
        """
        fills = []
        best_bid_p = bids_l2[0][0] if bids_l2 else None
        best_ask_p = asks_l2[0][0] if asks_l2 else None

        # 1. Update Queue Depth for BID
        if self.active_maker_orders["bid"]:
            order = self.active_maker_orders["bid"]
            instr = order["instruction"]
            active_bid_p = instr.price
            
            cur_depth = 0.0
            for p, q in bids_l2:
                if p >= active_bid_p:
                    cur_depth += q
                else:
                    break
                    
            if best_ask_p is not None:
                if best_ask_p < active_bid_p:
                    order["queue_ahead"] = 0.0
                    order["touched"] = True
                elif best_ask_p == active_bid_p:
                    order["touched"] = True
                    delta_depth = order["prev_depth"] - cur_depth
                    if delta_depth > 0:
                        alpha = 0.40
                        v_traded = alpha * delta_depth
                        c_cancels = (1.0 - alpha) * delta_depth
                        ratio = order["queue_ahead"] / order["prev_depth"] if order["prev_depth"] > 0 else 0.0
                        v_cancelled_ahead = c_cancels * ratio
                        order["queue_ahead"] = max(0.0, order["queue_ahead"] - v_traded - v_cancelled_ahead)
                    
                    stoch_decay = 0.02 * order["queue_ahead"]
                    order["queue_ahead"] = max(0.0, order["queue_ahead"] - stoch_decay)
                    order["prev_depth"] = cur_depth
                    
        # 2. Update Queue Depth for ASK
        if self.active_maker_orders["ask"]:
            order = self.active_maker_orders["ask"]
            instr = order["instruction"]
            active_ask_p = instr.price
            
            cur_depth = 0.0
            for p, q in asks_l2:
                if p <= active_ask_p:
                    cur_depth += q
                else:
                    break
                    
            if best_bid_p is not None:
                if best_bid_p > active_ask_p:
                    order["queue_ahead"] = 0.0
                    order["touched"] = True
                elif best_bid_p == active_ask_p:
                    order["touched"] = True
                    delta_depth = order["prev_depth"] - cur_depth
                    if delta_depth > 0:
                        alpha = 0.40
                        v_traded = alpha * delta_depth
                        c_cancels = (1.0 - alpha) * delta_depth
                        ratio = order["queue_ahead"] / order["prev_depth"] if order["prev_depth"] > 0 else 0.0
                        v_cancelled_ahead = c_cancels * ratio
                        order["queue_ahead"] = max(0.0, order["queue_ahead"] - v_traded - v_cancelled_ahead)
                    
                    stoch_decay = 0.02 * order["queue_ahead"]
                    order["queue_ahead"] = max(0.0, order["queue_ahead"] - stoch_decay)
                    order["prev_depth"] = cur_depth

        # 3. Process Fills
        # Buy Fill
        if self.active_maker_orders["bid"]:
            order = self.active_maker_orders["bid"]
            instr = order["instruction"]
            if best_ask_p is not None:
                is_crossed = best_ask_p < instr.price
                is_touched_and_front = best_ask_p == instr.price and order["queue_ahead"] <= 0.0
                if is_crossed or is_touched_and_front:
                    # Execute fill
                    fill_result = await self._execute_trade_internal(
                        side=instr.side,
                        qty=instr.qty,
                        price=instr.price,
                        ev=0.0,
                        expected_slippage_bps=0.0,
                        context_state=context_state,
                        is_maker=True
                    )
                    fill_result["instruction"] = instr
                    fills.append(fill_result)
                    self.active_maker_orders["bid"] = None

        # Sell Fill
        if self.active_maker_orders["ask"]:
            order = self.active_maker_orders["ask"]
            instr = order["instruction"]
            if best_bid_p is not None:
                is_crossed = best_bid_p > instr.price
                is_touched_and_front = best_bid_p == instr.price and order["queue_ahead"] <= 0.0
                if is_crossed or is_touched_and_front:
                    # We might need to split into SELL_YES and BUY_NO if yes_shares are insufficient.
                    # This logic should be here.
                    yes_shares = self.get_position_size("YES")
                    exec_qty = instr.qty
                    exec_price = instr.price
                    
                    if yes_shares >= exec_qty:
                        fill_result = await self._execute_trade_internal(
                            side="SELL_YES",
                            qty=exec_qty,
                            price=exec_price,
                            ev=0.0,
                            expected_slippage_bps=0.0,
                            context_state=context_state,
                            is_maker=True
                        )
                        fill_result["instruction"] = instr
                        fills.append(fill_result)
                    else:
                        if yes_shares > 0.0:
                            fill1 = await self._execute_trade_internal(
                                side="SELL_YES",
                                qty=yes_shares,
                                price=exec_price,
                                ev=0.0,
                                expected_slippage_bps=0.0,
                                context_state=context_state,
                                is_maker=True
                            )
                            fill1["instruction"] = instr
                            fills.append(fill1)
                            
                        rem_q = exec_qty - yes_shares
                        no_price = 1.0 - exec_price
                        fill2 = await self._execute_trade_internal(
                            side="BUY_NO",
                            qty=rem_q,
                            price=no_price,
                            ev=0.0,
                            expected_slippage_bps=0.0,
                            context_state=context_state,
                            is_maker=True
                        )
                        fill2["instruction"] = instr
                        fill2["is_split"] = True
                        fills.append(fill2)
                        
                    self.active_maker_orders["ask"] = None

        return fills
