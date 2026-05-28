import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from src.core.interfaces import IExecutionClient, IDataRecorder
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
        
    @property
    def cash_balance(self) -> float:
        return self._cash_balance

    def get_position_size(self, side: str) -> float:
        """Returns the holding quantity of YES or NO contracts."""
        clean_side = side.upper().replace("BUY_", "").replace("SELL_", "")
        return self.positions.get(clean_side, 0.0)

    async def execute_trade(
        self, 
        side: str, 
        qty: float, 
        price: float, 
        ev: float, 
        expected_slippage_bps: float, 
        context_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulates HFT execution, performs bookkeeping, deducts shadow liquidity,
        and logs outcomes immediately. Now implements True L2 walking, IOC behavior, 
        and stochastic rejection.
        """
        import random
        import math

        gas = self.config.arbitrage.GAS_FEE_USD
        top_bid, top_ask = self.shadow_book.get_top_of_book()
        
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
        
        if side == "SELL_YES" and qty > self.positions["YES"]:
            qty = self.positions["YES"]
        elif side == "SELL_NO" and qty > self.positions["NO"]:
            qty = self.positions["NO"]
            
        if qty <= 1e-9:
            return {"success": False, "reason": "INSUFFICIENT_POSITION"}
            
        # 2. L2 Book Walking for Fills (IOC)
        taker_fee_multiplier = self.config.arbitrage.TAKER_FEE_MULTIPLIER
        filled_qty = 0.0
        total_usd = 0.0
        total_taker_fee = 0.0
        remaining_qty = qty
        
        if side == "BUY_YES":
            levels = self.shadow_book.get_sorted_asks()
            for p, q in levels:
                if p > limit_price:
                    break
                fill = min(remaining_qty, q)
                filled_qty += fill
                total_usd += fill * p
                total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                remaining_qty -= fill
                if remaining_qty <= 1e-9:
                    break
                    
        elif side == "BUY_NO":
            levels = self.shadow_book.get_sorted_bids()
            for p, q in levels:
                if p < limit_price:
                    break
                fill = min(remaining_qty, q)
                filled_qty += fill
                total_usd += fill * (1.0 - p)
                total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                remaining_qty -= fill
                if remaining_qty <= 1e-9:
                    break
                    
        elif side == "SELL_YES":
            levels = self.shadow_book.get_sorted_bids()
            for p, q in levels:
                if p < limit_price:
                    break
                fill = min(remaining_qty, q)
                filled_qty += fill
                total_usd += fill * p
                total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                remaining_qty -= fill
                if remaining_qty <= 1e-9:
                    break
                    
        elif side == "SELL_NO":
            levels = self.shadow_book.get_sorted_asks()
            for p, q in levels:
                if p > limit_price:
                    break
                fill = min(remaining_qty, q)
                filled_qty += fill
                total_usd += fill * (1.0 - p)
                total_taker_fee += fill * taker_fee_multiplier * p * (1.0 - p)
                remaining_qty -= fill
                if remaining_qty <= 1e-9:
                    break

        vwap_exec = total_usd / filled_qty if filled_qty > 0 else 0.0
        
        if (filled_qty * vwap_exec) < 50.0:
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
        if side in ["BUY_YES", "BUY_NO"]:
            realized_slippage_bps = ((vwap_exec - p_market) / p_market) * 10000.0 if p_market > 0.0 else 0.0
        else:
            realized_slippage_bps = ((p_market - vwap_exec) / p_market) * 10000.0 if p_market > 0.0 else 0.0
            
        realized_pnl = 0.0
        
        # Bookkeeping based on trade side
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
            self.shadow_book.paper_execute(0.0, filled_qty, is_bid=False)
            
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
            self.shadow_book.paper_execute(0.0, filled_qty, is_bid=True)
            
        elif side == "SELL_YES":
            revenue = total_usd - gas - total_taker_fee
            self._cash_balance += revenue
            purchase_cost = filled_qty * self.entry_prices["YES"]
            realized_pnl = revenue - purchase_cost
            
            self.positions["YES"] -= filled_qty
            if self.positions["YES"] <= 1e-9:
                self.positions["YES"] = 0.0
                self.entry_prices["YES"] = 0.0
                
            self.shadow_book.paper_execute(0.0, filled_qty, is_bid=True)
            
        elif side == "SELL_NO":
            revenue = total_usd - gas - total_taker_fee
            self._cash_balance += revenue
            purchase_cost = filled_qty * self.entry_prices["NO"]
            realized_pnl = revenue - purchase_cost
            
            self.positions["NO"] -= filled_qty
            if self.positions["NO"] <= 1e-9:
                self.positions["NO"] = 0.0
                self.entry_prices["NO"] = 0.0
                
            self.shadow_book.paper_execute(0.0, filled_qty, is_bid=False)
            
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

    def settle_positions(self, settlement_price: float, strike_price: float, timestamp: float) -> float:
        """
        Settles all remaining holdings at expiration and resets contracts to zero.
        Returns the net payout in USD.
        """
        payout = 0.0
        gas = self.config.arbitrage.GAS_FEE_USD
        
        # 1. Settle YES positions
        if self.positions["YES"] > 0.0:
            won = settlement_price > strike_price
            payoff_per_contract = 1.0 if won else 0.0
            
            qty = self.positions["YES"]
            total_payoff = qty * payoff_per_contract
            purchase_cost = qty * self.entry_prices["YES"]
            pnl = total_payoff - purchase_cost - gas
            
            self._cash_balance += total_payoff
            payout += total_payoff
            
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
            logger.info(f"[MockClient] Settled YES position. Qty: {qty} | Won: {won} | PnL: ${pnl:+.2f}")
            self.positions["YES"] = 0.0
            self.entry_prices["YES"] = 0.0
            
        # 2. Settle NO positions
        if self.positions["NO"] > 0.0:
            won = settlement_price <= strike_price
            payoff_per_contract = 1.0 if won else 0.0
            
            qty = self.positions["NO"]
            total_payoff = qty * payoff_per_contract
            purchase_cost = qty * self.entry_prices["NO"]
            pnl = total_payoff - purchase_cost - gas
            
            self._cash_balance += total_payoff
            payout += total_payoff
            
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
            logger.info(f"[MockClient] Settled NO position. Qty: {qty} | Won: {won} | PnL: ${pnl:+.2f}")
            self.positions["NO"] = 0.0
            self.entry_prices["NO"] = 0.0
            
        return payout
