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
        and logs outcomes immediately.
        """
        gas = self.config.arbitrage.GAS_FEE_USD
        top_bid, top_ask = self.shadow_book.get_top_of_book()
        
        # Get market reference price (best bid/ask before execution slippage)
        p_market = price
        if side == "BUY_YES":
            p_market = top_ask[0] if top_ask else price
        elif side == "BUY_NO":
            p_market = (1.0 - top_bid[0]) if top_bid else price
        elif side == "SELL_YES":
            p_market = top_bid[0] if top_bid else price
        elif side == "SELL_NO":
            p_market = (1.0 - top_ask[0]) if top_ask else price
            
        # Calculate actual slippage relative to market reference
        realized_slippage_bps = 0.0
        if side in ["BUY_YES", "BUY_NO"]:
            realized_slippage_bps = ((price - p_market) / p_market) * 10000.0 if p_market > 0.0 else 0.0
        else:
            realized_slippage_bps = ((p_market - price) / p_market) * 10000.0 if p_market > 0.0 else 0.0
            
        realized_pnl = 0.0
        total_usd = qty * price
        
        # Calculate dynamic taker fee: shares * 0.072 * p * (1 - p)
        taker_fee_multiplier = self.config.arbitrage.TAKER_FEE_MULTIPLIER
        taker_fee = qty * taker_fee_multiplier * price * (1.0 - price)
        
        # Bookkeeping based on trade side
        if side == "BUY_YES":
            cost = total_usd + gas + taker_fee
            if cost > self._cash_balance:
                logger.warning(f"[MockClient] BUY_YES rejected: Insufficient cash balance. Needed ${cost:.2f}, Balance: ${self._cash_balance:.2f}")
                return {"success": False, "reason": "INSUFFICIENT_FUNDS"}
            
            # Deduct funds
            self._cash_balance -= cost
            # Update YES positions
            cur_qty = self.positions["YES"]
            new_qty = cur_qty + qty
            if new_qty > 0.0:
                self.entry_prices["YES"] = (self.entry_prices["YES"] * cur_qty + price * qty) / new_qty
            self.positions["YES"] = new_qty
            
            # Paper execution (BUY_YES depletes asks on the book)
            self.shadow_book.paper_execute(price, qty, is_bid=False)
            
        elif side == "BUY_NO":
            cost = total_usd + gas + taker_fee
            if cost > self._cash_balance:
                logger.warning(f"[MockClient] BUY_NO rejected: Insufficient cash. Needed ${cost:.2f}")
                return {"success": False, "reason": "INSUFFICIENT_FUNDS"}
                
            self._cash_balance -= cost
            cur_qty = self.positions["NO"]
            new_qty = cur_qty + qty
            if new_qty > 0.0:
                self.entry_prices["NO"] = (self.entry_prices["NO"] * cur_qty + price * qty) / new_qty
            self.positions["NO"] = new_qty
            
            # Paper execution (BUY_NO depletes YES bids at price 1.0 - price)
            self.shadow_book.paper_execute(1.0 - price, qty, is_bid=True)
            
        elif side == "SELL_YES":
            if qty > self.positions["YES"]:
                logger.warning(f"[MockClient] SELL_YES rejected: Selling size {qty} exceeds YES holdings {self.positions['YES']}")
                return {"success": False, "reason": "INSUFFICIENT_POSITION"}
                
            revenue = total_usd - gas - taker_fee
            self._cash_balance += revenue
            
            # Calculate realized PnL
            purchase_cost = qty * self.entry_prices["YES"]
            realized_pnl = revenue - purchase_cost
            
            # Reduce YES positions
            self.positions["YES"] -= qty
            if self.positions["YES"] <= 1e-9:
                self.positions["YES"] = 0.0
                self.entry_prices["YES"] = 0.0
                
            # Paper execution (SELL_YES depletes bids on the book)
            self.shadow_book.paper_execute(price, qty, is_bid=True)
            
        elif side == "SELL_NO":
            if qty > self.positions["NO"]:
                logger.warning(f"[MockClient] SELL_NO rejected: Selling size {qty} exceeds NO holdings {self.positions['NO']}")
                return {"success": False, "reason": "INSUFFICIENT_POSITION"}
                
            revenue = total_usd - gas - taker_fee
            self._cash_balance += revenue
            
            purchase_cost = qty * self.entry_prices["NO"]
            realized_pnl = revenue - purchase_cost
            
            self.positions["NO"] -= qty
            if self.positions["NO"] <= 1e-9:
                self.positions["NO"] = 0.0
                self.entry_prices["NO"] = 0.0
                
            # Paper execution (SELL_NO depletes asks on the book at price 1.0 - price)
            self.shadow_book.paper_execute(1.0 - price, qty, is_bid=False)
            
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
            qty=qty,
            vwap=price,
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
            f"[MockClient] Executed {side} | Qty: {qty:.2f} | Avg Price: ${price:.4f} | "
            f"Slippage: {realized_slippage_bps:.2f} bps | realized PnL: ${realized_pnl:+.2f} | "
            f"Cash: ${self._cash_balance:.2f}"
        )
        
        return {
            "success": True,
            "side": side,
            "qty": qty,
            "price": price,
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
