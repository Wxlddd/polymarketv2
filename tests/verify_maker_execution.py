import unittest
from typing import Optional, List, Dict, Tuple, Any
from src.core.market_context import MarketContext
from src.core.base_strategy import BaseStrategy
from src.core.interfaces import IExecutionClient
from config.settings import SystemConfig
from src.execution.maker_execution import InventoryManager, ExecutionRouter, OrderInstruction, MakerExecutionEngine

class MockStrategy(BaseStrategy):
    def __init__(self, p_hat: Optional[float] = 0.55):
        self.p_hat = p_hat
        
    def get_probability(self, context: MarketContext) -> Optional[float]:
        return self.p_hat


class MockClientForTest(IExecutionClient):
    def __init__(self, cash: float = 1000.0, yes: float = 0.0, no: float = 0.0):
        self._cash = cash
        self._yes = yes
        self._no = no
        
    @property
    def cash_balance(self) -> float:
        return self._cash
        
    def get_position_size(self, side: str) -> float:
        clean = side.upper()
        if "YES" in clean:
            return self._yes
        if "NO" in clean:
            return self._no
        return 0.0
        
    async def execute_trade(self, *args, **kwargs):
        return {}


class TestMakerExecution(unittest.TestCase):
    
    def setUp(self):
        self.gamma = 0.1
        self.inv_manager = InventoryManager(gamma=self.gamma, fixed_horizon_sec=300.0)
        self.router = ExecutionRouter(
            gamma=self.gamma,
            min_fee_buffer=0.005,
            toxicity_buffer=0.005,
            maker_size=100.0,
            max_inventory=500.0,
            unwind_threshold=0.01,
            taker_edge_epsilon=0.01,
            tick_size=0.01,
            requote_threshold=0.01,
            kelly_fraction=0.15
        )

    def test_reservation_price_skew(self):
        p_hat = 0.55
        sigma_sq = 0.25
        tau_sec = 150.0
        p_res_0 = self.inv_manager.calculate_reservation_price(p_hat, q=0.0, sigma_sq=sigma_sq, tau_seconds=tau_sec)
        self.assertAlmostEqual(p_res_0, 0.55)

        p_res_pos = self.inv_manager.calculate_reservation_price(p_hat, q=100.0, sigma_sq=sigma_sq, tau_seconds=tau_sec)
        self.assertTrue(p_res_pos < p_hat)

        p_res_neg = self.inv_manager.calculate_reservation_price(p_hat, q=-100.0, sigma_sq=sigma_sq, tau_seconds=tau_sec)
        self.assertTrue(p_res_neg > p_hat)

    def test_spread_calculation(self):
        sigma_sq = 0.25
        tau_sec = 300.0
        delta = self.router.calculate_spread(sigma_sq, tau_sec)
        expected = 0.005 + 0.5 * self.gamma * sigma_sq * (tau_sec / (365.25 * 24.0 * 3600.0)) + 0.005
        self.assertAlmostEqual(delta, expected)

    def test_regime_a_maker_quoting(self):
        context = MarketContext(
            timestamp=1000.0,
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.5,
            ofi=0.0,
            bids_l2=[(0.53, 500.0)],
            asks_l2=[(0.57, 500.0)]
        )
        
        instructions = self.router.evaluate_regimes(
            p_hat=0.55,
            sigma_sq=0.25,
            context=context,
            yes_shares=0.0,
            no_shares=0.0,
            cash_balance=1000.0
        )
        
        self.assertEqual(len(instructions), 2)
        bid_instr = next(x for x in instructions if x.side == "BUY_YES")
        ask_instr = next(x for x in instructions if x.side == "SELL_YES")
        self.assertEqual(bid_instr.action, "NEW")
        self.assertEqual(ask_instr.action, "NEW")

    def test_regime_b_taker_mode_buy(self):
        context = MarketContext(
            timestamp=1000.0,
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.5,
            ofi=0.0,
            bids_l2=[(0.38, 500.0)],
            asks_l2=[(0.40, 500.0)]
        )
        
        self.router.reset_active_orders()
        self.router.active_bid_id = "some_bid"
        self.router.active_ask_id = "some_ask"

        instructions = self.router.evaluate_regimes(
            p_hat=0.60,
            sigma_sq=0.25,
            context=context,
            yes_shares=0.0,
            no_shares=0.0,
            cash_balance=1000.0
        )

        self.assertEqual(len(instructions), 3)
        cancels = [x for x in instructions if x.action == "CANCEL"]
        self.assertEqual(len(cancels), 2)
        taker_order = next(x for x in instructions if x.action == "NEW")
        self.assertEqual(taker_order.side, "BUY_YES")
        self.assertEqual(taker_order.price, 0.40)
        self.assertAlmostEqual(taker_order.qty, 125.0)

    def test_regime_c_unwind_mode_positive_inventory(self):
        context = MarketContext(
            timestamp=1000.0,
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.5,
            ofi=0.0,
            bids_l2=[(0.53, 500.0)],
            asks_l2=[(0.57, 500.0)]
        )
        
        self.router.reset_active_orders()
        self.router.active_bid_id = "active_bid"
        self.router.active_ask_id = "active_ask"
        
        instructions = self.router.evaluate_regimes(
            p_hat=0.55,
            sigma_sq=0.25,
            context=context,
            yes_shares=600.0,
            no_shares=0.0,
            cash_balance=1000.0
        )
        
        self.assertEqual(len(instructions), 2)
        bid_cancel = next(x for x in instructions if x.action == "CANCEL" and x.side == "BUY_YES")
        self.assertEqual(bid_cancel.order_id, "active_bid")
        ask_quote = next(x for x in instructions if x.side == "SELL_YES")
        self.assertEqual(ask_quote.regime, "C")

    def test_regime_c_unwind_mode_realignment(self):
        context = MarketContext(
            timestamp=1000.0,
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.5,
            ofi=0.0,
            bids_l2=[(0.54, 500.0)],
            asks_l2=[(0.56, 500.0)]
        )
        
        self.router.reset_active_orders()
        self.router.active_bid_id = "active_bid"
        
        instructions = self.router.evaluate_regimes(
            p_hat=0.552,
            sigma_sq=0.25,
            context=context,
            yes_shares=100.0,
            no_shares=0.0,
            cash_balance=1000.0
        )
        
        bid_cancel = next(x for x in instructions if x.action == "CANCEL" and x.side == "BUY_YES")
        self.assertEqual(bid_cancel.order_id, "active_bid")
        ask_quote = next(x for x in instructions if x.side == "SELL_YES")
        self.assertEqual(ask_quote.regime, "C")

    def test_maker_execution_engine_orchestration(self):
        # Verify MakerExecutionEngine orchestrates strategy, client and router correctly
        config = SystemConfig()
        strategy = MockStrategy(p_hat=0.58)
        client = MockClientForTest(cash=2000.0, yes=50.0, no=0.0)
        
        engine = MakerExecutionEngine(strategy, client, config)
        
        context = MarketContext(
            timestamp=1000.0,
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.5,
            ofi=0.0,
            bids_l2=[(0.53, 500.0)],
            asks_l2=[(0.57, 500.0)]
        )
        
        instructions = engine.evaluate_and_route(context)
        
        # Should yield maker quotes under normal edge (Regime A)
        self.assertTrue(len(instructions) > 0)
        for instr in instructions:
            self.assertEqual(instr.regime, "A")

        # Now test with strategy returning None (Safe cancel mode)
        strategy_none = MockStrategy(p_hat=None)
        engine_none = MakerExecutionEngine(strategy_none, client, config)
        
        # Set active orders so they are cancelled
        engine_none.execution_router.active_bid_id = "bid_1"
        engine_none.execution_router.active_ask_id = "ask_1"
        
        instructions_none = engine_none.evaluate_and_route(context)
        self.assertEqual(len(instructions_none), 2)
        for instr in instructions_none:
            self.assertEqual(instr.action, "CANCEL")
            self.assertEqual(instr.regime, "SAFE")

if __name__ == '__main__':
    unittest.main()
