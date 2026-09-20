import unittest
from typing import Optional, List, Dict, Tuple, Any
from src.core.market_context import MarketContext
from src.core.base_strategy import BaseStrategy
from src.core.interfaces import IExecutionClient
from config.settings import SystemConfig
from src.execution.engine import ExecutionRouter, ExecutionEngine
from src.core.interfaces import OrderInstruction

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
        
    async def _execute_trade_internal(self, *args, **kwargs):
        return {}

    async def process_instruction(self, instruction, context_state):
        return {'success': True, 'action': instruction.action, 'qty': instruction.qty, 'price': instruction.price, 'side': instruction.side}

    def process_market_data(self, bids_l2, asks_l2, context_state):
        return []



class TestMakerExecution(unittest.TestCase):
    
    def setUp(self):
        self.gamma = 0.1
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

    def _quoted_bid(self, yes_shares: float, no_shares: float) -> float:
        """
        The router's Regime A bid, which sits at P_res - delta and therefore tracks the
        inventory skew. The book is deliberately wide (0.30/0.70) so p_hat stays far from
        the mid (no Regime C unwind) while the target bid stays inside the ask (no Regime B).
        """
        ctx = MarketContext(
            timestamp=1000.0, spot_price=67500.0, strike_price=67500.0, tau_seconds=150.0,
            volatility=0.5, ofi=0.0, bids_l2=[(0.30, 5000.0)], asks_l2=[(0.70, 5000.0)]
        )
        instrs = self.router.evaluate_regimes(
            p_hat=0.55, sigma_sq=1.0, context=ctx,
            yes_shares=yes_shares, no_shares=no_shares, cash_balance=100000.0
        )
        self.router.reset_active_orders()
        bids = [i for i in instrs if i.side == "BUY_YES" and i.action in ("NEW", "REPLACE")]
        self.assertTrue(bids, f"router produced no bid for q={yes_shares - no_shares}: {instrs}")
        self.assertEqual(bids[0].regime, "A")
        return bids[0].price

    def test_reservation_price_skew(self):
        """Long inventory must skew quotes down, short inventory up (P_res = p_hat - gamma*q_norm*sigma^2)."""
        flat = self._quoted_bid(yes_shares=0.0, no_shares=0.0)
        long_inv = self._quoted_bid(yes_shares=250.0, no_shares=0.0)
        short_inv = self._quoted_bid(yes_shares=0.0, no_shares=250.0)
        self.assertLess(long_inv, flat)
        self.assertGreater(short_inv, flat)

    def test_spread_calculation(self):
        """Half-spread = buffers + half the risk term, in probability points."""
        sigma_sq = 1e-5          # points^2 per second, the measured order of magnitude
        tau_sec = 300.0
        delta = self.router.calculate_spread(sigma_sq, tau_sec)
        expected = 0.005 + 0.005 + 0.5 * self.gamma * sigma_sq * tau_sec
        self.assertAlmostEqual(delta, expected)

    def test_risk_term_units_are_shared(self):
        """The skew and the half-spread must be built from the same quantity, and that
        quantity must scale with variance and with the horizon."""
        sigma_sq, tau = 1e-5, 300.0
        risk = self.router.risk_term(sigma_sq, tau)
        self.assertAlmostEqual(risk, self.gamma * sigma_sq * tau)
        # half-spread carries exactly half of it
        self.assertAlmostEqual(self.router.calculate_spread(sigma_sq, tau) - 0.01, 0.5 * risk)
        # linear in variance, linear in horizon (below the cap)
        self.assertAlmostEqual(self.router.risk_term(2 * sigma_sq, tau), 2 * risk)
        self.assertAlmostEqual(self.router.risk_term(sigma_sq, tau / 2), risk / 2)

    def test_horizon_capped_at_holding_time(self):
        """A 4h contract must not be quoted as if the inventory were held to settlement."""
        sigma_sq = 1e-5
        five_min = self.router.risk_term(sigma_sq, 300.0)
        four_hours = self.router.risk_term(sigma_sq, 14400.0)
        self.assertAlmostEqual(five_min, four_hours,
                               msg="horizon not capped: 4h quotes would be ~48x wider")

    def test_spread_capped_on_variance_spike(self):
        from src.execution.engine import MAX_HALF_SPREAD
        self.assertLessEqual(self.router.calculate_spread(1.0, 300.0), MAX_HALF_SPREAD)

    # ── liquidity-aware inventory cap ────────────────────────────────────────────
    @staticmethod
    def _book(total_depth):
        """Book with total_depth resting within 3 ticks of the top on each side,
        plus a far level that must not be counted."""
        per = total_depth / 3.0
        bids = [(0.47, per), (0.46, per), (0.45, per), (0.30, 9999.0)]
        asks = [(0.49, per), (0.50, per), (0.51, per), (0.70, 9999.0)]
        return bids, asks

    def _cap_for(self, depth):
        self.router.exit_depth_ewma = None
        self.router.update_exit_depth(*self._book(depth))
        return self.router.effective_max_inventory()

    def test_cap_tracks_book_depth(self):
        """A thin book must cap inventory harder than a thick one."""
        thin, mid, thick = self._cap_for(260), self._cap_for(1000), self._cap_for(2500)
        self.assertLess(thin, mid)
        self.assertLess(mid, thick)
        self.assertAlmostEqual(mid, 350.0)          # 0.33 * 1000, quantised to half a clip

    def test_cap_respects_floor_and_ceiling(self):
        """Never below one clip (we could not quote), never above the configured ceiling."""
        self.assertEqual(self._cap_for(1.0), self.router.maker_size)
        self.assertEqual(self._cap_for(100_000.0), self.router.max_inventory)

    def test_cap_ignores_liquidity_beyond_the_band(self):
        """Only size within cap_depth_ticks of the top counts: the far level is 9999."""
        self.assertEqual(self._cap_for(900), 300.0)

    def test_cap_drops_fast_and_recovers_slowly(self):
        """Thinning is believed at once; thickening is not."""
        self.router.exit_depth_ewma = 1000.0
        self.router.update_exit_depth(*self._book(200))
        dropped = self.router.exit_depth_ewma

        self.router.exit_depth_ewma = 1000.0
        self.router.update_exit_depth(*self._book(3000))
        raised = self.router.exit_depth_ewma

        self.assertLess(1000.0 - dropped * 1.0, 1000.0)
        self.assertGreater(1000.0 - dropped, raised - 1000.0,
                           "cap must react to a thinning book faster than to a thickening one")

    def test_cap_is_quantised(self):
        """Tiny depth wobbles must not resize the quote: a REPLACE costs queue position."""
        self.assertEqual(self._cap_for(1000), self._cap_for(1010))

    def test_cap_resets_on_rollover(self):
        """The next cycle is a different contract with a different book."""
        self._cap_for(1000)
        self.assertIsNotNone(self.router.exit_depth_ewma)
        self.router.reset_active_orders()
        self.assertIsNone(self.router.exit_depth_ewma)
        self.assertEqual(self.router.effective_max_inventory(), self.router.max_inventory)

    def test_cap_can_be_disabled(self):
        """liquidity_fraction=0 keeps the old fixed behaviour."""
        self.router.liquidity_fraction = 0.0
        self.assertEqual(self._cap_for(100), self.router.max_inventory)

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
        
        # Test case 1: yes_shares = 0.0 (only bid is quoted, ask is cancelled/skipped)
        instructions = self.router.evaluate_regimes(
            p_hat=0.55,
            sigma_sq=0.25,
            context=context,
            yes_shares=0.0,
            no_shares=0.0,
            cash_balance=1000.0
        )
        
        self.assertEqual(len(instructions), 1)
        bid_instr = next(x for x in instructions if x.side == "BUY_YES")
        self.assertEqual(bid_instr.action, "NEW")
        
        # Test case 2: yes_shares = 100.0 (both bid and ask are quoted)
        self.router.reset_active_orders()
        instructions_with_shares = self.router.evaluate_regimes(
            p_hat=0.60,
            sigma_sq=0.25,
            context=context,
            yes_shares=100.0,
            no_shares=0.0,
            cash_balance=1000.0
        )
        self.assertEqual(len(instructions_with_shares), 2)
        bid_instr2 = next(x for x in instructions_with_shares if x.side == "BUY_YES")
        ask_instr2 = next(x for x in instructions_with_shares if x.side == "SELL_YES")
        self.assertEqual(bid_instr2.action, "NEW")
        self.assertEqual(ask_instr2.action, "NEW")

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
        # Verify ExecutionEngine orchestrates strategy, client and router correctly
        config = SystemConfig()
        strategy = MockStrategy(p_hat=0.58)
        client = MockClientForTest(cash=2000.0, yes=50.0, no=0.0)
        
        engine = ExecutionEngine(strategy, client, config)
        
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
        engine_none = ExecutionEngine(strategy_none, client, config)
        
        # Set active orders so they are cancelled
        engine_none.execution_router.active_bid_id = "bid_1"
        engine_none.execution_router.active_ask_id = "ask_1"
        
        instructions_none = engine_none.evaluate_and_route(context)
        self.assertEqual(len(instructions_none), 2)
        for instr in instructions_none:
            self.assertEqual(instr.action, "CANCEL")
            self.assertEqual(instr.regime, "SAFE")

    def test_mock_client_maker_execution(self):
        from src.execution.clients import MockExecutionClient
        from src.execution.shadow_book import ShadowOrderBook
        from src.logging.recorder import DataRecorder
        import asyncio

        config = SystemConfig()
        
        recorder = DataRecorder(
            base_log_dir=config.LOG_DIR,
            strategy_name="test_maker",
            run_id="test_run_maker",
            buffer_size=1
        )
        
        shadow_book = ShadowOrderBook()
        shadow_book.update_book(
            bid_updates=[(0.45, 1000.0)],
            ask_updates=[(0.47, 1000.0)],
            is_snapshot=True
        )
        
        client = MockExecutionClient(config, recorder, shadow_book)
        initial_cash = client.cash_balance
        
        context_state = {
            "timestamp": 1000.0,
            "strike_price": 60000.0,
            "volatility": 0.25
        }
        
        async def run_maker_trade():
            return await client._execute_trade_internal(
                side="BUY_YES",
                qty=100.0,
                price=0.46,
                ev=0.1,
                expected_slippage_bps=0.0,
                context_state=context_state,
                is_maker=True
            )
            
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(run_maker_trade())
        
        self.assertTrue(res["success"])
        self.assertEqual(res["qty"], 100.0)
        self.assertEqual(res["price"], 0.46)
        self.assertEqual(res["realized_slippage_bps"], 0.0)
        
        expected_cash = initial_cash - (100.0 * 0.46 + 0.03)
        self.assertAlmostEqual(client.cash_balance, expected_cash)
        self.assertEqual(client.get_position_size("YES"), 100.0)
        self.assertEqual(client.entry_prices["YES"], 0.46)
        
        asks = shadow_book.get_sorted_asks()
        self.assertEqual(asks[0][1], 1000.0)

if __name__ == '__main__':
    unittest.main()
