import unittest
import asyncio
import time
from typing import Dict, Any
from config.settings import SystemConfig
from src.core.market_context import MarketContext
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.execution.shadow_book import ShadowOrderBook
from src.logging.recorder import DataRecorder

class TestPositionCapEnforcement(unittest.TestCase):
    
    def setUp(self):
        # Override SystemConfig defaults for testing
        self.config = SystemConfig()
        
        # Setup test-specific config values via environment or dictionary if editable,
        # but config is frozen. We can construct an ExecutionEngine and mock client,
        # and manually patch self.config parameters or configure via environment.
        # Since config properties are read from env vars, let's verify if we can patch them.
        # Let's write a helper to create configs with overridden properties if possible,
        # or since the config class is frozen we can use patch or mock.
        
    def get_custom_config(self, max_pos_usd: float = 250.0, panic_concession: float = 0.15) -> SystemConfig:
        # We can write a quick custom class subclassing or mocking,
        # or we can construct SystemConfig by patching environment variables first.
        import os
        os.environ["MAX_POSITION_SIZE_USD"] = str(max_pos_usd)
        os.environ["PANIC_CONCESSION"] = str(panic_concession)
        # Re-instantiate config so it reads new env values
        return SystemConfig()

    def test_headroom_cap_buy_yes(self):
        config = self.get_custom_config(max_pos_usd=250.0, panic_concession=0.15)
        engine = ExecutionEngine(config)
        
        recorder = DataRecorder(
            base_log_dir=config.LOG_DIR,
            strategy_name="test_cap",
            run_id="test_run",
            buffer_size=10
        )
        shadow_book = ShadowOrderBook()
        # Set a massive book so liquidity isn't the limiting factor
        shadow_book.update_book(
            bid_updates=[(0.48, 1000.0)],
            ask_updates=[(0.50, 1000.0)],
            is_snapshot=True
        )
        
        # Test case 1: YES position is 0 -> allowed increment should be up to cap
        client = MockExecutionClient(config, recorder, shadow_book)
        client.positions["YES"] = 0.0
        
        context = MarketContext(
            timestamp=time.time(),
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.25,
            ofi=0.0,
            bids_l2=shadow_book.get_sorted_bids(),
            asks_l2=shadow_book.get_sorted_asks()
        )
        
        # Model probability is 0.70 (huge edge over 0.50 ask, Kelly would size large)
        decision = engine.evaluate_and_trade(p_yes=0.70, context=context, client=client)
        self.assertEqual(decision["side"], "BUY_YES")
        # Position cap limit in units = 250 USD / 0.50 = 500 contracts
        self.assertLessEqual(decision["size"], 500.0)
        
        # Test case 2: YES position is already 400 -> allowed increment should be capped at 100
        client.positions["YES"] = 400.0
        client.entry_prices["YES"] = 0.50
        decision = engine.evaluate_and_trade(p_yes=0.70, context=context, client=client)
        self.assertEqual(decision["side"], "BUY_YES")
        # Remaining headroom is 500 - 400 = 100 contracts
        self.assertLessEqual(decision["size"], 100.1)  # small delta allowed for float precision
        
        # Test case 3: YES position is already 500 -> no buying allowed
        client.positions["YES"] = 500.0
        client.entry_prices["YES"] = 0.50
        decision = engine.evaluate_and_trade(p_yes=0.70, context=context, client=client)
        self.assertEqual(decision["side"], "HOLD")

    def test_headroom_cap_buy_no(self):
        config = self.get_custom_config(max_pos_usd=250.0, panic_concession=0.15)
        engine = ExecutionEngine(config)
        
        recorder = DataRecorder(
            base_log_dir=config.LOG_DIR,
            strategy_name="test_cap_no",
            run_id="test_run",
            buffer_size=10
        )
        shadow_book = ShadowOrderBook()
        # Set asks/bids on YES. If we buy NO, we sell YES (implied price of NO is 1 - YES price)
        # If YES bid is 0.50, then YES ask is 0.52. Implied NO ask is 1 - 0.50 = 0.50.
        shadow_book.update_book(
            bid_updates=[(0.50, 1000.0)],
            ask_updates=[(0.52, 1000.0)],
            is_snapshot=True
        )
        
        # Test case 1: NO position is 0 -> allowed increment should be up to cap
        client = MockExecutionClient(config, recorder, shadow_book)
        client.positions["NO"] = 0.0
        
        context = MarketContext(
            timestamp=time.time(),
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=150.0,
            volatility=0.25,
            ofi=0.0,
            bids_l2=shadow_book.get_sorted_bids(),
            asks_l2=shadow_book.get_sorted_asks()
        )
        
        # Model probability YES = 0.20 (so model prob NO = 0.80)
        # Implied ask NO = 1.0 - best_bid_yes = 1.0 - 0.50 = 0.50
        # Edge on NO is 0.80 - 0.50 = 0.30 (huge edge, Kelly would size large)
        decision = engine.evaluate_and_trade(p_yes=0.20, context=context, client=client)
        self.assertEqual(decision["side"], "BUY_NO")
        # Position cap limit in units = 250 USD / 0.50 = 500 contracts
        self.assertLessEqual(decision["size"], 500.0)
        
        # Test case 2: NO position is already 400 -> allowed increment should be capped at 100
        client.positions["NO"] = 400.0
        client.entry_prices["NO"] = 0.50
        decision = engine.evaluate_and_trade(p_yes=0.20, context=context, client=client)
        self.assertEqual(decision["side"], "BUY_NO")
        # Remaining headroom is 500 - 400 = 100 contracts
        self.assertLessEqual(decision["size"], 100.1)
        
        # Test case 3: NO position is already 500 -> no buying allowed
        client.positions["NO"] = 500.0
        client.entry_prices["NO"] = 0.50
        decision = engine.evaluate_and_trade(p_yes=0.20, context=context, client=client)
        self.assertEqual(decision["side"], "HOLD")

    def test_panic_concession_price_protection(self):
        config = self.get_custom_config(max_pos_usd=250.0, panic_concession=0.15)
        engine = ExecutionEngine(config)
        
        recorder = DataRecorder(
            base_log_dir=config.LOG_DIR,
            strategy_name="test_panic",
            run_id="test_run",
            buffer_size=10
        )
        
        # Test Case 1: Panic unwind with terrible bids -> should reject fill
        shadow_book_bad = ShadowOrderBook()
        # Bids are highly depleted/low: e.g. best bid is 0.10.
        shadow_book_bad.update_book(
            bid_updates=[(0.10, 1000.0)],
            ask_updates=[(0.55, 1000.0)],
            is_snapshot=True
        )
        
        client = MockExecutionClient(config, recorder, shadow_book_bad)
        client.positions["YES"] = 200.0
        client.entry_prices["YES"] = 0.50
        
        context = MarketContext(
            timestamp=time.time(),
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=10.0, # <= 15s triggers panic mode!
            volatility=0.25,
            ofi=0.0,
            bids_l2=shadow_book_bad.get_sorted_bids(),
            asks_l2=shadow_book_bad.get_sorted_asks()
        )
        
        # Model probability is 0.50. Concession is 0.15.
        # Target limit price for selling YES should be p_yes - concession = 0.50 - 0.15 = 0.35.
        # Bids are at 0.10, which is < 0.35, so no levels should match.
        decision = engine.evaluate_and_trade(p_yes=0.50, context=context, client=client)
        self.assertEqual(decision["side"], "SELL_YES")
        self.assertEqual(decision["limit_price"], 0.35)
        
        # Now run the execution and check that it returns failure / insufficient fill
        async def run_exec():
            return await client.execute_trade(
                side=decision["side"],
                qty=decision["size"],
                price=decision["vwap"],
                ev=decision["ev"],
                expected_slippage_bps=decision["expected_slippage_bps"],
                context_state={"timestamp": context.timestamp, "strike_price": context.strike_price, "limit_price": decision["limit_price"]}
            )
            
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_exec())
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "IOC_FILL_TOO_SMALL")
        
        # Test Case 2: Panic unwind with acceptable bids -> should fill
        shadow_book_good = ShadowOrderBook()
        # Bids are acceptable: e.g. best bid is 0.40 (which is >= 0.35)
        shadow_book_good.update_book(
            bid_updates=[(0.40, 1000.0)],
            ask_updates=[(0.55, 1000.0)],
            is_snapshot=True
        )
        
        client_good = MockExecutionClient(config, recorder, shadow_book_good)
        client_good.positions["YES"] = 200.0
        client_good.entry_prices["YES"] = 0.50
        
        context_good = MarketContext(
            timestamp=time.time(),
            spot_price=67500.0,
            strike_price=67500.0,
            tau_seconds=10.0,
            volatility=0.25,
            ofi=0.0,
            bids_l2=shadow_book_good.get_sorted_bids(),
            asks_l2=shadow_book_good.get_sorted_asks()
        )
        
        decision_good = engine.evaluate_and_trade(p_yes=0.50, context=context_good, client=client_good)
        self.assertEqual(decision_good["side"], "SELL_YES")
        self.assertEqual(decision_good["limit_price"], 0.35)
        
        async def run_exec_good():
            return await client_good.execute_trade(
                side=decision_good["side"],
                qty=decision_good["size"],
                price=decision_good["vwap"],
                ev=decision_good["ev"],
                expected_slippage_bps=decision_good["expected_slippage_bps"],
                context_state={"timestamp": context_good.timestamp, "strike_price": context_good.strike_price, "limit_price": decision_good["limit_price"]}
            )
            
        result_good = loop.run_until_complete(run_exec_good())
        self.assertTrue(result_good["success"])
        self.assertEqual(result_good["qty"], 200.0)
        self.assertEqual(result_good["price"], 0.40)

if __name__ == "__main__":
    unittest.main()
