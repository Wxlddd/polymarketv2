import asyncio
import time
from config.settings import SystemConfig
from src.ingestion.market_manager import MarketManager
from src.ingestion.live_feeds import ChainlinkSpotFeed
from src.execution.shadow_book import ShadowOrderBook
from src.strategies.merton_strategy import MertonStrategy
from src.execution.clients import MockExecutionClient
from src.execution.engine import ExecutionEngine
from src.logging.recorder import DataRecorder
from src.ui.dashboard import run_terminal_dashboard

async def main():
    print("=== Phase 5 CLI Dashboard Verification ===")
    
    config = SystemConfig()
    recorder = DataRecorder(
        base_log_dir=config.LOG_DIR,
        strategy_name="merton_ui_test",
        run_id="test_run",
        buffer_size=10
    )
    
    # 1. Initialize components
    market_manager = MarketManager(config)
    spot_feed = ChainlinkSpotFeed(config)
    shadow_book = ShadowOrderBook()
    strategy = MertonStrategy(config)
    client = MockExecutionClient(config, recorder, shadow_book)
    engine = ExecutionEngine(strategy, client, config)
    
    # Pre-populate dummy states
    market_manager.current_expiry = int(time.time() + 150)
    market_manager.current_slug = "btc-updown-5m-test"
    market_manager.condition_id = "0xTestConditionID"
    market_manager.yes_token_id = "0xYesToken"
    market_manager.no_token_id = "0xNoToken"
    market_manager.strike_price = 68000.0
    
    spot_feed._price = 67950.50
    spot_feed._last_updated = time.time()
    
    # Populate order book
    bids = [(0.48, 500.0), (0.47, 1000.0)]
    asks = [(0.50, 600.0), (0.51, 1200.0)]
    shadow_book.update_book(bids, asks, is_snapshot=True)
    
    # Feed ticks to calibrate volatility
    for i in range(15):
        strategy.vol_calibrator.add_tick(67950.0 + i, time.time() - (15 - i) * 5)
        
    latest_decision_ref = [{"side": "HOLD", "size": 0.0, "vwap": 0.0, "reason": "TESTING_DASHBOARD"}]
    
    # Stop event for UI loop
    stop_event = asyncio.Event()
    
    # Launch UI task
    print("Launching terminal dashboard UI task for 3 seconds...")
    ui_task = asyncio.create_task(
        run_terminal_dashboard(
            config=config,
            market_manager=market_manager,
            spot_feed=spot_feed,
            shadow_book=shadow_book,
            strategy=strategy,
            client=client,
            engine=engine,
            latest_decision_ref=latest_decision_ref,
            stop_event=stop_event
        )
    )
    
    # Run loop updates
    for count in range(3):
        await asyncio.sleep(1.0)
        # Simulate some decision changes
        latest_decision_ref[0] = {
            "side": "BUY_YES" if count % 2 == 0 else "HOLD",
            "size": 150.0 * (count + 1),
            "vwap": 0.50,
            "reason": f"MOCK_DECISION_TICK_{count}"
        }
        
    print("Shutting down dashboard...")
    stop_event.set()
    await ui_task
    print("[OK] Dashboard UI verification finished successfully.")

if __name__ == "__main__":
    asyncio.run(main())
