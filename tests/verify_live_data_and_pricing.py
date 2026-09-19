import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Optional

from config.settings import SystemConfig
from src.ingestion.market_manager import MarketManager
from src.ingestion.live_feeds import ChainlinkSpotFeed, ClobOrderBookFeed
from src.core.strike_manager import StrikeManager
from src.core.market_context import MarketContext
from src.strategies.merton_strategy import MertonStrategy
from src.execution.shadow_book import ShadowOrderBook

# Configure minimalist logging to file so it doesn't pollute terminal output
_LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(os.path.join(_LOG_DIR, "verify_live_data.log"), encoding="utf-8")]
)
logger = logging.getLogger("VerifyLiveDataAndPricing")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class LiveDataVerifier:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.market_manager = MarketManager(config)
        self.spot_feed = ChainlinkSpotFeed(config)
        self.shadow_book = ShadowOrderBook()
        self.strategy = MertonStrategy(config)
        
        self.strike_manager: Optional[StrikeManager] = None
        self.clob_feed: Optional[ClobOrderBookFeed] = None
        
        self.tick_count = 0
        self.max_ticks = 15  # Collect 15 ticks and exit
        self.tick_data_rows: List[dict] = []
        self.stop_event = asyncio.Event()

    async def run(self):
        print("=== Polymarket V2: Live Data & Pricing Verifier ===")
        print("1. Starting Spot Feed...")
        await self.spot_feed.start()
        
        print("Waiting for initial Spot price from WebSocket...")
        for _ in range(20):
            if self.spot_feed.price is not None:
                break
            await asyncio.sleep(0.5)
            
        if self.spot_feed.price is None:
            print("[FAIL] Could not retrieve spot price from feed. Is the WebSocket feed online?")
            await self.spot_feed.stop()
            return False
            
        print(f"[OK] Received spot price: ${self.spot_feed.price:,.2f}")
        
        print("\n2. Querying Gamma API for Active 5m Option Cycle...")
        t_now = time.time()
        await self.market_manager.update_market_cycle(t_now)
        
        cycle_start_time = self.market_manager.current_expiry - 300
        if t_now - cycle_start_time > 10.0:
            print("Ok, aspetto il prossimo ciclo...")
            print(f"Current time: {datetime.fromtimestamp(t_now).strftime('%Y-%m-%d %H:%M:%S')} (mid-cycle). Waiting for rollover at {datetime.fromtimestamp(self.market_manager.current_expiry).strftime('%H:%M:%S')}...")
            while time.time() < self.market_manager.current_expiry:
                await asyncio.sleep(1.0)
            
            # Now update to the new cycle
            t_now = time.time()
            await self.market_manager.update_market_cycle(t_now)
            print("\nRollover reached! Running Market Discovery for the new cycle...")
            
        print(f"  - Active Slug:  {self.market_manager.current_slug}")
        print(f"  - Expiry Time:  {datetime.fromtimestamp(self.market_manager.current_expiry).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - YES Token ID: {self.market_manager.yes_token_id}")
        print(f"  - NO Token ID:  {self.market_manager.no_token_id}")
        print(f"  - Strike Price: {self.market_manager.strike_price}")
        print(f"  - Condition ID: {self.market_manager.condition_id}")
        
        if not self.market_manager.yes_token_id or not self.market_manager.no_token_id:
            print("[FAIL] Missing Token IDs from market discovery. Cannot proceed.")
            await self.spot_feed.stop()
            return False
            
        print("[OK] Market Discovery successful.")
        
        # Instantiate Strike Manager
        self.strike_manager = StrikeManager(
            presumed_strike=self.market_manager.strike_price,
            expiration_timestamp=self.market_manager.current_expiry
        )
        
        # Inject dynamic tokens into configuration so CLOB feed uses them
        self.config.polymarket.__dict__["YES_TOKEN_ID"] = self.market_manager.yes_token_id
        self.config.polymarket.__dict__["NO_TOKEN_ID"] = self.market_manager.no_token_id
        
        print("\n3. Connecting to CLOB Order Book WebSocket...")
        self.clob_feed = ClobOrderBookFeed(self.config, self._clob_callback)
        await self.clob_feed.start()
        
        print("Waiting for CLOB data stream and Merton computations...")
        print("Printing data table below. Will auto-exit after 15 updates.")
        print("-" * 105)
        
        # Simple Terminal Live view using Rich
        if RICH_AVAILABLE:
            console = Console()
            with Live(self._generate_table(), console=console, refresh_per_second=2) as live:
                while not self.stop_event.is_set() and self.tick_count < self.max_ticks:
                    await asyncio.sleep(0.2)
                    live.update(self._generate_table())
        else:
            print("Timestamp           | Spot Price   | Strike K     | YES          | Market YES   | OFI   | Volatility")
            print("-" * 105)
            while not self.stop_event.is_set() and self.tick_count < self.max_ticks:
                await asyncio.sleep(0.5)
                
        # Cleanup
        await self.clob_feed.stop()
        await self.spot_feed.stop()
        print("\n=== Live Data verification complete ===")
        return True

    def _clob_callback(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], is_snapshot: bool) -> None:
        if self.spot_feed.price is None or self.strike_manager is None or self.stop_event.is_set():
            return
            
        t_now = time.time()
        spot = self.spot_feed.price
        
        # Reconcile L2 Order Book
        ofi = self.shadow_book.update_book(bids, asks, is_snapshot)
        
        # Resolve active Strike Price only after the cycle has actually started
        if self.strike_manager and (self.strike_manager.presumed_strike is None or self.strike_manager.presumed_strike == 0.0):
            if self.market_manager.current_expiry is not None:
                cycle_start_time = self.market_manager.current_expiry - 300
                if t_now >= cycle_start_time:
                    strike_tick = self.spot_feed.get_first_tick_after(cycle_start_time)
                    if strike_tick is not None:
                        _, strike_price_val = strike_tick
                        self.strike_manager.presumed_strike = strike_price_val
                    
        active_strike = self.strike_manager.get_strike(t_now, spot)
        tau_sec = max(0.0, self.market_manager.current_expiry - t_now)
        
        # Update Volatility
        self.strategy.vol_calibrator.add_tick(spot, t_now)
        vol = self.strategy.vol_calibrator.calculate_volatility(self.config.merton.DEFAULT_SIGMA)
        
        # Build Context & Price Merton option
        context = MarketContext(
            timestamp=t_now,
            spot_price=spot,
            strike_price=active_strike,
            tau_seconds=tau_sec,
            volatility=vol,
            ofi=ofi,
            bids_l2=self.shadow_book.get_sorted_bids(),
            asks_l2=self.shadow_book.get_sorted_asks()
        )
        
        p_yes = self.strategy.get_probability(context)
        
        # Get Implied Market price
        top_b, top_a = self.shadow_book.get_top_of_book()
        p_mkt = 0.5 * (top_b[0] + top_a[0]) if top_b and top_a else None
        
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "spot": spot,
            "strike": active_strike,
            "p_yes": p_yes,
            "p_mkt": p_mkt,
            "ofi": ofi,
            "vol": vol
        }
        
        self.tick_data_rows.append(row)
        self.tick_count += 1
        
        if not RICH_AVAILABLE:
            p_yes_str = f"{p_yes*100:.2f}%" if p_yes is not None else "—"
            p_mkt_str = f"{p_mkt*100:.2f}%" if p_mkt is not None else "—"
            strike_str = f"${active_strike:,.2f}" if active_strike is not None else "None"
            print(f"{row['timestamp']} | ${spot:,.2f}  | {strike_str:<12} | {p_yes_str:<12} | {p_mkt_str:<12} | {ofi:+.1f} | {vol:.2%}")
            
        if self.tick_count >= self.max_ticks:
            self.stop_event.set()

    def _generate_table(self) -> Table:
        table = Table(title="Live Tick & Pricing Update Logs", border_style="bright_black")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Spot Price", justify="right", style="gold1")
        table.add_column("Strike K", justify="right", style="white")
        table.add_column("YES", justify="right", style="bold green")
        table.add_column("Market YES", justify="right", style="magenta")
        table.add_column("OFI", justify="right")
        table.add_column("Volatility", justify="right", style="dim")
        
        # Show last 10 rows
        for r in self.tick_data_rows[-10:]:
            p_yes_str = f"{r['p_yes']*100:.2f}%" if r['p_yes'] is not None else "—"
            p_mkt_str = f"{r['p_mkt']*100:.2f}%" if r['p_mkt'] is not None else "—"
            strike_str = f"${r['strike']:,.2f}" if r['strike'] is not None else "[yellow]Wait rollover...[/]"
            
            table.add_row(
                r["timestamp"],
                f"${r['spot']:,.2f}",
                strike_str,
                p_yes_str,
                p_mkt_str,
                f"{r['ofi']:+.1f}",
                f"{r['vol']:.2%}"
            )
        return table

async def main():
    config = SystemConfig()
    verifier = LiveDataVerifier(config)
    
    success = await verifier.run()
    if success:
        print("\n[SUCCESS] Integration verification successful. Prices, strikes and Merton probability parsed cleanly.")
    else:
        print("\n[ERROR] Integration verification encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting verifier...")
        sys.exit(0)
