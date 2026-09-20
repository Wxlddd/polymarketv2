import asyncio
import json
import logging
import time
import aiohttp
from typing import Dict, Any, Optional, Tuple, List
from config.settings import SystemConfig
from src.core.market_context import MarketContext

logger = logging.getLogger("MarketManager")

class MarketManager:
    """
    Orchestrates the dynamic discovery of Polymarket fixed-length Up/Down cycle markets
    (e.g. 5-minute or 4-hour cycles, configured via CYCLE_DURATION_SEC / MARKET_SLUG_TYPE).
    Generates deterministic slugs, fetches metadata via the Gamma API,
    and handles rollovers when expiration milestones are crossed.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.ticker = config.TICKER.lower()
        self.cycle_duration_sec = config.polymarket.CYCLE_DURATION_SEC
        self.slug_type = config.polymarket.MARKET_SLUG_TYPE
        self.preempt_sec = config.polymarket.ROLLOVER_PREEMPT_SEC

        self.current_expiry: Optional[int] = None
        self.current_slug: Optional[str] = None

        # Extracted active market details
        self.condition_id: Optional[str] = None
        self.yes_token_id: Optional[str] = None
        self.no_token_id: Optional[str] = None
        self.strike_price: Optional[float] = None

        self._lock = asyncio.Lock()

    def get_next_expiry(self, current_time: float) -> int:
        """
        Calculates the UNIX expiration timestamp of the next cycle window using
        modular arithmetic. Expirations are multiples of CYCLE_DURATION_SEC
        (aligned to the Unix epoch, matching Polymarket's own rollover clock).
        Preempts the rollover by 15 seconds so the bot subscribes to the next cycle early.
        """
        cycle = self.cycle_duration_sec
        t_int = int(current_time + self.preempt_sec)
        return t_int - (t_int % cycle) + cycle

    def get_slug_for_expiry(self, expiry: int) -> str:
        """Generates the deterministic Polymarket event slug for the target expiration."""
        # Polymarket event slugs use the start time of the cycle (expiry - cycle_duration)
        return f"{self.ticker}-updown-{self.slug_type}-{expiry - self.cycle_duration_sec}"

    async def update_market_cycle(self, current_time: float) -> bool:
        """
        Evaluates the current time against the active expiry. If a rollover occurs,
        recalculates the new expiry slug, queries the Gamma API, and updates in-memory states.
        
        Returns True if a new cycle was successfully resolved, False otherwise.
        """
        next_expiry = self.get_next_expiry(current_time)
        
        if self.current_expiry is None or next_expiry != self.current_expiry:
            async with self._lock:
                self.current_expiry = next_expiry
                self.current_slug = self.get_slug_for_expiry(next_expiry)
                logger.info(f"[MarketManager] Rollover detected. New active cycle expiry: {self.current_expiry} | Slug: {self.current_slug}")
                
                # Fetch metadata for the new cycle
                await self.fetch_market_context()
                return True
        return False

    async def fetch_market_context(self) -> None:
        """
        Queries the public Gamma API to extract market details for the active slug.
        Traverses JSON nodes to locate active yes/no tokens, condition ID, and strike price.
        """
        if not self.current_slug:
            return
            
        url = f"https://gamma-api.polymarket.com/events?slug={self.current_slug}"
        logger.info(f"[MarketManager] Querying Gamma API events: {url}")
        
        # Reset current extraction states to avoid stale values
        self.condition_id = None
        self.yes_token_id = None
        self.no_token_id = None
        self.strike_price = None
        
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                try:
                    async with session.get(url, headers=headers, timeout=10.0) as resp:
                        if resp.status == 200:
                            events = await resp.json()
                            # Gamma events can return a single dict or a list of event dicts
                            if isinstance(events, dict):
                                events = [events]
                                
                            if not events or not isinstance(events, list):
                                logger.warning(f"[MarketManager] Empty or invalid response format from Gamma API for slug {self.current_slug}")
                                return
                                
                            for event in events:
                                markets = event.get("markets", [])
                                if not markets:
                                    continue
                                    
                                for m in markets:
                                    # We only focus on Yes/No binary LOB pairs (has clobTokenIds)
                                    clob_tokens_raw = m.get("clobTokenIds")
                                    if clob_tokens_raw:
                                        if isinstance(clob_tokens_raw, str):
                                            clob_tokens = json.loads(clob_tokens_raw)
                                        else:
                                            clob_tokens = clob_tokens_raw
                                            
                                        if isinstance(clob_tokens, list) and len(clob_tokens) >= 2:
                                            self.yes_token_id = clob_tokens[0]
                                            self.no_token_id = clob_tokens[1]
                                            self.condition_id = m.get("conditionId")
                                            
                                            # Extract priceToBeat (Strike K)
                                            for field in ("priceToBeat", "price_to_beat", "strikePrice", "strike_price"):
                                                val = m.get(field)
                                                if val is not None:
                                                    try:
                                                        self.strike_price = float(val)
                                                        logger.info(f"[MarketManager] Extracted Strike price (K) from Gamma API: ${self.strike_price:,.2f}")
                                                        break
                                                    except (ValueError, TypeError):
                                                        pass
                                            
                                            logger.info(
                                                f"[MarketManager] Successfully resolved active market parameters:\n"
                                                f"  - Condition ID: {self.condition_id}\n"
                                                f"  - YES Token: {self.yes_token_id}\n"
                                                f"  - NO Token: {self.no_token_id}\n"
                                                f"  - Strike K: {self.strike_price}"
                                            )
                                            return
                            
                            logger.warning(f"[MarketManager] No matching market structure found in events list for slug {self.current_slug}")
                            break  # Successful request but no matching market structure - don't retry
                        else:
                            logger.warning(f"[MarketManager] Gamma API returned HTTP status {resp.status} for slug {self.current_slug} (Attempt {attempt+1}/{max_retries})")
                except Exception as e:
                    logger.error(f"[MarketManager] Failed to query Gamma API (Attempt {attempt+1}/{max_retries}): {e}")
                    
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                
        # If strike price is not resolved from REST, it remains None (which forces waiting for rollover spot tick)
        if self.strike_price is None:
            logger.warning(f"[MarketManager] Strike price K could not be resolved from REST. Engine will wait for rollover spot tick.")
