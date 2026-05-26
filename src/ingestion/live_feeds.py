import asyncio
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import websockets
from config.settings import SystemConfig
from src.core.interfaces import ISpotFeed

logger = logging.getLogger("LiveFeeds")

class ChainlinkSpotFeed(ISpotFeed):
    """
    Connects to Polymarket's internal WebSocket spot price feed
    (relay of Chainlink Oracles) to fetch the real-time settlement price.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.ticker = config.TICKER.upper()
        self.wss_url = config.polymarket.INTERNAL_WSS_URL
        
        # Parse payload
        payload_str = config.polymarket.INTERNAL_SUBSCRIBE_PAYLOAD
        try:
            self.subscribe_payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        except Exception as e:
            logger.error(f"[{self.ticker} SpotFeed] Failed to parse subscribe payload: {e}")
            self.subscribe_payload = None

        symbol = f"{self.ticker.lower()}/usd"
        if not self.subscribe_payload or (isinstance(self.subscribe_payload, dict) and self.subscribe_payload.get("type") == "subscribe"):
            logger.info(f"[{self.ticker} SpotFeed] Upgrading to subscription format for symbol: {symbol}")
            self.subscribe_payload = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                        "filters": json.dumps({"symbol": symbol}, separators=(',', ':'))
                    }
                ]
            }
            
        self._price: Optional[float] = None
        self._last_updated: float = 0.0
        self._is_connected = False
        self._is_running = False
        
        # Tick cache: list of (timestamp_seconds, price)
        self.ticks: List[Tuple[float, float]] = []
        self.max_ticks_age_sec = 1200  # Keep 20 minutes of ticks
        
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def price(self) -> Optional[float]:
        return self._price

    @property
    def last_updated(self) -> float:
        return self._last_updated

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def start(self) -> None:
        self._is_running = True
        self._task = asyncio.create_task(self._connect_loop())
        logger.info(f"[{self.ticker} SpotFeed] Started background listener.")

    async def stop(self) -> None:
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._is_connected = False
        logger.info(f"[{self.ticker} SpotFeed] Stopped background listener.")

    async def _connect_loop(self) -> None:
        base_delay = 1.0
        max_delay = 60.0
        delay = base_delay

        while self._is_running:
            try:
                logger.info(f"[{self.ticker} SpotFeed] Connecting to {self.wss_url}")
                async with websockets.connect(self.wss_url) as ws:
                    self._is_connected = True
                    delay = base_delay  # Reset delay on success
                    
                    # Send subscription
                    await ws.send(json.dumps(self.subscribe_payload))
                    logger.info(f"[{self.ticker} SpotFeed] Subscribed with payload: {self.subscribe_payload}")
                    
                    while self._is_running:
                        msg = await ws.recv()
                        await self._process_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._is_connected = False
                logger.error(f"[{self.ticker} SpotFeed] Connection lost: {e}. Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)

    async def _process_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            payload = data.get("payload", {})
            if isinstance(payload, dict):
                # Historical tick snapshot on connection
                if "data" in payload and isinstance(payload["data"], list):
                    async with self._lock:
                        for tick in payload["data"]:
                            val = tick.get("value")
                            ts_ms = tick.get("timestamp")
                            if val is not None and ts_ms is not None:
                                price = float(val)
                                ts = float(ts_ms) / 1000.0
                                self.ticks.append((ts, price))
                                self._price = price
                                self._last_updated = time.time()
                        self.ticks.sort(key=lambda x: x[0])
                        self._prune_ticks()
                    return

                symbol = payload.get("symbol")
                target_symbol = f"{self.ticker.lower()}/usd"
                if symbol == target_symbol:
                    val = payload.get("value")
                    ts_ms = payload.get("timestamp")
                    if val is not None and ts_ms is not None:
                        price = float(val)
                        ts = float(ts_ms) / 1000.0
                        
                        async with self._lock:
                            self._price = price
                            self._last_updated = time.time()
                            self.ticks.append((ts, price))
                            self._prune_ticks()
                        logger.debug(f"[{self.ticker} SpotFeed] Spot Price Update: ${price:,.4f}")
        except Exception as e:
            logger.warning(f"[{self.ticker} SpotFeed] Error parsing message: {e}")

    def _prune_ticks(self) -> None:
        if not self.ticks:
            return
        cutoff = self.ticks[-1][0] - self.max_ticks_age_sec
        self.ticks = [t for t in self.ticks if t[0] >= cutoff]


class ClobOrderBookFeed:
    """
    Connects to Polymarket CLOB WebSocket to fetch real-time
    order book updates for YES/NO tokens.
    """
    
    def __init__(self, config: SystemConfig, book_callback):
        self.config = config
        self.wss_url = config.polymarket.WS_URL
        self.yes_token = config.polymarket.YES_TOKEN_ID
        self.no_token = config.polymarket.NO_TOKEN_ID
        self.book_callback = book_callback
        
        self._is_connected = False
        self._is_running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._is_running = True
        self._task = asyncio.create_task(self._connect_loop())
        logger.info("[CLOB Feed] Started background listener.")

    async def stop(self) -> None:
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._is_connected = False
        logger.info("[CLOB Feed] Stopped background listener.")

    async def _connect_loop(self) -> None:
        base_delay = 1.0
        max_delay = 60.0
        delay = base_delay

        while self._is_running:
            try:
                logger.info(f"[CLOB Feed] Connecting to {self.wss_url}")
                async with websockets.connect(self.wss_url) as ws:
                    self._is_connected = True
                    delay = base_delay
                    
                    # Subscribe to L2 books for YES and NO
                    sub_message = {
                        "type": "market",
                        "assets_ids": [self.yes_token, self.no_token]
                    }
                    await ws.send(json.dumps(sub_message))
                    logger.info("[CLOB Feed] Subscribed to YES/NO books.")
                    
                    while self._is_running:
                        msg = await ws.recv()
                        await self._process_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._is_connected = False
                logger.error(f"[CLOB Feed] Connection lost: {e}. Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)

    async def _process_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            events = data if isinstance(data, list) else [data]
            
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                
                # 1. Snapshot book event
                if "bids" in ev or "asks" in ev or ev.get("event_type") == "book":
                    asset_id = ev.get("asset_id") or ev.get("token_id")
                    if asset_id == self.yes_token:
                        bids = []
                        for b in ev.get("bids", []):
                            if isinstance(b, dict):
                                bids.append((float(b.get("price", b.get("p", 0.0))), float(b.get("size", b.get("qty", 0.0)))))
                            else:
                                bids.append((float(b[0]), float(b[1])))
                        
                        asks = []
                        for a in ev.get("asks", []):
                            if isinstance(a, dict):
                                asks.append((float(a.get("price", a.get("p", 0.0))), float(a.get("size", a.get("qty", 0.0)))))
                            else:
                                asks.append((float(a[0]), float(a[1])))
                                
                        self.book_callback(bids, asks, is_snapshot=True)
                
                # 2. Incremental L2 delta updates
                elif ev.get("event_type") == "price_change":
                    price_changes = ev.get("price_changes", [])
                    yes_bids = []
                    yes_asks = []
                    
                    for pc in price_changes:
                        asset_id = pc.get("asset_id")
                        if asset_id == self.yes_token:
                            price = float(pc.get("price"))
                            qty = float(pc.get("size"))
                            side = pc.get("side")
                            if side == "BUY":
                                yes_bids.append((price, qty))
                            elif side == "SELL":
                                yes_asks.append((price, qty))
                                
                    if yes_bids or yes_asks:
                        self.book_callback(yes_bids, yes_asks, is_snapshot=False)
        except Exception as e:
            logger.warning(f"[CLOB Feed] Error processing message: {e}", exc_info=True)
