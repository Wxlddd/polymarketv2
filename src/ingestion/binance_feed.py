"""Binance spot feed, used to nowcast the Chainlink oracle.

Measured on recorded sessions: Binance BTCUSDT leads the Chainlink oracle by ~4s
(correlation 0.31-0.39) and the Polymarket book by 1-2s, so the book moves BEFORE the
oracle prints. A fair value built on the last oracle print is therefore the last link in
the chain, which is what the -0.021 one-second markout on maker fills was measuring.

Chainlink still settles the contract, so it stays the target. This feed only supplies the
move that has already happened on the exchange but has not reached the oracle yet:

    nowcast = last_oracle_price * (binance_now / binance_at_that_oracle_print)

With no Binance move since the print the nowcast is exactly the oracle, so the feed can
only add information, never invent a level of its own.
"""
import asyncio
import json
import logging
import time
from collections import deque
from typing import Deque, Optional, Tuple

import websockets

logger = logging.getLogger("BinanceFeed")


class BinanceSpotFeed:
    """Best bid/ask midpoint from Binance's public bookTicker stream. No API key needed."""

    def __init__(self, config):
        cfg = config.binance
        self.symbol = cfg.SYMBOL.lower()
        self.ws_url = f"{cfg.WS_URL.rstrip('/')}/{self.symbol}@bookTicker"
        self.max_age_sec = cfg.MAX_AGE_SEC

        self._price: Optional[float] = None
        self._last_updated: float = 0.0
        self._is_connected = False
        self._is_running = False
        self._task: Optional[asyncio.Task] = None

        # (timestamp, mid) kept just long enough to look back to the last oracle print
        self._ticks: Deque[Tuple[float, float]] = deque(maxlen=20000)

    @property
    def price(self) -> Optional[float]:
        return self._price

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_fresh(self) -> bool:
        """False when the stream is down or stalled: callers must fall back to the oracle."""
        return (self._price is not None
                and self._is_connected
                and (time.time() - self._last_updated) <= self.max_age_sec)

    async def start(self) -> None:
        self._is_running = True
        self._task = asyncio.create_task(self._connect_loop())
        logger.info(f"[BinanceFeed] Started listener for {self.symbol}@bookTicker.")

    async def stop(self) -> None:
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._is_connected = False
        logger.info("[BinanceFeed] Stopped listener.")

    async def _connect_loop(self) -> None:
        delay, max_delay = 1.0, 60.0
        while self._is_running:
            try:
                logger.info(f"[BinanceFeed] Connecting to {self.ws_url}")
                async with websockets.connect(self.ws_url, open_timeout=10,
                                              ping_interval=15, ping_timeout=15) as ws:
                    self._is_connected = True
                    delay = 1.0
                    while self._is_running:
                        self._handle(await ws.recv())
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._is_connected = False
                logger.warning(f"[BinanceFeed] Connection lost: {e}. Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)

    def _handle(self, message: str) -> None:
        try:
            d = json.loads(message)
            bid, ask = float(d["b"]), float(d["a"])
        except (ValueError, KeyError, TypeError):
            return
        if bid <= 0.0 or ask <= 0.0:
            return
        now = time.time()
        self._price = 0.5 * (bid + ask)
        self._last_updated = now
        self._ticks.append((now, self._price))

    def price_at(self, timestamp: float) -> Optional[float]:
        """Last Binance mid at or before `timestamp`."""
        best = None
        for ts, px in self._ticks:
            if ts <= timestamp:
                best = px
            else:
                break
        return best

    def nowcast(self, oracle_price: float, oracle_timestamp: float) -> float:
        """The oracle price carried forward by the Binance move since that print.

        Returns `oracle_price` unchanged when the stream is stale or there is no Binance
        observation from the time of the print, so a broken feed degrades to today's
        behaviour instead of trading on a guess.
        """
        if not self.is_fresh or oracle_price is None or oracle_price <= 0.0:
            return oracle_price
        ref = self.price_at(oracle_timestamp)
        if ref is None or ref <= 0.0:
            return oracle_price
        return oracle_price * (self._price / ref)
