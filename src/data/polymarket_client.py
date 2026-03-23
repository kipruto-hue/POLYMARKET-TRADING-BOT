"""Polymarket Gamma + CLOB client wrappers."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
import websockets
from websockets.asyncio.client import connect as ws_connect

from src.config.settings import get_settings

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Market:
    token_id: str
    condition_id: str
    question: str
    slug: str
    active: bool
    liquidity: float
    volume_24h: float
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_price: float = 0.0
    neg_risk: bool = False
    tick_size: str = "0.01"


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class OrderBookSnapshot:
    token_id: str
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 1.0

    @property
    def bid_volume(self) -> float:
        return sum(sz for _, sz in self.bids)

    @property
    def ask_volume(self) -> float:
        return sum(sz for _, sz in self.asks)

    @property
    def imbalance(self) -> float:
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total


# ---------------------------------------------------------------------------
# Gamma API (market discovery)
# ---------------------------------------------------------------------------
class GammaClient:
    """Read-only Gamma API for market/event discovery."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.base_url = cfg.polymarket_gamma_host
        self._http = httpx.AsyncClient(timeout=15)

    async def get_active_markets(
        self,
        limit: int = 50,
        min_liquidity: float = 1000.0,
        keywords: list[str] | None = None,
    ) -> list[Market]:
        params: dict[str, Any] = {
            "limit": limit,
            "active": "true",
            "closed": "false",
            "order": "liquidity",
            "ascending": "false",
        }
        resp = await self._http.get(f"{self.base_url}/events", params=params)
        resp.raise_for_status()
        events = resp.json()

        markets: list[Market] = []
        for event in events:
            title = (event.get("title", "") or "").lower()
            slug = (event.get("slug", "") or "").lower()
            for mkt in event.get("markets", []):
                question = (mkt.get("question", "") or "").lower()
                searchable = f"{title} {slug} {question}"

                if keywords:
                    if not any(kw in searchable for kw in keywords):
                        continue

                liq = float(mkt.get("liquidity", 0) or 0)
                if liq < min_liquidity:
                    continue
                tokens_raw = mkt.get("clobTokenIds", "")
                if isinstance(tokens_raw, str):
                    try:
                        tokens = json.loads(tokens_raw)
                    except (json.JSONDecodeError, TypeError):
                        tokens = [t.strip().strip('"') for t in tokens_raw.strip("[]").split(",") if t.strip()]
                elif isinstance(tokens_raw, list):
                    tokens = tokens_raw
                else:
                    tokens = []
                if not tokens:
                    continue
                markets.append(
                    Market(
                        token_id=tokens[0],
                        condition_id=mkt.get("conditionId", ""),
                        question=mkt.get("question", event.get("title", "")),
                        slug=event.get("slug", ""),
                        active=True,
                        liquidity=liq,
                        volume_24h=float(mkt.get("volume24hr", 0) or 0),
                        neg_risk=bool(mkt.get("negRisk", False)),
                    )
                )
        return markets

    async def close(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# CLOB REST client (orderbook + order placement)
# ---------------------------------------------------------------------------
class CLOBClient:
    """Handles REST order book fetches and order placement via py-clob-client."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.host = cfg.polymarket_clob_host
        self._http = httpx.AsyncClient(timeout=15)
        self._clob = None  # lazy init of py_clob_client.ClobClient

    def _get_clob(self):
        if self._clob is None:
            from py_clob_client.client import ClobClient
            cfg = get_settings()
            self._clob = ClobClient(
                host=cfg.polymarket_clob_host,
                chain_id=cfg.polymarket_chain_id,
                key=cfg.polymarket_private_key,
            )
            creds = self._clob.derive_api_key()
            self._clob.set_api_creds(creds)
        return self._clob

    async def get_orderbook(self, token_id: str) -> OrderBookSnapshot:
        resp = await self._http.get(
            f"{self.host}/book", params={"token_id": token_id}
        )
        resp.raise_for_status()
        data = resp.json()
        bids = [(float(l["price"]), float(l["size"])) for l in data.get("bids", [])]
        asks = [(float(l["price"]), float(l["size"])) for l in data.get("asks", [])]
        return OrderBookSnapshot(
            token_id=token_id,
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x: x[0]),
            timestamp=time.time(),
        )

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        neg_risk: bool = False,
        tick_size: str = "0.01",
        order_type: str = "GTC",
    ) -> dict[str, Any]:
        """Place order via py-clob-client (sync call wrapped in executor)."""
        from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
        from py_clob_client.order_builder.constants import BUY, SELL

        loop = asyncio.get_running_loop()
        clob = self._get_clob()
        side_val = BUY if side.upper() == "BUY" else SELL

        def _place():
            return clob.create_and_post_order(
                OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side=side_val,
                ),
                options=PartialCreateOrderOptions(
                    tick_size=tick_size,
                    neg_risk=neg_risk,
                ),
            )

        result = await loop.run_in_executor(None, _place)
        log.info("order_placed", token_id=token_id, side=side, price=price, size=size, result=result)
        return result

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        clob = self._get_clob()
        result = await loop.run_in_executor(None, clob.cancel, order_id)
        log.info("order_cancelled", order_id=order_id)
        return result

    async def get_open_orders(self) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        clob = self._get_clob()
        orders = await loop.run_in_executor(None, clob.get_orders)
        return orders if isinstance(orders, list) else []

    async def close(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# WebSocket market data stream
# ---------------------------------------------------------------------------
class MarketWebSocket:
    """Streams real-time price/book updates for subscribed markets."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.ws_url = cfg.polymarket_ws_host
        self._ws = None
        self._running = False
        self._callbacks: list = []

    def on_message(self, callback) -> None:
        self._callbacks.append(callback)

    async def subscribe(self, token_ids: list[str]) -> None:
        self._running = True
        while self._running:
            try:
                async with ws_connect(self.ws_url) as ws:
                    self._ws = ws
                    sub_msg = json.dumps({
                        "assets_ids": token_ids,
                        "type": "market",
                        "custom_feature_enabled": True,
                    })
                    await ws.send(sub_msg)
                    log.info("ws_subscribed", token_count=len(token_ids))
                    async for raw in ws:
                        msg = json.loads(raw)
                        for cb in self._callbacks:
                            try:
                                await cb(msg)
                            except Exception:
                                log.exception("ws_callback_error")
            except websockets.ConnectionClosed:
                log.warning("ws_disconnected, reconnecting in 5s")
                await asyncio.sleep(5)
            except Exception:
                log.exception("ws_error, reconnecting in 10s")
                await asyncio.sleep(10)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
