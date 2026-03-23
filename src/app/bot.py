"""Core bot loop: 5-minute analysis candles + high-frequency execution targeting Bitcoin markets."""

from __future__ import annotations

import asyncio
import time

import structlog

from src.config.settings import get_settings
from src.data.candle_builder import CandleBuilder
from src.data.news_client import NewsClient
from src.data.polymarket_client import (
    CLOBClient,
    GammaClient,
    Market,
    MarketWebSocket,
)
from src.monitoring.metrics import AlertManager, BotMetrics
from src.storage import repository as repo
from src.strategy.features import build_features
from src.strategy.signal_engine import generate_decision, Signal
from src.trading.execution_engine import ExecutionEngine
from src.trading.risk_engine import RiskEngine

log = structlog.get_logger()


class PolymarketBot:
    """
    Orchestrates Bitcoin-focused prediction market trading.
    - 5-minute candle analysis window for signal generation
    - ~1.2s execution loop for high-frequency small trades ($2 max)
    """

    def __init__(self) -> None:
        self.cfg = get_settings()
        self.gamma = GammaClient()
        self.clob = CLOBClient()
        self.news = NewsClient()
        self.ws = MarketWebSocket()
        self.candles = CandleBuilder(interval_seconds=self.cfg.bot_interval_seconds)
        self.risk = RiskEngine()
        self.executor = ExecutionEngine(self.clob, self.risk)
        self.metrics = BotMetrics()
        self.alerts = AlertManager()
        self.markets: list[Market] = []
        self._running = False

        self._keywords = [
            kw.strip().lower()
            for kw in self.cfg.market_keywords.split(",")
            if kw.strip()
        ]
        self._latest_decisions: dict[str, object] = {}
        self._last_news_fetch: float = 0
        self._cached_news: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        log.info(
            "bot_starting",
            dry_run=self.cfg.dry_run,
            keywords=self._keywords,
            max_trade=self.cfg.max_position_size,
            exec_interval=self.cfg.execution_interval_seconds,
        )
        try:
            await repo.init_db()
        except Exception:
            log.warning("db_init_skipped (run without DB for now)")

        await self._refresh_markets()

        self._running = True
        ws_task = asyncio.create_task(self._run_websocket())
        analysis_task = asyncio.create_task(self._run_analysis_loop())
        exec_task = asyncio.create_task(self._run_execution_loop())

        try:
            await asyncio.gather(ws_task, analysis_task, exec_task)
        except asyncio.CancelledError:
            log.info("bot_shutting_down")
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        await self.gamma.close()
        await self.clob.close()
        await self.news.close()
        await self.ws.stop()
        log.info("bot_stopped")

    # ------------------------------------------------------------------
    # Market discovery (Bitcoin/crypto filtered)
    # ------------------------------------------------------------------
    async def _refresh_markets(self) -> None:
        try:
            all_markets = await self.gamma.get_active_markets(
                limit=100,
                min_liquidity=self.cfg.min_market_liquidity,
                keywords=self._keywords if self._keywords else None,
            )
            self.markets = all_markets[: self.cfg.max_markets]
            self.executor.register_markets(self.markets)
            log.info(
                "btc_markets_loaded",
                count=len(self.markets),
                keywords=self._keywords,
                questions=[m.question[:60] for m in self.markets[:5]],
            )
            if not self.markets:
                log.warning(
                    "no_btc_markets_found, falling back to top liquid markets"
                )
                fallback = await self.gamma.get_active_markets(
                    limit=self.cfg.max_markets,
                    min_liquidity=self.cfg.min_market_liquidity,
                )
                self.markets = fallback[: self.cfg.max_markets]
                self.executor.register_markets(self.markets)
                log.info("fallback_markets_loaded", count=len(self.markets))
        except Exception:
            log.exception("market_refresh_failed")
            self.metrics.errors += 1

    # ------------------------------------------------------------------
    # WebSocket data ingestion
    # ------------------------------------------------------------------
    async def _run_websocket(self) -> None:
        if not self.markets:
            log.warning("no_markets, ws_skipped")
            return
        token_ids = [m.token_id for m in self.markets]

        async def _handle(msg: dict) -> None:
            for event in msg if isinstance(msg, list) else [msg]:
                asset = event.get("asset_id", "")
                if not asset:
                    continue
                price = event.get("price")
                size = event.get("size", 0)
                if price is not None:
                    self.candles.add_tick(asset, float(price), float(size or 0))

        self.ws.on_message(_handle)
        await self.ws.subscribe(token_ids)

    # ------------------------------------------------------------------
    # 5-minute analysis loop (generates signals)
    # ------------------------------------------------------------------
    async def _run_analysis_loop(self) -> None:
        await asyncio.sleep(5)
        while self._running:
            try:
                await self._analysis_cycle()
                self.metrics.cycles_completed += 1
                self.metrics.last_cycle_ts = time.time()
            except Exception as exc:
                self.metrics.errors += 1
                self.metrics.last_error = str(exc)
                log.exception("analysis_cycle_error")
            await asyncio.sleep(self.cfg.bot_interval_seconds)

    async def _analysis_cycle(self) -> None:
        log.info("analysis_cycle_start", markets=len(self.markets))

        if self.metrics.cycles_completed % 12 == 0:
            await self._refresh_markets()

        # Fetch news once per analysis cycle (not per market)
        await self._refresh_news()

        for market in self.markets:
            try:
                decision = await self._analyze_market(market)
                if decision:
                    self._latest_decisions[market.token_id] = decision
            except Exception:
                log.exception("market_analysis_error", token=market.token_id[:12])

        active_signals = {
            tid: d for tid, d in self._latest_decisions.items()
            if d.signal != Signal.HOLD
        }
        log.info(
            "analysis_cycle_done",
            total_signals=len(self._latest_decisions),
            active_trades=len(active_signals),
        )

    async def _refresh_news(self) -> None:
        now = time.time()
        if now - self._last_news_fetch < 120:
            return
        self._last_news_fetch = now

        for keyword in ["bitcoin", "btc price", "crypto market"]:
            try:
                articles = await self.news.fetch_articles(
                    keyword, lookback_minutes=60, max_results=5
                )
                score = self.news.aggregate_sentiment(articles)
                self._cached_news[keyword] = score
                log.info("news_fetched", keyword=keyword, sentiment=round(score, 3), articles=len(articles))
            except Exception:
                log.exception("news_fetch_error", keyword=keyword)

    async def _analyze_market(self, market: Market):
        history = self.candles.get_history(market.token_id, n=20)
        current = self.candles.get_current_candle(market.token_id)
        if current:
            history = history + [current]
        if len(history) < 3:
            return None

        try:
            book = await self.clob.get_orderbook(market.token_id)
        except Exception:
            book = None

        avg_news = (
            sum(self._cached_news.values()) / len(self._cached_news)
            if self._cached_news
            else 0.0
        )

        features = build_features(
            token_id=market.token_id,
            candles=history,
            book=book,
            news_sentiment=avg_news,
        )

        decision = generate_decision(features)
        self.metrics.signals_generated += 1

        approved, reason = self.risk.check(decision)
        try:
            await repo.log_decision(
                token_id=market.token_id,
                signal=decision.signal.value,
                confidence=decision.confidence,
                score=0.0,
                reasons="; ".join(decision.reasons),
                target_price=decision.target_price,
                approved=approved,
                rejection_reason=reason if not approved else "",
            )
        except Exception:
            pass

        return decision

    # ------------------------------------------------------------------
    # High-frequency execution loop (~50 trades/min)
    # ------------------------------------------------------------------
    async def _run_execution_loop(self) -> None:
        await asyncio.sleep(10)
        log.info("execution_loop_started", interval=self.cfg.execution_interval_seconds)

        while self._running:
            try:
                await self._execute_pending()
            except Exception as exc:
                self.metrics.errors += 1
                self.metrics.last_error = str(exc)
                log.exception("execution_error")
            await asyncio.sleep(self.cfg.execution_interval_seconds)

    async def _execute_pending(self) -> None:
        if self.risk.state.kill_switch:
            return

        for token_id, decision in list(self._latest_decisions.items()):
            if decision.signal == Signal.HOLD:
                continue

            result = await self.executor.execute(decision)
            if result:
                self.metrics.trades_placed += 1
                log.info(
                    "trade_executed",
                    token=token_id[:12],
                    side=decision.signal.value,
                    price=decision.target_price,
                    size=self.risk.compute_order_size(decision),
                )
                try:
                    await repo.log_trade(
                        token_id=token_id,
                        side=decision.signal.value,
                        price=decision.target_price,
                        size=self.risk.compute_order_size(decision),
                        order_id=result.get("orderID", ""),
                        status="submitted",
                    )
                except Exception:
                    pass
            else:
                self.metrics.trades_rejected += 1

        # PnL snapshot every ~60 executions
        if self.metrics.trades_placed % 60 == 0:
            try:
                await repo.save_pnl_snapshot(
                    daily_pnl=self.risk.state.daily_pnl,
                    total_pnl=self.risk.state.daily_pnl,
                    open_positions=len(self.risk.state.positions),
                    open_orders=self.risk.state.open_order_count,
                )
            except Exception:
                pass
