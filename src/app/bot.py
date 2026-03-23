"""
Bot Orchestrator v2 — selective, adaptive, self-correcting.

Architecture:
  - Analysis loop (5-min): builds features, scores signals, gates via EV+confluence
  - Execution loop (1.2s): fires approved decisions, tracks latency, manages exits
  - Watchdog loop (30s):  monitors performance, cancels stale orders, logs metrics
  - Feedback engine:      recalibrates thresholds, enforces diagnostic/halt modes
"""

from __future__ import annotations

import asyncio
import time

import structlog

from src.config.settings import get_settings
from src.data.candle_builder import CandleBuilder
from src.data.news_client import NewsClient
from src.data.polymarket_client import CLOBClient, GammaClient, Market, MarketWebSocket
from src.monitoring.metrics import AlertManager, BotMetrics
from src.storage import repository as repo
from src.strategy.features import build_features
from src.strategy.feedback import FeedbackEngine
from src.strategy.signal_engine import THRESHOLDS, Signal, generate_decision
from src.trading.execution_engine import ExecutionEngine
from src.trading.risk_engine import RiskEngine

log = structlog.get_logger()


class PolymarketBot:
    def __init__(self) -> None:
        self.cfg = get_settings()
        self.gamma = GammaClient()
        self.clob = CLOBClient()
        self.news = NewsClient()
        self.ws = MarketWebSocket()
        self.candles = CandleBuilder(interval_seconds=self.cfg.bot_interval_seconds)
        self.risk = RiskEngine()
        self.feedback = FeedbackEngine()
        self.executor = ExecutionEngine(self.clob, self.risk, self.feedback)
        self.metrics = BotMetrics()
        self.alerts = AlertManager()
        self.markets: list[Market] = []
        self._running = False

        self._keywords = [
            kw.strip().lower()
            for kw in self.cfg.market_keywords.split(",")
            if kw.strip()
        ]
        # Stores (decision, signal_timestamp) per token
        self._pending_decisions: dict[str, tuple] = {}
        self._cached_news: dict[str, float] = {}
        self._last_news_fetch: float = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        log.info(
            "bot_starting_v2",
            dry_run=self.cfg.dry_run,
            keywords=self._keywords,
            max_trade=self.cfg.max_position_size,
            thresholds=THRESHOLDS.to_dict(),
        )
        try:
            await repo.init_db()
        except Exception:
            log.warning("db_init_skipped")

        await self._refresh_markets()
        self._running = True

        await asyncio.gather(
            asyncio.create_task(self._run_websocket()),
            asyncio.create_task(self._run_analysis_loop()),
            asyncio.create_task(self._run_execution_loop()),
            asyncio.create_task(self._run_watchdog_loop()),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        self._running = False
        await self.executor.cancel_all()
        await self.gamma.close()
        await self.clob.close()
        await self.news.close()
        await self.ws.stop()
        log.info("bot_stopped", **self.metrics.to_dict())

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------
    async def _refresh_markets(self) -> None:
        try:
            markets = await self.gamma.get_active_markets(
                limit=100,
                min_liquidity=self.cfg.min_market_liquidity,
                keywords=self._keywords or None,
            )
            if not markets:
                markets = await self.gamma.get_active_markets(
                    limit=self.cfg.max_markets,
                    min_liquidity=self.cfg.min_market_liquidity,
                )
            self.markets = markets[: self.cfg.max_markets]
            self.executor.register_markets(self.markets)
            log.info("markets_loaded", count=len(self.markets),
                     questions=[m.question[:50] for m in self.markets[:3]])
        except Exception:
            log.exception("market_refresh_failed")
            self.metrics.errors += 1

    # ------------------------------------------------------------------
    # WebSocket ingestion
    # ------------------------------------------------------------------
    async def _run_websocket(self) -> None:
        if not self.markets:
            return
        token_ids = [m.token_id for m in self.markets]

        async def _handle(msg) -> None:
            for event in (msg if isinstance(msg, list) else [msg]):
                asset = event.get("asset_id", "")
                price = event.get("price")
                size = event.get("size", 0)
                if asset and price is not None:
                    self.candles.add_tick(asset, float(price), float(size or 0))

        self.ws.on_message(_handle)
        await self.ws.subscribe(token_ids)

    # ------------------------------------------------------------------
    # 5-minute analysis loop
    # ------------------------------------------------------------------
    async def _run_analysis_loop(self) -> None:
        await asyncio.sleep(5)
        while self._running:
            cycle_start = time.time()
            try:
                await self._analysis_cycle()
                self.metrics.cycles_completed += 1
                self.metrics.last_cycle_ts = time.time()
            except Exception as exc:
                self.metrics.errors += 1
                self.metrics.last_error = str(exc)
                log.exception("analysis_cycle_error")
                self.alerts.warning(f"Analysis error: {exc}")
            await asyncio.sleep(max(0, self.cfg.bot_interval_seconds - (time.time() - cycle_start)))

    async def _analysis_cycle(self) -> None:
        if self.metrics.cycles_completed % 12 == 0:
            await self._refresh_markets()

        await self._refresh_news()

        approved_count = 0
        rejected_counts: dict[str, int] = {}

        for market in self.markets:
            try:
                decision, signal_ts = await self._analyze_market(market)
                self.metrics.signals_generated += 1

                if decision is None:
                    continue

                if decision.signal != Signal.HOLD:
                    self._pending_decisions[market.token_id] = (decision, signal_ts)
                    approved_count += 1
                    self.metrics.signals_approved += 1
                else:
                    reason = decision.rejected_reason
                    if "ev" in reason:
                        self.metrics.signals_rejected_ev += 1
                    elif "confluence" in reason:
                        self.metrics.signals_rejected_confluence += 1
                    elif "spread" in reason:
                        self.metrics.signals_rejected_spread += 1
                    elif "confidence" in reason:
                        self.metrics.signals_rejected_confidence += 1
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

                try:
                    approved, risk_reason = self.risk.check(decision)
                    await repo.log_decision(
                        token_id=market.token_id,
                        signal=decision.signal.value,
                        confidence=decision.confidence,
                        score=decision.expected_value,
                        reasons="; ".join(decision.reasons),
                        target_price=decision.target_price,
                        approved=approved,
                        rejection_reason=decision.rejected_reason or risk_reason,
                    )
                except Exception:
                    pass

            except Exception:
                log.exception("market_analysis_error", token=market.token_id[:12])

        log.info(
            "analysis_cycle_done",
            approved=approved_count,
            rejected=rejected_counts,
            mode=self.feedback.mode.value,
            win_rate=self.feedback.window.win_rate,
            thresholds=THRESHOLDS.to_dict(),
        )

    async def _refresh_news(self) -> None:
        if time.time() - self._last_news_fetch < 120:
            return
        self._last_news_fetch = time.time()
        for keyword in ["bitcoin", "btc price", "crypto market"]:
            try:
                articles = await self.news.fetch_articles(keyword, max_results=5)
                score = self.news.aggregate_sentiment(articles)
                relevance = min(1.0, len(articles) / 5.0)
                self._cached_news[keyword] = score
                log.info("news_fetched", keyword=keyword,
                         sentiment=round(score, 3), articles=len(articles))
            except Exception:
                log.exception("news_fetch_error", keyword=keyword)

    async def _analyze_market(self, market: Market):
        history = self.candles.get_history(market.token_id, n=20)
        current = self.candles.get_current_candle(market.token_id)
        if current:
            history = history + [current]
        if len(history) < 3:
            return None, None

        signal_ts = time.time()

        try:
            book = await self.clob.get_orderbook(market.token_id)
        except Exception:
            book = None

        avg_news = (
            sum(self._cached_news.values()) / len(self._cached_news)
            if self._cached_news else 0.0
        )
        news_relevance = min(1.0, len(self._cached_news) / 3.0)

        features = build_features(
            token_id=market.token_id,
            candles=history,
            book=book,
            news_sentiment=avg_news,
            news_relevance=news_relevance,
        )

        decision = generate_decision(features)
        return decision, signal_ts

    # ------------------------------------------------------------------
    # High-frequency execution loop
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

        for token_id, (decision, signal_ts) in list(self._pending_decisions.items()):
            if decision.signal == Signal.HOLD:
                continue

            result = await self.executor.execute(decision, signal_ts=signal_ts)
            if result:
                self.metrics.trades_placed += 1
                try:
                    await repo.log_trade(
                        token_id=token_id,
                        side=decision.signal.value,
                        price=decision.target_price,
                        size=self.risk.compute_order_size(decision),
                        order_id=result.get("orderID", "") if isinstance(result, dict) else "",
                        status="submitted",
                    )
                except Exception:
                    pass
            else:
                self.metrics.trades_rejected += 1

        # Dynamic exits: check all open positions
        for token_id, pos in list(self.risk.state.positions.items()):
            try:
                book = await self.clob.get_orderbook(token_id)
                mid = (book.bids[0][0] + book.asks[0][0]) / 2 if book.bids and book.asks else 0
                if mid > 0:
                    await self.executor.check_exit_conditions(token_id, mid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Watchdog loop (every 30s)
    # ------------------------------------------------------------------
    async def _run_watchdog_loop(self) -> None:
        await asyncio.sleep(30)
        while self._running:
            try:
                cancelled = await self.executor.cancel_stale_orders()
                if cancelled:
                    log.info("stale_orders_cleared", count=cancelled)

                feedback_stats = self.feedback.to_dict()
                metrics_snap = self.metrics.to_dict()

                log.info(
                    "watchdog_report",
                    mode=feedback_stats["mode"],
                    win_rate=feedback_stats["rolling_window"]["win_rate"],
                    total_trades=feedback_stats["total_trades"],
                    total_pnl=feedback_stats["total_pnl"],
                    bot_trades_placed=metrics_snap["trades_placed"],
                    avg_latency_ms=metrics_snap["latency_avg_ms"],
                    p95_latency_ms=metrics_snap["latency_p95_ms"],
                    daily_pnl=metrics_snap["daily_pnl"],
                    drawdown=metrics_snap["max_drawdown"],
                )

                if feedback_stats["mode"] == "halted":
                    self.alerts.critical("Bot halted by anti-decay engine — manual review needed")
                    self.risk.activate_kill_switch()

                # PnL snapshot
                try:
                    await repo.save_pnl_snapshot(
                        daily_pnl=self.risk.state.daily_pnl,
                        total_pnl=self.metrics.total_pnl,
                        open_positions=len(self.risk.state.positions),
                        open_orders=self.risk.state.open_order_count,
                    )
                except Exception:
                    pass

            except Exception:
                log.exception("watchdog_error")
            await asyncio.sleep(30)
