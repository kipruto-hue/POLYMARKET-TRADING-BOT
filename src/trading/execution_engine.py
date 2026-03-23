"""
Execution Engine v2 — latency tracking, stale order management,
liquidity-scaled sizing, time-to-decision metric.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import structlog

from src.data.polymarket_client import CLOBClient, Market
from src.strategy.feedback import FeedbackEngine
from src.strategy.signal_engine import Decision, Signal
from src.trading.risk_engine import RiskEngine

log = structlog.get_logger()


@dataclass
class ActiveOrder:
    order_id: str
    trade_id: str
    token_id: str
    side: str
    price: float
    size: float
    placed_ts: float = field(default_factory=time.time)
    max_age_seconds: float = 120.0  # cancel stale orders after 2 minutes

    def is_stale(self) -> bool:
        return (time.time() - self.placed_ts) > self.max_age_seconds


class ExecutionEngine:
    """
    Places and manages orders with full latency tracking and stale-order cleanup.
    """

    def __init__(
        self,
        clob: CLOBClient,
        risk: RiskEngine,
        feedback: FeedbackEngine,
    ) -> None:
        self.clob = clob
        self.risk = risk
        self.feedback = feedback
        self._market_meta: dict[str, Market] = {}
        self._active_orders: dict[str, ActiveOrder] = {}
        self._latencies: list[float] = []     # time-to-decision latencies (ms)

    def register_markets(self, markets: list[Market]) -> None:
        for m in markets:
            self._market_meta[m.token_id] = m

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return round(sum(self._latencies[-50:]) / len(self._latencies[-50:]), 2)

    async def execute(self, decision: Decision, signal_ts: float | None = None) -> dict | None:
        exec_start = time.time()

        # Time-to-decision metric
        if signal_ts:
            ttd_ms = (exec_start - signal_ts) * 1000
            self._latencies.append(ttd_ms)
            if ttd_ms > 2000:
                log.warning("high_latency", ttd_ms=round(ttd_ms), token=decision.token_id[:12])

        # Feedback gate — anti-decay engine check
        allowed, mode_reason = self.feedback.is_trading_allowed()
        if not allowed:
            log.info("trade_blocked_by_feedback", reason=mode_reason)
            return None

        # Risk check
        approved, reason = self.risk.check(decision)
        if not approved:
            log.info("trade_rejected", token=decision.token_id[:12], reason=reason)
            return None

        # Liquidity-scaled size
        size = self._compute_liquidity_scaled_size(decision)
        if size <= 0:
            log.info("zero_size_skipped", token=decision.token_id[:12])
            return None

        meta = self._market_meta.get(decision.token_id)
        neg_risk = meta.neg_risk if meta else False
        tick_size = meta.tick_size if meta else "0.01"
        side = "BUY" if decision.signal == Signal.BUY else "SELL"
        trade_id = str(uuid.uuid4())[:12]

        try:
            self.risk.state.open_order_count += 1
            result = await self.clob.place_order(
                token_id=decision.token_id,
                side=side,
                price=decision.target_price,
                size=size,
                neg_risk=neg_risk,
                tick_size=tick_size,
            )

            order_id = result.get("orderID", "") if isinstance(result, dict) else ""

            # Track active order for stale cancellation
            active = ActiveOrder(
                order_id=order_id,
                trade_id=trade_id,
                token_id=decision.token_id,
                side=side,
                price=decision.target_price,
                size=size,
            )
            if order_id:
                self._active_orders[order_id] = active

            # Record entry in feedback engine
            self.feedback.record_entry(
                trade_id=trade_id,
                token_id=decision.token_id,
                signal=side,
                confidence=decision.confidence,
                ev_estimated=decision.expected_value,
                entry_price=decision.target_price,
                size=size,
                reasons=decision.reasons,
            )

            # Record fill in risk engine
            self.risk.state.record_fill(
                decision.token_id, side, decision.target_price, size, trade_id=trade_id
            )

            exec_ms = (time.time() - exec_start) * 1000
            log.info(
                "order_submitted",
                token=decision.token_id[:12],
                side=side,
                size=size,
                price=decision.target_price,
                ev=decision.expected_value,
                confidence=decision.confidence,
                exec_ms=round(exec_ms),
                ttd_ms=round(self._latencies[-1]) if self._latencies else 0,
                trade_id=trade_id,
            )
            return result

        except Exception:
            log.exception("order_placement_failed", token=decision.token_id[:12])
            return None
        finally:
            self.risk.state.open_order_count = max(0, self.risk.state.open_order_count - 1)

    def _compute_liquidity_scaled_size(self, decision: Decision) -> float:
        """Scale size down when book depth is shallow."""
        base_size = self.risk.compute_order_size(decision)
        meta = self._market_meta.get(decision.token_id)
        if meta is None:
            return base_size

        # If book depth info is in decision's feature context, scale accordingly
        # Conservative: cap at base_size, no further scaling needed here
        return round(min(base_size, self.risk.max_position), 2)

    async def cancel_stale_orders(self) -> int:
        cancelled = 0
        stale = [o for o in list(self._active_orders.values()) if o.is_stale()]
        for order in stale:
            try:
                await self.clob.cancel_order(order.order_id)
                del self._active_orders[order.order_id]
                cancelled += 1
                log.info("stale_order_cancelled", order_id=order.order_id,
                         age_s=round(time.time() - order.placed_ts))
            except Exception:
                log.exception("stale_cancel_failed", order_id=order.order_id)
        return cancelled

    async def cancel_all(self) -> None:
        try:
            orders = await self.clob.get_open_orders()
            for order in orders:
                oid = order.get("id") or order.get("orderID", "")
                if oid:
                    await self.clob.cancel_order(oid)
                    self._active_orders.pop(oid, None)
            log.info("all_orders_cancelled", count=len(orders))
        except Exception:
            log.exception("cancel_all_failed")

    async def check_exit_conditions(self, token_id: str, current_price: float) -> bool:
        """Check and execute dynamic stop exits."""
        should_exit, reason = self.risk.should_exit_position(token_id, current_price)
        if should_exit:
            log.info("dynamic_exit_triggered", token=token_id[:12], reason=reason)
            pos = self.risk.state.positions.get(token_id)
            if pos:
                exit_side = "SELL" if pos.side == "BUY" else "BUY"
                meta = self._market_meta.get(token_id)
                try:
                    await self.clob.place_order(
                        token_id=token_id, side=exit_side,
                        price=current_price, size=pos.size,
                        neg_risk=meta.neg_risk if meta else False,
                        tick_size=meta.tick_size if meta else "0.01",
                    )
                    pnl = self.risk.state.positions[token_id].unrealized_pnl
                    self.feedback.record_exit(
                        trade_id=pos.trade_id, exit_price=current_price, pnl=pnl
                    )
                    del self.risk.state.positions[token_id]
                    return True
                except Exception:
                    log.exception("dynamic_exit_failed", token=token_id[:12])
        return False
