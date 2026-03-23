"""Execution engine: bridges signal decisions to live order placement."""

from __future__ import annotations

import structlog

from src.config.settings import get_settings
from src.data.polymarket_client import CLOBClient, Market
from src.strategy.signal_engine import Decision, Signal
from src.trading.risk_engine import RiskEngine

log = structlog.get_logger()


class ExecutionEngine:
    """Takes approved decisions and places/manages orders on Polymarket CLOB."""

    def __init__(self, clob: CLOBClient, risk: RiskEngine) -> None:
        self.clob = clob
        self.risk = risk
        self._market_meta: dict[str, Market] = {}

    def register_markets(self, markets: list[Market]) -> None:
        for m in markets:
            self._market_meta[m.token_id] = m

    async def execute(self, decision: Decision) -> dict | None:
        approved, reason = self.risk.check(decision)
        if not approved:
            log.info("trade_rejected", token=decision.token_id[:12], reason=reason)
            return None

        size = self.risk.compute_order_size(decision)
        if size <= 0:
            log.info("zero_size_skipped", token=decision.token_id[:12])
            return None

        meta = self._market_meta.get(decision.token_id)
        neg_risk = meta.neg_risk if meta else False
        tick_size = meta.tick_size if meta else "0.01"

        side = "BUY" if decision.signal == Signal.BUY else "SELL"

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
            log.info(
                "order_submitted",
                token=decision.token_id[:12],
                side=side,
                size=size,
                price=decision.target_price,
            )
            # Optimistically record (will reconcile on fill callback)
            self.risk.state.record_fill(
                decision.token_id, side, decision.target_price, size
            )
            return result
        except Exception:
            log.exception("order_placement_failed", token=decision.token_id[:12])
            return None
        finally:
            self.risk.state.open_order_count = max(
                0, self.risk.state.open_order_count - 1
            )

    async def cancel_all(self) -> None:
        try:
            orders = await self.clob.get_open_orders()
            for order in orders:
                oid = order.get("id")
                if oid:
                    await self.clob.cancel_order(oid)
            log.info("all_orders_cancelled", count=len(orders))
        except Exception:
            log.exception("cancel_all_failed")
