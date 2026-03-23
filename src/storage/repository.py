"""Persistence helpers for trade/decision logging and PnL snapshots."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select

from src.config.settings import get_settings
from src.storage.models import DecisionLog, PnlSnapshot, TradeLog

log = structlog.get_logger()

_engine = None
_session_factory = None


async def init_db() -> None:
    global _engine, _session_factory
    cfg = get_settings()
    _engine = create_async_engine(cfg.database_url, echo=False)
    _session_factory = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    log.info("database_initialized")


def _get_session() -> AsyncSession:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


async def log_trade(
    token_id: str,
    side: str,
    price: float,
    size: float,
    order_id: str = "",
    status: str = "submitted",
) -> None:
    async with _get_session() as session:
        entry = TradeLog(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_id=order_id,
            status=status,
        )
        session.add(entry)
        await session.commit()


async def log_decision(
    token_id: str,
    signal: str,
    confidence: float,
    score: float,
    reasons: str,
    target_price: float,
    approved: bool,
    rejection_reason: str = "",
) -> None:
    async with _get_session() as session:
        entry = DecisionLog(
            token_id=token_id,
            signal=signal,
            confidence=confidence,
            score=score,
            reasons=reasons,
            target_price=target_price,
            approved=approved,
            rejection_reason=rejection_reason,
        )
        session.add(entry)
        await session.commit()


async def save_pnl_snapshot(
    daily_pnl: float,
    total_pnl: float,
    open_positions: int,
    open_orders: int,
) -> None:
    async with _get_session() as session:
        entry = PnlSnapshot(
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            open_positions=open_positions,
            open_orders=open_orders,
        )
        session.add(entry)
        await session.commit()


async def get_recent_trades(limit: int = 50) -> list[TradeLog]:
    async with _get_session() as session:
        stmt = select(TradeLog).order_by(TradeLog.timestamp.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def get_recent_decisions(limit: int = 50) -> list[DecisionLog]:
    async with _get_session() as session:
        stmt = select(DecisionLog).order_by(DecisionLog.timestamp.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())
