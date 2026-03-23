"""Database models for trade log, decision log, and PnL snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class TradeLog(SQLModel, table=True):
    __tablename__ = "trade_log"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_id: str = Field(index=True)
    side: str
    price: float
    size: float
    order_id: str = ""
    status: str = "submitted"
    pnl: float = 0.0


class DecisionLog(SQLModel, table=True):
    __tablename__ = "decision_log"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_id: str = Field(index=True)
    signal: str
    confidence: float
    score: float = 0.0
    reasons: str = ""
    target_price: float = 0.0
    approved: bool = False
    rejection_reason: str = ""


class PnlSnapshot(SQLModel, table=True):
    __tablename__ = "pnl_snapshot"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    open_positions: int = 0
    open_orders: int = 0


class MarketCache(SQLModel, table=True):
    __tablename__ = "market_cache"

    id: int | None = Field(default=None, primary_key=True)
    token_id: str = Field(index=True, unique=True)
    condition_id: str = ""
    question: str = ""
    slug: str = ""
    liquidity: float = 0.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
