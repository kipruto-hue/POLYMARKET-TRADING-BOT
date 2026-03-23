"""
Performance metrics v2 — win rate, EV/trade, latency, slippage, drawdown,
signal accuracy vs outcome. All tracked in real-time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import deque

import structlog

log = structlog.get_logger()


@dataclass
class LatencyTracker:
    _samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, ms: float) -> None:
        self._samples.append(ms)

    @property
    def avg_ms(self) -> float:
        return round(sum(self._samples) / len(self._samples), 2) if self._samples else 0.0

    @property
    def p95_ms(self) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = int(len(s) * 0.95)
        return round(s[idx], 2)


@dataclass
class BotMetrics:
    start_time: float = field(default_factory=time.time)

    # Cycle tracking
    cycles_completed: int = 0
    last_cycle_ts: float = 0.0

    # Signal tracking
    signals_generated: int = 0
    signals_approved: int = 0
    signals_rejected_ev: int = 0
    signals_rejected_confluence: int = 0
    signals_rejected_spread: int = 0
    signals_rejected_confidence: int = 0

    # Trade tracking
    trades_placed: int = 0
    trades_rejected: int = 0
    trades_won: int = 0
    trades_lost: int = 0

    # Financial
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    total_ev_estimated: float = 0.0
    total_slippage: float = 0.0

    # Latency
    latency: LatencyTracker = field(default_factory=LatencyTracker)

    # Error tracking
    errors: int = 0
    last_error: str = ""

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def win_rate(self) -> float:
        done = self.trades_won + self.trades_lost
        return round(self.trades_won / done, 3) if done > 0 else 0.0

    @property
    def avg_ev_per_trade(self) -> float:
        return round(self.total_ev_estimated / max(self.trades_placed, 1), 5)

    @property
    def avg_slippage_per_trade(self) -> float:
        return round(self.total_slippage / max(self.trades_placed, 1), 5)

    @property
    def drawdown(self) -> float:
        return round(self.peak_pnl - self.total_pnl, 4)

    def record_trade(self, pnl: float, ev: float, slippage: float = 0.0) -> None:
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.total_ev_estimated += ev
        self.total_slippage += abs(slippage)
        if pnl > 0:
            self.trades_won += 1
        else:
            self.trades_lost += 1
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        if self.drawdown > self.max_drawdown:
            self.max_drawdown = self.drawdown

    def to_dict(self) -> dict:
        return {
            "uptime_s": round(self.uptime_seconds),
            "cycles": self.cycles_completed,

            "signals_generated": self.signals_generated,
            "signals_approved": self.signals_approved,
            "rejected_ev": self.signals_rejected_ev,
            "rejected_confluence": self.signals_rejected_confluence,
            "rejected_spread": self.signals_rejected_spread,
            "rejected_confidence": self.signals_rejected_confidence,

            "trades_placed": self.trades_placed,
            "trades_rejected": self.trades_rejected,
            "win_rate": self.win_rate,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,

            "total_pnl": round(self.total_pnl, 4),
            "daily_pnl": round(self.daily_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_ev_per_trade": self.avg_ev_per_trade,
            "avg_slippage": self.avg_slippage_per_trade,

            "latency_avg_ms": self.latency.avg_ms,
            "latency_p95_ms": self.latency.p95_ms,

            "errors": self.errors,
            "last_error": self.last_error,
        }


class AlertManager:
    def __init__(self) -> None:
        self._muted = False

    def alert(self, level: str, message: str, **ctx) -> None:
        if self._muted:
            return
        getattr(log, level.lower(), log.info)(f"ALERT: {message}", **ctx)

    def critical(self, message: str, **ctx) -> None:
        self.alert("critical", message, **ctx)

    def warning(self, message: str, **ctx) -> None:
        self.alert("warning", message, **ctx)

    def info(self, message: str, **ctx) -> None:
        self.alert("info", message, **ctx)

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False
