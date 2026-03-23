"""Lightweight telemetry, health checks, and alerting hooks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class BotMetrics:
    start_time: float = field(default_factory=time.time)
    cycles_completed: int = 0
    signals_generated: int = 0
    trades_placed: int = 0
    trades_rejected: int = 0
    errors: int = 0
    last_cycle_ts: float = 0.0
    last_error: str = ""

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "uptime_s": round(self.uptime_seconds),
            "cycles": self.cycles_completed,
            "signals": self.signals_generated,
            "trades_placed": self.trades_placed,
            "trades_rejected": self.trades_rejected,
            "errors": self.errors,
            "last_cycle": self.last_cycle_ts,
            "last_error": self.last_error,
        }


class AlertManager:
    """Simple alerting — prints to structured log.  Swap in Telegram/Slack/email later."""

    def __init__(self) -> None:
        self._muted = False

    def alert(self, level: str, message: str, **ctx) -> None:
        if self._muted:
            return
        log.msg(f"ALERT [{level}]: {message}", **ctx)

    def critical(self, message: str, **ctx) -> None:
        self.alert("CRITICAL", message, **ctx)

    def warning(self, message: str, **ctx) -> None:
        self.alert("WARNING", message, **ctx)

    def info(self, message: str, **ctx) -> None:
        self.alert("INFO", message, **ctx)

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False
