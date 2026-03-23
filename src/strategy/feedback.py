"""
Adaptive Feedback Engine — Anti-Decay + Learning Loop.

Responsibilities:
  1. Track every trade outcome and label it (correct/incorrect, early/late, quality)
  2. Detect performance degradation and trigger diagnostic mode
  3. Recalibrate signal thresholds based on rolling accuracy
  4. Maintain the meta-question: "Should this signal be trusted right now?"
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

import structlog

from src.strategy.signal_engine import THRESHOLDS, Signal

log = structlog.get_logger()


class TradeOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    PENDING = "pending"
    CANCELLED = "cancelled"


class FailureType(str, Enum):
    FALSE_SIGNAL = "false_signal"
    LATE_ENTRY = "late_entry"
    POOR_LIQUIDITY = "poor_liquidity"
    OVERTRADING = "overtrading"
    UNKNOWN = "unknown"


class BotMode(str, Enum):
    NORMAL = "normal"
    DIAGNOSTIC = "diagnostic"
    HALTED = "halted"


@dataclass
class TradeRecord:
    trade_id: str
    token_id: str
    signal: str
    confidence: float
    ev_estimated: float
    entry_price: float
    entry_ts: float
    size: float
    reasons: list[str]

    exit_price: float = 0.0
    exit_ts: float = 0.0
    pnl: float = 0.0
    outcome: TradeOutcome = TradeOutcome.PENDING

    quality_label: str = ""       # high / medium / low
    timing_label: str = ""        # early / on-time / late
    slippage: float = 0.0         # actual vs expected price


@dataclass
class PerformanceWindow:
    """Rolling window of last N trade outcomes."""
    window_size: int = 20
    _records: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=20))

    def add(self, record: TradeRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[TradeRecord]:
        return list(self._records)

    @property
    def completed(self) -> list[TradeRecord]:
        return [r for r in self._records if r.outcome != TradeOutcome.PENDING]

    @property
    def win_rate(self) -> float:
        done = self.completed
        if not done:
            return 0.5
        return sum(1 for r in done if r.outcome == TradeOutcome.WIN) / len(done)

    @property
    def avg_ev(self) -> float:
        done = self.completed
        if not done:
            return 0.0
        return sum(r.ev_estimated for r in done) / len(done)

    @property
    def consecutive_losses(self) -> int:
        streak = 0
        for r in reversed(self.completed):
            if r.outcome == TradeOutcome.LOSS:
                streak += 1
            else:
                break
        return streak

    @property
    def avg_slippage(self) -> float:
        done = self.completed
        if not done:
            return 0.0
        return sum(abs(r.slippage) for r in done) / len(done)

    @property
    def signal_accuracy(self) -> dict[str, float]:
        by_signal: dict[str, list[bool]] = {}
        for r in self.completed:
            by_signal.setdefault(r.signal, []).append(r.outcome == TradeOutcome.WIN)
        return {sig: sum(wins) / len(wins) for sig, wins in by_signal.items()}

    def to_dict(self) -> dict:
        return {
            "total_trades": len(self.completed),
            "win_rate": round(self.win_rate, 3),
            "avg_ev": round(self.avg_ev, 5),
            "consecutive_losses": self.consecutive_losses,
            "avg_slippage": round(self.avg_slippage, 5),
            "signal_accuracy": self.signal_accuracy,
        }


class FeedbackEngine:
    """Tracks trade outcomes, detects decay, recalibrates signal engine."""

    LOSS_STREAK_HALT = 4           # halt after this many consecutive losses
    DIAGNOSTIC_TRADES = 5          # run this many paper-style observations in diag mode
    RELAX_AFTER_WINS = 8           # relax thresholds after sustained wins
    TIGHTEN_WIN_RATE = 0.45        # tighten if win rate drops below this

    def __init__(self) -> None:
        self.window = PerformanceWindow(window_size=20)
        self.mode: BotMode = BotMode.NORMAL
        self._diagnostic_observations: int = 0
        self._pending: dict[str, TradeRecord] = {}
        self._total_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._max_drawdown: float = 0.0
        self._total_trades: int = 0
        self._wins: int = 0

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------
    def record_entry(
        self,
        trade_id: str,
        token_id: str,
        signal: str,
        confidence: float,
        ev_estimated: float,
        entry_price: float,
        size: float,
        reasons: list[str],
    ) -> None:
        rec = TradeRecord(
            trade_id=trade_id,
            token_id=token_id,
            signal=signal,
            confidence=confidence,
            ev_estimated=ev_estimated,
            entry_price=entry_price,
            entry_ts=time.time(),
            size=size,
            reasons=reasons,
        )
        self._pending[trade_id] = rec
        log.info("trade_entry_recorded", trade_id=trade_id, signal=signal, price=entry_price)

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        slippage: float = 0.0,
    ) -> None:
        rec = self._pending.pop(trade_id, None)
        if rec is None:
            return

        rec.exit_price = exit_price
        rec.exit_ts = time.time()
        rec.pnl = pnl
        rec.slippage = slippage
        rec.outcome = TradeOutcome.WIN if pnl > 0 else TradeOutcome.LOSS

        # Label timing
        hold_time = rec.exit_ts - rec.entry_ts
        if hold_time < 60:
            rec.timing_label = "early"
        elif hold_time < 600:
            rec.timing_label = "on-time"
        else:
            rec.timing_label = "late"

        # Label quality
        if abs(pnl) / max(rec.size * rec.entry_price, 1e-6) > 0.05:
            rec.quality_label = "high"
        elif abs(pnl) / max(rec.size * rec.entry_price, 1e-6) > 0.01:
            rec.quality_label = "medium"
        else:
            rec.quality_label = "low"

        self.window.add(rec)
        self._total_pnl += pnl
        self._total_trades += 1
        if pnl > 0:
            self._wins += 1

        # Drawdown tracking
        if self._total_pnl > self._peak_pnl:
            self._peak_pnl = self._total_pnl
        drawdown = self._peak_pnl - self._total_pnl
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        log.info(
            "trade_exit_recorded",
            trade_id=trade_id,
            pnl=round(pnl, 4),
            outcome=rec.outcome.value,
            timing=rec.timing_label,
            quality=rec.quality_label,
        )
        self._evaluate_performance()

    # ------------------------------------------------------------------
    # Performance watchdog
    # ------------------------------------------------------------------
    def _evaluate_performance(self) -> None:
        streak = self.window.consecutive_losses
        win_rate = self.window.win_rate
        n = len(self.window.completed)

        if streak >= self.LOSS_STREAK_HALT:
            if self.mode != BotMode.DIAGNOSTIC:
                log.warning(
                    "consecutive_loss_streak_triggered",
                    streak=streak,
                    entering="diagnostic_mode",
                )
                self._enter_diagnostic()
            return

        if n >= 10 and win_rate < self.TIGHTEN_WIN_RATE:
            log.warning("win_rate_degraded", win_rate=win_rate, tightening=True)
            THRESHOLDS.tighten(1.15)
            return

        if n >= self.RELAX_AFTER_WINS:
            recent = self.window.completed[-self.RELAX_AFTER_WINS:]
            sustained_wins = all(r.outcome == TradeOutcome.WIN for r in recent[-4:])
            if sustained_wins and win_rate > 0.65 and self.mode == BotMode.NORMAL:
                log.info("sustained_good_performance, relaxing thresholds slightly")
                THRESHOLDS.relax(0.95)

    def _enter_diagnostic(self) -> None:
        self.mode = BotMode.DIAGNOSTIC
        self._diagnostic_observations = 0
        THRESHOLDS.tighten(1.30)

        failure = self._diagnose_failure_type()
        log.warning("diagnostic_mode_entered", failure_type=failure.value,
                    thresholds=THRESHOLDS.to_dict())

        if failure == FailureType.FALSE_SIGNAL:
            THRESHOLDS.min_confidence = min(0.85, THRESHOLDS.min_confidence * 1.2)
        elif failure == FailureType.LATE_ENTRY:
            THRESHOLDS.min_confluence = min(3, THRESHOLDS.min_confluence + 1)
        elif failure == FailureType.POOR_LIQUIDITY:
            THRESHOLDS.min_depth *= 2
        elif failure == FailureType.OVERTRADING:
            THRESHOLDS.min_ev = min(0.02, THRESHOLDS.min_ev * 2)

    def _diagnose_failure_type(self) -> FailureType:
        recent = self.window.completed[-5:]
        if not recent:
            return FailureType.UNKNOWN

        high_conf_losses = sum(
            1 for r in recent if r.outcome == TradeOutcome.LOSS and r.confidence > 0.5
        )
        late_entries = sum(1 for r in recent if r.timing_label == "late")
        low_quality = sum(1 for r in recent if r.quality_label == "low")

        if high_conf_losses >= 3:
            return FailureType.FALSE_SIGNAL
        if late_entries >= 3:
            return FailureType.LATE_ENTRY
        if low_quality >= 3:
            return FailureType.OVERTRADING
        return FailureType.UNKNOWN

    # ------------------------------------------------------------------
    # Trading gate — meta-model question: "Should I trade now?"
    # ------------------------------------------------------------------
    def is_trading_allowed(self) -> tuple[bool, str]:
        if self.mode == BotMode.HALTED:
            return False, "system_halted"

        if self.mode == BotMode.DIAGNOSTIC:
            self._diagnostic_observations += 1
            if self._diagnostic_observations >= self.DIAGNOSTIC_TRADES:
                return self._evaluate_resume()
            return False, f"diagnostic_mode ({self._diagnostic_observations}/{self.DIAGNOSTIC_TRADES})"

        return True, "normal"

    def _evaluate_resume(self) -> tuple[bool, str]:
        win_rate = self.window.win_rate
        streak = self.window.consecutive_losses
        n = len(self.window.completed)

        if win_rate >= 0.50 and streak < 2 and n >= 3:
            self.mode = BotMode.NORMAL
            self._diagnostic_observations = 0
            log.info("resuming_normal_trading", win_rate=win_rate,
                     thresholds=THRESHOLDS.to_dict())
            return True, "resumed_after_diagnostic"

        if win_rate < 0.40:
            self.mode = BotMode.HALTED
            log.critical("bot_halted_after_diagnostic", win_rate=win_rate)
            return False, "halted_post_diagnostic"

        self._diagnostic_observations = 0
        log.warning("extending_diagnostic_mode", win_rate=win_rate)
        return False, "diagnostic_mode_extended"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def overall_win_rate(self) -> float:
        if self._total_trades == 0:
            return 0.0
        return self._wins / self._total_trades

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "total_trades": self._total_trades,
            "overall_win_rate": round(self.overall_win_rate, 3),
            "total_pnl": round(self._total_pnl, 4),
            "max_drawdown": round(self._max_drawdown, 4),
            "rolling_window": self.window.to_dict(),
            "thresholds": THRESHOLDS.to_dict(),
        }
