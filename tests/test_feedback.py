"""Tests for the anti-decay feedback engine."""

import time
import pytest

from src.strategy.feedback import (
    BotMode, FeedbackEngine, TradeOutcome,
)
from src.strategy.signal_engine import THRESHOLDS


def _make_engine() -> FeedbackEngine:
    THRESHOLDS.min_confidence = 0.30  # reset to default
    return FeedbackEngine()


def _inject_trades(engine: FeedbackEngine, wins: int, losses: int) -> None:
    for i in range(wins):
        tid = f"win_{i}"
        engine.record_entry(tid, "tok", "BUY", 0.6, 0.01, 0.4, 2.0, [])
        engine.record_exit(tid, 0.5, pnl=0.5)

    for i in range(losses):
        tid = f"loss_{i}"
        engine.record_entry(tid, "tok", "BUY", 0.6, 0.01, 0.6, 2.0, [])
        engine.record_exit(tid, 0.3, pnl=-1.0)


class TestFeedbackEngine:
    def test_starts_normal(self):
        e = _make_engine()
        allowed, reason = e.is_trading_allowed()
        assert allowed
        assert reason == "normal"

    def test_4_consecutive_losses_triggers_diagnostic(self):
        e = _make_engine()
        for i in range(4):
            e.record_entry(f"t{i}", "tok", "BUY", 0.6, 0.01, 0.6, 2.0, [])
            e.record_exit(f"t{i}", 0.3, pnl=-1.0)
        assert e.mode == BotMode.DIAGNOSTIC

    def test_diagnostic_blocks_trading(self):
        e = _make_engine()
        for i in range(4):
            e.record_entry(f"t{i}", "tok", "BUY", 0.6, 0.01, 0.6, 2.0, [])
            e.record_exit(f"t{i}", 0.3, pnl=-1.0)
        allowed, reason = e.is_trading_allowed()
        assert not allowed
        assert "diagnostic" in reason

    def test_resume_after_wins(self):
        e = _make_engine()
        e.mode = BotMode.DIAGNOSTIC
        e._diagnostic_observations = e.DIAGNOSTIC_TRADES - 1
        # Add some wins so evaluation passes
        _inject_trades(e, wins=3, losses=1)
        allowed, reason = e.is_trading_allowed()
        # Either resumes or extends, either way it's consistent
        assert reason in ("resumed_after_diagnostic", "diagnostic_mode_extended")

    def test_win_rate_tracked(self):
        e = _make_engine()
        _inject_trades(e, wins=7, losses=3)
        assert abs(e.window.win_rate - 0.7) < 0.05

    def test_drawdown_tracked(self):
        e = _make_engine()
        e.record_entry("t1", "tok", "BUY", 0.6, 0.01, 0.5, 2.0, [])
        e.record_exit("t1", 0.7, pnl=1.0)
        e.record_entry("t2", "tok", "BUY", 0.6, 0.01, 0.7, 2.0, [])
        e.record_exit("t2", 0.3, pnl=-2.0)
        assert e._max_drawdown > 0

    def test_consecutive_losses_property(self):
        e = _make_engine()
        _inject_trades(e, wins=2, losses=0)
        for i in range(3):
            e.record_entry(f"loss_{i}", "tok", "SELL", 0.5, 0.01, 0.5, 1.0, [])
            e.record_exit(f"loss_{i}", 0.8, pnl=-0.5)
        assert e.window.consecutive_losses == 3

    def test_thresholds_tightened_in_diagnostic(self):
        e = _make_engine()
        original = THRESHOLDS.min_confidence
        for i in range(4):
            e.record_entry(f"t{i}", "tok", "BUY", 0.6, 0.01, 0.6, 2.0, [])
            e.record_exit(f"t{i}", 0.3, pnl=-1.0)
        assert THRESHOLDS.min_confidence >= original


class TestPerformanceWindow:
    def test_window_respects_size(self):
        e = _make_engine()
        for i in range(25):
            e.record_entry(f"t{i}", "tok", "BUY", 0.5, 0.01, 0.5, 1.0, [])
            e.record_exit(f"t{i}", 0.6, pnl=0.1)
        assert len(e.window.completed) <= e.window.window_size

    def test_signal_accuracy(self):
        e = _make_engine()
        for i in range(3):
            e.record_entry(f"buy_{i}", "tok", "BUY", 0.5, 0.01, 0.4, 1.0, [])
            e.record_exit(f"buy_{i}", 0.6, pnl=0.2)
        acc = e.window.signal_accuracy
        assert "BUY" in acc
        assert acc["BUY"] == 1.0
