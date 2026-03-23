"""Tests for risk engine v2: EV gating, dynamic stops, capital-fraction sizing."""

import os
import time

import pytest

os.environ.setdefault("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)
os.environ.setdefault("DRY_RUN", "false")

from src.strategy.signal_engine import Decision, Signal
from src.trading.risk_engine import RiskEngine


def _decision(signal=Signal.BUY, confidence=0.6, size_frac=0.5, ev=0.01, token="tok1", price=0.4):
    return Decision(
        token_id=token, signal=signal, confidence=confidence,
        relevance=0.7, expected_value=ev,
        target_price=price, size_fraction=size_frac, reasons=["test"],
    )


class TestRiskEngine:
    def setup_method(self):
        self.risk = RiskEngine()
        self.risk.dry_run = False
        self.risk.state.last_trade_ts = 0

    def test_approve_normal_trade(self):
        ok, reason = self.risk.check(_decision())
        assert ok is True

    def test_reject_hold(self):
        ok, reason = self.risk.check(_decision(signal=Signal.HOLD))
        assert ok is False
        assert reason == "signal_is_hold"

    def test_reject_negative_ev(self):
        ok, reason = self.risk.check(_decision(ev=-0.01))
        assert ok is False
        assert "negative_ev" in reason

    def test_reject_zero_ev(self):
        ok, reason = self.risk.check(_decision(ev=0.0))
        assert ok is False
        assert "negative_ev" in reason

    def test_kill_switch_blocks(self):
        self.risk.activate_kill_switch()
        ok, reason = self.risk.check(_decision())
        assert ok is False

    def test_daily_loss_limit(self):
        self.risk.state.daily_pnl = -200
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert reason == "daily_loss_limit"

    def test_drawdown_10pct_halt(self):
        self.risk.set_capital(100.0)
        self.risk.state.daily_pnl = -15
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert "drawdown" in reason

    def test_cooldown_blocks(self):
        self.risk.cooldown = 60
        self.risk.state.last_trade_ts = time.time()
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert "cooldown" in reason

    def test_max_position_blocks(self):
        from src.trading.risk_engine import Position
        self.risk.state.positions["tok1"] = Position("tok1", "BUY", 999, 0.5)
        ok, reason = self.risk.check(_decision(size_frac=1.0))
        assert ok is False
        assert reason == "max_position_reached"

    def test_capital_scaled_size(self):
        self.risk.set_capital(200.0)
        size = self.risk.compute_order_size(_decision(size_frac=1.0))
        assert size <= 200 * 0.02

    def test_should_exit_time(self):
        from src.trading.risk_engine import Position
        old_pos = Position("tok1", "BUY", 1.0, 0.5)
        old_pos.entry_ts = time.time() - 700  # 700s ago
        self.risk.state.positions["tok1"] = old_pos
        should, reason = self.risk.should_exit_position("tok1", 0.5)
        assert should is True
        assert "time" in reason

    def test_should_exit_momentum_decay(self):
        from src.trading.risk_engine import Position
        pos = Position("tok1", "BUY", 1.0, 0.5)
        self.risk.state.positions["tok1"] = pos
        should, reason = self.risk.should_exit_position("tok1", 0.43)
        assert should is True
        assert "momentum" in reason

    def test_reset_daily(self):
        self.risk.state.daily_pnl = -30
        self.risk.state.kill_switch = True
        self.risk.reset_daily()
        assert self.risk.state.daily_pnl == 0.0
        assert self.risk.state.kill_switch is False
