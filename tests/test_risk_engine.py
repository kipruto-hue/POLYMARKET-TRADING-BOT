"""Tests for risk engine pre-trade checks."""

import os
import time

import pytest

os.environ.setdefault("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)
os.environ.setdefault("DRY_RUN", "false")

from src.strategy.signal_engine import Decision, Signal
from src.trading.risk_engine import RiskEngine


def _decision(signal=Signal.BUY, confidence=0.5, size_frac=0.5, token="tok1"):
    return Decision(
        token_id=token,
        signal=signal,
        confidence=confidence,
        target_price=0.50,
        size_fraction=size_frac,
        reasons=["test"],
    )


class TestRiskEngine:
    def setup_method(self):
        self.risk = RiskEngine()
        self.risk.dry_run = False
        self.risk.state.last_trade_ts = 0  # bypass cooldown

    def test_approve_normal_trade(self):
        ok, reason = self.risk.check(_decision())
        assert ok is True
        assert reason == "approved"

    def test_reject_hold_signal(self):
        ok, reason = self.risk.check(_decision(signal=Signal.HOLD))
        assert ok is False
        assert reason == "signal_is_hold"

    def test_kill_switch_blocks(self):
        self.risk.activate_kill_switch()
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert reason == "kill_switch_active"

    def test_daily_loss_limit(self):
        self.risk.state.daily_pnl = -200
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert reason == "daily_loss_limit"

    def test_cooldown_blocks(self):
        self.risk.cooldown = 60  # force cooldown for this test
        self.risk.state.last_trade_ts = time.time()
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert "cooldown" in reason

    def test_max_position_reached(self):
        self.risk.state.positions["tok1"] = type(
            "P", (), {"size": 999, "side": "BUY", "avg_entry": 0.5}
        )()
        ok, reason = self.risk.check(_decision(size_frac=1.0))
        assert ok is False
        assert reason == "max_position_reached"

    def test_too_many_open_orders(self):
        self.risk.state.open_order_count = 999
        ok, reason = self.risk.check(_decision())
        assert ok is False
        assert reason == "too_many_open_orders"

    def test_compute_order_size_capped(self):
        size = self.risk.compute_order_size(_decision(size_frac=1.0))
        assert size <= self.risk.max_position

    def test_reset_daily(self):
        self.risk.state.daily_pnl = -50
        self.risk.state.kill_switch = True
        self.risk.reset_daily()
        assert self.risk.state.daily_pnl == 0.0
        assert self.risk.state.kill_switch is False
