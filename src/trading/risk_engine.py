"""
Risk Engine v2 — EV-gated, capital-fraction sizing, dynamic stop conditions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from src.config.settings import get_settings
from src.strategy.signal_engine import Decision, Signal

log = structlog.get_logger()


@dataclass
class Position:
    token_id: str
    side: str
    size: float
    avg_entry: float
    entry_ts: float = field(default_factory=time.time)
    unrealized_pnl: float = 0.0
    trade_id: str = ""

    def age_seconds(self) -> float:
        return time.time() - self.entry_ts


@dataclass
class RiskState:
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    open_order_count: int = 0
    last_trade_ts: float = 0.0
    kill_switch: bool = False
    _trade_history: list[dict] = field(default_factory=list)

    def record_fill(
        self, token_id: str, side: str, price: float, size: float, trade_id: str = ""
    ) -> None:
        pos = self.positions.get(token_id)
        if pos is None:
            self.positions[token_id] = Position(
                token_id=token_id, side=side, size=size,
                avg_entry=price, trade_id=trade_id,
            )
        elif pos.side == side:
            total = pos.avg_entry * pos.size + price * size
            pos.size += size
            pos.avg_entry = total / pos.size if pos.size else price
        else:
            if size >= pos.size:
                pnl = (price - pos.avg_entry) * pos.size * (1 if pos.side == "BUY" else -1)
                self.daily_pnl += pnl
                remainder = size - pos.size
                if remainder > 0:
                    self.positions[token_id] = Position(
                        token_id=token_id, side=side, size=remainder, avg_entry=price
                    )
                else:
                    del self.positions[token_id]
            else:
                pnl = (price - pos.avg_entry) * size * (1 if pos.side == "BUY" else -1)
                self.daily_pnl += pnl
                pos.size -= size

        self.last_trade_ts = time.time()
        self._trade_history.append({
            "ts": self.last_trade_ts, "token_id": token_id,
            "side": side, "price": price, "size": size,
        })

    def stale_positions(self, max_age_seconds: float = 600) -> list[Position]:
        return [p for p in self.positions.values() if p.age_seconds() > max_age_seconds]


class RiskEngine:
    """
    Gate-keeper with EV gating, dynamic stop conditions, and capital-fraction sizing.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.max_position = cfg.max_position_size
        self.max_daily_loss = cfg.max_daily_loss
        self.max_open_orders = cfg.max_open_orders
        self.cooldown = cfg.cooldown_seconds
        self.dry_run = cfg.dry_run
        self.state = RiskState()

        # Dynamic stop parameters
        self.max_position_age_seconds: float = 600   # 10-min time-based exit
        self.momentum_decay_threshold: float = 0.02  # exit if momentum reverses this much

        # Capital tracking
        self._starting_capital: float = 0.0          # set externally once known
        self._capital: float = 0.0

    def set_capital(self, amount: float) -> None:
        self._starting_capital = amount
        self._capital = amount
        log.info("capital_set", amount=amount)

    def check(self, decision: Decision) -> tuple[bool, str]:
        if self.state.kill_switch:
            return False, "kill_switch_active"

        if decision.signal == Signal.HOLD:
            return False, "signal_is_hold"

        # Daily loss hard stop
        if self.state.daily_pnl <= -self.max_daily_loss:
            log.warning("daily_loss_limit_hit", pnl=self.state.daily_pnl)
            self.state.kill_switch = True
            return False, "daily_loss_limit"

        # Drawdown check against starting capital
        if self._capital > 0:
            drawdown_pct = abs(self.state.daily_pnl) / self._capital
            if drawdown_pct > 0.10:
                log.warning("10pct_drawdown_halt", drawdown_pct=round(drawdown_pct, 3))
                self.state.kill_switch = True
                return False, "drawdown_10pct_halt"

        # Cooldown
        elapsed = time.time() - self.state.last_trade_ts
        if self.cooldown > 0 and elapsed < self.cooldown:
            return False, f"cooldown({self.cooldown - elapsed:.0f}s)"

        # Position size
        desired = decision.size_fraction * self.max_position
        pos = self.state.positions.get(decision.token_id)
        current = pos.size if pos else 0.0
        if current + desired > self.max_position:
            if (self.max_position - current) <= 0:
                return False, "max_position_reached"

        # Open order limit
        if self.state.open_order_count >= self.max_open_orders:
            return False, "too_many_open_orders"

        # EV gate
        if decision.expected_value <= 0:
            return False, f"negative_ev({decision.expected_value:.5f})"

        # Dry run
        if self.dry_run:
            log.info("dry_run_trade", signal=decision.signal.value,
                     confidence=decision.confidence, ev=decision.expected_value)
            return False, "dry_run_mode"

        return True, "approved"

    def compute_order_size(self, decision: Decision) -> float:
        """
        Size = min(decision fraction, remaining room, 2% of capital if known).
        """
        max_by_capital = self._capital * 0.02 if self._capital > 0 else self.max_position
        desired = decision.size_fraction * min(self.max_position, max_by_capital)
        pos = self.state.positions.get(decision.token_id)
        current = pos.size if pos else 0.0
        room = self.max_position - current
        return round(max(0.0, min(desired, room, self.max_position)), 2)

    def should_exit_position(self, token_id: str, current_price: float) -> tuple[bool, str]:
        """Evaluate dynamic stop conditions for an open position."""
        pos = self.state.positions.get(token_id)
        if pos is None:
            return False, "no_position"

        # Time-based exit
        if pos.age_seconds() > self.max_position_age_seconds:
            return True, f"time_exit({pos.age_seconds():.0f}s)"

        # Momentum decay: adverse move exceeds threshold
        entry = pos.avg_entry
        if pos.side == "BUY" and entry > 0:
            move = (current_price - entry) / entry
            if move < -self.momentum_decay_threshold:
                return True, f"momentum_decay({move:.3f})"
        elif pos.side == "SELL" and entry > 0:
            move = (current_price - entry) / entry
            if move > self.momentum_decay_threshold:
                return True, f"momentum_reversal({move:.3f})"

        return False, "hold_position"

    def activate_kill_switch(self) -> None:
        self.state.kill_switch = True
        log.critical("kill_switch_activated")

    def deactivate_kill_switch(self) -> None:
        self.state.kill_switch = False
        log.info("kill_switch_deactivated")

    def reset_daily(self) -> None:
        self.state.daily_pnl = 0.0
        self.state.kill_switch = False
        log.info("daily_risk_reset")
