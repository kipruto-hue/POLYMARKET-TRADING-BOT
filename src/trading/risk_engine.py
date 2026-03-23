"""Pre-trade risk checks and position tracking."""

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
    unrealized_pnl: float = 0.0


@dataclass
class RiskState:
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    open_order_count: int = 0
    last_trade_ts: float = 0.0
    kill_switch: bool = False
    _trade_history: list[dict] = field(default_factory=list)

    def record_fill(self, token_id: str, side: str, price: float, size: float) -> None:
        pos = self.positions.get(token_id)
        if pos is None:
            self.positions[token_id] = Position(
                token_id=token_id, side=side, size=size, avg_entry=price
            )
        elif pos.side == side:
            total_cost = pos.avg_entry * pos.size + price * size
            pos.size += size
            pos.avg_entry = total_cost / pos.size if pos.size else price
        else:
            if size >= pos.size:
                pnl = (price - pos.avg_entry) * pos.size
                if pos.side == "SELL":
                    pnl = -pnl
                self.daily_pnl += pnl
                remainder = size - pos.size
                if remainder > 0:
                    self.positions[token_id] = Position(
                        token_id=token_id, side=side, size=remainder, avg_entry=price
                    )
                else:
                    del self.positions[token_id]
            else:
                pnl = (price - pos.avg_entry) * size
                if pos.side == "SELL":
                    pnl = -pnl
                self.daily_pnl += pnl
                pos.size -= size

        self.last_trade_ts = time.time()
        self._trade_history.append({
            "ts": self.last_trade_ts,
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
        })


class RiskEngine:
    """Gate-keeper that approves or rejects trade decisions."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.max_position = cfg.max_position_size
        self.max_daily_loss = cfg.max_daily_loss
        self.max_open_orders = cfg.max_open_orders
        self.cooldown = cfg.cooldown_seconds
        self.dry_run = cfg.dry_run
        self.state = RiskState()

    def check(self, decision: Decision) -> tuple[bool, str]:
        if self.state.kill_switch:
            return False, "kill_switch_active"

        if decision.signal == Signal.HOLD:
            return False, "signal_is_hold"

        # Daily loss check
        if self.state.daily_pnl <= -self.max_daily_loss:
            log.warning("daily_loss_limit_hit", pnl=self.state.daily_pnl)
            self.state.kill_switch = True
            return False, "daily_loss_limit"

        # Cooldown
        elapsed = time.time() - self.state.last_trade_ts
        if elapsed < self.cooldown:
            return False, f"cooldown ({self.cooldown - elapsed:.0f}s left)"

        # Position size
        desired_size = decision.size_fraction * self.max_position
        pos = self.state.positions.get(decision.token_id)
        current_size = pos.size if pos else 0.0
        if current_size + desired_size > self.max_position:
            desired_size = self.max_position - current_size
            if desired_size <= 0:
                return False, "max_position_reached"

        # Open order limit
        if self.state.open_order_count >= self.max_open_orders:
            return False, "too_many_open_orders"

        # Dry-run mode
        if self.dry_run:
            log.info("dry_run_trade", decision=decision)
            return False, "dry_run_mode"

        return True, "approved"

    def compute_order_size(self, decision: Decision) -> float:
        desired = decision.size_fraction * self.max_position
        pos = self.state.positions.get(decision.token_id)
        current = pos.size if pos else 0.0
        return round(min(desired, self.max_position - current), 2)

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
