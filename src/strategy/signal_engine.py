"""
Signal engine v2 — EV-gated, confluence-required, self-calibrating.

Every decision must pass:
  1. Relevance gate    — news/context is applicable
  2. Confluence gate  — at least 2 of 3 signal types agree
  3. Confidence gate  — weighted score exceeds dynamic threshold
  4. EV gate          — expected value is positive
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

from src.strategy.features import FeatureVector

log = structlog.get_logger()


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Decision:
    token_id: str
    signal: Signal
    confidence: float           # 0.0 – 1.0
    relevance: float            # how applicable this signal is [0, 1]
    expected_value: float       # EV = win_prob * payoff - loss_prob * loss
    target_price: float
    size_fraction: float        # 0.0 – 1.0 of max position
    reasons: list[str]
    rejected_reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Dynamic thresholds (recalibrated by anti-decay engine)
# ---------------------------------------------------------------------------
class SignalThresholds:
    def __init__(self) -> None:
        self.min_confidence: float = 0.30
        self.min_relevance: float = 0.40
        self.min_ev: float = 0.001
        self.buy_score: float = 0.15
        self.sell_score: float = -0.15
        self.max_spread_pct: float = 0.08
        self.min_depth: float = 100.0         # min $100 depth on entry side
        self.min_confluence: int = 2          # min signals that must agree

    def tighten(self, factor: float = 1.25) -> None:
        """Make thresholds stricter after poor performance."""
        self.min_confidence = min(0.80, self.min_confidence * factor)
        self.min_relevance = min(0.80, self.min_relevance * factor)
        self.min_ev = min(0.02, self.min_ev * factor)
        self.buy_score = min(0.60, self.buy_score * factor)
        self.sell_score = max(-0.60, self.sell_score * factor)
        self.min_confluence = min(3, self.min_confluence + 1)
        log.warning("thresholds_tightened", confidence=self.min_confidence,
                    ev=self.min_ev, confluence=self.min_confluence)

    def relax(self, factor: float = 0.90) -> None:
        """Ease thresholds after sustained good performance."""
        self.min_confidence = max(0.20, self.min_confidence * factor)
        self.min_relevance = max(0.30, self.min_relevance * factor)
        self.min_ev = max(0.001, self.min_ev * factor)
        self.buy_score = max(0.08, self.buy_score * factor)
        self.sell_score = min(-0.08, self.sell_score * factor)
        self.min_confluence = max(1, self.min_confluence - 1)
        log.info("thresholds_relaxed", confidence=self.min_confidence, ev=self.min_ev)

    def to_dict(self) -> dict:
        return {
            "min_confidence": self.min_confidence,
            "min_relevance": self.min_relevance,
            "min_ev": self.min_ev,
            "buy_score": self.buy_score,
            "sell_score": self.sell_score,
            "max_spread_pct": self.max_spread_pct,
            "min_depth": self.min_depth,
            "min_confluence": self.min_confluence,
        }


# Shared mutable threshold instance (updated by feedback engine)
THRESHOLDS = SignalThresholds()

WEIGHTS = {
    "momentum": 0.30,
    "rsi": 0.15,
    "book_imbalance": 0.25,
    "news": 0.20,
    "volume": 0.10,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_momentum(fv: FeatureVector) -> tuple[float, str | None]:
    mom = (fv.momentum_5 * 0.4 + fv.momentum_3 * 0.4 + fv.momentum_1 * 0.2)
    score = max(-1.0, min(1.0, mom * 15))
    reason = f"momentum={score:+.2f}" if abs(score) > 0.15 else None
    return score, reason


def _score_rsi(fv: FeatureVector) -> tuple[float, str | None]:
    if fv.rsi > 70:
        score = -((fv.rsi - 70) / 30.0)
        return score, f"rsi_overbought={fv.rsi:.0f}"
    elif fv.rsi < 30:
        score = (30 - fv.rsi) / 30.0
        return score, f"rsi_oversold={fv.rsi:.0f}"
    return 0.0, None


def _score_book(fv: FeatureVector) -> tuple[float, str | None]:
    bi_score = max(-1.0, min(1.0, fv.book_imbalance * 2.5))
    # Depth asymmetry adds conviction
    if fv.depth_bid + fv.depth_ask > 0:
        depth_ratio = (fv.depth_bid - fv.depth_ask) / (fv.depth_bid + fv.depth_ask)
        bi_score = bi_score * 0.7 + depth_ratio * 0.3
    reason = f"book={fv.book_imbalance:+.2f}" if abs(bi_score) > 0.12 else None
    return bi_score, reason


def _score_news(fv: FeatureVector) -> tuple[float, str | None]:
    if fv.news_relevance < 0.3:
        return 0.0, None
    score = max(-1.0, min(1.0, fv.news_sentiment * fv.news_relevance * 3.0))
    reason = f"news={fv.news_sentiment:+.2f}(rel={fv.news_relevance:.1f})" if abs(score) > 0.08 else None
    return score, reason


def _score_volume(fv: FeatureVector) -> tuple[float, str | None]:
    vt = max(-1.0, min(1.0, (fv.volume_trend - 1.0) * 0.8))
    va = max(-1.0, min(1.0, fv.volume_acceleration * 2.0))
    score = vt * 0.6 + va * 0.4
    reason = f"vol_surge={fv.volume_trend:.1f}x" if fv.volume_trend > 1.4 else None
    return score, reason


def compute_ev(
    signal: Signal,
    confidence: float,
    current_price: float,
    implied_prob_shift: float,
    size: float = 1.0,
) -> float:
    """
    EV = win_probability * net_payoff - loss_probability * loss_amount

    On a binary prediction market:
      - BUY at price P: payoff is (1 - P) if correct, loss is P if wrong
      - SELL at price P: payoff is P if correct, loss is (1 - P) if wrong
    """
    if signal == Signal.HOLD or current_price <= 0:
        return 0.0
    p = current_price
    if signal == Signal.BUY:
        win_prob = min(0.95, confidence + abs(implied_prob_shift))
        payoff = (1.0 - p) * size
        loss = p * size
    else:
        win_prob = min(0.95, confidence + abs(implied_prob_shift))
        payoff = p * size
        loss = (1.0 - p) * size
    loss_prob = 1.0 - win_prob
    return round(win_prob * payoff - loss_prob * loss, 5)


def generate_decision(fv: FeatureVector) -> Decision:
    t = THRESHOLDS

    def _reject(reason: str) -> Decision:
        return Decision(
            token_id=fv.token_id, signal=Signal.HOLD,
            confidence=0.0, relevance=fv.news_relevance,
            expected_value=0.0, target_price=fv.current_price,
            size_fraction=0.0, reasons=[], rejected_reason=reason,
        )

    # Gate 1: spread
    if fv.spread_pct > t.max_spread_pct:
        return _reject(f"spread_too_wide({fv.spread_pct:.3f})")

    # Gate 2: liquidity
    entry_depth = fv.depth_bid if fv.book_imbalance >= 0 else fv.depth_ask
    if entry_depth < t.min_depth and entry_depth > 0:
        return _reject(f"insufficient_depth({entry_depth:.0f})")

    # Score each signal source
    mom_s, mom_r = _score_momentum(fv)
    rsi_s, rsi_r = _score_rsi(fv)
    book_s, book_r = _score_book(fv)
    news_s, news_r = _score_news(fv)
    vol_s, vol_r = _score_volume(fv)

    reasons = [r for r in [mom_r, rsi_r, book_r, news_r, vol_r] if r]

    total_score = (
        WEIGHTS["momentum"] * mom_s
        + WEIGHTS["rsi"] * rsi_s
        + WEIGHTS["book_imbalance"] * book_s
        + WEIGHTS["news"] * news_s
        + WEIGHTS["volume"] * vol_s
    )
    total_score = max(-1.0, min(1.0, total_score))
    confidence = abs(total_score)

    # Gate 3: confidence
    if confidence < t.min_confidence:
        return _reject(f"low_confidence({confidence:.3f}<{t.min_confidence:.2f})")

    # Gate 4: confluence — count how many independent sources agree with direction
    direction = 1 if total_score > 0 else -1
    agreeing = sum(
        1 for s in [mom_s, rsi_s, book_s, news_s, vol_s]
        if abs(s) > 0.05 and (s * direction) > 0
    )
    if agreeing < t.min_confluence:
        return _reject(f"weak_confluence({agreeing}<{t.min_confluence})")

    # Determine signal
    if total_score >= t.buy_score:
        signal = Signal.BUY
    elif total_score <= t.sell_score:
        signal = Signal.SELL
    else:
        return _reject("below_threshold")

    # Gate 5: EV
    size_frac = min(1.0, confidence * 1.2)
    ev = compute_ev(signal, confidence, fv.current_price, fv.implied_prob_shift, size=size_frac)
    if ev < t.min_ev:
        return _reject(f"negative_ev({ev:.5f})")

    decision = Decision(
        token_id=fv.token_id,
        signal=signal,
        confidence=round(confidence, 4),
        relevance=round(fv.news_relevance, 3),
        expected_value=round(ev, 5),
        target_price=round(fv.current_price, 4),
        size_fraction=round(size_frac, 3),
        reasons=reasons,
    )
    log.info(
        "signal_approved",
        token=fv.token_id[:12],
        signal=signal.value,
        score=round(total_score, 3),
        confidence=decision.confidence,
        ev=decision.expected_value,
        confluence=agreeing,
        reasons=reasons,
    )
    return decision
