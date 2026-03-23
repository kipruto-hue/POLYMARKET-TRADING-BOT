"""Signal engine: produces BUY / SELL / HOLD decisions from feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
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
    confidence: float       # 0.0 – 1.0
    target_price: float
    size_fraction: float    # 0.0 – 1.0 of max position
    reasons: list[str]


# Tunable weights (v1: hand-set; v2: learn from replay)
WEIGHTS = {
    "momentum": 0.30,
    "rsi": 0.15,
    "book_imbalance": 0.20,
    "news_sentiment": 0.20,
    "volume_trend": 0.15,
}

# Thresholds (lowered for high-frequency small trades)
BUY_THRESHOLD = 0.10
SELL_THRESHOLD = -0.10
MIN_SPREAD_OK = 0.06  # allow slightly wider spreads for more opportunities


def score_features(fv: FeatureVector) -> tuple[float, list[str]]:
    """Return a composite score in [-1, 1] and list of contributing reasons."""
    reasons: list[str] = []
    components: dict[str, float] = {}

    # Momentum: positive momentum → bullish
    mom = (fv.momentum_5 + fv.momentum_3) / 2.0
    mom_score = max(-1.0, min(1.0, mom * 20))  # scale small moves
    components["momentum"] = mom_score
    if abs(mom_score) > 0.2:
        reasons.append(f"momentum={mom_score:+.2f}")

    # RSI: >70 overbought (bearish), <30 oversold (bullish)
    if fv.rsi > 70:
        rsi_score = -((fv.rsi - 70) / 30.0)
    elif fv.rsi < 30:
        rsi_score = (30 - fv.rsi) / 30.0
    else:
        rsi_score = 0.0
    components["rsi"] = rsi_score
    if abs(rsi_score) > 0.1:
        reasons.append(f"rsi={fv.rsi:.0f}")

    # Book imbalance: positive means more bids
    bi_score = max(-1.0, min(1.0, fv.book_imbalance * 2))
    components["book_imbalance"] = bi_score
    if abs(bi_score) > 0.15:
        reasons.append(f"book_imbal={fv.book_imbalance:+.2f}")

    # News sentiment
    ns_score = max(-1.0, min(1.0, fv.news_sentiment * 3))
    components["news_sentiment"] = ns_score
    if abs(ns_score) > 0.1:
        reasons.append(f"news={fv.news_sentiment:+.2f}")

    # Volume trend: >1.5 means unusual activity → amplify signal
    vt_multiplier = max(-1.0, min(1.0, (fv.volume_trend - 1.0)))
    components["volume_trend"] = vt_multiplier
    if fv.volume_trend > 1.5:
        reasons.append(f"vol_surge={fv.volume_trend:.1f}x")

    total = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return max(-1.0, min(1.0, total)), reasons


def generate_decision(fv: FeatureVector) -> Decision:
    score, reasons = score_features(fv)
    confidence = abs(score)

    # Spread filter: if spread is too wide, hold
    if fv.spread > MIN_SPREAD_OK:
        return Decision(
            token_id=fv.token_id,
            signal=Signal.HOLD,
            confidence=0.0,
            target_price=fv.current_price,
            size_fraction=0.0,
            reasons=["spread_too_wide"],
        )

    if score >= BUY_THRESHOLD:
        signal = Signal.BUY
        target = fv.current_price  # limit at current (GTC)
    elif score <= SELL_THRESHOLD:
        signal = Signal.SELL
        target = fv.current_price
    else:
        signal = Signal.HOLD
        target = fv.current_price

    size_frac = min(1.0, confidence * 1.5) if signal != Signal.HOLD else 0.0

    decision = Decision(
        token_id=fv.token_id,
        signal=signal,
        confidence=round(confidence, 3),
        target_price=round(target, 4),
        size_fraction=round(size_frac, 3),
        reasons=reasons,
    )
    log.info(
        "signal_generated",
        token=fv.token_id[:12],
        signal=signal.value,
        score=round(score, 3),
        confidence=decision.confidence,
    )
    return decision
