"""Tests for signal engine scoring and decision logic."""

import pytest

from src.strategy.features import FeatureVector
from src.strategy.signal_engine import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    Signal,
    generate_decision,
    score_features,
)


def _fv(**overrides) -> FeatureVector:
    defaults = dict(
        token_id="test_token",
        momentum_5=0.0,
        momentum_3=0.0,
        volatility=0.01,
        rsi=50.0,
        spread=0.01,
        book_imbalance=0.0,
        news_sentiment=0.0,
        current_price=0.50,
        volume_trend=1.0,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


class TestScoring:
    def test_neutral_features_produce_low_score(self):
        score, _ = score_features(_fv())
        assert -0.1 < score < 0.1

    def test_strong_bullish_momentum_pushes_positive(self):
        score, reasons = score_features(_fv(momentum_5=0.05, momentum_3=0.04))
        assert score > 0.2
        assert any("momentum" in r for r in reasons)

    def test_strong_bearish_momentum_pushes_negative(self):
        score, _ = score_features(_fv(momentum_5=-0.06, momentum_3=-0.05))
        assert score < -0.2

    def test_overbought_rsi_bearish(self):
        score, reasons = score_features(_fv(rsi=85))
        assert score < 0
        assert any("rsi" in r for r in reasons)

    def test_oversold_rsi_bullish(self):
        score, _ = score_features(_fv(rsi=15))
        assert score > 0

    def test_positive_sentiment_adds_bullish_bias(self):
        base, _ = score_features(_fv())
        boosted, _ = score_features(_fv(news_sentiment=0.8))
        assert boosted > base


class TestDecision:
    def test_hold_when_spread_too_wide(self):
        d = generate_decision(_fv(spread=0.10, momentum_5=0.1))
        assert d.signal == Signal.HOLD
        assert "spread_too_wide" in d.reasons

    def test_buy_signal_when_bullish(self):
        d = generate_decision(
            _fv(momentum_5=0.05, momentum_3=0.04, book_imbalance=0.5, news_sentiment=0.5)
        )
        assert d.signal == Signal.BUY
        assert d.confidence > 0
        assert d.size_fraction > 0

    def test_sell_signal_when_bearish(self):
        d = generate_decision(
            _fv(momentum_5=-0.06, momentum_3=-0.05, book_imbalance=-0.5, news_sentiment=-0.5)
        )
        assert d.signal == Signal.SELL

    def test_hold_when_neutral(self):
        d = generate_decision(_fv())
        assert d.signal == Signal.HOLD
        assert d.size_fraction == 0.0
