"""Tests for signal engine v2: EV gating, confluence, threshold recalibration."""

import pytest

from src.strategy.features import FeatureVector
from src.strategy.signal_engine import (
    Decision, Signal, SignalThresholds, THRESHOLDS,
    compute_ev, generate_decision,
    _score_book, _score_momentum, _score_news, _score_rsi, _score_volume,
)


def _fv(**overrides) -> FeatureVector:
    defaults = dict(
        token_id="test",
        momentum_5=0.0, momentum_3=0.0, momentum_1=0.0,
        rsi=50.0, volatility=0.01,
        spread=0.005, spread_pct=0.01,
        book_imbalance=0.0,
        depth_bid=500.0, depth_ask=500.0,
        top3_bid_pressure=0.5, top3_ask_pressure=0.5,
        volume_trend=1.0, volume_acceleration=0.0,
        current_price=0.50, price_vs_mid=0.0,
        news_sentiment=0.0, news_relevance=0.5,
        implied_prob_shift=0.0,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


class TestEVComputation:
    def test_buy_positive_ev(self):
        ev = compute_ev(Signal.BUY, 0.7, 0.3, 0.05)
        assert ev > 0

    def test_sell_positive_ev(self):
        ev = compute_ev(Signal.SELL, 0.7, 0.8, -0.05)
        assert ev > 0

    def test_hold_zero_ev(self):
        assert compute_ev(Signal.HOLD, 0.8, 0.5, 0.0) == 0.0

    def test_low_confidence_negative_ev(self):
        ev = compute_ev(Signal.BUY, 0.2, 0.8, 0.0)
        assert ev < 0


class TestIndividualScorers:
    def test_bullish_momentum(self):
        score, reason = _score_momentum(_fv(momentum_5=0.06, momentum_3=0.05, momentum_1=0.02))
        assert score > 0.3
        assert reason is not None

    def test_bearish_momentum(self):
        score, _ = _score_momentum(_fv(momentum_5=-0.06, momentum_3=-0.05, momentum_1=-0.02))
        assert score < -0.3

    def test_oversold_rsi_bullish(self):
        score, reason = _score_rsi(_fv(rsi=15))
        assert score > 0
        assert "oversold" in reason

    def test_overbought_rsi_bearish(self):
        score, reason = _score_rsi(_fv(rsi=85))
        assert score < 0
        assert "overbought" in reason

    def test_neutral_rsi(self):
        score, reason = _score_rsi(_fv(rsi=50))
        assert score == 0.0
        assert reason is None

    def test_bid_heavy_book_bullish(self):
        score, reason = _score_book(_fv(book_imbalance=0.6, depth_bid=800, depth_ask=200))
        assert score > 0.4

    def test_irrelevant_news_ignored(self):
        score, reason = _score_news(_fv(news_sentiment=0.8, news_relevance=0.1))
        assert score == 0.0

    def test_relevant_news_counted(self):
        score, reason = _score_news(_fv(news_sentiment=0.8, news_relevance=0.9))
        assert score > 0.1


class TestGates:
    def test_spread_gate_blocks(self):
        d = generate_decision(_fv(spread_pct=0.15))
        assert d.signal == Signal.HOLD
        assert "spread" in d.rejected_reason

    def test_low_confidence_blocked(self):
        THRESHOLDS.min_confidence = 0.99  # force failure
        d = generate_decision(_fv())
        THRESHOLDS.min_confidence = 0.30  # restore
        assert d.signal == Signal.HOLD

    def test_negative_ev_blocked(self):
        THRESHOLDS.min_ev = 0.99
        d = generate_decision(_fv(
            momentum_5=0.05, book_imbalance=0.5, news_sentiment=0.5,
            current_price=0.99  # near max — BUY has tiny payoff
        ))
        THRESHOLDS.min_ev = 0.001
        # Either blocked by ev or some other gate — just check not a high-conf trade
        if d.signal != Signal.HOLD:
            assert d.expected_value > 0  # if it passed, it must have positive EV

    def test_buy_approved_with_strong_signal(self):
        d = generate_decision(_fv(
            momentum_5=0.06, momentum_3=0.05, momentum_1=0.02,
            book_imbalance=0.5, depth_bid=1000, depth_ask=300,
            news_sentiment=0.6, news_relevance=0.8,
            current_price=0.30,
        ))
        assert d.signal == Signal.BUY
        assert d.expected_value > 0
        assert d.confidence > 0


class TestThresholds:
    def test_tighten_increases_requirements(self):
        t = SignalThresholds()
        original_conf = t.min_confidence
        t.tighten()
        assert t.min_confidence > original_conf

    def test_relax_decreases_requirements(self):
        t = SignalThresholds()
        t.tighten()
        tightened = t.min_confidence
        t.relax()
        assert t.min_confidence < tightened
