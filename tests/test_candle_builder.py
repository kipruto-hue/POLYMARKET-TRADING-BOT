"""Tests for the candle builder aggregation logic."""

import time
from unittest.mock import patch

from src.data.candle_builder import CandleBuilder


class TestCandleBuilder:
    def test_first_tick_initializes(self):
        cb = CandleBuilder(interval_seconds=60)
        result = cb.add_tick("tok1", 0.50, 10)
        assert result is None  # no closed candle yet
        c = cb.get_current_candle("tok1")
        assert c is not None
        assert c.open == 0.50
        assert c.close == 0.50

    def test_ticks_update_ohlc(self):
        cb = CandleBuilder(interval_seconds=60)
        cb.add_tick("tok1", 0.50, 10)
        cb.add_tick("tok1", 0.55, 5)
        cb.add_tick("tok1", 0.45, 8)
        cb.add_tick("tok1", 0.52, 3)
        c = cb.get_current_candle("tok1")
        assert c.open == 0.50
        assert c.high == 0.55
        assert c.low == 0.45
        assert c.close == 0.52
        assert c.volume == 26

    def test_bucket_rollover_produces_closed_candle(self):
        cb = CandleBuilder(interval_seconds=60)
        base = int(time.time()) // 60

        with patch("src.data.candle_builder.time.time", return_value=base * 60 + 10):
            cb.add_tick("tok1", 0.50, 10)
            cb.add_tick("tok1", 0.55, 5)

        with patch("src.data.candle_builder.time.time", return_value=(base + 1) * 60 + 5):
            closed = cb.add_tick("tok1", 0.60, 3)

        assert closed is not None
        assert closed.open == 0.50
        assert closed.high == 0.55
        assert closed.close == 0.55

        history = cb.get_history("tok1")
        assert len(history) == 1
        assert history[0] == closed

    def test_history_capped(self):
        cb = CandleBuilder(interval_seconds=60)
        cb._max_history = 5
        for i in range(10):
            with patch("src.data.candle_builder.time.time", return_value=i * 60):
                cb.add_tick("tok1", 0.50 + i * 0.01, 1)
        history = cb.get_history("tok1")
        assert len(history) <= 5
