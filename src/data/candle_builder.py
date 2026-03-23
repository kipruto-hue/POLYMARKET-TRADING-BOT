"""Aggregates trade/price ticks into 5-minute OHLCV candles."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from src.data.polymarket_client import Candle


@dataclass
class CandleBuilder:
    """
    Accumulates price ticks and produces fixed-interval candles.
    Call `add_tick` for every price update, then `get_candle` to retrieve
    the latest closed candle for a given token.
    """

    interval_seconds: int = 300  # 5 minutes
    _open: dict[str, float] = field(default_factory=dict)
    _high: dict[str, float] = field(default_factory=dict)
    _low: dict[str, float] = field(default_factory=dict)
    _close: dict[str, float] = field(default_factory=dict)
    _volume: dict[str, float] = field(default_factory=dict)
    _bucket: dict[str, int] = field(default_factory=dict)
    _history: dict[str, list[Candle]] = field(default_factory=lambda: defaultdict(list))
    _max_history: int = 100

    def _current_bucket(self) -> int:
        return int(time.time()) // self.interval_seconds

    def add_tick(self, token_id: str, price: float, size: float = 0.0) -> Candle | None:
        bucket = self._current_bucket()
        prev_bucket = self._bucket.get(token_id)

        closed_candle: Candle | None = None

        if prev_bucket is not None and bucket != prev_bucket:
            closed_candle = Candle(
                timestamp=prev_bucket * self.interval_seconds,
                open=self._open[token_id],
                high=self._high[token_id],
                low=self._low[token_id],
                close=self._close[token_id],
                volume=self._volume.get(token_id, 0.0),
            )
            hist = self._history[token_id]
            hist.append(closed_candle)
            if len(hist) > self._max_history:
                self._history[token_id] = hist[-self._max_history :]
            self._open[token_id] = price
            self._high[token_id] = price
            self._low[token_id] = price
            self._close[token_id] = price
            self._volume[token_id] = size
        elif prev_bucket is None:
            self._open[token_id] = price
            self._high[token_id] = price
            self._low[token_id] = price
            self._close[token_id] = price
            self._volume[token_id] = size
        else:
            self._high[token_id] = max(self._high[token_id], price)
            self._low[token_id] = min(self._low[token_id], price)
            self._close[token_id] = price
            self._volume[token_id] = self._volume.get(token_id, 0.0) + size

        self._bucket[token_id] = bucket
        return closed_candle

    def get_history(self, token_id: str, n: int = 20) -> list[Candle]:
        return self._history.get(token_id, [])[-n:]

    def get_current_candle(self, token_id: str) -> Candle | None:
        if token_id not in self._open:
            return None
        bucket = self._bucket[token_id]
        return Candle(
            timestamp=bucket * self.interval_seconds,
            open=self._open[token_id],
            high=self._high[token_id],
            low=self._low[token_id],
            close=self._close[token_id],
            volume=self._volume.get(token_id, 0.0),
        )
