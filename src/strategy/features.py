"""Feature extraction from candles, orderbook, and sentiment data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.polymarket_client import Candle, OrderBookSnapshot


@dataclass
class FeatureVector:
    token_id: str
    momentum_5: float         # 5-candle rate-of-change
    momentum_3: float         # 3-candle rate-of-change
    volatility: float         # stdev of returns over window
    rsi: float                # 14-period (or available) RSI
    spread: float             # current bid-ask spread
    book_imbalance: float     # (bid_vol - ask_vol) / total
    news_sentiment: float     # aggregated sentiment score
    current_price: float
    volume_trend: float       # ratio of recent volume to average


def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0  # neutral default
    deltas = np.diff(closes[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(
    token_id: str,
    candles: list[Candle],
    book: OrderBookSnapshot | None,
    news_sentiment: float = 0.0,
) -> FeatureVector:
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    current_price = closes[-1] if closes else 0.0

    # Momentum
    def _roc(n: int) -> float:
        if len(closes) < n + 1 or closes[-(n + 1)] == 0:
            return 0.0
        return (closes[-1] - closes[-(n + 1)]) / closes[-(n + 1)]

    momentum_5 = _roc(5)
    momentum_3 = _roc(3)

    # Volatility
    if len(closes) > 2:
        rets = np.diff(closes) / np.array(closes[:-1])
        volatility = float(np.std(rets))
    else:
        volatility = 0.0

    rsi = compute_rsi(closes)

    spread = book.spread if book else 0.0
    book_imbalance = book.imbalance if book else 0.0

    # Volume trend: recent 3 candles vs overall average
    if len(volumes) >= 3:
        avg_all = float(np.mean(volumes)) if np.mean(volumes) > 0 else 1.0
        avg_recent = float(np.mean(volumes[-3:]))
        volume_trend = avg_recent / avg_all
    else:
        volume_trend = 1.0

    return FeatureVector(
        token_id=token_id,
        momentum_5=momentum_5,
        momentum_3=momentum_3,
        volatility=volatility,
        rsi=rsi,
        spread=spread,
        book_imbalance=book_imbalance,
        news_sentiment=news_sentiment,
        current_price=current_price,
        volume_trend=volume_trend,
    )
