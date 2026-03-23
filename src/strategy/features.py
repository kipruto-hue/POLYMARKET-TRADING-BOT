"""
Feature extraction — rich microstructure + momentum + sentiment features.
Produces FeatureVector with all inputs the signal engine needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.polymarket_client import Candle, OrderBookSnapshot


@dataclass
class FeatureVector:
    token_id: str

    # Price momentum
    momentum_5: float           # 5-candle rate-of-change
    momentum_3: float           # 3-candle rate-of-change
    momentum_1: float           # 1-candle (last bar) change

    # Trend strength
    rsi: float                  # RSI (up to 14-period)
    volatility: float           # stdev of returns

    # Order book microstructure
    spread: float               # absolute bid-ask spread
    spread_pct: float           # spread as % of price
    book_imbalance: float       # (bid_vol - ask_vol) / total
    depth_bid: float            # total $ depth on bid side
    depth_ask: float            # total $ depth on ask side
    top3_bid_pressure: float    # volume concentration in top 3 bid levels
    top3_ask_pressure: float    # volume concentration in top 3 ask levels

    # Volume
    volume_trend: float         # recent vs average volume ratio
    volume_acceleration: float  # rate of change in volume

    # Price position
    current_price: float
    price_vs_mid: float         # price relative to book midpoint

    # External signal
    news_sentiment: float       # aggregated sentiment score [-1, 1]
    news_relevance: float       # how relevant news is [0, 1]

    # Derived probability estimate
    implied_prob_shift: float   # estimated probability shift from current price


def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _depth_value(levels: list[tuple[float, float]], top_n: int = 10) -> float:
    return sum(price * size for price, size in levels[:top_n])


def _top3_pressure(levels: list[tuple[float, float]]) -> float:
    if not levels:
        return 0.0
    top3 = sum(sz for _, sz in levels[:3])
    total = sum(sz for _, sz in levels) or 1.0
    return top3 / total


def build_features(
    token_id: str,
    candles: list[Candle],
    book: OrderBookSnapshot | None,
    news_sentiment: float = 0.0,
    news_relevance: float = 0.5,
) -> FeatureVector:
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    current_price = closes[-1] if closes else 0.0

    # --- Momentum ---
    def _roc(n: int) -> float:
        if len(closes) < n + 1 or closes[-(n + 1)] == 0:
            return 0.0
        return (closes[-1] - closes[-(n + 1)]) / closes[-(n + 1)]

    momentum_5 = _roc(5)
    momentum_3 = _roc(3)
    momentum_1 = _roc(1)

    # --- Volatility ---
    if len(closes) > 2:
        rets = np.diff(closes) / (np.array(closes[:-1]) + 1e-10)
        volatility = float(np.std(rets))
    else:
        volatility = 0.0

    rsi = compute_rsi(closes)

    # --- Volume ---
    if len(volumes) >= 3:
        avg_all = float(np.mean(volumes)) if np.mean(volumes) > 0 else 1.0
        avg_recent = float(np.mean(volumes[-3:]))
        volume_trend = avg_recent / avg_all
        if len(volumes) >= 6:
            avg_older = float(np.mean(volumes[-6:-3])) if np.mean(volumes[-6:-3]) > 0 else 1.0
            volume_acceleration = (avg_recent - avg_older) / avg_older
        else:
            volume_acceleration = 0.0
    else:
        volume_trend = 1.0
        volume_acceleration = 0.0

    # --- Order book microstructure ---
    if book and book.bids and book.asks:
        spread = book.spread
        mid = (book.bids[0][0] + book.asks[0][0]) / 2
        spread_pct = spread / mid if mid > 0 else 0.0
        book_imbalance = book.imbalance
        depth_bid = _depth_value(book.bids)
        depth_ask = _depth_value(book.asks)
        top3_bid_pressure = _top3_pressure(book.bids)
        top3_ask_pressure = _top3_pressure(book.asks)
        price_vs_mid = (current_price - mid) / mid if mid > 0 else 0.0
    else:
        spread = 1.0
        spread_pct = 1.0
        book_imbalance = 0.0
        depth_bid = 0.0
        depth_ask = 0.0
        top3_bid_pressure = 0.0
        top3_ask_pressure = 0.0
        price_vs_mid = 0.0

    # --- Implied probability shift ---
    # On binary markets price IS probability; momentum → estimated future shift
    implied_prob_shift = (momentum_3 * 0.6 + book_imbalance * 0.4) * current_price

    return FeatureVector(
        token_id=token_id,
        momentum_5=momentum_5,
        momentum_3=momentum_3,
        momentum_1=momentum_1,
        rsi=rsi,
        volatility=volatility,
        spread=spread,
        spread_pct=spread_pct,
        book_imbalance=book_imbalance,
        depth_bid=depth_bid,
        depth_ask=depth_ask,
        top3_bid_pressure=top3_bid_pressure,
        top3_ask_pressure=top3_ask_pressure,
        volume_trend=volume_trend,
        volume_acceleration=volume_acceleration,
        current_price=current_price,
        price_vs_mid=price_vs_mid,
        news_sentiment=news_sentiment,
        news_relevance=news_relevance,
        implied_prob_shift=implied_prob_shift,
    )
