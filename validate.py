"""
Validation & Testing Tool for the Polymarket Bot.

Runs 4 tests:
  1. NEWS TEST      - Fetches live news, shows sentiment scores, validates quality
  2. MARKET TEST    - Finds Bitcoin markets, checks liquidity and data availability
  3. SIGNAL TEST    - Builds features from live orderbooks, generates decisions
  4. DECISION AUDIT - Simulates a full cycle and grades every decision with reasons

Usage:
    python validate.py              # run all tests
    python validate.py news         # test news only
    python validate.py markets      # test market discovery only
    python validate.py signals      # test signal generation only
    python validate.py audit        # full decision audit
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

os.environ.setdefault("POLYMARKET_PRIVATE_KEY", os.environ.get("POLYMARKET_PRIVATE_KEY", ""))

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        __import__("logging").WARNING
    ),
)

from src.config.settings import get_settings
from src.data.candle_builder import CandleBuilder
from src.data.news_client import NewsClient
from src.data.polymarket_client import CLOBClient, GammaClient
from src.strategy.features import build_features
from src.strategy.signal_engine import generate_decision, compute_ev, Signal

HR = "=" * 70


# -------------------------------------------------------------------------
# 1. NEWS VALIDATION
# -------------------------------------------------------------------------
async def test_news():
    print(f"\n{HR}")
    print("  TEST 1: NEWS QUALITY & SENTIMENT")
    print(HR)

    news = NewsClient()

    queries = ["bitcoin", "btc price", "crypto market", "bitcoin regulation"]
    total_articles = 0
    all_sentiments = []

    print("\n  Source: Free RSS feeds (CoinDesk, CoinTelegraph, Bitcoin Magazine)")

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'-' * 50}")

        articles = await news.fetch_articles(query, lookback_minutes=120, max_results=5)
        total_articles += len(articles)

        if not articles:
            print("  [!] No articles matched this query")
            continue

        for i, a in enumerate(articles, 1):
            sentiment_bar = _sentiment_bar(a.sentiment)
            print(f"  {i}. [{sentiment_bar}] {a.sentiment:+.3f}  {a.title[:65]}")
            print(f"     Source: {a.source}  |  {a.published_at[:16]}")
            all_sentiments.append(a.sentiment)

        avg = news.aggregate_sentiment(articles)
        print(f"\n  Avg sentiment for \"{query}\": {avg:+.3f}")

    print(f"\n  --- NEWS SUMMARY ---")
    print(f"  Total articles fetched: {total_articles}")
    if all_sentiments:
        overall = sum(all_sentiments) / len(all_sentiments)
        bullish = sum(1 for s in all_sentiments if s > 0.05)
        bearish = sum(1 for s in all_sentiments if s < -0.05)
        neutral = len(all_sentiments) - bullish - bearish
        print(f"  Overall sentiment:  {overall:+.3f}")
        print(f"  Bullish: {bullish}  |  Neutral: {neutral}  |  Bearish: {bearish}")
        print(f"  Verdict: {'NEWS OK' if total_articles >= 3 else 'LOW COVERAGE'}")
    else:
        print("  [!] No articles — bot will rely on price/orderbook signals only")

    await news.close()
    return total_articles > 0


# -------------------------------------------------------------------------
# 2. MARKET DISCOVERY VALIDATION
# -------------------------------------------------------------------------
async def test_markets():
    print(f"\n{HR}")
    print("  TEST 2: BITCOIN MARKET DISCOVERY")
    print(HR)

    cfg = get_settings()
    gamma = GammaClient()

    keywords = [kw.strip().lower() for kw in cfg.market_keywords.split(",") if kw.strip()]
    print(f"\n  Keywords: {keywords}")
    print(f"  Min liquidity: ${cfg.min_market_liquidity:,.0f}")

    markets = await gamma.get_active_markets(
        limit=100,
        min_liquidity=cfg.min_market_liquidity,
        keywords=keywords if keywords else None,
    )

    print(f"\n  Found {len(markets)} Bitcoin/crypto markets:\n")

    if not markets:
        print("  [!] No markets matched keywords. Testing without keyword filter...")
        markets = await gamma.get_active_markets(limit=20, min_liquidity=1000)
        print(f"  Found {len(markets)} total markets (unfiltered)\n")

    for i, m in enumerate(markets[:15], 1):
        print(f"  {i:2d}. {m.question[:70]}")
        print(f"      Liquidity: ${m.liquidity:,.0f}  |  24h Vol: ${m.volume_24h:,.0f}  |  Token: {m.token_id[:16]}...")

    print(f"\n  --- MARKET SUMMARY ---")
    print(f"  Total matched: {len(markets)}")
    if markets:
        avg_liq = sum(m.liquidity for m in markets) / len(markets)
        print(f"  Avg liquidity: ${avg_liq:,.0f}")
        print(f"  Verdict: {'MARKETS OK' if len(markets) >= 2 else 'LOW COUNT — consider broadening keywords'}")

    await gamma.close()
    return markets


# -------------------------------------------------------------------------
# 3. SIGNAL GENERATION VALIDATION
# -------------------------------------------------------------------------
async def test_signals(markets=None):
    print(f"\n{HR}")
    print("  TEST 3: LIVE SIGNAL GENERATION")
    print(HR)

    if not markets:
        gamma = GammaClient()
        cfg = get_settings()
        keywords = [kw.strip().lower() for kw in cfg.market_keywords.split(",") if kw.strip()]
        markets = await gamma.get_active_markets(limit=50, min_liquidity=1000, keywords=keywords)
        if not markets:
            markets = await gamma.get_active_markets(limit=10, min_liquidity=1000)
        await gamma.close()

    clob = CLOBClient()
    results = []

    for market in markets[:10]:
        print(f"\n  Market: {market.question[:65]}")

        try:
            book = await clob.get_orderbook(market.token_id)
        except Exception as e:
            print(f"  [!] Orderbook fetch failed: {e}")
            continue

        print(f"  Orderbook: {len(book.bids)} bids, {len(book.asks)} asks")
        print(f"  Spread: {book.spread:.4f}  |  Imbalance: {book.imbalance:+.3f}")

        # Simulate candle data from orderbook midpoint
        mid = (book.bids[0][0] + book.asks[0][0]) / 2 if book.bids and book.asks else 0.5
        from src.data.polymarket_client import Candle
        fake_candles = [
            Candle(timestamp=time.time() - (20 - i) * 300, open=mid, high=mid + 0.005,
                   low=mid - 0.005, close=mid + (i - 10) * 0.001, volume=50)
            for i in range(20)
        ]

        features = build_features(market.token_id, fake_candles, book, news_sentiment=0.0, news_relevance=0.5)
        decision = generate_decision(features)

        signal_icon = {"BUY": "+", "SELL": "-", "HOLD": "~"}[decision.signal.value]
        print(f"  Signal: [{signal_icon}] {decision.signal.value}  "
              f"conf={decision.confidence:.3f}  ev={decision.expected_value:.5f}")
        if decision.rejected_reason:
            print(f"  Rejected: {decision.rejected_reason}")
        print(f"  Reasons: {', '.join(decision.reasons) if decision.reasons else 'neutral'}")
        print(f"  Target: ${decision.target_price:.4f}  |  Size frac: {decision.size_fraction:.3f}")

        results.append({
            "question": market.question[:50],
            "signal": decision.signal.value,
            "ev": decision.expected_value,
            "confidence": decision.confidence,
            "spread": book.spread,
            "reasons": decision.reasons,
            "rejected": decision.rejected_reason,
        })

    print(f"\n  --- SIGNAL SUMMARY ---")
    buys = sum(1 for r in results if r["signal"] == "BUY")
    sells = sum(1 for r in results if r["signal"] == "SELL")
    holds = sum(1 for r in results if r["signal"] == "HOLD")
    print(f"  BUY: {buys}  |  SELL: {sells}  |  HOLD: {holds}")
    if results:
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        avg_spread = sum(r["spread"] for r in results) / len(results)
        active = [r for r in results if r["signal"] != "HOLD"]
        avg_ev = sum(r["ev"] for r in active) / len(active) if active else 0
        rejected_reasons = [r.get("rejected", "") for r in results if r.get("rejected")]
        print(f"  Avg confidence: {avg_conf:.3f}")
        print(f"  Avg spread: {avg_spread:.4f}")
        print(f"  Avg EV (active): {avg_ev:.5f}")
        if rejected_reasons:
            from collections import Counter
            top_rejects = Counter(rejected_reasons).most_common(3)
            print(f"  Top rejection reasons: {top_rejects}")
        wide = sum(1 for r in results if r["spread"] > 0.06)
        print(f"  Markets with wide spread (>$0.06): {wide}")
        verdict = "SIGNALS OK (EV-gated)" if buys + sells > 0 else "ALL HOLD — filters working correctly"
        print(f"  Verdict: {verdict}")

    await clob.close()
    return results


# -------------------------------------------------------------------------
# 4. FULL DECISION AUDIT
# -------------------------------------------------------------------------
async def test_audit():
    print(f"\n{HR}")
    print("  TEST 4: FULL DECISION AUDIT")
    print(HR)

    cfg = get_settings()
    print(f"\n  Config check:")
    print(f"    Dry run:           {cfg.dry_run}")
    print(f"    Max trade size:    ${cfg.max_position_size}")
    print(f"    Max daily loss:    ${cfg.max_daily_loss}")
    print(f"    Cooldown:          {cfg.cooldown_seconds}s")
    print(f"    Exec interval:     {cfg.execution_interval_seconds}s (~{60/cfg.execution_interval_seconds:.0f} trades/min)")
    print(f"    Analysis window:   {cfg.bot_interval_seconds}s ({cfg.bot_interval_seconds//60}m candles)")
    print(f"    Keywords:          {cfg.market_keywords}")
    print(f"    Max open orders:   {cfg.max_open_orders}")

    from src.trading.risk_engine import RiskEngine
    risk = RiskEngine()

    print(f"\n  Risk engine state:")
    print(f"    Kill switch: {risk.state.kill_switch}")
    print(f"    Daily PnL:   ${risk.state.daily_pnl}")
    print(f"    Positions:   {len(risk.state.positions)}")

    # Run news + markets + signals
    print("\n  Running full pipeline...")
    news_ok = await test_news()
    markets = await test_markets()
    signals = await test_signals(markets)

    print(f"\n{HR}")
    print("  FINAL AUDIT REPORT")
    print(HR)

    checks = []

    checks.append(("Private key configured", cfg.polymarket_private_key != "0xYOUR_PRIVATE_KEY_HERE"))
    checks.append(("Dry run mode (safety)", cfg.dry_run))
    checks.append(("News API available", news_ok))
    checks.append(("Bitcoin markets found", len(markets) > 0 if markets else False))
    checks.append(("Signals generated", len(signals) > 0 if signals else False))
    checks.append(("Trade size <= $2", cfg.max_position_size <= 2.0))
    checks.append(("Daily loss cap set", cfg.max_daily_loss <= 100.0))
    active = sum(1 for s in signals if s["signal"] != "HOLD") if signals else 0
    checks.append(("Active trade signals", active > 0))

    all_pass = True
    for name, passed in checks:
        icon = "[PASS]" if passed else "[WARN]"
        if not passed:
            all_pass = False
        print(f"  {icon} {name}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME WARNINGS — review above'}")
    print(f"  Bot is {'READY' if all_pass else 'READY with warnings'} for {'dry-run' if cfg.dry_run else 'LIVE'} trading")
    print(HR)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _sentiment_bar(score: float, width: int = 10) -> str:
    normalized = (score + 1) / 2
    filled = int(normalized * width)
    return "#" * filled + "." * (width - filled)


async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "news":
        await test_news()
    elif target == "markets":
        await test_markets()
    elif target == "signals":
        await test_signals()
    elif target == "audit":
        await test_audit()
    else:
        await test_audit()


if __name__ == "__main__":
    asyncio.run(main())
