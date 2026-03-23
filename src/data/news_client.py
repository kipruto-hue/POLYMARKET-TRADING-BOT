"""Crypto news ingestion via free RSS feeds (no API key needed) + sentiment scoring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
import structlog
from textblob import TextBlob

log = structlog.get_logger()

CRYPTO_RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed"),
]


@dataclass
class NewsArticle:
    title: str
    description: str
    source: str
    published_at: str
    url: str
    sentiment: float = 0.0  # -1.0 to 1.0


class NewsClient:
    """Fetches crypto news from free RSS feeds and scores sentiment. No API key required."""

    def __init__(self) -> None:
        self._cache: list[NewsArticle] = []
        self._last_fetch: float = 0

    async def fetch_articles(
        self,
        query: str = "",
        lookback_minutes: int = 60,
        max_results: int = 10,
    ) -> list[NewsArticle]:
        loop = asyncio.get_running_loop()

        all_articles: list[NewsArticle] = []
        for source_name, feed_url in CRYPTO_RSS_FEEDS:
            try:
                feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
                for entry in feed.entries[:15]:
                    title = entry.get("title", "") or ""
                    summary = entry.get("summary", "") or ""
                    published = entry.get("published", "") or ""
                    link = entry.get("link", "") or ""

                    text = f"{title}. {summary[:300]}"
                    sentiment = TextBlob(text).sentiment.polarity

                    all_articles.append(
                        NewsArticle(
                            title=title,
                            description=summary[:200],
                            source=source_name,
                            published_at=published,
                            url=link,
                            sentiment=sentiment,
                        )
                    )
            except Exception:
                log.exception("rss_feed_error", source=source_name)

        if query:
            query_words = query.lower().split()
            filtered = [
                a for a in all_articles
                if any(w in f"{a.title} {a.description}".lower() for w in query_words)
            ]
        else:
            filtered = all_articles

        self._cache = filtered[:max_results]
        return self._cache

    def aggregate_sentiment(self, articles: list[NewsArticle]) -> float:
        if not articles:
            return 0.0
        return sum(a.sentiment for a in articles) / len(articles)

    async def close(self) -> None:
        pass
