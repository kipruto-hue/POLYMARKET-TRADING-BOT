from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Polymarket credentials
    polymarket_private_key: str
    polymarket_chain_id: int = 137
    polymarket_clob_host: str = "https://clob.polymarket.com"
    polymarket_gamma_host: str = "https://gamma-api.polymarket.com"
    polymarket_ws_host: str = (
        "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/polybot"
    redis_url: str = "redis://localhost:6379/0"

    # Risk limits
    max_position_size: float = 2.0
    max_daily_loss: float = 50.0
    max_open_orders: int = 100
    cooldown_seconds: int = 0

    # News
    news_api_key: str = ""
    news_api_url: str = "https://newsapi.org/v2/everything"

    # Bot behaviour
    bot_interval_seconds: int = 300  # 5-min candle window for analysis
    execution_interval_seconds: float = 1.2  # ~50 trades/min execution loop
    log_level: str = "INFO"
    dry_run: bool = True

    # Market filters
    market_keywords: str = "bitcoin,btc,crypto"
    min_market_liquidity: float = 1000.0
    max_markets: int = 20


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
