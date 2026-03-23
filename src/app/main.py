"""FastAPI server: health endpoints, bot controls, and metrics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.app.bot import PolymarketBot
from src.config.settings import get_settings

log = structlog.get_logger()

bot: PolymarketBot | None = None
bot_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, bot_task
    bot = PolymarketBot()
    bot_task = asyncio.create_task(bot.start())
    log.info("fastapi_started")
    yield
    if bot:
        await bot.stop()
    if bot_task:
        bot_task.cancel()


app = FastAPI(title="Polymarket Bot", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dry_run": get_settings().dry_run,
        "metrics": bot.metrics.to_dict() if bot else {},
        "feedback": bot.feedback.to_dict() if bot else {},
    }


@app.get("/metrics")
async def metrics():
    if not bot:
        return {"error": "bot not initialized"}
    return {**bot.metrics.to_dict(), **bot.feedback.to_dict()}


@app.get("/performance")
async def performance():
    if not bot:
        return {"error": "bot not initialized"}
    fb = bot.feedback
    return {
        "mode": fb.mode.value,
        "overall_win_rate": fb.overall_win_rate,
        "rolling": fb.window.to_dict(),
        "thresholds": __import__("src.strategy.signal_engine", fromlist=["THRESHOLDS"]).THRESHOLDS.to_dict(),
        "latency_avg_ms": bot.executor.avg_latency_ms,
    }


@app.get("/positions")
async def positions():
    if not bot:
        return {"error": "bot not initialized"}
    return {
        tid: {
            "side": p.side,
            "size": p.size,
            "avg_entry": p.avg_entry,
            "age_seconds": round(p.age_seconds()),
            "unrealized_pnl": p.unrealized_pnl,
        }
        for tid, p in bot.risk.state.positions.items()
    }


@app.get("/markets")
async def active_markets():
    if not bot:
        return []
    return [
        {
            "token_id": m.token_id[:16] + "...",
            "question": m.question[:80],
            "liquidity": m.liquidity,
            "volume_24h": m.volume_24h,
        }
        for m in bot.markets
    ]


@app.post("/kill")
async def kill_switch():
    if bot:
        bot.risk.activate_kill_switch()
        await bot.executor.cancel_all()
        return {"status": "kill_switch_activated"}
    return {"error": "bot not initialized"}


@app.post("/resume")
async def resume():
    if bot:
        bot.risk.deactivate_kill_switch()
        return {"status": "resumed"}
    return {"error": "bot not initialized"}


@app.post("/reset-daily")
async def reset_daily():
    if bot:
        bot.risk.reset_daily()
        return {"status": "daily_risk_reset"}
    return {"error": "bot not initialized"}
