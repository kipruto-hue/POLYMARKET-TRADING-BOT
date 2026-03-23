"""Quick-start script: run bot directly without Docker."""

import asyncio
import sys

import structlog
import uvicorn

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        __import__("logging").INFO
    ),
)


def main() -> None:
    if "--bot-only" in sys.argv:
        from src.app.bot import PolymarketBot
        bot = PolymarketBot()
        asyncio.run(bot.start())
    else:
        uvicorn.run(
            "src.app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
        )


if __name__ == "__main__":
    main()
