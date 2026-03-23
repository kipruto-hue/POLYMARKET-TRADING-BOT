# Polymarket 5-Minute Trading Bot

Autonomous bot that trades Polymarket prediction markets on a 5-minute decision cycle using price momentum, order book microstructure, and news sentiment signals.

## Architecture

```
Gamma API  -->  Market Selector
CLOB WS    -->  Candle Builder  -->  Feature Engine  -->  Signal Engine
News API   -->  Sentiment Score -/                         |
                                                     Risk Engine
                                                           |
                                                     Execution Engine  -->  CLOB REST
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required: `POLYMARKET_PRIVATE_KEY` (your Polygon wallet private key).

### 3. Run (local, dry-run mode)

```bash
python run.py
```

The bot starts in **DRY_RUN=true** mode by default — it will generate signals and log decisions but will **not** place real orders.

### 4. Dashboard

In a separate terminal:

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` to see live status, positions, and controls.

### 5. Docker

```bash
docker-compose up --build
```

This starts the bot, dashboard, PostgreSQL, and Redis together.

## Going Live

1. Set `DRY_RUN=false` in `.env`
2. Set conservative risk limits (`MAX_POSITION_SIZE`, `MAX_DAILY_LOSS`)
3. Monitor via the dashboard and `/health` endpoint
4. Use the **Kill Switch** button or `POST /kill` to halt trading instantly

## API Endpoints

| Endpoint        | Method | Description                    |
|-----------------|--------|--------------------------------|
| `/health`       | GET    | Bot health + metrics           |
| `/metrics`      | GET    | Detailed bot metrics           |
| `/positions`    | GET    | Current open positions         |
| `/markets`      | GET    | Active monitored markets       |
| `/kill`         | POST   | Activate kill switch           |
| `/resume`       | POST   | Deactivate kill switch         |
| `/reset-daily`  | POST   | Reset daily PnL counter        |

## Safety

- **DRY_RUN** mode on by default
- **Kill switch** halts all trading and cancels open orders
- **Daily loss limit** auto-triggers kill switch
- **Cooldown timer** prevents rapid-fire trading
- **Position caps** per market and globally
- All decisions and trades logged to database

## Project Structure

```
src/
  app/          main.py (FastAPI), bot.py (orchestrator)
  config/       settings.py (typed config from .env)
  data/         polymarket_client.py, news_client.py, candle_builder.py
  strategy/     features.py, signal_engine.py
  trading/      risk_engine.py, execution_engine.py
  storage/      models.py, repository.py
  monitoring/   metrics.py
dashboard.py    Streamlit operator UI
run.py          Quick-start launcher
```
