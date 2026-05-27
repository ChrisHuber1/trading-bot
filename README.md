# Trading Bot

A dual crypto trading system running 24/7 on Kraken. Two bots, two approaches ;  one rule-based, one ML-driven ;  sharing the same risk management and infrastructure.

I started with a rule-based bot to learn the mechanics of automated trading. Then I built an ML system on top of it to see if a model could find edges that hand-tuned indicators miss. Both run simultaneously on paper trading while I validate performance.

## Architecture

```
Windows (GPU workstation)              Linux Server
+-------------------+          +---------------------------+
| ML Training       |          | Bot1: MACD + StochRSI     |
| XGBoost on CUDA   |--deploy->| Bot3: ML-driven, 40 pairs |
| RTX 4070 Ti       |          | Risk manager              |
| Feature pipeline  |          | Web dashboards            |
+-------------------+          +---------------------------+
                                        |
                               +--------v--------+
                               | TimescaleDB     |
                               | PostgreSQL 16   |
                               | 650+ pairs OHLCV|
                               +-----------------+
```

## Bot1 ;  Rule-Based (MACD + StochRSI)

Straightforward technical analysis. MACD histogram crossover with ADX trend filter, confirmed by Stochastic RSI extremes. Trades 12 pairs on 1-hour candles, both long and short.

The value of Bot1 isn't the strategy ;  it's the infrastructure. Building it forced me to solve order execution, position tracking, risk enforcement, and exchange API reliability. All of that carries over to Bot3.

## Bot3 ;  ML-Driven

This is the interesting one. XGBoost classifier trained on 71 features derived from OHLCV data across 650+ Kraken pairs. The model scores every candidate trade with a confidence value; only trades above 0.65 get executed.

**ML Pipeline:**
- Feature engineering runs in TimescaleDB (SQL-based, 71 features including 7 short-bias indicators)
- Training runs on my RTX 4070 Ti using XGBoost with CUDA acceleration
- Champion-challenger model management ;  new model must beat current champion on holdout data before promotion
- Market regime filter using BTC and ETH EMA-24 as macro sentiment
- ATR-based dynamic stop losses instead of fixed percentages

**What makes it different from tutorial trading bots:**
- Real data pipeline ingesting from a live exchange, not CSV files
- GPU-accelerated training with automated retraining schedule
- Model deployment via SCP with post-copy verification (learned the hard way ;  silent SCP failures created empty model dirs)
- Paper trading gate: 14 days minimum before any model touches real capital

## Risk Management

These rules are enforced in code and cannot be bypassed by the trading logic:

| Rule | Value |
|---|---|
| Capital floor | $700 (30% drawdown = halt) |
| Per-trade risk | 1.5% of portfolio |
| Stop-loss | 1.5% from entry |
| Take-profit | 1.5% from entry |
| Max positions | 5 concurrent |
| Max daily loss | -5% (halt for day) |
| Order type | Limit only (maker) |
| Paper gate | 14 days before live |

## Decisions and Tradeoffs

**XGBoost over neural networks:** For tabular financial data with 71 engineered features, gradient-boosted trees consistently outperform deep learning in benchmarks. XGBoost trains in seconds on GPU, is interpretable via feature importance, and doesn't need the data volume that neural nets require to generalize.

**TimescaleDB over raw PostgreSQL:** Time-series queries (rolling windows, OHLCV aggregation, feature computation) are the core workload. TimescaleDB's hypertables and continuous aggregates make these queries 10-100x faster than vanilla PostgreSQL on the same data.

**Symmetric SL/TP (1.5%/1.5%) over asymmetric:** I started with 4% SL / 1.5% TP. The math looked fine on paper but in practice the bot held losing positions too long, turning small losses into trend-following disasters. Symmetric limits cut average loss duration significantly.

**Champion-challenger over A/B testing:** Can't A/B test with real money safely. Instead, the new model paper-trades alongside the champion. If it outperforms on holdout data and paper PnL, it gets promoted. The old champion becomes the fallback.

## Project Structure

```
trading-bot/
├── config.py                 # All tunable parameters
├── main.py                   # Entry point: --mode paper|live|backtest
├── bot3/                     # ML-driven bot
│   ├── config.py             # Bot3-specific settings
│   └── main.py               # Bot3 entry point
├── bot3_retrain/
│   ├── retrain.py            # GPU training pipeline
│   └── deploy.py             # SCP deployment to server
├── ml_pipeline/
│   ├── features/             # Feature engineering SQL + Python
│   ├── labels/               # Label generation
│   ├── train.py              # XGBoost training with CUDA
│   ├── inference.py          # Model scoring
│   └── runner.py             # Full pipeline orchestration
├── exchange/
│   └── kraken_client.py      # CCXT wrapper for Kraken
├── strategy/
│   ├── macd_stochrsi.py      # Bot1 strategy
│   └── signals.py            # Signal types
├── risk/
│   └── manager.py            # Risk enforcement (never bypassed)
├── execution/
│   └── order_manager.py      # Order placement and fill tracking
├── data/
│   ├── feed.py               # OHLCV aggregation + indicators
│   └── ingestion.py          # TimescaleDB data pipeline
├── monitor/
│   ├── dashboard.py          # Terminal dashboard
│   └── web_server.py         # Browser dashboard (WebSocket)
├── backtest/
│   └── runner.py             # Historical backtesting
└── logs/
    └── trades.jsonl           # Append-only trade log
```

## Current State

Both bots are paper trading on Kraken. Bot1 has been running since mid-April 2026. Bot3 since late April. The ML pipeline retrains automatically and has produced 20+ model versions. Cumulative paper PnL on Bot3 is slightly positive.

I'm not live trading yet. The paper gate exists because I don't trust a model that hasn't survived at least one market regime change.

## What I'd Do Differently

- Start with the ML pipeline first. Building Bot1's rule-based system taught me infrastructure, but the strategy itself doesn't have an edge. Should have used that time building a better feature set.
- Use a proper experiment tracker (MLflow or Weights & Biases) instead of timestamped model directories. The champion-challenger logic works but the model lineage is harder to trace than it should be.

## A Note on Source Code

This is a live trading system. The repository shows the full architecture, project structure, and infrastructure, but strategy implementations, feature engineering, model training, and backtesting logic are stubbed. Class signatures and docstrings show what each component does, but the implementations that represent my trading edge are withheld.

The components you can inspect in full: exchange client, risk manager, order execution, data pipeline, monitoring dashboard, and deployment tooling.
