# Bybit Quant Bot v9.3

<div align="center">

**Tiered real-time trading bot for Bybit V5 perpetuals**

Quant-driven signals · Multi-setup classification · DeepSeek agentic analysis · Full backtest engine

</div>

---

## Overview

A production crypto futures bot for Bybit USDT perpetuals. Scans ~260 pairs hourly, monitors the top 20 in real-time via WebSocket, classifies signals into scalp/intraday/swing setups, gates execution behind multi-signal agreement and Bayesian confidence, and applies delayed execution via a signal queue.

Revived from 4 months of dormancy in v9.1 after a root-cause analysis of the original failure modes (deprecated V2 API, silent thread crashes, over-filtering).

---

## Version History

| Version | Date | Highlights |
|---|---|---|
| **9.3** | 2026-10-21 | Tiered WS + scheduled scan · Backtest portfolio simulator · DeepSeek advisor |
| 9.2 | 2026-10-19 | Signal queue · Multi-TP fix · Stalled detection · Reflection memory · SIGHUP soft reload |
| 9.1 | 2026-10-17 | V5 API migration · Full-pair scanner · Quant composite · Bayesian confidence |
| 8.1 | 2025-06 | Legacy — dead for 4 months |

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Bybit V5 API key (**Read + Trade + Contract Trading** permissions)
- DeepSeek API key ([platform.deepseek.com](https://platform.deepseek.com))
- Discord webhook URLs (optional)

### Installation

```bash
git clone https://github.com/youruser/Bot-Auto-Screening-Bybit.git
cd Bot-Auto-Screening-Bybit

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install tqdm

createdb bybit_bot_v9

cp config.json.example config.json
nano config.json
```

### DB Migration

```bash
sudo -u postgres psql -d bybit_bot_v9 << 'SQL'
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS execution_status VARCHAR(20) DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS execution_attempts INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS execute_after TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS executed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS execution_result JSONB,
    ADD COLUMN IF NOT EXISTS plan JSONB,
    ADD COLUMN IF NOT EXISTS funding_rate_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS funding_regime VARCHAR(32),
    ADD COLUMN IF NOT EXISTS funding_contrarian_bias VARCHAR(8),
    ADD COLUMN IF NOT EXISTS technical_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS composite_score_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS derivative_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sentiment_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pattern_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS llm_score DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_signals_exec_queue
    ON signals (execution_status, execute_after)
    WHERE execution_status = 'pending';

CREATE TABLE IF NOT EXISTS signal_cooldowns (
    symbol          VARCHAR(32) PRIMARY KEY,
    alerted_at      TIMESTAMPTZ NOT NULL,
    direction       VARCHAR(8),
    confidence      DOUBLE PRECISION,
    signal_id       BIGINT
);

CREATE TABLE IF NOT EXISTS decision_memory (
    id                  BIGSERIAL PRIMARY KEY,
    signal_id           BIGINT REFERENCES signals(id),
    symbol              VARCHAR(32) NOT NULL,
    direction           VARCHAR(8) NOT NULL,
    entry_price         DOUBLE PRECISION,
    exit_price          DOUBLE PRECISION,
    realized_pnl_pct    DOUBLE PRECISION,
    setup_type          VARCHAR(16),
    confidence          DOUBLE PRECISION,
    holding_bars        INT,
    context_json        JSONB,
    settled             BOOLEAN DEFAULT FALSE,
    benchmark_pnl_pct   DOUBLE PRECISION,
    alpha_pct           DOUBLE PRECISION,
    reflection_text     TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    settled_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_dm_symbol_settled
    ON decision_memory (symbol, settled);
SQL
```

### Run

```bash
python main.py
```

---

## Configuration

### Minimal `config.json`

```json
{
  "api": {
    "bybit_key": "YOUR_KEY",
    "bybit_secret": "YOUR_SECRET",
    "testnet": false,
    "discord_servers": [
      {
        "name": "Main",
        "mode": "combined",
        "webhook": "https://discord.com/api/webhooks/..."
      }
    ]
  },
  "deepseek": {
    "api_key": "sk-...",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-pro",
    "advisor_model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 1500,
    "timeout": 45
  },
  "database": {
    "host": "127.0.0.1",
    "database": "bybit_bot_v9",
    "user": "nazhier",
    "password": "no-special-chars",
    "port": "5432"
  },
  "system": {
    "check_interval_hours": 1,
    "max_concurrent": 3,
    "scan_top_n": 0,
    "min_turnover_24h": 1000000
  },
  "risk": {
    "default_leverage": 10,
    "max_position_pct": 3.0,
    "max_open_positions": 5,
    "max_daily_loss_pct": 5.0,
    "max_drawdown_pct": 10.0
  },
  "strategy": {
    "min_raw_threshold": 50.0,
    "min_calibrated": 0.50,
    "composite_weights": {
      "mean_reversion": 0.20,
      "basis": 0.10,
      "volume": 0.15,
      "derivatives": 0.25,
      "funding": 0.15,
      "momentum": 0.15
    }
  },
  "execution": {
    "entry_order_type": "limit_preferred",
    "market_tolerance_pct": 0.15,
    "max_deviation_pct": 4.0,
    "cancel_on_rally_pct": 2.5,
    "momentum_guard_roc_pct": 3.0,
    "min_atr_pct": 0.3,
    "max_atr_pct": 5.0,
    "atr_multiplier_sl": 1.5,
    "tp_r_multiples": [1.0, 2.0, 3.0],
    "tp_weights": [0.40, 0.35, 0.25],
    "tp_attach_mode": "limits_only",
    "emergency_sl_pct": 2.5,
    "max_hold_hours": 72,
    "stall_timeout_candles": 8,
    "signal_cooldown_hours": 4
  },
  "realtime": {
    "enabled": true,
    "max_connections": 2,
    "selection": "top_by_turnover",
    "max_symbols": 20,
    "tier2_enabled": true,
    "tier2_deviation_multiplier": 1.5
  },
  "auto_trade": {
    "enabled": true,
    "delay_seconds": 10,
    "delay_seconds_by_setup": {
      "scalp": 5, "intraday": 10, "swing": 20
    },
    "execution_interval_seconds": 5
  },
  "agreement": {
    "min_agree": 3
  },
  "setups": {
    "scalp_rvol_min": 1.5,
    "intraday_rvol_min": 1.0,
    "swing_rvol_min": 0.7
  },
  "auto_tune": {
    "run_hour": 7,
    "auto_tune_min_samples": 30
  }
}
```

### Critical warnings

**`database.password` cannot contain `@ : / ? # & %`** — these break URL parsing. Use hyphens/underscores only.

**`composite_weights` must sum to 1.0** — the engine will renormalize if not, but you'll see a warning every scan.

**`max_position_pct` × `default_leverage` should be ≤ 100** — e.g., 3.0 × 10 = 30% notional. Values above 100 mean one bad trade can wipe the account.

### Parameter tuning reference

| Parameter | Default | Effect |
|---|---|---|
| `system.max_concurrent` | 3 | Higher → faster scan, risk of rate limits |
| `execution.max_deviation_pct` | 4.0 | Skip signals if price moved more than this |
| `execution.cancel_on_rally_pct` | 2.5 | Cancel limit orders if price rallies this far |
| `execution.momentum_guard_roc_pct` | 3.0 | Skip symbols with 5-bar ROC above this |
| `execution.min_atr_pct` | 0.3 | Skip dead symbols (below this ATR%) |
| `execution.max_atr_pct` | 5.0 | Skip pump-mode memecoins |
| `agreement.min_agree` | 3 | Min sources agreeing (of 5) |
| `realtime.tier2_deviation_multiplier` | 1.5 | Tier 2 relaxed deviation = ×1.5 |
| `auto_trade.delay_seconds_by_setup` | 5/10/20 | Delay before execution per setup type |

---

## Subsystems

### Real-Time Monitor (`core/realtime_monitor.py`)

WebSocket kline stream for the top 20 symbols by 24h turnover, split across 2 connections (10 topics each).

- **`confirm=true` gate** — only closed candles trigger analysis
- **Heartbeat** every 20s (Bybit requirement)
- **Auto-reconnect** with exponential backoff
- **Gap-fill** on reconnect — fetches missed candles via REST
- **Bounded store** — 300 candles per symbol max

Latency vs scheduled scan: **~1 second vs up to 60 minutes**.

### Tiered Signal Recovery (`main.py`)

When WS fires a signal:

| Tier | Execution | Fallback |
|---|---|---|
| 1 | Strict: 4% dev, 2.5% rally cancel, limit preferred | Tier 2 |
| 2 | Relaxed: 6% dev, no rally cancel, market only | Tier 3 |
| 3 | Next scheduled scan (natural retry) | — |

**Analysis quality is never compromised** — Tier 2 only relaxes execution, not the agreement gate, confidence scorer, momentum guard, or ATR filter.

### Signal Queue (`execution/signal_queue.py`)

Decouples signal generation from execution:

1. Signal logged as `pending` with `execute_after = now + delay`
2. Discord alert fires **immediately**
3. Queue worker executes after delay
4. Stall detection cancels unfilled limits

Prevents: signal chases, stale entries, race conditions.

### Setup Classifier (`analysis/setup_classifier.py`)

Every signal classified into one of:
- **Scalp** — 15m, high RVOL, short half-life
- **Intraday** — 1h, mean-reverting, mid half-life
- **Swing** — 4h, trend-following, long half-life

**Direction lock** — all valid setups must agree on direction. Prevents `[scalp long, swing short]` contradictions.

Alert routing:
- 3 setups → 3 separate Discord messages
- 2 setups → 1 combined message
- 1 setup → 1 single message

### Agreement Gate (`analysis/agreement_gate.py`)

Requires `min_agree` sources to agree:
- SMC (BOS/CHoCH)
- Derivatives (CVD + smart money)
- Quant composite
- Orderbook heatmap
- Fibonacci

Contradictions → signal rejected. This is the primary false-positive filter.

### Momentum Guard (`main.py`)

Direction-aware + ATR-normalized skip:

```
skip if:
  ROC moves in bias direction  AND
  |ROC| >= 3%                  AND
  sigma >= 2.5                 (sigma = |ROC| / (ATR% × √lookback))
```

**Key behavior:** a 5% drop before a LONG signal **does not** skip — it's a better entry. Only chases (moves in your trade direction) are filtered.

### Position Management (`execution/trade_executor.py`)

**Multi-TP:**
- Entry order attaches SL only
- TP1/TP2/TP3 placed as separate reduce-only limits (40%/35%/25%)
- SL cascades: breakeven after TP1, TP1 price after TP2

**Trailing stop:** activates at 1% profit, trails 0.5% behind favorable price.

**Emergency close:** 2.5% adverse move → market close. 72h max hold.

**Orphan prevention:** pending Bybit orders cancelled on shutdown before tasks die.

---

## Backtest

Full historical replay with the same production code.

### Quick Start

```bash
source venv/bin/activate

python -m backtest.run_all_pairs --days 30 --json-out bt.json

python -m backtest.run_all_pairs --days 30 --recommend

python -m backtest.run_all_pairs --days 30 --recommend --apply
```

### What's Simulated

| Component | Realistic? |
|---|---|
| OHLCV, funding, OI | Real Bybit data |
| CVD | Approximated from kline |
| Orderbook heatmap | Neutral (no historical data) |
| SMC | Simplified swing-break proxy |
| Fill logic | Fees + slippage + SL-first tiebreak |
| Stall detection | 8-bar limit expiry |
| Portfolio equity | Ruin detection |

### Example Result (Aggressive Config)

With `20% margin × 20x leverage` ($100 account):

```
PORTFOLIO RUINED
Peak equity: $147.32 (+47%)
Ended with: $0.00
Trades: 12
Killed by: PUMPFUNUSDT
```

**The strategy has edge** (peak +47%) **but sizing destroys it.**

### Recommended Safe Config

```json
"risk": {
  "default_leverage": 10,
  "max_position_pct": 3.0,
  "max_open_positions": 5,
  "max_daily_loss_pct": 5.0,
  "max_drawdown_pct": 10.0
}
```

Expected: +5–10% over 30 days, max drawdown <6%, no ruin.

### DeepSeek Advisor

After backtest, sends metrics to DeepSeek for config recommendations. Validates every change against a whitelist and bounds. `--apply` writes to config with automatic backup.

**20 tunable parameters** — cannot touch API keys, database, or `auto_trade.enabled`.

---

## Operation

### systemd

```ini
[Unit]
Description=Bybit Quant Bot v9.3
After=network-online.target postgresql.service
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=20

[Service]
Type=simple
User=nazhier
WorkingDirectory=/home/nazhier/Bot-Auto-Screening-Bybit
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/home/nazhier/Bot-Auto-Screening-Bybit/.env
ExecStart=/home/nazhier/Bot-Auto-Screening-Bybit/venv/bin/python -u main.py
Restart=always
RestartSec=15
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bybit-quant
sudo journalctl -u bybit-quant -f
```

### Soft Reload (SIGHUP)

Apply config changes without closing positions:

```bash
sudo kill -HUP $(pidof -f 'python.*main.py')
```

Preserves: active positions, pending limits, cooldowns, WS connections.

### Daily Git Pull (00:00 WIB)

```bash
cat > /home/nazhier/Bot-Auto-Screening-Bybit/deploy/git-pull.sh << 'EOF'
#!/bin/bash
cd /home/nazhier/Bot-Auto-Screening-Bybit
git fetch origin main
git pull --ff-only origin main
echo "Pull complete at $(date -u)"
EOF
chmod +x /home/nazhier/Bot-Auto-Screening-Bybit/deploy/git-pull.sh

echo "0 17 * * * /home/nazhier/Bot-Auto-Screening-Bybit/deploy/git-pull.sh" | \
  sudo -u nazhier crontab -
```

---

## Troubleshooting

### No signals for hours

```bash
sudo journalctl -u bybit-quant --since "1 hour ago" | \
  grep -E "skip|invalid" | head -20
```

Common filters:
- `momentum_guard` — price moving too fast
- `atr_too_high` — memecoin pump
- `agreement_failed` — sources disagree
- `low_confidence` — below thresholds
- `cooldown` — traded recently

### Rate limit (HTTP 403)

```json
"system": { "max_concurrent": 2 }
```

Verify server is not US/CN:
```bash
curl -s https://ipinfo.io/$(curl -s https://api.ipify.org)/country
```

### DeepSeek timeout

```json
"deepseek": {
  "advisor_model": "deepseek-chat",
  "timeout": 120
}
```

### Signal not executing

```bash
sudo -u postgres psql -d bybit_bot_v9 -c "
  SELECT id, symbol, execution_status,
         execution_result->>'reason' AS reason
  FROM signals
  WHERE timestamp > NOW() - INTERVAL '1 hour'
  ORDER BY id DESC LIMIT 10;
"
```

Statuses: `pending` → `executing` → `filled` / `skipped` / `failed` / `stalled` / `expired`.

### Position close spamming

`del self.active_positions[symbol]` must be the **first executable line** in `_handle_position_closed`, before any `await`.

Verify:
```bash
grep -A 5 "async def _handle_position_closed" execution/trade_executor.py | head -8
```

### Config JSON invalid

```bash
python3 -m json.tool config.json > /dev/null && echo "OK"
```

Common causes: trailing comma before `}`, missing comma between keys, unquoted strings.

---

## File Structure

```
Bot-Auto-Screening-Bybit/
├── main.py
├── config.json
├── requirements.txt
├── README.md
│
├── core/
│   ├── bybit_client.py
│   └── realtime_monitor.py
│
├── analysis/
│   ├── derivatives.py
│   ├── confidence.py
│   ├── quant_composite.py
│   ├── setup_classifier.py
│   ├── fibonacci.py
│   ├── orderbook_heatmap.py
│   ├── agreement_gate.py
│   ├── smc.py
│   ├── mean_reversion.py
│   ├── basis.py
│   ├── volume_anomaly.py
│   ├── funding.py
│   ├── technicals.py
│   └── patterns.py
│
├── agent/
│   └── deepseek_agent.py
│
├── execution/
│   ├── trade_executor.py
│   ├── risk_manager.py
│   ├── signal_queue.py
│   └── order_state.py
│
├── learning/
│   ├── signal_journal.py
│   ├── outcome_tracker.py
│   ├── auto_tuner.py
│   └── reflection_memory.py
│
├── persistence/
│   ├── database.py
│   └── models.py
│
├── notification/
│   ├── discord_bot.py
│   └── chart_generator.py
│
├── backtest/
│   ├── data_fetcher.py
│   ├── strategy.py
│   ├── simulator.py
│   ├── portfolio_simulator.py
│   ├── metrics.py
│   ├── report.py
│   ├── deepseek_advisor.py
│   └── run_all_pairs.py
│
├── tests/
│   ├── conftest.py
│   ├── test_momentum_guard.py
│   ├── test_atr_filter.py
│   ├── test_setup_classifier.py
│   ├── test_agreement_gate.py
│   ├── test_tp_sl.py
│   └── test_confidence.py
│
└── deploy/
    ├── bot.service
    ├── git-pull.sh
    └── restart_bot.sh
```

---

## Testing

```bash
source venv/bin/activate
pytest tests/ -v
```

Runs 38 tests covering:
- Momentum guard (direction-aware, sigma)
- ATR filter (band edges)
- Setup classifier (direction lock, contradiction rejection)
- Agreement gate (min_agree, contradiction detection)
- TP/SL math (1R minimum, weighted RR)
- Confidence scorer (Bayesian blend)

---

## Contributing

1. Fork
2. Feature branch (`git checkout -b feature/x`)
3. Add tests for new quant formulas
4. Include docstrings with derivations
5. PR with justification for composite weight changes

---

## License

MIT — see [LICENSE](LICENSE)

---

## Disclaimer

**This software is for research and educational purposes only.**

Trading cryptocurrency futures involves substantial risk of loss. The backtest system shows exactly how aggressive configs lead to ruin — **run the backtest before deploying**.

The authors assume **no responsibility** for financial losses.