"""
persistence/database.py
Async PostgreSQL connection pool + schema migrations.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(32) NOT NULL,
    direction       VARCHAR(8) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    raw_score       DOUBLE PRECISION,
    calibrated_confidence DOUBLE PRECISION,
    is_high_confidence BOOLEAN DEFAULT FALSE,
    gate_reasons    JSONB,
    llm_response    JSONB,
    indicators      JSONB,
    composite_score DOUBLE PRECISION,
    composite_direction VARCHAR(8),
    composite_conviction VARCHAR(16),
    zeta            DOUBLE PRECISION,
    hurst           DOUBLE PRECISION,
    half_life_hours DOUBLE PRECISION,
    basis_pct       DOUBLE PRECISION,
    funding_annualized_pct DOUBLE PRECISION,
    volume_zscore   DOUBLE PRECISION,
    volume_anomaly_level VARCHAR(24),
    executed        BOOLEAN DEFAULT FALSE,
    execution_result JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_high_conf ON signals(is_high_confidence) WHERE is_high_confidence = TRUE;

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id       BIGINT PRIMARY KEY REFERENCES signals(id) ON DELETE CASCADE,
    result          VARCHAR(24) NOT NULL,
    won             BOOLEAN NOT NULL,
    pnl_pct         DOUBLE PRECISION,
    elapsed_hours   DOUBLE PRECISION,
    perps_data      JSONB,
    resolved_at     TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tuning_history (
    id              BIGSERIAL PRIMARY KEY,
    parameter       VARCHAR(64) NOT NULL,
    old_value       DOUBLE PRECISION,
    new_value       DOUBLE PRECISION,
    reason          TEXT,
    sample_size     INTEGER,
    applied         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS positions (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       BIGINT REFERENCES signals(id),
    symbol          VARCHAR(32) NOT NULL,
    side            VARCHAR(8) NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    qty             DOUBLE PRECISION NOT NULL,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    trailing_stop   DOUBLE PRECISION,
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    exit_price      DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    exit_reason     VARCHAR(32),
    raw_payload     JSONB
);

CREATE TABLE IF NOT EXISTS basis_history (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(32) NOT NULL,
    basis_pct       DOUBLE PRECISION,
    funding_rate    DOUBLE PRECISION,
    ts              TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_basis_symbol_ts ON basis_history(symbol, ts DESC);
"""


class Database:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        dsn = (
            f"postgresql://{self.cfg['user']}:{self.cfg['password']}"
            f"@{self.cfg['host']}:{self.cfg['port']}/{self.cfg['database']}"
        )
        self._pool = await asyncpg.create_pool(
            dsn, min_size=2, max_size=10, command_timeout=30,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("Database connected and schema applied")

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute(self, query: str, *args) -> str:
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)