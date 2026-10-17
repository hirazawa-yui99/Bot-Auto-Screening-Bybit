"""
learning/signal_journal.py
Persist and query signal records + outcomes.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalJournal:
    def __init__(self, db):
        self.db = db

    async def log_signal(self, signal: Dict) -> int:
        breakdown = signal.get("_breakdown") or {}
        row_id = await self.db.fetchval(
            """
            INSERT INTO signals (
                symbol, direction, timestamp, entry_price,
                raw_score, calibrated_confidence, is_high_confidence,
                gate_reasons, llm_response, indicators,
                composite_score, composite_direction, composite_conviction,
                zeta, hurst, half_life_hours, basis_pct,
                funding_annualized_pct, volume_zscore, volume_anomaly_level,
                technical_score, composite_score_pct, derivative_score,
                sentiment_score, pattern_score, llm_score
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
                $21,$22,$23,$24,$25,$26
            ) RETURNING id
            """,
            signal["symbol"],
            signal["direction"],
            datetime.fromisoformat(signal["timestamp"]),
            signal["entry_price"],
            signal.get("raw_score", 0.0),
            signal.get("calibrated_confidence", 0.0),
            signal.get("is_high_confidence", False),
            json.dumps(signal.get("gate_reasons", [])),
            json.dumps(signal.get("llm_response", {})),
            json.dumps(signal.get("indicators", {})),
            signal.get("composite_score", 0.0),
            signal.get("composite_direction", "neutral"),
            signal.get("composite_conviction", "low"),
            signal.get("zeta", 0.0),
            signal.get("hurst", 0.5),
            signal.get("half_life_hours", 0.0),
            signal.get("basis_pct", 0.0),
            signal.get("funding_annualized_pct", 0.0),
            signal.get("volume_zscore", 0.0),
            signal.get("volume_anomaly_level", "normal"),
            breakdown.get("technical_score", 0.0),
            breakdown.get("composite_score", 0.0),
            breakdown.get("derivative_score", 0.0),
            breakdown.get("sentiment_score", 0.0),
            breakdown.get("pattern_score", 0.0),
            breakdown.get("llm_score", 0.0),
        )
        return int(row_id)

    async def mark_executed(self, signal_id: int, result: Dict):
        await self.db.execute(
            "UPDATE signals SET executed = TRUE, execution_result = $1 WHERE id = $2",
            json.dumps(result, default=str), signal_id,
        )

    async def record_outcome(self, signal_id: int, outcome: Dict):
        await self.db.execute(
            """
            INSERT INTO outcomes (
                signal_id, result, won, pnl_pct, elapsed_hours,
                perps_data, resolved_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (signal_id) DO UPDATE SET
                result = EXCLUDED.result,
                won = EXCLUDED.won,
                pnl_pct = EXCLUDED.pnl_pct,
                resolved_at = EXCLUDED.resolved_at
            """,
            signal_id,
            outcome["result"],
            outcome["won"],
            outcome.get("pnl_pct", 0.0),
            outcome.get("elapsed_hours", 0.0),
            json.dumps(outcome.get("perps_data", {})),
            datetime.fromisoformat(outcome["resolved_at"]),
        )

    def get_signals_since(self, cutoff: datetime) -> List[Dict]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run():
            rows = await self.db.fetch(
                "SELECT * FROM signals WHERE timestamp >= $1 ORDER BY timestamp",
                cutoff,
            )
            return [dict(r) for r in rows]

        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(_run(), loop).result()
        return loop.run_until_complete(_run())

    def get_outcome(self, signal_id: int) -> Optional[Dict]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run():
            row = await self.db.fetchrow(
                "SELECT * FROM outcomes WHERE signal_id = $1", signal_id,
            )
            return dict(row) if row else None

        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(_run(), loop).result()
        return loop.run_until_complete(_run())

    def get_unresolved_signals(self) -> List[Dict]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run():
            rows = await self.db.fetch(
                """
                SELECT s.* FROM signals s
                LEFT JOIN outcomes o ON o.signal_id = s.id
                WHERE o.signal_id IS NULL
                  AND s.timestamp >= NOW() - INTERVAL '7 days'
                ORDER BY s.timestamp
                """
            )
            return [dict(r) for r in rows]

        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(_run(), loop).result()
        return loop.run_until_complete(_run())