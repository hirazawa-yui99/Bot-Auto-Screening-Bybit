"""
persistence/models.py
Lightweight dataclasses for DB records.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SignalRow:
    id: Optional[int] = None
    symbol: str = ""
    direction: str = ""
    timestamp: datetime = None
    entry_price: float = 0.0
    raw_score: float = 0.0
    calibrated_confidence: float = 0.0
    is_high_confidence: bool = False
    gate_reasons: List[str] = field(default_factory=list)
    llm_response: Dict[str, Any] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)
    composite_score: float = 0.0
    composite_direction: str = "neutral"
    composite_conviction: str = "low"
    zeta: float = 0.0
    hurst: float = 0.5
    half_life_hours: float = 0.0
    basis_pct: float = 0.0
    funding_annualized_pct: float = 0.0
    volume_zscore: float = 0.0
    volume_anomaly_level: str = "normal"
    executed: bool = False
    execution_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRow:
    signal_id: int = 0
    result: str = ""
    won: bool = False
    pnl_pct: float = 0.0
    elapsed_hours: float = 0.0
    perps_data: Dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime = None


@dataclass
class PositionRow:
    id: Optional[int] = None
    signal_id: Optional[int] = None
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    qty: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: Optional[float] = None
    opened_at: datetime = None
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None