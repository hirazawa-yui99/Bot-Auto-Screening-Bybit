"""
analysis/agreement_gate.py
Strict multi-signal agreement gate.

Inputs:
  smc_dir       "long" | "short" | "neutral"
  deriv_dir     "long" | "short" | "neutral"
  quant_dir     "long" | "short" | "neutral"
  heatmap_dir   "long" | "short" | "neutral"
  fib_direction "long" | "short" | "neutral"  (fib is valid only if in_zone)

Rules:
  - Collect non-neutral directions.
  - If any two disagree → FAIL (contradiction).
  - Require >= 2 non-neutral signals agreeing → PASS.
  - If 0 or 1 non-neutral → FAIL (insufficient evidence).
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class AgreementResult:
    passed: bool = False
    direction: str = "neutral"
    agreement_count: int = 0
    total_active: int = 0
    reason: str = ""
    details: Dict[str, str] = field(default_factory=dict)
    disagreements: List[str] = field(default_factory=list)


class AgreementGate:
    def __init__(self, min_agree: int = 2):
        self.min_agree = min_agree

    def check(
        self,
        smc_dir: str,
        deriv_dir: str,
        quant_dir: str,
        heatmap_dir: str,
        fib_direction: str = "neutral",
        fib_valid: bool = True,
    ) -> AgreementResult:
        r = AgreementResult()

        signals = {
            "smc": (smc_dir or "neutral").lower(),
            "deriv": (deriv_dir or "neutral").lower(),
            "quant": (quant_dir or "neutral").lower(),
            "heatmap": (heatmap_dir or "neutral").lower(),
        }
        if fib_valid and fib_direction in ("long", "short"):
            signals["fib"] = fib_direction

        r.details = dict(signals)

        active = {
            k: v for k, v in signals.items()
            if v in ("long", "short")
        }
        r.total_active = len(active)

        if r.total_active == 0:
            r.reason = "no active signals (all neutral)"
            return r

        directions = set(active.values())

        if len(directions) > 1:
            r.disagreements = [
                f"{k}={v}" for k, v in active.items()
            ]
            r.reason = (
                f"contradiction: "
                f"{', '.join(r.disagreements)}"
            )
            return r

        direction = directions.pop()
        r.direction = direction
        r.agreement_count = len(active)

        if r.agreement_count < self.min_agree:
            r.reason = (
                f"insufficient agreement: "
                f"{r.agreement_count} < {self.min_agree} required "
                f"({', '.join(active.keys())})"
            )
            return r

        r.passed = True
        r.reason = (
            f"all {r.agreement_count} signals agree: "
            f"{', '.join(active.keys())} = {direction}"
        )
        return r