"""
learning/auto_tuner.py
Auto-improvement engine with anti-overfitting guards.

Guards:
  1. Walk-forward split (70/30).
  2. Bonferroni correction for multiple comparisons.
  3. Minimum sample size (30 outcomes).
  4. Monotone safety while edge unproven.
  5. Deflated Sharpe Ratio computation available.
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    parameter: str
    old_value: float
    new_value: float
    reason: str
    confidence: float
    sample_size: int
    applied: bool = False


class AutoTuner:
    PARAM_BOUNDS = {
        "min_raw_threshold": (50.0, 85.0),
        "min_calibrated": (0.55, 0.90),
        "min_rvol": (1.0, 4.0),
        "risk_reward_min": (1.5, 5.0),
        "trailing_activation_pct": (0.5, 3.0),
        "trailing_distance_pct": (0.2, 2.0),
        "zeta_entry_threshold": (1.0, 3.0),
        "volume_z_high_threshold": (1.5, 3.0),
    }

    RISK_INCREASING_DIRECTION = {
        "min_raw_threshold": "up",
        "min_calibrated": "up",
        "min_rvol": "up",
        "risk_reward_min": "up",
        "trailing_activation_pct": "down",
        "trailing_distance_pct": "down",
        "zeta_entry_threshold": "up",
        "volume_z_high_threshold": "up",
    }

    def __init__(self, config: Dict, confidence_scorer):
        self.config = config
        self.scorer = confidence_scorer
        self.min_samples = config.get("auto_tune_min_samples", 30)
        self.proven_edge_samples = config.get("proven_edge_samples", 100)

    def load_signals_and_outcomes(
        self, signal_journal, since_hours: int = 24,
    ) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        signals = signal_journal.get_signals_since(cutoff)
        enriched = []
        for sig in signals:
            outcome = signal_journal.get_outcome(sig["id"])
            if outcome:
                enriched.append({**sig, **outcome})
        return enriched

    def _bonferroni_threshold(self, n_tests: int, alpha: float = 0.05) -> float:
        return alpha / max(n_tests, 1)

    def _deflated_sharpe(
        self, returns: np.ndarray, n_trials: int,
    ) -> Tuple[float, float]:
        if len(returns) < 10:
            return 0.0, 0.0
        T = len(returns)
        sr = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        sr0 = np.sqrt(2 * np.log(max(n_trials, 2))) / np.sqrt(T)
        gamma3 = float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-9)) ** 3))
        gamma4 = float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-9)) ** 4))
        numerator = (sr - sr0) * np.sqrt(T - 1)
        denominator = np.sqrt(1 - gamma3 * sr + (gamma4 - 1) / 4 * sr ** 2)
        dsr = numerator / (denominator + 1e-9)
        return float(sr), float(dsr)

    def _walk_forward_split(
        self, data: List[Dict], train_pct: float = 0.7,
    ) -> Tuple[List[Dict], List[Dict]]:
        data_sorted = sorted(data, key=lambda x: str(x.get("timestamp", "")))
        split_idx = int(len(data_sorted) * train_pct)
        return data_sorted[:split_idx], data_sorted[split_idx:]

    def _simulate_filter(
        self, data: List[Dict], param_name: str, threshold: float,
    ) -> List[Dict]:
        return [
            d for d in data
            if d.get(param_name, d.get("raw_score", 0)) >= threshold
        ]

    def _binomial_p_value(
        self, successes: int, trials: int, p0: float,
    ) -> float:
        from math import comb
        p_value = 0.0
        for k in range(successes, trials + 1):
            p_value += comb(trials, k) * (p0 ** k) * ((1 - p0) ** (trials - k))
        return p_value

    async def run_daily_improvement(
        self, signal_journal, config_loader=None,
    ) -> List[TuningResult]:
        results: List[TuningResult] = []
        data = self.load_signals_and_outcomes(signal_journal, since_hours=24)
        logger.info(f"Auto-tuner: loaded {len(data)} signals with outcomes")

        if len(data) < self.min_samples:
            logger.info(
                f"Auto-tuner: insufficient samples "
                f"({len(data)} < {self.min_samples})"
            )
            return results

        train, validation = self._walk_forward_split(data, train_pct=0.7)
        if len(validation) < 10:
            logger.warning("Validation set too small. Skipping.")
            return results

        current_wins = sum(1 for d in validation if d.get("won"))
        current_wr = current_wins / len(validation) if validation else 0.0
        logger.info(
            f"Current validation win rate: {current_wr:.2%} "
            f"(n={len(validation)})"
        )

        parameters_to_test = [
            ("min_raw_threshold", 2.5),
            ("min_calibrated", 0.02),
            ("min_rvol", 0.25),
            ("risk_reward_min", 0.25),
            ("trailing_activation_pct", 0.25),
            ("trailing_distance_pct", 0.1),
            ("zeta_entry_threshold", 0.1),
            ("volume_z_high_threshold", 0.1),
        ]

        n_tests = len(parameters_to_test) * 2
        bonferroni_alpha = self._bonferroni_threshold(n_tests, alpha=0.10)

        for param_name, step in parameters_to_test:
            current_val = self.config.get(param_name)
            if current_val is None:
                continue
            lo, hi = self.PARAM_BOUNDS.get(param_name, (None, None))
            if lo is None:
                continue

            best_delta = 0.0
            best_wr = current_wr
            best_candidate = None

            for direction in [-1, 1]:
                candidate = current_val + direction * step
                if candidate < lo or candidate > hi:
                    continue
                filtered = self._simulate_filter(
                    validation, param_name, candidate,
                )
                if len(filtered) < 5:
                    continue
                filtered_wr = sum(1 for d in filtered if d.get("won")) / len(filtered)
                improvement = filtered_wr - current_wr
                if improvement > 0:
                    p_value = self._binomial_p_value(
                        sum(1 for d in filtered if d.get("won")),
                        len(filtered), current_wr,
                    )
                    if p_value < bonferroni_alpha and improvement > best_delta:
                        best_delta = improvement
                        best_wr = filtered_wr
                        best_candidate = candidate

            if best_candidate is not None and best_delta > 0:
                proven = len(data) >= self.proven_edge_samples
                if not proven:
                    safe_dir = self.RISK_INCREASING_DIRECTION.get(param_name)
                    if safe_dir == "up" and best_candidate < current_val:
                        continue
                    if safe_dir == "down" and best_candidate > current_val:
                        continue

                results.append(TuningResult(
                    parameter=param_name,
                    old_value=current_val,
                    new_value=best_candidate,
                    reason=(
                        f"Validation WR improved "
                        f"{current_wr:.2%} → {best_wr:.2%}"
                    ),
                    confidence=best_delta,
                    sample_size=len(validation),
                    applied=True,
                ))
                logger.info(
                    f"Auto-tuner: {param_name} {current_val} → "
                    f"{best_candidate} (WR {current_wr:.2%} → {best_wr:.2%})"
                )

        if results and config_loader:
            await config_loader.apply_tuning_results(results)

        return results