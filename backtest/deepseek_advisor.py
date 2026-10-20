"""
backtest/deepseek_advisor.py
Post-backtest analysis via DeepSeek API.

Flow:
  1. Receive backtest metrics + current config
  2. Build a compact prompt (metrics + tunable params only)
  3. Call DeepSeek → get structured JSON recommendations
  4. Validate recommendations against a whitelist
  5. Optionally apply to config.json (with backup)

Safety:
  - Never touches: api keys, webhooks, database, deepseek key
  - Never touches: system.*, auto_trade.enabled
  - All numeric changes must have a "current" value from config
  - All recommended values must be within allowed ranges
"""
import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Prompt
# ====================================================================== #
ADVISOR_SYSTEM_PROMPT = """You are a senior quantitative risk engineer auditing a crypto futures trading bot's config against backtest results.

Your task: analyze the backtest metrics and recommend specific config.json parameter changes that improve risk-adjusted returns.

CRITICAL RULES:
1. Return ONLY valid JSON — no markdown fences, no commentary
2. Every numeric recommendation MUST be justified by the backtest data
3. NEVER recommend changes to: api.*, deepseek.api_key, deepseek.base_url, database.*, auto_trade.enabled
4. If portfolio ruined (equity <= 0): prioritize position sizing + drawdown limits before anything else
5. If win rate < 45%: tighten entry filters (min_agree, momentum_guard, atr_pct)
6. If trades executed < 5: loosen filters (min_raw_threshold, min_agree, agreement thresholds)
7. If max drawdown < -15%: reduce max_position_pct and leverage
8. Be specific — use numeric values with clear reasoning

PARAMETER RANGES (must stay within):
- risk.max_position_pct:        0.5 – 5.0
- risk.default_leverage:        1 – 15
- risk.max_open_positions:      1 – 10
- risk.max_daily_loss_pct:      1.0 – 10.0
- risk.max_drawdown_pct:        5.0 – 25.0
- strategy.min_raw_threshold:   40.0 – 80.0
- strategy.min_calibrated:      0.35 – 0.85
- strategy.risk_reward_min:     1.0 – 3.0
- execution.max_deviation_pct:  1.5 – 10.0
- execution.cancel_on_rally_pct: 0.8 – 5.0
- execution.momentum_guard_roc_pct: 1.5 – 10.0
- execution.min_atr_pct:        0.1 – 1.5
- execution.max_atr_pct:        2.0 – 10.0
- execution.stall_timeout_candles: 3 – 15
- agreement.min_agree:          2 – 4
- setups.scalp_rvol_min:        0.8 – 3.0
- setups.intraday_rvol_min:     0.5 – 2.5
- setups.swing_rvol_min:        0.3 – 2.0

Return this exact JSON structure:
{
  "verdict": "PASS" | "NEEDS_CHANGES" | "FAIL",
  "summary": "1-3 sentences summarizing the diagnosis",
  "parameter_changes": [
    {
      "path": "risk.max_position_pct",
      "current": 20.0,
      "recommended": 3.0,
      "reason": "Portfolio ruined at 20% margin x 20x leverage (400% notional/trade). 3% x 10x = 30% notional.",
      "confidence": "high"
    }
  ],
  "risk_warnings": ["string", "..."],
  "keep_unchanged": ["risk.max_open_positions"]
}

If the strategy passed cleanly with good metrics, return:
{"verdict": "PASS", "summary": "...", "parameter_changes": [], "risk_warnings": [], "keep_unchanged": []}
"""


# ====================================================================== #
#  Whitelist + bounds
# ====================================================================== #
TUNABLE_PARAMS = {
    "risk.max_position_pct":        (0.5, 5.0),
    "risk.default_leverage":        (1, 15),
    "risk.max_open_positions":      (1, 10),
    "risk.max_daily_loss_pct":      (1.0, 10.0),
    "risk.max_drawdown_pct":        (5.0, 25.0),
    "strategy.min_raw_threshold":   (40.0, 80.0),
    "strategy.min_calibrated":      (0.35, 0.85),
    "strategy.risk_reward_min":     (1.0, 3.0),
    "execution.max_deviation_pct":  (1.5, 10.0),
    "execution.cancel_on_rally_pct": (0.8, 5.0),
    "execution.momentum_guard_roc_pct": (1.5, 10.0),
    "execution.momentum_guard_lookback_bars": (2, 15),
    "execution.momentum_guard_sigma": (1.5, 5.0),
    "execution.min_atr_pct":        (0.1, 1.5),
    "execution.max_atr_pct":        (2.0, 10.0),
    "execution.stall_timeout_candles": (3, 15),
    "agreement.min_agree":          (2, 4),
    "setups.scalp_rvol_min":        (0.8, 3.0),
    "setups.intraday_rvol_min":     (0.5, 2.5),
    "setups.swing_rvol_min":        (0.3, 2.0),
}

# Params the advisor must never touch — for belt-and-braces
BLOCKED_PARAMS = {
    "api.bybit_key", "api.bybit_secret", "api.discord_servers",
    "deepseek.api_key", "deepseek.base_url",
    "database.host", "database.database", "database.user",
    "database.password", "database.port",
    "auto_trade.enabled",
    "system.timezone",
}


# ====================================================================== #
#  Advisor
# ====================================================================== #
class DeepSeekAdvisor:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-v4-pro",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Build compact context
    # ------------------------------------------------------------------ #
    def _build_context(
        self,
        metrics: Dict,
        positions: List,
        current_config: Dict,
    ) -> Dict:
        """Compact backtest summary + current tunable params."""

        # Aggregate metrics
        m = metrics

        # Top 5 best + worst trades
        sorted_pos = sorted(
            positions,
            key=lambda p: p.equity_delta_usd,
            reverse=True,
        )
        top_wins = [
            {
                "symbol": p.symbol,
                "dir": p.direction,
                "pnl_pct": round(p.equity_delta_pct, 2),
                "pnl_usd": round(p.equity_delta_usd, 4),
            }
            for p in sorted_pos[:5] if p.equity_delta_usd > 0
        ]
        top_losses = [
            {
                "symbol": p.symbol,
                "dir": p.direction,
                "pnl_pct": round(p.equity_delta_pct, 2),
                "pnl_usd": round(p.equity_delta_usd, 4),
            }
            for p in sorted_pos[-5:] if p.equity_delta_usd < 0
        ]

        # Per-setup summary
        by_setup = {}
        for p in positions:
            s = getattr(p.trade, "setup_type", "unknown")
            if s not in by_setup:
                by_setup[s] = {
                    "trades": 0, "wins": 0, "total_pnl_usd": 0.0,
                }
            by_setup[s]["trades"] += 1
            by_setup[s]["wins"] += 1 if p.equity_delta_usd > 0 else 0
            by_setup[s]["total_pnl_usd"] += p.equity_delta_usd

        # Only tunable params from config
        tunable = {}
        for path in TUNABLE_PARAMS:
            val = self._get_nested(current_config, path)
            if val is not None:
                tunable[path] = val

        return {
            "backtest": {
                "total_trades": m.get("total_trades", 0),
                "wins": m.get("wins", 0),
                "losses": m.get("losses", 0),
                "win_rate": round(m.get("win_rate", 0.0), 4),
                "started_with_usd": m.get("started_with", 0),
                "ended_with_usd": m.get("ended_with", 0),
                "total_return_pct": round(m.get("total_return_pct", 0), 2),
                "peak_equity_usd": round(m.get("peak_equity", 0), 2),
                "max_drawdown_pct": round(m.get("max_drawdown_pct", 0), 2),
                "ruined": m.get("ruined", False),
                "peak_open_positions": m.get("peak_open_positions", 0),
                "skipped_due_to_margin": m.get("skipped_due_to_margin", 0),
                "skipped_due_to_positions": m.get("skipped_due_to_positions", 0),
                "skipped_due_to_drawdown": m.get("skipped_due_to_drawdown", 0),
            },
            "top_winners": top_wins,
            "top_losers": top_losses,
            "by_setup": by_setup,
            "current_config": tunable,
        }

    @staticmethod
    def _get_nested(d: Dict, path: str):
        parts = path.split(".")
        cur = d
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    @staticmethod
    def _set_nested(d: Dict, path: str, value: Any):
        parts = path.split(".")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    # ------------------------------------------------------------------ #
    #  Call DeepSeek
    # ------------------------------------------------------------------ #
    async def analyze(
        self,
        metrics: Dict,
        positions: List,
        current_config: Dict,
    ) -> Optional[Dict]:
        if not self.api_key:
            logger.warning(
                "DeepSeek advisor: no api_key configured — skipping"
            )
            return None

        context = self._build_context(metrics, positions, current_config)

        user_msg = (
            "Backtest results:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Return ONLY the JSON object described in the system prompt. "
            "No markdown, no explanation, no preamble — just the JSON."
        )

        # Attempt 1: user-configured model
        primary = self._build_payload(
            self.model, user_msg, with_json_format=True,
        )
        logger.info(
            f"DeepSeek advisor: calling {self.model} "
            f"(timeout={self.timeout}s, max_tokens=8000)"
        )
        data = await self._call_api(primary, expect_json=True)

        # Attempt 2: retry without response_format
        if data is None:
            logger.warning(
                f"{self.model} failed — retrying without "
                f"response_format (timeout={self.timeout}s)"
            )
            fallback_payload = self._build_payload(
                self.model, user_msg, with_json_format=False,
            )
            data = await self._call_api(
                fallback_payload, expect_json=False,
            )

        # Attempt 3: fast model fallback (deepseek-chat)
        if data is None and self.model != "deepseek-chat":
            logger.warning(
                f"{self.model} failed twice — falling back to "
                f"deepseek-chat (fast, non-reasoning)"
            )
            fallback_payload = self._build_payload(
                "deepseek-chat", user_msg, with_json_format=True,
            )
            data = await self._call_api(
                fallback_payload, expect_json=True,
            )

        if data is None:
            logger.error(
                f"DeepSeek advisor: ALL attempts failed "
                f"(primary={self.model}). "
                f"Advisor skipped — check network and API key."
            )
            return None

        if "error" in data:
            logger.error(f"DeepSeek API error: {data['error']}")
            return None

        # Parse
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            finish_reason = choice.get("finish_reason", "?")
            usage = data.get("usage", {})
            actual_model = data.get("model", "?")

            logger.info(
                f"DeepSeek response: model={actual_model} "
                f"finish={finish_reason} "
                f"content_len={len(content)} "
                f"reasoning_len={len(reasoning)} "
                f"tokens={usage}"
            )

            if not content.strip() and reasoning.strip():
                logger.warning(
                    "DeepSeek returned empty content but non-empty "
                    "reasoning_content — extracting JSON from reasoning"
                )
                content = reasoning

            if not content.strip():
                logger.error(
                    f"DeepSeek returned empty content. "
                    f"finish={finish_reason}, usage={usage}. "
                    f"Raw response: {json.dumps(data)[:1000]}"
                )
                return None

            if finish_reason == "length":
                logger.error(
                    f"DeepSeek truncated at max_tokens. "
                    f"Content: {content[:500]}"
                )
                return None

            content = content.strip()
            if content.startswith("```"):
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.split("```")[0]
                content = content.strip()

            first = content.find("{")
            last = content.rfind("}")
            if first >= 0 and last > first:
                content = content[first:last + 1]

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse DeepSeek JSON: {e}. "
                f"Raw content: {content[:1000]}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error parsing DeepSeek response: {e}"
            )
            return None

    def _build_payload(
        self, model: str, user_msg: str, with_json_format: bool,
    ) -> Dict:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 8000,
        }
        # Only send reasoning_effort for reasoning models
        if "pro" in model.lower() or "reasoner" in model.lower():
            payload["reasoning_effort"] = "low"
        if with_json_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _call_api(
        self, payload: Dict, expect_json: bool = True,
    ) -> Optional[Dict]:
        """Send payload to DeepSeek. Returns parsed JSON response or None."""
        model = payload.get("model", "?")
        url = f"{self.base_url}/chat/completions"

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout,
                    connect=15,            # fail fast if unreachable
                    sock_read=self.timeout,
                ),
            ) as session:
                async with session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(
                            f"DeepSeek HTTP {resp.status} "
                            f"(model={model}): {body[:500]}"
                        )
                        return None

                    try:
                        return await resp.json()
                    except aiohttp.ContentTypeError:
                        text = await resp.text()
                        logger.error(
                            f"DeepSeek returned non-JSON content: "
                            f"{text[:500]}"
                        )
                        return None

        except asyncio.TimeoutError:
            # Most common failure for reasoning models — this is what
            # was happening before. Log it explicitly so it's not blank.
            logger.error(
                f"DeepSeek TIMEOUT after {self.timeout}s "
                f"(model={model}). "
                f"Reasoning models can take 90-180s — increase timeout "
                f"or switch to a fast model."
            )
            return None

        except aiohttp.ClientConnectorError as e:
            logger.error(
                f"DeepSeek CONNECTION ERROR (model={model}): "
                f"cannot reach {self.base_url} — {e}"
            )
            return None

        except aiohttp.ClientError as e:
            logger.error(
                f"DeepSeek CLIENT ERROR (model={model}): "
                f"{type(e).__name__}: {e or '(no message)'}"
            )
            return None

        except Exception as e:
            logger.error(
                f"DeepSeek UNEXPECTED ERROR (model={model}): "
                f"{type(e).__name__}: {e or '(no message)'}"
            )
            return None

    # ------------------------------------------------------------------ #
    #  Validate + sanitize
    # ------------------------------------------------------------------ #
    def validate(
        self,
        recommendation: Dict,
        current_config: Dict,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Check each proposed change against whitelist + bounds.
        Returns (valid_changes, rejection_reasons).
        """
        valid = []
        rejected = []

        for change in recommendation.get("parameter_changes", []):
            path = change.get("path", "")
            rec_val = change.get("recommended")
            cur_val = change.get("current")

            # 1. Blocked path
            if path in BLOCKED_PARAMS:
                rejected.append(f"{path}: BLOCKED (never tunable)")
                continue

            # 2. Whitelist
            if path not in TUNABLE_PARAMS:
                rejected.append(f"{path}: not in tunable whitelist")
                continue

            # 3. Numeric
            if not isinstance(rec_val, (int, float)):
                rejected.append(
                    f"{path}: recommended value not numeric "
                    f"({rec_val})"
                )
                continue

            # 4. Bounds
            lo, hi = TUNABLE_PARAMS[path]
            if not (lo <= rec_val <= hi):
                rejected.append(
                    f"{path}: {rec_val} outside bounds "
                    f"[{lo}, {hi}]"
                )
                continue

            # 5. Current matches actual config (prevents hallucinated deltas)
            actual_current = self._get_nested(current_config, path)
            if actual_current is None:
                rejected.append(
                    f"{path}: not present in config"
                )
                continue
            if (
                cur_val is not None
                and isinstance(cur_val, (int, float))
                and abs(float(cur_val) - float(actual_current)) > 1e-6
            ):
                rejected.append(
                    f"{path}: claimed current={cur_val} but "
                    f"actual={actual_current}"
                )
                continue

            # 6. No-op check
            if abs(float(rec_val) - float(actual_current)) < 1e-6:
                rejected.append(
                    f"{path}: no change (same value)"
                )
                continue

            valid.append({
                "path": path,
                "current": actual_current,
                "recommended": rec_val,
                "reason": change.get("reason", ""),
                "confidence": change.get("confidence", "medium"),
            })

        return valid, rejected

    # ------------------------------------------------------------------ #
    #  Apply to config
    # ------------------------------------------------------------------ #
    def apply_changes(
        self,
        config: Dict,
        valid_changes: List[Dict],
    ) -> Dict:
        """Return a deep-copied config with changes applied."""
        import copy
        new_config = copy.deepcopy(config)
        for ch in valid_changes:
            self._set_nested(
                new_config, ch["path"], ch["recommended"],
            )
        return new_config


# ====================================================================== #
#  Config file helpers
# ====================================================================== #
def load_config(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def save_config(path: str, config: Dict, backup: bool = True):
    """Atomic write with optional backup."""
    if backup and os.path.exists(path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.bak.{ts}"
        shutil.copy2(path, backup_path)
        logger.info(f"Backup written: {backup_path}")

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, path)
    logger.info(f"Config written: {path}")


def format_recommendation_report(rec: Dict, valid: List[Dict],
                                 rejected: List[str]) -> str:
    """Human-readable summary for stdout."""
    lines = []
    lines.append("=" * 78)
    lines.append("  DEEPSEEK ADVISOR RECOMMENDATION")
    lines.append("=" * 78)
    lines.append("")

    verdict = rec.get("verdict", "?")
    emoji = {
        "PASS": "✅",
        "NEEDS_CHANGES": "⚠️",
        "FAIL": "❌",
    }.get(verdict, "❓")

    lines.append(f"  {emoji} Verdict: {verdict}")
    lines.append("")
    lines.append(f"  {rec.get('summary', 'No summary')}")
    lines.append("")

    if valid:
        lines.append("  PARAMETER CHANGES (validated, will apply)")
        lines.append("  " + "-" * 74)
        lines.append(
            f"  {'Parameter':<38} {'Current':>10} → {'New':>10}  Conf"
        )
        lines.append("  " + "-" * 74)
        for ch in valid:
            lines.append(
                f"  {ch['path']:<38} "
                f"{ch['current']:>10} → {ch['recommended']:>10}  "
                f"{ch.get('confidence', '?'):<6}"
            )
            if ch.get("reason"):
                lines.append(f"    └─ {ch['reason'][:68]}")
        lines.append("")

    if rejected:
        lines.append("  REJECTED CHANGES (out of bounds / not whitelisted)")
        lines.append("  " + "-" * 74)
        for r in rejected:
            lines.append(f"  ✗ {r}")
        lines.append("")

    warnings = rec.get("risk_warnings", [])
    if warnings:
        lines.append("  RISK WARNINGS")
        lines.append("  " + "-" * 74)
        for w in warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    unchanged = rec.get("keep_unchanged", [])
    if unchanged:
        lines.append("  ADVISOR SAYS KEEP UNCHANGED")
        lines.append("  " + "-" * 74)
        lines.append(f"  {', '.join(unchanged[:8])}")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)