"""
agent/deepseek_agent.py
DeepSeek LLM integration for agentic trading analysis.

Configured for deepseek-v4-pro by default. Pro has stronger agentic
reasoning and multi-step tool use than Flash, at ~4-5x the cost.
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


DEEPSEEK_SYSTEM_PROMPT = """You are a senior quantitative trading analyst at a crypto hedge fund.
You specialise in Bybit perpetual futures. You receive structured market data
including: price action, technical indicators, CVD, Open Interest, funding rates,
account long/short ratios, top-trader long/short ratios, mean-reversion metrics
(Zeta, Hurst, half-life, ADF p-value), basis spread, volume anomaly data, and
funding regime classification.

Your job:
1. Assess whether the quantitative model's directional bias is supported by the data.
2. Identify specific invalidation conditions (price levels or metric thresholds).
3. Flag conflicting signals across timeframes.
4. Do NOT invent data. If a metric is missing, say so.
5. Do NOT output a confidence percentage.

Return ONLY a JSON object with these exact fields:
{
  "direction": "long" | "short" | "neutral",
  "agrees_with_quant": true | false,
  "key_reasons": ["...", "..."],
  "invalidation_conditions": ["...", "..."],
  "risk_notes": ["...", "..."],
  "conflicting_signals": ["...", "..."]
}
"""


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"
    model_high_confidence: str = "deepseek-v4-pro"
    temperature: float = 0.3
    max_tokens: int = 1500
    timeout: int = 45
    error_cooldown_sec: int = 900


class DeepSeekAgent:
    """
    Agentic LLM layer for trading analysis.

    Uses deepseek-v4-pro for all analysis. Disables itself for
    error_cooldown_sec after a fatal API error (balance, auth) so
    the scan loop doesn't hammer the API.
    """

    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._disabled_until: float = 0.0
        self._last_error: str = ""
        logger.info(
            f"DeepSeekAgent initialised: "
            f"model={self.cfg.model} "
            f"model_high_confidence={self.cfg.model_high_confidence}"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.cfg.timeout)
            )
        return self._session

    def _build_tools(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_klines",
                    "description": "Fetch OHLCV kline data for a symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "interval": {"type": "string"},
                            "limit": {"type": "integer", "default": 100},
                        },
                        "required": ["symbol", "interval"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_orderbook",
                    "description": "Fetch current orderbook depth for a symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "limit": {"type": "integer", "default": 50},
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_derivative_snapshot",
                    "description": "Fetch CVD, OI, funding, and L/S ratio data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                        },
                        "required": ["symbol"],
                    },
                },
            },
        ]

    def _select_model(self, prelim_score: float):
        """
        Returns (model_name, reasoning_effort).

        With both fields set to deepseek-v4-pro, always returns Pro.
        Reasoning effort scales with prelim_score to save tokens on
        weak signals.
        """
        if prelim_score >= 55:
            return self.cfg.model_high_confidence, "high"
        # Below 55: still Pro, but low reasoning effort to save output tokens
        return self.cfg.model, "low"

    def _is_disabled(self) -> bool:
        return time.time() < self._disabled_until

    def _disable(self, reason: str):
        self._disabled_until = time.time() + self.cfg.error_cooldown_sec
        self._last_error = reason
        logger.warning(
            f"DeepSeek agent disabled for "
            f"{self.cfg.error_cooldown_sec}s — reason: {reason}"
        )

    def _neutral_response(self, reason: str = "disabled") -> Dict:
        return {
            "direction": "neutral",
            "agrees_with_quant": False,
            "key_reasons": [],
            "invalidation_conditions": [],
            "risk_notes": [f"LLM unavailable: {reason}"],
            "conflicting_signals": [],
            "_llm_unavailable": True,
        }

    async def analyse(
        self,
        symbol: str,
        market_context: Dict,
        tool_executor: Optional[Any] = None,
    ) -> Dict:
        # Short-circuit if disabled
        if self._is_disabled():
            remaining = int(self._disabled_until - time.time())
            logger.debug(
                f"DeepSeek disabled for {remaining}s — "
                f"returning neutral for {symbol}"
            )
            return self._neutral_response("cooldown")

        prelim_score = market_context.get("preliminary_score", 0)
        model, reasoning_effort = self._select_model(prelim_score)

        messages = [
            {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyse {symbol}.\n\n"
                    f"Market context:\n"
                    f"{json.dumps(market_context, indent=2, default=str)}\n\n"
                    f"Respond ONLY in the required JSON format."
                ),
            },
        ]

        tools = self._build_tools()
        session = await self._get_session()

        for _ in range(5):
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            # Pro supports reasoning_effort
            if "pro" in model.lower():
                payload["reasoning_effort"] = reasoning_effort

            try:
                async with session.post(
                    f"{self.cfg.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    data = await resp.json()
            except Exception as e:
                logger.error(f"DeepSeek request failed: {e}")
                self._disable(f"request error: {e}")
                return self._neutral_response(str(e))

            # Handle API errors
            if "error" in data:
                err_msg = data["error"].get("message", "")
                logger.error(f"DeepSeek API error: {data['error']}")

                low = err_msg.lower()
                if any(kw in low for kw in (
                    "insufficient balance",
                    "insufficient_balance",
                    "authentication",
                    "unauthorized",
                    "invalid api key",
                    "quota",
                )):
                    self._disable(err_msg)
                    return self._neutral_response(err_msg)

                # Retry once at low effort on Pro failure
                if reasoning_effort != "low":
                    logger.warning(
                        "Retrying at low reasoning_effort"
                    )
                    reasoning_effort = "low"
                    continue

                return self._neutral_response(err_msg)

            choice = data["choices"][0]
            msg = choice["message"]

            if msg.get("tool_calls"):
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    logger.info(
                        f"DeepSeek requested tool: {fn_name}({fn_args})"
                    )

                    if tool_executor:
                        try:
                            result = await tool_executor(fn_name, fn_args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": "no tool_executor provided"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str),
                    })
                continue

            content = msg.get("content", "")
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                logger.warning(
                    f"DeepSeek returned non-JSON: {content[:200]}"
                )
                parsed = self._neutral_response("non-json response")

            return parsed

        return self._neutral_response("max_tool_rounds_exceeded")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()