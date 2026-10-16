"""
agent/deepseek_agent.py
DeepSeek LLM integration for agentic trading analysis.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


DEEPSEEK_SYSTEM_PROMPT = """You are a senior quantitative trading analyst at a crypto hedge fund.
You specialise in Bybit perpetual futures. You receive structured market data
including: price action, technical indicators, CVD, Open Interest, funding rates,
account long/short ratios, top-trader long/short ratios, mean-reversion metrics
(Zeta, Hurst, half-life, ADF p-value), basis spread, and volume anomaly data.

Your job:
1. Assess whether the quantitative model's directional bias is supported.
2. Identify specific invalidation conditions (price levels or metric thresholds).
3. Flag conflicting signals across timeframes.
4. Do NOT invent data. If a metric is missing, say so.
5. Do NOT output a confidence percentage.

Return ONLY a JSON object:
{
  "direction": "long" | "short" | "neutral",
  "agrees_with_quant": true | false,
  "key_reasons": ["..."],
  "invalidation_conditions": ["..."],
  "risk_notes": ["..."],
  "conflicting_signals": ["..."]
}
"""


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 1500
    timeout: int = 30


class DeepSeekAgent:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

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
                    "description": "Fetch OHLCV kline data",
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
                    "description": "Fetch orderbook depth",
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
        ]

    async def analyse(
        self,
        symbol: str,
        market_context: Dict,
        tool_executor: Optional[Any] = None,
    ) -> Dict:
        messages = [
            {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyse {symbol}.\n\n"
                    f"Market context:\n"
                    f"{json.dumps(market_context, indent=2, default=str)}\n\n"
                    f"Respond in JSON."
                ),
            },
        ]
        tools = self._build_tools()
        session = await self._get_session()

        for _ in range(5):
            payload = {
                "model": self.cfg.model,
                "messages": messages,
                "tools": tools,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
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
                return {"direction": "neutral", "agrees_with_quant": False,
                        "error": str(e)}

            if "error" in data:
                logger.error(f"DeepSeek API error: {data['error']}")
                return {"direction": "neutral", "agrees_with_quant": False}

            choice = data["choices"][0]
            msg = choice["message"]

            if msg.get("tool_calls"):
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    if tool_executor:
                        try:
                            result = await tool_executor(fn_name, fn_args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": "no tool_executor"}
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
                parsed = {
                    "direction": "neutral",
                    "agrees_with_quant": False,
                    "raw_response": content[:500],
                }
            return parsed

        return {"direction": "neutral", "agrees_with_quant": False,
                "error": "max_tool_rounds_exceeded"}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()