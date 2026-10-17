"""
notification/discord_bot.py
Rich Discord alerts for signals, trades, and tuning.
"""
import logging
from datetime import datetime
from typing import Dict, List

import aiohttp

logger = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(self, config: Dict):
        self.webhook = config.get("discord_webhook", "")
        self.live_webhook = config.get("discord_live_webhook", "") or self.webhook
        self.dashboard_webhook = (
            config.get("discord_dashboard_webhook", "") or self.webhook
        )

    async def _post(self, url: str, payload: Dict):
        if not url:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status >= 400:
                        logger.error(
                            f"Discord webhook error {resp.status}: "
                            f"{await resp.text()}"
                        )
        except Exception as e:
            logger.error(f"Discord post failed: {e}")

    async def send_signal_alert(self, signal, breakdown, llm_response: Dict):
        color = 0x00FF00 if signal.direction == "long" else 0xFF0000
        entry = signal.entry_price

        def pct(target: float) -> str:
            p = (target - entry) / entry * 100
            if signal.direction == "short":
                p = -p
            return f"{p:+.2f}%"

        # Build TP lines with weights
        tp_lines = []
        tps = signal.tp_list
        weights = signal.tp_weights[:len(tps)]
        for i, (tp, w) in enumerate(zip(tps, weights)):
            rr = abs(tp - entry) / abs(entry - signal.stop_loss) if entry != signal.stop_loss else 0
            tp_lines.append(
                f"**TP{i+1}**: `${tp:.6f}` ({pct(tp)}, R={rr:.1f}, {w*100:.0f}%)"
            )

        sl_pct = (signal.stop_loss - entry) / entry * 100
        if signal.direction == "short":
            sl_pct = -sl_pct

        embed = {
            "title": f"🚨 {signal.direction.upper()} SIGNAL: {signal.symbol}",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"`{entry:.6f}`", "inline": True},
                {"name": "Stop Loss", "value": f"`{signal.stop_loss:.6f}` ({sl_pct:+.2f}%)", "inline": True},
                {"name": "R:R (weighted)", "value": f"{signal.risk_reward:.2f}", "inline": True},
                {"name": "Confidence", "value": f"{signal.confidence:.1%}", "inline": True},
                {"name": "Raw Score", "value": f"{breakdown.raw_total:.1f}/100", "inline": True},
                {"name": "Composite", "value": f"{breakdown.composite_score:.1f}/20", "inline": True},
                {"name": "🎯 Take Profits", "value": "\n".join(tp_lines), "inline": False},
            ],
            "footer": {"text": f"Bybit Quant v9.2 | {datetime.utcnow().isoformat()}"},
        }
        if llm_response and llm_response.get("key_reasons"):
            embed["fields"].append({
                "name": "LLM Reasons",
                "value": "\n".join(f"• {r}" for r in llm_response["key_reasons"][:5]),
                "inline": False,
            })
        await self._post(self.live_webhook, {"embeds": [embed]})

    async def send_tuning_report(self, results: List):
        if not results:
            return
        lines = []
        for r in results:
            lines.append(
                f"**{r.parameter}**: `{r.old_value}` → `{r.new_value}` "
                f"({r.reason}, n={r.sample_size})"
            )
        embed = {
            "title": "🔧 Daily Auto-Tuning Report",
            "color": 0x3498DB,
            "description": "\n".join(lines),
            "footer": {"text": datetime.utcnow().isoformat()},
        }
        await self._post(self.dashboard_webhook, {"embeds": [embed]})

    async def send_emergency_alert(self, symbol: str, reason: str):
        embed = {
            "title": f"🛑 EMERGENCY: {symbol}",
            "color": 0xFF0000,
            "description": reason,
            "footer": {"text": datetime.utcnow().isoformat()},
        }
        await self._post(self.live_webhook, {"embeds": [embed]})