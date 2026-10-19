"""
notification/discord_bot.py
Per-server Discord routing with combined and split modes.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(self, config: Dict):
        servers = config.get("discord_servers")
        if servers:
            self._servers = servers
        else:
            default = config.get("discord_webhook", "") or ""
            self._servers = [{
                "name": "default",
                "mode": "split",
                "live": config.get("discord_live_webhook") or default,
                "quant": config.get("discord_quant_webhook") or default,
                "dashboard": (
                    config.get("discord_dashboard_webhook") or default
                ),
                "alerts": config.get("discord_alerts_webhook") or default,
            }]
        modes = [s.get("mode", "split") for s in self._servers]
        logger.info(
            f"DiscordNotifier: {len(self._servers)} server(s) — modes={modes}"
        )

    # ================================================================== #
    #  Low-level HTTP
    # ================================================================== #
    @staticmethod
    def _split_urls(s: str) -> List[str]:
        if not s:
            return []
        return [u.strip() for u in s.split(",") if u.strip()]

    async def _post_single(
        self, url: str, payload: Dict,
        file_path: Optional[str] = None,
    ):
        try:
            async with aiohttp.ClientSession() as session:
                if file_path and os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        data = f.read()
                    form = aiohttp.FormData()
                    form.add_field(
                        "payload_json", json.dumps(payload),
                        content_type="application/json",
                    )
                    form.add_field(
                        "file", data,
                        filename=os.path.basename(file_path),
                        content_type="image/png",
                    )
                    async with session.post(url, data=form) as resp:
                        await self._check_response(resp, url)
                else:
                    async with session.post(url, json=payload) as resp:
                        await self._check_response(resp, url)
        except Exception as e:
            raise RuntimeError(f"{url[:50]}... → {e}")

    @staticmethod
    async def _check_response(resp, url: str):
        if resp.status in (200, 204):
            return
        if resp.status == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            logger.warning(f"Discord rate limit — wait {retry_after}s")
            await asyncio.sleep(retry_after)
            raise RuntimeError("Rate limited")
        body = await resp.text()
        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

    # ================================================================== #
    #  Dispatchers
    # ================================================================== #
    async def _dispatch_embeds(
        self,
        channel: str,
        alert_embed: Optional[Dict] = None,
        quant_embed: Optional[Dict] = None,
        file_path: Optional[str] = None,
    ):
        tasks = []

        for server in self._servers:
            mode = server.get("mode", "split")
            name = server.get("name", "?")

            if mode == "combined":
                wh = (server.get("webhook") or "").strip()
                if not wh:
                    continue
                embeds: List[Dict] = []
                if alert_embed:
                    e = dict(alert_embed)
                    e.pop("image", None)
                    embeds.append(e)
                if quant_embed:
                    e = dict(quant_embed)
                    e.pop("image", None)
                    embeds.append(e)
                if not embeds:
                    continue
                if file_path:
                    embeds[-1]["image"] = {
                        "url": "attachment://" + os.path.basename(file_path)
                    }
                payload = {"embeds": embeds}
                tasks.append(
                    (name, self._post_single(wh, payload, file_path))
                )
            else:
                if alert_embed:
                    for url in self._split_urls(server.get(channel, "")):
                        e = dict(alert_embed)
                        e.pop("image", None)
                        if file_path:
                            e["image"] = {
                                "url": "attachment://"
                                + os.path.basename(file_path)
                            }
                        tasks.append(
                            (name, self._post_single(
                                url, {"embeds": [e]}, file_path,
                            ))
                        )
                if quant_embed:
                    for url in self._split_urls(server.get("quant", "")):
                        e = dict(quant_embed)
                        e.pop("image", None)
                        if file_path:
                            e["image"] = {
                                "url": "attachment://"
                                + os.path.basename(file_path)
                            }
                        tasks.append(
                            (name, self._post_single(
                                url, {"embeds": [e]}, file_path,
                            ))
                        )

        if not tasks:
            logger.debug("No matching webhooks — nothing dispatched")
            return
        results = await asyncio.gather(
            *[t[1] for t in tasks], return_exceptions=True,
        )
        for (name, _), r in zip(tasks, results):
            if isinstance(r, Exception):
                logger.error(f"Discord send failed (server={name}): {r}")

    async def _dispatch_simple(
        self,
        channel: str,
        embed: Dict,
        file_path: Optional[str] = None,
    ):
        tasks = []
        for server in self._servers:
            mode = server.get("mode", "split")
            name = server.get("name", "?")

            if mode == "combined":
                wh = (server.get("webhook") or "").strip()
                if not wh:
                    continue
                e = dict(embed)
                e.pop("image", None)
                if file_path:
                    e["image"] = {
                        "url": "attachment://"
                        + os.path.basename(file_path)
                    }
                tasks.append(
                    (name, self._post_single(
                        wh, {"embeds": [e]}, file_path,
                    ))
                )
            else:
                for url in self._split_urls(server.get(channel, "")):
                    e = dict(embed)
                    e.pop("image", None)
                    if file_path:
                        e["image"] = {
                            "url": "attachment://"
                            + os.path.basename(file_path)
                        }
                    tasks.append(
                        (name, self._post_single(
                            url, {"embeds": [e]}, file_path,
                        ))
                    )

        if not tasks:
            return
        results = await asyncio.gather(
            *[t[1] for t in tasks], return_exceptions=True,
        )
        for (name, _), r in zip(tasks, results):
            if isinstance(r, Exception):
                logger.error(f"Discord send failed (server={name}): {r}")

    # ================================================================== #
    #  Embed builders
    # ================================================================== #
    def _build_alert_embed(
        self,
        signal,
        breakdown,
        llm_response: Dict,
        setup_label: Optional[str] = None,
        signal_id: int = 0,
        delay_seconds: int = 0,
    ) -> Dict:
        entry = signal.entry_price
        color = 0x3BA55D if signal.direction == "long" else 0xED4245

        def pct_str(target: float) -> str:
            p = (target - entry) / entry * 100
            if signal.direction == "short":
                p = -p
            return f"{p:+.2f}%"

        tp_lines = []
        tps = signal.tp_list
        weights = signal.tp_weights[:len(tps)]
        for i, (tp, w) in enumerate(zip(tps, weights)):
            rr = (
                abs(tp - entry) / abs(entry - signal.stop_loss)
                if entry != signal.stop_loss else 0
            )
            tp_lines.append(
                f"**TP{i+1}**: `${tp:.6f}` "
                f"({pct_str(tp)}, R={rr:.1f}, {w*100:.0f}%)"
            )

        sl_pct = (signal.stop_loss - entry) / entry * 100
        if signal.direction == "short":
            sl_pct = -sl_pct

        title_prefix = f"[{setup_label.upper()}] " if setup_label else ""

        fields = [
            {"name": "Entry", "value": f"`{entry:.6f}`", "inline": True},
            {"name": "Stop Loss",
             "value": f"`{signal.stop_loss:.6f}` ({sl_pct:+.2f}%)",
             "inline": True},
            {"name": "R:R (weighted)",
             "value": f"{signal.risk_reward:.2f}", "inline": True},
            {"name": "Confidence",
             "value": f"{signal.confidence:.1%}", "inline": True},
            {"name": "Raw Score",
             "value": f"{breakdown.raw_total:.1f}/100", "inline": True},
            {"name": "Composite",
             "value": f"{breakdown.composite_score:.1f}/20",
             "inline": True},
            {"name": "🎯 Take Profits",
             "value": "\n".join(tp_lines) or "—",
             "inline": False},
        ]

        if signal_id > 0:
            if delay_seconds > 0:
                fields.append({
                    "name": "⏱ Queue",
                    "value": (
                        f"Signal `#{signal_id}` · auto-trade ON\n"
                        f"Executes in **{delay_seconds}s**"
                    ),
                    "inline": False,
                })
            else:
                fields.append({
                    "name": "⏱ Queue",
                    "value": (
                        f"Signal `#{signal_id}` · observation mode"
                    ),
                    "inline": False,
                })

        embed = {
            "title": (
                f"{title_prefix}"
                f"{'🟢' if signal.direction == 'long' else '🔴'} "
                f"{signal.direction.upper()} SIGNAL · {signal.symbol}"
            ),
            "color": color,
            "fields": fields,
            "footer": {
                "text": (
                    "Bybit Quant v9.2 · "
                    f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        if llm_response and llm_response.get("key_reasons"):
            embed["fields"].append({
                "name": "LLM Reasons",
                "value": "\n".join(
                    f"• {r}" for r in llm_response["key_reasons"][:5]
                ),
                "inline": False,
            })
        return embed

    def _build_quant_embed(
        self,
        symbol: str,
        composite,
        breakdown,
        deriv,
        indicators: Dict,
    ) -> Optional[Dict]:
        if composite is None or breakdown is None or deriv is None:
            return None

        mr = composite.mean_reversion
        basis = composite.basis
        vol = composite.volume
        funding = composite.funding

        def f(x, fmt=".2f"):
            return f"{x:{fmt}}"

        return {
            "title": f"📊 QUANT ANALYSIS · {symbol}",
            "color": 0x5865F2,
            "fields": [
                {
                    "name": "🎯 Composite",
                    "value": (
                        f"Score: **{f(composite.total_score, '+.3f')}**\n"
                        f"Direction: **{composite.direction.upper()}**\n"
                        f"Conviction: **{composite.conviction.upper()}**\n"
                        f"Agreement: "
                        f"{f(composite.agreement_ratio * 100, '.0f')}%"
                    ),
                    "inline": True,
                },
                {
                    "name": "📊 Confidence",
                    "value": (
                        f"Raw: **{f(breakdown.raw_total, '.1f')}** / 100\n"
                        f"Calibrated: "
                        f"**{f(breakdown.calibrated_confidence * 100, '.1f')}%**\n"
                        f"Tech: {f(breakdown.technical_score, '.1f')} | "
                        f"Deriv: {f(breakdown.derivative_score, '.1f')}\n"
                        f"Sent: {f(breakdown.sentiment_score, '.1f')} | "
                        f"Pattern: {f(breakdown.pattern_score, '.1f')}"
                    ),
                    "inline": True,
                },
                {
                    "name": "🔄 Mean Reversion",
                    "value": (
                        f"Zeta: **{f(mr.zeta, '+.2f')}**\n"
                        f"Hurst: {f(mr.hurst, '.3f')}\n"
                        f"Regime: {mr.regime}\n"
                        f"Half-life: {f(mr.half_life_hours, '.1f')}h"
                    ),
                    "inline": True,
                },
                {
                    "name": "📈 Derivatives",
                    "value": (
                        f"CVD div: **{deriv.cvd_divergence}**\n"
                        f"OI 1h: {f(deriv.oi_change_1h, '+.2f')}%\n"
                        f"OI 4h: {f(deriv.oi_change_4h, '+.2f')}%\n"
                        f"OI div: {deriv.oi_divergence}\n"
                        f"Top L/S: {f(deriv.top_trader_ls_ratio, '.2f')}"
                    ),
                    "inline": True,
                },
                {
                    "name": "💰 Funding",
                    "value": (
                        f"Rate: **{f(funding.rate_pct, '+.4f')}%**\n"
                        f"Annual: {f(funding.annualized_pct, '+.1f')}%\n"
                        f"Regime: {funding.regime}\n"
                        f"Bias: {funding.contrarian_bias}"
                    ),
                    "inline": True,
                },
                {
                    "name": "📦 Volume",
                    "value": (
                        f"RVOL: {f(vol.rvol, '.2f')}\n"
                        f"Z: **{f(vol.z_score, '+.2f')}**\n"
                        f"Level: {vol.anomaly_level}\n"
                        f"VWPD: {f(vol.vwpd, '+.2f')}"
                    ),
                    "inline": True,
                },
                {
                    "name": "⚖️ Basis",
                    "value": (
                        f"Basis: **{f(basis.basis_pct, '+.4f')}%**\n"
                        f"Regime: {basis.regime}\n"
                        f"Signal: {basis.signal_bias}"
                    ),
                    "inline": True,
                },
                {
                    "name": "📐 Indicators",
                    "value": (
                        f"EMA9: `{f(indicators.get('ema_9', 0), '.4f')}`\n"
                        f"EMA21: `{f(indicators.get('ema_21', 0), '.4f')}`\n"
                        f"RSI: **{f(indicators.get('rsi', 0), '.1f')}**\n"
                        f"ATR: {f(indicators.get('atr', 0), '.4f')}"
                    ),
                    "inline": True,
                },
            ],
            "footer": {
                "text": (
                    "Bybit Quant v9.2 · "
                    f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ================================================================== #
    #  Public API
    # ================================================================== #
    async def send_signal_alert(
        self,
        signal,
        breakdown,
        llm_response: Dict,
        chart_path: Optional[str] = None,
        composite=None,
        deriv=None,
        indicators: Optional[Dict] = None,
        setup_label: Optional[str] = None,
        signal_id: int = 0,
        delay_seconds: int = 0,
    ):
        alert_embed = self._build_alert_embed(
            signal, breakdown, llm_response,
            setup_label=setup_label,
            signal_id=signal_id,
            delay_seconds=delay_seconds,
        )
        quant_embed = None
        if (
            composite is not None
            and deriv is not None
            and indicators is not None
        ):
            quant_embed = self._build_quant_embed(
                signal.symbol, composite, breakdown, deriv, indicators,
            )
        await self._dispatch_embeds(
            channel="live",
            alert_embed=alert_embed,
            quant_embed=quant_embed,
            file_path=chart_path,
        )

    async def send_quant_analysis(
        self,
        symbol: str,
        composite,
        breakdown,
        deriv,
        indicators: Dict,
        chart_path: Optional[str] = None,
    ):
        quant_embed = self._build_quant_embed(
            symbol, composite, breakdown, deriv, indicators,
        )
        if not quant_embed:
            return
        if chart_path:
            quant_embed["image"] = {
                "url": "attachment://" + os.path.basename(chart_path)
            }
        await self._dispatch_embeds(
            channel="quant",
            alert_embed=None,
            quant_embed=quant_embed,
            file_path=chart_path,
        )

    async def send_multi_setup_alert(
        self,
        symbol: str,
        setups: list,
        breakdown,
        llm_response: Dict,
        chart_path: Optional[str] = None,
        composite=None,
        deriv=None,
        indicators: Optional[Dict] = None,
        agreement=None,
        smc=None,
        fibo=None,
        heatmap=None,
        signal_id: int = 0,
        delay_seconds: int = 0,
    ):
        """
        Combined message for 1–2 setups. Each setup field shows
        entry zone, SL, and all three TPs.
        """
        if not setups:
            return

        labels = [s.setup_type.upper() for s in setups]
        title_label = " + ".join(labels)

        first_dir = getattr(setups[0], "direction", "neutral")
        if first_dir == "long":
            color = 0x3BA55D
        elif first_dir == "short":
            color = 0xED4245
        else:
            color = 0x5865F2

        fields = []

        # ---- Per-setup field: entry, SL, TP1/TP2/TP3 ----
        for s in setups:
            direction = getattr(s, "direction", "neutral")
            entry_low = getattr(s, "entry_zone_low", 0.0)
            entry_high = getattr(s, "entry_zone_high", 0.0)
            entry_mid = (entry_low + entry_high) / 2
            atr = getattr(s, "atr", 0.0)
            sl = getattr(s, "sl", 0.0)
            tp1 = getattr(s, "tp1", 0.0)
            tp2 = getattr(s, "tp2", 0.0)
            tp3 = getattr(s, "tp3", 0.0)

            def _pct(target: float, entry: float) -> str:
                if entry <= 0 or target <= 0:
                    return "—"
                p = (target - entry) / entry * 100
                if direction == "short":
                    p = -p
                return f"{p:+.2f}%"

            lines = [
                f"Direction: **{direction.upper()}** · "
                f"Conf **{s.confidence:.0%}**",
                f"Entry zone: `{entry_low:.6f}` – `{entry_high:.6f}` "
                f"(mid `{entry_mid:.6f}`)",
            ]

            if sl > 0:
                lines.append(
                    f"🛑 **SL**: `{sl:.6f}` "
                    f"({_pct(sl, entry_mid)})"
                )
            else:
                lines.append("🛑 **SL**: —")

            if tp1 > 0:
                lines.append(
                    f"🎯 **TP1**: `{tp1:.6f}` "
                    f"({_pct(tp1, entry_mid)}) · 40%"
                )
            if tp2 > 0:
                lines.append(
                    f"🎯 **TP2**: `{tp2:.6f}` "
                    f"({_pct(tp2, entry_mid)}) · 35%"
                )
            if tp3 > 0:
                lines.append(
                    f"🎯 **TP3**: `{tp3:.6f}` "
                    f"({_pct(tp3, entry_mid)}) · 25%"
                )

            lines.append(f"ATR: `{atr:.6f}` · {s.reason}")

            fields.append({
                "name": f"📐 {s.setup_type.upper()} ({s.timeframe})",
                "value": "\n".join(lines),
                "inline": False,
            })

        # ---- Agreement ----
        if agreement is not None:
            details_str = ", ".join(
                f"{k}={v}" for k, v in agreement.details.items()
                if v != "neutral"
            ) or "all neutral"
            fields.append({
                "name": "🔒 Agreement",
                "value": (
                    f"Direction: **{agreement.direction.upper()}**\n"
                    f"Signals: {agreement.agreement_count} agree "
                    f"({details_str})"
                ),
                "inline": False,
            })

        # ---- Fib + heatmap + SMC ----
        if fibo is not None and getattr(fibo, "valid", False):
            fields.append({
                "name": "📊 Fibonacci",
                "value": (
                    f"Zone: {fibo.zone_name}\n"
                    f"Range: `{fibo.entry_zone_low:.6f}` – "
                    f"`{fibo.entry_zone_high:.6f}`\n"
                    f"In zone: {'✅' if fibo.in_zone else '❌'}"
                ),
                "inline": True,
            })

        if heatmap is not None and getattr(heatmap, "valid", False):
            fields.append({
                "name": "🔥 Heatmap",
                "value": (
                    f"Imbalance: **{heatmap.imbalance:+.2f}**\n"
                    f"Direction: {heatmap.direction}\n"
                    f"Strength: "
                    f"{'strong' if getattr(heatmap, 'strong', False) else 'mild'}"
                ),
                "inline": True,
            })

        if smc is not None:
            fields.append({
                "name": "🧱 SMC",
                "value": (
                    f"BOS: {getattr(smc, 'bos', 'none')}\n"
                    f"CHoCH: {getattr(smc, 'choch', 'none')}\n"
                    f"Sweep: {getattr(smc, 'liquidity_sweep', 'none')}"
                ),
                "inline": True,
            })

        # ---- Queue badge ----
        if signal_id > 0:
            if delay_seconds > 0:
                fields.append({
                    "name": "⏱ Queue",
                    "value": (
                        f"Signal `#{signal_id}` · auto-trade ON\n"
                        f"Executes in **{delay_seconds}s**"
                    ),
                    "inline": False,
                })
            else:
                fields.append({
                    "name": "⏱ Queue",
                    "value": (
                        f"Signal `#{signal_id}` · observation mode"
                    ),
                    "inline": False,
                })

        # ---- Confidence ----
        if breakdown is not None:
            fields.append({
                "name": "📊 Confidence",
                "value": (
                    f"Raw: **{breakdown.raw_total:.1f}**/100\n"
                    f"Calibrated: "
                    f"**{breakdown.calibrated_confidence:.1%}**"
                ),
                "inline": True,
            })

        embed = {
            "title": (
                f"{'🟢' if first_dir == 'long' else '🔴' if first_dir == 'short' else '⚪'} "
                f"{title_label} SETUP · {symbol}"
            ),
            "color": color,
            "fields": fields,
            "footer": {
                "text": (
                    "Bybit Quant v9.2 · "
                    f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        if chart_path:
            embed["image"] = {
                "url": "attachment://" + os.path.basename(chart_path)
            }

        await self._dispatch_embeds(
            channel="live",
            alert_embed=embed,
            quant_embed=None,
            file_path=chart_path,
        )

    async def send_trade_executed(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        tps: list,
        qty: float,
        signal_id: int,
    ):
        color = 0x3BA55D if direction == "long" else 0xED4245
        tp_lines = "\n".join(
            f"TP{i+1}: `${tp:.6f}`"
            for i, tp in enumerate(tps)
        )
        embed = {
            "title": (
                f"{'🟢' if direction == 'long' else '🔴'} "
                f"TRADE EXECUTED · {symbol}"
            ),
            "color": color,
            "fields": [
                {"name": "Direction",
                 "value": direction.upper(), "inline": True},
                {"name": "Qty", "value": f"`{qty}`", "inline": True},
                {"name": "Signal ID",
                 "value": f"`#{signal_id}`", "inline": True},
                {"name": "Entry",
                 "value": f"`{entry:.6f}`", "inline": True},
                {"name": "Stop Loss",
                 "value": f"`{stop_loss:.6f}`", "inline": True},
                {"name": "🎯 TPs",
                 "value": tp_lines or "—", "inline": False},
            ],
            "footer": {
                "text": (
                    "Bybit Quant v9.2 · "
                    f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self._dispatch_simple("live", embed, None)

    async def send_position_closed(
        self,
        symbol: str,
        side: str,
        entry: float,
        exit_price: float,
        pnl_pct: float,
        pnl_usd: float,
        filled_tps: List[float],
        total_tps: int,
        exit_reason: str,
        chart_path: Optional[str] = None,
    ):
        color = 0x3BA55D if pnl_pct >= 0 else 0xED4245
        emoji = {
            "tp_all": "🎯",
            "sl_hit": "🛑",
            "emergency_sl": "🚨",
            "max_hold": "⏰",
            "manual": "👤",
        }.get(exit_reason, "❓")

        embed = {
            "title": f"{emoji} POSITION CLOSED · {symbol}",
            "color": color,
            "fields": [
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Entry",
                 "value": f"`{entry:.6f}`", "inline": True},
                {"name": "Exit",
                 "value": f"`{exit_price:.6f}`", "inline": True},
                {"name": "PnL %",
                 "value": f"**{pnl_pct:+.2f}%**", "inline": True},
                {"name": "PnL USDT",
                 "value": f"**{pnl_usd:+.4f}**", "inline": True},
                {"name": "TPs Hit",
                 "value": f"{len(filled_tps)}/{total_tps}",
                 "inline": True},
                {"name": "Exit Reason",
                 "value": exit_reason, "inline": False},
            ],
            "footer": {"text": "Bybit Quant v9.2"},
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self._dispatch_simple("alerts", embed, chart_path)

    async def send_emergency_alert(
        self,
        symbol: str,
        reason: str,
        detail: str = "",
        chart_path: Optional[str] = None,
    ):
        embed = {
            "title": f"🚨 EMERGENCY · {symbol}",
            "color": 0xFF0000,
            "description": reason,
            "fields": [
                {"name": "Detail", "value": detail or "—", "inline": False}
            ],
            "footer": {
                "text": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            },
        }
        await self._dispatch_simple("alerts", embed, chart_path)

    async def send_tuning_report(self, results: List):
        if not results:
            return
        lines = [
            f"**{r.parameter}**: `{r.old_value:.3f}` → `{r.new_value:.3f}` "
            f"({r.reason}, n={r.sample_size})"
            for r in results
        ]
        embed = {
            "title": "🔧 Daily Auto-Tuning Report",
            "color": 0x3498DB,
            "description": "\n".join(lines),
            "footer": {"text": datetime.utcnow().isoformat()},
        }
        await self._dispatch_simple("dashboard", embed, None)