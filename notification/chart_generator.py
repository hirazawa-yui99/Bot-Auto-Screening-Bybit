"""
notification/chart_generator.py
Render candlestick + quant metrics charts for Discord.

Layout:
  - Left column (3 rows): price + EMAs, volume, RSI
  - Right column (full height): quant metrics text panel

Output: PNG file path, deleted by caller after upload.
"""
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHART_DIR = "charts"

# Discord dark theme palette
BG = "#1e2124"
PANEL_BG = "#2c2f33"
GRID = "#3a3d42"
TEXT = "#dcddde"
TEXT_DIM = "#b9bbbe"
GREEN = "#26a69a"
RED = "#ef5350"
EMA9_C = "#faa61a"
EMA21_C = "#5865f2"
EMA50_C = "#eb459e"
ENTRY_C = "#ffffff"
SL_C = "#ed4245"
TP_C = "#3ba55d"
HEADER = "#faa61a"


class ChartGenerator:
    def __init__(self, output_dir: str = CHART_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._warmup()

    def _warmup(self):
        """Pre-load matplotlib fonts so first real chart is fast."""
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot([1, 2], [1, 2])
            path = os.path.join(self.output_dir, "_warmup.png")
            fig.savefig(path)
            plt.close(fig)
            os.unlink(path)
        except Exception as e:
            logger.debug(f"Matplotlib warmup: {e}")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    async def render_signal_chart(
        self, df: pd.DataFrame, context: Dict,
    ) -> Optional[str]:
        """Async wrapper — runs matplotlib in thread pool."""
        try:
            path = await asyncio.to_thread(
                self._render_sync, df, context,
            )
            return path
        except Exception as e:
            logger.error(
                f"Chart generation failed for "
                f"{context.get('symbol', '?')}: {e}",
                exc_info=True,
            )
            return None

    def cleanup_old_charts(self, max_age_seconds: int = 86_400):
        """Delete charts older than max_age_seconds."""
        try:
            now = time.time()
            for name in os.listdir(self.output_dir):
                if not name.endswith(".png"):
                    continue
                path = os.path.join(self.output_dir, name)
                try:
                    if now - os.path.getmtime(path) > max_age_seconds:
                        os.unlink(path)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Chart cleanup: {e}")

    # ------------------------------------------------------------------ #
    #  Sync render
    # ------------------------------------------------------------------ #
    def _render_sync(self, df: pd.DataFrame, ctx: Dict) -> str:
        symbol = ctx.get("symbol", "?")
        ts_sec = int(time.time())
        path = os.path.join(self.output_dir, f"{symbol}_{ts_sec}.png")

        if df is None or len(df) < 20:
            raise ValueError(f"Insufficient data for {symbol}")

        df = df.copy().reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        n = len(df)
        if n < 20:
            raise ValueError(f"Too few candles: {n}")

        # EMAs
        ema9 = df["close"].ewm(span=9).mean()
        ema21 = df["close"].ewm(span=21).mean()
        ema50 = df["close"].ewm(span=50).mean()

        # RSI(14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))

        # Figure
        fig = plt.figure(figsize=(16, 10), facecolor=BG, dpi=100)
        gs = fig.add_gridspec(
            3, 2,
            width_ratios=[5, 1],
            height_ratios=[6, 1.5, 2.5],
            hspace=0.05, wspace=0.02,
        )
        ax_p = fig.add_subplot(gs[0, 0])
        ax_v = fig.add_subplot(gs[1, 0], sharex=ax_p)
        ax_r = fig.add_subplot(gs[2, 0], sharex=ax_p)
        ax_m = fig.add_subplot(gs[:, 1])

        for ax in (ax_p, ax_v, ax_r):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.grid(True, color=GRID, alpha=0.3, linewidth=0.5)
        ax_m.set_facecolor(PANEL_BG)
        ax_m.axis("off")

        # Candles
        for i in range(n):
            o = df["open"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            c = df["close"].iloc[i]
            color = GREEN if c >= o else RED
            ax_p.vlines(i, l, h, color=color, linewidth=0.6, alpha=0.85)
            body_lo = min(o, c)
            body_h = max(abs(c - o), (h - l) * 0.01)
            ax_p.add_patch(Rectangle(
                (i - 0.32, body_lo), 0.64, body_h,
                facecolor=color, edgecolor=color, linewidth=0.4,
            ))

        # EMAs
        ax_p.plot(ema9.values, color=EMA9_C, linewidth=0.9, label="EMA9", alpha=0.85)
        ax_p.plot(ema21.values, color=EMA21_C, linewidth=0.9, label="EMA21", alpha=0.85)
        ax_p.plot(ema50.values, color=EMA50_C, linewidth=0.9, label="EMA50", alpha=0.85)

        # Entry / SL / TPs
        si = ctx.get("signal_info") or {}
        entry = si.get("entry")
        sl = si.get("sl")
        tps = si.get("tps") or []
        direction = si.get("direction", "")

        if entry:
            ax_p.axhline(entry, color=ENTRY_C, linestyle="--", linewidth=1.2, alpha=0.9)
            ax_p.text(n - 1, entry, f"  Entry {entry:.4f}", color=ENTRY_C,
                     fontsize=8, va="center", ha="left", weight="bold")

        if sl:
            ax_p.axhline(sl, color=SL_C, linestyle="--", linewidth=1.2, alpha=0.9)
            pct = (sl - entry) / entry * 100 if entry else 0
            ax_p.text(n - 1, sl, f"  SL {sl:.4f} ({pct:+.2f}%)", color=SL_C,
                     fontsize=8, va="center", ha="left", weight="bold")

        for i, tp in enumerate(tps):
            ax_p.axhline(tp, color=TP_C, linestyle="--", linewidth=1.0, alpha=0.8)
            pct = (tp - entry) / entry * 100 if entry else 0
            ax_p.text(n - 1, tp, f"  TP{i+1} {tp:.4f} ({pct:+.2f}%)", color=TP_C,
                     fontsize=7, va="center", ha="left")

        title = (
            f"{symbol}  |  "
            f"{direction.upper() if direction else 'ANALYSIS'}  |  "
            f"{ctx.get('price', 0):.4f}"
        )
        ax_p.set_title(title, color=TEXT, fontsize=12, weight="bold", loc="left", pad=8)
        ax_p.legend(loc="upper left", facecolor=PANEL_BG, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=8, framealpha=0.8)
        ax_p.set_ylabel("Price", color=TEXT_DIM, fontsize=9)

        # Volume
        vol_colors = [
            GREEN if df["close"].iloc[i] >= df["open"].iloc[i] else RED
            for i in range(n)
        ]
        ax_v.bar(range(n), df["volume"].values, color=vol_colors, width=0.7, alpha=0.7)
        ax_v.set_ylabel("Vol", color=TEXT_DIM, fontsize=9)

        # RSI
        ax_r.plot(rsi.values, color="#aaff00", linewidth=1.3)
        ax_r.axhline(70, color=RED, linestyle="--", linewidth=0.8, alpha=0.6)
        ax_r.axhline(30, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
        ax_r.axhline(50, color=GRID, linestyle=":", linewidth=0.6, alpha=0.5)
        ax_r.fill_between(range(n), 70, 100, color=RED, alpha=0.08)
        ax_r.fill_between(range(n), 0, 30, color=GREEN, alpha=0.08)
        ax_r.set_ylim(0, 100)
        ax_r.set_ylabel("RSI", color=TEXT_DIM, fontsize=9)

        # X-axis labels
        ticks = list(range(0, n, max(1, n // 8)))
        labels = []
        for i in ticks:
            ts = df["ts"].iloc[i]
            try:
                dt = pd.to_datetime(int(ts), unit="ms", utc=True)
                labels.append(dt.strftime("%m-%d %H:%M"))
            except Exception:
                labels.append(str(i))
        ax_r.set_xticks(ticks)
        ax_r.set_xticklabels(labels, rotation=0, color=TEXT_DIM, fontsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_v.tick_params(labelbottom=False)

        # Metrics panel
        self._render_metrics(ax_m, ctx)

        fig.text(
            0.01, 0.005,
            f"Bybit Quant Bot v9.1  |  "
            f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            color=TEXT_DIM, fontsize=7, alpha=0.7,
        )

        plt.savefig(
            path, facecolor=fig.get_facecolor(),
            dpi=100, bbox_inches="tight", pad_inches=0.15,
        )
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    #  Metrics panel
    # ------------------------------------------------------------------ #
    def _render_metrics(self, ax, ctx: Dict):
        composite = ctx.get("composite")
        breakdown = ctx.get("breakdown")
        deriv = ctx.get("deriv")
        indicators = ctx.get("indicators") or {}

        y = 0.98
        lh = 0.0225

        def header(text):
            nonlocal y
            ax.text(0.05, y, text, color=HEADER, fontsize=8.5,
                    weight="bold", family="monospace",
                    transform=ax.transAxes)
            y -= lh * 1.05

        def row(label, value, color=TEXT):
            nonlocal y
            ax.text(0.05, y, label, color=TEXT_DIM, fontsize=7.5,
                    family="monospace", transform=ax.transAxes)
            ax.text(0.95, y, str(value), color=color, fontsize=7.5,
                    weight="bold", family="monospace",
                    ha="right", transform=ax.transAxes)
            y -= lh

        def blank():
            nonlocal y
            y -= lh * 0.5

        symbol = ctx.get("symbol", "?")
        price = ctx.get("price", 0.0)

        header(f">>> {symbol}")
        row("Price", f"{price:.4f}")
        blank()

        if composite is not None:
            header("COMPOSITE")
            s = composite.total_score
            row("Score", f"{s:+.3f}", GREEN if s > 0 else RED if s < 0 else TEXT)
            d = composite.direction
            row("Direction", d.upper(),
                GREEN if d == "long" else RED if d == "short" else TEXT)
            row("Conviction", composite.conviction.upper())
            row("Agreement", f"{composite.agreement_ratio * 100:.0f}%")
            blank()

            mr = composite.mean_reversion
            header("MEAN REVERSION")
            row("Zeta", f"{mr.zeta:+.2f}")
            row("Hurst", f"{mr.hurst:.3f}")
            row("Regime", mr.regime)
            row("ADF p", f"{mr.adf_pvalue:.3f}")
            blank()

        if breakdown is not None:
            header("CONFIDENCE")
            row("Raw", f"{breakdown.raw_total:.1f}/100")
            row("Calibrated", f"{breakdown.calibrated_confidence:.1%}")
            row("Tech", f"{breakdown.technical_score:.1f}")
            row("Comp", f"{breakdown.composite_score:.1f}")
            row("Deriv", f"{breakdown.derivative_score:.1f}")
            row("Sent", f"{breakdown.sentiment_score:.1f}")
            row("Pattern", f"{breakdown.pattern_score:.1f}")
            row("LLM", f"{breakdown.llm_score:.1f}")
            blank()

        if deriv is not None:
            header("DERIVATIVES")
            cvd_d = deriv.cvd_divergence
            row("CVD div", cvd_d,
                GREEN if cvd_d == "bullish" else RED if cvd_d == "bearish" else TEXT_DIM)
            row("OI 1h", f"{deriv.oi_change_1h:+.2f}%")
            row("OI 4h", f"{deriv.oi_change_4h:+.2f}%")
            row("OI div", deriv.oi_divergence)
            row("Top L/S", f"{deriv.top_trader_ls_ratio:.2f}")
            row("Smart", deriv.smart_money_bias)
            blank()

        if composite is not None:
            funding = composite.funding
            header("FUNDING")
            fr = funding.rate_pct
            row("Rate", f"{fr:+.4f}%",
                RED if fr > 0.05 else GREEN if fr < -0.05 else TEXT)
            row("Annual", f"{funding.annualized_pct:+.1f}%")
            row("Regime", funding.regime)
            if funding.should_block_long:
                row("BLOCK", "LONG", RED)
            if funding.should_block_short:
                row("BLOCK", "SHORT", GREEN)
            blank()

            vol = composite.volume
            header("VOLUME")
            row("RVOL", f"{vol.rvol:.2f}")
            row("Z-score", f"{vol.z_score:+.2f}")
            row("Level", vol.anomaly_level)
            row("VWPD", f"{vol.vwpd:+.2f}")
            blank()

            basis = composite.basis
            header("BASIS")
            row("Basis %", f"{basis.basis_pct:+.4f}%")
            row("Regime", basis.regime)
            if basis.signal_bias != "neutral":
                row("Bias", basis.signal_bias)