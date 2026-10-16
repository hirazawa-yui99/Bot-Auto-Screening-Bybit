"""
notification/chart_generator.py
Render price + indicator charts for Discord attachment.
"""
import logging
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class ChartGenerator:
    def __init__(self, output_dir: str = "/tmp"):
        self.output_dir = output_dir

    def render_signal_chart(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal_info: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Render a signal chart with EMA/RSI overlay."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.patch.set_facecolor("#1a1a1a")
        for ax in axes:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#444")

        # Price + EMAs
        ax = axes[0]
        ax.plot(df["close"], color="#00d4ff", linewidth=1.5, label="Close")
        ax.plot(
            df["close"].ewm(span=9).mean(),
            color="#ffcc00", linewidth=1, label="EMA9",
        )
        ax.plot(
            df["close"].ewm(span=21).mean(),
            color="#ff6600", linewidth=1, label="EMA21",
        )
        ax.plot(
            df["close"].ewm(span=50).mean(),
            color="#ff0066", linewidth=1, label="EMA50",
        )
        if signal_info:
            ax.axhline(
                signal_info.get("entry_price", 0),
                color="white", linestyle="--", alpha=0.6, label="Entry",
            )
            ax.axhline(
                signal_info.get("stop_loss", 0),
                color="red", linestyle="--", alpha=0.6, label="SL",
            )
            ax.axhline(
                signal_info.get("take_profit_1", 0),
                color="green", linestyle="--", alpha=0.6, label="TP",
            )
        ax.set_title(f"{symbol} — Price Action", color="white")
        ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
        ax.grid(alpha=0.2)

        # Volume
        ax = axes[1]
        ax.bar(range(len(df)), df["volume"], color="#555", width=0.8)
        ax.set_title("Volume", color="white")
        ax.grid(alpha=0.2)

        # RSI
        ax = axes[2]
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        ax.plot(rsi, color="#aaff00", linewidth=1.5)
        ax.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax.set_title("RSI(14)", color="white")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        out = f"{self.output_dir}/{filename or symbol}_signal.png"
        plt.savefig(out, facecolor=fig.get_facecolor(), dpi=100)
        plt.close(fig)
        return out