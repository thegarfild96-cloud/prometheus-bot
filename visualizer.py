"""
visualizer.py — графики с RSI и MACD панелями.
"""
import os, logging
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

COLORS = {
    "price":   "#2196F3",
    "forecast":"#FF9800",
    "anomaly": "#F4433620",
    "grid":    "#2D2D2D",
    "bg":      "#0D1117",
    "card":    "#161B22",
    "text":    "#C9D1D9",
    "green":   "#39D353",
    "red":     "#F78166",
    "yellow":  "#E3B341",
    "macd":    "#58A6FF",
    "signal":  "#FF9800",
    "hist_pos":"#39D353",
    "hist_neg":"#F78166",
}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_forecast_chart(symbol, df, predicted, ci_low, ci_high,
                         horizon_h, anomaly_df=None, cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs(cfg["paths"]["charts"], exist_ok=True)

    plot_df = df.tail(168).copy()
    plot_df["datetime"] = pd.to_datetime(plot_df["open_time"], unit="ms")

    # Считаем RSI
    delta = plot_df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    plot_df["rsi"] = 100 - (100 / (1 + rs))

    # Считаем MACD
    ema12 = plot_df["close"].ewm(span=12).mean()
    ema26 = plot_df["close"].ewm(span=26).mean()
    plot_df["macd"]   = ema12 - ema26
    plot_df["signal"] = plot_df["macd"].ewm(span=9).mean()
    plot_df["hist"]   = plot_df["macd"] - plot_df["signal"]

    last_dt   = plot_df["datetime"].iloc[-1]
    target_dt = last_dt + timedelta(hours=horizon_h)

    # 3 панели: цена, RSI, MACD
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9),
        facecolor=COLORS["bg"],
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.08}
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(COLORS["card"])
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.grid(color=COLORS["grid"], linewidth=0.4, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#30363D")

    # ── Панель 1: Цена ──────────────────────────────────────────
    ax1.plot(plot_df["datetime"], plot_df["close"],
             color=COLORS["price"], linewidth=1.5, label="Цена")
    ax1.fill_between(plot_df["datetime"], plot_df["close"],
                     alpha=0.05, color=COLORS["price"])

    # Точка прогноза
    ax1.scatter(target_dt, predicted, color=COLORS["forecast"], s=120, zorder=5)
    ax1.plot([last_dt, target_dt],
             [plot_df["close"].iloc[-1], predicted],
             color=COLORS["forecast"], linewidth=1.5, linestyle="--", alpha=0.8)
    ax1.vlines(target_dt, ci_low, ci_high,
               colors=COLORS["forecast"], linewidth=6, alpha=0.25)
    ax1.annotate(f"${predicted:,.0f}",
                 xy=(target_dt, predicted),
                 xytext=(8, 0), textcoords="offset points",
                 fontsize=10, fontweight="bold", color=COLORS["forecast"])

    # Аномалии
    if anomaly_df is not None and not anomaly_df.empty:
        for _, row in anomaly_df.iterrows():
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(row["datetime"] - timedelta(minutes=30),
                           row["datetime"] + timedelta(minutes=30),
                           color=COLORS["anomaly"], zorder=0)

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title(f"Prometheus — {symbol}  |  Прогноз +{horizon_h}ч",
                  fontsize=12, fontweight="bold",
                  color=COLORS["text"], pad=8)
    ax1.legend(loc="upper left", fontsize=8,
               facecolor=COLORS["card"], labelcolor=COLORS["text"])
    ax1.set_xticklabels([])

    # ── Панель 2: RSI ────────────────────────────────────────────
    ax2.plot(plot_df["datetime"], plot_df["rsi"],
             color=COLORS["yellow"], linewidth=1.2)
    ax2.axhline(70, color=COLORS["red"],   linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(30, color=COLORS["green"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.fill_between(plot_df["datetime"], plot_df["rsi"], 70,
                     where=plot_df["rsi"] >= 70,
                     color=COLORS["red"], alpha=0.15)
    ax2.fill_between(plot_df["datetime"], plot_df["rsi"], 30,
                     where=plot_df["rsi"] <= 30,
                     color=COLORS["green"], alpha=0.15)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color=COLORS["text"], fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Текущее значение RSI
    rsi_now = plot_df["rsi"].iloc[-1]
    rsi_color = COLORS["red"] if rsi_now > 70 else (
                COLORS["green"] if rsi_now < 30 else COLORS["yellow"])
    ax2.annotate(f"RSI: {rsi_now:.1f}",
                 xy=(0.02, 0.8), xycoords="axes fraction",
                 fontsize=9, color=rsi_color, fontweight="bold")
    ax2.set_xticklabels([])

    # ── Панель 3: MACD ────────────────────────────────────────────
    ax3.plot(plot_df["datetime"], plot_df["macd"],
             color=COLORS["macd"], linewidth=1.2, label="MACD")
    ax3.plot(plot_df["datetime"], plot_df["signal"],
             color=COLORS["signal"], linewidth=1.0, label="Signal")
    colors_hist = [COLORS["hist_pos"] if v >= 0 else COLORS["hist_neg"]
                   for v in plot_df["hist"]]
    ax3.bar(plot_df["datetime"], plot_df["hist"],
            color=colors_hist, alpha=0.6, width=0.03)
    ax3.axhline(0, color="#30363D", linewidth=0.8)
    ax3.set_ylabel("MACD", color=COLORS["text"], fontsize=8)
    ax3.legend(loc="upper left", fontsize=7,
               facecolor=COLORS["card"], labelcolor=COLORS["text"])

    # MACD текущий
    macd_now = plot_df["macd"].iloc[-1]
    sig_now  = plot_df["signal"].iloc[-1]
    macd_col = COLORS["green"] if macd_now > sig_now else COLORS["red"]
    ax3.annotate(f"MACD: {macd_now:.1f}",
                 xy=(0.02, 0.8), xycoords="axes fraction",
                 fontsize=9, color=macd_col, fontweight="bold")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=24))

    fig.tight_layout()
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(cfg["paths"]["charts"],
                       f"{symbol}_{horizon_h}h_{ts}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return out


def build_history_chart(symbol, df, cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    os.makedirs(cfg["paths"]["charts"], exist_ok=True)

    plot_df = df.tail(720).copy()
    plot_df["datetime"] = pd.to_datetime(plot_df["open_time"], unit="ms")

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["card"])
    ax.plot(plot_df["datetime"], plot_df["close"],
            color=COLORS["price"], linewidth=1.2)
    ax.fill_between(plot_df["datetime"], plot_df["close"],
                    alpha=0.08, color=COLORS["price"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    ax.set_title(f"{symbol} — последние 30 дней",
                 fontsize=12, fontweight="bold", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color("#30363D")
    fig.tight_layout()

    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(cfg["paths"]["charts"],
                       f"{symbol}_history_{ts}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return out
