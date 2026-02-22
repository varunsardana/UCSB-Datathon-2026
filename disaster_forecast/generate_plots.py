"""
generate_plots.py — Time Series Visualization

Reads prophet_state_forecasts.json and produces:
  1. disaster_forecast/plots/sample_grid.png   — 6-panel overview grid
  2. disaster_forecast/plots/<key>.png          — one detailed plot per combo

Run from repo root:
    python disaster_forecast/generate_plots.py
"""

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

HIST_COLOR    = "#1a3a5c"   # dark navy  — historical
FORE_COLOR    = "#e05c1a"   # burnt orange — forecast line
CI_COLOR      = "#e05c1a"   # same, lighter for fill
BOUNDARY_CLR  = "#888888"   # gray dashed vertical line
SHADE_ALPHA   = 0.15

# ── 6 combos chosen for variety across disaster type, geography, seasonality ──
FEATURED = [
    "FL_Hurricane",       # Atlantic hurricane season — classic Sep peak
    "LA_Hurricane",       # Gulf Coast — extreme counts, tight season
    "CA_Fire",            # Wildfire — summer/fall, increasing trend
    "TX_Severe_Storm",    # Tornado alley — spring peak, very low MAE
    "IA_Flood",           # River flooding — spring snowmelt pattern
    "OK_Severe_Storm",    # Tornado alley — strong May peak
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_combo(data: dict, key: str):
    """Parse one combo entry into DataFrames and metadata."""
    entry = data[key]
    info  = entry.get("model_info", {})

    hist_df = pd.DataFrame({
        "date":  pd.to_datetime([d + "-01" for d in entry["historical"]["dates"]]),
        "count": entry["historical"]["counts"],
    })

    fore_df = pd.DataFrame({
        "date":      pd.to_datetime([d + "-01" for d in entry["forecast"]["dates"]]),
        "predicted": entry["forecast"]["predicted_counts"],
        "lower":     entry["forecast"]["lower_bound"],
        "upper":     entry["forecast"]["upper_bound"],
    })

    meta = {
        "state":        entry["state"],
        "disaster":     entry["disaster_type"],
        "total":        info.get("total_historical", "?"),
        "cv_mae":       info.get("cv_mae"),
        "cv_rmse":      info.get("cv_rmse"),
        "peak_months":  info.get("peak_months", []),
    }
    return hist_df, fore_df, meta


# ── Single-panel plot ─────────────────────────────────────────────────────────

def plot_combo(ax, hist_df: pd.DataFrame, fore_df: pd.DataFrame, meta: dict,
               show_xlabel: bool = True, compact: bool = False):
    """Draw historical + forecast onto an existing Axes object."""

    forecast_start = fore_df["date"].iloc[0]

    # ── Historical ────────────────────────────────────────────────────────────
    ax.plot(
        hist_df["date"], hist_df["count"],
        color=HIST_COLOR, linewidth=1.4 if compact else 1.8,
        label="Historical", zorder=3,
    )

    # ── Forecast line ─────────────────────────────────────────────────────────
    ax.plot(
        fore_df["date"], fore_df["predicted"],
        color=FORE_COLOR, linewidth=1.4 if compact else 1.8,
        linestyle="--", label="Prophet forecast", zorder=3,
    )

    # ── 95% CI band ───────────────────────────────────────────────────────────
    ax.fill_between(
        fore_df["date"], fore_df["lower"], fore_df["upper"],
        color=CI_COLOR, alpha=SHADE_ALPHA, label="95% CI", zorder=2,
    )

    # ── Forecast boundary line ────────────────────────────────────────────────
    ax.axvline(forecast_start, color=BOUNDARY_CLR, linewidth=1.0,
               linestyle=":", zorder=4, label="Forecast start (Mar 2026)")

    # ── Title & labels ────────────────────────────────────────────────────────
    peaks_str = " · ".join(meta["peak_months"][:2]) if meta["peak_months"] else "N/A"
    mae_str   = f"CV MAE={meta['cv_mae']:.2f}" if meta["cv_mae"] is not None else ""

    if compact:
        ax.set_title(
            f"{meta['state']} — {meta['disaster']}\n"
            f"Peaks: {peaks_str}   {mae_str}",
            fontsize=9, fontweight="bold", pad=4,
        )
    else:
        ax.set_title(
            f"{meta['state']} — {meta['disaster']}",
            fontsize=13, fontweight="bold", pad=8,
        )
        ax.set_subtitle = lambda *a, **k: None   # placeholder
        ax.annotate(
            f"Peaks: {peaks_str}   |   {mae_str}   |   "
            f"{int(meta['total']):,} total declarations",
            xy=(0.01, 0.97), xycoords="axes fraction",
            fontsize=9, color="#555555", va="top",
        )

    ax.set_ylabel("Declarations / month", fontsize=9 if compact else 10)
    if show_xlabel:
        ax.set_xlabel("Date", fontsize=9 if compact else 10)

    # ── X-axis date formatting ────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.YearLocator(4 if compact else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── Shade forecast region ─────────────────────────────────────────────────
    ax.axvspan(
        forecast_start, fore_df["date"].iloc[-1],
        color="#f5f0e8", alpha=0.5, zorder=1,
    )

    ax.set_xlim(hist_df["date"].iloc[0], fore_df["date"].iloc[-1])
    ax.set_ylim(bottom=0)

    if not compact:
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85)


# ── Full-size individual plot ─────────────────────────────────────────────────

def save_individual_plot(key: str, hist_df, fore_df, meta, out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    plot_combo(ax, hist_df, fore_df, meta, compact=False)

    fig.suptitle(
        f"FEMA Disaster Declaration Frequency — Prophet 72-Month Forecast",
        fontsize=11, color="#444444", y=1.01,
    )
    fig.tight_layout()
    out = out_dir / f"{key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── 6-panel grid ─────────────────────────────────────────────────────────────

def save_grid_plot(keys, data: dict, out_dir: Path):
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        hist_df, fore_df, meta = load_combo(data, key)
        plot_combo(
            axes[i], hist_df, fore_df, meta,
            show_xlabel=(i >= ncols),   # only bottom row
            compact=True,
        )
        if i % ncols != 0:
            axes[i].set_ylabel("")

    fig.suptitle(
        "FEMA Disaster Frequency — Prophet 72-Month Forecast by State & Disaster Type\n"
        "Shaded region = 2026-03 → 2032-02 (forecast period)   |   Dashed = 95% CI",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # Shared legend in bottom-right corner of figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    out = out_dir / "sample_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING TIME SERIES PLOTS")
    print("=" * 60)

    forecast_path = Path("disaster_forecast/prophet_state_forecasts.json")
    if not forecast_path.exists():
        print("ERROR: prophet_state_forecasts.json not found.")
        print("Run: python disaster_forecast/prophet_forecast.py")
        return

    with open(forecast_path) as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} combos from forecast JSON")

    out_dir = Path("disaster_forecast/plots")
    out_dir.mkdir(exist_ok=True)

    # ── 1. Grid overview ──────────────────────────────────────────────────────
    print("\nGenerating 6-panel overview grid...")
    available = [k for k in FEATURED if k in data]
    missing   = [k for k in FEATURED if k not in data]
    if missing:
        print(f"  ⚠  Not in JSON (skipped): {missing}")

    save_grid_plot(available, data, out_dir)

    # ── 2. Individual full-size plots ─────────────────────────────────────────
    print("\nGenerating individual plots...")
    for key in available:
        hist_df, fore_df, meta = load_combo(data, key)
        save_individual_plot(key, hist_df, fore_df, meta, out_dir)

    print(f"\n✓ All plots saved to {out_dir}/")
    print(f"  Files: {[f.name for f in sorted(out_dir.glob('*.png'))]}")


if __name__ == "__main__":
    main()
