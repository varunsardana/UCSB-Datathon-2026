"""
Model Comparison: Baseline (Trend+Seasonality) vs Prophet vs Negative Binomial GLM

Runs all 3 models on every state × disaster type combination using the same
train/test split. Selects the best model per combo by lowest test-set MAE.

Outputs
-------
disaster_forecast/model_comparison_results.csv   — side-by-side MAE/RMSE table
disaster_forecast/best_model_forecasts.json      — 72-month forecast from best model
disaster_forecast/rag_optimized_profiles.json    — narrative text for ChromaDB ingestion
disaster_forecast/plots/<key>.png                — per-combo time series plot
disaster_forecast/plots/model_comparison_summary.png  — grouped bar chart

Run from repo root:
    python disaster_forecast/model_comparison.py
"""

import json
import logging
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # non-interactive — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from prophet import Prophet

# ── Constants ─────────────────────────────────────────────────────────────────

START_DATE       = "2000-01-01"
END_DATE         = "2026-02-01"
FORECAST_HORIZON = 72
N_TEST           = 12

MAJOR_DISASTERS = [
    "Hurricane", "Severe Storm", "Flood", "Fire",
    "Tornado", "Snowstorm", "Severe Ice Storm",
]

COLORS = {
    "baseline": "#2196F3",   # blue
    "prophet":  "#FF9800",   # orange
    "negbin":   "#4CAF50",   # green
    "history":  "#212121",   # near-black
    "test_bg":  "#F5F5F5",   # light gray
}


# ── Shared utility ────────────────────────────────────────────────────────────

def create_complete_series(df_subset):
    """Fill every month from START_DATE → END_DATE; missing months → 0."""
    dr       = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    complete = pd.DataFrame({"date": dr})
    complete = complete.merge(df_subset[["date", "disaster_count"]], on="date", how="left")
    complete["disaster_count"] = complete["disaster_count"].fillna(0).astype(int)
    return complete


def evaluate(y_true, y_pred):
    mae  = round(float(np.mean(np.abs(y_true - y_pred))), 3)
    rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3)
    return mae, rmse


# ── Model A: Baseline — replicates proper_timeseries_v2.py logic ─────────────

def _calc_trend(y):
    x = np.arange(len(y))
    if len(y) < 3:
        return 0.0, float(np.mean(y))
    slope, intercept, *_ = stats.linregress(x, y)
    return float(slope), float(intercept)


def _calc_seasonality(y, period=12):
    if len(y) < period * 2:
        return np.ones(period)
    n   = (len(y) // period) * period
    mat = y[:n].reshape(-1, period)
    avg = mat.mean(axis=0)
    mu  = y[:n].mean()
    return avg / mu if mu > 0 else np.ones(period)


def baseline_forecast(y_train, horizon):
    """Trend × Seasonality decomposition — Aliza's existing method."""
    if len(y_train) < 24:
        avg = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
        return np.full(horizon, avg), 0.0

    slope, intercept = _calc_trend(y_train)
    seas = _calc_seasonality(y_train)
    x    = np.arange(len(y_train))
    trend_line = slope * x + intercept
    baseline   = float(np.mean(y_train - trend_line))

    preds = []
    for i in range(horizon):
        tv  = slope * (len(y_train) + i) + intercept
        sf  = seas[(len(y_train) + i) % 12]
        val = (tv if tv > 0 else baseline) * sf
        preds.append(max(0.0, val))

    residuals = y_train - (trend_line + baseline * seas[x % 12])
    std_err   = float(np.std(residuals))
    return np.array(preds), std_err


# ── Model B: Negative Binomial GLM ───────────────────────────────────────────

def _glm_features(start_idx, n):
    t  = np.arange(start_idx, start_idx + n)
    mi = t % 12
    X  = np.zeros((n, 13))
    X[:, 0] = 1
    X[:, 1] = t / 100.0
    for m in range(1, 12):
        X[:, m + 1] = (mi == m).astype(float)
    return X


def negbin_forecast(y_train, n_train, n_test, horizon):
    if y_train.sum() == 0:
        return None, None, None, None, False
    try:
        model = sm.GLM(
            y_train, _glm_features(0, n_train),
            family=sm.families.NegativeBinomial(),
        ).fit(maxiter=200, disp=False)

        test_pred = np.maximum(0.0, model.predict(_glm_features(n_train, n_test)))

        ci       = model.get_prediction(_glm_features(n_train + n_test, horizon)).summary_frame(alpha=0.05)
        forecast = np.maximum(0.0, ci["mean"].values)
        lower    = np.maximum(0.0, ci["mean_ci_lower"].values)
        upper    = ci["mean_ci_upper"].values

        return test_pred, forecast, lower, upper, True
    except Exception:
        return None, None, None, None, False


# ── Model C: Prophet ──────────────────────────────────────────────────────────

def prophet_forecast(ts_df, n_train, horizon):
    n_test   = len(ts_df) - n_train
    train_df = (
        ts_df[["date", "disaster_count"]]
        .iloc[:n_train]
        .rename(columns={"date": "ds", "disaster_count": "y"})
        .copy()
    )
    if len(train_df) < 2 or train_df["y"].sum() == 0:
        return None, None, None, None, False
    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.95,
            changepoint_prior_scale=0.05,
        )
        m.fit(train_df)
        future   = m.make_future_dataframe(periods=n_test + horizon, freq="MS")
        forecast = m.predict(future)

        test_pred = forecast.iloc[n_train: n_train + n_test]["yhat"].clip(lower=0).values
        fut_rows  = forecast.iloc[n_train + n_test:]
        return (
            test_pred,
            fut_rows["yhat"].clip(lower=0).values,
            fut_rows["yhat_lower"].clip(lower=0).values,
            fut_rows["yhat_upper"].values,
            True,
        )
    except Exception:
        return None, None, None, None, False


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_combo(key, ts_df, y, n_train, results, future_dates, out_dir):
    """Time series plot showing all 3 model forecasts for one combo."""
    dates      = ts_df["date"].tolist()
    hist_dates = [pd.Timestamp(d) for d in dates]
    fd         = list(future_dates)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Shade test window
    ax.axvspan(
        pd.Timestamp(dates[n_train]), pd.Timestamp(dates[-1]),
        alpha=0.08, color="gray", label=f"Test window ({N_TEST} mo)",
    )

    # Vertical line: forecast start
    ax.axvline(pd.Timestamp(dates[-1]), color="black", lw=0.8, linestyle=":")

    # Historical
    ax.plot(hist_dates, y, color=COLORS["history"], lw=1.5, zorder=5, label="Actual")

    # Each model
    for mname, color in [
        ("baseline", COLORS["baseline"]),
        ("prophet",  COLORS["prophet"]),
        ("negbin",   COLORS["negbin"]),
    ]:
        r = results.get(mname)
        if r is None or r.get("forecast") is None:
            continue
        fc    = np.array(r["forecast"])
        lo    = np.array(r.get("lower", fc))
        hi    = np.array(r.get("upper", fc))
        mae   = r.get("mae")
        label = f"{mname.capitalize()} (MAE={mae:.2f})" if isinstance(mae, float) else mname.capitalize()
        ls    = "-" if mname == results["best_model"] else "--"
        ax.plot(fd, fc, color=color, lw=2.2, linestyle=ls, label=label, alpha=0.9)
        ax.fill_between(fd, lo, hi, color=color, alpha=0.12)

    state, disaster = key.split("_", 1)
    ax.set_title(
        f"{state} — {disaster.replace('_', ' ')}  |  "
        f"Best: {results['best_model'].capitalize()} (MAE={results['best_mae']:.2f})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Disaster declarations / month")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=120)
    plt.close()


def plot_summary(comparison_df, out_dir):
    """Grouped bar chart: MAE per model across all combos."""
    # Exclude combos where test set was all zeros (artificially perfect scores)
    df = comparison_df[
        (comparison_df["test_zeros"] == 0) &
        (comparison_df["baseline_mae"].notna())
    ].sort_values("baseline_mae", ascending=False).head(30)

    if df.empty:
        print("   ⚠ No non-zero test-set combos for summary chart")
        return

    labels = [f"{r.state}·{r.disaster_type[:6]}" for r in df.itertuples()]
    x      = np.arange(len(labels))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.8), 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.bar(x - w, df["baseline_mae"],             w, label="Baseline (Trend+Seasonal)", color=COLORS["baseline"], alpha=0.85)
    ax.bar(x,     df["prophet_mae"].fillna(0),    w, label="Prophet",                  color=COLORS["prophet"],  alpha=0.85)
    ax.bar(x + w, df["negbin_mae"].fillna(0),     w, label="NegBin GLM",               color=COLORS["negbin"],   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAE (disaster declarations / month)")
    ax.set_title(
        "Model Comparison — MAE by State × Disaster Type\n"
        "(lower is better  |  combos with all-zero test sets excluded)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison_summary.png"), dpi=130)
    plt.close()
    print("   ✓ Summary chart saved")


# ── RAG narrative builder ─────────────────────────────────────────────────────

def build_rag_text(state, disaster, results, future_dates, seas):
    """
    Build a human-readable chunk for ChromaDB ingestion.
    Each chunk is self-contained so it retrieves well on its own.
    """
    best     = results["best_model"]
    best_r   = results[best]
    best_mae = results["best_mae"]
    baseline_mae = (results.get("baseline") or {}).get("mae", "N/A")

    total    = results.get("total_historical", "?")
    avg_yr   = round(total / 26, 1) if isinstance(total, (int, float)) else "?"
    trend    = results.get("trend_direction", "stable")

    # Top 3 peak months by seasonal index
    peak_idx    = np.argsort(seas)[-3:][::-1]
    peak_months = [datetime(2000, int(i) + 1, 1).strftime("%B") for i in peak_idx]

    # Monthly risk profile
    risk_lines = []
    for i, avg in enumerate(seas):
        mn    = datetime(2000, i + 1, 1).strftime("%B")
        level = ("HIGH" if avg >= 2.0 else "moderate" if avg >= 1.0 else "low" if avg >= 0.5 else "minimal")
        risk_lines.append(f"  {mn}: {level} (index={avg:.2f})")

    # 12-month outlook
    fc12 = best_r["forecast"][:12]
    fd12 = list(future_dates)[:12]
    fc_lines = ", ".join(f"{fd.strftime('%b %Y')}={v:.1f}" for fd, v in zip(fd12, fc12))

    bm_str = f"{baseline_mae:.2f}" if isinstance(baseline_mae, float) else str(baseline_mae)

    text = (
        f"{state} {disaster.replace('_', ' ')} Disaster Seasonal Risk Profile (2000–2026):\n"
        f"State: {state} | Disaster type: {disaster.replace('_', ' ')}\n"
        f"Historical total: {total} FEMA declarations over 26 years (~{avg_yr}/year). "
        f"Long-term trend: {trend}.\n"
        f"Peak activity months: {', '.join(peak_months)}.\n\n"
        f"Monthly seasonal pattern (relative to annual average):\n"
        + "\n".join(risk_lines) + "\n\n"
        f"12-Month Outlook (best model: {best.capitalize()}, "
        f"MAE={best_mae:.2f} vs baseline MAE={bm_str}):\n"
        f"{fc_lines}\n\n"
        f"Important: these forecasts predict FEMA disaster declaration frequency "
        f"(county-level declarations per month), not direct job loss or economic impact."
    )
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MODEL COMPARISON: Baseline vs Prophet vs NegBin GLM")
    print("=" * 70)

    # 1. Load data
    print("\n1. Loading pre-committed data...")
    monthly_agg = pd.read_csv("disaster_forecast/fema_monthly_aggregated.csv")
    monthly_agg["date"] = pd.to_datetime(monthly_agg["date"])
    state_stats = pd.read_csv("disaster_forecast/fema_state_incident_stats.csv")
    print(f"   ✓ {len(monthly_agg):,} monthly records  |  {len(state_stats):,} combos")

    # 2. Select combos
    print("\n2. Selecting combinations (major disasters, ≥10 total, ≥12 months)...")
    filtered = (
        state_stats[
            state_stats["incidentType"].isin(MAJOR_DISASTERS)
            & (state_stats["total_disasters"] >= 10)
            & (state_stats["months_with_data"] >= 12)
        ]
        .sort_values("total_disasters", ascending=False)
        .head(50)
    )
    print(f"   ✓ {len(filtered)} combinations selected")

    # 3. Prepare output directory
    out_dir = "disaster_forecast/plots"
    os.makedirs(out_dir, exist_ok=True)

    # 4. Run models
    print("\n3. Running all 3 models per combination...")
    print("=" * 70)

    comparison_rows = []
    best_forecasts  = {}
    rag_profiles    = []

    for idx, (_, row) in enumerate(filtered.iterrows(), 1):
        state    = row["state"]
        disaster = row["incidentType"]
        key      = f"{state}_{disaster.replace(' ', '_')}"

        print(f"\n[{idx}/{len(filtered)}] {key}")

        # Build complete time series
        subset = monthly_agg[
            (monthly_agg["state"] == state) & (monthly_agg["incidentType"] == disaster)
        ]
        ts    = create_complete_series(subset)
        y     = ts["disaster_count"].values.astype(float)
        dates = ts["date"].values

        n_total = len(y)
        n_train = n_total - N_TEST

        if n_train < 24:
            print("   ⚠ Skipped (< 24 train months)")
            continue

        y_train = y[:n_train]
        y_test  = y[n_train:]

        future_dates = pd.date_range(
            start=pd.Timestamp(dates[-1]) + pd.DateOffset(months=1),
            periods=FORECAST_HORIZON, freq="MS",
        )

        seas = _calc_seasonality(y_train)
        slope, _ = _calc_trend(y)
        trend_dir = ("increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable")

        results = {
            "total_historical": int(row["total_disasters"]),
            "trend_direction":  trend_dir,
        }

        # ── Model A: Baseline ─────────────────────────────────────────────
        print("   [A] Baseline     ", end="")
        fc_a_full, std_a = baseline_forecast(y_train, N_TEST + FORECAST_HORIZON)
        test_a = fc_a_full[:N_TEST]
        fut_a  = fc_a_full[N_TEST:]
        mae_a, rmse_a = evaluate(y_test, test_a)
        results["baseline"] = {
            "mae": mae_a, "rmse": rmse_a,
            "forecast": fut_a.tolist(),
            "lower":    np.maximum(0, fut_a - 1.96 * std_a).tolist(),
            "upper":    (fut_a + 1.96 * std_a).tolist(),
        }
        print(f"→  MAE={mae_a:.3f}   RMSE={rmse_a:.3f}")

        # ── Model B: NegBin GLM ───────────────────────────────────────────
        print("   [B] NegBin GLM  ", end="")
        tp_b, fc_b, lo_b, hi_b, ok_b = negbin_forecast(y_train, n_train, N_TEST, FORECAST_HORIZON)
        if ok_b:
            mae_b, rmse_b = evaluate(y_test, tp_b)
            results["negbin"] = {
                "mae": mae_b, "rmse": rmse_b,
                "forecast": fc_b.tolist(), "lower": lo_b.tolist(), "upper": hi_b.tolist(),
            }
            print(f"→  MAE={mae_b:.3f}   RMSE={rmse_b:.3f}")
        else:
            results["negbin"] = None
            mae_b = rmse_b = None
            print("→  ⚠ failed to converge")

        # ── Model C: Prophet ──────────────────────────────────────────────
        print("   [C] Prophet     ", end="")
        tp_c, fc_c, lo_c, hi_c, ok_c = prophet_forecast(ts, n_train, FORECAST_HORIZON)
        if ok_c:
            mae_c, rmse_c = evaluate(y_test, tp_c)
            results["prophet"] = {
                "mae": mae_c, "rmse": rmse_c,
                "forecast": fc_c.tolist(), "lower": lo_c.tolist(), "upper": hi_c.tolist(),
            }
            print(f"→  MAE={mae_c:.3f}   RMSE={rmse_c:.3f}")
        else:
            results["prophet"] = None
            mae_c = rmse_c = None
            print("→  ⚠ failed")

        # ── Best model selection ──────────────────────────────────────────
        candidates = {"baseline": mae_a}
        if ok_b: candidates["negbin"]  = mae_b
        if ok_c: candidates["prophet"] = mae_c
        best_model = min(candidates, key=candidates.get)
        best_mae   = candidates[best_model]
        results["best_model"] = best_model
        results["best_mae"]   = best_mae
        print(f"   → Best: {best_model.upper()} (MAE={best_mae:.3f})")

        # ── Per-combo plot ────────────────────────────────────────────────
        try:
            plot_combo(key, ts, y, n_train, results, future_dates, out_dir)
            print(f"   ✓ Plot: plots/{key}.png")
        except Exception as e:
            print(f"   ⚠ Plot failed: {e}")

        # ── Comparison row ────────────────────────────────────────────────
        comparison_rows.append({
            "state":           state,
            "disaster_type":   disaster,
            "total_historical": int(row["total_disasters"]),
            "test_zeros":       int(y_test.sum() == 0),
            "baseline_mae":     mae_a,   "baseline_rmse":  rmse_a,
            "negbin_mae":       mae_b,   "negbin_rmse":    rmse_b,
            "prophet_mae":      mae_c,   "prophet_rmse":   rmse_c,
            "best_model":       best_model,
            "best_mae":         best_mae,
        })

        # ── Best-model forecast JSON ──────────────────────────────────────
        best_r = results[best_model]
        best_forecasts[key] = {
            "state":        state,
            "disaster_type": disaster,
            "best_model":   best_model,
            "model_comparison": {
                "baseline": {"mae": mae_a, "rmse": rmse_a},
                "negbin":   {"mae": mae_b, "rmse": rmse_b} if ok_b else None,
                "prophet":  {"mae": mae_c, "rmse": rmse_c} if ok_c else None,
            },
            "historical": {
                "dates":  [pd.Timestamp(d).strftime("%Y-%m") for d in dates],
                "counts": y.tolist(),
            },
            "forecast": {
                "dates":            [d.strftime("%Y-%m") for d in future_dates],
                "predicted_counts": [round(v, 2) for v in best_r["forecast"]],
                "lower_bound":      [round(v, 2) for v in best_r["lower"]],
                "upper_bound":      [round(v, 2) for v in best_r["upper"]],
            },
        }

        # ── RAG profile ───────────────────────────────────────────────────
        rag_text = build_rag_text(state, disaster, results, future_dates, seas)
        rag_profiles.append({
            "id":           f"forecast_{key}",
            "disaster_type": disaster.lower(),
            "state":        state,
            "source":       "ts_forecast_model",
            "text":         rag_text,
        })

    # 5. Summary plot
    print("\n4. Generating summary bar chart...")
    comparison_df = pd.DataFrame(comparison_rows)
    plot_summary(comparison_df, out_dir)

    # 6. Save outputs
    print("\n5. Saving outputs...")
    comparison_df.to_csv("disaster_forecast/model_comparison_results.csv", index=False)
    print(f"   ✓ model_comparison_results.csv   ({len(comparison_df)} rows)")

    with open("disaster_forecast/best_model_forecasts.json", "w") as f:
        json.dump(best_forecasts, f, indent=2)
    print(f"   ✓ best_model_forecasts.json      ({len(best_forecasts)} combos)")

    with open("disaster_forecast/rag_optimized_profiles.json", "w") as f:
        json.dump(rag_profiles, f, indent=2)
    print(f"   ✓ rag_optimized_profiles.json    ({len(rag_profiles)} profiles)")

    # 7. Final leaderboard
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    valid = comparison_df[comparison_df["test_zeros"] == 0]
    if len(valid) > 0:
        print(f"\nAverage MAE across {len(valid)} non-trivial combos:")
        print(f"   Baseline (Trend+Seasonal) : {valid['baseline_mae'].mean():.3f}")
        if valid["prophet_mae"].notna().any():
            print(f"   Prophet                   : {valid['prophet_mae'].dropna().mean():.3f}")
        if valid["negbin_mae"].notna().any():
            print(f"   NegBin GLM                : {valid['negbin_mae'].dropna().mean():.3f}")

        wins = valid["best_model"].value_counts()
        print(f"\nBest model wins ({len(valid)} combos):")
        for model_name, n in wins.items():
            pct = 100 * n / len(valid)
            print(f"   {model_name.capitalize():<10}: {n:>3} combos  ({pct:.0f}%)")

    trivial = comparison_df[comparison_df["test_zeros"] == 1]
    if len(trivial) > 0:
        print(f"\nNote: {len(trivial)} combos excluded from leaderboard "
              f"(test set had all-zero disaster counts — MAE trivially ≈ 0).")

    print(f"\nPlots  →  disaster_forecast/plots/")
    print(f"To ingest forecasts into RAG: cd backend && python -m rag.ingest")
    print("=" * 70)


if __name__ == "__main__":
    main()
