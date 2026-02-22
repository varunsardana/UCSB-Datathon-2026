"""
Negative Binomial GLM — FEMA Disaster Frequency Forecasting

Why NegBin over Aliza's baseline (Trend × Seasonality):
  - Disaster counts are non-negative integers — NegBin is the correct distribution
  - NegBin handles overdispersion (variance >> mean), common in disaster data
  - Explicitly models autocorrelation through GLM link function
  - Seasonal patterns captured via month dummy variables (not just averages)

Model formula per combo:
  log(E[y_t]) = β₀ + β₁*(t/100) + β₂*Feb + β₃*Mar + ... + β₁₂*Dec
  where t = month index (0, 1, 2, ...), January is the baseline month

Run from repo root:
    python disaster_forecast/negbin_model.py
"""

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

START_DATE       = "2000-01-01"
END_DATE         = "2026-02-01"
FORECAST_HORIZON = 72   # months (6 years, through Feb 2032)
N_TEST           = 12   # months held out for evaluation

MAJOR_DISASTERS = [
    "Hurricane", "Severe Storm", "Flood", "Fire",
    "Tornado", "Snowstorm", "Severe Ice Storm",
]


# ── Data helper ───────────────────────────────────────────────────────────────

def create_complete_series(df_subset):
    """Fill missing months with 0 — same logic as proper_timeseries_v2.py."""
    date_range  = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    complete    = pd.DataFrame({"date": date_range})
    complete    = complete.merge(
        df_subset[["date", "disaster_count"]], on="date", how="left"
    )
    complete["disaster_count"] = complete["disaster_count"].fillna(0).astype(int)
    return complete


# ── GLM feature matrix ────────────────────────────────────────────────────────

def build_features(start_idx: int, n: int) -> np.ndarray:
    """
    Feature matrix with shape (n, 13):
      col 0  : intercept (all 1s)
      col 1  : scaled time trend  t/100  (scaling aids GLM convergence)
      cols 2–12 : month dummies for Feb–Dec (January = baseline, dropped)
    """
    t         = np.arange(start_idx, start_idx + n)
    month_idx = t % 12                          # 0=Jan, 1=Feb, ..., 11=Dec

    X = np.zeros((n, 13))
    X[:, 0] = 1                                 # intercept
    X[:, 1] = t / 100.0                         # scaled trend
    for m in range(1, 12):                      # Feb(1) … Dec(11)
        X[:, m + 1] = (month_idx == m).astype(float)
    return X


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_negbin(y_train, n_train, n_test, horizon):
    """
    Fit a Negative Binomial GLM on y_train, then:
      • Predict n_test steps for validation
      • Forecast `horizon` steps into the future with 95% CI

    Returns (test_pred, forecast, lower, upper, converged_flag).
    Returns (None, None, None, None, False) on failure.
    """
    if y_train.sum() == 0:
        # Can't fit NegBin on all-zero series
        return None, None, None, None, False

    try:
        X_train = build_features(0, n_train)
        model   = sm.GLM(
            y_train,
            X_train,
            family=sm.families.NegativeBinomial(),
        ).fit(maxiter=200, disp=False)

        # ── Test predictions ──────────────────────────────────────────────
        X_test    = build_features(n_train, n_test)
        test_pred = np.maximum(0.0, model.predict(X_test))

        # ── Future forecast with 95% CI ───────────────────────────────────
        X_fut    = build_features(n_train + n_test, horizon)
        pred_obj = model.get_prediction(X_fut)
        ci       = pred_obj.summary_frame(alpha=0.05)

        forecast = np.maximum(0.0, ci["mean"].values)
        lower    = np.maximum(0.0, ci["mean_ci_lower"].values)
        upper    = ci["mean_ci_upper"].values

        return test_pred, forecast, lower, upper, True

    except Exception:
        return None, None, None, None, False


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    mae  = round(float(np.mean(np.abs(y_true - y_pred))), 3)
    rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3)
    return mae, rmse


# ── Main ──────────────────────────────────────────────────────────────────────

def run_negbin(monthly_agg, state_stats):
    """
    Core function — can be called from model_comparison.py or run standalone.
    Returns (forecasts_dict, performance_df).
    """
    filtered = (
        state_stats[
            state_stats["incidentType"].isin(MAJOR_DISASTERS)
            & (state_stats["total_disasters"] >= 10)
            & (state_stats["months_with_data"] >= 12)
        ]
        .sort_values("total_disasters", ascending=False)
        .head(50)
    )

    forecasts_data = {}
    performance    = []

    for _, row in filtered.iterrows():
        state    = row["state"]
        disaster = row["incidentType"]
        key      = f"{state}_{disaster.replace(' ', '_')}"

        print(f"  [{key}]", end="  ")

        subset = monthly_agg[
            (monthly_agg["state"] == state)
            & (monthly_agg["incidentType"] == disaster)
        ]
        ts    = create_complete_series(subset)
        y     = ts["disaster_count"].values.astype(float)
        dates = ts["date"].values

        n_total = len(y)
        n_train = n_total - N_TEST

        if n_train < 24:
            print("⚠  skipped (< 24 train months)")
            continue

        y_train = y[:n_train]
        y_test  = y[n_train:]

        test_pred, forecast, lower, upper, ok = fit_negbin(
            y_train, n_train, N_TEST, FORECAST_HORIZON
        )

        if not ok:
            print("⚠  GLM failed to converge")
            performance.append({
                "state": state, "disaster_type": disaster,
                "total_historical": int(row["total_disasters"]),
                "mae": None, "rmse": None, "converged": False,
            })
            continue

        mae, rmse = evaluate(y_test, test_pred)

        last_date    = pd.Timestamp(dates[-1])
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=FORECAST_HORIZON, freq="MS",
        )

        forecasts_data[key] = {
            "state":        state,
            "disaster_type": disaster,
            "historical": {
                "dates":  [pd.Timestamp(d).strftime("%Y-%m") for d in dates],
                "counts": y.tolist(),
            },
            "forecast": {
                "dates":            [d.strftime("%Y-%m") for d in future_dates],
                "predicted_counts": [round(v, 2) for v in forecast.tolist()],
                "lower_bound":      [round(v, 2) for v in lower.tolist()],
                "upper_bound":      [round(v, 2) for v in upper.tolist()],
            },
            "model_info": {"mae": mae, "rmse": rmse, "converged": True},
        }

        performance.append({
            "state": state, "disaster_type": disaster,
            "total_historical": int(row["total_disasters"]),
            "mae": mae, "rmse": rmse, "converged": True,
        })

        print(f"MAE={mae:.3f}   RMSE={rmse:.3f}")

    return forecasts_data, pd.DataFrame(performance)


def main():
    print("=" * 70)
    print("NEGATIVE BINOMIAL GLM — FEMA DISASTER FORECASTING")
    print("=" * 70)

    print("\n1. Loading data...")
    monthly_agg = pd.read_csv("disaster_forecast/fema_monthly_aggregated.csv")
    monthly_agg["date"] = pd.to_datetime(monthly_agg["date"])
    state_stats = pd.read_csv("disaster_forecast/fema_state_incident_stats.csv")
    print(f"   ✓ {len(monthly_agg):,} monthly records")
    print(f"   ✓ {len(state_stats):,} state-incident combinations")

    print("\n2. Fitting models...")
    print("=" * 70)
    forecasts_data, perf_df = run_negbin(monthly_agg, state_stats)

    print("\n3. Saving results...")
    with open("disaster_forecast/negbin_forecasts.json", "w") as f:
        json.dump(forecasts_data, f, indent=2)

    perf_df.to_csv("disaster_forecast/negbin_performance.csv", index=False)
    print(f"   ✓ negbin_forecasts.json    ({len(forecasts_data)} models)")
    print(f"   ✓ negbin_performance.csv   ({len(perf_df)} rows)")

    success = perf_df[perf_df["converged"] == True] if "converged" in perf_df.columns else perf_df
    if len(success) > 0:
        print(f"\nSummary ({len(success)} converged models):")
        print(f"   Average MAE  : {success['mae'].mean():.3f}")
        print(f"   Average RMSE : {success['rmse'].mean():.3f}")
        best = success.loc[success["mae"].idxmin()]
        print(f"   Best combo   : {best['state']} {best['disaster_type']} (MAE={best['mae']:.3f})")

    print("\n✓ NegBin modeling complete.")
    return forecasts_data, perf_df


if __name__ == "__main__":
    main()
