"""
Prophet Time Series Forecasting — FEMA Disaster Frequency

Why Prophet over Aliza's baseline (Trend × Seasonality):
  - Automatically detects trend changepoints (the baseline assumes constant linear trend)
  - Handles zero-heavy count series without manual intervention
  - Robust to outlier months (e.g., a single 165-declaration month in FL hurricanes)
  - Returns proper uncertainty intervals via Monte Carlo sampling
  - Built specifically for the kind of seasonal business/event time series
    that disaster declarations represent

Run from repo root:
    python disaster_forecast/prophet_model.py
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from prophet import Prophet

# ── Constants ─────────────────────────────────────────────────────────────────

START_DATE       = "2000-01-01"
END_DATE         = "2026-02-01"
FORECAST_HORIZON = 72   # months (6 years)
N_TEST           = 12   # months held out for evaluation

MAJOR_DISASTERS = [
    "Hurricane", "Severe Storm", "Flood", "Fire",
    "Tornado", "Snowstorm", "Severe Ice Storm",
]


# ── Data helper ───────────────────────────────────────────────────────────────

def create_complete_series(df_subset):
    """Fill missing months with 0 — same logic as proper_timeseries_v2.py."""
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    complete   = pd.DataFrame({"date": date_range})
    complete   = complete.merge(
        df_subset[["date", "disaster_count"]], on="date", how="left"
    )
    complete["disaster_count"] = complete["disaster_count"].fillna(0).astype(int)
    return complete


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_prophet(ts_df, n_train, horizon):
    """
    Fit Prophet on the first n_train rows of ts_df.

    Returns:
      test_pred   — predictions for the n_test held-out rows
      fut_pred    — point forecast for `horizon` future months
      fut_lower   — lower 95% CI
      fut_upper   — upper 95% CI
      success     — bool

    All returned arrays are clipped to >= 0 (no negative disaster counts).
    """
    n_test = len(ts_df) - n_train

    # Prophet requires columns named 'ds' and 'y'
    train_df = (
        ts_df[["date", "disaster_count"]]
        .iloc[:n_train]
        .rename(columns={"date": "ds", "disaster_count": "y"})
        .copy()
    )

    # Guard: need at least 2 data points and at least 1 non-zero
    if len(train_df) < 2 or train_df["y"].sum() == 0:
        return None, None, None, None, False

    try:
        m = Prophet(
            yearly_seasonality=True,    # captures hurricane season, flood season, etc.
            weekly_seasonality=False,   # monthly data — weekly cycle irrelevant
            daily_seasonality=False,
            seasonality_mode="additive",  # additive safer for zero-heavy series
            interval_width=0.95,        # 95% uncertainty interval
            changepoint_prior_scale=0.05,  # moderate flexibility for trend changepoints
        )
        m.fit(train_df)

        # Build future dataframe covering test + forecast periods
        future   = m.make_future_dataframe(periods=n_test + horizon, freq="MS")
        forecast = m.predict(future)

        # Test period: rows n_train through n_train+n_test-1
        test_pred = forecast.iloc[n_train: n_train + n_test]["yhat"].clip(lower=0).values

        # Future horizon: last `horizon` rows
        fut_rows  = forecast.iloc[n_train + n_test:]
        fut_pred  = fut_rows["yhat"].clip(lower=0).values
        fut_lower = fut_rows["yhat_lower"].clip(lower=0).values
        fut_upper = fut_rows["yhat_upper"].values

        return test_pred, fut_pred, fut_lower, fut_upper, True

    except Exception:
        return None, None, None, None, False


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    mae  = round(float(np.mean(np.abs(y_true - y_pred))), 3)
    rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3)
    return mae, rmse


# ── Main ──────────────────────────────────────────────────────────────────────

def run_prophet(monthly_agg, state_stats):
    """
    Core function — called from model_comparison.py or run standalone.
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

        n_total = len(ts)
        n_train = n_total - N_TEST

        if n_train < 24:
            print("⚠  skipped (< 24 train months)")
            continue

        y_test = y[n_train:]

        test_pred, fut_pred, fut_lower, fut_upper, ok = fit_prophet(
            ts, n_train, FORECAST_HORIZON
        )

        if not ok:
            print("⚠  Prophet failed")
            performance.append({
                "state": state, "disaster_type": disaster,
                "total_historical": int(row["total_disasters"]),
                "mae": None, "rmse": None, "success": False,
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
                "predicted_counts": [round(v, 2) for v in fut_pred.tolist()],
                "lower_bound":      [round(v, 2) for v in fut_lower.tolist()],
                "upper_bound":      [round(v, 2) for v in fut_upper.tolist()],
            },
            "model_info": {"mae": mae, "rmse": rmse, "success": True},
        }

        performance.append({
            "state": state, "disaster_type": disaster,
            "total_historical": int(row["total_disasters"]),
            "mae": mae, "rmse": rmse, "success": True,
        })

        print(f"MAE={mae:.3f}   RMSE={rmse:.3f}")

    return forecasts_data, pd.DataFrame(performance)


def main():
    print("=" * 70)
    print("PROPHET MODEL — FEMA DISASTER FORECASTING")
    print("=" * 70)

    print("\n1. Loading data...")
    monthly_agg = pd.read_csv("disaster_forecast/fema_monthly_aggregated.csv")
    monthly_agg["date"] = pd.to_datetime(monthly_agg["date"])
    state_stats = pd.read_csv("disaster_forecast/fema_state_incident_stats.csv")
    print(f"   ✓ {len(monthly_agg):,} monthly records")
    print(f"   ✓ {len(state_stats):,} state-incident combinations")

    print("\n2. Fitting models...")
    print("=" * 70)
    forecasts_data, perf_df = run_prophet(monthly_agg, state_stats)

    print("\n3. Saving results...")
    with open("disaster_forecast/prophet_forecasts.json", "w") as f:
        json.dump(forecasts_data, f, indent=2)

    perf_df.to_csv("disaster_forecast/prophet_performance.csv", index=False)
    print(f"   ✓ prophet_forecasts.json    ({len(forecasts_data)} models)")
    print(f"   ✓ prophet_performance.csv   ({len(perf_df)} rows)")

    success = perf_df[perf_df["success"] == True] if "success" in perf_df.columns else perf_df
    if len(success) > 0:
        print(f"\nSummary ({len(success)} successful models):")
        print(f"   Average MAE  : {success['mae'].mean():.3f}")
        print(f"   Average RMSE : {success['rmse'].mean():.3f}")
        best = success.loc[success["mae"].idxmin()]
        print(f"   Best combo   : {best['state']} {best['disaster_type']} (MAE={best['mae']:.3f})")

    print("\n✓ Prophet modeling complete.")
    return forecasts_data, perf_df


if __name__ == "__main__":
    main()
