"""
Prophet Forecasting — State-Specific Disaster Frequency

Loads the combos selected by state_disaster_selector.py, fits a Prophet model
on the FULL historical series (2000-01 → 2026-02) for each state+disaster combo,
and forecasts the next 72 months (through ~Feb 2032).

A 12-month cross-validation window is run first to report accuracy (cv_mae, cv_rmse)
without leaking future data into training.

Outputs:
  disaster_forecast/prophet_state_forecasts.json   — full forecast per combo
  disaster_forecast/prophet_forecast_performance.csv — accuracy summary

# ── Climate regressor hook (future enhancement) ──────────────────────────────
# Prophet supports extra regressors via m.add_regressor("enso_index").
# To add ENSO / global temp anomaly / sea-level data:
#   1. Load a monthly climate DataFrame aligned to the same date range
#   2. Merge into train_df and future_df before fit/predict
#   3. Uncomment the add_regressor() calls below (marked with TODO_CLIMATE)
# See: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects.html
# ─────────────────────────────────────────────────────────────────────────────

Run from repo root:
    python disaster_forecast/state_disaster_selector.py   # generates selected_combos.csv
    python disaster_forecast/prophet_forecast.py
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
FORECAST_HORIZON = 72   # months (~6 years, through Feb 2032)
N_CV             = 12   # months held out for cross-validation accuracy estimate


# ── Data helpers ──────────────────────────────────────────────────────────────

def create_complete_series(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing months with 0 so Prophet sees a contiguous monthly series.
    Months with no FEMA declaration get count=0.
    """
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    complete = pd.DataFrame({"date": date_range})
    complete = complete.merge(
        df_subset[["date", "disaster_count"]], on="date", how="left"
    )
    complete["disaster_count"] = complete["disaster_count"].fillna(0).astype(int)
    return complete


def make_prophet_df(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the ds/y format Prophet requires."""
    return (
        ts_df[["date", "disaster_count"]]
        .rename(columns={"date": "ds", "disaster_count": "y"})
        .copy()
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def build_prophet() -> Prophet:
    """
    Shared Prophet configuration used for both CV and final model.

    yearly_seasonality=True   captures hurricane season, wildfire season, etc.
    weekly/daily = False      monthly data — sub-month cycles are irrelevant
    seasonality_mode=additive safer for zero-heavy count series (vs multiplicative)
    changepoint_prior_scale   moderate flexibility: adapts to long-run trend shifts
                              without over-fitting short-term noise
    interval_width=0.95       95% uncertainty bands on the forecast
    """
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    # TODO_CLIMATE: m.add_regressor("enso_index")
    # TODO_CLIMATE: m.add_regressor("global_temp_anomaly")


def fit_and_forecast(ts_df: pd.DataFrame, horizon: int = FORECAST_HORIZON, n_cv: int = N_CV):
    """
    1. Cross-validate: train on [0 : n_total-n_cv], predict last n_cv months → cv_mae, cv_rmse
    2. Final model: train on ALL data, forecast `horizon` months ahead

    Returns:
        fut_rows  — DataFrame with yhat / yhat_lower / yhat_upper for future months
        cv_mae    — cross-validation MAE (float)
        cv_rmse   — cross-validation RMSE (float)
        success   — bool

    Returns (None, None, None, False) on failure or all-zero series.
    """
    n_total = len(ts_df)
    full_df = make_prophet_df(ts_df)

    # Guard: can't model an all-zero series
    if n_total < 2 or full_df["y"].sum() == 0:
        return None, None, None, False

    try:
        # ── Cross-validation ──────────────────────────────────────────────────
        n_train_cv = n_total - n_cv
        cv_train   = full_df.iloc[:n_train_cv]

        m_cv = build_prophet()
        m_cv.fit(cv_train)

        cv_future   = m_cv.make_future_dataframe(periods=n_cv, freq="MS")
        cv_forecast = m_cv.predict(cv_future)

        cv_pred = cv_forecast.iloc[n_train_cv:]["yhat"].clip(lower=0).values
        cv_true = full_df.iloc[n_train_cv:]["y"].values

        cv_mae  = round(float(np.mean(np.abs(cv_true - cv_pred))), 3)
        cv_rmse = round(float(np.sqrt(np.mean((cv_true - cv_pred) ** 2))), 3)

        # ── Final model on full history ───────────────────────────────────────
        m = build_prophet()
        m.fit(full_df)

        future   = m.make_future_dataframe(periods=horizon, freq="MS")
        forecast = m.predict(future)

        # Future rows start at index n_total (rows 0..n_total-1 are historical)
        fut_rows = forecast.iloc[n_total:].reset_index(drop=True)

        return fut_rows, cv_mae, cv_rmse, True

    except Exception:
        return None, None, None, False


# ── Evaluation helpers ────────────────────────────────────────────────────────

def peak_months(ts_df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Return month names with highest average disaster count."""
    month_avg = (
        ts_df.assign(month=ts_df["date"].dt.month)
        .groupby("month")["disaster_count"]
        .mean()
        .sort_values(ascending=False)
    )
    month_names = [
        pd.Timestamp(2000, int(m), 1).strftime("%B")
        for m in month_avg.head(top_n).index
    ]
    return month_names


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PROPHET FORECAST — STATE-SPECIFIC DISASTER FREQUENCY")
    print("=" * 70)

    # ── Load inputs ───────────────────────────────────────────────────────────
    monthly_agg = pd.read_csv("disaster_forecast/fema_monthly_aggregated.csv")
    monthly_agg["date"] = pd.to_datetime(monthly_agg["date"])

    selected = pd.read_csv("disaster_forecast/selected_combos.csv")

    print(f"\n  Monthly records loaded : {len(monthly_agg):,}")
    print(f"  Combos to forecast     : {len(selected)}")
    print(f"  Forecast horizon       : {FORECAST_HORIZON} months")
    print(f"  CV window              : last {N_CV} months of history")
    print("=" * 70)

    forecasts  = {}
    perf_rows  = []
    failed     = []

    for _, row in selected.iterrows():
        state    = row["state"]
        disaster = row["incidentType"]
        key      = f"{state}_{disaster.replace(' ', '_')}"

        print(f"  [{key}]", end="  ")

        # Build complete monthly series for this combo
        subset = monthly_agg[
            (monthly_agg["state"] == state) &
            (monthly_agg["incidentType"] == disaster)
        ]
        ts    = create_complete_series(subset)
        y     = ts["disaster_count"].values.astype(float)
        dates = ts["date"].values

        # Fit + forecast
        fut_rows, cv_mae, cv_rmse, ok = fit_and_forecast(ts)

        if not ok:
            print("⚠  Prophet failed — skipped")
            failed.append(key)
            continue

        # Future date index
        last_date    = pd.Timestamp(dates[-1])
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=FORECAST_HORIZON,
            freq="MS",
        )

        # Seasonal context: which months historically peak?
        peaks = peak_months(ts)

        forecasts[key] = {
            "state":         state,
            "disaster_type": disaster,
            "historical": {
                "dates":  [pd.Timestamp(d).strftime("%Y-%m") for d in dates],
                "counts": y.tolist(),
            },
            "forecast": {
                "dates":            [d.strftime("%Y-%m") for d in future_dates],
                "predicted_counts": [round(v, 2) for v in fut_rows["yhat"].clip(lower=0).tolist()],
                "lower_bound":      [round(v, 2) for v in fut_rows["yhat_lower"].clip(lower=0).tolist()],
                "upper_bound":      [round(v, 2) for v in fut_rows["yhat_upper"].tolist()],
            },
            "model_info": {
                "cv_mae":           cv_mae,
                "cv_rmse":          cv_rmse,
                "train_months":     int(len(ts)),
                "forecast_horizon": FORECAST_HORIZON,
                "peak_months":      peaks,
                "total_historical": int(row["total_disasters"]),
            },
        }

        perf_rows.append({
            "state":            state,
            "disaster_type":    disaster,
            "total_historical": int(row["total_disasters"]),
            "cv_mae":           cv_mae,
            "cv_rmse":          cv_rmse,
            "peak_months":      ", ".join(peaks),
        })

        print(f"CV MAE={cv_mae:.3f}   CV RMSE={cv_rmse:.3f}   peaks: {', '.join(peaks)}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)

    with open("disaster_forecast/prophet_state_forecasts.json", "w") as f:
        json.dump(forecasts, f, indent=2)

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv("disaster_forecast/prophet_forecast_performance.csv", index=False)

    print(f"✓  prophet_state_forecasts.json         ({len(forecasts)} combos)")
    print(f"✓  prophet_forecast_performance.csv     ({len(perf_df)} rows)")

    if failed:
        print(f"\n  ⚠  {len(failed)} combos failed (all-zero or convergence issue): {failed}")

    if len(perf_df) > 0:
        print(f"\nPerformance summary:")
        print(f"   Avg CV MAE  : {perf_df['cv_mae'].mean():.3f}")
        print(f"   Avg CV RMSE : {perf_df['cv_rmse'].mean():.3f}")
        best = perf_df.loc[perf_df["cv_mae"].idxmin()]
        worst = perf_df.loc[perf_df["cv_mae"].idxmax()]
        print(f"   Best combo  : {best['state']} {best['disaster_type']}  (MAE={best['cv_mae']:.3f})")
        print(f"   Worst combo : {worst['state']} {worst['disaster_type']}  (MAE={worst['cv_mae']:.3f})")

    print("\n✓ Forecasting complete.")
    print("  Output ready for RAG ingestion and frontend charts.")

    return forecasts, perf_df


if __name__ == "__main__":
    main()
