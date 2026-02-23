"""
chart_data.py — Time Series Chart Data Formatter

Reads the pre-computed Prophet forecasts (prophet_state_forecasts.json)
and returns chart-ready JSON dicts for a given state + disaster combo.

Usage (standalone test):
    python disaster_forecast/chart_data.py --state FL --disaster Hurricane

Imported by backend/routers/forecast.py to serve the frontend API.

Output shape (designed for Recharts / Chart.js / Plotly / Nivo):
{
  "state": "FL",
  "disaster_type": "Hurricane",
  "meta": {
    "total_historical_declarations": 856,
    "avg_monthly": 29.52,
    "peak_months": ["September", "August", "October"],
    "cv_mae": 1.23,
    "cv_rmse": 2.45,
    "data_start": "2000-01",
    "data_end": "2026-02",
    "forecast_start": "2026-03",
    "forecast_end": "2032-02"
  },
  "historical": [
    {"date": "2000-01", "count": 3},
    ...
  ],
  "forecast": [
    {"date": "2026-03", "predicted": 1.5, "lower": 0.0, "upper": 4.2},
    ...
  ]
}
"""

import json
from pathlib import Path

# Default path — works when called from repo root OR imported by backend
_DEFAULT_FORECAST_PATH = (
    Path(__file__).parent / "prophet_state_forecasts.json"
)


# ── Core loader ───────────────────────────────────────────────────────────────

def _load_forecasts(path: Path) -> dict:
    """Load and cache the forecast JSON. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {path}\n"
            "Run: python disaster_forecast/prophet_forecast.py"
        )
    with open(path) as f:
        return json.load(f)


# ── Public API ────────────────────────────────────────────────────────────────

def get_forecast_chart_data(
    state: str,
    disaster_type: str,
    forecasts_path: Path | str = _DEFAULT_FORECAST_PATH,
) -> dict:
    """
    Return chart-ready JSON for one state + disaster combo.

    Parameters
    ----------
    state          : Two-letter state code, e.g. "FL"
    disaster_type  : Disaster label, e.g. "Hurricane" or "Severe Storm"
    forecasts_path : Path to prophet_state_forecasts.json (override for testing)

    Returns
    -------
    dict with keys: state, disaster_type, meta, historical, forecast

    Raises
    ------
    FileNotFoundError  — forecast JSON not found (run prophet_forecast.py first)
    KeyError           — requested combo not in forecast JSON
    """
    forecasts = _load_forecasts(Path(forecasts_path))

    # Build the lookup key (same convention as prophet_forecast.py)
    key = f"{state}_{disaster_type.replace(' ', '_')}"

    if key not in forecasts:
        available = list(forecasts.keys())
        raise KeyError(
            f"Combo '{key}' not in forecasts. "
            f"Available ({len(available)}): {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    entry = forecasts[key]
    info  = entry.get("model_info", {})
    hist  = entry["historical"]
    fcast = entry["forecast"]

    # ── Historical series ─────────────────────────────────────────────────────
    historical = [
        {"date": d, "count": c}
        for d, c in zip(hist["dates"], hist["counts"])
    ]

    # ── Forecast series ───────────────────────────────────────────────────────
    forecast = [
        {
            "date":      d,
            "predicted": p,
            "lower":     lo,
            "upper":     hi,
        }
        for d, p, lo, hi in zip(
            fcast["dates"],
            fcast["predicted_counts"],
            fcast["lower_bound"],
            fcast["upper_bound"],
        )
    ]

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "total_historical_declarations": info.get("total_historical"),
        "peak_months":                   info.get("peak_months", []),
        "cv_mae":                        info.get("cv_mae"),
        "cv_rmse":                       info.get("cv_rmse"),
        "train_months":                  info.get("train_months"),
        "forecast_horizon_months":       info.get("forecast_horizon"),
        "data_start":                    hist["dates"][0]  if hist["dates"]  else None,
        "data_end":                      hist["dates"][-1] if hist["dates"]  else None,
        "forecast_start":                fcast["dates"][0]  if fcast["dates"] else None,
        "forecast_end":                  fcast["dates"][-1] if fcast["dates"] else None,
    }

    return {
        "state":        entry["state"],
        "disaster_type": entry["disaster_type"],
        "meta":         meta,
        "historical":   historical,
        "forecast":     forecast,
    }


def list_available_combos(
    forecasts_path: Path | str = _DEFAULT_FORECAST_PATH,
) -> list[dict]:
    """
    Return all state + disaster combos that have a forecast.

    Useful for populating frontend dropdowns / map filters.

    Returns list of dicts:
      [{"state": "FL", "disaster_type": "Hurricane", "peak_months": [...]}, ...]
    """
    forecasts = _load_forecasts(Path(forecasts_path))

    combos = []
    for entry in forecasts.values():
        info = entry.get("model_info", {})
        combos.append({
            "state":        entry["state"],
            "disaster_type": entry["disaster_type"],
            "peak_months":  info.get("peak_months", []),
            "cv_mae":       info.get("cv_mae"),
        })

    # Sort: state A→Z, then disaster type A→Z
    combos.sort(key=lambda x: (x["state"], x["disaster_type"]))
    return combos


def list_states(
    forecasts_path: Path | str = _DEFAULT_FORECAST_PATH,
) -> list[str]:
    """Return sorted list of unique states that have at least one forecast."""
    combos = list_available_combos(forecasts_path)
    return sorted({c["state"] for c in combos})


def list_disaster_types_for_state(
    state: str,
    forecasts_path: Path | str = _DEFAULT_FORECAST_PATH,
) -> list[str]:
    """Return disaster types available for a given state."""
    combos = list_available_combos(forecasts_path)
    return sorted(
        c["disaster_type"] for c in combos if c["state"] == state
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="Preview chart data for one combo")
    parser.add_argument("--state",    default="FL",        help="State code, e.g. FL")
    parser.add_argument("--disaster", default="Hurricane", help="Disaster type, e.g. Hurricane")
    args = parser.parse_args()

    print(f"\nAvailable states: {list_states()}\n")

    data = get_forecast_chart_data(args.state, args.disaster)

    print(f"State:         {data['state']}")
    print(f"Disaster:      {data['disaster_type']}")
    print(f"Meta:          {data['meta']}")
    print(f"Historical pts:{len(data['historical'])}  (first 3: {data['historical'][:3]})")
    print(f"Forecast pts:  {len(data['forecast'])}  (first 3: {data['forecast'][:3]})")
