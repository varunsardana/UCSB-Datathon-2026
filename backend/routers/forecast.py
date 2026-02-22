"""
GET /api/forecast/available  — list all state+disaster combos with forecasts
GET /api/forecast/states     — list unique states
GET /api/forecast/chart      — chart data for one combo (state + disaster_type)

The heavy lifting is done by disaster_forecast/chart_data.py.
This router just exposes it over HTTP with proper error handling.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

# Resolve path to forecast JSON relative to THIS file's location:
# backend/routers/ → backend/ → repo_root/ → disaster_forecast/
_FORECAST_JSON = (
    Path(__file__).parent.parent.parent
    / "disaster_forecast"
    / "prophet_state_forecasts.json"
)

# Import the formatter from the disaster_forecast module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "disaster_forecast"))

from chart_data import (
    get_forecast_chart_data,
    list_available_combos,
    list_states,
    list_disaster_types_for_state,
)

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/available")
def get_available_combos():
    """
    List every state+disaster combo that has a pre-computed Prophet forecast.

    Response:
        {
          "count": 142,
          "combos": [
            {"state": "AL", "disaster_type": "Severe Storm", "peak_months": ["March"], "cv_mae": 0.8},
            ...
          ]
        }

    Frontend use: populate the state + disaster dropdowns / map legend.
    """
    try:
        combos = list_available_combos(_FORECAST_JSON)
        return {"count": len(combos), "combos": combos}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/states")
def get_states():
    """
    Return unique states that have at least one forecast.

    Response: {"states": ["AL", "AK", "AZ", ...]}

    Frontend use: first-level filter dropdown.
    """
    try:
        states = list_states(_FORECAST_JSON)
        return {"states": states}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/types")
def get_disaster_types(state: str = Query(..., description="Two-letter state code, e.g. FL")):
    """
    Return disaster types available for a given state.

    Response: {"state": "FL", "disaster_types": ["Hurricane", "Severe Storm"]}

    Frontend use: second-level filter dropdown (depends on selected state).
    """
    try:
        types = list_disaster_types_for_state(state.upper(), _FORECAST_JSON)
        if not types:
            raise HTTPException(
                status_code=404,
                detail=f"No forecasts found for state '{state}'. Check /api/forecast/states for valid options.",
            )
        return {"state": state.upper(), "disaster_types": types}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/chart")
def get_chart_data(
    state: str = Query(..., description="Two-letter state code, e.g. FL"),
    disaster_type: str = Query(..., description="Disaster type, e.g. Hurricane"),
):
    """
    Return chart-ready time series data for one state + disaster combo.

    Response shape:
    {
      "state": "FL",
      "disaster_type": "Hurricane",
      "meta": {
        "total_historical_declarations": 856,
        "peak_months": ["September", "August", "October"],
        "cv_mae": 1.23,
        "cv_rmse": 2.45,
        "data_start": "2000-01",
        "data_end": "2026-02",
        "forecast_start": "2026-03",
        "forecast_end": "2032-02",
        "train_months": 314,
        "forecast_horizon_months": 72
      },
      "historical": [
        {"date": "2000-01", "count": 3},
        {"date": "2000-02", "count": 0},
        ...
      ],
      "forecast": [
        {"date": "2026-03", "predicted": 1.52, "lower": 0.0, "upper": 4.21},
        ...
      ]
    }

    Frontend notes:
    - `historical` covers 2000-01 → 2026-02 (monthly, zero-filled)
    - `forecast` covers 72 months starting from 2026-03
    - Render historical as a solid line; forecast as dashed + confidence band
    - lower/upper are 95% uncertainty bounds — suitable for area shading
    """
    try:
        data = get_forecast_chart_data(
            state=state.upper(),
            disaster_type=disaster_type,
            forecasts_path=_FORECAST_JSON,
        )
        return data
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Combo not found: {state.upper()} / {disaster_type}. "
                   f"Check /api/forecast/types?state={state.upper()} for valid options.",
        )
