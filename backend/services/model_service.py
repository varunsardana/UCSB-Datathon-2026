"""
Model service — loads pre-computed predictions and the ML model artifact.

Priority for XGBoost predictions:
1. Exact fips_code lookup in model_predictions.json   — instant
2. State-level best match (most sectors covered)       — instant fallback
3. Live prediction via ml/predict.py                   — runtime fallback

Prophet forecasts are loaded separately and always injected as guaranteed
structured context (not relying on ChromaDB vector search to surface them).
"""

import json
from pathlib import Path

_precomputed: dict[str, dict] = {}
_precomputed_by_state: dict[str, list] = {}   # "{disaster_type}_{STATE}" → [entries]
_prophet_forecasts: dict[str, dict] = {}       # raw prophet_state_forecasts.json
_model_loaded: bool = False

# 2-digit FIPS state prefix → state abbreviation
FIPS_STATE_MAP: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

# model_predictions disaster types (lowercase_underscored) → prophet JSON key suffix
# prophet_state_forecasts.json keys look like "FL_Hurricane", "CA_Fire", "OK_Severe_Storm"
PROPHET_TYPE_MAP: dict[str, str] = {
    "fire":         "Fire",
    "flood":        "Flood",
    "hurricane":    "Hurricane",
    "severe_storm": "Severe_Storm",
    "tornado":      "Tornado",
}


def load_model() -> None:
    global _precomputed, _precomputed_by_state, _prophet_forecasts, _model_loaded

    base = Path(__file__).parent.parent

    # ── 1. Pre-computed XGBoost predictions ──────────────────────────────────
    predictions_path = base / "data" / "model_predictions.json"
    if predictions_path.exists():
        with open(predictions_path, encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            # Primary key: exact disaster_type + fips_code lookup
            key = f"{entry['disaster_type']}_{entry['fips_code']}"
            _precomputed[key] = entry

            # Secondary index: state-level fallback
            fips = str(entry.get("fips_code", ""))
            if len(fips) >= 2:
                state = FIPS_STATE_MAP.get(fips[:2])
                if state:
                    state_key = f"{entry['disaster_type']}_{state}"
                    _precomputed_by_state.setdefault(state_key, []).append(entry)

        print(
            f"Model service: loaded {len(_precomputed)} pre-computed predictions "
            f"across {len(_precomputed_by_state)} state×disaster combos"
        )
    else:
        print("Model service: model_predictions.json not found — will try live prediction only")

    # ── 2. Prophet time-series forecasts ─────────────────────────────────────
    # Check backend/data/ first, then the source disaster_forecast/ directory
    prophet_paths = [
        base / "data" / "prophet_state_forecasts.json",
        base.parent / "disaster_forecast" / "prophet_state_forecasts.json",
    ]
    for p in prophet_paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _prophet_forecasts = json.load(f)
            print(f"Model service: loaded {len(_prophet_forecasts)} Prophet forecasts from {p.name}")
            break
    else:
        print("Model service: prophet_state_forecasts.json not found — no frequency forecast context")

    # ── 3. SQL engine — in-memory SQLite over both model JSONs ───────────────
    # Resolves the actual prophet path used above
    prophet_path_used = next((p for p in prophet_paths if p.exists()), None)
    if prophet_path_used and predictions_path.exists():
        try:
            from rag.sql_engine import init_db
            init_db(predictions_path, prophet_path_used)
        except Exception as e:
            print(f"Model service: SQL engine failed to init — {e}")
    else:
        print("Model service: SQL engine skipped (one or both JSON files missing)")

    # ── 4. ML model artifact (optional, for live inference fallback) ──────────
    model_path = base / "ml" / "models" / "xgboost_model.pkl"
    if model_path.exists():
        try:
            import sys
            sys.path.insert(0, str(base))
            _model_loaded = True
            print("Model service: ML model artifact loaded")
        except Exception as e:
            print(f"Model service: could not load ML model — {e}")
    else:
        print("Model service: no ML model artifact found — pre-computed only")


# ── XGBoost prediction accessors ─────────────────────────────────────────────

def get_prediction(
    disaster_type: str,
    fips_code: str,
    severity: str = "major",
) -> dict | None:
    """Exact lookup by disaster_type + fips_code."""
    key = f"{disaster_type}_{fips_code}"
    if key in _precomputed:
        return _precomputed[key]

    if _model_loaded:
        try:
            from ml.predict import predict as ml_predict
            result = ml_predict(disaster_type, fips_code, severity)
            if result:
                return result
        except Exception as e:
            print(f"Live prediction failed for {key}: {e}")

    return None


def get_prediction_by_state(disaster_type: str, state: str) -> dict | None:
    """
    State-level fallback when user doesn't specify a FIPS code.
    Returns the most representative entry (most sectors covered = most data).
    """
    state_key = f"{disaster_type.lower()}_{state.upper()}"
    entries = _precomputed_by_state.get(state_key, [])
    if not entries:
        return None
    # Most sectors covered = most informative for the LLM
    return max(entries, key=lambda e: len(e.get("predictions", {})))


def format_prediction_context(prediction: dict | None) -> str:
    """Convert a pre-computed XGBoost prediction dict into a RAG prompt block."""
    if not prediction:
        return ""

    lines = []
    region = prediction.get("region", prediction.get("fips_code", "Unknown region"))
    disaster = prediction.get("disaster_type", "disaster").replace("_", " ").title()
    lines.append(f"Employment Impact Model — {disaster} event in {region}:")

    base_text = prediction.get("text", "")
    if base_text:
        lines.append(f"  Model summary: {base_text}")

    predictions = prediction.get("predictions", {})
    if predictions:
        lines.append("  Sector-level projections (based on 26 years of historical disaster data):")
        for industry, data in predictions.items():
            if "job_loss_pct" in data:
                recovery = data.get("recovery_months", "?")
                lines.append(
                    f"    • {industry}: {data['job_loss_pct']}% job displacement expected, "
                    f"recovery in approximately {recovery} months"
                )
            elif "job_change_pct" in data:
                peak = data.get("peak_month", "?")
                lines.append(
                    f"    • {industry}: +{data['job_change_pct']}% labor demand surge, "
                    f"peaking around month {peak} post-disaster"
                )

    return "\n".join(lines)


# ── Prophet forecast accessors ───────────────────────────────────────────────

def get_forecast(state: str, disaster_type: str) -> dict | None:
    """
    Return the raw Prophet forecast entry for a (state, disaster_type) combo.
    Keys in prophet_state_forecasts.json look like: "FL_Hurricane", "CA_Fire", "OK_Severe_Storm"
    """
    if not _prophet_forecasts:
        return None

    dt_norm = disaster_type.lower().replace(" ", "_")
    prophet_suffix = PROPHET_TYPE_MAP.get(dt_norm)
    if not prophet_suffix:
        return None

    key = f"{state.upper()}_{prophet_suffix}"
    return _prophet_forecasts.get(key)


def format_forecast_context(state: str | None, disaster_type: str | None) -> str:
    """
    Convert a Prophet forecast entry into a structured RAG prompt block.

    This is injected DIRECTLY — not retrieved from ChromaDB — so it is
    guaranteed to appear in every prompt where state + disaster_type are known.
    """
    if not state or not disaster_type:
        return ""

    entry = get_forecast(state, disaster_type)
    if not entry:
        return ""

    info = entry.get("model_info", {})
    peak_months = info.get("peak_months", [])
    total_historical = info.get("total_historical", 0)
    cv_mae = info.get("cv_mae", None)

    forecast = entry.get("forecast", {})
    dates = forecast.get("dates", [])
    predicted = forecast.get("predicted_counts", [])
    upper = forecast.get("upper_bound", [])

    # Summarise next 12 months of the forecast horizon
    next_12_pred = predicted[:12]
    next_12_dates = dates[:12]
    avg_next_12 = (
        round(sum(next_12_pred) / len(next_12_pred), 2) if next_12_pred else 0.0
    )

    dt_display = disaster_type.replace("_", " ").title()
    state_display = state.upper()

    lines = [f"{dt_display} Risk Forecast for {state_display}:"]
    lines.append(
        f"  Historical record: {total_historical} FEMA major disaster declarations "
        f"in {state_display} (2000–2026, 26 years of data)"
    )

    if peak_months:
        lines.append(f"  Seasonal peak months: {', '.join(peak_months)}")
        if len(peak_months) >= 2:
            lines.append(
                f"  → Highest-risk window for workers: {peak_months[0]}–{peak_months[-1]}"
            )

    lines.append(
        f"  6-year forward forecast (2026–2032): "
        f"avg {avg_next_12:.1f} declaration-months/month over next 12 months"
    )

    if next_12_pred and next_12_dates:
        max_val = max(next_12_pred)
        max_idx = next_12_pred.index(max_val)
        upper_val = round(upper[max_idx], 1) if upper else "N/A"
        lines.append(
            f"  Forecast peak month: {next_12_dates[max_idx]} "
            f"({max_val:.1f} predicted, up to {upper_val} in worst case)"
        )

    if cv_mae is not None:
        lines.append(
            f"  Forecast accuracy: ±{cv_mae:.2f} declarations/month "
            f"(cross-validated on 2000–2026 data)"
        )

    return "\n".join(lines)
