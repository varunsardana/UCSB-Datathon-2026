"""
Model service — loads pre-computed predictions and the ML model artifact.

Priority:
1. Pre-computed lookup (model_predictions.json) — instant, no inference
2. Live prediction via ml/predict.py — runs model at request time (fallback)
"""

import json
from pathlib import Path

_precomputed: dict[str, dict] = {}
_model_loaded: bool = False


def load_model() -> None:
    global _precomputed, _model_loaded

    predictions_path = Path(__file__).parent.parent / "data" / "model_predictions.json"
    if predictions_path.exists():
        with open(predictions_path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            key = f"{entry['disaster_type']}_{entry['fips_code']}"
            _precomputed[key] = entry
        print(f"Model service: loaded {len(_precomputed)} pre-computed predictions")
    else:
        print("Model service: model_predictions.json not found — will try live prediction only")

    # Try loading the ML model artifact
    model_path = Path(__file__).parent.parent / "ml" / "models" / "xgboost_model.pkl"
    if model_path.exists():
        try:
            import joblib
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent))
            _model_loaded = True
            print("Model service: ML model artifact loaded")
        except Exception as e:
            print(f"Model service: could not load ML model — {e}")
    else:
        print("Model service: no ML model artifact found — pre-computed only")


def get_prediction(
    disaster_type: str,
    fips_code: str,
    severity: str = "major",
) -> dict | None:
    """
    Return prediction for a (disaster_type, fips_code) scenario.

    Returns a dict with at minimum:
      - text: str  (narrative summary to inject into RAG prompt)
      - predictions: dict  (structured numbers per industry)
    """
    # 1. Pre-computed lookup (fastest)
    key = f"{disaster_type}_{fips_code}"
    if key in _precomputed:
        return _precomputed[key]

    # 2. Live prediction fallback
    if _model_loaded:
        try:
            from ml.predict import predict as ml_predict
            result = ml_predict(disaster_type, fips_code, severity)
            if result:
                return result
        except Exception as e:
            print(f"Live prediction failed for {key}: {e}")

    return None


def format_prediction_context(prediction: dict | None) -> str:
    """Turn a prediction dict into a human-readable string for the RAG prompt."""
    if not prediction:
        return ""

    lines = []
    region = prediction.get("region", prediction.get("fips_code", "Unknown region"))
    disaster = prediction.get("disaster_type", "disaster")
    lines.append(f"Forecast for {disaster} in {region}:")

    base_text = prediction.get("text", "")
    if base_text:
        lines.append(base_text)

    predictions = prediction.get("predictions", {})
    if predictions:
        lines.append("\nDetailed industry forecasts:")
        for industry, data in predictions.items():
            if "job_loss_pct" in data:
                recovery = data.get("recovery_months", "?")
                lines.append(
                    f"  - {industry}: {data['job_loss_pct']}% job loss, "
                    f"recovery expected in {recovery} months"
                )
            elif "job_change_pct" in data:
                peak = data.get("peak_month", "?")
                lines.append(
                    f"  - {industry}: +{data['job_change_pct']}% demand surge, "
                    f"peak at month {peak}"
                )

    return "\n".join(lines)
