"""
ML team prediction interface.

This file is a STUB — the ML team will replace the body of predict()
with their actual trained model inference.

Contract:
  Input:  disaster_type (str), fips_code (str), severity (str)
  Output: dict with keys:
    - disaster_type: str
    - fips_code: str
    - region: str
    - text: str  (narrative summary — this gets embedded into the RAG prompt)
    - predictions: dict[industry -> {job_loss_pct|job_change_pct, recovery_months|peak_month}]
"""


def predict(disaster_type: str, fips_code: str, severity: str = "major") -> dict | None:
    """
    ML team: replace this stub with your model inference.

    The backend will call this when a user query doesn't match any
    pre-computed entry in data/model_predictions.json.
    """
    # ── STUB: return None so the backend gracefully falls back ─────────────
    # Replace this with your actual model call, e.g.:
    #   model = load_your_model()
    #   features = build_features(disaster_type, fips_code, severity)
    #   raw = model.predict(features)
    #   return format_output(raw, disaster_type, fips_code)
    return None