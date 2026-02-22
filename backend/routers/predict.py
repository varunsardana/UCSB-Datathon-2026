from fastapi import APIRouter
from pydantic import BaseModel

from services.model_service import format_prediction_context, get_prediction

router = APIRouter()


class PredictRequest(BaseModel):
    disaster_type: str
    fips_code: str
    severity: str = "major"


@router.post("/predict")
def predict(req: PredictRequest):
    """
    Return workforce disruption prediction for a given disaster scenario.

    Used by the frontend to populate prediction cards and charts,
    and internally by /api/chat to ground the RAG response.

    Response shape (mirrors model_predictions.json entries):
    {
      "disaster_type": "wildfire",
      "fips_code": "06037",
      "region": "Los Angeles County, CA",
      "text": "Narrative summary...",
      "predictions": {
        "Retail": { "job_loss_pct": 20, "recovery_months": 6 },
        "Construction": { "job_change_pct": 180, "peak_month": 3 },
        ...
      }
    }
    """
    result = get_prediction(
        disaster_type=req.disaster_type,
        fips_code=req.fips_code,
        severity=req.severity,
    )

    if result is None:
        return {
            "error": "No prediction available for this scenario",
            "disaster_type": req.disaster_type,
            "fips_code": req.fips_code,
        }

    return result


@router.get("/predict/scenarios")
def list_scenarios():
    """Return all available pre-computed prediction scenarios."""
    from services.model_service import _precomputed

    scenarios = [
        {
            "key": key,
            "disaster_type": entry.get("disaster_type"),
            "fips_code": entry.get("fips_code"),
            "region": entry.get("region"),
        }
        for key, entry in _precomputed.items()
    ]
    return {"count": len(scenarios), "scenarios": scenarios}
