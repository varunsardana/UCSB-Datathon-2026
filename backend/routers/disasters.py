from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "disasters.csv"


def _load_disasters() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


@router.get("/disasters")
def list_disasters(
    state: str | None = Query(None, description="Filter by state code e.g. CA"),
    disaster_type: str | None = Query(None, description="Filter by type e.g. wildfire"),
    limit: int = Query(200, ge=1, le=1000),
):
    """
    Return disaster events for the frontend map.

    Expected CSV columns:
      disaster_id, disaster_type, declaration_date, state, county,
      fips_code, lat, lng, severity, title
    """
    df = _load_disasters()
    if df.empty:
        return []

    if state:
        df = df[df["state"].str.upper() == state.upper()]
    if disaster_type:
        df = df[df["disaster_type"].str.lower() == disaster_type.lower()]

    return df.head(limit).fillna("").to_dict(orient="records")


@router.get("/disasters/{disaster_id}")
def get_disaster(disaster_id: str):
    """Return a single disaster event by ID."""
    df = _load_disasters()
    if df.empty:
        raise HTTPException(status_code=404, detail="No disaster data available")

    row = df[df["disaster_id"].astype(str) == disaster_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Disaster {disaster_id} not found")

    return row.iloc[0].fillna("").to_dict()
