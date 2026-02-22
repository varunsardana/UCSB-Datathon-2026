from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()

ANALYTICS_PATH = Path(__file__).parent.parent / "data" / "processed" / "regional_analytics.csv"


def _load_analytics() -> pd.DataFrame:
    if not ANALYTICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ANALYTICS_PATH)


@router.get("/analytics")
def get_analytics(
    fips_code: str | None = Query(None, description="Filter by county FIPS code e.g. 06037"),
    disaster_type: str | None = Query(None, description="Filter by disaster type"),
    state: str | None = Query(None, description="Filter by state code e.g. CA"),
    industry: str | None = Query(None, description="Filter by industry group"),
):
    """
    Return time-series analytics for the frontend charts.

    Expected CSV columns:
      fips_code, state, industry_group, disaster_type, month_offset,
      job_change_count, job_change_pct, recovery_rate
    """
    df = _load_analytics()
    if df.empty:
        return []

    if fips_code:
        df = df[df["fips_code"].astype(str) == str(fips_code)]
    if disaster_type:
        df = df[df["disaster_type"].str.lower() == disaster_type.lower()]
    if state:
        df = df[df["state"].str.upper() == state.upper()]
    if industry:
        df = df[df["industry_group"].str.lower().str.contains(industry.lower())]

    return df.fillna(0).to_dict(orient="records")


@router.get("/analytics/summary")
def get_summary():
    """
    Return top-level summary stats for the landing page.
    """
    df = _load_analytics()
    if df.empty:
        return {"total_events": 0, "regions_analyzed": 0, "industries_tracked": 0}

    return {
        "total_events": int(df["disaster_type"].nunique()),
        "regions_analyzed": int(df["fips_code"].nunique()),
        "industries_tracked": int(df["industry_group"].nunique()),
    }
