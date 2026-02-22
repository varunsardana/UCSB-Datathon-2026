"""
generate_rag_profiles.py — Forecast → RAG Narrative Generator

Reads prophet_state_forecasts.json and converts each state+disaster combo
into a self-contained narrative text chunk suitable for ChromaDB embedding.

Each chunk is written to answer natural language questions like:
  "When is hurricane season in Florida?"
  "How often do wildfires happen in California?"
  "Which months should workers in Oklahoma prepare for tornado disruptions?"

Output: backend/data/forecast_profiles.json
  Ingested by: cd backend && python -m rag.ingest  (step [3/3])

Run from repo root:
    python disaster_forecast/generate_rag_profiles.py
"""

import json
from pathlib import Path

import pandas as pd

# ── State code → full name ────────────────────────────────────────────────────

STATE_NAMES = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
    "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri",
    "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
    "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VA": "Virginia", "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
    "WV": "West Virginia", "WY": "Wyoming",
}

# Disaster type → plain English description for narrative
DISASTER_DESCRIPTIONS = {
    "Hurricane":        "hurricane",
    "Severe Storm":     "severe storm (including tornadoes, damaging winds, and hail)",
    "Flood":            "flood",
    "Fire":             "wildfire",
    "Tornado":          "tornado",
    "Snowstorm":        "snowstorm and winter storm",
    "Severe Ice Storm": "severe ice storm",
}

# Disaster type → most affected industries (for workforce context)
INDUSTRY_CONTEXT = {
    "Hurricane":        "hospitality, tourism, construction, oil and gas, and coastal retail",
    "Severe Storm":     "agriculture, construction, utilities, and outdoor industries",
    "Flood":            "agriculture, real estate, retail, and manufacturing near waterways",
    "Fire":             "forestry, outdoor recreation, hospitality, and rural agriculture",
    "Tornado":          "agriculture, manufacturing, construction, and mobile home communities",
    "Snowstorm":        "transportation, logistics, retail, and outdoor construction",
    "Severe Ice Storm": "utilities, transportation, agriculture, and retail",
}


# ── Forecast analysis helpers ─────────────────────────────────────────────────

def get_forecast_by_month(forecast: dict) -> dict[int, list[float]]:
    """Group forecast predicted_counts by calendar month number."""
    by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for date_str, val in zip(forecast["dates"], forecast["predicted_counts"]):
        month = int(date_str.split("-")[1])
        by_month[month].append(val)
    return by_month


def get_season_description(peak_months: list[str]) -> str:
    """Turn a peak month list into a readable season description (sentence-ready)."""
    months = [m.capitalize() for m in peak_months]
    if not months:
        return "Activity is not strongly seasonal"
    if len(months) == 1:
        return f"Activity peaks in {months[0]}"
    if len(months) == 2:
        return f"Activity peaks in {months[0]} and {months[1]}"
    return f"Activity peaks in {months[0]}, {months[1]}, and {months[2]}"


def get_risk_window(peak_months: list[str]) -> str:
    """Describe the high-risk window for workforce planning."""
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    months = [m.capitalize() for m in peak_months]
    if not months:
        return "year-round with no strong seasonal peak"
    sorted_peaks = sorted(months, key=lambda m: month_order.index(m) if m in month_order else 99)
    if len(sorted_peaks) == 1:
        return sorted_peaks[0]
    return f"{sorted_peaks[0]}–{sorted_peaks[-1]}"


# ── Narrative builder ─────────────────────────────────────────────────────────

def build_narrative(state: str, disaster_type: str, entry: dict) -> str:
    """
    Build a self-contained, embeddable narrative text chunk for one combo.
    Designed to answer natural-language questions about seasonal disaster risk.
    """
    info        = entry.get("model_info", {})
    hist        = entry["historical"]
    forecast    = entry["forecast"]

    state_name  = STATE_NAMES.get(state, state)
    disaster_en = DISASTER_DESCRIPTIONS.get(disaster_type, disaster_type.lower())
    industries  = INDUSTRY_CONTEXT.get(disaster_type, "multiple industries")

    total       = info.get("total_historical", "N/A")
    cv_mae      = info.get("cv_mae")
    peak_months = [m.capitalize() for m in info.get("peak_months", [])]
    train_months = info.get("train_months", 314)

    # Historical stats
    hist_counts  = [c for c in hist["counts"] if c > 0]
    avg_active   = sum(hist_counts) / len(hist_counts) if hist_counts else 0
    years_data   = round(train_months / 12, 0)

    # Forecast stats — split peak vs. off-season months
    by_month     = get_forecast_by_month(forecast)
    month_names  = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    peak_idxs    = [month_names.index(m) + 1 for m in peak_months if m in month_names]
    off_idxs     = [m for m in range(1, 13) if m not in peak_idxs]

    peak_vals    = [v for idx in peak_idxs for v in by_month[idx]]
    off_vals     = [v for idx in off_idxs  for v in by_month[idx]]

    avg_peak     = round(sum(peak_vals) / len(peak_vals), 1) if peak_vals else 0
    avg_off      = round(sum(off_vals)  / len(off_vals),  1) if off_vals  else 0

    season_desc  = get_season_description(peak_months)
    risk_window  = get_risk_window(peak_months)
    mae_note     = f"(model CV MAE={cv_mae:.2f})" if cv_mae is not None else ""

    narrative = f"""{state_name} {disaster_type} Seasonal Risk Profile

{state_name} has recorded {int(total):,} FEMA {disaster_type.lower()} declarations from 2000–2026 ({int(years_data)} years of data). During active months, the state averages {avg_active:.1f} declarations per month. {season_desc}, with the highest-risk window being {risk_window}.

6-Year Forecast {mae_note}:
Our Prophet time series model projects {disaster_type.lower()} declaration frequency in {state_name} through February 2032. During forecasted peak months ({', '.join(peak_months) if peak_months else 'N/A'}), the model predicts an average of {avg_peak:.1f} declarations/month. During off-season months, activity is expected to remain near {avg_off:.1f} declarations/month.

Workforce Impact:
Industries most affected by {disaster_type.lower()} declarations in {state_name} include {industries}. Workers and employers in {state_name} should anticipate potential workforce disruption during {risk_window}. Disaster declaration frequency is seasonal — some years may see significantly more or fewer events depending on weather patterns.

Important note: These forecasts predict FEMA declaration frequency (how often disasters are formally declared), not disaster probability or severity. A single major event can generate many county-level declarations in one month."""

    return narrative.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAG PROFILE GENERATOR")
    print("=" * 60)

    forecast_path = Path("disaster_forecast/prophet_state_forecasts.json")
    out_path      = Path("backend/data/forecast_profiles.json")

    if not forecast_path.exists():
        print("ERROR: prophet_state_forecasts.json not found.")
        print("Run: python disaster_forecast/prophet_forecast.py")
        return

    with open(forecast_path) as f:
        data = json.load(f)

    print(f"\nProcessing {len(data)} forecast combos...")
    print("-" * 60)

    profiles = []

    for key, entry in data.items():
        state        = entry["state"]
        disaster     = entry["disaster_type"]

        text = build_narrative(state, disaster, entry)

        profile = {
            "id":           f"forecast_{key}",
            "state":        state,
            "disaster_type": disaster.lower(),
            "text":         text,
            "source":       "prophet_forecast",
        }
        profiles.append(profile)
        print(f"  ✓  {key:<35}  ({len(text)} chars)")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓  {len(profiles)} profiles written → {out_path}")
    print(f"   Avg text length: {sum(len(p['text']) for p in profiles) // len(profiles)} chars")
    print(f"\nNext step:")
    print(f"   cd backend && python -m rag.ingest")


if __name__ == "__main__":
    main()
