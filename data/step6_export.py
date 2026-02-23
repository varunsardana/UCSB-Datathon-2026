"""
STEP 6: Export predictions for the backend API + RAG chatbot

What this does:
- Loads predictions.csv (11,796 rows with XGBoost predictions)
- Aggregates by disaster_type × fips_code × sector across windows
- Computes job_change_pct from excess_exits / baseline
- Estimates recovery_months from the 3-window pattern
- Outputs two files the backend expects:
    1. model_predictions.json — per-scenario sector impact summaries
    2. regional_analytics.csv — month-by-month time series for charts

Mapping our data → backend format:
  - excess_exits > 0 + baseline > 0 → job_loss_pct (sector contracting)
  - excess_exits < 0 → job_change_pct (sector expanding / demand surge)
  - 3 windows → recovery_months estimate
  - exits_month_1..6 → month_offset time series
"""

import pandas as pd
import numpy as np
import json

df = pd.read_csv("predictions.csv")
df['fips_code'] = df['fips_code'].astype(str).str.zfill(5)
print(f"Loaded {len(df)} prediction rows")

# ── FIPS → region name mapping ──────────────────────────
# Build from our data: fips_code → "designatedArea, STATE"
fips_region = (df[['fips_code', 'designatedArea', 'state']]
    .drop_duplicates(subset='fips_code')
    .set_index('fips_code'))

def get_region_name(fips):
    if fips in fips_region.index:
        row = fips_region.loc[fips]
        area = row['designatedArea'] if isinstance(row, pd.Series) else row.iloc[0]['designatedArea']
        state = row['state'] if isinstance(row, pd.Series) else row.iloc[0]['state']
        # Clean up FEMA's area names (e.g., "Los Angeles (County)" → "Los Angeles County")
        area = str(area).replace(' (County)', ' County').replace(' (City)', '')
        return f"{area}, {state}"
    return f"FIPS {fips}"

# ── Disaster type name normalization ────────────────────
# Backend uses lowercase; our data has title case
def normalize_disaster_type(t):
    return t.lower().replace(' ', '_')

# ══════════════════════════════════════════════════════════
# 1. MODEL PREDICTIONS JSON
# ══════════════════════════════════════════════════════════
print("\nGenerating model_predictions.json...")

# For each disaster_type × fips_code combo, aggregate sector impacts
scenarios = df.groupby(['incidentType', 'fips_code']).agg(
    state=('state', 'first'),
    designatedArea=('designatedArea', 'first'),
).reset_index()

results = []

for _, scenario in scenarios.iterrows():
    dtype = scenario['incidentType']
    fips = scenario['fips_code']

    # Get all rows for this scenario
    mask = (df['incidentType'] == dtype) & (df['fips_code'] == fips)
    scenario_df = df[mask]

    predictions = {}
    sector_summaries = []

    for sector in scenario_df['sector'].unique():
        sector_data = scenario_df[scenario_df['sector'] == sector]

        # Get predicted excess exits per window
        w1 = sector_data[sector_data['window'] == 'window_1']['xgb_predicted'].mean()
        w2 = sector_data[sector_data['window'] == 'window_2']['xgb_predicted'].mean()
        w3 = sector_data[sector_data['window'] == 'window_3']['xgb_predicted'].mean()

        # Skip if any window has no data
        if any(pd.isna(x) for x in [w1, w2, w3]):
            continue

        # Get baseline for percentage calculation
        baseline = sector_data['baseline_exits'].mean()

        if baseline < 1 or pd.isna(baseline):
            continue  # skip sectors with no baseline data

        # Average predicted excess across all windows for overall impact
        avg_excess = (w1 + w2 + w3) / 3

        # Job change percentage (relative to baseline)
        job_change_pct = (avg_excess / baseline) * 100

        if abs(job_change_pct) < 1:
            continue  # skip negligible impacts

        # Cap at 95% — can't lose more than all jobs
        job_change_pct = max(-95, min(95, job_change_pct))

        # Per-window percentages (capped)
        w1_pct = max(-95, min(95, (w1 / baseline) * 100))
        w2_pct = max(-95, min(95, (w2 / baseline) * 100))
        w3_pct = max(-95, min(95, (w3 / baseline) * 100))

        # Window trajectory — the actual recovery curve
        window_trajectory = {
            "window_1_pct": round(w1_pct, 1),
            "window_2_pct": round(w2_pct, 1),
            "window_3_pct": round(w3_pct, 1),
        }

        if avg_excess > 0:
            # Sector is LOSING jobs (more exits than baseline)
            predictions[sector] = {
                "job_loss_pct": round(abs(job_change_pct)),
                **window_trajectory,
            }
            sector_summaries.append(
                f"{sector} is projected to lose {round(abs(job_change_pct))}% of jobs "
                f"({round(abs(w1_pct))}% in months 0-6, {round(abs(w2_pct))}% in 6-12, "
                f"{round(abs(w3_pct))}% in 12-18)."
            )
        else:
            # Sector is GAINING jobs (fewer exits = people staying / demand surge)
            # Peak month = window with most negative excess (= most retention)
            if w1 < w2 and w1 < w3:
                peak_month = 3   # peak in window 1
            elif w2 < w3:
                peak_month = 9   # peak in window 2
            else:
                peak_month = 15  # peak in window 3

            predictions[sector] = {
                "job_change_pct": round(abs(job_change_pct)),
                "peak_month": peak_month,
                **window_trajectory,
            }
            sector_summaries.append(
                f"{sector} sees a {round(abs(job_change_pct))}% demand increase, "
                f"peaking around month {peak_month}."
            )

    if not predictions:
        continue

    # Sort by impact magnitude — top sectors first
    sorted_sectors = sorted(predictions.items(),
        key=lambda x: x[1].get('job_loss_pct', x[1].get('job_change_pct', 0)),
        reverse=True)
    predictions = dict(sorted_sectors[:8])  # top 8 sectors

    # Build narrative text
    dtype_lower = normalize_disaster_type(dtype)
    region = get_region_name(fips)
    text = (
        f"Following a {dtype.lower()} event in {region} (FIPS {fips}), "
        f"our model predicts the following workforce impacts. "
        + " ".join(sector_summaries[:6])
    )

    results.append({
        "id": f"{dtype_lower}_{fips}",
        "disaster_type": dtype_lower,
        "fips_code": fips,
        "region": region,
        "text": text,
        "predictions": predictions,
    })

print(f"Generated {len(results)} scenario predictions")

# Save
with open("../backend/data/model_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to backend/data/model_predictions.json")


# ══════════════════════════════════════════════════════════
# 2. REGIONAL ANALYTICS CSV
# ══════════════════════════════════════════════════════════
print("\nGenerating regional_analytics.csv...")

monthly_cols = [f'exits_month_{i}' for i in range(1, 7)]

analytics_rows = []

for _, scenario in scenarios.iterrows():
    dtype = scenario['incidentType']
    fips = scenario['fips_code']
    state = scenario['state']

    mask = (df['incidentType'] == dtype) & (df['fips_code'] == fips)
    scenario_df = df[mask]

    for sector in scenario_df['sector'].unique():
        sector_data = scenario_df[scenario_df['sector'] == sector]

        baseline = sector_data['baseline_exits'].mean()
        if baseline < 1:
            continue

        # Build month-by-month from our 3 windows × 6 months each
        for _, row in sector_data.iterrows():
            window_num = int(row['window'].replace('window_', ''))
            window_offset = (window_num - 1) * 6  # 0, 6, or 12

            for m in range(1, 7):
                col = f'exits_month_{m}'
                if col not in row.index:
                    continue

                month_exits = row[col]
                month_offset = window_offset + m - 1  # 0-17

                # Calculate change relative to monthly baseline (baseline / 6)
                monthly_baseline = baseline / 6
                if monthly_baseline > 0:
                    change_count = month_exits - monthly_baseline
                    change_pct = (change_count / monthly_baseline) * 100
                else:
                    change_count = month_exits
                    change_pct = 0

                # Recovery rate: 1.0 if at or below baseline, 0.0 if at peak excess
                peak_excess = sector_data['excess_exits'].abs().max()
                if peak_excess > 0:
                    recovery = max(0, 1 - abs(change_count) / (peak_excess / 6))
                else:
                    recovery = 1.0

                analytics_rows.append({
                    'fips_code': fips,
                    'state': state,
                    'industry_group': sector,
                    'disaster_type': normalize_disaster_type(dtype),
                    'month_offset': month_offset,
                    'job_change_count': round(change_count, 1),
                    'job_change_pct': round(change_pct, 1),
                    'recovery_rate': round(min(1.0, max(0.0, recovery)), 2),
                })

analytics_df = pd.DataFrame(analytics_rows)

# Aggregate duplicate month_offset entries (multiple disasters of same type in same FIPS)
analytics_df = analytics_df.groupby(
    ['fips_code', 'state', 'industry_group', 'disaster_type', 'month_offset']
).agg(
    job_change_count=('job_change_count', 'mean'),
    job_change_pct=('job_change_pct', 'mean'),
    recovery_rate=('recovery_rate', 'mean'),
).reset_index()

analytics_df['fips_code'] = analytics_df['fips_code'].astype(str).str.zfill(5)
analytics_df = analytics_df.sort_values(['fips_code', 'disaster_type', 'industry_group', 'month_offset'])

print(f"Generated {len(analytics_df)} time-series rows")
print(f"Unique scenarios: {analytics_df.groupby(['fips_code', 'disaster_type']).ngroups}")

analytics_df.to_csv("../backend/data/processed/regional_analytics.csv", index=False)
print(f"Saved to backend/data/processed/regional_analytics.csv")


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"STEP 6 COMPLETE — Backend data exported")
print(f"{'='*60}")
print(f"  model_predictions.json: {len(results)} scenarios")
print(f"  regional_analytics.csv: {len(analytics_df)} time-series rows")
print(f"  Disaster types covered: {analytics_df['disaster_type'].nunique()}")
print(f"  FIPS codes covered: {analytics_df['fips_code'].nunique()}")
print(f"  Sectors covered: {analytics_df['industry_group'].nunique()}")
