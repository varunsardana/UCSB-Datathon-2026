"""
STEP 3: Merge FEMA disasters with job endings (3-window structure)

What this does:
- Loads both clean datasets (fema_clean.csv + jobs_clean.csv)
- For each disaster: counts job endings in 3 post-disaster windows:
    Window 1 (0-6 months):   Immediate shock
    Window 2 (6-12 months):  Recovery phase
    Window 3 (12-18 months): Normalization
- Baseline = average of same period 2 years ago and 3 years ago
  (avoids overlap with post-disaster windows)
- Groups by industry so we get per-industry displacement numbers
- excess_exits = post_disaster_exits - avg_baseline_exits

Output: one row per disaster × industry × window
"""

import pandas as pd
import numpy as np

fema = pd.read_csv("fema_clean.csv", parse_dates=['incidentBeginDate', 'incidentEndDate'])
jobs = pd.read_csv("jobs_clean.csv", parse_dates=['started_at', 'ended_at'])

print(f"FEMA disasters: {len(fema)} rows ({fema['disasterNumber'].nunique()} unique)")
print(f"Job endings: {len(jobs)} rows")


fema_fips = set(fema['fips_code'].unique())
jobs_fips = set(jobs['fips_code'].unique())
overlap = fema_fips & jobs_fips

print(f"\nFEMA FIPS codes: {len(fema_fips)}")
print(f"Jobs FIPS codes: {len(jobs_fips)}")
print(f"Overlapping FIPS codes: {len(overlap)}")

# Filter FEMA to only disasters in counties we have job data for
fema = fema[fema['fips_code'].isin(overlap)]
print(f"FEMA rows after filtering to overlap: {len(fema)}")
print(f"Unique disasters with job data: {fema['disasterNumber'].nunique()}")

# 3-window structure with averaged baselines (2yr + 3yr ago)
# Baseline uses 2 and 3 years prior to avoid overlap with post-disaster windows
print("\nProcessing disasters (3 windows per disaster)...")

# Define the 3 windows: (name, start_offset_months, end_offset_months)
windows = [
    ('window_1', 0, 6),     # 0-6 months: immediate shock
    ('window_2', 6, 12),    # 6-12 months: recovery
    ('window_3', 12, 18),   # 12-18 months: normalization
]

results = []

unique_disasters = fema.drop_duplicates(subset=['disasterNumber', 'fips_code'])

for idx, disaster in unique_disasters.iterrows():
    fips = disaster['fips_code']
    start = disaster['incidentBeginDate']

    if pd.isna(start):
        continue

    fips_jobs = jobs[jobs['fips_code'] == fips]

    if len(fips_jobs) == 0:
        continue

    for window_name, offset_start, offset_end in windows:
        # Post-disaster window
        post_start = start + pd.DateOffset(months=offset_start)
        post_end = start + pd.DateOffset(months=offset_end)

        # Baseline 1: same window, 2 years before disaster
        b1_start = post_start - pd.DateOffset(years=2)
        b1_end = post_end - pd.DateOffset(years=2)

        # Baseline 2: same window, 3 years before disaster
        b2_start = post_start - pd.DateOffset(years=3)
        b2_end = post_end - pd.DateOffset(years=3)

        # Count post-disaster exits
        post_jobs = fips_jobs[
            (fips_jobs['ended_at'] >= post_start) &
            (fips_jobs['ended_at'] < post_end)
        ]

        # Count baseline exits (2 years ago)
        baseline1_jobs = fips_jobs[
            (fips_jobs['ended_at'] >= b1_start) &
            (fips_jobs['ended_at'] < b1_end)
        ]

        # Count baseline exits (3 years ago)
        baseline2_jobs = fips_jobs[
            (fips_jobs['ended_at'] >= b2_start) &
            (fips_jobs['ended_at'] < b2_end)
        ]

        post_counts = post_jobs.groupby('industry').size().to_dict()
        b1_counts = baseline1_jobs.groupby('industry').size().to_dict()
        b2_counts = baseline2_jobs.groupby('industry').size().to_dict()

        all_industries = set(list(post_counts.keys()) + list(b1_counts.keys()) + list(b2_counts.keys()))

        for industry in all_industries:
            post_count = post_counts.get(industry, 0)
            b1_count = b1_counts.get(industry, 0)
            b2_count = b2_counts.get(industry, 0)
            avg_baseline = (b1_count + b2_count) / 2

            results.append({
                'disasterNumber': disaster['disasterNumber'],
                'incidentType': disaster['incidentType'],
                'declarationTitle': disaster['declarationTitle'],
                'fips_code': fips,
                'state': disaster['state'],
                'designatedArea': disaster['designatedArea'],
                'incidentBeginDate': start,
                'window': window_name,
                'industry': industry,
                'post_disaster_exits': post_count,
                'baseline_yr2_exits': b1_count,
                'baseline_yr3_exits': b2_count,
                'baseline_exits': avg_baseline,
                'excess_exits': post_count - avg_baseline,
            })


df = pd.DataFrame(results)
print(f"\nTotal rows (disaster × industry × window): {len(df)}")

if len(df) > 0:
    print(f"\n{'='*50}")
    print(f"MERGED DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Unique disasters matched: {df['disasterNumber'].nunique()}")
    print(f"Unique industries: {df['industry'].nunique()}")
    print(f"Unique FIPS codes: {df['fips_code'].nunique()}")
    print(f"Windows: {df['window'].unique().tolist()}")
    print(f"Rows per window: {df['window'].value_counts().to_dict()}")

    print(f"\nAverage excess exits by window:")
    print(df.groupby('window')['excess_exits'].mean())

    print(f"\nAverage excess exits by disaster type:")
    print(df.groupby('incidentType')['excess_exits'].mean().sort_values(ascending=False).head(10))

    print(f"\nTop 10 disaster-industry-window combos by excess exits:")
    top = df.nlargest(10, 'excess_exits')[['declarationTitle', 'industry', 'window', 'post_disaster_exits', 'baseline_exits', 'excess_exits']]
    print(top.to_string())

    df.to_csv("merged_disaster_jobs.csv", index=False)
    print(f"\nSaved to merged_disaster_jobs.csv ({len(df)} rows)")
else:
    print("WARNING: No matches found. Check that FIPS codes and dates overlap.")
