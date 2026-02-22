"""
STEP 3: Merge FEMA disasters with job endings

What this does:
- Loads both clean datasets (fema_clean.csv + jobs_clean.csv)
- For each disaster: finds all job endings in the same FIPS code
  in the 6 months AFTER the disaster started
- Also counts job endings in the same period ONE YEAR PRIOR (baseline)
- Groups by industry so we get per-industry displacement numbers
- The difference between post-disaster and baseline = disaster-driven job loss

Output: one row per disaster per industry with monthly job ending counts
This is the training data for the ML model.
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

# For every disaster:
#   - Look at the 6 months AFTER the disaster started (post-disaster window)
#   - Count how many jobs ended per industry per month in that FIPS code
#   - Do the same for the SAME 6-month window ONE YEAR EARLIER (baseline)
#   - Baseline tells us "normal" turnover so we can subtract it out
print("\nProcessing disasters (this may take a minute)...")

results = []

unique_disasters = fema.drop_duplicates(subset=['disasterNumber', 'fips_code'])

for idx, disaster in unique_disasters.iterrows():
    fips = disaster['fips_code']
    start = disaster['incidentBeginDate']

    # Skip if start date is missing
    if pd.isna(start):
        continue

    # Define time windows
    post_start = start
    post_end = start + pd.DateOffset(months=6)
    baseline_start = start - pd.DateOffset(months=18)  # same period, one year earlier
    baseline_end = start - pd.DateOffset(months=12)

    # Get jobs in this FIPS code
    fips_jobs = jobs[jobs['fips_code'] == fips]

    if len(fips_jobs) == 0:
        continue

    # Post-disaster job endings (6 months after disaster)
    post_jobs = fips_jobs[
        (fips_jobs['ended_at'] >= post_start) &
        (fips_jobs['ended_at'] < post_end)
    ]

    # Baseline job endings (same 6-month window, one year earlier)
    baseline_jobs = fips_jobs[
        (fips_jobs['ended_at'] >= baseline_start) &
        (fips_jobs['ended_at'] < baseline_end)
    ]

    # Count per industry
    post_counts = post_jobs.groupby('industry').size().to_dict()
    baseline_counts = baseline_jobs.groupby('industry').size().to_dict()

    # Get all industries from both periods
    all_industries = set(list(post_counts.keys()) + list(baseline_counts.keys()))

    for industry in all_industries:
        post_count = post_counts.get(industry, 0)
        baseline_count = baseline_counts.get(industry, 0)

        results.append({
            'disasterNumber': disaster['disasterNumber'],
            'incidentType': disaster['incidentType'],
            'declarationTitle': disaster['declarationTitle'],
            'fips_code': fips,
            'state': disaster['state'],
            'designatedArea': disaster['designatedArea'],
            'incidentBeginDate': start,
            'industry': industry,
            'post_disaster_exits': post_count,
            'baseline_exits': baseline_count,
            'excess_exits': post_count - baseline_count,  # positive = more exits than normal
        })


df = pd.DataFrame(results)
print(f"\nTotal rows (disaster × industry): {len(df)}")

if len(df) > 0:
    print(f"\n{'='*50}")
    print(f"MERGED DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Unique disasters matched: {df['disasterNumber'].nunique()}")
    print(f"Unique industries: {df['industry'].nunique()}")
    print(f"Unique FIPS codes: {df['fips_code'].nunique()}")

    print(f"\nAverage excess exits by disaster type:")
    print(df.groupby('incidentType')['excess_exits'].mean().sort_values(ascending=False).head(10))

    print(f"\nTop 10 disaster-industry pairs by excess exits:")
    top = df.nlargest(10, 'excess_exits')[['declarationTitle', 'industry', 'post_disaster_exits', 'baseline_exits', 'excess_exits']]
    print(top.to_string())

 
    df.to_csv("merged_disaster_jobs.csv", index=False)
    print(f"\nSaved to data/merged_disaster_jobs.csv ({len(df)} rows)")
else:
    print("WARNING: No matches found. Check that FIPS codes and dates overlap.")
