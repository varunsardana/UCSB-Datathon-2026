"""
STEP 2: Clean the job/employment data

What this does:
- Loads the 875MB JSON of employment records (75,139 people)
- Each person has multiple jobs in their history
- We extract every job that has an ended_at date (= person left/lost that job)
- Pull out: fips_code, industry, job title, start date, end date
- Drop rows missing fips_code or industry (can't use them)
- Save a clean CSV that's small enough to work with easily

Why we care about ended_at:
- If a job has no ended_at → person still works there (not displaced)
- If a job has ended_at → person left that job
- We'll later check if that end date lines up with a disaster in that FIPS code
"""

import json
import pandas as pd

print("Loading job data (this takes a minute)...")
with open("../live_data_persons_history_combined.json") as f:
    data = json.load(f)
print(f"Total people: {len(data)}")

job_endings = []

for person in data:
    for job in person.get('jobs', []):
        # Skip jobs that are still active (no end date)
        if not job.get('ended_at'):
            continue

        # Skip jobs with no location data (can't match to disasters)
        if not job.get('location_details'):
            continue

        # Skip jobs with no company info (can't get industry)
        if not job.get('company'):
            continue

        job_endings.append({
            'fips_code': job['location_details'].get('fips_code'),
            'industry': job['company'].get('industry'),
            'company_name': job['company'].get('name'),
            'title': job.get('title'),
            'started_at': job.get('started_at'),
            'ended_at': job.get('ended_at'),
            'locality': job['location_details'].get('locality'),
            'region': job['location_details'].get('region'),
        })

print(f"Total job endings extracted: {len(job_endings)}")

df = pd.DataFrame(job_endings)

before = len(df)
df = df.dropna(subset=['fips_code', 'industry'])
print(f"Dropped {before - len(df)} rows with missing fips/industry")
print(f"Remaining: {len(df)} job endings")


df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

# Add a year-month column for ease later
df['end_year_month'] = df['ended_at'].dt.to_period('M')

print(f"\n{'='*50}")
print(f"CLEAN JOB DATA SUMMARY")
print(f"{'='*50}")
print(f"Total job endings: {len(df)}")
print(f"Unique FIPS codes: {df['fips_code'].nunique()}")
print(f"Date range: {df['ended_at'].min()} to {df['ended_at'].max()}")
print(f"\nTop 15 industries by job endings:")
print(df['industry'].value_counts().head(15))
print(f"\nTop 10 locations by job endings:")
print(df[['locality', 'region']].value_counts().head(10))

df.to_csv("jobs_clean.csv", index=False)
print(f"\nSaved to data/jobs_clean.csv ({len(df)} rows)")
