"""
STEP 1: Load and clean FEMA disaster data

What this does:
- Loads the raw FEMA CSV (69,635 rows of every disaster declaration in US history)
- Keeps only the columns we need
- Builds a full FIPS code (the key we'll use to join with job data)
- Filters to only major disasters (type "DR") — these are the ones big enough to affect employment
- Filters out rows with missing FIPS codes (can't join without them)
- Saves a clean version to data/fema_clean.csv
"""

import pandas as pd
df = pd.read_csv("../DisasterDeclarationsSummaries.csv")

cols_to_keep = [
    'disasterNumber',       # unique ID for the disaster
    'state',                # state abbreviation (CA, FL, TX...)
    'declarationType',      # DR=major disaster, EM=emergency, FM=fire mgmt
    'incidentType',         # Fire, Hurricane, Flood, Severe Storm, etc.
    'declarationTitle',     # human readable name ("WOOLSEY FIRE")
    'incidentBeginDate',    # when the disaster actually started
    'incidentEndDate',      # when it ended
    'fipsStateCode',        # first part of FIPS 
    'fipsCountyCode',       # second part of FIPS
    'designatedArea',       # human readable county name
]
df = df[cols_to_keep]


df['fipsStateCode'] = df['fipsStateCode'].astype(str).str.zfill(2)   # pad to 2 digits
df['fipsCountyCode'] = df['fipsCountyCode'].astype(str).str.zfill(3)  # pad to 3 digits
df['fips_code'] = df['fipsStateCode'] + df['fipsCountyCode']
print(f"\nSample FIPS codes: {df['fips_code'].head(5).tolist()}")

print(f"\nDeclaration type breakdown:")

df = df[df['declarationType'] == 'DR']
print(f"\nAfter filtering to major disasters (DR): {len(df)} rows")

bad_fips = df['fipsCountyCode'] == '000'
print(f"Rows with statewide (000) FIPS: {bad_fips.sum()}")
df = df[~bad_fips]
print(f"After dropping bad FIPS: {len(df)} rows")

df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'])
df['incidentEndDate'] = pd.to_datetime(df['incidentEndDate'])


print(f"\n{'='*50}")
print(f"CLEAN FEMA DATA SUMMARY")
print(f"{'='*50}")
print(f"Total disaster-county rows: {len(df)}")
print(f"Unique disasters: {df['disasterNumber'].nunique()}")
print(f"Date range: {df['incidentBeginDate'].min()} to {df['incidentBeginDate'].max()}")
print(f"\nDisaster types:")
print(df['incidentType'].value_counts())
print(f"\nTop 10 states by disaster count:")
print(df['state'].value_counts().head(10))

df.to_csv("fema_clean.csv", index=False)
print(f"\nSaved to data/fema_clean.csv ({len(df)} rows)")
