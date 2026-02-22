"""
STEP 4: Feature engineering (optimized — no row-by-row loops)

What this does:
- Takes the merged disaster-jobs dataset (9,639 rows)
- Converts everything into numbers the ML model can learn from
- Creates meaningful derived features that capture real-world patterns
- Saves the final feature matrix + metadata for decoding predictions later

Feature categories:
  1. Disaster type (one-hot: is it a fire? hurricane? flood?)
  2. Temporal (month, quarter, year — captures seasonality + economic trends)
  3. Scale features (how big is this industry/region normally?)
  4. Interaction features (combines disaster type with other signals)
  5. Disaster duration (how long did it last?)
  6. Recent disaster history (has this county been hit before?)
  7. Industry sector grouping (broad categories for better generalization)
  8. Monthly exit breakdown (month-by-month displacement for time series)
  9. Peak exit timing (immediate vs delayed impact)
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. Load all datasets we need
# ============================================================
df = pd.read_csv("merged_disaster_jobs.csv", parse_dates=['incidentBeginDate'])
fema = pd.read_csv("fema_clean.csv", parse_dates=['incidentBeginDate', 'incidentEndDate'])
jobs = pd.read_csv("jobs_clean.csv", parse_dates=['started_at', 'ended_at'])
print(f"Loaded {len(df)} rows")

# ============================================================
# 2. Disaster type — one-hot encoding
# ============================================================
disaster_dummies = pd.get_dummies(df['incidentType'], prefix='disaster')
df = pd.concat([df, disaster_dummies], axis=1)
print(f"Added {len(disaster_dummies.columns)} disaster type columns")

# ============================================================
# 3. Temporal features
# ============================================================
df['disaster_month'] = df['incidentBeginDate'].dt.month
df['disaster_quarter'] = df['incidentBeginDate'].dt.quarter
df['disaster_year'] = df['incidentBeginDate'].dt.year

# Cyclical encoding so the model knows Jan and Dec are neighbors
df['month_sin'] = np.sin(2 * np.pi * df['disaster_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['disaster_month'] / 12)
print("Added temporal features (month, quarter, year, cyclical month)")

# ============================================================
# 4. Disaster duration
# ============================================================
fema_dates = fema[['disasterNumber', 'fips_code', 'incidentEndDate']].drop_duplicates(
    subset=['disasterNumber', 'fips_code']
)
df = df.merge(fema_dates, on=['disasterNumber', 'fips_code'], how='left')

df['disaster_duration_days'] = (df['incidentEndDate'] - df['incidentBeginDate']).dt.days
median_duration = df['disaster_duration_days'].median()
df['disaster_duration_days'] = df['disaster_duration_days'].fillna(median_duration).clip(lower=0)
df['log_duration'] = np.log1p(df['disaster_duration_days'])

print(f"Added disaster duration (median: {median_duration:.0f} days)")

# ============================================================
# 5. Recent disaster history per county (VECTORIZED)
# ============================================================
# Instead of looping row by row, we:
#   1. Get unique disaster-fips combos from our data
#   2. For each, count prior disasters using a merge + filter approach

print("Computing disaster history (vectorized)...")

# Get unique disaster-fips-date combos (no need to repeat for every industry)
disaster_fips = df[['disasterNumber', 'fips_code', 'incidentBeginDate']].drop_duplicates()

# Get all FEMA events with just what we need
fema_slim = fema[['disasterNumber', 'fips_code', 'incidentBeginDate']].drop_duplicates()

# Self-join: for each disaster, find all other disasters in the same FIPS
# Then filter to ones that happened in the 2 years before
merged = disaster_fips.merge(fema_slim, on='fips_code', suffixes=('', '_prior'))

# Keep only prior disasters (different disaster, within 2 years before)
merged = merged[
    (merged['disasterNumber_prior'] != merged['disasterNumber']) &
    (merged['incidentBeginDate_prior'] < merged['incidentBeginDate']) &
    (merged['incidentBeginDate_prior'] >= merged['incidentBeginDate'] - pd.DateOffset(years=2))
]

# Count unique prior disasters per disaster-fips combo
prior_counts = (merged.groupby(['disasterNumber', 'fips_code'])['disasterNumber_prior']
    .nunique()
    .reset_index()
    .rename(columns={'disasterNumber_prior': 'prior_disasters_2yr'})
)

# Merge back — disasters with no prior hits get 0
df = df.merge(prior_counts, on=['disasterNumber', 'fips_code'], how='left')
df['prior_disasters_2yr'] = df['prior_disasters_2yr'].fillna(0).astype(int)
df['recently_hit'] = (df['prior_disasters_2yr'] > 0).astype(int)

print(f"Added disaster history (avg prior disasters in 2yr: {df['prior_disasters_2yr'].mean():.1f})")

# ============================================================
# 6. Industry sector grouping
# ============================================================
sector_map = {
    # Tech
    'Computer Software': 'Tech', 'Internet': 'Tech',
    'Information Technology and Services': 'Tech',
    'Computer Hardware': 'Tech', 'Semiconductors': 'Tech',
    'Computer Networking': 'Tech', 'Telecommunications': 'Tech',
    'Computer & Network Security': 'Tech', 'Wireless': 'Tech',
    'Computer Games': 'Tech', 'Nanotechnology': 'Tech',

    # Healthcare
    'Hospital & Health Care': 'Healthcare', 'Medical Devices': 'Healthcare',
    'Pharmaceuticals': 'Healthcare', 'Biotechnology': 'Healthcare',
    'Mental Health Care': 'Healthcare', 'Health, Wellness and Fitness': 'Healthcare',
    'Medical Practice': 'Healthcare', 'Veterinary': 'Healthcare',
    'Alternative Medicine': 'Healthcare',

    # Education
    'Higher Education': 'Education', 'Education Management': 'Education',
    'E-Learning': 'Education', 'Primary/Secondary Education': 'Education',
    'Professional Training & Coaching': 'Education',

    # Finance
    'Financial Services': 'Finance', 'Banking': 'Finance',
    'Insurance': 'Finance', 'Investment Banking': 'Finance',
    'Investment Management': 'Finance', 'Venture Capital & Private Equity': 'Finance',
    'Capital Markets': 'Finance', 'Accounting': 'Finance',

    # Government & Nonprofit
    'Government Administration': 'Government', 'Military': 'Government',
    'Non-Profit Organization Management': 'Nonprofit',
    'Civic & Social Organization': 'Nonprofit', 'Philanthropy': 'Nonprofit',

    # Retail & Hospitality
    'Retail': 'Retail & Hospitality', 'Restaurants': 'Retail & Hospitality',
    'Food & Beverages': 'Retail & Hospitality', 'Hospitality': 'Retail & Hospitality',
    'Leisure, Travel & Tourism': 'Retail & Hospitality',
    'Consumer Goods': 'Retail & Hospitality', 'Apparel & Fashion': 'Retail & Hospitality',

    # Construction & Real Estate
    'Construction': 'Construction & Real Estate', 'Real Estate': 'Construction & Real Estate',
    'Architecture & Planning': 'Construction & Real Estate',
    'Building Materials': 'Construction & Real Estate',
    'Civil Engineering': 'Construction & Real Estate',

    # Legal
    'Law Practice': 'Legal', 'Legal Services': 'Legal',

    # Media & Entertainment
    'Entertainment': 'Media & Entertainment', 'Media Production': 'Media & Entertainment',
    'Motion Pictures and Film': 'Media & Entertainment',
    'Broadcast Media': 'Media & Entertainment', 'Music': 'Media & Entertainment',
    'Publishing': 'Media & Entertainment', 'Online Media': 'Media & Entertainment',
    'Newspapers': 'Media & Entertainment',

    # Manufacturing
    'Automotive': 'Manufacturing', 'Aviation & Aerospace': 'Manufacturing',
    'Mechanical or Industrial Engineering': 'Manufacturing',
    'Electrical/Electronic Manufacturing': 'Manufacturing',
    'Industrial Automation': 'Manufacturing',

    # Energy
    'Oil & Energy': 'Energy', 'Renewables & Environment': 'Energy',
    'Utilities': 'Energy', 'Mining & Metals': 'Energy',

    # Marketing & Creative
    'Marketing and Advertising': 'Marketing & Creative',
    'Design': 'Marketing & Creative', 'Graphic Design': 'Marketing & Creative',
    'Public Relations and Communications': 'Marketing & Creative',

    # Research
    'Research': 'Research', 'Think Tanks': 'Research',

    # Transportation & Logistics
    'Transportation/Trucking/Railroad': 'Transportation',
    'Logistics and Supply Chain': 'Transportation',
    'Airlines/Aviation': 'Transportation', 'Maritime': 'Transportation',

    # Management Consulting
    'Management Consulting': 'Consulting',
    'Human Resources': 'Consulting', 'Staffing and Recruiting': 'Consulting',
}

df['sector'] = df['industry'].map(sector_map).fillna('Other')

sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
df = pd.concat([df, sector_dummies], axis=1)
df['sector_code'] = df['sector'].astype('category').cat.codes

print(f"Grouped {df['industry'].nunique()} industries into {df['sector'].nunique()} sectors")

# ============================================================
# 7. Monthly exit breakdown (VECTORIZED)
# ============================================================
# Instead of looping 9,639 × 6 = 57,834 times, we:
#   1. Add a month-offset column to jobs
#   2. Merge jobs with disasters on fips + industry
#   3. Calculate which month bucket each job ending falls into
#   4. Pivot to get month columns

print("Computing monthly exit breakdown (vectorized)...")

# Get unique disaster-fips-industry combos with their start dates
disaster_keys = df[['disasterNumber', 'fips_code', 'industry', 'incidentBeginDate']].copy()

# Merge jobs with disaster keys on fips + industry
jobs_with_disaster = jobs.merge(
    disaster_keys,
    on=['fips_code', 'industry'],
    how='inner'
)

# Calculate months between disaster start and job end
jobs_with_disaster['months_after'] = (
    (jobs_with_disaster['ended_at'].dt.year - jobs_with_disaster['incidentBeginDate'].dt.year) * 12 +
    (jobs_with_disaster['ended_at'].dt.month - jobs_with_disaster['incidentBeginDate'].dt.month)
)

# Keep only months 0-5 (= month 1 through month 6 post-disaster)
jobs_with_disaster = jobs_with_disaster[
    (jobs_with_disaster['months_after'] >= 0) &
    (jobs_with_disaster['months_after'] < 6)
]

# Map to month buckets (0→month_1, 1→month_2, etc.)
jobs_with_disaster['month_bucket'] = jobs_with_disaster['months_after'] + 1

# Count exits per disaster × fips × industry × month
monthly_counts = (jobs_with_disaster
    .groupby(['disasterNumber', 'fips_code', 'industry', 'month_bucket'])
    .size()
    .reset_index(name='exits')
)

# Pivot so each month becomes a column
monthly_pivot = monthly_counts.pivot_table(
    index=['disasterNumber', 'fips_code', 'industry'],
    columns='month_bucket',
    values='exits',
    fill_value=0
).reset_index()

# Rename columns to exits_month_1 through exits_month_6
monthly_pivot.columns = [
    f'exits_month_{int(c)}' if isinstance(c, (int, float)) else c
    for c in monthly_pivot.columns
]

# Make sure all 6 month columns exist (some disasters may not have exits in every month)
for i in range(1, 7):
    col = f'exits_month_{i}'
    if col not in monthly_pivot.columns:
        monthly_pivot[col] = 0

# Merge back into main dataframe
df = df.merge(
    monthly_pivot[['disasterNumber', 'fips_code', 'industry'] + [f'exits_month_{i}' for i in range(1, 7)]],
    on=['disasterNumber', 'fips_code', 'industry'],
    how='left'
)

# Fill NaN with 0 (disasters with no exits in a given month)
for i in range(1, 7):
    df[f'exits_month_{i}'] = df[f'exits_month_{i}'].fillna(0).astype(int)

print("Added monthly exit columns (exits_month_1 through exits_month_6)")

# ============================================================
# 8. Peak exit timing
# ============================================================
monthly_cols_temp = [f'exits_month_{i}' for i in range(1, 7)]
df['peak_exit_month'] = df[monthly_cols_temp].idxmax(axis=1).str.extract(r'(\d)').astype(float)
df['peak_exits'] = df[monthly_cols_temp].max(axis=1)
# Was the damage immediate (month 1-2) or delayed (month 3+)?
df['immediate_impact'] = (df['peak_exit_month'] <= 2).astype(int)

# Handle edge case: if all months are 0, peak_exit_month is meaningless
all_zero = df[monthly_cols_temp].sum(axis=1) == 0
df.loc[all_zero, 'peak_exit_month'] = 0
df.loc[all_zero, 'immediate_impact'] = 0

print("Added peak exit features (peak_exit_month, peak_exits, immediate_impact)")

# ============================================================
# 9. Scale features
# ============================================================
df['exit_ratio'] = np.where(
    df['baseline_exits'] > 0,
    df['post_disaster_exits'] / df['baseline_exits'],
    df['post_disaster_exits']
)

df['log_baseline'] = np.log1p(df['baseline_exits'])

median_baseline = df['baseline_exits'].median()
df['high_turnover'] = (df['baseline_exits'] > median_baseline).astype(int)

print("Added scale features (exit_ratio, log_baseline, high_turnover)")

# ============================================================
# 10. Industry encoding + vulnerability
# ============================================================
df['industry_code'] = df['industry'].astype('category').cat.codes

industry_vuln = df.groupby('industry')['excess_exits'].mean().to_dict()
df['industry_vulnerability'] = df['industry'].map(industry_vuln)

sector_vuln = df.groupby('sector')['excess_exits'].mean().to_dict()
df['sector_vulnerability'] = df['sector'].map(sector_vuln)

industry_freq = df['industry'].value_counts().to_dict()
df['industry_frequency'] = df['industry'].map(industry_freq)

print(f"Encoded {df['industry'].nunique()} industries + vulnerability scores")

# ============================================================
# 11. State / region encoding
# ============================================================
df['state_code'] = df['state'].astype('category').cat.codes

state_disaster_freq = df.groupby('state')['disasterNumber'].nunique().to_dict()
df['state_disaster_frequency'] = df['state'].map(state_disaster_freq)

print(f"Encoded {df['state'].nunique()} states + disaster frequency")

# ============================================================
# 12. Interaction features
# ============================================================
df['disaster_x_baseline'] = df['log_baseline'] * df[disaster_dummies.columns].max(axis=1)
df['vuln_x_baseline'] = df['industry_vulnerability'] * df['log_baseline']
df['duration_x_baseline'] = df['log_duration'] * df['log_baseline']
df['prior_x_vulnerability'] = df['prior_disasters_2yr'] * df['industry_vulnerability']

print("Added interaction features")

# ============================================================
# 13. Define feature columns vs metadata columns
# ============================================================
metadata_cols = [
    'disasterNumber', 'incidentType', 'declarationTitle', 'fips_code',
    'state', 'designatedArea', 'incidentBeginDate', 'incidentEndDate',
    'industry', 'sector'
]

target_col = 'excess_exits'
monthly_cols = [f'exits_month_{i}' for i in range(1, 7)]

feature_cols = [col for col in df.columns
    if col not in metadata_cols + [target_col, 'post_disaster_exits', 'end_year_month'] + monthly_cols
]

print(f"\n{'='*50}")
print(f"FEATURE MATRIX SUMMARY")
print(f"{'='*50}")
print(f"Total rows: {len(df)}")
print(f"Metadata columns: {len(metadata_cols)}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Monthly target columns: {len(monthly_cols)}")
print(f"Primary target: {target_col}")

print(f"\nFeature columns:")
for col in feature_cols:
    print(f"  {col} — dtype: {df[col].dtype}, nulls: {df[col].isna().sum()}")

print(f"\nTarget variable (excess_exits):")
print(f"  Mean: {df[target_col].mean():.2f}")
print(f"  Std:  {df[target_col].std():.2f}")
print(f"  Min:  {df[target_col].min()}")
print(f"  Max:  {df[target_col].max()}")

print(f"\nMonthly exit breakdown averages:")
for col in monthly_cols:
    print(f"  {col}: {df[col].mean():.2f}")

# ============================================================
# 14. Save everything
# ============================================================
df.to_csv("features.csv", index=False)
print(f"\nSaved to data/features.csv ({len(df)} rows, {len(df.columns)} columns)")

with open("feature_columns.txt", "w") as f:
    f.write("\n".join(feature_cols))
print("Saved feature column list to data/feature_columns.txt")

with open("monthly_columns.txt", "w") as f:
    f.write("\n".join(monthly_cols))
print("Saved monthly column list to data/monthly_columns.txt")

industry_mapping = df[['industry_code', 'industry']].drop_duplicates().sort_values('industry_code')
industry_mapping.to_csv("industry_map.csv", index=False)

sector_mapping = df[['sector_code', 'sector']].drop_duplicates().sort_values('sector_code')
sector_mapping.to_csv("sector_map.csv", index=False)
print("Saved industry + sector mappings")
