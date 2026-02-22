"""
FEMA Disaster Time Series Analysis - Phase 1: Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("FEMA DISASTER TIME SERIES ANALYSIS - PHASE 1: EDA")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('/Users/alizasamad/Downloads/projects/datathon26/UCSB-Datathon-2026/data/fema_clean.csv')
print(f"✓ Loaded {len(df):,} records")
print(f"✓ Shape: {df.shape}")

# Display basic info
print("\n2. DATASET OVERVIEW:")
print(df.head(10))
print("\n" + "="*80)
print("Column Info:")
print(df.info())

# Convert dates
print("\n3. DATE PROCESSING...")
df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'])
df['incidentEndDate'] = pd.to_datetime(df['incidentEndDate'])

# Extract temporal features
df['year'] = df['incidentBeginDate'].dt.year
df['month'] = df['incidentBeginDate'].dt.month
df['quarter'] = df['incidentBeginDate'].dt.quarter
df['year_month'] = df['incidentBeginDate'].dt.to_period('M')

print(f"✓ Date range: {df['incidentBeginDate'].min()} to {df['incidentBeginDate'].max()}")
print(f"✓ Years of data: {df['year'].max() - df['year'].min() + 1}")

# Check for missing values
print("\n4. DATA QUALITY:")
print("Missing values:")
print(df.isnull().sum())

# Unique values
print("\n5. UNIQUE VALUES:")
print(f"States: {df['state'].nunique()}")
print(f"Counties (fips_code): {df['fips_code'].nunique()}")
print(f"Disaster Types: {df['incidentType'].nunique()}")
print(f"Disaster Numbers: {df['disasterNumber'].nunique()}")

# Display disaster types
print("\n6. DISASTER TYPES BREAKDOWN:")
disaster_counts = df['incidentType'].value_counts()
print(disaster_counts)

# Save for reference
disaster_types_df = disaster_counts.reset_index()
disaster_types_df.columns = ['Disaster_Type', 'Count']
print(f"\n✓ Found {len(disaster_types_df)} unique disaster types")

# State analysis
print("\n7. TOP 10 STATES BY DISASTER COUNT:")
state_counts = df['state'].value_counts().head(10)
print(state_counts)

# Temporal trends
print("\n8. YEARLY DISASTER TRENDS:")
yearly_counts = df.groupby('year').size()
print(yearly_counts)

# Monthly seasonality
print("\n9. MONTHLY PATTERNS (All disasters combined):")
monthly_counts = df.groupby('month').size()
print(monthly_counts)

# Create monthly aggregation
print("\n10. CREATING MONTHLY TIME SERIES AGGREGATION...")

# Aggregate by state, incident type, and month
monthly_agg = df.groupby(['state', 'incidentType', 'year_month']).size().reset_index(name='disaster_count')
monthly_agg['date'] = monthly_agg['year_month'].dt.to_timestamp()

print(f"✓ Created {len(monthly_agg):,} state-incident-month records")
print("\nSample of aggregated data:")
print(monthly_agg.head(20))

# Calculate statistics per state-incident combination
print("\n11. STATISTICS BY STATE-INCIDENT TYPE:")
state_incident_stats = monthly_agg.groupby(['state', 'incidentType'])['disaster_count'].agg([
    ('total_disasters', 'sum'),
    ('avg_monthly', 'mean'),
    ('max_monthly', 'max'),
    ('months_with_data', 'count')
]).round(2)

state_incident_stats = state_incident_stats.sort_values('total_disasters', ascending=False)
print("\nTop 20 State-Incident combinations:")
print(state_incident_stats.head(20))

# Save aggregated data
monthly_agg.to_csv('disaster_forecast/fema_monthly_aggregated.csv', index=False)
state_incident_stats.to_csv('disaster_forecast/fema_state_incident_stats.csv')
print(f"\n✓ Saved aggregated data to fema_monthly_aggregated.csv")
print(f"✓ Saved statistics to fema_state_incident_stats.csv")

# Identify top combinations for modeling
print("\n12. TOP COMBINATIONS FOR TIME SERIES MODELING:")
top_combinations = state_incident_stats[
    (state_incident_stats['total_disasters'] >= 10) &  # At least 10 total disasters
    (state_incident_stats['months_with_data'] >= 12)    # At least 12 months of data
].head(50)

print(f"\n✓ Identified {len(top_combinations)} state-incident combinations suitable for modeling")
print(f"  (Criteria: ≥10 total disasters, ≥12 months of data)")
print("\nTop 30:")
print(top_combinations.head(30))

# Analyze climate change trends
print("\n13. CLIMATE CHANGE TREND ANALYSIS:")
yearly_total = df.groupby('year').size()
if len(yearly_total) >= 5:
    from scipy import stats
    years = yearly_total.index.values
    counts = yearly_total.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
    
    print(f"Linear trend: slope = {slope:.2f} disasters/year")
    print(f"R² = {r_value**2:.3f}")
    print(f"P-value = {p_value:.4f}")
    
    if p_value < 0.05:
        if slope > 0:
            print("✓ Statistically significant INCREASING trend detected")
        else:
            print("✓ Statistically significant DECREASING trend detected")
    else:
        print("No statistically significant trend")

# Save key metrics
summary = {
    'total_records': len(df),
    'date_range': f"{df['incidentBeginDate'].min()} to {df['incidentBeginDate'].max()}",
    'years_of_data': df['year'].max() - df['year'].min() + 1,
    'unique_states': df['state'].nunique(),
    'unique_counties': df['fips_code'].nunique(),
    'unique_disaster_types': df['incidentType'].nunique(),
    'unique_disaster_events': df['disasterNumber'].nunique(),
    'state_incident_combinations': len(state_incident_stats),
    'modelable_combinations': len(top_combinations)
}

print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("="*80)
for key, value in summary.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("PHASE 1 COMPLETE - EDA DONE!")
print("="*80)
print("\nNext Steps:")
print("1. Review fema_monthly_aggregated.csv for time series structure")
print("2. Review fema_state_incident_stats.csv for top combinations")
print("3. Proceed to Phase 2: Time Series Modeling")
print("\n" + "="*80)
