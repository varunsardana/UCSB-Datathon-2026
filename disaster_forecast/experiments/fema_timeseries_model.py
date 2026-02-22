"""
FEMA Disaster Time Series Analysis - Phase 2: Time Series Modeling
Using seasonal decomposition and simple forecasting for monthly patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FEMA DISASTER TIME SERIES - PHASE 2: MODELING")
print("="*80)

# Load aggregated data
print("\n1. LOADING AGGREGATED DATA...")
monthly_agg = pd.read_csv('disaster_forecast/fema_monthly_aggregated.csv')
monthly_agg['date'] = pd.to_datetime(monthly_agg['date'])
state_stats = pd.read_csv('disaster_forecast/fema_state_incident_stats.csv')

print(f"✓ Loaded {len(monthly_agg):,} monthly records")
print(f"✓ Loaded {len(state_stats):,} state-incident combinations")

# Select top combinations for modeling
print("\n2. SELECTING TOP COMBINATIONS FOR MODELING...")
top_combos = state_stats[
    (state_stats['total_disasters'] >= 50) &
    (state_stats['months_with_data'] >= 12)
].head(30)

print(f"✓ Selected {len(top_combos)} top combinations")
print("\nTop 15 combinations to model:")
print(top_combos.head(15)[['state', 'incidentType', 'total_disasters', 'avg_monthly']])

# Function to create complete monthly time series
def create_complete_timeseries(df, state, incident_type):
    """Create complete monthly time series with missing months filled as 0"""
    
    # Filter data
    subset = df[(df['state'] == state) & (df['incidentType'] == incident_type)].copy()
    
    if len(subset) == 0:
        return None
    
    # Create complete date range
    min_date = subset['date'].min()
    max_date = subset['date'].max()
    
    # Create monthly range
    date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    # Create complete dataframe
    complete_df = pd.DataFrame({'date': date_range})
    complete_df = complete_df.merge(subset[['date', 'disaster_count']], on='date', how='left')
    complete_df['disaster_count'] = complete_df['disaster_count'].fillna(0)
    
    # Add temporal features
    complete_df['year'] = complete_df['date'].dt.year
    complete_df['month'] = complete_df['date'].dt.month
    complete_df['quarter'] = complete_df['date'].dt.quarter
    
    return complete_df

# Function to calculate seasonal patterns
def calculate_seasonal_patterns(ts_df):
    """Calculate average monthly patterns and trends"""
    
    # Monthly averages (seasonality)
    monthly_avg = ts_df.groupby('month')['disaster_count'].agg(['mean', 'std', 'max']).round(2)
    
    # Yearly trend
    yearly_avg = ts_df.groupby('year')['disaster_count'].sum()
    
    # Calculate trend
    if len(yearly_avg) >= 3:
        from scipy import stats
        years = yearly_avg.index.values
        counts = yearly_avg.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
        trend_info = {
            'slope': round(slope, 2),
            'r_squared': round(r_value**2, 3),
            'p_value': round(p_value, 4),
            'trend': 'increasing' if slope > 0 and p_value < 0.05 else 'stable' if p_value >= 0.05 else 'decreasing'
        }
    else:
        trend_info = {'trend': 'insufficient_data'}
    
    return monthly_avg, trend_info

# Function to generate forecasts
def generate_forecast(ts_df, monthly_patterns, trend_info, months_ahead=12):
    """Generate simple forecast based on historical patterns"""
    
    forecasts = []
    last_date = ts_df['date'].max()
    
    for i in range(1, months_ahead + 1):
        # Next month
        forecast_date = last_date + pd.DateOffset(months=i)
        month = forecast_date.month
        
        # Base forecast from seasonal average
        base_forecast = monthly_patterns.loc[month, 'mean']
        
        # Adjust for trend
        if trend_info['trend'] == 'increasing':
            trend_adjustment = trend_info['slope'] * (i / 12)  # Pro-rate annual trend
            forecast_value = base_forecast + trend_adjustment
        elif trend_info['trend'] == 'decreasing':
            trend_adjustment = trend_info['slope'] * (i / 12)
            forecast_value = base_forecast + trend_adjustment
        else:
            forecast_value = base_forecast
        
        # Calculate confidence interval (based on historical std)
        std_dev = monthly_patterns.loc[month, 'std']
        lower_bound = max(0, forecast_value - 1.96 * std_dev)
        upper_bound = forecast_value + 1.96 * std_dev
        
        forecasts.append({
            'date': forecast_date.strftime('%Y-%m'),
            'month': month,
            'predicted_count': round(forecast_value, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2)
        })
    
    return forecasts

# Process each top combination
print("\n3. BUILDING TIME SERIES MODELS...")
print("="*80)

all_predictions = {}
model_summaries = []

for idx, row in top_combos.iterrows():
    state = row['state']
    incident_type = row['incidentType']
    
    print(f"\nProcessing: {state} - {incident_type}")
    
    # Create complete time series
    ts_df = create_complete_timeseries(monthly_agg, state, incident_type)
    
    if ts_df is None or len(ts_df) < 12:
        print("  ⚠ Insufficient data, skipping")
        continue
    
    print(f"  ✓ Time series: {len(ts_df)} months ({ts_df['date'].min().strftime('%Y-%m')} to {ts_df['date'].max().strftime('%Y-%m')})")
    
    # Calculate patterns
    monthly_patterns, trend_info = calculate_seasonal_patterns(ts_df)
    
    print(f"  ✓ Trend: {trend_info['trend']}")
    
    # Find peak months
    peak_months = monthly_patterns.nlargest(3, 'mean').index.tolist()
    peak_month_names = [datetime(2000, m, 1).strftime('%B') for m in peak_months]
    
    print(f"  ✓ Peak months: {', '.join(peak_month_names)}")
    
    # Generate forecast
    forecasts = generate_forecast(ts_df, monthly_patterns, trend_info, months_ahead=12)
    
    # Store results
    key = f"{state}_{incident_type}"
    all_predictions[key] = {
        'state': state,
        'incident_type': incident_type,
        'monthly_patterns': monthly_patterns.to_dict(),
        'trend': trend_info,
        'peak_months': peak_months,
        'forecasts': forecasts,
        'historical_summary': {
            'total_disasters': int(ts_df['disaster_count'].sum()),
            'avg_per_month': round(ts_df['disaster_count'].mean(), 2),
            'max_in_month': int(ts_df['disaster_count'].max()),
            'data_span_months': len(ts_df)
        }
    }
    
    model_summaries.append({
        'state': state,
        'incident_type': incident_type,
        'total_disasters': int(ts_df['disaster_count'].sum()),
        'avg_monthly': round(ts_df['disaster_count'].mean(), 2),
        'trend': trend_info['trend'],
        'peak_months': ', '.join(peak_month_names)
    })
    
    print(f"  ✓ Generated 12-month forecast")

print(f"\n✓ Completed modeling for {len(all_predictions)} combinations")

# Save predictions
print("\n4. SAVING RESULTS...")
with open('disaster_forecast/disaster_predictions.json', 'w') as f:
    json.dump(all_predictions, f, indent=2)

model_summary_df = pd.DataFrame(model_summaries)
model_summary_df.to_csv('disaster_forecast/model_summary.csv', index=False)

print(f"✓ Saved predictions to disaster_predictions.json")
print(f"✓ Saved summary to model_summary.csv")

# Create simplified output for RAG chatbot
print("\n5. CREATING RAG-FRIENDLY OUTPUT FORMAT...")

rag_output = {}

for key, data in all_predictions.items():
    state = data['state']
    incident_type = data['incident_type']
    
    # Create monthly risk profile
    monthly_risk = []
    for month in range(1, 13):
        month_name = datetime(2000, month, 1).strftime('%B')
        pattern = data['monthly_patterns']['mean'].get(month, 0)
        
        # Risk level categorization
        if pattern >= 2.0:
            risk_level = 'high'
        elif pattern >= 1.0:
            risk_level = 'moderate'
        elif pattern >= 0.5:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        monthly_risk.append({
            'month': month,
            'month_name': month_name,
            'avg_disasters': round(pattern, 2),
            'risk_level': risk_level
        })
    
    # Create location risk profile
    rag_output[key] = {
        'location': {
            'state': state,
            'disaster_type': incident_type
        },
        'risk_summary': {
            'avg_disasters_per_year': round(data['historical_summary']['total_disasters'] / 
                                           (data['historical_summary']['data_span_months'] / 12), 1),
            'trend': data['trend']['trend'],
            'peak_season_months': [datetime(2000, m, 1).strftime('%B') for m in data['peak_months']]
        },
        'monthly_risk_profile': monthly_risk,
        'next_12_months_forecast': data['forecasts']
    }

# Save RAG output
with open('disaster_forecast/rag_disaster_profiles.json', 'w') as f:
    json.dump(rag_output, f, indent=2)

print(f"✓ Created RAG-friendly profiles for {len(rag_output)} combinations")
print(f"✓ Saved to rag_disaster_profiles.json")

# Display sample output
print("\n6. SAMPLE RAG OUTPUT:")
print("="*80)
sample_key = list(rag_output.keys())[0]
sample = rag_output[sample_key]
print(f"\nLocation: {sample['location']['state']} - {sample['location']['disaster_type']}")
print(f"Avg disasters/year: {sample['risk_summary']['avg_disasters_per_year']}")
print(f"Trend: {sample['risk_summary']['trend']}")
print(f"Peak months: {', '.join(sample['risk_summary']['peak_season_months'])}")
print("\nMonthly Risk Profile (first 6 months):")
for month_data in sample['monthly_risk_profile'][:6]:
    print(f"  {month_data['month_name']:10s} - {month_data['risk_level']:10s} ({month_data['avg_disasters']} avg)")

print("\n" + "="*80)
print("PHASE 2 COMPLETE - TIME SERIES MODELING DONE!")
print("="*80)
print("\nFiles created:")
print("  1. disaster_predictions.json - Full prediction data")
print("  2. model_summary.csv - Summary of all models")
print("  3. rag_disaster_profiles.json - RAG chatbot format")
print("\nNext Steps:")
print("  1. Review predictions and validate against known patterns")
print("  2. Integrate with slider visualization")
print("  3. Connect to RAG chatbot system")
print("="*80)
