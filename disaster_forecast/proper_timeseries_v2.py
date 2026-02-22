"""
PROPER Time Series Forecasting - FEMA Disasters
Build forecasting models using exponential smoothing + trend + seasonality
Generate 72-month forecasts (through Feb 2032)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PROPER TIME SERIES FORECASTING - STATE × DISASTER TYPE")
print("="*80)

# Load data
print("\n1. LOADING AND PREPARING DATA...")
df = pd.read_csv('/Users/alizasamad/Downloads/projects/datathon26/UCSB-Datathon-2026/data/fema_clean.csv')
df['incidentBeginDate'] = pd.to_datetime(df['incidentBeginDate'])
df['year_month'] = df['incidentBeginDate'].dt.to_period('M')

print(f"✓ Loaded {len(df):,} records")
print(f"✓ Date range: {df['incidentBeginDate'].min()} to {df['incidentBeginDate'].max()}")

# Focus on major disaster types
major_disasters = ['Hurricane', 'Severe Storm', 'Flood', 'Fire', 'Tornado', 
                   'Snowstorm', 'Severe Ice Storm']

all_states = sorted(df['state'].unique())
print(f"✓ {len(all_states)} states/territories")

# ============================================================================
# AGGREGATE MONTHLY DATA
# ============================================================================

print("\n2. CREATING COMPLETE MONTHLY TIME SERIES...")

monthly_data = df.groupby(['state', 'incidentType', 'year_month']).size().reset_index(name='disaster_count')
monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()

print(f"✓ Aggregated to {len(monthly_data):,} monthly records")

def create_complete_series(df_subset, start_date, end_date):
    """Fill missing months with 0 disasters"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    complete_df = pd.DataFrame({'date': date_range})
    complete_df = complete_df.merge(df_subset[['date', 'disaster_count']], 
                                    on='date', how='left')
    complete_df['disaster_count'] = complete_df['disaster_count'].fillna(0).astype(int)
    return complete_df

start_date = '2000-01-01'
end_date = '2026-02-01'

# ============================================================================
# TIME SERIES FORECASTING FUNCTIONS
# ============================================================================

def calculate_trend(y):
    """Calculate linear trend"""
    x = np.arange(len(y))
    if len(y) < 3:
        return 0, 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept

def calculate_seasonality(y, period=12):
    """Calculate seasonal indices"""
    if len(y) < period * 2:
        return np.ones(period)
    
    # Reshape into years
    n_complete = (len(y) // period) * period
    y_complete = y[:n_complete]
    reshaped = y_complete.reshape(-1, period)
    
    # Calculate average for each month
    seasonal_avg = reshaped.mean(axis=0)
    
    # Normalize
    overall_mean = y_complete.mean()
    if overall_mean > 0:
        seasonal_indices = seasonal_avg / overall_mean
    else:
        seasonal_indices = np.ones(period)
    
    return seasonal_indices

def exponential_smoothing_forecast(y, alpha=0.3, horizon=72):
    """Simple exponential smoothing"""
    if len(y) == 0:
        return np.zeros(horizon)
    
    # Initialize
    forecast = []
    s = y[0]  # Initial smoothed value
    
    # Smooth historical data
    for val in y:
        s = alpha * val + (1 - alpha) * s
    
    # Forecast (constant at last smoothed value)
    for _ in range(horizon):
        forecast.append(s)
    
    return np.array(forecast)

def trend_seasonal_forecast(y, horizon=72):
    """Forecast using trend + seasonality"""
    if len(y) < 24:
        # Not enough data, use simple average
        avg = np.mean(y) if len(y) > 0 else 0
        return np.full(horizon, avg), 0, 0
    
    # Calculate components
    trend_slope, trend_intercept = calculate_trend(y)
    seasonal_indices = calculate_seasonality(y, period=12)
    
    # Detrend
    x = np.arange(len(y))
    trend_line = trend_slope * x + trend_intercept
    detrended = y - trend_line
    
    # Calculate baseline (mean of detrended data)
    baseline = np.mean(detrended)
    
    # Generate forecast
    forecast = []
    current_x = len(y)
    
    for i in range(horizon):
        # Trend component
        trend_value = trend_slope * (current_x + i) + trend_intercept
        
        # Seasonal component
        month_idx = (len(y) + i) % 12
        seasonal_factor = seasonal_indices[month_idx]
        
        # Combine
        if trend_value > 0:
            pred = trend_value * seasonal_factor
        else:
            pred = baseline * seasonal_factor
        
        forecast.append(max(0, pred))
    
    # Calculate uncertainty (std of residuals)
    residuals = y - (trend_line + baseline * seasonal_indices[x % 12])
    std_error = np.std(residuals)
    
    return np.array(forecast), std_error, trend_slope

# ============================================================================
# BUILD MODELS FOR STATE-DISASTER COMBINATIONS
# ============================================================================

print("\n3. BUILDING TIME SERIES FORECAST MODELS...")
print("="*80)

forecasts_data = {}
model_performance = []

# Select combinations with sufficient data
state_disaster_combos = []

for disaster in major_disasters:
    disaster_df = monthly_data[monthly_data['incidentType'] == disaster]
    
    for state in all_states:
        state_disaster_df = disaster_df[disaster_df['state'] == state]
        
        if len(state_disaster_df) >= 12:  # At least 12 months
            total_disasters = state_disaster_df['disaster_count'].sum()
            if total_disasters >= 10:  # At least 10 total disasters
                state_disaster_combos.append((state, disaster, total_disasters))

# Sort by total disasters and take top 50
state_disaster_combos.sort(key=lambda x: x[2], reverse=True)
top_combos = state_disaster_combos[:50]

print(f"\n✓ Selected {len(top_combos)} state-disaster combinations for forecasting")

forecast_horizon = 72  # 6 years

for idx, (state, disaster, total) in enumerate(top_combos):
    print(f"[{idx+1}/{len(top_combos)}] {state} - {disaster} ({total} disasters)")
    
    try:
        # Get data
        subset = monthly_data[
            (monthly_data['state'] == state) & 
            (monthly_data['incidentType'] == disaster)
        ]
        
        # Create complete time series
        ts_data = create_complete_series(subset, start_date, end_date)
        
        y = ts_data['disaster_count'].values
        dates = ts_data['date'].values
        
        # Train/test split
        train_size = len(y) - 12
        train_data = y[:train_size]
        test_data = y[train_size:]
        
        if len(train_data) < 12:
            continue
        
        # Generate forecast
        forecast, std_error, trend = trend_seasonal_forecast(train_data, horizon=forecast_horizon + 12)
        
        # Validate on test set
        predictions_test = forecast[:12]
        mae = np.mean(np.abs(test_data - predictions_test))
        rmse = np.sqrt(np.mean((test_data - predictions_test) ** 2))
        
        # Use full data for final forecast
        forecast_full, std_error_full, trend_full = trend_seasonal_forecast(y, horizon=forecast_horizon)
        
        # Generate future dates
        last_date = pd.Timestamp(dates[-1])
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                    periods=forecast_horizon, freq='MS')
        
        # Store results
        key = f"{state}_{disaster.replace(' ', '_')}"
        forecasts_data[key] = {
            'state': state,
            'disaster_type': disaster,
            'historical': {
                'dates': [pd.Timestamp(d).strftime('%Y-%m') for d in dates],
                'counts': y.tolist()
            },
            'forecast': {
                'dates': [pd.Timestamp(d).strftime('%Y-%m') for d in future_dates],
                'predicted_counts': [round(v, 2) for v in forecast_full.tolist()],
                'lower_bound': [max(0, round(v - 1.96 * std_error_full, 2)) for v in forecast_full.tolist()],
                'upper_bound': [round(v + 1.96 * std_error_full, 2) for v in forecast_full.tolist()]
            },
            'model_info': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'trend_slope': round(trend_full, 4),
                'trend_direction': 'increasing' if trend_full > 0.01 else 'decreasing' if trend_full < -0.01 else 'stable'
            }
        }
        
        model_performance.append({
            'state': state,
            'disaster_type': disaster,
            'total_historical': int(total),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'trend_slope': round(trend_full, 4),
            'forecast_avg_annual': round(np.mean(forecast_full[:12]) * 12, 1)
        })
        
    except Exception as e:
        print(f"  ⚠ Error: {str(e)[:80]}")
        continue

print(f"\n✓ Successfully built {len(forecasts_data)} forecast models")

# ============================================================================
# CREATE SLIDER-READY FORMAT
# ============================================================================

print("\n4. CREATING US MAP SLIDER DATA FORMAT...")

slider_data = {}

for disaster in major_disasters:
    print(f"  Processing {disaster}...")
    
    disaster_keys = [k for k in forecasts_data.keys() if disaster.replace(' ', '_') in k]
    
    if len(disaster_keys) == 0:
        continue
    
    # Collect all dates
    all_dates_set = set()
    for key in disaster_keys:
        all_dates_set.update(forecasts_data[key]['historical']['dates'])
        all_dates_set.update(forecasts_data[key]['forecast']['dates'])
    
    all_dates = sorted(list(all_dates_set))
    
    # Structure: { date: { state: count } }
    date_state_map = {}
    
    for date in all_dates:
        date_state_map[date] = {}
    
    # Fill in data
    for key in disaster_keys:
        state = forecasts_data[key]['state']
        
        # Historical
        for d, c in zip(forecasts_data[key]['historical']['dates'], 
                       forecasts_data[key]['historical']['counts']):
            date_state_map[d][state] = {'count': c, 'type': 'historical'}
        
        # Forecast
        for d, c in zip(forecasts_data[key]['forecast']['dates'], 
                       forecasts_data[key]['forecast']['predicted_counts']):
            date_state_map[d][state] = {'count': c, 'type': 'forecast'}
    
    slider_data[disaster] = {
        'disaster_type': disaster,
        'dates': all_dates,
        'data_by_date': date_state_map
    }

# Save files
print("\n5. SAVING RESULTS...")

with open('disaster_forecast/ts_forecasts_proper.json', 'w') as f:
    json.dump(forecasts_data, f, indent=2)

performance_df = pd.DataFrame(model_performance)
performance_df.to_csv('disaster_forecast/ts_model_performance.csv', index=False)

with open('disaster_forecast/us_map_slider_data.json', 'w') as f:
    json.dump(slider_data, f, indent=2)

print(f"✓ Saved full forecasts: ts_forecasts_proper.json ({len(forecasts_data)} models)")
print(f"✓ Saved performance: ts_model_performance.csv")
print(f"✓ Saved slider data: us_map_slider_data.json ({len(slider_data)} disaster types)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TIME SERIES FORECASTING COMPLETE!")
print("="*80)

if len(performance_df) > 0:
    print(f"\nModel Performance Summary:")
    print(f"  Models built: {len(performance_df)}")
    print(f"  Average MAE: {performance_df['mae'].mean():.2f} disasters/month")
    print(f"  Average RMSE: {performance_df['rmse'].mean():.2f}")
    
    print(f"\n  Best performing model:")
    best_idx = performance_df['mae'].idxmin()
    print(f"    {performance_df.loc[best_idx, 'state']} - {performance_df.loc[best_idx, 'disaster_type']}")
    print(f"    MAE: {performance_df.loc[best_idx, 'mae']:.2f}")
    
    print(f"\n  Trends detected:")
    increasing = len(performance_df[performance_df['trend_slope'] > 0.01])
    decreasing = len(performance_df[performance_df['trend_slope'] < -0.01])
    stable = len(performance_df) - increasing - decreasing
    print(f"    Increasing: {increasing} models")
    print(f"    Stable: {stable} models")
    print(f"    Decreasing: {decreasing} models")

print(f"\nForecast Details:")
print(f"  Historical data: 2000-01 to 2026-02")
print(f"  Forecast period: 2026-03 to 2032-02 (72 months)")
print(f"  Disaster types covered: {len(slider_data)}")

print("\n" + "="*80)
print("Next: Build interactive US map with time slider!")
print("="*80)
