# DisasterShift — Project Log

## Assumptions & Design Decisions

### 3-Window Post-Disaster Structure
- We measure job displacement across 3 consecutive 6-month windows after each disaster:
  - Window 1 (0-6 months): Immediate shock
  - Window 2 (6-12 months): Recovery phase
  - Window 3 (12-18 months): Normalization
- Why 3 windows: a single 6-month window misses delayed effects (recovery jobs, delayed layoffs). Different sectors respond at different speeds — construction may boom in window 2 while retail collapses in window 1.
- Results validate this design: avg excess exits decrease across windows (0.31 → 0.20 → 0.03)

### Baseline Selection: Average of 2yr + 3yr Prior
**Problem with 1-year-prior baseline:**
- With the 3-window structure, window 3 (12-18 months post-disaster) baseline at 1yr prior would overlap with window 1 (0-6 months post-disaster). The "baseline" period would include disaster-affected data.
- Additionally, the prior year might have had its own disaster or economic event (e.g., COVID), contaminating the baseline.

**Options considered:**
1. **Flag contaminated baselines** — mark rows where baseline year had a disaster. Pros: transparent. Cons: loses data, doesn't fix the overlap problem.
2. **Average multiple baseline years (2yr + 3yr ago)** — use the same window shifted back 2 and 3 years, average the counts. Pros: avoids overlap, smooths out single-year anomalies. Cons: older data may be less relevant.
3. **Raw counts only, no baseline** — skip baseline subtraction entirely. Pros: simplest. Cons: loses seasonal adjustment, can't distinguish disaster-driven exits from normal turnover.

**Decision: Option 2 (average of 2yr + 3yr ago)**
- Avoids the 1yr overlap problem entirely
- Averaging 2 years reduces the chance of either year being anomalous
- We tested adding a 4th year (2yr + 3yr + 4yr):

| Baseline | MAE | R² | Stability (±) |
|---|---|---|---|
| 2yr + 3yr avg | 0.636 | 0.887 | ± 0.089 |
| 2yr + 3yr + 4yr avg | 0.634 | 0.895 | ± 0.068 |

- Difference is negligible (0.008 R² improvement). Per parsimony principle, we go with the simpler 2yr+3yr approach — easier to justify, fewer assumptions about relevance of 4-year-old economic data.

### Major Disasters Only (DR declarations)
- Filtered to FEMA "DR" (Major Disaster) declarations only
- Excluded EM (Emergency) and FM (Fire Management) — too small to affect employment
- Still left us with 2,638 unique disasters and 45,197 county-level rows

## Known Limitations / Weaknesses

### Baseline year may not be "normal"
- If the prior year had its own disaster or economic event (e.g., COVID 2020), the baseline is inflated
- This makes the current disaster look less impactful than it actually is
- Ideal fix: average multiple baseline years instead of just one

### Fixed 6-month window doesn't fit all disaster types
- Wildfires may displace people in week 1, floods might take 9+ months to show full economic damage
- One fixed window for all disaster types is a simplification
- Ideal fix: vary the window per disaster type based on historical patterns

### Job ending ≠ disaster-caused job loss
- Someone might quit for a better job, retire, or relocate independently of the disaster
- We count ALL job endings, not just disaster-caused ones
- Baseline subtraction helps but doesn't fully solve this — some noise remains

### Small numbers problem
- Some FIPS/industry combos have only 2-3 job endings normally
- Going from 2 to 5 endings could be random variation, not disaster signal
- With counts this small, the model may learn noise instead of real patterns
- Ideal fix: statistical significance tests to filter out low-confidence rows

## Key Findings

### Hurricanes Show Negative Excess Exits
- Average excess exits for hurricanes: -0.17 (fewer people left jobs than baseline)
- Possible explanations:
  - People hold onto jobs tighter during crisis
  - Reconstruction/recovery jobs keep people employed
  - Employers may delay layoffs during declared emergencies
  - Federal disaster aid may temporarily prop up businesses
- Worth investigating further — could be a strong talking point in presentation

### Data Constraints
- Job data covers only 75 unique FIPS codes (major metros: LA, SF, NYC, etc.)
- FEMA data has 3,266 FIPS codes — only 74 overlap with job data
- This means we only see disasters that hit major metro areas
- Limits model generalizability to rural/smaller counties

### Sector-Level Pivot
- Pivoted from industry-level (302 industries, 9,639 rows) to sector-level (17 sectors, 3,417 rows)
- Why: industry-level had small numbers problem (2-3 exits per combo), too noisy
- Aggregating to sectors gives stronger signal and cleaner model
- Sectors: Tech, Healthcare, Finance, Education, Retail & Hospitality, Construction & Real Estate, Energy, Manufacturing, Transportation, Media & Entertainment, Marketing & Creative, Legal, Consulting, Government, Nonprofit, Research, Other

### Feature Optimization (318 → 63 features)
- Removed 238 disaster×sector one-hot interaction columns — redundant with disaster_sector_avg_displacement
- Removed 17 season×sector one-hot columns — model learns from quarter + sector_code
- Kept disaster + sector one-hot for LSTM compatibility
- Added disaster_code numeric encoding for XGBoost
- Final ratio: 3,417 rows / 63 features = 54:1 (healthy)

## Model Training Approach

### Two Models — XGBoost vs LSTM
- Training both, picking whichever performs better on held-out data
- **XGBoost**: Builds 500 small decision trees sequentially. Each new tree focuses on the rows the previous trees got wrong. Old trees never change — new trees get stacked on top. Final prediction = sum of all trees. Fast, great for tabular data, works well with small datasets (3,417 rows is plenty).
- **LSTM**: You take your 6 sequential features and feed them one by one into a series of gates. With each month of data, the gates update 64 memory values — deciding what to remember and what to forget. After all 6 months, you have 64 numbers that summarize the whole sequence pattern. Then you take those 64 memory numbers and combine them with the 57 static features to get 121 numbers. You pass those through linear combinations: 121 → 64 → 32 → 1. That 1 number is your prediction. You compare the prediction to the actual excess_exits, calculate the error, and adjust all the weights (both the gate weights and the linear layer weights). Repeat for 100 epochs until the predictions are good.
- **In short**: Feed sequential features one by one into gates, updating 64 memory values about what to remember and forget. Take those 64 values plus context features and create linear combinations through the layers. Compare final value to actual value and adjust weights accordingly.
- Key difference: XGBoost adds new trees to fix mistakes. LSTM adjusts the same set of weights over and over (across 100 epochs) until predictions converge.

### GroupKFold Cross-Validation
- Split data into 5 folds, grouped by disaster number
- All rows from the same disaster go into the same fold (either all train or all test)
- Prevents data leakage: if Hurricane Katrina has 17 sector rows, the model can't see Katrina+Tech in training and then "predict" Katrina+Finance in test — that would be too easy
- Forces the model to generalize: "I've never seen this disaster, but based on patterns from similar ones, I predict X"
- Every disaster gets tested exactly once across the 5 folds

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual excess exits. "On average, our predictions are off by X exits." Lower = better.
- **RMSE (Root Mean Squared Error)**: Same concept but squares errors first, penalizing big misses harder. If RMSE >> MAE, the model has some outlier predictions.
- **R² (R-squared)**: What % of variation in excess exits does the model explain? 0 = no better than guessing the mean, 1 = perfect, <0 = worse than guessing.
- **Baseline comparison**: We also compare model MAE to "always predict the mean" MAE. This proves the model is actually learning something beyond the average.

## Model Results

### Single-Window Results (initial, now superseded)
| Model | MAE | Improvement over baseline | R² |
|---|---|---|---|
| Baseline (predict mean) | 4.141 | — | — |
| **XGBoost** | **1.275** | **69.2%** | 0.59 - 0.93 |
| LSTM | 2.583 | 37.6% | 0.20 - 0.77 |

### 3-Window Results (current production)
| Model | MAE | Improvement over baseline | R² |
|---|---|---|---|
| Baseline (predict mean) | 3.783 | — | — |
| **XGBoost** | **0.636 ± 0.267** | **83.2%** | **0.887 ± 0.089** |

- 3-window structure improved MAE from 1.275 → 0.636 (50% reduction in error)
- R² improved from ~0.76 avg → 0.887 avg
- Model stability: worst fold R² went from 0.59 → 0.77
- `window_num` appears in top 15 features — model learns shock→recovery→normalization

### Why XGBoost Won (over LSTM)
- XGBoost is designed for tabular data with mixed feature types — exactly what we have
- LSTM needs long sequences and lots of data. We only have 6 time steps (months) and 3,417 rows — not enough for LSTM to learn meaningful gate weights
- XGBoost handles the 65 features (numeric codes, one-hot, ratios, scores) natively. LSTM needs scaling and struggles with sparse categorical features
- With more granular time series data (daily/weekly exits, thousands of rows), LSTM would likely improve

### XGBoost is the Production Model
- Final model trained on all 11,796 rows (after evaluation on held-out folds)
- Saved as xgb_model.joblib for the API to load
- Average prediction error: 0.636 excess exits (vs 3.783 baseline)
- Model explains 77-97% of variation in excess exits depending on the fold

## Data Pipeline Summary
- Step 1: Cleaned FEMA data → 45,197 rows, 2,638 disasters
- Step 2: Cleaned job data → 125,433 job endings from 75,139 people
- Step 3: Merged on FIPS code + 3 time windows → 34,857 rows (530 disasters × 334 industries × 3 windows)
  - Baseline: average of 2yr + 3yr prior (avoids 1yr overlap problem)
- Step 4: Sector aggregation + feature engineering → 11,796 rows, 85 columns (65 features)
  - New features: sector concentration, Herfindahl index, sector size, growth trend,
    decay-weighted disaster exposure, historical displacement rate per disaster×sector, window_num
- Step 5: XGBoost training → MAE=0.636, R²=0.887, 83.2% improvement over baseline
