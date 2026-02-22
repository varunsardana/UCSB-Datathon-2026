# DisasterShift — Project Log

## Assumptions & Design Decisions

### 6-Month Post-Disaster Window
- We measure job displacement in the 6 months after a disaster's start date
- Too short (1-2 months) misses delayed layoffs (businesses holding on before shutting down)
- Too long (12+ months) captures job endings unrelated to the disaster
- 6 months is a standard window in labor economics research

### Baseline = Same 6-Month Window One Year Prior
- We compare post-disaster job endings to the same period one year earlier
- This controls for seasonal patterns (e.g., retail always has January layoffs)
- Limitation: if the baseline year had its own disaster or economic event (e.g., COVID), the comparison breaks down

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

## Data Pipeline Summary
- Step 1: Cleaned FEMA data → 45,197 rows, 2,638 disasters
- Step 2: Cleaned job data → 125,433 job endings from 75,139 people
- Step 3: Merged on FIPS code + time window → 9,639 training rows (404 disasters × 302 industries)
