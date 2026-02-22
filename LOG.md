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
