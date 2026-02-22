# DisasterShift — Workforce Disruption Intelligence Platform

> **UCSB Datathon 2026** — A full-stack AI system that predicts, explains, and advises on the employment impact of natural disasters across the United States.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Machine Learning Models](#4-machine-learning-models)
   - [XGBoost Employment Impact Model](#41-xgboost-employment-impact-model)
   - [Prophet Disaster Frequency Forecasting](#42-prophet-disaster-frequency-forecasting)
   - [LSTM (Rejected — Documented for Reproducibility)](#43-lstm-rejected--documented-for-reproducibility)
5. [RAG Intelligence Pipeline](#5-rag-intelligence-pipeline)
   - [ChromaDB Knowledge Base](#51-chromadb-knowledge-base)
   - [SQL Aggregation Engine](#52-sql-aggregation-engine)
   - [Query Router](#53-query-router)
   - [Audience Detection & System Prompt](#54-audience-detection--system-prompt)
6. [REST API Reference](#6-rest-api-reference)
7. [Frontend Application](#7-frontend-application)
8. [Setup & Running](#8-setup--running)
9. [Design Decisions & Assumptions](#9-design-decisions--assumptions)
10. [Known Limitations](#10-known-limitations)
11. [Key Findings](#11-key-findings)
12. [Model Performance](#12-model-performance)
13. [Data Coverage](#13-data-coverage)
14. [Tech Stack Summary](#14-tech-stack-summary)

---

## 1. Project Overview

**DisasterShift** answers a single critical question: *When a natural disaster hits, who loses their job, how many, and how long does recovery take?*

The platform integrates three data-driven models into one intelligent advisory system:

| Layer | Model | Output |
|---|---|---|
| **When** | Prophet time-series forecasting | Predicted disaster frequency per state × disaster type, 6 years forward |
| **Who** | XGBoost gradient boosting | Sector-level job loss %, recovery timeline (months), demand surge % |
| **Why / What next** | RAG + Claude API | Plain-language guidance tailored to workers, employers, policymakers, investors, and insurers |

**Target audiences:**
- **Workers** — "Will I lose my restaurant job if a hurricane hits Florida?"
- **Employers** — "How many workers will we lose? How long until full staffing?"
- **Policymakers** — "Which sectors need retraining programs and when?"
- **Investors / Insurers** — "What is our portfolio exposure across the Gulf Coast?"

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                    │
│  DisasterMap · KeyMetrics · AnalyticsTabs · AdvisorChat          │
│                     Port 3000 → proxied to 8000                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP / SSE (Server-Sent Events)
┌──────────────────────────▼───────────────────────────────────────┐
│                      FASTAPI BACKEND  (Port 8000)                 │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  /api/chat   │  │ /api/predict │  │  /api/forecast          │ │
│  │  /api/       │  │ /api/        │  │  /api/disasters         │ │
│  │  analytics   │  │ predict/     │  │  /api/analytics         │ │
│  └──────┬───────┘  │ by-state     │  └────────────┬────────────┘ │
│         │          └──────────────┘               │              │
│  ┌──────▼───────────────────────────────────────┐ │              │
│  │              RAG PIPELINE                     │ │              │
│  │                                               │ │              │
│  │  1. Detect audience (worker/employer/etc.)    │ │              │
│  │  2. Prophet forecast → direct injection       │ │              │
│  │  3. XGBoost prediction → direct injection     │ │              │
│  │  4. SQL aggregation → structured context      │ │              │
│  │  5. ChromaDB similarity search → KB chunks    │ │              │
│  │  6. Build system prompt → LLM                 │ │              │
│  └──────────────────────┬────────────────────────┘ │              │
│                         │                          │              │
│  ┌──────────────────────▼──────┐  ┌───────────────▼────────────┐ │
│  │   LLM Provider              │  │  Data Services              │ │
│  │   Anthropic Claude API      │  │  model_predictions.json     │ │
│  │   (or Ollama local)         │  │  prophet_state_forecasts.   │ │
│  └─────────────────────────────┘  │  json                       │ │
│                                   │  SQLite (in-memory)         │ │
│  ┌──────────────────────────────┐ │  ChromaDB (disk-persistent) │ │
│  │  Embedding Model             │ └─────────────────────────────┘ │
│  │  all-MiniLM-L6-v2 (384-dim) │                                  │
│  └──────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘

DATA PIPELINE (offline, run once)
  FEMA Declarations → step1 → step2 → step3 → step4 → step5 → step6
  Jobs Data                                          XGBoost → JSON
  Prophet Forecast  →  disaster_forecast/prophet_forecast.py → JSON
```

---

## 3. Data Pipeline

The ML pipeline runs offline and produces the two JSON files the backend serves at runtime. Six numbered Python scripts in `data/` transform raw public data into model-ready features.

### Step 1 — Clean FEMA Declarations (`data/step1_clean_fema.py`)

- **Source**: FEMA OpenFEMA API — Major Disaster (DR) declarations
- **Filtered to**: DR declarations only (excludes EM and FM — too small to affect employment)
- **Output**: `data/fema_clean.csv`

| Column | Description |
|---|---|
| disasterNumber | Unique FEMA disaster ID |
| fips_code | 5-digit county FIPS code |
| incidentBeginDate | Disaster start date |
| incidentType | hurricane, flood, fire, etc. |
| state | 2-letter state code |
| county | County name |

**Result**: 45,198 rows · 2,638 unique disasters

---

### Step 2 — Clean Job Data (`data/step2_clean_jobs.py`)

- **Source**: Employment separation records from a state workforce database
- **Captures**: Every "job ending" (quit, layoff, retirement) with FIPS code and industry
- **Output**: `data/jobs_clean.csv`

**Result**: 125,434 job endings · 75 unique FIPS codes (major metros)

---

### Step 3 — Merge FEMA × Jobs (`data/step3_merge.py`)

The core statistical design: for each disaster, find how many job endings happened in the same county across three 6-month windows post-disaster, compared to a baseline from 2–3 years earlier.

**3-Window Post-Disaster Structure:**

```
Baseline period:    2yr ago + 3yr ago (averaged)
                    ←──────────────────────────→

Window 1 (0–6 mo):  Immediate shock
Window 2 (6–12 mo): Recovery phase
Window 3 (12–18 mo): Normalization

Excess exits = actual job endings − baseline job endings (per window)
```

- **Output**: `data/merged_disaster_jobs.csv`
- **Result**: 34,858 rows after FIPS × temporal join

---

### Step 4 — Feature Engineering (`data/step4_features.py`)

Fully vectorized (no loops). Creates 85 columns from the merged dataset.

**Feature Groups:**

| Group | Features | Count |
|---|---|---|
| Disaster type one-hot | biological, coastal_storm, earthquake, fire, flood, etc. | 16 |
| Sector one-hot | Healthcare, Tech, Retail & Hospitality, Construction, etc. | 17 |
| Temporal | disaster_month, disaster_quarter, disaster_year, month_sin, month_cos | 5 |
| Scale | baseline_exits, log_baseline, exit_ratio | 3 |
| Window | window_num (1, 2, or 3) — captures the shock→recovery pattern | 1 |
| Disaster code | Numeric encoding for XGBoost | 1 |
| Interaction | disaster_sector_avg_displacement, sector_concentration, herfindahl_index, sector_size, growth_trend, decay_weighted_disaster_exposure, recovery_rate_historical | ~18 |

**Sector Aggregation**: 302 raw industries → 17 sectors via `industry_map.csv`

**17 Sectors:**
Tech · Healthcare · Finance · Education · Retail & Hospitality · Construction & Real Estate · Energy · Manufacturing · Transportation · Media & Entertainment · Marketing & Creative · Legal · Consulting · Government · Nonprofit · Research · Other

**Feature Optimization**: Originally 318 features. Removed:
- 238 disaster×sector one-hot interaction columns (redundant with `disaster_sector_avg_displacement`)
- 17 season×sector columns (model already learns from quarter + sector_code)

**Final**: 11,797 rows × 65 features. Feature:row ratio = 181:1 (healthy)

**Output**: `data/features.csv`, `data/feature_columns.txt`

---

### Step 5 — Model Training (`data/step5_train.py`)

Two models trained and compared. See [Section 4](#4-machine-learning-models) for full details.

**Output**: `data/xgb_model.joblib` (production model)

---

### Step 6 — Export Predictions (`data/step6_export.py`)

Runs the trained XGBoost model on all disaster × FIPS combinations and serializes results to JSON for the API to serve at runtime (no inference at request time).

**Output**: `backend/data/model_predictions.json` (pre-computed predictions for ~4,000 disaster × FIPS combos)

---

## 4. Machine Learning Models

### 4.1 XGBoost Employment Impact Model

**Type**: Gradient Boosting Regressor — 500 sequential decision trees

**What it predicts**: `excess_exits` — the number of job endings above the 2–3 year baseline, per sector, per disaster, per 6-month window

**Architecture:**

```python
XGBRegressor(
    n_estimators=500,       # 500 stacked trees
    max_depth=6,            # Max tree depth
    learning_rate=0.05,     # Shrinkage factor (slow learning → better generalization)
    subsample=0.8,          # 80% row sampling per tree (reduces overfitting)
    colsample_bytree=0.8,   # 80% feature sampling per tree
    min_child_weight=5,     # Min samples in a leaf (prevents tiny splits)
    reg_alpha=0.1,          # L1 regularization (sparsity)
    reg_lambda=1.0,         # L2 regularization (weight decay)
)
```

**How it works**: XGBoost builds trees sequentially. Each new tree focuses on the residual errors from the previous trees. The final prediction is the sum of all 500 trees. Unlike LSTM (which adjusts the same weights), XGBoost never changes old trees — it stacks new ones to correct past mistakes. This is why it excels on tabular data with mixed feature types.

**Cross-Validation — GroupKFold (5 folds):**

All rows from a given disaster go into the same fold — train or test, never split. This prevents data leakage: if Hurricane Katrina has 17 sector rows, the model can't see Katrina+Tech in training and then "predict" Katrina+Finance in test. It forces true generalization: "I've never seen this exact disaster, but I can predict based on similar ones."

**Performance (3-Window Model — Production):**

| Metric | Mean | Std Dev | Range |
|---|---|---|---|
| MAE | **0.636** | ± 0.267 | 0.45 – 1.23 |
| RMSE | 0.897 | ± 0.417 | 0.56 – 1.85 |
| R² | **0.887** | ± 0.089 | 0.77 – 0.97 |
| vs Baseline MAE | 3.783 | — | — |
| **Improvement** | **83.2%** | — | — |

**Top 10 Features by Importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | baseline_exits | 0.1197 |
| 2 | log_baseline | 0.1117 |
| 3 | exit_ratio | 0.1092 |
| 4 | disaster_code | 0.1055 |
| 5–10 | interaction + sector features | — |

**Output Format** (`model_predictions.json`):

```json
{
  "id": "hurricane_12086",
  "disaster_type": "hurricane",
  "fips_code": "12086",
  "region": "Miami-Dade County, FL",
  "text": "Following a hurricane event in Miami-Dade County, FL...",
  "predictions": {
    "Retail & Hospitality": {
      "job_loss_pct": 44,
      "recovery_months": 18
    },
    "Construction & Real Estate": {
      "job_change_pct": 23,
      "peak_month": 3
    },
    "Healthcare": {
      "job_loss_pct": 12,
      "recovery_months": 9
    }
  }
}
```

Sectors that **lose** workers have `job_loss_pct` + `recovery_months`. Sectors that **gain** workers (demand surge from rebuilding) have `job_change_pct` + `peak_month`.

---

### 4.2 Prophet Disaster Frequency Forecasting

**What it forecasts**: Monthly count of FEMA disaster declarations per state × disaster type, 6 years into the future (2026–2032)

**Model**: Facebook Prophet — additive decomposition of trend + seasonality + holidays

**Training Data**: Monthly FEMA DR declarations aggregated from 2000–2026 (26 years = 312 months per combo)

**Coverage**: 142 state × disaster type combinations with sufficient historical data

**Forecast horizon**: 72 months (6 years) from March 2026

**Output Format** (`prophet_state_forecasts.json`):

```json
{
  "FL_Hurricane": {
    "state": "FL",
    "disaster_type": "Hurricane",
    "model_info": {
      "total_historical": 856,
      "peak_months": ["September", "August", "October"],
      "cv_mae": 4.88,
      "cv_rmse": 7.21,
      "train_months": 312,
      "forecast_horizon": 72
    },
    "historical": {
      "dates": ["2000-01", "2000-02", "..."],
      "counts": [3, 0, "..."]
    },
    "forecast": {
      "dates": ["2026-03", "2026-04", "..."],
      "predicted_counts": [1.52, 0.80, "..."],
      "lower_bound": [0.0, 0.0, "..."],
      "upper_bound": [4.21, 3.50, "..."]
    }
  }
}
```

**Key Metrics:**
- `cv_mae`: Cross-validated Mean Absolute Error on historical holdout
- `peak_months`: Seasonal peaks detected (e.g., September–October for Atlantic hurricanes)
- `avg_next_12`: Average monthly predicted declarations over the next 12 months (used for SQL risk scoring)

**Experiment Files** (`disaster_forecast/experiments/`):

Multiple models were evaluated before settling on Prophet:

| Model | Notes |
|---|---|
| ARIMA (fema_timeseries_model.py) | Baseline — no seasonality |
| Negative Binomial (negbin_model.py) | Better for count data, less interpretable |
| LSTM time-series (proper_timeseries_v2.py) | Too little data per combo |
| **Prophet (prophet_model.py)** | **Winner** — handles seasonality, missing data, outliers |

**Sample Forecast Plots** (`disaster_forecast/plots/`):
- `FL_Hurricane.png` — Florida hurricane frequency forecast
- `CA_Fire.png` — California wildfire frequency forecast
- `TX_Severe_Storm.png` — Texas severe storm forecast
- `LA_Hurricane.png`, `IA_Flood.png`, `OK_Severe_Storm.png`
- `sample_grid.png` — Overview grid of all six

---

### 4.3 LSTM (Rejected — Documented for Reproducibility)

**Architecture:**
1. 6 sequential monthly features fed one-by-one through LSTM gates
2. 64 memory cells (what to remember vs. forget per timestep)
3. Memory cells concatenated with 57 static features → 121 inputs
4. Dense layers: 121 → 64 → 32 → 1 (predicted excess exits)
5. Trained for 100 epochs with Adam optimizer

**Single-Window Results (original experiment):**

| Model | MAE | R² | Improvement |
|---|---|---|---|
| Baseline (predict mean) | 4.141 | — | — |
| LSTM | 2.583 | 0.20 – 0.77 | 37.6% |
| **XGBoost** | **1.275** | **0.59 – 0.93** | **69.2%** |

**Why LSTM was rejected:**
- Only 3,417 rows at sector level — not enough for LSTM to learn meaningful gate weights
- Only 6 time steps (months) — LSTM needs longer sequences to learn temporal dependencies
- XGBoost handles the 65 mixed features (one-hot, ratios, numeric codes) natively; LSTM requires scaling and struggles with sparse categorical features
- With daily/weekly job data and thousands more rows, LSTM would likely improve

**Code**: `data/step5_train.py` (trains both; picks winner)

---

## 5. RAG Intelligence Pipeline

The chat advisor runs a 7-step pipeline per request, streaming status updates to the frontend as each step completes.

```
User message
     │
     ▼
Step 0: Detect audience (worker/employer/policymaker/investor/insurer)
     │
     ▼
Step 1a: Prophet forecast → look up state × disaster_type in JSON
     │
     ▼
Step 1b: XGBoost prediction → FIPS lookup → state fallback → None
     │
     ▼
Step 2: SQL query router → keyword detection → structured SQL context
     │
     ▼
Step 3: ChromaDB retrieval → top-6 chunks by cosine similarity
     │
     ▼
Step 4: Build augmented system prompt (all 4 contexts merged)
     │
     ▼
Step 5: Stream via LLM (Claude API or Ollama)
     │
     ▼
SSE tokens → frontend
```

**File**: `backend/services/chat_service.py`

Each step emits a `__status__` SSE event the frontend displays as a live pipeline progress indicator.

---

### 5.1 ChromaDB Knowledge Base

**Vector Store**: ChromaDB (PersistentClient, saved to `./chroma_db`)
**Embedding Model**: `all-MiniLM-L6-v2` — 384-dimensional sentence embeddings
**Collection**: `disaster_kb` — cosine similarity search
**Total Chunks**: ~441 at runtime (after ingestion)

**Three data sources ingested** (`backend/rag/ingest.py`):

#### Source 1 — Knowledge Base Markdown Files (`backend/data/knowledge/*.md`)

40+ curated documents covering:

| Category | Documents |
|---|---|
| State Unemployment | CA, FL, TX, NC, LA, AZ, CO, CT, DC, GA, IL, IN, MA, MD, MI, MN, MO, NV, NY, OH, OK, OR, PA, RI, TN, UT, VA, WA, WI |
| FEMA Guidance | fema_flood.md, fema_hurricane.md, fema_wildfire.md |
| Benefits Programs | cobra_health.md, warn_act.md, retraining.md, transferable_skills.md, recovery_timelines.md, disaster_financial_aid.md |

Each document is chunked to 800 characters with 100-character overlap.

#### Source 2 — Model Prediction Narratives

The `text` field from each entry in `model_predictions.json` is ingested as a chunk with:
- `category: "model_output"`
- Metadata: `state`, `disaster_type`, `fips_code`

#### Source 3 — Forecast Profiles

Prophet forecasts generate 142 narrative summaries (`forecast_profiles.json`):
- Describes seasonal risk, peak months, and year-over-year trend
- `category: "forecast"`

**Retrieval Strategy** (`backend/rag/retriever.py`):

The retriever uses metadata pre-filtering before cosine search to keep results relevant:

| Context | Filter Strategy |
|---|---|
| State + Disaster | State unemployment + disaster docs + model outputs + Prophet profiles + WARN Act + COBRA + recovery timelines + transferable skills |
| State only | State unemployment + national KB |
| Disaster only | Disaster docs + model outputs + Prophet forecasts + FEMA programs + recovery timelines |
| Neither | Pure cosine similarity across all chunks |

Default `top_k = 6` (configurable in `.env`)

---

### 5.2 SQL Aggregation Engine

**File**: `backend/rag/sql_engine.py`

ChromaDB retrieval can find relevant documents but cannot *aggregate* — it cannot rank sectors by risk, compute portfolio exposure across states, or cross-join forecast frequency with job loss severity. The SQL engine solves this.

**Implementation**: Python built-in `sqlite3`, loaded once at startup into memory from both JSON files. No external database required.

**Two Tables:**

```sql
employment_impact (
    fips_code      TEXT,
    state          TEXT,
    disaster_type  TEXT,
    region         TEXT,
    sector         TEXT,
    job_loss_pct   REAL,
    job_change_pct REAL,
    recovery_months REAL,
    peak_month     INTEGER
)  -- 707 rows

disaster_forecast (
    state             TEXT,
    disaster_type     TEXT,
    total_historical  INTEGER,
    peak_months       TEXT,   -- JSON array
    avg_next_12       REAL,   -- avg monthly declarations next 12 months
    cv_mae            REAL
)  -- 74 rows
```

**Six SQL Query Functions:**

#### `query_sector_ranking(state, disaster_type)`
Ranks sectors by job loss severity and recovery speed for a given state × disaster.
```sql
SELECT sector, AVG(job_loss_pct) AS loss, AVG(recovery_months) AS recovery
FROM employment_impact
WHERE state = ? AND disaster_type = ?
GROUP BY sector ORDER BY loss DESC
```

#### `query_top_risk_combos(limit=10)`
The marquee cross-model query: Prophet frequency × XGBoost job loss = weighted risk score.
```sql
SELECT ei.state, ei.sector, ei.disaster_type,
       AVG(ei.job_loss_pct) * df.avg_next_12 AS risk_score
FROM employment_impact ei
JOIN disaster_forecast df ON ei.state = df.state AND ei.disaster_type = df.disaster_type
GROUP BY ei.state, ei.sector, ei.disaster_type
ORDER BY risk_score DESC LIMIT ?
```

#### `query_portfolio(states, disaster_type)`
Aggregate sector-level risk across a custom list of states (investor portfolio use case).

#### `query_variance()`
Which sectors have the most variation in recovery time across states and disaster types — least reliable predictions.

#### `query_demand_surge(disaster_type)`
Which sectors see *positive* labor demand after a disaster (Construction, Energy, Government, etc.).

#### `query_preposition(limit=10)`
Pre-positioning: where to deploy retraining resources. Ranked by `forecast_frequency × avg_job_loss`.

**Region Groups** (for portfolio queries by region name):

```python
REGION_GROUPS = {
    "southeast":  ["FL", "GA", "NC", "SC", "VA", "AL", "MS", "TN"],
    "gulf_coast": ["FL", "LA", "TX", "MS", "AL"],
    "midwest":    ["IL", "IN", "MI", "MN", "MO", "OH", "WI"],
    "west_coast": ["CA", "OR", "WA"],
    "northeast":  ["NY", "PA", "MA", "CT", "RI"],
    "southwest":  ["AZ", "TX", "NM", "CO", "UT"],
}
```

---

### 5.3 Query Router

**File**: `backend/rag/query_router.py`

Keyword-based classifier that detects question intent and routes to the appropriate SQL query. Questions that don't match any pattern fall through to ChromaDB-only retrieval.

| Keyword Pattern | SQL Query Triggered |
|---|---|
| "top 10 cities", "rank", "which sectors recover fastest", "most at risk" | `query_top_risk_combos()` |
| "pre-position", "deploy resources", "intervention priority", "next 18 months" | `query_preposition()` |
| "portfolio", "southeast", "gulf coast", "across our", "multiple states" | `query_portfolio()` |
| "variance", "reliable", "how accurate", "prediction accuracy" | `query_variance()` |
| "gain workers", "hiring", "construction boom", "which jobs increase" | `query_demand_surge()` |
| "which sector", "worst hit", "recover fastest", "most affected" | `query_sector_ranking()` |

The router also extracts state codes and disaster types from natural language before calling SQL functions.

---

### 5.4 Audience Detection & System Prompt

**File**: `backend/rag/prompts.py`

#### Audience Detection

Before building the prompt, the system detects who it's talking to from the `job_title` + `question` text:

| Audience | Trigger Keywords | Response Style |
|---|---|---|
| **insurer** | "insur", "actuar", "underwrite", "reinsur" | Quantified risk metrics, confidence intervals |
| **investor** | "investor", "portfolio", "fund manager", "analyst" | Risk scores, geographic exposure, recovery timelines |
| **policymaker** | "mayor", "policy", "city council", "government", "agency director" | Intervention priorities, sector-level data |
| **employer** | "owner", "ceo", "hr", "hiring manager", "operations manager" | Workforce planning figures, staffing timelines |
| **worker** (default) | everything else | Plain empathetic language, concrete next steps |

#### Job-to-Sector Mapping (built into prompt)

When the user mentions a specific job, the advisor maps it to the nearest modeled sector:

```
restaurant / waiter      → Retail & Hospitality
nurse / doctor           → Healthcare
teacher                  → Education
contractor / builder     → Construction & Real Estate
store / cashier          → Retail & Hospitality
```

#### System Prompt Structure

The final prompt has 4 data sections injected before the LLM generates a response:

```
--- DISASTER FREQUENCY FORECAST ---
[Prophet: state × disaster, peak months, avg monthly frequency]

--- EMPLOYMENT IMPACT PREDICTION ---
[XGBoost: sector job loss %, recovery months, demand surge %]

--- STRUCTURED DATA RESULTS ---
[SQL: ranked table or portfolio aggregation if applicable]

--- RETRIEVED KNOWLEDGE ---
[ChromaDB: top-6 chunks — unemployment programs, FEMA guides, etc.]

--- USER CONTEXT ---
State: FL | Disaster: hurricane | Job/Industry: restaurant worker
```

The LLM receives all four sources merged into one prompt, is instructed to use only that data (no invented URLs), and must keep responses under 4 short paragraphs.

---

## 6. REST API Reference

Base URL: `http://localhost:8000/api`

### Chat

#### `POST /api/chat`

Runs the full RAG pipeline and streams the response as Server-Sent Events.

**Request Body:**
```json
{
  "message": "Will I lose my restaurant job if a hurricane hits Florida?",
  "state": "FL",
  "disaster_type": "hurricane",
  "job_title": "restaurant worker",
  "fips_code": null,
  "audience_type": null
}
```

**SSE Stream Format:**
```
data: __status__Mode: Worker guidance
data: __status__Fetching disaster frequency forecast for FL hurricane...
data: __status__Analyzing sector-level employment impact...
data: __status__Running structured data analysis...
data: __status__Searching knowledge base for programs and resources...
data: __status__Generating response...
data: Based on our hurricane impact model for Florida, the Retail &
data:  Hospitality sector faces a 44% job loss in the first six months...
data: [DONE]
```

`__status__` prefixed lines are pipeline progress events. All other `data:` lines are LLM content tokens. `[DONE]` signals stream completion. `[ERROR] message` signals a failure.

---

### Predictions

#### `POST /api/predict`

Returns pre-computed sector predictions for a given FIPS code + disaster type.

**Request Body:**
```json
{ "disaster_type": "hurricane", "fips_code": "12086" }
```

**Response:**
```json
{
  "disaster_type": "hurricane",
  "fips_code": "12086",
  "region": "Miami-Dade County, FL",
  "text": "Following a hurricane event in Miami-Dade County, FL...",
  "predictions": {
    "Retail & Hospitality": { "job_loss_pct": 44, "recovery_months": 18 },
    "Construction & Real Estate": { "job_change_pct": 23, "peak_month": 3 }
  }
}
```

#### `POST /api/predict/by-state`

Returns the best-available prediction for a state + disaster type (most sectors covered).

**Request Body:**
```json
{ "disaster_type": "hurricane", "state": "FL" }
```

#### `GET /api/predict/scenarios`

Lists all available pre-computed prediction keys.

**Response:**
```json
{
  "count": 4000,
  "scenarios": [
    { "key": "hurricane_12086", "disaster_type": "hurricane", "fips_code": "12086", "region": "Miami-Dade County, FL" }
  ]
}
```

---

### Forecasts

#### `GET /api/forecast/chart?state=FL&disaster_type=Hurricane`

Returns chart-ready time-series data with historical and 72-month forecast.

**Response:**
```json
{
  "state": "FL",
  "disaster_type": "Hurricane",
  "meta": {
    "total_historical_declarations": 856,
    "peak_months": ["September", "August", "October"],
    "cv_mae": 4.88,
    "cv_rmse": 7.21,
    "train_months": 312,
    "forecast_horizon_months": 72,
    "data_start": "2000-01",
    "data_end": "2026-02",
    "forecast_start": "2026-03",
    "forecast_end": "2032-02"
  },
  "historical": [
    { "date": "2000-01", "count": 3 }
  ],
  "forecast": [
    { "date": "2026-03", "predicted": 1.52, "lower": 0.0, "upper": 4.21 }
  ]
}
```

#### `GET /api/forecast/available` — All available state × disaster combos
#### `GET /api/forecast/states` — All states with forecast data
#### `GET /api/forecast/types?state=FL` — Disaster types available for a state

---

### Disasters

#### `GET /api/disasters?state=FL&disaster_type=hurricane&limit=200`

Returns individual disaster events for the map.

**Response:**
```json
[
  {
    "disaster_id": "DR-4234",
    "disaster_type": "hurricane",
    "declaration_date": "2017-09-10",
    "state": "FL",
    "county": "Miami-Dade County",
    "fips_code": "12086",
    "lat": 25.7617,
    "lng": -80.1918,
    "severity": "major",
    "title": "Hurricane Irma"
  }
]
```

#### `GET /api/disasters/{disaster_id}` — Single disaster event

---

### Analytics

#### `GET /api/analytics?fips_code=06037&state=CA&disaster_type=wildfire`

Returns time-series employment change data for charting.

**Response:**
```json
[
  {
    "fips_code": "06037",
    "state": "CA",
    "industry_group": "Retail & Hospitality",
    "disaster_type": "wildfire",
    "month_offset": 1,
    "job_change_count": -234,
    "job_change_pct": -12.4,
    "recovery_rate": 0.67
  }
]
```

#### `GET /api/analytics/summary` — Returns `{total_events, regions_analyzed, industries_tracked}`

---

## 7. Frontend Application

**Tech**: React 19 · TypeScript · Vite · Tailwind CSS · Recharts · Leaflet

**Dev Server**: Port 3000 (Vite proxy forwards `/api/*` to `localhost:8000`)

### Layout

```
Navbar
├── SidebarLeft (collapsible on mobile)
│   ├── Disaster Type selector
│   ├── State selector
│   ├── FIPS / Scenario selector
│   └── Run Prediction button
│
├── Main Content (center)
│   ├── DisasterMap (Leaflet — event markers with popups)
│   ├── KeyMetrics (Total events / Regions / Industries tracked)
│   └── AnalyticsTabs
│       ├── Tab 1: Displacement Curve (line chart — months vs job loss %)
│       └── Tab 2: Industry Impact (bar chart — sector vs job loss %)
│
└── AdvisorChat (right sidebar — desktop only)
    ├── State + Disaster dropdowns
    ├── Message history with streaming
    ├── Pipeline progress steps + live timer
    └── Suggestion chips
```

### Component Details

#### `SidebarLeft.jsx`
- Loads available scenarios from `GET /api/predict/scenarios`
- Calls `POST /api/predict` or `POST /api/predict/by-state` on "Run Prediction"
- Passes prediction data to `AnalyticsTabs` for charting
- Resizable (default 270px)

#### `DisasterMap.jsx`
- Leaflet map (OpenStreetMap tiles)
- Event markers: red = major, orange = minor
- Popups with disaster name, date, severity
- Data from `GET /api/disasters`

#### `AnalyticsTabs.jsx`
- **Displacement Curve**: Line chart with 13 time points (-6mo, -3mo, 0, +1 through +18mo), one line per sector
- **Industry Impact**: Bar chart with job loss % and demand surge % per sector, risk badge (High/Medium/Low)
- Fallback to `mockData.ts` for demo if no prediction loaded

#### `AdvisorChat.jsx`
- Full SSE streaming handler
- Parses `__status__` prefixed events → pipeline progress list (`✓` for completed, `→` for active)
- Live elapsed timer (updates every 100ms)
- `⚡ X.Xs` generation time badge shown on completed messages
- Suggestion chips: "Will I lose my job?", "Which sectors are hiring now?", "What aid programs can help me?", "How long until the job market recovers?"

### Vite Proxy Configuration (`frontend/vite.config.ts`)

```typescript
server: {
  hmr: process.env.DISABLE_HMR !== 'true',
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
},
```

This eliminates all CORS issues — the browser only ever talks to port 3000.

---

## 8. Setup & Running

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) Ollama if using local LLM

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

**Create your `.env` file** (already present — just fill in your key):

```env
# LLM Provider
LLM_PROVIDER=claude          # "claude" or "local" (Ollama)
ANTHROPIC_API_KEY=sk-ant-... # Required if LLM_PROVIDER=claude

# Ollama (only if LLM_PROVIDER=local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b       # Recommend llama3.2:3b or larger

# RAG / ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=6

# CORS
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

**Ingest the knowledge base** (run once, or after adding new documents):

```bash
cd backend
python -m rag.ingest
```

Expected output:
```
Ingesting 40 knowledge documents...
Ingesting 148 model prediction narratives...
Ingesting 142 forecast profiles...
Done. Collection 'disaster_kb' has 441 chunks.
```

**Start the backend:**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Expected startup output:
```
Starting DisasterShift API...
Loading embedding model: all-MiniLM-L6-v2...
Embedding model ready (dim=384)
Initializing ChromaDB at: ./chroma_db
ChromaDB ready — collection 'disaster_kb' has 441 chunks
Model service: loaded 148 pre-computed predictions across 87 state×disaster combos
Model service: loaded 74 Prophet forecasts from prophet_state_forecasts.json
SQL engine: 707 sector rows + 74 forecast rows loaded into SQLite
All services ready.
INFO: Uvicorn running on http://127.0.0.1:8000
```

**API docs available at**: `http://localhost:8000/docs`

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`. API calls to `/api/*` are proxied to the backend automatically.

---

### Running the Data Pipeline (Offline — Model Retraining)

If you want to retrain the XGBoost model from scratch:

```bash
cd data
python step1_clean_fema.py       # → fema_clean.csv (45K rows)
python step2_clean_jobs.py       # → jobs_clean.csv (125K rows)
python step3_merge.py            # → merged_disaster_jobs.csv (34K rows)
python step4_features.py         # → features.csv (11.8K rows × 85 cols)
python step5_train.py            # → xgb_model.joblib (prints CV results)
python step6_export.py           # → backend/data/model_predictions.json
```

For Prophet retraining:
```bash
cd disaster_forecast
pip install -r requirements_forecast.txt
python prophet_forecast.py       # → prophet_state_forecasts.json
python generate_rag_profiles.py  # → ../backend/data/forecast_profiles.json
```

---

## 9. Design Decisions & Assumptions

### 3-Window Post-Disaster Structure

A single 6-month window misses delayed effects: construction booms during window 2 while retail collapsed in window 1. By measuring three consecutive 6-month windows, the model learns the full shock → recovery → normalization arc.

Results validate this: average excess exits decrease across windows (0.31 → 0.20 → 0.03), and `window_num` appears in the top 15 features, confirming the model learned this temporal pattern.

### Baseline: Average of 2yr + 3yr Prior

**Why not 1-year prior?** With a 3-window structure, window 3 (12–18 months post-disaster) using a 1-year baseline would overlap with window 1 (0–6 months post-disaster) of the same event — the baseline period would include disaster-affected data. Additionally, the prior year may have had its own economic shocks (COVID 2020, etc.).

**Solution**: Average the same windows from 2 and 3 years ago. Tested adding a 4th year:

| Baseline | MAE | R² |
|---|---|---|
| 2yr + 3yr avg | 0.636 | 0.887 |
| 2yr + 3yr + 4yr avg | 0.634 | 0.895 |

Difference is negligible (0.008 R²). Parsimony wins — simpler assumption, fewer years of potentially-stale economic data.

### Major Disasters Only (DR Declarations)

Only FEMA Major Disaster (DR) declarations are used. Emergency (EM) and Fire Management (FM) declarations are excluded — they are typically too small and too localized to produce measurable employment effects at the county level.

### Sector Aggregation: 302 Industries → 17 Sectors

At industry level: 9,639 rows, many FIPS/industry combos with 2–3 exits — statistically meaningless. At sector level: 3,417 rows, 17 sectors — strong signal, 54:1 feature-to-row ratio (healthy for ML).

### XGBoost Over LSTM

XGBoost is designed for mixed-type tabular data with a moderate number of rows. LSTM needs long time sequences and large datasets. With only 6 time steps and 3,417 rows, LSTM cannot learn meaningful gate weights. XGBoost won by a large margin (MAE 0.636 vs 2.583) and was chosen as the production model.

### Pre-Computed Predictions (No Runtime Inference)

The XGBoost model runs offline during `step6_export.py` for all disaster × FIPS combinations and saves to JSON. The backend serves these pre-computed results at request time — zero ML inference per API call. This makes the API fast, predictable, and deployable without GPU/heavy dependencies.

### Four-Track RAG (Prophet + XGBoost + SQL + ChromaDB)

Pure vector search (ChromaDB) cannot aggregate across 148 rows — it cannot answer "which 10 sectors are most at risk nationally?" Pure SQL cannot answer "what unemployment programs exist in Florida?" Both are needed. Prophet and XGBoost outputs are *guaranteed* to be in the prompt (direct injection, not retrieval-dependent). SQL results are injected when a structured query pattern is detected. ChromaDB adds context from programs and guidance documents.

---

## 10. Known Limitations

| Limitation | Impact | Potential Fix |
|---|---|---|
| Baseline year may not be "normal" | Disasters during baseline inflate loss estimates | Multi-year baseline averaging (partially implemented) |
| Fixed 6-month windows for all disasters | Wildfires hit immediately; floods take 9+ months | Variable windows per disaster type |
| Job endings include non-disaster causes | Noise in excess exits (people who quit for other reasons) | Causal inference methods; propensity scoring |
| Small counts at FIPS/sector level | 2→5 endings looks like signal but may be noise | Statistical significance filtering before training |
| Only 75 FIPS codes in job data | Model sees only major metro areas | Expand job data to rural counties |
| gemma3:1b echoes system prompt headers | Unusable for complex prompt following | Use Claude API (default) or llama3.2:3b+ |
| Prophet assumes stable seasonality | Category 5+ hurricanes break historical patterns | Exogenous regressors (ENSO, sea surface temp) |

---

## 11. Key Findings

### Hurricanes Show Negative Excess Exits

Average excess exits for hurricanes: **−0.17** (fewer people left jobs than baseline). This counterintuitive result has several explanations:
- People hold onto jobs tighter during crisis periods
- Reconstruction and recovery jobs absorb displaced workers
- Employers delay layoffs during federally declared emergencies
- Federal disaster aid temporarily props up businesses

This is a strong narrative finding: hurricane impact on employment is more complex than simple job loss.

### Recovery Is Non-Linear

The 3-window structure reveals that sector recovery follows different timelines:
- **Retail & Hospitality**: Sharp shock in window 1, slow recovery (12–18 months)
- **Construction & Real Estate**: Often *gains* workers in window 2 (rebuilding demand)
- **Healthcare**: Moderate initial drop, fastest recovery (emergency workers in demand)
- **Government**: Near-zero impact (public sector employment is protected)

### Top Risk Score: Prophet × XGBoost Cross-Join

Multiplying monthly disaster frequency (Prophet) × average job loss percentage (XGBoost) produces a composite risk score. This reveals which state × sector × disaster combinations represent the highest aggregate workforce risk nationally — a metric no single model can produce alone.

---

## 12. Model Performance

### XGBoost Production Model

| Experiment | MAE | R² | Improvement |
|---|---|---|---|
| Single-window baseline (predict mean) | 4.141 | — | — |
| Single-window LSTM | 2.583 | 0.20–0.77 | 37.6% |
| Single-window XGBoost | 1.275 | 0.59–0.93 | 69.2% |
| **3-window XGBoost (production)** | **0.636 ± 0.267** | **0.887 ± 0.089** | **83.2%** |

The 3-window structure alone reduced XGBoost MAE by 50% (1.275 → 0.636).

### Prophet Forecast Model

| State × Disaster | Total Historical | CV MAE | Peak Months |
|---|---|---|---|
| FL Hurricane | 856 | 4.88 | Sep, Aug, Oct |
| CA Fire | 312 | varies | Jul, Aug, Sep |
| TX Severe Storm | varies | varies | Apr, May, Jun |

Cross-validation uses time-series holdout (not random split) — Prophet is evaluated on future periods it never saw during training.

---

## 13. Data Coverage

### XGBoost (Employment Impact Model)

- **FIPS codes**: 75 unique counties (major metro areas)
- **Disasters modeled**: 2,638 unique FEMA DR declarations (2000–2026)
- **Sectors**: 17 (see list above)
- **Disaster types**: 13 (biological, coastal_storm, earthquake, fire, flood, freezing, hurricane, severe_ice_storm, severe_storm, snowstorm, tornado, tropical_storm, winter_storm)
- **States in model output**: 29 (via FIPS-to-state mapping)

### Prophet (Disaster Frequency Model)

- **State × Disaster combos**: 142 with sufficient history
- **States covered**: 41
- **Disaster types**: 6 (Fire, Flood, Hurricane, Severe Storm, Tornado, Typhoon)
- **Historical data**: 2000–2026 (312 months)
- **Forecast horizon**: 72 months (2026–2032)

### Both Models (Overlap — Supports SQL Cross-Join)

21 states appear in both models: FL, GA, IL, IN, LA, MI, MN, MO, NC, NY, OH, OK, OR, PA, TN, TX, VA, WA, WI, AZ, CA

The SQL cross-join risk score (`query_top_risk_combos`) only works for these overlapping states.

### Knowledge Base

- **Documents**: 40+ markdown files
- **ChromaDB chunks**: ~441
- **State unemployment guides**: 29 states
- **FEMA program guides**: 3 (flood, hurricane, wildfire)
- **Benefits / program guides**: 7 (COBRA, WARN Act, retraining, transferable skills, recovery timelines, financial aid)

---

## 14. Tech Stack Summary

### Backend

| Component | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ML — employment impact | XGBoost + scikit-learn + joblib |
| ML — disaster frequency | Prophet (Facebook) |
| Vector database | ChromaDB (PersistentClient) |
| Embedding model | SentenceTransformers `all-MiniLM-L6-v2` (384-dim) |
| Structured queries | SQLite (in-memory, via Python stdlib `sqlite3`) |
| LLM — cloud | Anthropic Claude API (`claude-opus-4-6`) |
| LLM — local | Ollama (any model, default `gemma3:1b`) |
| Data processing | Pandas, NumPy |
| Config management | Pydantic Settings + python-dotenv |
| Streaming | Server-Sent Events (SSE) via FastAPI StreamingResponse |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 19 + TypeScript |
| Build tool | Vite 6 (dev on port 3000) |
| Styling | Tailwind CSS 4 |
| Charts | Recharts 3 |
| Maps | Leaflet 1.9 + react-leaflet 5 |
| Icons | Lucide React |
| Animations | Motion 12 |
| CSS utilities | clsx + tailwind-merge |

### Infrastructure

| Component | Technology |
|---|---|
| Version control | Git |
| Dev backend | `uvicorn main:app --reload --port 8000` |
| Dev frontend | `npm run dev` (Vite, port 3000) |
| LLM (production) | Anthropic Claude API |
| LLM (development) | Ollama local (gemma3:1b or llama3.2:3b+) |

---

## Project Structure

```
UCSB-Datathon-2026/
├── backend/
│   ├── .env                         # LLM provider + API key (not committed)
│   ├── config.py                    # Pydantic settings
│   ├── main.py                      # FastAPI app + lifespan startup
│   ├── requirements.txt
│   ├── data/
│   │   ├── model_predictions.json   # XGBoost pre-computed predictions
│   │   ├── forecast_profiles.json   # Prophet RAG narratives
│   │   └── knowledge/               # 40+ markdown KB documents
│   ├── ml/
│   │   └── predict.py               # Runtime prediction helpers
│   ├── rag/
│   │   ├── embeddings.py            # SentenceTransformer wrapper
│   │   ├── ingest.py                # ChromaDB ingestion script
│   │   ├── prompts.py               # System prompt + audience detection
│   │   ├── query_router.py          # Keyword → SQL query routing
│   │   ├── retriever.py             # ChromaDB retrieval + metadata filtering
│   │   ├── sql_engine.py            # In-memory SQLite aggregation layer
│   │   └── vectorstore.py           # ChromaDB client init
│   ├── routers/
│   │   ├── analytics.py             # GET /api/analytics
│   │   ├── chat.py                  # POST /api/chat (SSE streaming)
│   │   ├── disasters.py             # GET /api/disasters
│   │   ├── forecast.py              # GET /api/forecast/*
│   │   └── predict.py               # POST /api/predict
│   └── services/
│       ├── chat_service.py          # RAG pipeline orchestrator
│       ├── data_service.py          # Disaster event data access
│       └── model_service.py         # Prediction + forecast loaders + SQL init
│
├── data/                            # Offline ML pipeline
│   ├── step1_clean_fema.py
│   ├── step2_clean_jobs.py
│   ├── step3_merge.py
│   ├── step4_features.py
│   ├── step5_train.py
│   ├── step6_export.py
│   ├── fema_clean.csv               # 45,198 rows
│   ├── jobs_clean.csv               # 125,434 rows
│   ├── merged_disaster_jobs.csv     # 34,858 rows
│   ├── features.csv                 # 11,797 rows × 85 cols
│   ├── predictions.csv              # Full model predictions
│   ├── feature_importance.csv       # XGBoost feature importances
│   ├── xgb_model.joblib             # Trained XGBoost model
│   └── industry_map.csv             # 302 industries → 17 sectors
│
├── disaster_forecast/               # Prophet forecasting pipeline
│   ├── prophet_forecast.py          # Main Prophet training + export
│   ├── generate_rag_profiles.py     # Narrative summaries for RAG
│   ├── chart_data.py                # Recharts-format converter
│   ├── state_disaster_selector.py   # Filter combos with enough data
│   ├── prophet_state_forecasts.json # 142 combos × 72-month forecasts
│   ├── requirements_forecast.txt
│   ├── experiments/                 # Model comparison experiments
│   │   ├── model_comparison.py      # ARIMA vs NegBin vs Prophet
│   │   ├── prophet_model.py
│   │   ├── negbin_model.py
│   │   └── model_comparison_results.csv
│   └── plots/                       # Sample forecast visualizations
│       ├── FL_Hurricane.png
│       ├── CA_Fire.png
│       └── sample_grid.png
│
├── frontend/
│   ├── vite.config.ts               # Vite proxy → backend:8000
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── App.tsx                  # Root layout
│       ├── constants.ts             # App-wide constants
│       ├── mockData.ts              # Demo fallback data
│       └── components/
│           ├── Navbar.jsx
│           ├── SidebarLeft.jsx      # Disaster/state selector + prediction trigger
│           ├── DisasterMap.jsx      # Leaflet map with event markers
│           ├── KeyMetrics.jsx       # Summary stat cards
│           ├── AnalyticsTabs.jsx    # Displacement curve + Industry impact charts
│           └── AdvisorChat.jsx      # SSE chat interface + status pipeline UI
│
├── chroma_db/                       # ChromaDB vector store (auto-generated)
├── LOG.md                           # Full design decision log + experiment notes
└── README.md                        # This file
```

---

*Built for UCSB Datathon 2026.*
