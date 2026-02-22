"""
SQL Engine — in-memory SQLite over the two model output JSONs.

Loaded once at startup (alongside ChromaDB). Handles ranking, comparison,
portfolio aggregation, and weighted-risk queries that pure RAG can't do.

Tables:
  employment_impact   — XGBoost sector predictions (148 rows)
  disaster_forecast   — Prophet state×disaster forecasts (74 rows)
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

_conn: Optional[sqlite3.Connection] = None

FIPS_STATE_MAP = {
    "04": "AZ", "06": "CA", "08": "CO", "09": "CT", "11": "DC",
    "12": "FL", "13": "GA", "17": "IL", "18": "IN", "22": "LA",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "29": "MO",
    "32": "NV", "36": "NY", "37": "NC", "39": "OH", "40": "OK",
    "41": "OR", "42": "PA", "44": "RI", "47": "TN", "48": "TX",
    "49": "UT", "51": "VA", "53": "WA", "55": "WI",
}

PROPHET_TYPE_MAP = {
    "Fire": "fire", "Flood": "flood", "Hurricane": "hurricane",
    "Severe_Storm": "severe_storm", "Tornado": "tornado", "Typhoon": "typhoon",
}

# Southeast / Gulf Coast state groups for portfolio queries
REGION_GROUPS = {
    "southeast": ["FL", "GA", "NC", "SC", "VA", "AL", "MS", "TN"],
    "gulf_coast": ["FL", "LA", "TX", "MS", "AL"],
    "midwest":   ["IL", "IN", "MI", "MN", "MO", "OH", "WI"],
    "west_coast": ["CA", "OR", "WA"],
    "northeast": ["NY", "PA", "MA", "CT", "RI"],
    "southwest": ["AZ", "TX", "NM", "CO", "UT"],
}


# ── Initialisation ─────────────────────────────────────────────────────────────

def init_db(predictions_path: Path, prophet_path: Path) -> None:
    global _conn
    _conn = sqlite3.connect(":memory:", check_same_thread=False)
    _conn.row_factory = sqlite3.Row

    _conn.executescript("""
        CREATE TABLE employment_impact (
            fips_code     TEXT,
            state         TEXT,
            disaster_type TEXT,
            region        TEXT,
            sector        TEXT,
            job_loss_pct  REAL,
            job_change_pct REAL,
            recovery_months REAL,
            peak_month    INTEGER
        );

        CREATE TABLE disaster_forecast (
            state            TEXT,
            disaster_type    TEXT,
            total_historical INTEGER,
            peak_months      TEXT,
            avg_next_12      REAL,
            cv_mae           REAL
        );

        CREATE INDEX idx_ei_state   ON employment_impact(state, disaster_type);
        CREATE INDEX idx_df_state   ON disaster_forecast(state, disaster_type);
    """)

    # ── Load XGBoost predictions ───────────────────────────────────────────────
    if predictions_path.exists():
        with open(predictions_path, encoding="utf-8") as f:
            entries = json.load(f)

        rows = []
        for entry in entries:
            fips = str(entry.get("fips_code", ""))
            state = FIPS_STATE_MAP.get(fips[:2], "")
            region = entry.get("region", fips)
            disaster = entry.get("disaster_type", "")
            for sector, data in entry.get("predictions", {}).items():
                rows.append((
                    fips, state, disaster, region, sector,
                    data.get("job_loss_pct"),
                    data.get("job_change_pct"),
                    data.get("recovery_months"),
                    data.get("peak_month"),
                ))
        _conn.executemany("INSERT INTO employment_impact VALUES (?,?,?,?,?,?,?,?,?)", rows)

    # ── Load Prophet forecasts ─────────────────────────────────────────────────
    if prophet_path.exists():
        with open(prophet_path, encoding="utf-8") as f:
            prophet = json.load(f)

        rows = []
        for key, entry in prophet.items():
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            state = parts[0]
            dtype = PROPHET_TYPE_MAP.get(parts[1], parts[1].lower())
            info = entry.get("model_info", {})
            forecast = entry.get("forecast", {})
            predicted = forecast.get("predicted_counts", [])[:12]
            avg_next_12 = round(sum(predicted) / len(predicted), 3) if predicted else 0.0
            rows.append((
                state, dtype,
                info.get("total_historical", 0),
                json.dumps(info.get("peak_months", [])),
                avg_next_12,
                info.get("cv_mae"),
            ))
        _conn.executemany("INSERT INTO disaster_forecast VALUES (?,?,?,?,?,?)", rows)

    _conn.commit()
    ei_count = _conn.execute("SELECT COUNT(*) FROM employment_impact").fetchone()[0]
    df_count = _conn.execute("SELECT COUNT(*) FROM disaster_forecast").fetchone()[0]
    print(f"SQL engine: {ei_count} sector rows + {df_count} forecast rows loaded into SQLite")


def _get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("SQL engine not initialised. Call init_db() first.")
    return _conn


# ── Query 1: Sector ranking for a given state × disaster ──────────────────────

def query_sector_ranking(state: str | None, disaster_type: str | None) -> str:
    """
    Rank sectors by job loss severity and recovery speed.
    Returns a formatted table the LLM can summarise.
    """
    conn = _get_conn()
    params: list = []
    where = []
    if state:
        where.append("state = ?"); params.append(state.upper())
    if disaster_type:
        where.append("disaster_type = ?"); params.append(disaster_type.lower())
    where_clause = "WHERE " + " AND ".join(where) if where else ""

    # Job loss sectors — ranked worst to least
    loss_rows = conn.execute(f"""
        SELECT sector, AVG(job_loss_pct) as loss, AVG(recovery_months) as recovery
        FROM employment_impact
        {where_clause} AND job_loss_pct IS NOT NULL
        GROUP BY sector
        ORDER BY loss DESC
    """, params).fetchall()

    # Demand-surge sectors
    surge_rows = conn.execute(f"""
        SELECT sector, AVG(job_change_pct) as surge, AVG(peak_month) as peak
        FROM employment_impact
        {where_clause} AND job_change_pct IS NOT NULL
        GROUP BY sector
        ORDER BY surge DESC
    """, params).fetchall()

    if not loss_rows and not surge_rows:
        return ""

    context = f"Sector Employment Rankings"
    if state and disaster_type:
        context += f" — {state} {disaster_type.replace('_', ' ').title()}"
    context += ":\n"

    if loss_rows:
        context += "\nSectors losing jobs (from most to least severe):\n"
        for i, r in enumerate(loss_rows, 1):
            loss = f"{r['loss']:.0f}%"
            rec = f"{r['recovery']:.0f} months" if r['recovery'] else "unknown"
            context += f"  {i}. {r['sector']:<28} — {loss} job loss, recovers in {rec}\n"

    if surge_rows:
        context += "\nSectors gaining workers (demand surge from rebuilding/response):\n"
        for i, r in enumerate(surge_rows, 1):
            peak = f"month {r['peak']:.0f}" if r['peak'] else "unknown timing"
            context += f"  {i}. {r['sector']:<28} — +{r['surge']:.0f}% demand increase, peaks at {peak}\n"

    return context.strip()


# ── Query 2: Top N state-sector-disaster combos by weighted risk ───────────────

def query_top_risk_combos(limit: int = 10) -> str:
    """
    Cross-join Prophet frequency × XGBoost job loss → weighted risk score.
    The marquee 'top 10 most at risk city-sector combinations' query.
    """
    conn = _get_conn()
    rows = conn.execute(f"""
        SELECT
            ei.state,
            ei.sector,
            ei.disaster_type,
            ROUND(AVG(ei.job_loss_pct), 1)    AS avg_loss_pct,
            ROUND(AVG(ei.recovery_months), 0) AS avg_recovery,
            ROUND(df.avg_next_12, 2)           AS freq_per_month,
            ROUND(AVG(ei.job_loss_pct) * df.avg_next_12, 1) AS risk_score
        FROM employment_impact ei
        JOIN disaster_forecast df
          ON ei.state = df.state AND ei.disaster_type = df.disaster_type
        WHERE ei.job_loss_pct IS NOT NULL
          AND df.avg_next_12  IS NOT NULL
        GROUP BY ei.state, ei.sector, ei.disaster_type
        ORDER BY risk_score DESC
        LIMIT ?
    """, [limit]).fetchall()

    if not rows:
        return ""

    lines = [f"Top {limit} Highest-Risk State × Sector × Disaster Combinations",
             "(Risk Score = Job Loss % × Avg Monthly Disaster Frequency)", ""]
    for i, r in enumerate(rows, 1):
        disaster_label = r['disaster_type'].replace("_", " ").title()
        lines.append(
            f"  {i:>2}. {r['state']} · {r['sector']} · {disaster_label}"
        )
        lines.append(
            f"       Job loss: {r['avg_loss_pct']}% | Recovery: {r['avg_recovery']} months "
            f"| Frequency: {r['freq_per_month']} events/month | Risk score: {r['risk_score']}"
        )
    return "\n".join(lines)


# ── Query 3: Portfolio aggregation across a list of states ────────────────────

def query_portfolio(states: list[str], disaster_type: str) -> str:
    """
    Aggregate sector-level risk across a portfolio of states for one disaster type.
    """
    conn = _get_conn()
    placeholders = ",".join("?" * len(states))
    rows = conn.execute(f"""
        SELECT
            ei.sector,
            ROUND(AVG(ei.job_loss_pct), 1)     AS avg_loss,
            ROUND(MAX(ei.job_loss_pct), 1)     AS worst_loss,
            ROUND(AVG(ei.recovery_months), 0)  AS avg_recovery,
            COUNT(DISTINCT ei.state)           AS states_covered
        FROM employment_impact ei
        WHERE ei.state IN ({placeholders})
          AND ei.disaster_type = ?
          AND ei.job_loss_pct IS NOT NULL
        GROUP BY ei.sector
        ORDER BY avg_loss DESC
    """, [*states, disaster_type.lower()]).fetchall()

    if not rows:
        return ""

    disaster_label = disaster_type.replace("_", " ").title()
    state_list = ", ".join(states)
    lines = [
        f"Portfolio Workforce Risk — {disaster_label} exposure across: {state_list}",
        ""
    ]
    for r in rows:
        covered = f"{r['states_covered']}/{len(states)} states"
        lines.append(
            f"  {r['sector']:<30} avg {r['avg_loss']}% loss "
            f"(worst: {r['worst_loss']}%), {r['avg_recovery']}mo recovery [{covered}]"
        )
    return "\n".join(lines)


# ── Query 4: Recovery variance — prediction reliability ───────────────────────

def query_variance() -> str:
    """
    Show which sectors have the most variation in recovery time across
    different states and disaster types (= least reliable predictions).
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            sector,
            ROUND(AVG(recovery_months), 1)                            AS avg_recovery,
            ROUND(MAX(recovery_months) - MIN(recovery_months), 1)     AS range_months,
            ROUND(AVG(job_loss_pct), 1)                               AS avg_loss,
            COUNT(*)                                                   AS n_data_points
        FROM employment_impact
        WHERE recovery_months IS NOT NULL
        GROUP BY sector
        HAVING n_data_points >= 2
        ORDER BY range_months DESC
    """).fetchall()

    if not rows:
        return ""

    lines = ["Sector Recovery Variance (high range = less reliable predictions):", ""]
    for i, r in enumerate(rows, 1):
        reliability = "HIGH variance" if r['range_months'] > 9 else (
            "MEDIUM variance" if r['range_months'] > 4 else "LOW variance"
        )
        lines.append(
            f"  {i:>2}. {r['sector']:<30} avg {r['avg_recovery']}mo recovery, "
            f"range: {r['range_months']}mo spread — {reliability} ({r['n_data_points']} data points)"
        )
    return "\n".join(lines)


# ── Query 5: Sectors that gain workers after disasters ────────────────────────

def query_demand_surge(disaster_type: str | None = None) -> str:
    """
    List sectors that see positive labor demand after a disaster.
    """
    conn = _get_conn()
    params: list = []
    where = "WHERE job_change_pct IS NOT NULL"
    if disaster_type:
        where += " AND disaster_type = ?"; params.append(disaster_type.lower())

    rows = conn.execute(f"""
        SELECT
            sector,
            disaster_type,
            ROUND(AVG(job_change_pct), 1) AS avg_surge,
            ROUND(AVG(peak_month), 0)     AS avg_peak,
            COUNT(DISTINCT state)          AS states
        FROM employment_impact
        {where}
        GROUP BY sector, disaster_type
        ORDER BY avg_surge DESC
    """, params).fetchall()

    if not rows:
        return ""

    label = f" after {disaster_type.replace('_',' ').title()}" if disaster_type else ""
    lines = [f"Sectors gaining workers{label} (labor demand increases):", ""]
    for r in rows:
        disaster_label = r['disaster_type'].replace("_", " ").title()
        lines.append(
            f"  {r['sector']:<30} +{r['avg_surge']}% demand surge{label or ' (' + disaster_label + ')'}, "
            f"peaks at month {r['avg_peak']:.0f} ({r['states']} states)"
        )
    return "\n".join(lines)


# ── Query 6: Pre-positioning — highest risk states in next 12 months ──────────

def query_preposition(limit: int = 10) -> str:
    """
    Which states have the highest forecast disaster frequency in next 12 months
    combined with high average job loss — for pre-positioning retraining resources.
    """
    conn = _get_conn()
    rows = conn.execute(f"""
        SELECT
            df.state,
            df.disaster_type,
            ROUND(df.avg_next_12, 2)           AS freq,
            df.peak_months,
            ROUND(AVG(ei.job_loss_pct), 1)     AS avg_loss,
            ROUND(df.avg_next_12 * AVG(ei.job_loss_pct), 1) AS priority_score
        FROM disaster_forecast df
        JOIN employment_impact ei
          ON df.state = ei.state AND df.disaster_type = ei.disaster_type
        WHERE df.avg_next_12 > 0 AND ei.job_loss_pct IS NOT NULL
        GROUP BY df.state, df.disaster_type
        ORDER BY priority_score DESC
        LIMIT ?
    """, [limit]).fetchall()

    if not rows:
        return ""

    lines = [
        f"Top {limit} States/Disasters to Pre-Position Workforce Resources",
        "(Ranked by: Forecast Frequency × Average Job Loss = Priority Score)", ""
    ]
    for i, r in enumerate(rows, 1):
        disaster_label = r['disaster_type'].replace("_", " ").title()
        peaks = json.loads(r['peak_months'] or "[]")
        peak_str = f", peak: {', '.join(peaks[:2])}" if peaks else ""
        lines.append(
            f"  {i:>2}. {r['state']} — {disaster_label}{peak_str}"
        )
        lines.append(
            f"       Frequency: {r['freq']} events/month | Avg job loss: {r['avg_loss']}% | Priority: {r['priority_score']}"
        )
    return "\n".join(lines)
