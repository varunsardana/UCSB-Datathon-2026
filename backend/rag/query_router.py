"""
Query Router — classifies the user's question and runs the right SQL query.

Two tracks:
  STRUCTURED  → SQL against SQLite (rankings, comparisons, portfolio, variance)
  RAG         → ChromaDB retrieval (programs, eligibility, guidance, general)

Most questions hit both tracks. The router decides what SQL pre-context to add.
"""

import re
from rag import sql_engine

# ── Keyword sets ───────────────────────────────────────────────────────────────

_TOP_RANK = {
    "top 10", "top 5", "top ten", "top five", "top cities", "top states",
    "top sectors", "rank", "ranked", "ranking", "which sectors recover fastest",
    "which sectors recover slowest", "fastest recovery", "slowest recovery",
    "most at risk", "highest risk", "least risk", "safest sector",
    "compare sectors", "sector comparison",
}
_SECTOR_RANK = {
    "which sector", "which industry", "recover fastest", "recover slowest",
    "recover quickest", "quickest recovery", "worst hit", "hardest hit",
    "most affected", "least affected", "bounce back",
}
_PORTFOLIO = {
    "portfolio", "southeast", "gulf coast", "west coast", "midwest", "northeast",
    "across our", "across states", "multiple states", "all states",
    "real estate holdings", "our properties", "our exposure",
}
_VARIANCE = {
    "variance", "reliable", "reliability", "confidence", "uncertain",
    "how accurate", "least reliable", "most reliable", "how sure",
    "how certain", "prediction accuracy", "model accuracy",
}
_DEMAND_SURGE = {
    "gain workers", "gain jobs", "gaining jobs", "hiring", "increase jobs",
    "which sectors grow", "which sectors hire", "opportunity after",
    "benefit from disaster", "boom after", "construction boom",
    "who benefits", "which jobs increase",
}
_PREPOSITION = {
    "pre-position", "preposition", "where should we", "deploy resources",
    "prioritize retraining", "where to focus", "resource allocation",
    "intervention priority", "forecast risk", "next 18 months", "next year risk",
    "upcoming risk", "prepare for",
}


def _match(text: str, keywords: set[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _extract_states(text: str) -> list[str]:
    """Pull 2-letter state codes from the question."""
    all_states = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID",
        "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO",
        "MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
        "RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
    }
    found = re.findall(r'\b([A-Z]{2})\b', text.upper())
    return [s for s in found if s in all_states]


def _extract_disaster(text: str) -> str | None:
    t = text.lower()
    for kw, dt in [
        ("hurricane", "hurricane"), ("flood", "flood"), ("wildfire", "fire"),
        ("fire", "fire"), ("tornado", "tornado"), ("earthquake", "earthquake"),
        ("storm", "severe_storm"), ("severe storm", "severe_storm"),
    ]:
        if kw in t:
            return dt
    return None


def _extract_region_states(text: str) -> list[str]:
    """Map named regions to their state lists."""
    t = text.lower()
    for region_name, states in sql_engine.REGION_GROUPS.items():
        if region_name.replace("_", " ") in t or region_name in t:
            return states
    return []


# ── Main router ────────────────────────────────────────────────────────────────

def route_and_query(
    question: str,
    state: str | None = None,
    disaster_type: str | None = None,
) -> str:
    """
    Classify the question and run the appropriate SQL query.
    Returns a formatted string (may be empty if no SQL query applies).
    The result is injected into the prompt as STRUCTURED DATA context.
    """
    text = question.lower()

    # ── Top-N weighted risk — the marquee cross-model question ─────────────
    if _match(text, _TOP_RANK) and any(
        kw in text for kw in ["city", "state", "combined", "overall", "weighted", "10", "5", "ten", "five"]
    ):
        return sql_engine.query_top_risk_combos(limit=10)

    # ── Pre-positioning / resource allocation across the country ────────────
    if _match(text, _PREPOSITION):
        return sql_engine.query_preposition(limit=10)

    # ── Portfolio / multi-state aggregation ─────────────────────────────────
    if _match(text, _PORTFOLIO):
        states = _extract_region_states(question) or _extract_states(question)
        dtype = disaster_type or _extract_disaster(question) or "hurricane"
        if states:
            return sql_engine.query_portfolio(states, dtype)

    # ── Variance / reliability ───────────────────────────────────────────────
    if _match(text, _VARIANCE):
        return sql_engine.query_variance()

    # ── Demand surge — which sectors gain workers ────────────────────────────
    if _match(text, _DEMAND_SURGE):
        dtype = disaster_type or _extract_disaster(question)
        return sql_engine.query_demand_surge(dtype)

    # ── Sector ranking within a state × disaster ─────────────────────────────
    if _match(text, _SECTOR_RANK) or _match(text, _TOP_RANK):
        q_state = state or ((_extract_states(question) or [None])[0])
        q_dtype = disaster_type or _extract_disaster(question)
        result = sql_engine.query_sector_ranking(q_state, q_dtype)
        if result:
            return result

    # No structured query matched — return empty, fall through to RAG only
    return ""
