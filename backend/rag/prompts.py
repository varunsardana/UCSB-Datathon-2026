SYSTEM_PROMPT_TEMPLATE = """\
You are the DisasterShift Workforce Advisor. Answer the user's question using only \
the data provided below. Do not invent numbers, URLs, phone numbers, or program names. \
If data is missing for a scenario, say so explicitly.

Audience: {audience_type}. Tailor your tone accordingly: workers get plain empathetic \
language and concrete next steps; employers get workforce planning figures; \
policymakers get intervention priorities; investors/insurers get quantified risk metrics.

Job mapping: if the user mentions a specific role, map it to the nearest sector in the \
impact data (restaurant/waiter → Retail & Hospitality; nurse/doctor → Healthcare; \
teacher → Education; contractor/builder → Construction & Real Estate; \
store/cashier → Retail & Hospitality) and tell the user which category their job falls under.

For scenario questions ("if X hits, what happens to Y"): lead with the sector impact \
figures, then the seasonal timing, then actions.
For ranking questions ("which sectors recover fastest"): present the ranked list \
directly from the structured data results — do not re-rank or modify the order.
For resource questions ("what programs are available"): list only programs, phone \
numbers, and websites that appear explicitly in the retrieved knowledge below.
For risk/portfolio questions: show the key numbers, state your uncertainty, give a \
summary metric.

Keep the response under 4 short paragraphs. No invented URLs. Only use phone numbers \
and websites that appear word-for-word in the retrieved knowledge section below.

--- DISASTER FREQUENCY FORECAST ---
{forecast_context}

--- EMPLOYMENT IMPACT PREDICTION ---
{prediction_context}

--- STRUCTURED DATA RESULTS ---
{sql_context}

--- RETRIEVED KNOWLEDGE ---
{retrieved_docs}

--- USER CONTEXT ---
State: {state} | Disaster: {disaster_type} | Job/Industry: {job_title}

Answer the user's question now.\
"""


AUDIENCE_TYPES = {"worker", "employer", "policymaker", "investor", "insurer", "unknown"}


def detect_audience(job_title: str | None, question: str | None = None) -> str:
    """
    Keyword heuristic for audience detection.
    In production the UI would pass this explicitly; this covers the default case.
    """
    if not job_title and not question:
        return "unknown"

    text = f"{job_title or ''} {question or ''}".lower()

    if any(k in text for k in ["insur", "actuar", "underwrite", "reinsur"]):
        return "insurer"
    if any(k in text for k in ["investor", "portfolio", "real estate fund", "analyst", "fund manager"]):
        return "investor"
    if any(k in text for k in ["mayor", "planner", "policy", "city council", "government", "agency director", "official"]):
        return "policymaker"
    if any(k in text for k in ["owner", "employer", "ceo", "hr ", "hiring manager", "workforce manager", "operations manager"]):
        return "employer"
    # Default displaced worker / community member
    return "worker"


def format_retrieved_docs(docs: list[dict]) -> str:
    """Format retrieved chunks with source labels for the prompt."""
    if not docs:
        return "No relevant documents retrieved."
    parts = []
    for i, doc in enumerate(docs, 1):
        category = doc["metadata"].get("category", "unknown")
        source = doc["metadata"].get("source", "")
        label = f"[Source {i} — {category}]"
        if source:
            label += f" ({source})"
        parts.append(f"{label}\n{doc['text']}")
    return "\n\n".join(parts)


def build_system_prompt(
    forecast_context: str,
    prediction_context: str,
    retrieved_docs: list[dict],
    state: str | None,
    disaster_type: str | None,
    job_title: str | None,
    question: str | None = None,
    audience_type: str | None = None,
    sql_context: str | None = None,
) -> str:
    resolved_audience = (
        audience_type
        if audience_type in AUDIENCE_TYPES
        else detect_audience(job_title, question)
    )

    return SYSTEM_PROMPT_TEMPLATE.format(
        forecast_context=(
            forecast_context
            or "No frequency forecast available for this state/disaster combination."
        ),
        prediction_context=(
            prediction_context
            or "No employment impact prediction available for this scenario."
        ),
        sql_context=(
            sql_context
            or "No structured ranking/aggregation query was needed for this question."
        ),
        retrieved_docs=format_retrieved_docs(retrieved_docs),
        state=state or "Not specified",
        disaster_type=disaster_type or "Not specified",
        job_title=job_title or "Not specified",
        audience_type=resolved_audience,
    )
