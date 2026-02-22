"""
Ingestion script — run once (or whenever KB changes) to populate ChromaDB.

Usage:
    cd backend
    python -m rag.ingest

It reads:
  1. backend/data/knowledge/*.md  — curated resource documents
  2. backend/data/model_predictions.json  — ML team's pre-computed outputs

Each document is chunked, embedded, and upserted into the 'disaster_kb' collection.
"""

import json
import re
import sys
from pathlib import Path

# Add backend/ to path so relative imports work when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.embeddings import embed_texts, load_embedding_model
from rag.vectorstore import get_collection, init_vectorstore

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Maps filename prefixes → metadata categories
CATEGORY_MAP = {
    "fema": "fema",
    "ca_": "unemployment",
    "fl_": "unemployment",
    "tx_": "unemployment",
    "nc_": "unemployment",
    "unemployment": "unemployment",
    "warn": "warn_act",
    "retraining": "retraining",
    "transferable": "transferable_skills",
    "cobra": "cobra",
    "recovery": "recovery_timelines",
    "disaster_financial": "financial_aid",
}

# Maps filename keywords → disaster types
DISASTER_MAP = {
    "wildfire": "wildfire",
    "hurricane": "hurricane",
    "flood": "flood",
    "earthquake": "earthquake",
    "tornado": "tornado",
}

# Maps filename prefix → state code (all model-supported states + national fallback)
STATE_MAP = {
    # Original 5
    "ca_": "CA",
    "fl_": "FL",
    "tx_": "TX",
    "nc_": "NC",
    "la_": "LA",
    # Batch 1 additions
    "az_": "AZ",
    "co_": "CO",
    "ct_": "CT",
    "dc_": "DC",
    "ga_": "GA",
    "il_": "IL",
    "in_": "IN",
    "ma_": "MA",
    "md_": "MD",
    "mi_": "MI",
    # Batch 2 additions
    "mn_": "MN",
    "mo_": "MO",
    "nv_": "NV",
    "ny_": "NY",
    "oh_": "OH",
    "ok_": "OK",
    "or_": "OR",
    "pa_": "PA",
    "ri_": "RI",
    "tn_": "TN",
    "ut_": "UT",
    "va_": "VA",
    "wa_": "WA",
    "wi_": "WI",
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def infer_metadata(filename: str) -> dict:
    """Infer category, state, disaster_type from filename."""
    name = filename.lower()
    metadata = {"category": "general", "state": None, "disaster_type": None}

    for prefix, category in CATEGORY_MAP.items():
        if name.startswith(prefix) or prefix in name:
            metadata["category"] = category
            break

    for keyword, disaster_type in DISASTER_MAP.items():
        if keyword in name:
            metadata["disaster_type"] = disaster_type
            break

    for prefix, state in STATE_MAP.items():
        if name.startswith(prefix):
            metadata["state"] = state
            break

    return metadata


def ingest_knowledge_docs(knowledge_dir: Path, collection) -> int:
    """Ingest all .md files from the knowledge directory."""
    md_files = list(knowledge_dir.glob("*.md"))
    if not md_files:
        print(f"  No .md files found in {knowledge_dir}")
        return 0

    total_chunks = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        metadata_base = infer_metadata(md_file.name)
        metadata_base["source"] = md_file.name

        # Clean up None values — ChromaDB doesn't accept None metadata values
        metadata_base = {k: v for k, v in metadata_base.items() if v is not None}

        chunks = chunk_text(text)
        ids = [f"{md_file.stem}_chunk_{i}" for i in range(len(chunks))]
        embeddings = embed_texts(chunks)
        metadatas = [metadata_base.copy() for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunk(s) [{metadata_base.get('category')}]")

    return total_chunks


def ingest_model_predictions(predictions_path: Path, collection) -> int:
    """Ingest model predictions JSON into ChromaDB."""
    if not predictions_path.exists():
        print(f"  model_predictions.json not found at {predictions_path} — skipping")
        return 0

    with open(predictions_path, encoding="utf-8") as f:
        entries = json.load(f)

    total_chunks = 0
    for entry in entries:
        text = entry.get("text", "").strip()
        if not text:
            continue

        entry_id = entry.get("id", f"pred_{entry['disaster_type']}_{entry['fips_code']}")
        metadata = {
            "category": "model_output",
            "disaster_type": entry.get("disaster_type", ""),
            "fips_code": entry.get("fips_code", ""),
            "region": entry.get("region", ""),
            "source": "model_predictions.json",
        }
        # Remove empty string values
        metadata = {k: v for k, v in metadata.items() if v}

        embeddings = embed_texts([text])

        collection.upsert(
            ids=[entry_id],
            documents=[text],
            embeddings=embeddings,
            metadatas=[metadata],
        )

        total_chunks += 1
        print(f"  Prediction [{entry.get('region', entry_id)}] ingested")

    return total_chunks


def ingest_forecast_profiles(profiles_path: Path, collection) -> int:
    """
    Ingest Prophet time series forecast profiles into ChromaDB.

    Each entry in forecast_profiles.json has:
      id, state, disaster_type, text, source

    These are narrative chunks describing seasonal disaster risk per state,
    generated by disaster_forecast/generate_rag_profiles.py.
    """
    if not profiles_path.exists():
        print(f"  forecast_profiles.json not found at {profiles_path} — skipping")
        print(f"  Run: python disaster_forecast/generate_rag_profiles.py")
        return 0

    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)

    total_chunks = 0
    for profile in profiles:
        text = profile.get("text", "").strip()
        if not text:
            continue

        profile_id = profile.get("id", f"forecast_{profile['state']}_{profile['disaster_type']}")
        metadata = {
            "category":     "forecast",
            "state":        profile.get("state", ""),
            "disaster_type": profile.get("disaster_type", ""),
            "source":       "prophet_forecast",
        }
        # Remove empty string values — ChromaDB doesn't accept them
        metadata = {k: v for k, v in metadata.items() if v}

        chunks = chunk_text(text)
        ids = [f"{profile_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = embed_texts(chunks)
        metadatas = [metadata.copy() for _ in chunks]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

    print(f"  Ingested {len(profiles)} forecast profiles → {total_chunks} chunk(s)")
    return total_chunks


def main():
    base_dir = Path(__file__).parent.parent
    knowledge_dir = base_dir / "data" / "knowledge"
    predictions_path = base_dir / "data" / "model_predictions.json"
    forecast_profiles_path = base_dir / "data" / "forecast_profiles.json"

    print("=== DisasterShift KB Ingestion ===")
    load_embedding_model()
    init_vectorstore()
    collection = get_collection()

    print(f"\n[1/3] Ingesting knowledge docs from {knowledge_dir}...")
    kb_chunks = ingest_knowledge_docs(knowledge_dir, collection)

    print(f"\n[2/3] Ingesting model predictions from {predictions_path}...")
    pred_chunks = ingest_model_predictions(predictions_path, collection)

    print(f"\n[3/3] Ingesting forecast profiles from {forecast_profiles_path}...")
    forecast_chunks = ingest_forecast_profiles(forecast_profiles_path, collection)

    total = kb_chunks + pred_chunks + forecast_chunks
    print(f"\n=== Done: {total} total chunks ingested ===")
    print(f"    Collection '{collection.name}' now has {collection.count()} chunks")


if __name__ == "__main__":
    main()
