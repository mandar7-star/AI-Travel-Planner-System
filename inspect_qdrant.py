"""
Qdrant collection inspection utility.

Verifies the embedded Qdrant collection is populated and shows
high-level stats.

Usage:
    python inspect_qdrant.py
"""

from collections import Counter
from qdrant_client import QdrantClient

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "travel_knowledge"


def inspect():
    client = QdrantClient(path=QDRANT_PATH, check_compatibility=False)

    try:
        info = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' not found: {e}")
        print("Run `python ingest_knowledge.py` first.")
        return

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )

    dests = Counter(p.payload.get("destination", "?") for p in points)
    cats = Counter(p.payload.get("category", "?") for p in points)
    lengths = [len(p.payload.get("text", "")) for p in points]
    avg_len = sum(lengths) // len(lengths) if lengths else 0

    print("=" * 60)
    print(f"  QDRANT — {COLLECTION_NAME}")
    print("=" * 60)
    print(f"  Total points        : {info.points_count}")
    print(f"  Vector dimension    : {info.config.params.vectors.size}")
    print(f"  Distance metric     : {info.config.params.vectors.distance}")
    print(f"  Unique destinations : {len(dests)}")
    print(f"  Unique categories   : {len(cats)}")
    print(f"  Average chunk length: {avg_len} chars")
    print(f"  Min / Max chunk     : {min(lengths)} / {max(lengths)} chars")
    print("=" * 60)

if __name__ == "__main__":
    inspect()