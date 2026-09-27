"""
Document Ingestion Pipeline for Travel Planner System
Ingests structured travel guides into local embedded Qdrant vector database (./qdrant_data).
"""

import os
import sys
import glob
import re
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

COLLECTION_NAME = "travel_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension


def parse_markdown_doc(file_path: str):
    """
    Parses a markdown travel document with sections:
    # Destination: <Name>
    ## Category: <Category>
    <Content>
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract default destination and normalize
    dest_match = re.search(r"^#\s*Destination:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
    default_dest = dest_match.group(1).strip().title() if dest_match else "General"

    # Split sections by ## Category:
    sections = re.split(r"(?=^##\s*Category:)", content, flags=re.MULTILINE | re.IGNORECASE)
    docs = []

    for section in sections:
        section = section.strip()
        if not section or section.startswith("# Destination:"):
            continue

        cat_match = re.search(r"^##\s*Category:\s*(.+)$", section, re.MULTILINE | re.IGNORECASE)
        category = cat_match.group(1).strip().title() if cat_match else "General"

        # Remove header line to get body
        body = re.sub(r"^##\s*Category:\s*.+$", "", section, flags=re.MULTILINE | re.IGNORECASE).strip()
        if not body:
            continue

        docs.append({
            "destination": default_dest,
            "category": category,
            "text": body,
            "source_file": os.path.basename(file_path)
        })

    return docs


def get_qdrant_client():
    return QdrantClient(path="./qdrant_data")


def ingest():
    client = get_qdrant_client()

    # Initialize FastEmbed Dense Embedder
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # Setup Qdrant collection with clean fresh indexing
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Recreating collection '{COLLECTION_NAME}' for fresh indexing...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}' (dimension: {VECTOR_SIZE}, Cosine)...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        ),
        # Enable scalar quantization for memory efficiency
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )
    # Create payload indexes for pre-filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="destination",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print("✅ Collection and payload indices created successfully.")

    # Load and chunk documents (1000-char chunks for rich cross-encoder context)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    docs_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge")
    doc_files = glob.glob(os.path.join(docs_dir, "*.md"))
    print(f"Found {len(doc_files)} knowledge documents in {docs_dir}")

    all_chunks = []
    for file_path in doc_files:
        sections = parse_markdown_doc(file_path)
        for sec in sections:
            split_texts = splitter.split_text(sec["text"])
            for idx, chunk_text in enumerate(split_texts):
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "destination": sec["destination"],
                    "category": sec["category"],
                    "source_file": sec["source_file"],
                    "chunk_id": f"{sec['source_file']}_{sec['category']}_{idx}",
                    "char_count": len(chunk_text)
                })

    print(f"Generated {len(all_chunks)} semantic chunks. Generating dense embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = list(embed_model.embed(texts))

    points = [
        models.PointStruct(
            id=chunk["id"],
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "destination": chunk["destination"],
                "category": chunk["category"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "char_count": chunk["char_count"]
            }
        )
        for chunk, embedding in zip(all_chunks, embeddings)
    ]

    print(f"Upserting {len(points)} points into Qdrant collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"✅ Ingestion complete! {len(points)} points indexed in Qdrant.")


if __name__ == "__main__":
    ingest()