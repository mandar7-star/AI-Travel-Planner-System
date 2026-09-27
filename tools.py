"""
Tools module for AI Travel Planner System.
"""

import os
import sys
import json
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

COLLECTION_NAME = "travel_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Initialize Groq LLM
llm = ChatGroq(model=GROQ_MODEL)


# Initialize shared local embedding and reranker instances (lazy/cached)
_qdrant_client = None
_embed_model = None
_reranker = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path="./qdrant_data")
    return _qdrant_client


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embed_model


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    return _reranker


CITY_TO_COUNTRY = {
    "Paris": "France",
    "Tokyo": "Japan",
    "Kyoto": "Japan",
    "Bangkok": "Thailand",
    "Bali": "Indonesia",
    "Dubai": "UAE",
    "London": "UK",
    "Rome": "Italy",
}


# ── Two-Stage RAG (Qdrant Retrieval + FlashRank Reranker) ──
def search_destination_knowledge(destination: str, query: str = "") -> str:
    """
    Two-stage retrieval pipeline:
    1. Dense vector search in Qdrant with destination metadata filter (Top 10 candidates - Recall@10)
    2. Cross-encoder reranking via FlashRank (Top 3 results - Precision@3)
    """
    normalized = destination.strip().title()
    variants = [normalized, "General"]
    if normalized in CITY_TO_COUNTRY:
        variants.append(CITY_TO_COUNTRY[normalized])

    search_query = query if query else f"Travel guidelines, visa, transport, and tips for {normalized}"
    print(f"[RAG] Query: {search_query}")
    print(f"[RAG] Filter variants: {variants}")

    try:
        client = get_qdrant_client()
        embed_model = get_embed_model()
        reranker = get_reranker()

        # Generate query embedding
        query_vectors = list(embed_model.embed([search_query]))
        if not query_vectors:
            return ""
        query_vector = query_vectors[0].tolist()

        # Stage 1: Dense Retrieval with Qdrant payload filtering across variants
        dest_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="destination",
                    match=models.MatchValue(value=v)
                ) for v in variants
            ]
        )

        search_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=dest_filter,
            limit=10,
            with_payload=True
        )
        search_results = list(search_response.points if hasattr(search_response, "points") else search_response)

        # Fallback if fewer than 5 results
        if len(search_results) < 5:
            fallback_response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=10,
                with_payload=True
            )
            fallback_points = list(fallback_response.points if hasattr(fallback_response, "points") else fallback_response)
            existing_ids = {h.id for h in search_results}
            for p in fallback_points:
                if p.id not in existing_ids:
                    search_results.append(p)
                    existing_ids.add(p.id)
                if len(search_results) >= 10:
                    break

        print(f"[RAG] Retrieved {len(search_results)} candidates")
        for h in search_results[:3]:
            print(f"[RAG]   dest={h.payload.get('destination')} "
                  f"cat={h.payload.get('category')} "
                  f"len={len(h.payload.get('text',''))}")

        if not search_results:
            print("[RAG] ⚠️ Zero results — collection may be empty or filter too strict")
            return ""

        # Prepare passages for Stage 2 reranking
        passages = [
            {
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "meta": {
                    "destination": hit.payload.get("destination", ""),
                    "category": hit.payload.get("category", ""),
                    "source_file": hit.payload.get("source_file", "")
                }
            }
            for hit in search_results
            if hit.payload and hit.payload.get("text")
        ]

        if not passages:
            return ""

        # Stage 2: Cross-Encoder Reranking
        rerank_request = RerankRequest(query=search_query, passages=passages)
        reranked_results = reranker.rerank(rerank_request)

        if reranked_results:
            print(f"[RAG] Reranked top score: {reranked_results[0].get('score', 0):.4f}")

        # Take Top 3 high-precision chunks
        top_chunks = reranked_results[:3]

        formatted_knowledge = []
        for i, chunk in enumerate(top_chunks, 1):
            category = chunk.get("meta", {}).get("category", "General")
            source = chunk.get("meta", {}).get("source_file", "knowledge-base")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()
            formatted_knowledge.append(
                f"**[{category}] (Source: {source}, Relevance: {score:.2f})**\n{text}"
            )

        return "\n\n".join(formatted_knowledge)

    except Exception as e:
        print(f"[Warning] Knowledge search error: {e}")
        return ""


# ── Tavily Web Search ──
def search_web(query: str, max_results: int = 5) -> str:
    """Search web via Tavily API."""
    if not TAVILY_API_KEY:
        return "Tavily API Key is not configured."
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results
            },
            timeout=15
        )
        results = response.json().get("results", [])
        if not results:
            return "No results found."
        lines = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "").strip()
            snippet = content[:250] + ("..." if len(content) > 250 else "")
            lines.append(f"**{i}. [{title}]({url})**")
            if snippet:
                lines.append(snippet)
            lines.append(f"🔗 {url}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Search unavailable: {str(e)}"


# ── OpenWeatherMap Direct API ──
def get_current_weather(city: str) -> str:
    """Get live weather from OpenWeatherMap."""
    if not OPENWEATHER_API_KEY:
        return "OpenWeather API Key is not configured."
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10
        )
        data = response.json()
        if response.status_code != 200:
            return f"Weather unavailable: {data.get('message', 'Unknown error')}"
        return (
            f"📍 City: {data['name']}\n"
            f"🌡️ Temperature: {data['main']['temp']}°C "
            f"(Feels like {data['main']['feels_like']}°C)\n"
            f"💧 Humidity: {data['main']['humidity']}%\n"
            f"🌤️ Condition: {data['weather'][0]['description'].title()}\n"
            f"💨 Wind Speed: {data['wind']['speed']} m/s"
        )
    except Exception as e:
        return f"Weather unavailable: {str(e)}"


def get_forecast(city: str) -> str:
    """Get 5-period forecast from OpenWeatherMap."""
    if not OPENWEATHER_API_KEY:
        return "OpenWeather API Key is not configured."
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10
        )
        data = response.json()
        if response.status_code != 200:
            return f"Forecast unavailable: {data.get('message', 'Unknown error')}"
        lines = [f"📍 5-Period Forecast for {city}:"]
        for item in data["list"][:5]:
            lines.append(
                f"  • {item['dt_txt']} — "
                f"{item['main']['temp']}°C, "
                f"{item['weather'][0]['description'].title()}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Forecast unavailable: {str(e)}"


# ── Utility ──
def extract_destination(query: str) -> str:
    """Extracts the primary destination city or country from user query."""
    prompt = (
        f"Extract only the destination city or country name from this travel query. "
        f"Return ONLY the city or country name, nothing else.\n\nQuery: {query}"
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        return raw.strip().strip('"\'').title()
    except Exception:
        try:
            fallback = ChatGroq(model="openai/gpt-oss-20b", max_retries=0, timeout=30.0)
            raw = fallback.invoke(prompt).content.strip()
            return raw.strip().strip('"\'').title()
        except Exception:
            # Fallback heuristic
            words = query.split()
            return words[-1].title() if words else "General"
