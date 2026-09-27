"""
LangGraph Multi-Agent Travel Planner System
Architecture:
- Self-hosted PostgreSQL checkpointing via psycopg_pool ConnectionPool
- Supervisor Multi-Agent Orchestration with parallel specialist fan-out and fan-in
- Two-stage RAG (Qdrant + FlashRank Reranker)
- Dynamic task routing and fault isolation
"""

import os
import sys
import operator
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
import psycopg_pool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from tools import (
    search_destination_knowledge,
    extract_destination
)
from mcp_client import mcp_client

load_dotenv(override=True)

import time
import re

try:
    from groq import RateLimitError
except ImportError:
    RateLimitError = None

# ── Database & LLM Setup ──
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/travel_planner")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Configure LLM instances with max_retries=0 to fail-fast on 429 and trigger fallback immediately
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    max_retries=0,
    timeout=30.0
)
fallback_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_retries=0,
    timeout=30.0
)


def invoke_llm(messages, max_retries: int = 3):
    """
    Invokes the LLM with multi-tier resilience:
    1. Tries primary model (openai/gpt-oss-120b).
    2. On 429/413 rate limit, switches immediately to fallback model (openai/gpt-oss-20b).
    3. If the fallback also encounters a temporary TPM rate limit (e.g. 8k org limit),
       extracts the exact wait time from the error message (e.g. 1.7s - 6.0s),
       sleeps gracefully, and retries rather than crashing or displaying error JSON to the user.
    """
    for attempt in range(max_retries):
        # 1. Try Primary Model
        try:
            return llm.invoke(messages)
        except Exception as e1:
            err_str1 = str(e1).lower()
            is_rl1 = (
                (RateLimitError is not None and isinstance(e1, RateLimitError))
                or "429" in err_str1
                or "413" in err_str1
                or "rate" in err_str1
                or "tpm" in err_str1
                or "tokens per minute" in err_str1
            )
            if not is_rl1:
                raise e1
            
            print(f"[Fallback] Primary model ({GROQ_MODEL}) rate-limited (429/413) → switching immediately to openai/gpt-oss-20b")
            
            # 2. Try Fallback Model
            try:
                return fallback_llm.invoke(messages)
            except Exception as e2:
                err_str2 = str(e2).lower()
                is_rl2 = (
                    (RateLimitError is not None and isinstance(e2, RateLimitError))
                    or "429" in err_str2
                    or "413" in err_str2
                    or "rate" in err_str2
                    or "tpm" in err_str2
                    or "tokens per minute" in err_str2
                )
                if not is_rl2:
                    raise e2
                
                # Extract wait time from error message (e.g. "Please try again in 1.7325s.")
                match = re.search(r"try again in ([\d\.]+)s", str(e2))
                wait_sec = float(match.group(1)) + 0.5 if match else (2.0 * (attempt + 1))
                wait_sec = min(wait_sec, 8.0)
                
                print(f"[Fallback Throttle] Both models hit TPM limit. Waiting {wait_sec:.1f}s for quota window to reset (retry {attempt + 1}/{max_retries})...")
                time.sleep(wait_sec)
                
    # Final retry attempt on fallback
    return fallback_llm.invoke(messages)









# ── Graph State ──
class TravelState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    user_query: str
    destination: str
    trip_type: str
    plan_steps: List[str]
    active_agent: Annotated[str, lambda prev, new: new]
    research_results: str
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str
    llm_calls: Annotated[int, operator.add]



# ── Supervisor Node ──
def supervisor_node(state: TravelState) -> Dict[str, Any]:
    """
    Analyzes user query and determines required specialist agents.
    Sets planned execution branches and extracts destination metadata.
    """
    print("\n[Supervisor] Analyzing query and orchestrating agents...")
    query = state.get("user_query", "")
    destination = state.get("destination") or extract_destination(query)

    supervisor_prompt = f"""
    You are the Master Orchestrator for an AI Travel Planner System.
    Analyze the user's travel query and decide which specialist agents need to be invoked.
    
    User Query: "{query}"
    Detected Destination: "{destination}"
    
    Available Specialist Agents:
    1. research_agent (Visa, local culture, safety guidelines, transport tips)
    2. flight_agent (Flights, trains, road transport)
    3. hotel_agent (Hotels, resorts, stay areas, price brackets)
    4. weather_agent (Live weather, seasonal forecast, packing recommendations)
    5. budget_agent (Cost calculations and estimates in INR)
    6. itinerary_agent (Day-by-day complete schedule synthesis)
    
    Return a JSON object with:
    - "trip_type": "INTERNATIONAL" | "DOMESTIC_SHORT" | "DOMESTIC_LONG"
    - "required_specialists": list of agent names from above
    - "reasoning": brief one-sentence explanation
    
    Respond ONLY with valid JSON.
    """

    try:
        response = invoke_llm([
            SystemMessage(content="You are a precise routing supervisor. Output pure JSON only."),
            HumanMessage(content=supervisor_prompt)
        ])
        content = response.content.strip()
        # Clean JSON markdown if wrapped in ```json
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content.strip())
        trip_type = data.get("trip_type", "INTERNATIONAL")
        required_specialists = data.get("required_specialists", [
            "research_agent", "flight_agent", "hotel_agent", "weather_agent"
        ])
    except Exception as e:
        print(f"[Supervisor Error / Fallback]: {e}")
        trip_type = "INTERNATIONAL"
        required_specialists = ["research_agent", "flight_agent", "hotel_agent", "weather_agent"]

    return {
        "destination": destination,
        "trip_type": trip_type,
        "plan_steps": required_specialists,
        "active_agent": "supervisor",
        "messages": [AIMessage(content=f"Supervisor planned execution for destination: {destination}")],
        "llm_calls": 1
    }


# ── Specialist Agent 1: Research Agent (RAG + Qdrant + FlashRank) ──
def research_agent(state: TravelState) -> Dict[str, Any]:
    """Retrieves destination knowledge and generates comprehensive research brief."""
    print("\n[Agent: Research] Fetching RAG destination knowledge...")
    destination = state.get("destination") or "General"
    query = state.get("user_query", "")

    try:
        rag_data = search_destination_knowledge(destination=destination, query=query)
        rag_context = f"Verified Destination Knowledge (RAG):\n{rag_data}" if rag_data else "Note: No specific local database doc found, rely on curated knowledge."
        prompt = f"""
        You are a travel research expert.
        User Query: {query}
        Destination: {destination}
        
        {rag_context}

        Provide a structured research summary:
        1. Visa & Entry requirements (specifically for Indian citizens)
        2. Best seasons and timing
        3. Local currency, payment methods (cash vs card/UPI), money tips
        4. Cultural etiquette and local rules
        5. Key safety recommendations and scams to avoid
        """

        response = invoke_llm([
            SystemMessage(content="You are an expert travel researcher. Be factual, concise, and structured. Format output using headings, bullet lists, and short paragraphs. Do NOT use markdown tables.\n\nDo NOT include any URLs, links, or a 'Sources' section. The knowledge you receive comes from a curated local reference that does not expose URLs. Do not fabricate links, do not invent citations, and do not add a Sources section under any circumstance. Just produce the research content."),
            HumanMessage(content=prompt)
        ])
        result_text = response.content
    except Exception as e:
        result_text = f"Research briefing unavailable due to error: {str(e)}"

    return {
        "research_results": result_text,
        "active_agent": "research_agent",
        "messages": [AIMessage(content="Research and visa briefing completed.")],
        "llm_calls": 1
    }


# ── Specialist Agent 2: Flight & Transport Agent (MCP / Web Search) ──
def flight_agent(state: TravelState) -> Dict[str, Any]:
    """Researches flights, trains, or ground transport based on trip type via FastMCP."""
    print("\n[Agent: Flight/Transport] Invoking FastMCP tavily_search tool...")
    query = state.get("user_query", "")
    destination = state.get("destination", "")
    trip_type = state.get("trip_type", "INTERNATIONAL")

    try:
        no_tables_suffix = " Format output using headings, bullet lists, and short paragraphs. Do NOT use markdown tables under any circumstance. If the search results or retrieved context contain URLs, include a 'Sources' section at the end listing them as [Title](URL) bullets. If no URLs are available, write 'Sources: None retrieved.'"
        if trip_type == "DOMESTIC_SHORT":
            search_data = mcp_client.invoke_tool_sync("tavily_search", query=f"train bus taxi options {query} India")
            system_msg = "You are a local India transport specialist. Suggest ground transport (trains, buses, cabs). Never recommend flights for short distances under 500km. All fares in INR." + no_tables_suffix
            prompt_body = f"User Query: {query}\n\nSearch Results:\n{search_data}\n\nProvide:\n1. Train options (IRCTC)\n2. Bus options (RedBus)\n3. Cab options (Ola/Uber/Intercity)\n4. Recommended choice."
        elif trip_type == "DOMESTIC_LONG":
            search_data = mcp_client.invoke_tool_sync("tavily_search", query=f"trains and flights {query} prices 2025")
            system_msg = "You are a domestic travel specialist. Compare trains and flights with approximate prices in INR." + no_tables_suffix
            prompt_body = f"User Query: {query}\n\nSearch Results:\n{search_data}\n\nProvide flight and train comparisons with booking platforms."
        else:
            search_data = mcp_client.invoke_tool_sync("tavily_search", query=f"flights from India to {destination} airlines price duration 2025")
            system_msg = "You are an international flight advisor. Provide flight routes, airlines, typical economy/business prices in INR, and booking advice." + no_tables_suffix
            prompt_body = f"Destination: {destination}\nUser Query: {query}\n\nWeb Search:\n{search_data}\n\nProvide best flight options."

        response = invoke_llm([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt_body)
        ])
        result_text = response.content
    except Exception as e:
        result_text = f"Transit options search encountered an issue: {str(e)}"

    return {
        "flight_results": result_text,
        "active_agent": "flight_agent",
        "messages": [AIMessage(content="Flight & transportation options compiled via FastMCP.")],
        "llm_calls": 1
    }


# ── Specialist Agent 3: Hotel Agent (MCP / Web Search) ──
def hotel_agent(state: TravelState) -> Dict[str, Any]:
    """Researches accommodation across budget tiers via FastMCP."""
    print("\n[Agent: Hotel] Invoking FastMCP tavily_search tool...")
    query = state.get("user_query", "")
    destination = state.get("destination", "")

    try:
        search_data = mcp_client.invoke_tool_sync("tavily_search", query=f"best hotels and areas to stay in {destination} prices ratings 2025")
        prompt = f"""
        User Query: {query}
        Destination: {destination}
        
        Search Results (from FastMCP Tavily Tool):
        {search_data}
        
        Instructions:
        - You MUST only use specific hotel names that appear verbatim in the Search Results above.
        - You MUST NOT invent, guess, or fabricate hotel names under any circumstances.
        - Placeholder names such as "Hotel 1", "Hotel 2", "Hotel A", "Hotel B", "Hotel X", "Luxury Hotel", "Budget Stay", or similar generic labels are strictly forbidden.
        - If the Search Results contain specific hotel names:
          1. Recommend top budget, mid-range, and luxury options found in the results (with estimated prices per night in INR ₹ if mentioned).
          2. Recommend best neighborhoods/areas to stay.
          3. Provide practical booking tips.
        - If the Search Results DO NOT contain specific hotel names:
          1. Explicitly state: "Specific hotel names were not retrieved in the current live search results."
          2. Recommend key neighborhoods and districts in {destination} to stay in instead.
          3. For each neighborhood/area, detail its character, vibe, and what type of traveler it suits best (e.g. couples, families, backpackers, luxury travelers).
          4. Provide general booking recommendations and typical price ranges in INR ₹ for those areas.
        """
        response = invoke_llm([
            SystemMessage(content=(
                "You are an expert accommodation advisor. Provide clean price comparisons and neighborhood insights in INR ₹. "
                "CRITICAL: Only reference specific hotel names that appear verbatim in the provided search results. "
                "Do NOT invent, fabricate, or guess hotel names. Generic placeholder names like 'Hotel 1', 'Hotel A', 'Hotel X', 'Luxury Hotel', or 'Budget Stay' are strictly prohibited. "
                "If no specific hotel names are present in the search data, honestly declare that specific hotels were not found in the search results and provide detailed neighborhood/area recommendations instead (area character, traveler suitability, estimated price ranges). "
                "Format output using headings, bullet lists, and short paragraphs. Do NOT use markdown tables under any circumstance. "
                "If the search results or retrieved context contain URLs, include a 'Sources' section at the end listing them as [Title](URL) bullets. If no URLs are available, write 'Sources: None retrieved.'"
            )),
            HumanMessage(content=prompt)
        ])
        result_text = response.content
    except Exception as e:
        result_text = f"Hotel recommendations unavailable: {str(e)}"

    return {
        "hotel_results": result_text,
        "active_agent": "hotel_agent",
        "messages": [AIMessage(content="Hotel and accommodation options collected via FastMCP.")],
        "llm_calls": 1
    }


# ── Specialist Agent 4: Weather Agent (MCP / OpenWeatherMap) ──
def weather_agent(state: TravelState) -> Dict[str, Any]:
    """Fetches real-time weather and forecast data via FastMCP."""
    print("\n[Agent: Weather] Invoking FastMCP OpenWeather tools...")
    destination = state.get("destination", "")

    try:
        current_w = mcp_client.invoke_tool_sync("get_current_weather", city=destination)
        forecast_w = mcp_client.invoke_tool_sync("get_forecast", city=destination)

        response = invoke_llm([
            SystemMessage(content="You are a travel weather advisor. Format output using headings, bullet lists, and short paragraphs. Do NOT use markdown tables under any circumstance."),
            HumanMessage(content=f"""
            Destination: {destination}
            Current Weather (FastMCP):
            {current_w}
            
            Forecast (FastMCP):
            {forecast_w}
            
            Provide:
            1. Weather summary and seasonal suitability
            2. Recommended clothing & packing checklist
            3. Weather precautions (rain, heat, UV index)
            """)
        ])
        result_text = f"**Current Conditions (via FastMCP):**\n{current_w}\n\n**Forecast (via FastMCP):**\n{forecast_w}\n\n**Weather Advisory:**\n{response.content}"
    except Exception as e:
        result_text = f"Weather data unavailable: {str(e)}"

    return {
        "weather_results": result_text,
        "active_agent": "weather_agent",
        "messages": [AIMessage(content="Weather data and packing advisory processed via FastMCP.")],
        "llm_calls": 1
    }


# ── Specialist Agent 5: Budget Agent (Financial Calculator) ──
def budget_agent(state: TravelState) -> Dict[str, Any]:
    """Synthesizes transit and hotel findings into a comprehensive INR budget."""
    print("\n[Agent: Budget] Calculating financial breakdown...")
    query = state.get("user_query", "")
    flight_res = state.get("flight_results", "")[:1200]
    hotel_res = state.get("hotel_results", "")[:1200]

    try:
        prompt = f"""
        User Query: {query}
        Flight / Transport Highlights: {flight_res}
        Hotel / Accommodation Highlights: {hotel_res}
        
        Calculate a comprehensive budget breakdown in Indian Rupees (INR ₹):
        1. Estimated Transit/Flight Costs
        2. Accommodation Costs (per night & total duration)
        3. Daily Food & Dining Estimate (Budget / Mid-range / Fine Dining)
        4. Local Sightseeing, Entry Tickets & Activities
        5. Miscellaneous & Buffer (Sim card, local transport, insurance)
        6. Total Trip Estimated Cost across (A) Budget Backpacker, (B) Moderate Traveler, (C) Luxury Tier.
        7. Top money-saving tips.
        """
        response = invoke_llm([
            SystemMessage(content="You are a financial travel planner. Always display exact figures in INR ₹. Format output using headings, bullet lists, and short paragraphs. Do NOT use markdown tables under any circumstance."),
            HumanMessage(content=prompt)
        ])
        result_text = response.content
    except Exception as e:
        result_text = f"Budget computation error: {str(e)}"

    return {
        "budget_results": result_text,
        "active_agent": "budget_agent",
        "messages": [AIMessage(content="Budget breakdown calculated.")],
        "llm_calls": 1
    }


# ── Specialist Agent 6: Itinerary Agent (Final Synthesis) ──
def itinerary_agent(state: TravelState) -> Dict[str, Any]:
    """Synthesizes all specialist research into a cohesive day-by-day travel plan."""
    print("\n[Agent: Itinerary] Synthesizing final master travel itinerary...")
    query = state.get("user_query", "")
    research = state.get("research_results", "")[:1200]
    flights = state.get("flight_results", "")[:1000]
    hotels = state.get("hotel_results", "")[:1000]
    weather = state.get("weather_results", "")[:700]
    budget = state.get("budget_results", "")[:1000]

    try:
        prompt = f"""
        User Query: {query}
        Destination Research: {research}
        Transit Options: {flights}
        Hotels & Neighborhoods: {hotels}
        Weather Advisory: {weather}
        Budget Breakdown: {budget}
        
        Synthesize this into a polished, professional Day-by-Day Master Travel Plan:
        1. Executive Trip Summary (Best time, travel style, overall vibe)
        2. Day-by-Day Detailed Schedule with morning, afternoon, and evening activities
        3. Selected Hotel & Transport Recommendation
        4. Must-Try Local Cuisine & Dining spots
        5. Practical Checklist (Visa, currency, emergency numbers, essential apps)
        6. Final Cost Summary in INR (₹)
        """
        response = invoke_llm([
            SystemMessage(content="You are a master itinerary synthesizer. Provide a beautifully formatted markdown travel itinerary. Format output using headings, bullet lists, and short paragraphs. Do NOT use markdown tables under any circumstance."),
            HumanMessage(content=prompt)
        ])
        result_text = response.content
    except Exception as e:
        result_text = f"Itinerary generation failed: {str(e)}"


    return {
        "itinerary": result_text,
        "active_agent": "itinerary_agent",
        "messages": [AIMessage(content="Master travel itinerary generated.")],
        "llm_calls": 1
    }


# ── LangGraph Workflow Construction ──
workflow = StateGraph(TravelState)

# Add all nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_agent)
workflow.add_node("flight_agent", flight_agent)
workflow.add_node("hotel_agent", hotel_agent)
workflow.add_node("weather_agent", weather_agent)
workflow.add_node("budget_agent", budget_agent)
workflow.add_node("itinerary_agent", itinerary_agent)

# Start routes to supervisor
workflow.add_edge(START, "supervisor")

# Parallel Fan-Out from Supervisor to independent specialists
workflow.add_edge("supervisor", "research_agent")
workflow.add_edge("supervisor", "flight_agent")
workflow.add_edge("supervisor", "hotel_agent")
workflow.add_edge("supervisor", "weather_agent")

# Fan-In from parallel branches to Budget Agent
workflow.add_edge("research_agent", "budget_agent")
workflow.add_edge("flight_agent", "budget_agent")
workflow.add_edge("hotel_agent", "budget_agent")
workflow.add_edge("weather_agent", "budget_agent")

# Sequence from Budget to Itinerary Synthesis to END
workflow.add_edge("budget_agent", "itinerary_agent")
workflow.add_edge("itinerary_agent", END)


# ── PostgreSQL Connection Pool & Checkpointer ──
def get_compiled_graph():
    """
    Initializes PostgreSQL connection pool and compiles LangGraph StateGraph.
    Falls back gracefully to in-memory MemorySaver if PostgreSQL is offline or credentials fail.
    """
    print("=" * 60)
    print(" AI Travel Planner System — Starting")
    print("=" * 60)

    try:
        import psycopg
        # Fast connection test with 2s timeout
        with psycopg.connect(DATABASE_URL, connect_timeout=2) as conn:
            pass
        
        pool = psycopg_pool.ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=20,
            kwargs={"autocommit": True}
        )
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        print(" ✅ PostgreSQL: localhost:5432 / travel_planner")
        print(" ✅ Qdrant: embedded at ./qdrant_data")
        print("=" * 60)
        return workflow.compile(checkpointer=checkpointer)
    except Exception as e:
        print(" ❌ PostgreSQL unreachable — using in-memory MemorySaver")
        print("    Check DATABASE_URL in .env and that Postgres is running")
        print(" ✅ Qdrant: embedded at ./qdrant_data")
        print("=" * 60)
        from langgraph.checkpoint.memory import MemorySaver
        return workflow.compile(checkpointer=MemorySaver())


# Compiled Application Instance
app = get_compiled_graph()