# ✈️ AI Travel Planner System

An enterprise-grade, multi-agent AI Travel Planner System built on **LangGraph**, **FastMCP (Model Context Protocol)**, **Qdrant Vector DB (Two-Stage RAG)**, **PostgreSQL Checkpointing**, and **Groq Fast LLM Inference**.

Delivers complete, grounded, real-time travel plans with live web search, weather, flights, hotels, and INR budget breakdowns from a single natural language query.

---

## 🤖 Agent Details & Architecture

The system utilizes a **LangGraph Supervisor Multi-Agent Architecture** where specialized agents collaborate dynamically:

| Agent | Core Role | Tools / Integration | Primary Output |
|---|---|---|---|
| 👑 **Supervisor Agent** | Master Orchestrator | Groq LLM Routing | Analyzes query, extracts destination, detects trip type (`INTERNATIONAL`, `DOMESTIC_SHORT`, `DOMESTIC_LONG`), and dispatches parallel specialist execution. |
| 📚 **Research Agent** | Destination & Culture Expert | Two-Stage RAG (Embedded Qdrant + FlashRank Cross-Encoder) | Visa/entry rules for Indian citizens, currency & payments, safety guidelines, and local cultural etiquette. |
| 🛫 **Flight & Transit Agent** | Transit Specialist | FastMCP (`tavily_search`) | Live flight routes, IRCTC train options, intercity buses/cabs, and estimated fares in INR (₹). |
| 🏨 **Hotel Agent** | Accommodation Advisor | FastMCP (`tavily_search`) | Verified hotel options across budget, mid-range, and luxury tiers, plus neighborhood vibe and stay recommendations. |
| 🌤️ **Weather Agent** | Climate & Packing Advisor | FastMCP (`get_current_weather`, `get_forecast`) | Real-time weather, 5-period forecast, seasonal advisories, and tailored packing checklists. |
| 💰 **Budget Agent** | Financial Calculator | Groq LLM Synthesis | Consolidated INR (₹) cost breakdown (transit, stays, daily food, activities, buffer) for Backpacker, Mid-range, and Luxury tiers. |
| 🗓️ **Itinerary Agent** | Master Synthesizer | Groq LLM Synthesis | Cohesive day-by-day master travel plan with morning, afternoon, and evening schedules, food spots, and practical checklists. |

---

## 🔑 API Keys & Prerequisites

To run the system, you will need free API keys from the following services:

| Service | Link | Purpose | Environment Variable |
|---|---|---|---|
| **Groq** | [https://console.groq.com](https://console.groq.com) | Ultra-fast LLM inference (`openai/gpt-oss-120b` & `openai/gpt-oss-20b`) | `GROQ_API_KEY` |
| **Tavily** | [https://www.tavily.com](https://www.tavily.com) | Live web search for flight fares, train routes, and hotels | `TAVILY_API_KEY` |
| **OpenWeatherMap** | [https://openweathermap.org/api](https://openweathermap.org/api) | Real-time weather conditions and forecast data | `OPENWEATHER_API_KEY` |

### How to Get Your API Keys:
1. **Groq**:
   - Sign up / Log in at [console.groq.com](https://console.groq.com).
   - Navigate to **API Keys** in the left sidebar.
   - Click **Create API Key**, copy your key, and assign it to `GROQ_API_KEY`.
2. **Tavily**:
   - Register for a free account at [tavily.com](https://www.tavily.com).
   - Go to your dashboard to copy your API key (1,000 free searches/month).
   - Assign it to `TAVILY_API_KEY`.
3. **OpenWeatherMap**:
   - Create a free account at [openweathermap.org/api](https://openweathermap.org/api).
   - Go to your account profile → **My API Keys**.
   - Copy your default key (or generate a new one) and assign it to `OPENWEATHER_API_KEY`. *(Note: New keys activate within 10-15 minutes).*

---

## 🛠️ Complete Setup Guide

> **Python Requirement**: **Python 3.10+** (Python 3.10, 3.11, or 3.12 recommended).

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone https://github.com/mandar7-star/AI-Travel-Planner-System.git
cd AI-Travel-Planner-System

# Create virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup & Environment Variables
1. **Create Database in PostgreSQL**:
   - Open your PostgreSQL app (e.g., pgAdmin, Postgres.app, DBeaver, or psql).
   - Create a new database (e.g., `travel_planner`):
     ```sql
     CREATE DATABASE travel_planner;
     ```
2. **Configure `.env`**:
   - Create a `.env` file in the project root (see `.env.example`).
   - Copy your database connection URL and paste it as `DATABASE_URL`:
     ```env
     GROQ_API_KEY=your_groq_api_key
     TAVILY_API_KEY=your_tavily_api_key
     OPENWEATHER_API_KEY=your_openweathermap_api_key
     DATABASE_URL=postgresql://postgres:your_password@localhost:5432/travel_planner
     ```
     *(Format: `postgresql://<username>:<password>@<host>:<port>/<database_name>`)*
   - *(Note: If PostgreSQL is offline or unconfigured, the system automatically falls back to in-memory `MemorySaver`).*

### 4. Knowledge Ingestion
Ingest destination knowledge vectors into embedded Qdrant:
```bash
python ingest_knowledge.py
```

### 5. Launch the Web Application
```bash
streamlit run frontend.py
```

---

## 🚀 Key Highlights

- 🤖 **LangGraph Supervisor Graph** — Dynamic supervisor with parallel fan-out (`research`, `flight`, `hotel`, `weather`) and fan-in (`budget`, `itinerary`).
- 🔌 **FastMCP (Model Context Protocol)** — Official MCP server running over `stdio` for standardized, decoupled tool discovery and execution.
- 🎯 **Two-Stage RAG Pipeline** — Qdrant vector retrieval (Bi-Encoder Recall@10) + FlashRank Cross-Encoder reranking (Precision@3).
- 🐘 **PostgreSQL Checkpointing** — Full state persistence, session continuity, and time-travel debugging via native PostgreSQL on port 5432 with `psycopg_pool` (with automatic graceful fallback to `MemorySaver`).
- ⚡ **High-Speed Groq Inference** — Fail-fast `openai/gpt-oss-120b` with instant automated fallback to `openai/gpt-oss-20b` on rate limits.
- 🌤️ **Live Weather & Forecast** — OpenWeatherMap API integration with clothing, packing, and activity advisories.
- 💰 **INR Budget Breakdown** — Comprehensive trip cost estimates across Budget, Mid-Range, and Luxury tiers.
- 🖥️ **Interactive Web UI** — Modern Streamlit UI with live agent execution timeline, MCP status chips, and session management.

---

## 🧠 System Architecture

```
                      ┌──────────────────┐
                      │    User Query    │
                      └─────────┬────────┘
                                │
                      ┌─────────▼────────┐
                      │    Supervisor    │
                      └─────────┬────────┘
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
 ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
 │  Research   │         │Flight/Travel│         │    Hotel    │
 │    Agent    │         │    Agent    │         │    Agent    │
 └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
        │ (Embedded Qdrant)     │ (FastMCP)             │ (FastMCP)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                         ┌──────▼──────┐
                         │   Weather   │
                         │    Agent    │
                         └──────┬──────┘
                                │ (FastMCP)
                         ┌──────▼──────┐
                         │Budget Agent │
                         └──────┬──────┘
                                │ (INR Synthesis)
                         ┌──────▼──────┐
                         │  Itinerary  │
                         │    Agent    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │PostgreSQL DB│ (Native Checkpointing on 5432)
                         └─────────────┘
```

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph & LangChain Core | Supervisor state machine with parallel execution & state reducers |
| **Tool Protocol** | FastMCP (`mcp_server.py`) | Model Context Protocol over `stdio` with JSON-RPC 2.0 |
| **Vector DB** | Qdrant (Embedded `./qdrant_data`) | Embedded vector store with payload filtering on destination |
| **Embeddings** | FastEmbed (`bge-small-en-v1.5`) | Local, lightweight 384-dimensional dense embeddings |
| **Reranker** | FlashRank (`ms-marco-TinyBERT-L-2-v2`) | Cross-encoder reranking for maximum Precision@3 |
| **LLM Inference** | Groq (`openai/gpt-oss-120b` / `20b`) | Ultra-fast token generation with fail-fast fallback |
| **State Persistence** | PostgreSQL + `psycopg_pool` | Production checkpointing on native port 5432 (with MemorySaver fallback) |
| **Search & Weather** | Tavily Search + OpenWeatherMap | Real-time live web search and forecast data |
| **Frontend** | Streamlit | Real-time streaming UI with MCP badges and expandable agent tabs |

---

## 📸 Screenshots

The following screenshots demonstrating the system UI, agent execution pipeline, and exported itinerary can be found in the [`screenshots/`](screenshots/) folder:

| File Name | Description | Preview |
|---|---|---|
| **`1_main_ui.png`** | **Interactive Streamlit Web UI** — Query input, destination configuration, and active search status | ![Main UI](screenshots/1_main_ui.png) |
| **`2_agents_pipeline.png`** | **Agent Execution Pipeline** — Supervisor graph orchestration, parallel fan-out agents, and real-time execution timeline | ![Agents Pipeline](screenshots/2_agents_pipeline.png) |
| **`3_download.png`** | **Trip Plan & Export** — Synthesized day-wise itinerary, weather forecast, INR budget breakdown, and PDF/Markdown download options | ![Download & Export](screenshots/3_download.png) |

---

## 💡 Example Queries

- `Plan a 7-day Japan trip for cherry blossom season under ₹2.5 Lakhs`
- `5-day luxury holiday in Dubai from Mumbai with 5-star hotel options`
- `Weekend road trip from Pune to Nashik for 2 people with wine tasting`
- `Budget 4-day trip to Bali for digital nomads with coworking cafe recommendations`

---

## 👨‍💻 Author

**Mandar Borhade**
- LinkedIn: [mandarborhade](https://www.linkedin.com/in/mandarborhade)
- GitHub: [mandar7-star](https://github.com/mandar7-star)