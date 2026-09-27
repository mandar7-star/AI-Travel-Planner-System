"""
Streamlit Interactive UI for Production Multi-Agent AI Travel Planner System
Visualizes:
- Supervisor Master Orchestrator
- Parallel Fan-Out Specialists (Research, Flights, Hotels, Weather)
- Synthesis Pipeline (Budget & Itinerary)
- Real-time Checkpointing & Session Memory via PostgreSQL
"""

import os
import re
import io
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage
from main import app


@st.cache_data(show_spinner=False)
def generate_pdf_bytes(markdown_text: str) -> bytes:
    """Converts a Markdown travel plan into PDF bytes using markdown-pdf (cached)."""
    from markdown_pdf import MarkdownPdf, Section
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(markdown_text))
    buf = io.BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return buf.getvalue()


def clean_llm_output(text: str) -> str:
    """Convert LLM-generated <br> to newlines and strip dangerous HTML injection tags."""
    if not text:
        return text
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    dangerous = r"</?(?:script|iframe|object|embed|style|link|meta|form|input)[^>]*>"
    text = re.sub(dangerous, "", text, flags=re.IGNORECASE)
    return text


def tidy_weather(text: str) -> str:
    """Normalize weather text: preserve double-newline section breaks, 
    strip trailing spaces, remove excessive blank lines, and inject markdown hard breaks."""
    if not text:
        return text
    # Collapse 3+ newlines to exactly 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Convert single newlines to markdown hard breaks (two trailing spaces)
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        # If next line is not blank and this line is not blank, add hard break
        if i < len(lines) - 1 and stripped and lines[i + 1].strip():
            out.append(stripped + "  ")
        else:
            out.append(stripped)
    return "\n".join(out).strip()



def strip_markdown_tables(text: str) -> str:
    """Convert markdown tables to bullet lists."""
    if not text:
        return text
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and line.endswith("|"):
            if i + 1 < len(lines):
                sep = lines[i + 1].strip()
                if set(sep.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")) == set():
                    headers = [c.strip() for c in line.strip("|").split("|")]
                    i += 2
                    while i < len(lines):
                        row_line = lines[i].strip()
                        if not (row_line.startswith("|") and row_line.endswith("|")):
                            break
                        cells = [c.strip() for c in row_line.strip("|").split("|")]
                        for h, v in zip(headers, cells):
                            if v:
                                out.append(f"**{h}:** {v}")
                        out.append("")
                        i += 1
                    continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)



st.set_page_config(
    page_title="AI Travel Planner System — Multi-Agent Supervisor",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; background: #FFFFFF; }

.hero-wrapper { 
    position: relative; 
    border-radius: 20px; 
    overflow: hidden; 
    margin-bottom: 2rem; 
    min-height: 250px; 
}
.hero-content { 
    position: relative; 
    z-index: 2; 
    min-height: 250px; 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    justify-content: center; 
    text-align: center; 
    padding: 2.2rem 2rem; 
}
.hero-badge { 
    background: rgba(37,99,235,0.25); 
    border: 1px solid rgba(255,255,255,0.5); 
    color: #ffffff !important; 
    font-size: 0.78rem; 
    font-weight: 700; 
    letter-spacing: 0.14em; 
    text-transform: uppercase; 
    padding: 0.35rem 1rem; 
    border-radius: 20px; 
    margin-bottom: 0.8rem; 
    display: inline-block; 
}
.hero-title { 
    font-size: 2.4rem; 
    font-weight: 800; 
    color: #ffffff; 
    margin: 0 0 0.5rem; 
    line-height: 1.2; 
    text-shadow: 0 2px 20px rgba(0,0,0,0.5); 
}
.hero-sub { 
    color: #f1f5f9; 
    font-size: 0.98rem; 
    max-width: 650px; 
    text-shadow: 0 2px 10px rgba(0,0,0,0.5); 
}

/* Primary Action Button (e.g. Generate My Travel Plan) */
div[data-testid="stButton"] > button { 
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important; 
    color: #ffffff !important; 
    border: none !important; 
    border-radius: 12px !important; 
    padding: 0.85rem 2.5rem !important; 
    font-size: 1.05rem !important; 
    font-weight: 700 !important; 
    width: 100% !important; 
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important; 
    transition: all 0.3s ease !important; 
}
div[data-testid="stButton"] > button:hover { 
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important; 
    transform: translateY(-2px) !important; 
}

/* Download Section Buttons */
div[data-testid="stDownloadButton"] {
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stDownloadButton"] > button { 
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important; 
    color: #ffffff !important; 
    border: none !important; 
    border-radius: 12px !important; 
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    box-sizing: border-box !important;
    padding: 0 1.2rem !important; 
    font-size: 0.95rem !important; 
    font-weight: 700 !important; 
    width: 100% !important; 
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important; 
    transition: all 0.3s ease !important; 
    margin: 0 !important;
}
div[data-testid="stDownloadButton"] > button:hover { 
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important; 
    transform: translateY(-2px) !important; 
}

div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    display: flex !important;
}

.save-bar {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0 1.2rem;
    color: #64748B;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    height: 48px;
    min-height: 48px;
    max-height: 48px;
    box-sizing: border-box;
    margin: 0;
    line-height: normal;
}
.save-bar code {
    background: #EDF2F7;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.82rem;
    color: #334155;
}

.sec-head { 
    display: flex; 
    align-items: center; 
    gap: 0.6rem; 
    margin: 1.8rem 0 0.75rem; 
    padding-bottom: 0.5rem; 
    border-bottom: 1px solid #E5E7EB; 
}
.sec-head span { 
    font-size: 1.2rem; 
    font-weight: 700; 
    color: #111827; 
}

.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
.metric-box { 
    flex: 1; 
    background: #F8FAFC; 
    border: 1px solid #E2E8F0; 
    border-radius: 12px; 
    padding: 1rem 1.2rem; 
    text-align: center; 
}
.metric-val { font-size: 1.8rem; font-weight: 800; color: #2563EB; }
.metric-lbl { 
    font-size: 0.78rem; 
    color: #64748B; 
    margin-top: 0.2rem; 
    text-transform: uppercase; 
    letter-spacing: 0.08em; 
    font-weight: 600;
}

.weather-card { 
    background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 100%); 
    border: 1px solid #BFDBFE; 
    border-radius: 14px; 
    padding: 1.4rem 1.6rem; 
    margin-bottom: 1rem; 
}
.weather-title { 
    font-size: 0.8rem; 
    font-weight: 700; 
    color: #2563EB; 
    letter-spacing: 0.1em; 
    text-transform: uppercase; 
    margin-bottom: 0.8rem; 
}
.weather-card p {
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.4;
}
.weather-card ul, .weather-card ol {
    margin: 0.3rem 0 0.5rem 1.2rem;
    padding-left: 0.8rem;
}
.weather-card li {
    margin-bottom: 0.15rem;
    line-height: 1.4;
}
.weather-card h1, .weather-card h2, .weather-card h3 {
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
    line-height: 1.3;
}
/* Kill Streamlit's default paragraph margin inside the card */
.weather-card > p,
.weather-card > div > p {
    margin-bottom: 0.35rem !important;
}
.weather-card .stMarkdown {
    margin-bottom: 0.3rem;
}
.final-card { 
    background: #FFFFFF; 
    border: 1px solid #E2E8F0; 
    border-left: 5px solid #2563EB; 
    border-radius: 14px; 
    padding: 1.8rem; 
    line-height: 1.8; 
    color: #1F2937; 
    font-size: 0.95rem; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}

.sb-tech-chip { 
    display: flex; 
    align-items: center; 
    gap: 0.55rem; 
    background: #FFFFFF; 
    border: 1px solid #E2E8F0; 
    border-radius: 8px; 
    padding: 0.45rem 0.75rem; 
    margin-bottom: 0.4rem; 
    font-size: 0.82rem; 
}
.agent-card { 
    background: #FFFFFF; 
    border: 1px solid #E2E8F0; 
    border-radius: 12px; 
    padding: 0.75rem 0.9rem; 
    margin-bottom: 0.55rem; 
    position: relative; 
    overflow: hidden; 
}
.agent-card-accent { 
    position: absolute; 
    left: 0; 
    top: 0; 
    bottom: 0; 
    width: 3.5px; 
    border-radius: 12px 0 0 12px; 
}
.agent-card-inner { 
    display: flex; 
    align-items: center; 
    gap: 0.65rem; 
    padding-left: 0.4rem; 
}
.agent-icon-box { 
    width: 36px; 
    height: 36px; 
    border-radius: 9px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    font-size: 1.1rem; 
    flex-shrink: 0; 
}
.agent-name { font-size: 0.88rem !important; font-weight: 700 !important; color: #0F172A !important; }
.agent-desc { font-size: 0.74rem !important; color: #1E293B !important; font-weight: 600 !important; margin-top: 0.15rem !important; }
.agent-badge { 
    font-size: 0.62rem; 
    font-weight: 700 !important; 
    padding: 0.22rem 0.55rem; 
    border-radius: 20px; 
    letter-spacing: 0.05em; 
    flex-shrink: 0; 
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 0.2rem;">
        <div style="font-size:1.25rem;font-weight:800;color:#0F172A;">✈️ AI Travel Planner</div>
        <div style="font-size:0.78rem;color:#334155;font-weight:600;margin-top:0.2rem;">Supervisor Orchestrator · MCP · Qdrant RAG</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#E2E8F0;margin:0.8rem 0;'>", unsafe_allow_html=True)

    thread_id = st.text_input(
        "👤 Session / Thread ID",
        value="mandar_traveller_7",
        help="Maps directly to LangGraph PostgreSQL checkpoint thread_id"
    )

    st.markdown("<hr style='border-color:#E2E8F0;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;color:#0F172A;margin-bottom:0.6rem;'>⚡ Architecture Stack</div>", unsafe_allow_html=True)

    TECHS = [
        ("🎯", "LangGraph Supervisor", "Dynamic routing & parallel fan-out"),
        ("🧠", "Groq Reasoning Engine", "High-throughput LLM reasoning"),
        ("⚡", "Qdrant + FlashRank", "Two-stage dense RAG + reranker"),
        ("🔌", "FastMCP Server (stdio)", "Critical-path JSON-RPC 2.0 tool execution"),
        ("🐘", "PostgreSQL Pool", "psycopg_pool state checkpointer"),
    ]

    for icon, name, desc in TECHS:
        st.markdown(f"""
        <div class="sb-tech-chip">
            <span style="font-size:1.05rem;">{icon}</span>
            <span><strong style="color:#0F172A;font-weight:700;">{name}</strong>
                  <span style="color:#1E293B;font-size:0.75rem;font-weight:600;"> — {desc}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E2E8F0;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;color:#0F172A;margin-bottom:0.6rem;'>🤖 Multi-Agent Graph Topology</div>", unsafe_allow_html=True)

    GRAPH_NODES = [
        {"role":"ORCHESTRATOR","icon":"🎯","name":"Supervisor Node","desc":"Task decomposition & dynamic routing","badge":"Master","accent":"#2563EB","bg":"rgba(37,99,235,0.12)","badge_bg":"rgba(37,99,235,0.15)","badge_color":"#1D4ED8"},
        {"role":"PARALLEL FAN-OUT","icon":"🔬","name":"Research Agent","desc":"Two-Stage Qdrant RAG + Visa Rules","badge":"Qdrant RAG","accent":"#4F46E5","bg":"rgba(79,70,229,0.12)","badge_bg":"rgba(79,70,229,0.15)","badge_color":"#4338CA"},
        {"role":"PARALLEL FAN-OUT","icon":"🚆","name":"Flight & Transit Agent","desc":"Tavily MCP Search & Transit Options","badge":"MCP Search","accent":"#2563EB","bg":"rgba(37,99,235,0.12)","badge_bg":"rgba(37,99,235,0.15)","badge_color":"#1D4ED8"},
        {"role":"PARALLEL FAN-OUT","icon":"🏨","name":"Hotel Agent","desc":"Accommodation & Area Recommendations","badge":"MCP Search","accent":"#7C3AED","bg":"rgba(124,58,237,0.12)","badge_bg":"rgba(124,58,237,0.15)","badge_color":"#6D28D9"},
        {"role":"PARALLEL FAN-OUT","icon":"🌤️","name":"Weather Agent","desc":"OpenWeather MCP Live Temp & Forecast","badge":"MCP Weather","accent":"#0891B2","bg":"rgba(8,145,178,0.12)","badge_bg":"rgba(8,145,178,0.15)","badge_color":"#0E7490"},
        {"role":"FAN-IN SYNTHESIS","icon":"💰","name":"Budget Agent","desc":"Multi-tier Cost Calculations in INR ₹","badge":"Financial LLM","accent":"#D97706","bg":"rgba(217,119,6,0.12)","badge_bg":"rgba(217,119,6,0.15)","badge_color":"#B45309"},
        {"role":"FAN-IN SYNTHESIS","icon":"🗓️","name":"Itinerary Agent","desc":"Final Day-by-Day Comprehensive Plan","badge":"Synthesizer","accent":"#059669","bg":"rgba(5,150,105,0.12)","badge_bg":"rgba(5,150,105,0.15)","badge_color":"#047857"},
    ]
    for ag in GRAPH_NODES:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-card-accent" style="background:{ag['accent']};"></div>
            <div class="agent-card-inner">
                <div class="agent-icon-box" style="background:{ag['bg']};">{ag['icon']}</div>
                <div style="flex:1;">
                    <div style="font-size:0.62rem;font-weight:800;color:{ag['accent']};letter-spacing:0.08em;">{ag['role']}</div>
                    <div class="agent-name">{ag['name']}</div>
                    <div class="agent-desc">{ag['desc']}</div>
                </div>
                <div class="agent-badge" style="background:{ag['badge_bg']};color:{ag['badge_color']};border:1px solid {ag['badge_color']}40;">{ag['badge']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Hero Section ──
st.markdown("""
<div class="hero-wrapper">
    <img style="width:100%;height:100%;object-fit:cover;filter:brightness(0.45);position:absolute;inset:0;"
         src="https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1400&q=80"/>
    <div class="hero-content">
        <div class="hero-badge">✦ LangGraph Supervisor · MCP · Two-Stage Qdrant RAG</div>
        <div class="hero-title">✈️ AI Multi-Agent Travel Planner</div>
        <div class="hero-sub">Orchestrated by a Master Supervisor with parallel specialist execution across destination RAG, transport, accommodations, live weather, budgeting, and comprehensive day-wise itinerary synthesis.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Destination Quick Select ──
DESTINATIONS = [
    ("🇯🇵 Tokyo, Japan",    "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=300&q=70"),
    ("🇫🇷 Paris, France",    "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=300&q=70"),
    ("🇦🇪 Dubai, UAE",      "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=300&q=70"),
    ("🇮🇩 Bali, Indonesia",  "https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=300&q=70"),
    ("🇹🇭 Bangkok, Thailand","https://images.unsplash.com/photo-1508009603885-50cf7c579365?w=300&q=70"),
]
cols = st.columns(5)
for col, (name, img_url) in zip(cols, DESTINATIONS):
    with col:
        st.markdown(f"""
        <div style="border-radius:10px;overflow:hidden;position:relative;height:85px;border:1px solid #E2E8F0;">
            <img src="{img_url}" style="width:100%;height:100%;object-fit:cover;filter:brightness(0.6);"/>
            <div style="position:absolute;bottom:8px;left:0;right:0;text-align:center;color:#fff;font-size:0.8rem;font-weight:700;text-shadow:0 2px 10px rgba(0,0,0,0.6);">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Query Input ──
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

QUICK = [
    "7-day Japan cherry blossom trip under ₹2.5 Lakhs",
    "Paris 5-day cultural holiday with museums and food",
    "Dubai 4-day luxury weekend with desert safari",
    "Bali 10-day budget backpacking and scuba diving"
]
qcols = st.columns(len(QUICK))
for qc, label in zip(qcols, QUICK):
    with qc:
        if st.button(label, key=f"q_{label}"):
            st.session_state["user_query"] = label

user_query = st.text_area(
    "Describe your trip:",
    value=st.session_state["user_query"],
    placeholder="e.g. Plan a 7-day trip to Tokyo and Kyoto including visa guidelines, flights from Mumbai, boutique hotels, and budget breakdown in INR",
    height=95,
    label_visibility="collapsed"
)
st.session_state["user_query"] = user_query
generate = st.button("🚀  Orchestrate Travel Multi-Agent Graph", use_container_width=True)

AGENT_META = {
    "supervisor":      ("🎯", "Supervisor Orchestrator", "Routing & Planning"),
    "research_agent":  ("🔬", "Research Agent", "Two-Stage Qdrant RAG + Visa Rules"),
    "flight_agent":    ("🚆", "Flight & Transport Agent", "MCP Web Search"),
    "hotel_agent":     ("🏨", "Hotel Agent", "MCP Accommodation Search"),
    "weather_agent":   ("🌤️", "Weather Agent", "MCP Live Weather & Forecast"),
    "budget_agent":    ("💰", "Budget Agent", "INR Cost Calculation"),
    "itinerary_agent": ("🗓️", "Itinerary Agent", "Comprehensive Master Plan Synthesis"),
}


# ── Graph Execution Streaming ──
if generate:
    if not user_query.strip():
        st.warning("Please describe your trip before generating.")
    else:
        config = {"configurable": {"thread_id": thread_id}}
        collected = {
            "research_results": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "llm_calls": 0,
            "destination": "",
            "trip_type": ""
        }

        st.markdown("---")
        st.markdown("<div class='sec-head'><span>⚡ Live Multi-Agent Graph Execution</span></div>", unsafe_allow_html=True)

        for chunk in app.stream(
            {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "destination": "",
                "trip_type": "",
                "plan_steps": [],
                "active_agent": "start",
                "research_results": "",
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "budget_results": "",
                "itinerary": "",
                "llm_calls": 0,
            },
            config=config,
            stream_mode="updates",
        ):
            for node_name, state_update in chunk.items():
                icon, label, subtitle = AGENT_META.get(node_name, ("🔧", node_name, "Processing"))

                st.markdown(
                    f"<div style='background:#F8FAFC;color:#0F172A;border:1.5px solid #E2E8F0;padding:10px 16px;"
                    f"border-radius:10px 10px 0 0;font-size:1.05rem;font-weight:700;"
                    f"letter-spacing:0.02em;margin-top:1rem;display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span>{icon} {label}</span>"
                    f"<span style='font-size:0.75rem;color:#64748B;font-weight:500;'>{subtitle}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                with st.container(border=True):
                    if node_name == "supervisor":
                        dest = state_update.get("destination", "Detected")
                        ttype = state_update.get("trip_type", "Trip")
                        steps = state_update.get("plan_steps", [])
                        collected["destination"] = dest
                        collected["trip_type"] = ttype
                        st.info(f"**Supervisor Decision:** Destination: `{dest}` | Trip Type: `{ttype}` | Specialists Dispatched: `{', '.join(steps)}`")

                    elif node_name == "research_agent":
                        text = state_update.get("research_results", "")
                        collected["research_results"] = text
                        st.markdown(strip_markdown_tables(clean_llm_output(text)) or "_No research data._")

                    elif node_name == "flight_agent":
                        text = state_update.get("flight_results", "")
                        collected["flight_results"] = text
                        st.markdown(strip_markdown_tables(clean_llm_output(text)) or "_No transit data._")

                    elif node_name == "hotel_agent":
                        text = state_update.get("hotel_results", "")
                        collected["hotel_results"] = text
                        st.markdown(strip_markdown_tables(clean_llm_output(text)) or "_No hotel data._")

                    elif node_name == "weather_agent":
                        text = state_update.get("weather_results", "")
                        collected["weather_results"] = text
                        if text:
                            st.markdown('<div class="weather-card">', unsafe_allow_html=True)
                            st.markdown('<div class="weather-title">🌤️ Real-Time Weather & Forecast</div>', unsafe_allow_html=True)
                            st.markdown(tidy_weather(clean_llm_output(text)))
                            st.markdown('</div>', unsafe_allow_html=True)

                    elif node_name == "budget_agent":
                        text = state_update.get("budget_results", "")
                        collected["budget_results"] = text
                        st.markdown(strip_markdown_tables(clean_llm_output(text)) or "_No budget data._")

                    elif node_name == "itinerary_agent":
                        text = state_update.get("itinerary", "")
                        collected["itinerary"] = text
                        st.markdown(strip_markdown_tables(clean_llm_output(text)) or "_No itinerary._")

                    collected["llm_calls"] += state_update.get("llm_calls", 1)

        # ── Metrics ──
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">7</div><div class="metric-lbl">Graph Nodes</div></div>
            <div class="metric-box"><div class="metric-val">{collected['llm_calls']}</div><div class="metric-lbl">LLM Inferences</div></div>
            <div class="metric-box"><div class="metric-val">Parallel</div><div class="metric-lbl">Execution Mode</div></div>
            <div class="metric-box"><div class="metric-val">PostgreSQL</div><div class="metric-lbl">Checkpoint State</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Final Master Itinerary ──
        if collected["itinerary"]:
            st.markdown("<div class='sec-head'><span>🗓️ Master Synthesized Travel Itinerary</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='final-card'>{strip_markdown_tables(clean_llm_output(collected['itinerary']))}</div>", unsafe_allow_html=True)

        # ── Save & Download ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_md = f"travel_plan_{timestamp}.md"
        filename_pdf = f"travel_plan_{timestamp}.pdf"
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "travel_plans")
        os.makedirs(save_dir, exist_ok=True)

        file_content = f"""# Master Travel Plan
**Query:** {user_query}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Session ID:** {thread_id}
**Orchestration:** LangGraph Supervisor (Parallel Fan-Out & Fan-In)

---

## 🎯 Supervisor Strategy
- **Destination:** {collected['destination']}
- **Classification:** {collected['trip_type']}

---

## 🔬 Destination Research (Two-Stage Qdrant RAG)
{collected['research_results'] or 'N/A'}

---

## ✈️ Flight & Ground Transportation (MCP Search)
{collected['flight_results'] or 'N/A'}

---

## 🏨 Accommodation & Stays (MCP Search)
{collected['hotel_results'] or 'N/A'}

---

## 🌤️ Weather Conditions & Advisory (FastMCP OpenWeather)
{collected['weather_results'] or 'N/A'}

---

## 💰 Budget Breakdown (INR ₹)
{collected['budget_results'] or 'N/A'}

---

## 🗓️ Master Synthesized Itinerary
{collected['itinerary'] or 'N/A'}
"""
        with open(os.path.join(save_dir, filename_md), "w", encoding="utf-8") as f:
            f.write(file_content)

        pdf_bytes = generate_pdf_bytes(file_content)
        try:
            with open(os.path.join(save_dir, filename_pdf), "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            print(f"Warning: PDF file write failed: {e}")

        st.download_button(
            "📄 Download PDF",
            data=pdf_bytes,
            file_name=filename_pdf,
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf"
        )