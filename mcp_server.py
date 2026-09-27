"""
Travel Tools MCP Server
Exposes Tavily Search and OpenWeatherMap tools via standard Model Context Protocol (MCP).
Can be run via stdio (for local subprocess transport) or SSE.
"""

import sys
import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Initialize FastMCP Server
mcp = FastMCP("TravelPlanningTools")


@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the live web using the Tavily Search API.
    Returns structured markdown links, titles, and snippets.
    
    Args:
        query: The search query string (e.g. flight, hotel, or transit search).
        max_results: Maximum number of search results to return (default: 5).
    """
    print(f"[MCP Server stdio] Executing tavily_search(query='{query[:40]}...', max_results={max_results})", file=sys.stderr)
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY is not set."
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
        data = response.json()
        results = data.get("results", [])
        if not results:
            return "No web results found."

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
        return f"MCP Tavily Search Error: {str(e)}"


@mcp.tool()
def get_current_weather(city: str) -> str:
    """
    Get real-time live weather conditions for a specified city or location.
    
    Args:
        city: Destination city or locality name (e.g. 'Tokyo', 'Paris').
    """
    print(f"[MCP Server stdio] Executing get_current_weather(city='{city}')", file=sys.stderr)
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY is not set."
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
        return f"MCP Weather Error: {str(e)}"


@mcp.tool()
def get_forecast(city: str) -> str:
    """
    Get a 5-period forecast for a specified city or location.
    
    Args:
        city: Destination city or locality name (e.g. 'Tokyo', 'Paris').
    """
    print(f"[MCP Server stdio] Executing get_forecast(city='{city}')", file=sys.stderr)
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY is not set."
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
        return f"MCP Forecast Error: {str(e)}"


if __name__ == "__main__":
    # Run FastMCP via standard I/O transport
    mcp.run()
