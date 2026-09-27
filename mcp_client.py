"""
MCP Client Manager module for AI Travel Planner System.
Connects to FastMCP Server (mcp_server.py) over stdio using the official MCP client protocol.
Provides cached tool invocation and graceful fallback to direct tools if needed.
"""

import sys
import os
import asyncio
import threading
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Fallback imports
import tools as fallback_tools

load_dotenv()


class TravelMCPClientManager:
    """
    Manages client-side MCP lifecycle over stdio transport.
    Exposes synchronous and asynchronous tool invocation interfaces for LangGraph agents.
    """

    def __init__(self):
        self.server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        self.python_exe = sys.executable
        self._lock = threading.Lock()
        self._tools_cache = None

    async def _call_mcp_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Connects to the FastMCP server via stdio and invokes the requested tool via JSON-RPC 2.0.
        """
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession

        server_params = StdioServerParameters(
            command=self.python_exe,
            args=[self.server_script],
            env=dict(os.environ)
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                # Call tool over MCP
                result = await session.call_tool(tool_name, arguments=arguments)
                
                # Extract text content from result
                if result and result.content:
                    texts = [c.text for c in result.content if hasattr(c, "text")]
                    return "\n".join(texts)
                return "No content returned from MCP tool."

    def invoke_tool_sync(self, tool_name: str, **kwargs) -> str:
        """
        Synchronously invokes an MCP tool over stdio transport with automatic fallback.
        Safe for use inside synchronous LangGraph nodes.
        """
        try:
            # Run async MCP invocation in a clean event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If running inside an existing loop (e.g. Streamlit or Jupyter), run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self._call_mcp_tool_async(tool_name, kwargs))
                    return future.result(timeout=25)
            else:
                return loop.run_until_complete(self._call_mcp_tool_async(tool_name, kwargs))

        except Exception as e:
            print(f"[Warning] MCP Tool invocation '{tool_name}' failed via stdio ({e}). Using resilience fallback...")
            return self._invoke_fallback(tool_name, **kwargs)

    def _invoke_fallback(self, tool_name: str, **kwargs) -> str:
        """Emergency fallback directly calling tools.py if MCP subprocess transport fails."""
        if tool_name == "tavily_search":
            return fallback_tools.search_web(
                query=kwargs.get("query", ""),
                max_results=kwargs.get("max_results", 5)
            )
        elif tool_name == "get_current_weather":
            return fallback_tools.get_current_weather(city=kwargs.get("city", ""))
        elif tool_name == "get_forecast":
            return fallback_tools.get_forecast(city=kwargs.get("city", ""))
        return f"Tool '{tool_name}' unavailable in fallback."


# Global client instance
mcp_client = TravelMCPClientManager()
