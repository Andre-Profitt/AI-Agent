# TODO: Fix undefined variables: BaseModel, Field, TavilySearch, attempt, base_delay, delay, e, max_retries, query, results, search, tool
"""
Tavily search tool implementations.
"""
from agent import query
from agent import search
from examples.parallel_execution_example import results

from src.database.models import tool
from src.gaia_components.adaptive_tool_system import attempt
from src.utils.base_tool import TavilySearch
from src.utils.tavily_search import base_delay
from src.utils.tavily_search import delay
from src.utils.tavily_search import max_retries


import time

from langchain_core.tools import tool
from pydantic import BaseModel, Field
# TODO: Fix undefined variables: attempt, base_delay, delay, e, max_retries, query, results, search, time
from pydantic import Field

from src.tools.base_tool import tool
from src.utils.tools_enhanced import TavilySearch


class TavilySearchInput(BaseModel):
    """Input schema for Tavily search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=3, description="Maximum number of results to return")

@tool
def tavily_search(query: str, max_results: int = 3) -> str:
    """
    Search using Tavily API.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return

    Returns:
        str: Search results or error message
    """
    try:
        from langchain_tavily import TavilySearch

        search = TavilySearch()
        results = search.run(query)

        return str(results)

    except Exception as e:
        return f"Error performing Tavily search: {str(e)}"

@tool
def tavily_search_backoff(query: str, max_results: int = 3) -> str:
    """
    Search using Tavily API with exponential backoff.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return

    Returns:
        str: Search results or error message
    """
    try:

        search = TavilySearch()
        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries):
            try:
                results = search.run(query)
                return str(results)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

    except Exception as e:
        return f"Error performing Tavily search with backoff: {str(e)}"
