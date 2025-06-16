import os
import logging
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding

from src.database import get_vector_store

# Configure logging
logger = logging.getLogger(__name__)

# --- Tool Definitions ---

def get_tools() -> list:
    """
    Initializes and returns a list of all tools available to the agent.
    """
    # 1. Tavily Search Tool for real-time web searches
    tavily_search_tool = TavilySearchResults(
        max_results=3,
        description='A search engine optimized for comprehensive, accurate, and trusted results. Useful for searching the web for current events, facts, and real-time information.'
    )

    # 2. LlamaIndex Knowledge Base Tool for private data retrieval
    # Initialize the vector store and index
    try:
        vector_store = get_vector_store()
        # Ensure embedding model is configured for the index
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # Create a query engine from the index
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        # Wrap the query engine in a tool
        knowledge_base_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="knowledge_base_retriever",
            description=(
                "Useful for answering questions about internal company policies, project documentation (like Project Orion), and historical data. Use this for private, non-public information."
            )
        )
    except Exception as e:
        logger.error(f"Failed to initialize Knowledge Base tool: {e}. It will not be available.")
        knowledge_base_tool = None


    # 3. Python Code Interpreter Tool for calculations and code execution
    # SECURITY WARNING: This tool executes arbitrary Python code.
    # Do not use in a production environment with untrusted inputs without proper sandboxing.
    python_repl_tool = PythonREPLTool()

    # Assemble the list of tools, filtering out any that failed to initialize
    tools = [tavily_search_tool, python_repl_tool]
    if knowledge_base_tool:
        tools.append(knowledge_base_tool)
        
    logger.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")
    return tools

# Example of a custom tool using the @tool decorator
@tool
def get_weather(city: str) -> str:
    """
    A dummy tool to get the weather for a given city.
    In a real application, this would call a weather API.
    """
    if "san francisco" in city.lower():
        return "The weather in San Francisco is 65°F and sunny."
    elif "new york" in city.lower():
        return "The weather in New York is 75°F and humid."
    else:
        return f"Sorry, I don't have weather information for {city}."

# To add the custom tool, you would append it to the list in get_tools()
# tools.append(get_weather) 