# TODO: Fix undefined variables: BaseModel, Field, SentenceTransformer, df, document_embeddings, e, filename, idx, model, query, query_embedding, results, similarities, tool, top_indices, top_k
"""
Semantic search tool implementation.
"""
from agent import query
from benchmarks.cot_performance import df
from benchmarks.cot_performance import filename
from examples.parallel_execution_example import results

from src.core.optimized_chain_of_thought import idx
from src.database.models import tool
from src.gaia_components.production_vector_store import model
from src.gaia_components.production_vector_store import query_embedding
from src.gaia_components.production_vector_store import similarities
from src.utils.semantic_search_tool import document_embeddings
from src.utils.semantic_search_tool import top_indices


import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import logging
# TODO: Fix undefined variables: SentenceTransformer, df, document_embeddings, e, filename, idx, logging, model, os, query, query_embedding, results, similarities, top_indices, top_k
from pydantic import Field

from src.tools.base_tool import tool


logger = logging.getLogger(__name__)

# Try to import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not available - semantic search disabled")

class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(description="Search query")
    filename: str = Field(description="Path to the knowledge base file")
    top_k: int = Field(default=3, description="Number of results to return")

@tool
def semantic_search_tool(query: str, filename: str, top_k: int = 3) -> str:
    """
    Perform semantic search on a knowledge base.

    Args:
        query (str): Search query
        filename (str): Path to the knowledge base file
        top_k (int): Number of results to return

    Returns:
        str: Search results or error message
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return "Error: sentence_transformers not available. Please install with: pip install sentence-transformers"

    try:
        if not os.path.exists(filename):
            return f"Error: Knowledge base file not found: {filename}"

        # Load knowledge base
        df = pd.read_csv(filename)

        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode query
        query_embedding = model.encode(query)

        # Encode documents
        document_embeddings = model.encode(df['text'].tolist())

        # Calculate similarities
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'text': df.iloc[idx]['text'],
                'similarity': float(similarities[idx])
            })

        return str(results)

    except Exception as e:
        return f"Error performing semantic search: {str(e)}"
