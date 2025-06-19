"""
Semantic search tool implementation.
"""

import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import logging

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