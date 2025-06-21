# TODO: Fix undefined variables: OpenAI, SentenceTransformer, cls, data, e, embedding, embeddings, response, self, text, texts
"""

from langchain.llms import OpenAI
Embedding Manager - Centralized Embedding Management
Fixes the critical embedding consistency issue where different components
were using different embedding methods (random vs real embeddings).
"""
from agent import response
from tests.load_test import data

from src.database.models import text
from src.gaia_components.advanced_reasoning_engine import embedding
from src.gaia_components.advanced_reasoning_engine import embeddings
from src.gaia_components.production_vector_store import texts


from typing import Any
from typing import List

import os
import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Centralized embedding management to ensure consistency across all components"""

    _instance: Optional['EmbeddingManager'] = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize embedding model once"""
        self._client = None
        self._model = None

        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self._client = OpenAI()
                self.method = "openai"
                self.dimension = 1536
                logger.info("Using OpenAI embeddings")
            except ImportError:
                logger.warning("OpenAI not available, falling back to local embeddings")
                self._setup_local_embeddings()
        else:
            self._setup_local_embeddings()

    def _setup_local_embeddings(self) -> Any:
        """Setup local sentence transformer embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self.method = "local"
            self.dimension = 384
            logger.info("Using local sentence transformer embeddings")
        except ImportError:
            logger.error("No embedding models available!")
            self.method = "none"
            self.dimension = 0

    def embed(self, text: str) -> List[float]:
        """Get embedding for text"""
        if not text or self.method == "none":
            # Return zero vector as fallback
            return [0.0] * max(self.dimension, 384)

        if self.method == "openai":
            try:
                response = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error("OpenAI embedding failed: {}", extra={"e": e})
                return [0.0] * self.dimension
        else:
            try:
                embedding = self._model.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.error("Local embedding failed: {}", extra={"e": e})
                return [0.0] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (more efficient)"""
        if not texts:
            return []

        if self.method == "openai":
            try:
                response = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"OpenAI batch embedding failed: {e}")
                return [[0.0] * self.dimension for _ in texts]
        else:
            try:
                embeddings = self._model.encode(texts)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Local batch embedding failed: {e}")
                return [[0.0] * self.dimension for _ in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_method(self) -> str:
        """Get embedding method being used"""
        return self.method

# Global embedding manager instance
embedding_manager = EmbeddingManager()

def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance"""
    return embedding_manager