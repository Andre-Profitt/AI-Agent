from src.agents.advanced_agent_fsm import Agent

"""

from typing import Any
from typing import Dict
from typing import Optional
Centralized embedding manager for the AI Agent system.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from functools import lru_cache

# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global embedding manager instance
_embedding_manager = None


class EmbeddingManager:
    """Centralized embedding manager with multiple backends"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", backend: str = "sentence_transformers"):
        self.model_name = model_name
        self.backend = backend
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize the appropriate backend
        self._initialize_backend()
        
        logger.info("Embedding manager initialized with {} backend using {}", extra={"backend": backend, "model_name": model_name})
    
    def _initialize_backend(self):
        """Initialize the embedding backend"""
        if self.backend == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info("Initialized SentenceTransformer with dimension {}", extra={"self_dimension": self.dimension})
            except Exception as e:
                logger.error("Failed to initialize SentenceTransformer: {}", extra={"e": e})
                self._fallback_initialization()
        
        elif self.backend == "openai" and OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")
                
                self.model = "text-embedding-ada-002"
                self.dimension = 1536  # OpenAI ada-002 dimension
                logger.info("Initialized OpenAI embedding backend")
            except Exception as e:
                logger.error("Failed to initialize OpenAI backend: {}", extra={"e": e})
                self._fallback_initialization()
        
        else:
            self._fallback_initialization()
    
    def _fallback_initialization(self):
        """Fallback to simple hash-based embeddings"""
        logger.warning("Using fallback hash-based embeddings")
        self.backend = "fallback"
        self.dimension = 128
    
    @lru_cache(maxsize=1000)
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a text string"""
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        try:
            if self.backend == "sentence_transformers" and self.model:
                embedding = self.model.encode(text)
                return embedding.tolist()
            
            elif self.backend == "openai":
                return self._openai_embed(text)
            
            else:
                return self._fallback_embed(text)
                
        except Exception as e:
            logger.error(f"Embedding failed for text: {text[:50]}... Error: {e}")
            return [0.0] * self.dimension
    
    def _openai_embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error("OpenAI embedding failed: {}", extra={"e": e})
            return [0.0] * self.dimension
    
    def _fallback_embed(self, text: str) -> List[float]:
        """Generate simple hash-based embedding"""
        import hashlib
        
        # Create a simple hash-based embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float values
        embedding = []
        for i in range(0, min(len(hash_bytes), self.dimension * 4), 4):
            if i + 3 < len(hash_bytes):
                value = int.from_bytes(hash_bytes[i:i+4], byteorder='big')
                embedding.append((value % 10000) / 10000.0)  # Normalize to [0, 1]
        
        # Pad or truncate to required dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        
        return embedding[:self.dimension]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not texts:
            return []
        
        try:
            if self.backend == "sentence_transformers" and self.model:
                embeddings = self.model.encode(texts)
                return embeddings.tolist()
            
            elif self.backend == "openai":
                return self._openai_embed_batch(texts)
            
            else:
                return [self._fallback_embed(text) for text in texts]
                
        except Exception as e:
            logger.error("Batch embedding failed: {}", extra={"e": e})
            return [[0.0] * self.dimension for _ in texts]
    
    def _openai_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using OpenAI API"""
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error("OpenAI batch embedding failed: {}", extra={"e": e})
            return [[0.0] * self.dimension for _ in texts]
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error("Similarity calculation failed: {}", extra={"e": e})
            return 0.0
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend"""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "openai_available": OPENAI_AVAILABLE
        }


def get_embedding_manager(model_name: Optional[str] = None, backend: Optional[str] = None) -> EmbeddingManager:
    """Get or create the global embedding manager instance"""
    global _embedding_manager
    
    if _embedding_manager is None:
        # Use environment variables or defaults
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        backend = backend or os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
        
        _embedding_manager = EmbeddingManager(model_name=model_name, backend=backend)
    
    return _embedding_manager


def reset_embedding_manager():
    """Reset the global embedding manager (useful for testing)"""
    global _embedding_manager
    _embedding_manager = None


# Convenience functions
def embed_text(text: str) -> List[float]:
    """Quick function to embed a single text"""
    manager = get_embedding_manager()
    return manager.embed(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Quick function to embed a batch of texts"""
    manager = get_embedding_manager()
    return manager.embed_batch(texts)


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Quick function to calculate similarity between embeddings"""
    manager = get_embedding_manager()
    return manager.similarity(embedding1, embedding2) 