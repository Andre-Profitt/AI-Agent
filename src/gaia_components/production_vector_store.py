from agent import match
from agent import query
from examples.parallel_execution_example import results
from performance_dashboard import stats

from src.config.integrations import api_key
from src.core.langgraph_compatibility import batch
from src.core.langgraph_compatibility import loop
from src.core.llamaindex_enhanced import documents
from src.core.optimized_chain_of_thought import similarity
from src.database.models import metadata
from src.database.models import model_name
from src.database.models import text
from src.database.models import vector_store
from src.database.supabase_manager import cache_key
from src.gaia_components.adaptive_tool_system import dot_product
from src.gaia_components.adaptive_tool_system import norm1
from src.gaia_components.adaptive_tool_system import norm2
from src.gaia_components.advanced_reasoning_engine import embedding
from src.gaia_components.advanced_reasoning_engine import embeddings
from src.gaia_components.advanced_reasoning_engine import hash_val
from src.gaia_components.performance_optimization import batch_size
from src.gaia_components.production_vector_store import cached_indices
from src.gaia_components.production_vector_store import count
from src.gaia_components.production_vector_store import dummy_text
from src.gaia_components.production_vector_store import environment
from src.gaia_components.production_vector_store import final_embeddings
from src.gaia_components.production_vector_store import formatted_results
from src.gaia_components.production_vector_store import ids
from src.gaia_components.production_vector_store import model
from src.gaia_components.production_vector_store import new_embeddings
from src.gaia_components.production_vector_store import new_idx
from src.gaia_components.production_vector_store import persist_dir
from src.gaia_components.production_vector_store import query_embedding
from src.gaia_components.production_vector_store import similarities
from src.gaia_components.production_vector_store import store_type
from src.gaia_components.production_vector_store import texts
from src.gaia_components.production_vector_store import texts_to_embed
from src.gaia_components.production_vector_store import vectors
from src.infrastructure.embedding_manager import vec1
from src.infrastructure.embedding_manager import vec2
from src.query_classifier import doc
from src.utils.base_tool import device
from src.utils.knowledge_utils import doc_id

"""
from abc import abstractmethod
from typing import List
from src.config.settings import Settings
# TODO: Fix undefined variables: ABC, Any, Dict, List, Path, SentenceTransformer, Tuple, a, abstractmethod, api_key, b, batch, batch_size, cache_key, cached_indices, concurrent, count, device, doc, doc_id, doc_vector, documents, dot_product, dummy_text, e, embedding, embeddings, environment, final_embeddings, formatted_results, hash_val, i, id_, ids, index_name, k, logging, loop, match, metadata, metadatas, model, model_name, new_embeddings, new_idx, norm1, norm2, os, persist_dir, persist_directory, query, query_embedding, results, similarities, similarity, stats, store_type, text, texts, texts_to_embed, time, vec1, vec2, vector_store, vectors, x
# TODO: Fix undefined variables: SentenceTransformer, a, api_key, b, batch, batch_size, cache_key, cached_indices, chromadb, concurrent, count, device, doc, doc_id, doc_vector, documents, dot_product, dummy_text, e, embedding, embeddings, environment, final_embeddings, formatted_results, hash_val, hashlib, i, id_, ids, index_name, k, loop, match, metadata, metadatas, model, model_name, new_embeddings, new_idx, norm1, norm2, persist_dir, persist_directory, pinecone, query, query_embedding, results, self, similarities, similarity, stats, store_type, text, texts, texts_to_embed, torch, vec1, vec2, vector_store, vectors, x

Production-Ready Vector Store Implementation for GAIA System
Supports multiple vector store providers with real embeddings
"""

from typing import Tuple
from typing import Dict
from typing import Any

import os
import logging
import hashlib
import time

from abc import ABC, abstractmethod
import asyncio
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreInterface(ABC):
    """Abstract interface for vector stores"""

    @abstractmethod
    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass

class ProductionEmbeddings:
    """Production-ready embedding system with multiple providers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """Initialize embeddings with automatic device selection"""
        self.model_name = model_name
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")

        # Initialize embedding model
        self.model = self._initialize_model()

        # Cache for frequently embedded texts
        self.embedding_cache = {}
        self.cache_size = 10000

        logger.info(f"Production embeddings initialized with model: {model_name} on {self.device}")

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model with device specification
            model = SentenceTransformer(self.model_name, device=self.device)

            # Warm up the model
            dummy_text = "This is a warmup text for the embedding model."
            _ = model.encode(dummy_text, convert_to_tensor=False)

            return model

        except ImportError:
            logger.warning("SentenceTransformers not available, using fallback embeddings")
            return self._create_fallback_embeddings()

    def _create_fallback_embeddings(self):
        """Create fallback embedding function"""
        class FallbackEmbeddings:
            def encode(self, texts, convert_to_tensor=False, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]

                embeddings = []
                for text in texts:
                    # Simple hash-based embedding
                    hash_val = hash(text.lower()) % 1000
                    embedding = [float(hash_val + i) / 1000.0 for i in range(384)]
                    embeddings.append(embedding)

                if len(embeddings) == 1:
                    return embeddings[0]
                return embeddings

        return FallbackEmbeddings()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.cache_size,
            "cache_utilization": len(self.embedding_cache) / self.cache_size,
            "model_name": self.model_name,
            "device": self.device
        }

        """Embed multiple documents with caching"""
        embeddings = []
        texts_to_embed = []
        cached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                cached_indices.append(i)
            else:
                texts_to_embed.append(text)

        # Embed uncached texts
        if texts_to_embed:
            try:
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    convert_to_tensor=False,
                    show_progress_bar=len(texts_to_embed) > 100,
                    batch_size=32
                )

                # Convert to list if needed
                if hasattr(new_embeddings, 'tolist'):
                    new_embeddings = new_embeddings.tolist()

                # Add to cache
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    cache_key = hashlib.md5(text.encode()).hexdigest()
                    if len(self.embedding_cache) < self.cache_size:
                        self.embedding_cache[cache_key] = embedding

                # Merge cached and new embeddings in correct order
                final_embeddings = []
                new_idx = 0
                for i in range(len(texts)):
                    if i in cached_indices:
                        final_embeddings.append(embeddings[cached_indices.index(i)])
                    else:
                        new_idx += 1

                return final_embeddings

            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                # Return fallback embeddings
                return self._generate_fallback_embeddings(texts)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with optimization"""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            embedding = self.model.encode(text, convert_to_tensor=False)

            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            if len(self.embedding_cache) < self.cache_size:
                self.embedding_cache[cache_key] = embedding
            
            return embedding

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            # Return fallback embedding
            hash_val = hash(text.lower()) % 1000
            return [float(hash_val + i) / 1000.0 for i in range(384)]

    def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate fallback embeddings"""
        embeddings = []
        for text in texts:
            hash_val = hash(text.lower()) % 1000
            embedding = [float(hash_val + i) / 1000.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings

class ChromaVectorStore(VectorStoreInterface):
    """Production ChromaDB vector store implementation"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embeddings = ProductionEmbeddings()
        self.client = None
        self.collection = None

        self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False
            ))

            self.collection = self.client.get_or_create_collection(
                name="gaia_agent",
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"ChromaDB initialized at {self.persist_directory}")

        except ImportError:
            logger.error("ChromaDB not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add documents to ChromaDB"""
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)

            # Create IDs
            ids = [f"doc_{hash(doc)}_{int(time.time())}_{i}" for i, doc in enumerate(documents)]

            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append((
                        results['documents'][0][i],
                        1 - results['distances'][0][i],  # Convert distance to similarity
                        results['metadatas'][0][i] if results['metadatas'] else {}
                    ))

            return formatted_results

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        if not self.collection:
            return False

        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        if not self.collection:
            return {"error": "Collection not initialized"}

        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "persist_directory": str(self.persist_directory),
                "embedding_cache_stats": self.embeddings.get_cache_stats()
            }
        except Exception as e:
            return {"error": str(e)}

class PineconeVectorStore(VectorStoreInterface):
    """Production Pinecone vector store implementation"""

    def __init__(self, api_key: str, environment: str, index_name: str = "gaia-agent"):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.embeddings = ProductionEmbeddings()
        self.index = None

        self._initialize_pinecone()

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            import pinecone

            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=384,  # For all-MiniLM-L6-v2
                    metric='cosine',
                    pods=1,
                    replicas=1,
                    pod_type='p1.x1'
                )

            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone initialized with index: {self.index_name}")

        except ImportError:
            logger.error("Pinecone not available")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add documents to Pinecone"""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")

        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)

            # Create IDs
            ids = [f"doc_{hash(doc)}_{int(time.time())}_{i}" for i, doc in enumerate(documents)]

            # Prepare vectors
            vectors = []
            for i, (id_, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
                vectors.append({
                    'id': id_,
                    'values': embedding,
                    'metadata': metadata
                })

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Added {len(documents)} documents to Pinecone")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise

    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        if not self.index:
            raise RuntimeError("Pinecone index not initialized")

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )

            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append((
                    match['id'],
                    match['score'],
                    match.get('metadata', {})
                ))

            return formatted_results

        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        if not self.index:
            return False

        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from Pinecone: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics"""
        if not self.index:
            return {"error": "Index not initialized"}

        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_name": self.index_name,
                "embedding_cache_stats": self.embeddings.get_cache_stats()
            }
        except Exception as e:
            return {"error": str(e)}

class InMemoryVectorStore(VectorStoreInterface):
    """In-memory vector store for testing and development"""

    def __init__(self):
        self.embeddings = ProductionEmbeddings()
        self.documents = {}
        self.metadatas = {}
        self.vectors = {}

        logger.info("In-memory vector store initialized")

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add documents to in-memory store"""
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)

            # Create IDs
            ids = [f"doc_{hash(doc)}_{int(time.time())}_{i}" for i, doc in enumerate(documents)]

            # Store documents
            for id_, doc, embedding, metadata in zip(ids, documents, embeddings, metadatas):
                self.documents[id_] = doc
                self.vectors[id_] = embedding
                self.metadatas[id_] = metadata

            logger.info(f"Added {len(documents)} documents to in-memory store")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents to in-memory store: {e}")
            raise

    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Calculate similarities
            similarities = []
            for doc_id, doc_vector in self.vectors.items():
                similarity = self._cosine_similarity(query_embedding, doc_vector)
                similarities.append((doc_id, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top k results
            results = []
            for doc_id, similarity in similarities[:k]:
                results.append((
                    self.documents[doc_id],
                    similarity,
                    self.metadatas.get(doc_id, {})
                ))

            return results

        except Exception as e:
            logger.error(f"In-memory search failed: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            for doc_id in ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                if doc_id in self.vectors:
                    del self.vectors[doc_id]
                if doc_id in self.metadatas:
                    del self.metadatas[doc_id]

            logger.info(f"Deleted {len(ids)} documents from in-memory store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from in-memory store: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get in-memory store statistics"""
        return {
            "total_documents": len(self.documents),
            "store_type": "in_memory",
            "embedding_cache_stats": self.embeddings.get_cache_stats()
        }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

# Factory function to create appropriate vector store
def create_vector_store(store_type: str = "chroma") -> VectorStoreInterface:
    """Create a vector store based on configuration"""

    if store_type == "pinecone":
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        if not api_key or not environment:
            logger.warning("Pinecone API key or environment not set, falling back to ChromaDB")
            store_type = "chroma"
        else:
            return PineconeVectorStore(api_key, environment)

    if store_type == "chroma":
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        return ChromaVectorStore(persist_dir)

    elif store_type == "memory":
        return InMemoryVectorStore()

    else:
        logger.warning(f"Unknown vector store type: {store_type}, using in-memory")
        return InMemoryVectorStore()

# Async wrapper for vector store operations
class AsyncVectorStore:
    """Async wrapper for vector store operations"""

    def __init__(self, vector_store: VectorStoreInterface):
        self.vector_store = vector_store
        self.executor = None

    async def __aenter__(self):
        import concurrent.futures
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Add documents asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(self.vector_store.add_documents(documents, metadatas))
        )

    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(self.vector_store.search(query, k))
        )

    async def delete(self, ids: List[str]) -> bool:
        """Delete documents asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(self.vector_store.delete(ids))
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get stats asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(self.vector_store.get_stats())
        )
