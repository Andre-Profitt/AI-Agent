from supabase import create_client, Client
from contextlib import asynccontextmanager
import asyncio
from typing import Optional, List, Dict, Any
import aiohttp
from dataclasses import dataclass
from langchain.schema import Document
import numpy as np
import logging
import os
from functools import lru_cache

# Import circuit breaker protection
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    circuit_breaker
)

try:
    from .config.integrations import integration_config
except ImportError:
    try:
        from config.integrations import integration_config
    except ImportError:
        # Fallback for when running as standalone script
        integration_config = None
        logging.warning("Could not import integration_config - using defaults")

# Import centralized embedding manager
from .embedding_manager import get_embedding_manager

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured search result"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

class SupabaseConnectionPool:
    """Enhanced Supabase client with connection pooling"""
    
    def __init__(self, url: str, key: str, pool_size: int = 10):
        self.url = url
        self.key = key
        self.pool_size = pool_size
        self._pool = asyncio.Queue(maxsize=pool_size)
        self._session = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize connection pool"""
        if self._initialized:
            return
            
        # Create custom aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.pool_size,
            limit_per_host=self.pool_size,
            ttl_dns_cache=300,
            keepalive_timeout=30
        )
        self._session = aiohttp.ClientSession(connector=connector)
        
        # Pre-create clients
        for _ in range(self.pool_size):
            client = create_client(self.url, self.key)
            await self._pool.put(client)
        
        self._initialized = True
        logger.info(f"Supabase connection pool initialized with {self.pool_size} connections")
    
    @asynccontextmanager
    async def get_client(self):
        """Get client from pool"""
        if not self._initialized:
            await self.initialize()
            
        client = await self._pool.get()
        try:
            yield client
        finally:
            await self._pool.put(client)
    
    async def close(self):
        """Close all connections"""
        if self._session:
            await self._session.close()
        self._initialized = False

class OptimizedVectorStore:
    """Optimized vector store with batch operations and caching"""
    
    def __init__(self, pool: SupabaseConnectionPool):
        self.pool = pool
        self.config = integration_config
        
        # Use centralized embedding manager instead of local initialization
        self.embedding_manager = get_embedding_manager()
        
        # Use functools.lru_cache for proper caching
        self._embedding_cache = lru_cache(maxsize=1000)(self._compute_embedding)
        
        self._batch_size = config.supabase.batch_size if config else 100
        
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute actual embeddings using centralized manager"""
        embedding = self.embedding_manager.embed(text)
        return np.array(embedding)
    
    async def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        # Use the LRU cached method
        return self._embedding_cache(text)
        
    @circuit_breaker("batch_insert", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60))
    async def batch_insert_embeddings(
        self, 
        documents: List[Document],
        batch_size: int = None
    ):
        """Batch insert for better performance with circuit breaker protection"""
        if batch_size is None:
            batch_size = self._batch_size
            
        async with self.pool.get_client() as client:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                batch_data = []
                for doc in batch:
                    embedding = await self._get_cached_embedding(doc.page_content)
                    batch_data.append({
                        "node_id": doc.metadata.get("id", str(hash(doc.page_content))),
                        "embedding": embedding.tolist(),
                        "text": doc.page_content,
                        "metadata_": doc.metadata
                    })
                
                # Use upsert for conflict resolution
                try:
                    result = await client.table("knowledge_base").upsert(batch_data).execute()
                    logger.info(f"Inserted {len(batch_data)} documents with circuit breaker protection")
                except Exception as e:
                    logger.error(f"Batch insert failed: {e}")
                    raise

class HybridVectorSearch:
    """Combine vector similarity with metadata filtering and BM25"""
    
    def __init__(self, pool: SupabaseConnectionPool):
        self.pool = pool
        # Use centralized embedding manager
        self.embedding_manager = get_embedding_manager()
        
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for query using centralized manager"""
        # FIXED: Use real embeddings instead of random
        embedding = self.embedding_manager.embed(text)
        return np.array(embedding)
        
    @circuit_breaker("vector_search", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60))
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Enhanced search with multiple ranking strategies and circuit breaker protection"""
        
        # 1. Vector similarity search
        query_embedding = await self.get_embedding(query)
        
        async with self.pool.get_client() as client:
            try:
                # Use RPC for complex queries
                results = await client.rpc(
                    'hybrid_match_documents',
                    {
                        'query_embedding': query_embedding.tolist(),
                        'match_count': top_k * 3,  # Get more for reranking
                        'metadata_filter': metadata_filter or {},
                        'query_text': query  # For BM25
                    }
                ).execute()
                
                # Convert to SearchResult objects
                search_results = []
                for result in results.data:
                    search_results.append(SearchResult(
                        content=result.get('text', ''),
                        metadata=result.get('metadata_', {}),
                        score=result.get('similarity', 0.0),
                        source=result.get('source', 'unknown')
                    ))
                
                if rerank:
                    search_results = await self._rerank_results(query, search_results)
                
                return search_results[:top_k]
                
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                # Fallback to simple vector search
                return await self._fallback_search(client, query_embedding, top_k)
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using additional signals"""
        # Simple reranking based on content length and metadata
        for result in results:
            # Boost results with more metadata
            metadata_boost = len(result.metadata) * 0.1
            result.score += metadata_boost
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    async def _fallback_search(self, client, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Fallback to simple vector similarity search"""
        try:
            # Simple vector similarity search
            results = await client.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding.tolist(),
                    'match_count': top_k
                }
            ).execute()
            
            search_results = []
            for result in results.data:
                search_results.append(SearchResult(
                    content=result.get('text', ''),
                    metadata=result.get('metadata_', {}),
                    score=result.get('similarity', 0.0),
                    source=result.get('source', 'unknown')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []

class SupabaseRealtimeManager:
    """Manage realtime subscriptions"""
    
    def __init__(self, client: Client):
        self.client = client
        self.subscriptions = {}
    
    async def subscribe_to_tool_metrics(self, callback):
        """Subscribe to tool execution metrics"""
        try:
            subscription = self.client.table('tool_metrics').on('INSERT', callback).subscribe()
            self.subscriptions['tool_metrics'] = subscription
            logger.info("Subscribed to tool metrics")
        except Exception as e:
            logger.error(f"Failed to subscribe to tool metrics: {e}")
    
    async def subscribe_to_knowledge_updates(self, callback):
        """Subscribe to knowledge base updates"""
        try:
            subscription = self.client.table('knowledge_base').on('INSERT', callback).subscribe()
            self.subscriptions['knowledge_updates'] = subscription
            logger.info("Subscribed to knowledge updates")
        except Exception as e:
            logger.error(f"Failed to subscribe to knowledge updates: {e}")
    
    async def unsubscribe_all(self):
        """Unsubscribe from all subscriptions"""
        for name, subscription in self.subscriptions.items():
            try:
                await subscription.unsubscribe()
                logger.info(f"Unsubscribed from {name}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {name}: {e}")
        self.subscriptions.clear()

@circuit_breaker("supabase_initialization", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60))
async def initialize_supabase_enhanced(url: Optional[str] = None, key: Optional[str] = None):
    """Initialize Supabase with circuit breaker protection"""
    try:
        # Use provided URL/key or fall back to config
        if url is None or key is None:
            if integration_config:
                url = url or integration_config.supabase_url
                key = key or integration_config.supabase_key
            else:
                url = url or os.getenv('SUPABASE_URL')
                key = key or os.getenv('SUPABASE_KEY')
        
        if not url or not key:
            raise ValueError("Supabase URL and key are required")
        
        # Create connection pool with circuit breaker protection
        pool = SupabaseConnectionPool(url, key)
        await pool.initialize()
        
        # Create optimized vector store
        vector_store = OptimizedVectorStore(pool)
        
        # Create hybrid search
        hybrid_search = HybridVectorSearch(pool)
        
        # Create realtime manager
        client = create_client(url, key)
        realtime_manager = SupabaseRealtimeManager(client)
        
        return {
            'client': client,
            'connection_pool': pool,
            'vector_store': vector_store,
            'hybrid_search': hybrid_search,
            'realtime_manager': realtime_manager
        }
        
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")
        raise

# Global instances for backward compatibility
vector_store = None
hybrid_search = None 