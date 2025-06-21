from agent import query
from examples.parallel_execution_example import results
from performance_dashboard import stats
from setup_environment import value
from tests.load_test import data

from src.application.tools.tool_executor import operation
from src.core.monitoring import key
from src.core.optimized_chain_of_thought import similarity
from src.database.models import agent_id
from src.database.models import memory_type
from src.database.models import metadata
from src.gaia_components.performance_optimization import entry
from src.gaia_components.performance_optimization import sorted_keys
from src.gaia_components.production_vector_store import similarities
from src.services.integration_hub import limit
from src.unified_architecture.shared_memory import access_counts
from src.unified_architecture.shared_memory import candidate_vectors
from src.unified_architecture.shared_memory import candidates
from src.unified_architecture.shared_memory import ctx_tags
from src.unified_architecture.shared_memory import ctx_type
from src.unified_architecture.shared_memory import entries
from src.unified_architecture.shared_memory import exp_tags
from src.unified_architecture.shared_memory import exp_type
from src.unified_architecture.shared_memory import experience_data
from src.unified_architecture.shared_memory import experiences
from src.unified_architecture.shared_memory import expired_keys
from src.unified_architecture.shared_memory import export_data
from src.unified_architecture.shared_memory import filtered_results
from src.unified_architecture.shared_memory import imported_count
from src.unified_architecture.shared_memory import keys_to_remove
from src.unified_architecture.shared_memory import num_to_remove
from src.unified_architecture.shared_memory import recent_access
from src.utils.tools_introspection import field

from src.agents.advanced_agent_fsm import Agent

from src.gaia_components.enhanced_memory_system import MemoryType

from src.gaia_components.enhanced_memory_system import MemoryEntry
from collections import deque
from dataclasses import field
from enum import auto
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
# TODO: Fix undefined variables: Any, Dict, Enum, List, Optional, Tuple, access_counts, agent_id, auto, c, candidate_vectors, candidates, cls, context, ctx_tags, ctx_type, data, dataclass, defaultdict, deque, e, entries, entry, entry_data, exp_tags, exp_type, experience, experience_data, experiences, expired_keys, export_data, field, filtered_results, i, imported_count, key, keys_to_remove, limit, logging, memory_type, metadata, min_similarity, num_to_remove, operation, query, query_vector, recent_access, results, similarities, similarity, sorted_keys, stats, tag, tags, time, top_k, ttl, uuid, value, vector, vector_dim, x
# TODO: Fix undefined variables: access_counts, agent_id, c, candidate_vectors, candidates, cls, context, ctx_tags, ctx_type, data, e, entries, entry, entry_data, exp_tags, exp_type, experience, experience_data, experiences, expired_keys, export_data, filtered_results, i, imported_count, key, keys_to_remove, limit, memory_type, metadata, min_similarity, num_to_remove, operation, query, query_vector, recent_access, results, self, similarities, similarity, sorted_keys, stats, tag, tags, top_k, ttl, value, vector, vector_dim, x

"""
Shared Memory System for Multi-Agent System

This module provides distributed knowledge base for agents:
- Memory storage and retrieval
- Semantic search capabilities
- Experience sharing between agents
- Memory lifecycle management
"""

import asyncio
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    cosine_similarity = None

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of shared memory"""
    KNOWLEDGE = auto()
    EXPERIENCE = auto()
    CONTEXT = auto()
    CACHE = auto()
    MODEL = auto()
    CONFIGURATION = auto()

@dataclass
class MemoryEntry:
    """Entry in shared memory"""
    key: str
    value: Any
    memory_type: MemoryType
    created_by: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None
    
    def is_expired(self) -> bool:
        """Check if memory entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type.name,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "ttl": self.ttl,
            "tags": self.tags,
            "metadata": self.metadata,
            "vector": self.vector.tolist() if self.vector is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data = data.copy()
        data["memory_type"] = MemoryType[data["memory_type"]]
        if data.get("vector"):
            data["vector"] = np.array(data["vector"])
        return cls(**data)

class SharedMemorySystem:
    """Distributed knowledge base for agents"""
    
    def __init__(self, vector_dim: int = 768):
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.vector_index: Dict[str, np.ndarray] = {}  # For semantic search
        self.access_log: deque = deque(maxlen=10000)
        self.memory_lock = asyncio.Lock()
        self.vector_dim = vector_dim
        self.cleanup_interval = 3600  # 1 hour
        self.max_memory_entries = 10000
        
        # Statistics
        self.stats = {
            "total_stores": 0,
            "total_retrieves": 0,
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Check if similarity computation is available
        if not SIMILARITY_AVAILABLE:
            logger.warning("scikit-learn not available, semantic search disabled")
    
    async def store(self, key: str, value: Any, memory_type: MemoryType,
                   agent_id: str, tags: Optional[List[str]] = None,
                   ttl: Optional[float] = None, vector: Optional[np.ndarray] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """Store information in shared memory"""
        async with self.memory_lock:
            try:
                # Check if we need to evict old entries
                if len(self.memory_store) >= self.max_memory_entries:
                    await self._evict_old_entries()
                
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    memory_type=memory_type,
                    created_by=agent_id,
                    tags=tags or [],
                    ttl=ttl,
                    metadata=metadata or {}
                )
                
                self.memory_store[key] = entry
                
                # Store vector representation if provided
                if vector is not None and vector.shape[0] == self.vector_dim:
                    entry.vector = vector
                    self.vector_index[key] = vector
                
                # Log access
                self._log_access(agent_id, key, "store")
                
                self.stats["total_stores"] += 1
                
                logger.debug("Agent {} stored key: {} (type: {})", extra={"agent_id": agent_id, "key": key, "memory_type_name": memory_type.name})
                
            except Exception as e:
                logger.error("Error storing memory entry {}: {}", extra={"key": key, "e": e})
    
    async def retrieve(self, key: str, agent_id: str) -> Optional[Any]:
        """Retrieve information from shared memory"""
        async with self.memory_lock:
            if key in self.memory_store:
                entry = self.memory_store[key]
                
                # Check TTL
                if entry.is_expired():
                    await self.delete(key)
                    self.stats["cache_misses"] += 1
                    return None
                
                # Update access info
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                # Log access
                self._log_access(agent_id, key, "retrieve")
                
                self.stats["total_retrieves"] += 1
                self.stats["cache_hits"] += 1
                
                return entry.value
            else:
                self.stats["cache_misses"] += 1
                return None
    
    async def search(self, query_vector: np.ndarray, top_k: int = 5,
                    memory_type: Optional[MemoryType] = None,
                    tags: Optional[List[str]] = None,
                    min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """Semantic search in shared memory"""
        if not SIMILARITY_AVAILABLE:
            logger.warning("Semantic search not available (scikit-learn required)")
            return []
        
        if not self.vector_index:
            return []
        
        self.stats["total_searches"] += 1
        
        # Filter candidates
        candidates = []
        for key, vector in self.vector_index.items():
            entry = self.memory_store.get(key)
            if not entry:
                continue
            
            # Check if expired
            if entry.is_expired():
                continue
            
            # Apply filters
            if memory_type and entry.memory_type != memory_type:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            candidates.append((key, vector))
        
        if not candidates:
            return []
        
        # Calculate similarities
        candidate_vectors = np.array([c[1] for c in candidates])
        similarities = cosine_similarity([query_vector], candidate_vectors)[0]
        
        # Filter by minimum similarity
        filtered_results = []
        for i, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                filtered_results.append((candidates[i][0], similarity))
        
        # Get top-k results
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:top_k]
    
    async def search_by_text(self, query: str, top_k: int = 5,
                           memory_type: Optional[MemoryType] = None,
                           tags: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Search by text (simplified - would use embeddings in practice)"""
        # This is a simplified text search - in practice would use embeddings
        results = []
        
        for key, entry in self.memory_store.items():
            if entry.is_expired():
                continue
            
            # Apply filters
            if memory_type and entry.memory_type != memory_type:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Simple text matching
            if isinstance(entry.value, str) and query.lower() in entry.value.lower():
                results.append((key, 0.8))  # Fixed similarity score
            elif isinstance(entry.value, dict):
                # Search in dictionary values
                for value in entry.value.values():
                    if isinstance(value, str) and query.lower() in value.lower():
                        results.append((key, 0.7))
                        break
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def share_experience(self, experience: Dict[str, Any], agent_id: str,
                             tags: Optional[List[str]] = None):
        """Share learning experience between agents"""
        key = f"experience:{agent_id}:{uuid.uuid4().hex[:8]}"
        
        # Add metadata
        experience_data = {
            "experience": experience,
            "shared_at": time.time(),
            "agent_id": agent_id
        }
        
        await self.store(
            key=key,
            value=experience_data,
            memory_type=MemoryType.EXPERIENCE,
            agent_id=agent_id,
            tags=tags or ["experience", experience.get("task_type", "unknown")],
            ttl=86400 * 7  # Keep for 7 days
        )
        
        logger.info("Agent {} shared experience: {}", extra={"agent_id": agent_id, "experience_get__task_type____unknown__": experience.get('task_type', 'unknown')})
    
    async def get_relevant_experiences(self, context: Dict[str, Any],
                                     limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant experiences for a given context"""
        experiences = []
        
        for key, entry in self.memory_store.items():
            if entry.memory_type == MemoryType.EXPERIENCE and not entry.is_expired():
                if self._is_relevant_experience(entry.value, context):
                    experiences.append(entry.value)
                    
                if len(experiences) >= limit:
                    break
        
        return experiences
    
    def _is_relevant_experience(self, experience: Dict[str, Any],
                               context: Dict[str, Any]) -> bool:
        """Check if an experience is relevant to current context"""
        # Simplified relevance check
        exp_type = experience.get("experience", {}).get("task_type")
        ctx_type = context.get("task_type")
        
        if exp_type and ctx_type:
            return exp_type == ctx_type
        
        # Check for tag overlap
        exp_tags = experience.get("experience", {}).get("tags", [])
        ctx_tags = context.get("tags", [])
        
        if exp_tags and ctx_tags:
            return any(tag in exp_tags for tag in ctx_tags)
        
        return False
    
    async def get_memory_stats(self, memory_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        stats = {
            "total_entries": len(self.memory_store),
            "vector_entries": len(self.vector_index),
            "memory_types": defaultdict(int),
            "popular_tags": defaultdict(int),
            "access_patterns": {},
            **self.stats
        }
        
        # Count by memory type
        for entry in self.memory_store.values():
            if not entry.is_expired():
                stats["memory_types"][entry.memory_type.name] += 1
                
                for tag in entry.tags:
                    stats["popular_tags"][tag] += 1
        
        # Filter by memory type if specified
        if memory_type:
            stats["total_entries"] = stats["memory_types"].get(memory_type.name, 0)
        
        # Get access patterns
        if self.access_log:
            recent_access = list(self.access_log)[-100:]  # Last 100 accesses
            stats["access_patterns"] = {
                "recent_operations": len(recent_access),
                "most_accessed_keys": self._get_most_accessed_keys(10)
            }
        
        return stats
    
    def _get_most_accessed_keys(self, limit: int) -> List[Tuple[str, int]]:
        """Get most frequently accessed keys"""
        access_counts = defaultdict(int)
        
        for entry in self.memory_store.values():
            if not entry.is_expired():
                access_counts[entry.key] = entry.access_count
        
        sorted_keys = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_keys[:limit]
    
    def _log_access(self, agent_id: str, key: str, operation: str):
        """Log memory access"""
        self.access_log.append({
            "agent_id": agent_id,
            "key": key,
            "operation": operation,
            "timestamp": time.time()
        })
    
    async def delete(self, key: str):
        """Delete a memory entry"""
        async with self.memory_lock:
            if key in self.memory_store:
                del self.memory_store[key]
                
            if key in self.vector_index:
                del self.vector_index[key]
            
            logger.debug("Deleted memory entry: {}", extra={"key": key})
    
    async def _evict_old_entries(self):
        """Evict old entries when memory is full"""
        # Sort by last accessed time (oldest first)
        entries = [(key, entry) for key, entry in self.memory_store.items()]
        entries.sort(key=lambda x: x[1].last_accessed)
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(entries) // 10)
        
        for i in range(num_to_remove):
            key, entry = entries[i]
            await self.delete(key)
        
        logger.info("Evicted {} old memory entries", extra={"num_to_remove": num_to_remove})
    
    async def cleanup_expired(self):
        """Clean up expired memory entries"""
        async with self.memory_lock:
            expired_keys = []
            
            for key, entry in self.memory_store.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self.delete(key)
            
            if expired_keys:
                logger.info("Cleaned up {} expired memory entries", extra={"len_expired_keys_": len(expired_keys)})
    
    async def export_memory(self, memory_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """Export memory data"""
        export_data = {
            "entries": {},
            "metadata": {
                "exported_at": time.time(),
                "total_entries": 0,
                "memory_type_filter": memory_type.name if memory_type else None
            }
        }
        
        for key, entry in self.memory_store.items():
            if not entry.is_expired():
                if memory_type is None or entry.memory_type == memory_type:
                    export_data["entries"][key] = entry.to_dict()
                    export_data["metadata"]["total_entries"] += 1
        
        return export_data
    
    async def import_memory(self, data: Dict[str, Any]) -> int:
        """Import memory data"""
        imported_count = 0
        
        for key, entry_data in data.get("entries", {}).items():
            try:
                entry = MemoryEntry.from_dict(entry_data)
                self.memory_store[key] = entry
                
                if entry.vector is not None:
                    self.vector_index[key] = entry.vector
                
                imported_count += 1
            except Exception as e:
                logger.error("Error importing memory entry {}: {}", extra={"key": key, "e": e})
        
        logger.info("Imported {} memory entries", extra={"imported_count": imported_count})
        return imported_count
    
    async def clear_memory(self, memory_type: Optional[MemoryType] = None):
        """Clear all memory or memory of specific type"""
        async with self.memory_lock:
            if memory_type:
                keys_to_remove = [
                    key for key, entry in self.memory_store.items()
                    if entry.memory_type == memory_type
                ]
            else:
                keys_to_remove = list(self.memory_store.keys())
            
            for key in keys_to_remove:
                await self.delete(key)
            
            logger.info(f"Cleared {len(keys_to_remove)} memory entries "
                       f"({'all' if memory_type is None else memory_type.name})") 