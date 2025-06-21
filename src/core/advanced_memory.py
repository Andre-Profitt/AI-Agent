"""
Advanced Memory System
Implements episodic, semantic, and procedural memory
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import asyncio
import pickle
import json
import hashlib
from enum import Enum
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    EPISODIC = "episodic"      # Specific experiences
    SEMANTIC = "semantic"      # General knowledge
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"        # Current context
    SENSORY = "sensory"        # Recent perceptions

@dataclass
class Memory:
    """Base memory unit"""
    id: str
    content: Any
    memory_type: MemoryType
    timestamp: datetime
    importance: float
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class EpisodicMemory(Memory):
    """Memory of specific events"""
    context: Dict[str, Any] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    
@dataclass
class SemanticMemory(Memory):
    """Factual knowledge"""
    category: str = ""
    confidence: float = 1.0
    source: Optional[str] = None
    verification_status: str = "unverified"
    
@dataclass
class ProceduralMemory(Memory):
    """Memory of how to perform tasks"""
    task_type: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    optimization_notes: List[str] = field(default_factory=list)

class AdvancedMemorySystem:
    """
    Sophisticated memory system with:
    - Multiple memory types
    - Associative retrieval
    - Memory consolidation
    - Forgetting curves
    - Memory networks
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memories = {
            MemoryType.EPISODIC: {},
            MemoryType.SEMANTIC: {},
            MemoryType.PROCEDURAL: {},
            MemoryType.WORKING: deque(maxlen=100),
            MemoryType.SENSORY: deque(maxlen=20)
        }
        
        self.memory_graph = nx.Graph()  # Association network
        self.importance_threshold = 0.3
        self.consolidation_queue = asyncio.Queue()
        self._start_background_tasks()
        
    def _start_background_tasks(self):
        """Start background memory processes"""
        asyncio.create_task(self._consolidation_loop())
        asyncio.create_task(self._forgetting_loop())
        asyncio.create_task(self._association_builder())
        
    async def store(
        self, 
        content: Any, 
        memory_type: MemoryType,
        importance: float = 0.5,
        **kwargs
    ) -> str:
        """Store new memory"""
        memory_id = self._generate_id(content)
        
        # Create appropriate memory type
        if memory_type == MemoryType.EPISODIC:
            memory = EpisodicMemory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.utcnow(),
                importance=importance,
                **kwargs
            )
        elif memory_type == MemoryType.SEMANTIC:
            memory = SemanticMemory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.utcnow(),
                importance=importance,
                **kwargs
            )
        elif memory_type == MemoryType.PROCEDURAL:
            memory = ProceduralMemory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.utcnow(),
                importance=importance,
                **kwargs
            )
        else:
            memory = Memory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.utcnow(),
                importance=importance,
                **kwargs
            )
            
        # Store in appropriate structure
        if memory_type in [MemoryType.WORKING, MemoryType.SENSORY]:
            self.memories[memory_type].append(memory)
        else:
            self.memories[memory_type][memory_id] = memory
            
        # Add to memory graph
        self.memory_graph.add_node(memory_id, memory=memory)
        
        # Build initial associations
        await self._build_associations(memory)
        
        # Check capacity
        if self._total_memories() > self.capacity:
            await self._manage_capacity()
            
        return memory_id
        
    async def retrieve(
        self, 
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        top_k: int = 5,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """Retrieve relevant memories"""
        if not memory_types:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
            
        candidates = []
        
        # Search each memory type
        for mem_type in memory_types:
            if mem_type in [MemoryType.WORKING, MemoryType.SENSORY]:
                # Search in deques
                for memory in self.memories[mem_type]:
                    if memory.importance >= min_importance:
                        relevance = self._calculate_relevance(query, memory)
                        if relevance > 0:
                            candidates.append((memory, relevance))
            else:
                # Search in dictionaries
                for memory in self.memories[mem_type].values():
                    if memory.importance >= min_importance:
                        relevance = self._calculate_relevance(query, memory)
                        if relevance > 0:
                            candidates.append((memory, relevance))
                            
        # Sort by relevance and recency
        candidates.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        
        # Get top k
        results = []
        for memory, relevance in candidates[:top_k]:
            # Update access patterns
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            
            # Activate associated memories
            associated = await self._activate_associations(memory.id)
            memory.metadata["activated_associations"] = associated
            
            results.append(memory)
            
        return results
        
    async def _build_associations(self, memory: Memory):
        """Build associations with existing memories"""
        # Find related memories
        related_memories = await self._find_related_memories(memory, top_k=10)
        
        for related_mem, similarity in related_memories:
            # Create edge in memory graph
            self.memory_graph.add_edge(
                memory.id, 
                related_mem.id,
                weight=similarity
            )
            
            # Update association lists
            memory.associations.append(related_mem.id)
            related_mem.associations.append(memory.id)
            
    async def _find_related_memories(
        self, 
        memory: Memory, 
        top_k: int = 10
    ) -> List[Tuple[Memory, float]]:
        """Find memories related to given memory"""
        related = []
        
        for mem_type in self.memories:
            if mem_type in [MemoryType.WORKING, MemoryType.SENSORY]:
                continue
                
            for other_memory in self.memories[mem_type].values():
                if other_memory.id != memory.id:
                    similarity = self._calculate_similarity(memory, other_memory)
                    if similarity > 0.3:
                        related.append((other_memory, similarity))
                        
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:top_k]
        
    def _calculate_relevance(self, query: str, memory: Memory) -> float:
        """Calculate relevance of memory to query"""
        # Simple implementation - would use embeddings in practice
        query_words = set(query.lower().split())
        memory_words = set(str(memory.content).lower().split())
        
        intersection = query_words & memory_words
        union = query_words | memory_words
        
        if not union:
            return 0.0
            
        jaccard = len(intersection) / len(union)
        
        # Factor in importance and recency
        recency_factor = self._calculate_recency_factor(memory.timestamp)
        importance_factor = memory.importance
        
        return jaccard * 0.5 + recency_factor * 0.3 + importance_factor * 0.2
        
    def _calculate_recency_factor(self, timestamp: datetime) -> float:
        """Calculate recency factor with decay"""
        age = datetime.utcnow() - timestamp
        days_old = age.total_seconds() / 86400
        
        # Exponential decay
        return np.exp(-days_old / 30)  # Half-life of 30 days
        
    async def consolidate(self):
        """Consolidate short-term to long-term memory"""
        # Move important working memories to episodic
        working_memories = list(self.memories[MemoryType.WORKING])
        
        for memory in working_memories:
            if memory.importance > 0.7:
                # Promote to episodic memory
                episodic_memory = EpisodicMemory(
                    id=memory.id,
                    content=memory.content,
                    memory_type=MemoryType.EPISODIC,
                    timestamp=memory.timestamp,
                    importance=memory.importance,
                    context=memory.metadata.get("context", {})
                )
                
                self.memories[MemoryType.EPISODIC][memory.id] = episodic_memory
                
        # Extract patterns from episodic memories
        patterns = await self._extract_patterns()
        
        # Create semantic memories from patterns
        for pattern in patterns:
            semantic_id = await self.store(
                content=pattern["knowledge"],
                memory_type=MemoryType.SEMANTIC,
                importance=pattern["confidence"],
                category=pattern["category"]
            )
            
    async def _extract_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from episodic memories"""
        patterns = []
        
        # Group similar episodic memories
        episodic_memories = list(self.memories[MemoryType.EPISODIC].values())
        
        # Cluster memories (simplified)
        clusters = self._cluster_memories(episodic_memories)
        
        for cluster in clusters:
            if len(cluster) >= 3:
                # Extract common pattern
                pattern = {
                    "knowledge": self._extract_common_knowledge(cluster),
                    "confidence": len(cluster) / 10.0,
                    "category": self._determine_category(cluster)
                }
                patterns.append(pattern)
                
        return patterns
        
    async def forget(self, decay_rate: float = 0.01):
        """Implement forgetting curve"""
        memories_to_forget = []
        
        for mem_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            for memory_id, memory in self.memories[mem_type].items():
                # Calculate retention probability
                retention_prob = self._calculate_retention(memory)
                
                if np.random.random() > retention_prob:
                    memories_to_forget.append((mem_type, memory_id))
                    
        # Remove forgotten memories
        for mem_type, memory_id in memories_to_forget:
            if memory_id in self.memories[mem_type]:
                del self.memories[mem_type][memory_id]
                
                # Remove from graph
                if self.memory_graph.has_node(memory_id):
                    self.memory_graph.remove_node(memory_id)
                    
    def _calculate_retention(self, memory: Memory) -> float:
        """Calculate probability of retaining memory"""
        # Ebbinghaus forgetting curve with modifications
        age = (datetime.utcnow() - memory.timestamp).total_seconds() / 86400
        
        # Base retention
        base_retention = np.exp(-age / 7)  # Weekly decay
        
        # Modifiers
        importance_modifier = memory.importance
        access_modifier = min(memory.access_count / 10, 1.0)
        association_modifier = min(len(memory.associations) / 5, 1.0)
        
        return base_retention * (0.4 + 0.2 * importance_modifier + 
                                0.2 * access_modifier + 0.2 * association_modifier)
                                
    async def _consolidation_loop(self):
        """Background consolidation process"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.consolidate()
            except Exception as e:
                logger.error(f"Consolidation error: {e}")
                
    async def _forgetting_loop(self):
        """Background forgetting process"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                await self.forget()
            except Exception as e:
                logger.error(f"Forgetting error: {e}")
                
    async def _association_builder(self):
        """Build associations between memories"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Strengthen existing associations
                for edge in self.memory_graph.edges():
                    node1, node2 = edge
                    if node1 in self.memory_graph and node2 in self.memory_graph:
                        # Check if both memories were recently accessed together
                        mem1 = self.memory_graph.nodes[node1]["memory"]
                        mem2 = self.memory_graph.nodes[node2]["memory"]
                        
                        time_diff = abs((mem1.last_accessed - mem2.last_accessed).total_seconds())
                        if time_diff < 60:  # Accessed within 1 minute
                            # Strengthen association
                            current_weight = self.memory_graph[node1][node2].get("weight", 0.5)
                            new_weight = min(current_weight * 1.1, 1.0)
                            self.memory_graph[node1][node2]["weight"] = new_weight
                            
            except Exception as e:
                logger.error(f"Association builder error: {e}")

class MemoryIndex:
    """Fast memory indexing and search"""
    
    def __init__(self):
        self.embedding_cache = {}
        self.index_structures = {
            "semantic": None,  # Would be FAISS, Annoy, etc.
            "temporal": None,
            "importance": None
        }
        
    async def build_index(self, memories: List[Memory]):
        """Build search indices"""
        # Build embedding index
        embeddings = []
        for memory in memories:
            embedding = await self._get_embedding(memory)
            embeddings.append(embedding)
            
        # Create vector index
        # self.index_structures["semantic"] = create_faiss_index(embeddings)
        
    async def search(
        self, 
        query: str, 
        filters: Dict[str, Any] = None
    ) -> List[Memory]:
        """Fast memory search"""
        query_embedding = await self._get_embedding(query)
        
        # Vector search
        # similar_indices = self.index_structures["semantic"].search(query_embedding)
        
        # Apply filters
        # filtered_results = self._apply_filters(similar_indices, filters)
        
        pass
