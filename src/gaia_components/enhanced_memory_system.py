"""
Enhanced Memory System for GAIA-enhanced FSMReActAgent
Implements sophisticated memory management with episodic, semantic, and working memory
"""

import logging
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"      # Event-based memories
    SEMANTIC = "semantic"      # Fact-based knowledge
    WORKING = "working"        # Short-term processing
    PROCEDURAL = "procedural"  # Skill-based memories
    EMOTIONAL = "emotional"    # Emotion-linked memories

class MemoryPriority(str, Enum):
    """Memory priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ARCHIVE = "archive"

@dataclass
class MemoryItem:
    """Represents a single memory item"""
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    decay_rate: float = 0.1  # Rate at which memory decays
    associations: Set[str] = field(default_factory=set)  # Associated memory IDs
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    def access(self):
        """Mark memory as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_strength(self) -> float:
        """Calculate memory strength based on access patterns and time"""
        time_factor = np.exp(-self.decay_rate * 
                           (datetime.now() - self.timestamp).total_seconds() / 3600)
        access_factor = min(self.access_count / 10.0, 1.0)  # Normalize access count
        return (time_factor + access_factor) / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "decay_rate": self.decay_rate,
            "associations": list(self.associations)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        item = cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            priority=MemoryPriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            decay_rate=data.get("decay_rate", 0.1)
        )
        if data.get("last_accessed"):
            item.last_accessed = datetime.fromisoformat(data["last_accessed"])
        item.associations = set(data.get("associations", []))
        return item

class EpisodicMemory:
    """Stores event-based memories with temporal context"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.memories: Dict[str, MemoryItem] = {}
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # Date -> memory IDs
        self.event_index: Dict[str, List[str]] = defaultdict(list)     # Event type -> memory IDs
    
    def store(self, content: str, event_type: str, metadata: Dict[str, Any] = None) -> str:
        """Store an episodic memory"""
        memory_id = self._generate_id(content)
        
        # Create memory item
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.MEDIUM,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in memory
        self.memories[memory_id] = memory
        
        # Update indices
        date_key = memory.timestamp.strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(memory_id)
        self.event_index[event_type].append(memory_id)
        
        # Maintain size limit
        self._enforce_size_limit()
        
        return memory_id
    
    def retrieve_by_time(self, start_date: datetime, end_date: datetime) -> List[MemoryItem]:
        """Retrieve memories within a time range"""
        memories = []
        current_date = start_date.date()
        end_date_obj = end_date.date()
        
        while current_date <= end_date_obj:
            date_key = current_date.strftime("%Y-%m-%d")
            for memory_id in self.temporal_index.get(date_key, []):
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    if start_date <= memory.timestamp <= end_date:
                        memory.access()
                        memories.append(memory)
            current_date += timedelta(days=1)
        
        return sorted(memories, key=lambda m: m.timestamp)
    
    def retrieve_by_event_type(self, event_type: str) -> List[MemoryItem]:
        """Retrieve memories by event type"""
        memories = []
        for memory_id in self.event_index.get(event_type, []):
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.access()
                memories.append(memory)
        
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
    
    def _enforce_size_limit(self):
        """Enforce maximum memory size"""
        if len(self.memories) <= self.max_size:
            return
        
        # Remove oldest, least accessed memories
        memories_list = list(self.memories.values())
        memories_list.sort(key=lambda m: (m.access_count, m.timestamp))
        
        # Remove excess memories
        excess_count = len(self.memories) - self.max_size
        for memory in memories_list[:excess_count]:
            self._remove_memory(memory.id)
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory and update indices"""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # Remove from temporal index
        date_key = memory.timestamp.strftime("%Y-%m-%d")
        if memory_id in self.temporal_index[date_key]:
            self.temporal_index[date_key].remove(memory_id)
        
        # Remove from event index
        for event_type, memory_ids in self.event_index.items():
            if memory_id in memory_ids:
                memory_ids.remove(memory_id)
        
        # Remove from main storage
        del self.memories[memory_id]

class SemanticMemory:
    """Stores fact-based knowledge with semantic relationships"""
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.memories: Dict[str, MemoryItem] = {}
        self.concept_index: Dict[str, List[str]] = defaultdict(list)  # Concept -> memory IDs
        self.relationship_graph: Dict[str, Set[str]] = defaultdict(set)  # Memory ID -> related IDs
    
    def store(self, content: str, concepts: List[str], metadata: Dict[str, Any] = None) -> str:
        """Store a semantic memory"""
        memory_id = self._generate_id(content)
        
        # Create memory item
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.HIGH,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in memory
        self.memories[memory_id] = memory
        
        # Update concept index
        for concept in concepts:
            self.concept_index[concept.lower()].append(memory_id)
        
        # Maintain size limit
        self._enforce_size_limit()
        
        return memory_id
    
    def retrieve_by_concept(self, concept: str) -> List[MemoryItem]:
        """Retrieve memories related to a concept"""
        memories = []
        concept_lower = concept.lower()
        
        for memory_id in self.concept_index.get(concept_lower, []):
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.access()
                memories.append(memory)
        
        return sorted(memories, key=lambda m: m.calculate_strength(), reverse=True)
    
    def retrieve_related(self, memory_id: str, max_related: int = 5) -> List[MemoryItem]:
        """Retrieve memories related to a specific memory"""
        if memory_id not in self.memories:
            return []
        
        related_ids = self.relationship_graph.get(memory_id, set())
        memories = []
        
        for related_id in list(related_ids)[:max_related]:
            if related_id in self.memories:
                memory = self.memories[related_id]
                memory.access()
                memories.append(memory)
        
        return sorted(memories, key=lambda m: m.calculate_strength(), reverse=True)
    
    def add_relationship(self, memory_id1: str, memory_id2: str):
        """Add a relationship between two memories"""
        if memory_id1 in self.memories and memory_id2 in self.memories:
            self.relationship_graph[memory_id1].add(memory_id2)
            self.relationship_graph[memory_id2].add(memory_id1)
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
    
    def _enforce_size_limit(self):
        """Enforce maximum memory size"""
        if len(self.memories) <= self.max_size:
            return
        
        # Remove weakest memories
        memories_list = list(self.memories.values())
        memories_list.sort(key=lambda m: m.calculate_strength())
        
        # Remove excess memories
        excess_count = len(self.memories) - self.max_size
        for memory in memories_list[:excess_count]:
            self._remove_memory(memory.id)
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory and update indices"""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # Remove from concept index
        for concept, memory_ids in self.concept_index.items():
            if memory_id in memory_ids:
                memory_ids.remove(memory_id)
        
        # Remove from relationship graph
        if memory_id in self.relationship_graph:
            del self.relationship_graph[memory_id]
        
        # Remove from main storage
        del self.memories[memory_id]

class WorkingMemory:
    """Short-term memory for active processing"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.memories: Dict[str, MemoryItem] = {}
        self.access_queue = deque()  # Track access order
    
    def store(self, content: str, priority: MemoryPriority = MemoryPriority.MEDIUM, 
              metadata: Dict[str, Any] = None) -> str:
        """Store an item in working memory"""
        memory_id = self._generate_id(content)
        
        # Create memory item
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=MemoryType.WORKING,
            priority=priority,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in memory
        self.memories[memory_id] = memory
        self.access_queue.append(memory_id)
        
        # Maintain size limit
        self._enforce_size_limit()
        
        return memory_id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access()
            
            # Move to end of access queue
            if memory_id in self.access_queue:
                self.access_queue.remove(memory_id)
            self.access_queue.append(memory_id)
            
            return memory
        return None
    
    def get_all(self) -> List[MemoryItem]:
        """Get all working memories"""
        memories = list(self.memories.values())
        return sorted(memories, key=lambda m: m.priority.value, reverse=True)
    
    def clear(self):
        """Clear all working memories"""
        self.memories.clear()
        self.access_queue.clear()
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
    
    def _enforce_size_limit(self):
        """Enforce maximum memory size using LRU eviction"""
        while len(self.memories) > self.max_size and self.access_queue:
            # Remove least recently used memory
            lru_id = self.access_queue.popleft()
            if lru_id in self.memories:
                del self.memories[lru_id]

class EnhancedMemorySystem:
    """Enhanced memory system integrating all memory types"""
    
    def __init__(self, persist_directory: str = "./memory_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize memory subsystems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.working_memory = WorkingMemory()
        
        # Memory consolidation settings
        self.consolidation_threshold = 0.7  # Strength threshold for consolidation
        self.consolidation_interval = 3600  # Consolidation interval in seconds
        self.last_consolidation = datetime.now()
        
        # Load existing memories
        self._load_memories()
        
        logger.info("Enhanced Memory System initialized")
    
    def store_episodic(self, content: str, event_type: str, metadata: Dict[str, Any] = None) -> str:
        """Store an episodic memory"""
        return self.episodic_memory.store(content, event_type, metadata)
    
    def store_semantic(self, content: str, concepts: List[str], metadata: Dict[str, Any] = None) -> str:
        """Store a semantic memory"""
        return self.semantic_memory.store(content, concepts, metadata)
    
    def store_working(self, content: str, priority: MemoryPriority = MemoryPriority.MEDIUM, 
                     metadata: Dict[str, Any] = None) -> str:
        """Store an item in working memory"""
        return self.working_memory.store(content, priority, metadata)
    
    def retrieve_episodic_by_time(self, start_date: datetime, end_date: datetime) -> List[MemoryItem]:
        """Retrieve episodic memories by time range"""
        return self.episodic_memory.retrieve_by_time(start_date, end_date)
    
    def retrieve_episodic_by_event(self, event_type: str) -> List[MemoryItem]:
        """Retrieve episodic memories by event type"""
        return self.episodic_memory.retrieve_by_event_type(event_type)
    
    def retrieve_semantic_by_concept(self, concept: str) -> List[MemoryItem]:
        """Retrieve semantic memories by concept"""
        return self.semantic_memory.retrieve_by_concept(concept)
    
    def retrieve_working(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a working memory"""
        return self.working_memory.retrieve(memory_id)
    
    def search_memories(self, query: str, memory_types: List[MemoryType] = None) -> List[MemoryItem]:
        """Search across all memory types"""
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.WORKING]
        
        results = []
        query_lower = query.lower()
        
        # Search episodic memories
        if MemoryType.EPISODIC in memory_types:
            for memory in self.episodic_memory.memories.values():
                if query_lower in memory.content.lower():
                    memory.access()
                    results.append(memory)
        
        # Search semantic memories
        if MemoryType.SEMANTIC in memory_types:
            for memory in self.semantic_memory.memories.values():
                if query_lower in memory.content.lower():
                    memory.access()
                    results.append(memory)
        
        # Search working memories
        if MemoryType.WORKING in memory_types:
            for memory in self.working_memory.memories.values():
                if query_lower in memory.content.lower():
                    memory.access()
                    results.append(memory)
        
        # Sort by relevance (strength and recency)
        results.sort(key=lambda m: (m.calculate_strength(), m.timestamp), reverse=True)
        
        return results
    
    def consolidate_memories(self):
        """Consolidate memories from working to long-term storage"""
        current_time = datetime.now()
        if (current_time - self.last_consolidation).total_seconds() < self.consolidation_interval:
            return
        
        # Consolidate strong working memories to semantic memory
        working_memories = self.working_memory.get_all()
        for memory in working_memories:
            if memory.calculate_strength() >= self.consolidation_threshold:
                # Extract concepts from content (simple keyword extraction)
                concepts = self._extract_concepts(memory.content)
                
                # Store in semantic memory
                self.semantic_memory.store(
                    content=memory.content,
                    concepts=concepts,
                    metadata=memory.metadata
                )
                
                # Remove from working memory
                self.working_memory.memories.pop(memory.id, None)
        
        self.last_consolidation = current_time
        logger.info("Memory consolidation completed")
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract concepts from content (simple implementation)"""
        # Simple keyword extraction - in a real implementation, this would use NLP
        words = content.lower().split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(concepts))[:10]  # Limit to 10 concepts
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            "episodic_count": len(self.episodic_memory.memories),
            "semantic_count": len(self.semantic_memory.memories),
            "working_count": len(self.working_memory.memories),
            "total_memories": (
                len(self.episodic_memory.memories) +
                len(self.semantic_memory.memories) +
                len(self.working_memory.memories)
            ),
            "episodic_events": len(self.episodic_memory.event_index),
            "semantic_concepts": len(self.semantic_memory.concept_index),
            "working_priority_distribution": self._get_priority_distribution()
        }
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of priorities in working memory"""
        distribution = defaultdict(int)
        for memory in self.working_memory.memories.values():
            distribution[memory.priority.value] += 1
        return dict(distribution)
    
    def _load_memories(self):
        """Load memories from persistent storage"""
        try:
            # Load episodic memories
            episodic_file = self.persist_directory / "episodic_memories.pkl"
            if episodic_file.exists():
                with open(episodic_file, 'rb') as f:
                    data = pickle.load(f)
                    self.episodic_memory.memories = {
                        k: MemoryItem.from_dict(v) for k, v in data.get('memories', {}).items()
                    }
                    self.episodic_memory.temporal_index = data.get('temporal_index', defaultdict(list))
                    self.episodic_memory.event_index = data.get('event_index', defaultdict(list))
            
            # Load semantic memories
            semantic_file = self.persist_directory / "semantic_memories.pkl"
            if semantic_file.exists():
                with open(semantic_file, 'rb') as f:
                    data = pickle.load(f)
                    self.semantic_memory.memories = {
                        k: MemoryItem.from_dict(v) for k, v in data.get('memories', {}).items()
                    }
                    self.semantic_memory.concept_index = data.get('concept_index', defaultdict(list))
                    self.semantic_memory.relationship_graph = data.get('relationship_graph', defaultdict(set))
            
            logger.info("Memories loaded from persistent storage")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def save_memories(self):
        """Save memories to persistent storage"""
        try:
            # Save episodic memories
            episodic_data = {
                'memories': {k: v.to_dict() for k, v in self.episodic_memory.memories.items()},
                'temporal_index': dict(self.episodic_memory.temporal_index),
                'event_index': dict(self.episodic_memory.event_index)
            }
            with open(self.persist_directory / "episodic_memories.pkl", 'wb') as f:
                pickle.dump(episodic_data, f)
            
            # Save semantic memories
            semantic_data = {
                'memories': {k: v.to_dict() for k, v in self.semantic_memory.memories.items()},
                'concept_index': dict(self.semantic_memory.concept_index),
                'relationship_graph': dict(self.semantic_memory.relationship_graph)
            }
            with open(self.persist_directory / "semantic_memories.pkl", 'wb') as f:
                pickle.dump(semantic_data, f)
            
            logger.info("Memories saved to persistent storage")
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def cleanup_old_memories(self, days_to_keep: int = 30):
        """Clean up old memories"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up episodic memories
        old_episodic = []
        for memory in self.episodic_memory.memories.values():
            if memory.timestamp < cutoff_date and memory.access_count < 5:
                old_episodic.append(memory.id)
        
        for memory_id in old_episodic:
            self.episodic_memory._remove_memory(memory_id)
        
        # Clean up weak semantic memories
        weak_semantic = []
        for memory in self.semantic_memory.memories.values():
            if memory.calculate_strength() < 0.1:
                weak_semantic.append(memory.id)
        
        for memory_id in weak_semantic:
            self.semantic_memory._remove_memory(memory_id)
        
        logger.info(f"Cleaned up {len(old_episodic)} old episodic and {len(weak_semantic)} weak semantic memories")