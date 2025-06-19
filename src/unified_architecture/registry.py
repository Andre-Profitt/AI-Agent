"""
Agent Registry for Multi-Agent System

This module provides dynamic agent registration and discovery:
- Agent registration and metadata management
- Capability-based discovery
- Status tracking and health monitoring
- Registry persistence and recovery
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

from .core import AgentMetadata, AgentStatus, AgentCapability

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Dynamic agent registration and discovery service"""
    
    def __init__(self):
        self.registry: Dict[str, AgentMetadata] = {}
        self.capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.discovery_cache: Dict[str, List[str]] = {}
        self.registry_lock = asyncio.Lock()
        
        # Health monitoring
        self.health_check_interval = 60  # seconds
        self.agent_heartbeats: Dict[str, float] = {}
        self.offline_threshold = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            "total_registrations": 0,
            "active_agents": 0,
            "discovery_queries": 0,
            "cache_hits": 0
        }
        
    async def register(self, metadata: AgentMetadata) -> bool:
        """Register an agent in the registry"""
        async with self.registry_lock:
            try:
                # Validate metadata
                if not metadata.agent_id or not metadata.name:
                    logger.error("Invalid agent metadata: missing agent_id or name")
                    return False
                
                # Check if agent already exists
                if metadata.agent_id in self.registry:
                    logger.warning("Agent {} already registered, updating metadata", extra={"metadata_agent_id": metadata.agent_id})
                    # Update existing registration
                    old_metadata = self.registry[metadata.agent_id]
                    
                    # Remove old indices
                    for capability in old_metadata.capabilities:
                        self.capability_index[capability].discard(metadata.agent_id)
                    for tag in old_metadata.tags:
                        self.tag_index[tag].discard(metadata.agent_id)
                else:
                    self.stats["total_registrations"] += 1
                
                # Add to registry
                self.registry[metadata.agent_id] = metadata
                
                # Update indices
                for capability in metadata.capabilities:
                    self.capability_index[capability].add(metadata.agent_id)
                
                for tag in metadata.tags:
                    self.tag_index[tag].add(metadata.agent_id)
                
                # Update heartbeat
                self.agent_heartbeats[metadata.agent_id] = time.time()
                
                # Clear discovery cache
                self.discovery_cache.clear()
                
                # Update status
                metadata.status = AgentStatus.AVAILABLE
                metadata.last_seen = datetime.utcnow()
                
                self.stats["active_agents"] = len(self.registry)
                
                logger.info("Registered agent {} ({}) "
                           f"with capabilities: {}", extra={"metadata_name": metadata.name, "metadata_agent_id": metadata.agent_id, "_cap_name_for_cap_in_metadata_capabilities": [cap.name for cap in metadata.capabilities]})
                return True
                
            except Exception as e:
                logger.error("Error registering agent {}: {}", extra={"metadata_agent_id": metadata.agent_id, "e": e})
                return False
    
    async def unregister(self, agent_id: str) -> bool:
        """Unregister an agent from the registry"""
        async with self.registry_lock:
            try:
                if agent_id not in self.registry:
                    logger.warning("Agent {} not found in registry", extra={"agent_id": agent_id})
                    return False
                
                metadata = self.registry[agent_id]
                
                # Remove from indices
                for capability in metadata.capabilities:
                    self.capability_index[capability].discard(agent_id)
                
                for tag in metadata.tags:
                    self.tag_index[tag].discard(agent_id)
                
                # Remove from registry
                del self.registry[agent_id]
                
                # Remove heartbeat
                if agent_id in self.agent_heartbeats:
                    del self.agent_heartbeats[agent_id]
                
                # Clear discovery cache
                self.discovery_cache.clear()
                
                self.stats["active_agents"] = len(self.registry)
                
                logger.info("Unregistered agent {}", extra={"agent_id": agent_id})
                return True
                
            except Exception as e:
                logger.error("Error unregistering agent {}: {}", extra={"agent_id": agent_id, "e": e})
                return False
    
    async def discover(self, 
                      capabilities: Optional[List[AgentCapability]] = None,
                      tags: Optional[List[str]] = None,
                      status: Optional[AgentStatus] = None,
                      min_reliability: float = 0.0,
                      max_load: Optional[int] = None) -> List[AgentMetadata]:
        """Discover agents matching criteria"""
        self.stats["discovery_queries"] += 1
        
        # Create cache key
        cache_key = f"{capabilities}:{tags}:{status}:{min_reliability}:{max_load}"
        
        # Check cache
        if cache_key in self.discovery_cache:
            self.stats["cache_hits"] += 1
            agent_ids = self.discovery_cache[cache_key]
            return [self.registry[aid] for aid in agent_ids if aid in self.registry]
        
        async with self.registry_lock:
            matching_agents = set(self.registry.keys())
            
            # Filter by capabilities
            if capabilities:
                capability_matches = set()
                for capability in capabilities:
                    capability_matches.update(self.capability_index.get(capability, set()))
                matching_agents &= capability_matches
            
            # Filter by tags
            if tags:
                tag_matches = set()
                for tag in tags:
                    tag_matches.update(self.tag_index.get(tag, set()))
                matching_agents &= tag_matches
            
            # Filter by status
            if status:
                matching_agents = {
                    aid for aid in matching_agents
                    if self.registry[aid].status == status
                }
            
            # Filter by reliability
            if min_reliability > 0:
                matching_agents = {
                    aid for aid in matching_agents
                    if self.registry[aid].reliability_score >= min_reliability
                }
            
            # Filter by load (if metadata contains load information)
            if max_load is not None:
                matching_agents = {
                    aid for aid in matching_agents
                    if self.registry[aid].performance_metrics.get("current_load", 0) <= max_load
                }
            
            # Cache results
            self.discovery_cache[cache_key] = list(matching_agents)
            
            return [self.registry[aid] for aid in matching_agents]
    
    async def update_status(self, agent_id: str, status: AgentStatus):
        """Update agent status"""
        async with self.registry_lock:
            if agent_id in self.registry:
                self.registry[agent_id].status = status
                self.registry[agent_id].last_seen = datetime.utcnow()
                
                # Update heartbeat
                self.agent_heartbeats[agent_id] = time.time()
                
                logger.debug("Updated status for agent {}: {}", extra={"agent_id": agent_id, "status_name": status.name})
    
    async def update_heartbeat(self, agent_id: str):
        """Update agent heartbeat"""
        async with self.registry_lock:
            if agent_id in self.registry:
                self.agent_heartbeats[agent_id] = time.time()
                self.registry[agent_id].last_seen = datetime.utcnow()
    
    async def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID"""
        async with self.registry_lock:
            return self.registry.get(agent_id)
    
    async def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentMetadata]:
        """Get all agents with a specific capability"""
        async with self.registry_lock:
            agent_ids = self.capability_index.get(capability, set())
            return [self.registry[aid] for aid in agent_ids if aid in self.registry]
    
    async def get_agents_by_tag(self, tag: str) -> List[AgentMetadata]:
        """Get all agents with a specific tag"""
        async with self.registry_lock:
            agent_ids = self.tag_index.get(tag, set())
            return [self.registry[aid] for aid in agent_ids if aid in self.registry]
    
    async def update_agent_metadata(self, agent_id: str, 
                                  updates: Dict[str, Any]) -> bool:
        """Update agent metadata"""
        async with self.registry_lock:
            if agent_id not in self.registry:
                return False
            
            metadata = self.registry[agent_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            # Update indices if capabilities or tags changed
            if "capabilities" in updates:
                # Remove old capabilities
                for capability in metadata.capabilities:
                    self.capability_index[capability].discard(agent_id)
                
                # Add new capabilities
                for capability in updates["capabilities"]:
                    self.capability_index[capability].add(agent_id)
            
            if "tags" in updates:
                # Remove old tags
                for tag in metadata.tags:
                    self.tag_index[tag].discard(agent_id)
                
                # Add new tags
                for tag in updates["tags"]:
                    self.tag_index[tag].add(agent_id)
            
            # Clear cache
            self.discovery_cache.clear()
            
            logger.debug("Updated metadata for agent {}", extra={"agent_id": agent_id})
            return True
    
    async def check_agent_health(self) -> Dict[str, Any]:
        """Check health of all registered agents"""
        current_time = time.time()
        health_results = {}
        
        async with self.registry_lock:
            for agent_id, metadata in self.registry.items():
                last_heartbeat = self.agent_heartbeats.get(agent_id, 0)
                time_since_heartbeat = current_time - last_heartbeat
                
                if time_since_heartbeat > self.offline_threshold:
                    # Mark agent as offline
                    metadata.status = AgentStatus.OFFLINE
                    health_results[agent_id] = {
                        "status": "offline",
                        "last_heartbeat": last_heartbeat,
                        "time_since_heartbeat": time_since_heartbeat
                    }
                else:
                    health_results[agent_id] = {
                        "status": "online",
                        "last_heartbeat": last_heartbeat,
                        "time_since_heartbeat": time_since_heartbeat
                    }
        
        return health_results
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        async with self.registry_lock:
            # Count agents by status
            status_counts = defaultdict(int)
            capability_counts = defaultdict(int)
            tag_counts = defaultdict(int)
            
            for metadata in self.registry.values():
                status_counts[metadata.status.name] += 1
                
                for capability in metadata.capabilities:
                    capability_counts[capability.name] += 1
                
                for tag in metadata.tags:
                    tag_counts[tag] += 1
            
            return {
                **self.stats,
                "agents_by_status": dict(status_counts),
                "agents_by_capability": dict(capability_counts),
                "popular_tags": dict(sorted(tag_counts.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]),
                "total_capabilities": len(self.capability_index),
                "total_tags": len(self.tag_index),
                "cache_size": len(self.discovery_cache)
            }
    
    async def clear_cache(self):
        """Clear the discovery cache"""
        async with self.registry_lock:
            self.discovery_cache.clear()
            logger.info("Cleared discovery cache")
    
    async def export_registry(self) -> Dict[str, Any]:
        """Export registry data for persistence"""
        async with self.registry_lock:
            return {
                "agents": {
                    aid: metadata.to_dict() 
                    for aid, metadata in self.registry.items()
                },
                "heartbeats": self.agent_heartbeats.copy(),
                "stats": self.stats.copy(),
                "exported_at": datetime.utcnow().isoformat()
            }
    
    async def import_registry(self, data: Dict[str, Any]) -> bool:
        """Import registry data from persistence"""
        try:
            async with self.registry_lock:
                # Clear existing data
                self.registry.clear()
                self.capability_index.clear()
                self.tag_index.clear()
                self.agent_heartbeats.clear()
                self.discovery_cache.clear()
                
                # Import agents
                for aid, agent_data in data.get("agents", {}).items():
                    metadata = AgentMetadata.from_dict(agent_data)
                    self.registry[aid] = metadata
                    
                    # Rebuild indices
                    for capability in metadata.capabilities:
                        self.capability_index[capability].add(aid)
                    
                    for tag in metadata.tags:
                        self.tag_index[tag].add(aid)
                
                # Import heartbeats
                self.agent_heartbeats.update(data.get("heartbeats", {}))
                
                # Import stats
                self.stats.update(data.get("stats", {}))
                
                logger.info("Imported registry with {} agents", extra={"len_self_registry_": len(self.registry)})
                return True
                
        except Exception as e:
            logger.error("Error importing registry: {}", extra={"e": e})
            return False
    
    async def cleanup_offline_agents(self, max_offline_time: float = 3600) -> int:
        """Remove agents that have been offline for too long"""
        current_time = time.time()
        agents_to_remove = []
        
        async with self.registry_lock:
            for agent_id, last_heartbeat in self.agent_heartbeats.items():
                if current_time - last_heartbeat > max_offline_time:
                    agents_to_remove.append(agent_id)
            
            for agent_id in agents_to_remove:
                await self.unregister(agent_id)
        
        if agents_to_remove:
            logger.info("Cleaned up {} offline agents", extra={"len_agents_to_remove_": len(agents_to_remove)})
        
        return len(agents_to_remove) 