"""
State Management for Multi-Agent System

This module provides unified state management:
- Distributed state storage with Redis
- Local state fallback
- State change notifications
- Checkpoint and recovery
"""

import asyncio
import time
import json
import pickle
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any, List, Union, Tuple

try:
    import aioredis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    msgpack = None

logger = logging.getLogger(__name__)

class StateManager:
    """Unified state management across different architectures"""
    
    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.local_state: Dict[str, Any] = {}
        self.state_history: deque = deque(maxlen=1000)
        self.redis_url = redis_url
        self.redis_client: Optional[Any] = None
        self.state_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.change_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.state_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.enable_redis = redis_url is not None and REDIS_AVAILABLE
        self.compression_enabled = True
        self.default_ttl = 86400  # 24 hours
        
    async def initialize(self) -> Any:
        """Initialize state manager with Redis connection"""
        if self.enable_redis:
            try:
                self.redis_client = await aioredis.create_redis_pool(self.redis_url)
                logger.info("Connected to Redis for distributed state management")
            except Exception as e:
                logger.error("Failed to connect to Redis: {}", extra={"e": e})
                self.enable_redis = False
        else:
            logger.info("Using local state management only")
    
    async def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value with fallback to local storage"""
        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    if self.compression_enabled:
                        return msgpack.unpackb(value, raw=False)
                    else:
                        return json.loads(value.decode())
            except Exception as e:
                logger.error("Redis get error: {}", extra={"e": e})
        
        # Fallback to local state
        return self.local_state.get(key, default)
    
    async def set_state(self, key: str, value: Any, ttl: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Set state value in both local and distributed storage"""
        async with self.state_locks[key]:
            # Update local state
            old_value = self.local_state.get(key)
            self.local_state[key] = value
            
            # Update metadata
            self.state_metadata[key] = {
                "last_updated": time.time(),
                "ttl": ttl or self.default_ttl,
                "size": len(str(value)),
                **(metadata or {})
            }
            
            # Record in history
            self.state_history.append({
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "timestamp": time.time(),
                "operation": "set"
            })
            
            # Update Redis
            if self.redis_client:
                try:
                    if self.compression_enabled:
                        packed_value = msgpack.packb(value, use_bin_type=True)
                    else:
                        packed_value = json.dumps(value).encode()
                    
                    if ttl:
                        await self.redis_client.setex(key, ttl, packed_value)
                    else:
                        await self.redis_client.set(key, packed_value)
                except Exception as e:
                    logger.error("Redis set error: {}", extra={"e": e})
            
            # Notify listeners
            await self._notify_listeners(key, old_value, value)
    
    async def delete_state(self, key: str) -> bool:
        """Delete state from storage"""
        async with self.state_locks[key]:
            # Remove from local state
            old_value = self.local_state.pop(key, None)
            
            # Remove metadata
            if key in self.state_metadata:
                del self.state_metadata[key]
            
            # Remove from Redis
            if self.redis_client:
                try:
                    await self.redis_client.delete(key)
                except Exception as e:
                    logger.error("Redis delete error: {}", extra={"e": e})
            
            # Record in history
            self.state_history.append({
                "key": key,
                "old_value": old_value,
                "new_value": None,
                "timestamp": time.time(),
                "operation": "delete"
            })
            
            # Notify listeners
            await self._notify_listeners(key, old_value, None)
    
    async def update_state(self, key: str, update_func: Callable[[Any], Any],
                          ttl: Optional[int] = None) -> bool:
        """Update state using a function"""
        async with self.state_locks[key]:
            current_value = await self.get_state(key)
            new_value = update_func(current_value)
            await self.set_state(key, new_value, ttl)
    
    def subscribe(self, key: str, callback: Callable) -> Any:
        """Subscribe to state changes"""
        self.change_listeners[key].append(callback)
        logger.debug("Added listener for key: {}", extra={"key": key})
    
    def unsubscribe(self, key: str, callback: Callable) -> Any:
        """Unsubscribe from state changes"""
        if key in self.change_listeners:
            try:
                self.change_listeners[key].remove(callback)
                logger.debug("Removed listener for key: {}", extra={"key": key})
            except ValueError:
                logger.warning("Callback not found for key: {}", extra={"key": key})
    
    async def _notify_listeners(self, key: str, old_value: Any, new_value: Any) -> Any:
        """Notify all listeners of state change"""
        if key not in self.change_listeners:
            return
        
        for callback in self.change_listeners[key]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(key, old_value, new_value)
                else:
                    callback(key, old_value, new_value)
            except Exception as e:
                logger.error("Listener error for key {}: {}", extra={"key": key, "e": e})
    
    async def create_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """Create a state checkpoint"""
        checkpoint_id = checkpoint_name or f"checkpoint_{int(time.time())}"
        
        checkpoint_data = {
            "id": checkpoint_id,
            "timestamp": time.time(),
            "state": self.local_state.copy(),
            "metadata": self.state_metadata.copy()
        }
        
        # Store checkpoint
        await self.set_state(f"checkpoint:{checkpoint_id}", checkpoint_data, ttl=604800)  # 7 days
        logger.info("Created checkpoint: {}", extra={"checkpoint_id": checkpoint_id})
        return checkpoint_id
    
    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint"""
        checkpoint_data = await self.get_state(f"checkpoint:{checkpoint_id}")
        
        if checkpoint_data:
            self.local_state = checkpoint_data["state"]
            self.state_metadata = checkpoint_data.get("metadata", {})
            logger.info("Restored checkpoint: {}", extra={"checkpoint_id": checkpoint_id})
            return True
        
        logger.error("Checkpoint not found: {}", extra={"checkpoint_id": checkpoint_id})
        return False
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoints = []
        
        if self.redis_client:
            try:
                # Get all checkpoint keys
                keys = await self.redis_client.keys("checkpoint:*")
                for key in keys:
                    checkpoint_data = await self.get_state(key.decode())
                    if checkpoint_data:
                        checkpoints.append({
                            "id": checkpoint_data["id"],
                            "timestamp": checkpoint_data["timestamp"],
                            "state_size": len(checkpoint_data["state"])
                        })
            except Exception as e:
                logger.error("Error listing checkpoints: {}", extra={"e": e})
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    async def get_state_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a state key"""
        value = await self.get_state(key)
        if value is None:
            return None
        
        metadata = self.state_metadata.get(key, {})
        
        return {
            "key": key,
            "value_type": type(value).__name__,
            "value_size": len(str(value)),
            "last_updated": metadata.get("last_updated"),
            "ttl": metadata.get("ttl"),
            "has_listeners": len(self.change_listeners.get(key, [])) > 0
        }
    
    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all state keys"""
        keys = list(self.local_state.keys())
        
        if self.redis_client and pattern:
            try:
                redis_keys = await self.redis_client.keys(pattern)
                keys.extend([k.decode() for k in redis_keys])
            except Exception as e:
                logger.error("Error listing Redis keys: {}", extra={"e": e})
        
        return sorted(set(keys))
    
    async def cleanup_expired(self) -> None:
        """Clean up expired state entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.state_metadata.items():
            if metadata.get("ttl"):
                last_updated = metadata.get("last_updated", 0)
                if current_time - last_updated > metadata["ttl"]:
                    expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete_state(key)
        
        if expired_keys:
            logger.info("Cleaned up {} expired state entries", extra={"len_expired_keys_": len(expired_keys)})
    
    async def get_state_stats(self) -> Dict[str, Any]:
        """Get statistics about state usage"""
        total_keys = len(self.local_state)
        total_size = sum(len(str(v)) for v in self.local_state.values())
        total_listeners = sum(len(listeners) for listeners in self.change_listeners.values())
        
        # Get Redis stats if available
        redis_stats = {}
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                redis_stats = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
            except Exception as e:
                logger.error("Error getting Redis stats: {}", extra={"e": e})
        
        return {
            "total_keys": total_keys,
            "total_size_bytes": total_size,
            "total_listeners": total_listeners,
            "history_size": len(self.state_history),
            "redis_enabled": self.enable_redis,
            "redis_stats": redis_stats,
            "compression_enabled": self.compression_enabled
        }
    
    async def shutdown(self) -> Any:
        """Shutdown the state manager"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
            logger.info("Redis connection closed") 