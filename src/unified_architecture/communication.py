"""
Communication Protocol for Multi-Agent System

This module provides inter-agent messaging system:
- Message types and routing
- Broadcast and topic-based messaging
- Request-response patterns
- Message queuing and delivery
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = auto()
    RESPONSE = auto()
    BROADCAST = auto()
    HEARTBEAT = auto()
    NEGOTIATION = auto()
    COORDINATION = auto()
    KNOWLEDGE_SHARE = auto()
    STATUS_UPDATE = auto()
    ERROR = auto()
    COMMAND = auto()

@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Any
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    correlation_id: Optional[str] = None
    priority: int = 5  # 1-10, higher is more important
    ttl: Optional[float] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message after initialization"""
        if self.priority < 1 or self.priority > 10:
            raise ValueError("Priority must be between 1 and 10")
        
        if self.ttl is not None and self.ttl <= 0:
            raise ValueError("TTL must be positive")
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl": self.ttl,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create from dictionary"""
        data = data.copy()
        data["message_type"] = MessageType[data["message_type"]]
        return cls(**data)

class CommunicationProtocol:
    """Inter-agent messaging system with various communication patterns"""
    
    def __init__(self):
        self.message_queues: Dict[str, asyncio.PriorityQueue] = defaultdict(
            lambda: asyncio.PriorityQueue()
        )
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_history: deque = deque(maxlen=10000)
        self.broadcast_topics: Dict[str, Set[str]] = defaultdict(set)
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.max_queue_size = 1000
        self.default_timeout = 30.0
        self.enable_message_history = True
        self.enable_priority_queuing = True
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "response_timeouts": 0
        }
        
    async def register_agent(self, agent_id: str, message_handler: Callable):
        """Register an agent for communication"""
        if agent_id in self.message_handlers:
            logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self.message_handlers[agent_id] = message_handler
        logger.info(f"Registered agent {agent_id} for communication")
        return True
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent from communication"""
        if agent_id in self.message_handlers:
            del self.message_handlers[agent_id]
        
        # Remove from all topics
        for topic in list(self.broadcast_topics.keys()):
            self.broadcast_topics[topic].discard(agent_id)
        
        # Clear pending responses
        expired_responses = []
        for msg_id, future in self.pending_responses.items():
            if future.done():
                expired_responses.append(msg_id)
        
        for msg_id in expired_responses:
            del self.pending_responses[msg_id]
        
        logger.info(f"Unregistered agent {agent_id} from communication")
    
    async def send_message(self, message: AgentMessage) -> Optional[Any]:
        """Send a message to an agent"""
        self.stats["messages_sent"] += 1
        
        # Check if message is expired
        if message.is_expired():
            logger.warning(f"Message {message.message_id} is expired")
            self.stats["messages_failed"] += 1
            return None
        
        # Record in history
        if self.enable_message_history:
            self.message_history.append(message)
        
        if message.recipient_id:
            # Direct message
            return await self._send_direct_message(message)
        else:
            # Broadcast message
            return await self._broadcast_message(message)
    
    async def _send_direct_message(self, message: AgentMessage) -> Optional[Any]:
        """Send a direct message to a specific agent"""
        if message.recipient_id not in self.message_queues:
            logger.error(f"Recipient {message.recipient_id} not found")
            self.stats["messages_failed"] += 1
            return None
        
        # Add to recipient's queue
        priority = -message.priority  # Negative for max-heap behavior
        await self.message_queues[message.recipient_id].put((priority, message))
        
        if message.requires_response:
            # Create future for response
            future = asyncio.Future()
            self.pending_responses[message.message_id] = future
            
            try:
                # Wait for response with timeout
                timeout = message.metadata.get("timeout", self.default_timeout)
                response = await asyncio.wait_for(future, timeout=timeout)
                self.stats["messages_delivered"] += 1
                return response
            except asyncio.TimeoutError:
                logger.error(f"Response timeout for message {message.message_id}")
                self.stats["response_timeouts"] += 1
                if message.message_id in self.pending_responses:
                    del self.pending_responses[message.message_id]
                return None
            except Exception as e:
                logger.error(f"Error waiting for response: {e}")
                if message.message_id in self.pending_responses:
                    del self.pending_responses[message.message_id]
                return None
        else:
            self.stats["messages_delivered"] += 1
            return None
    
    async def _broadcast_message(self, message: AgentMessage):
        """Broadcast a message to all agents"""
        tasks = []
        recipients = set(self.message_queues.keys())
        recipients.discard(message.sender_id)  # Don't send to self
        
        for agent_id in recipients:
            # Create a copy of the message for each recipient
            recipient_message = AgentMessage(
                message_id=f"{message.message_id}_{agent_id}",
                sender_id=message.sender_id,
                recipient_id=agent_id,
                message_type=message.message_type,
                payload=message.payload,
                timestamp=message.timestamp,
                requires_response=message.requires_response,
                correlation_id=message.correlation_id,
                priority=message.priority,
                ttl=message.ttl,
                metadata=message.metadata
            )
            
            task = self._send_direct_message(recipient_message)
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is not None)
            logger.debug(f"Broadcast sent to {successful}/{len(recipients)} agents")
    
    async def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe an agent to a broadcast topic"""
        self.broadcast_topics[topic].add(agent_id)
        self.agent_subscriptions[agent_id].add(topic)
        logger.debug(f"Agent {agent_id} subscribed to topic: {topic}")
    
    async def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a broadcast topic"""
        self.broadcast_topics[topic].discard(agent_id)
        self.agent_subscriptions[agent_id].discard(topic)
        logger.debug(f"Agent {agent_id} unsubscribed from topic: {topic}")
    
    async def publish_to_topic(self, topic: str, message: AgentMessage):
        """Publish a message to a specific topic"""
        subscribers = self.broadcast_topics.get(topic, set())
        
        if not subscribers:
            logger.debug(f"No subscribers for topic: {topic}")
            return
        
        tasks = []
        for agent_id in subscribers:
            if agent_id != message.sender_id and agent_id in self.message_queues:
                # Create topic-specific message
                topic_message = AgentMessage(
                    message_id=f"{message.message_id}_topic_{topic}",
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    message_type=message.message_type,
                    payload=message.payload,
                    timestamp=message.timestamp,
                    requires_response=message.requires_response,
                    correlation_id=message.correlation_id,
                    priority=message.priority,
                    ttl=message.ttl,
                    metadata={**message.metadata, "topic": topic}
                )
                
                task = self._send_direct_message(topic_message)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Published to topic {topic} with {len(subscribers)} subscribers")
    
    async def process_messages(self, agent_id: str):
        """Process messages for an agent"""
        if agent_id not in self.message_handlers:
            logger.error(f"No handler registered for agent {agent_id}")
            return
        
        handler = self.message_handlers[agent_id]
        
        while True:
            try:
                # Get next message from queue
                priority, message = await self.message_queues[agent_id].get()
                self.stats["messages_received"] += 1
                
                # Check if message is expired
                if message.is_expired():
                    logger.debug(f"Skipping expired message: {message.message_id}")
                    continue
                
                # Process message
                response = await handler(message)
                
                # Send response if required
                if message.requires_response and message.message_id in self.pending_responses:
                    future = self.pending_responses[message.message_id]
                    if not future.done():
                        future.set_result(response)
                    del self.pending_responses[message.message_id]
                
                # Mark task as done
                self.message_queues[agent_id].task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Message processing cancelled for agent {agent_id}")
                break
            except Exception as e:
                logger.error(f"Error processing message for {agent_id}: {e}")
                self.stats["messages_failed"] += 1
    
    async def send_response(self, original_message: AgentMessage, response: Any):
        """Send a response to a message"""
        response_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=original_message.recipient_id or "system",
            recipient_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            payload=response,
            timestamp=time.time(),
            requires_response=False,
            correlation_id=original_message.message_id,
            priority=original_message.priority,
            metadata={"response_to": original_message.message_id}
        )
        
        return await self.send_message(response_message)
    
    async def send_heartbeat(self, agent_id: str, status: Dict[str, Any]):
        """Send a heartbeat message"""
        heartbeat_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=agent_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.HEARTBEAT,
            payload=status,
            timestamp=time.time(),
            requires_response=False,
            priority=1,  # Low priority
            ttl=60.0  # 1 minute TTL
        )
        
        return await self.send_message(heartbeat_message)
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics"""
        return {
            **self.stats,
            "active_agents": len(self.message_handlers),
            "pending_responses": len(self.pending_responses),
            "total_topics": len(self.broadcast_topics),
            "history_size": len(self.message_history)
        }
    
    async def cleanup_expired_messages(self):
        """Clean up expired messages and responses"""
        # Clean up expired pending responses
        current_time = time.time()
        expired_responses = []
        
        for msg_id, future in self.pending_responses.items():
            if future.done():
                expired_responses.append(msg_id)
        
        for msg_id in expired_responses:
            del self.pending_responses[msg_id]
        
        # Clean up expired messages in history
        if self.enable_message_history:
            expired_messages = [
                msg for msg in self.message_history
                if msg.is_expired()
            ]
            
            for msg in expired_messages:
                self.message_history.remove(msg)
        
        if expired_responses or expired_messages:
            logger.debug(f"Cleaned up {len(expired_responses)} expired responses "
                        f"and {len(expired_messages)} expired messages")
    
    async def shutdown(self):
        """Shutdown the communication protocol"""
        # Cancel all pending responses
        for future in self.pending_responses.values():
            if not future.done():
                future.cancel()
        
        # Clear all queues
        for queue in self.message_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
        
        logger.info("Communication protocol shut down") 