# Real-time Collaboration System for AI Agent Platform
# src/collaboration/realtime_collaboration.py

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Data Models
class CollaborationType(str, Enum):
    AGENT_HANDOFF = "agent_handoff"
    SHARED_CONTEXT = "shared_context"
    PARALLEL_EXECUTION = "parallel_execution"
    REVIEW_APPROVAL = "review_approval"
    LIVE_EDITING = "live_editing"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class CollaborationSession:
    """Represents a real-time collaboration session"""
    session_id: str
    name: str
    participants: List[str]  # User IDs
    agents: List[str]  # Agent IDs
    created_at: datetime
    status: SessionStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class CollaborationEvent:
    """Event in a collaboration session"""
    event_id: str
    session_id: str
    event_type: str
    actor_id: str  # User or Agent ID
    timestamp: datetime
    data: Dict[str, Any]
    
@dataclass
class AgentHandoff:
    """Agent handoff protocol"""
    handoff_id: str
    session_id: str
    from_agent: str
    to_agent: str
    context: Dict[str, Any]
    reason: str
    timestamp: datetime
    accepted: bool = False

# WebSocket Connection Manager
class CollaborationManager:
    """Manages real-time collaboration sessions and WebSocket connections"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.sessions: Dict[str, CollaborationSession] = {}
        self.connections: Dict[str, Set[WebSocket]] = {}  # session_id -> websockets
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.agent_sessions: Dict[str, str] = {}  # agent_id -> session_id
        
    async def initialize(self):
        """Initialize Redis connection and pubsub"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
            
    # Session Management
    async def create_session(
        self,
        name: str,
        creator_id: str,
        participants: List[str] = None,
        agents: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> CollaborationSession:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())
        session = CollaborationSession(
            session_id=session_id,
            name=name,
            participants=participants or [creator_id],
            agents=agents or [],
            created_at=datetime.utcnow(),
            status=SessionStatus.ACTIVE,
            metadata=metadata or {}
        )
        
        # Store in memory and Redis
        self.sessions[session_id] = session
        await self._save_session_to_redis(session)
        
        # Update user mapping
        for user_id in session.participants:
            self.user_sessions[user_id] = session_id
            
        # Update agent mapping
        for agent_id in session.agents:
            self.agent_sessions[agent_id] = session_id
            
        # Broadcast session creation
        await self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="session_created",
            actor_id=creator_id,
            timestamp=datetime.utcnow(),
            data={"session": self._session_to_dict(session)}
        ))
        
        logger.info(f"Created collaboration session {session_id}")
        return session
        
    async def join_session(
        self,
        session_id: str,
        user_id: str,
        websocket: WebSocket
    ):
        """Join a collaboration session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        
        # Add user to session if not already present
        if user_id not in session.participants:
            session.participants.append(user_id)
            await self._save_session_to_redis(session)
            
        # Update mappings
        self.user_sessions[user_id] = session_id
        
        # Add WebSocket connection
        if session_id not in self.connections:
            self.connections[session_id] = set()
        self.connections[session_id].add(websocket)
        
        # Subscribe to session channel
        await self.pubsub.subscribe(f"session:{session_id}")
        
        # Send session state to new participant
        await websocket.send_json({
            "type": "session_state",
            "data": {
                "session": self._session_to_dict(session),
                "shared_context": session.shared_context
            }
        })
        
        # Broadcast join event
        await self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="user_joined",
            actor_id=user_id,
            timestamp=datetime.utcnow(),
            data={"user_id": user_id}
        ))
        
    async def leave_session(self, session_id: str, user_id: str, websocket: WebSocket):
        """Leave a collaboration session"""
        if session_id in self.connections and websocket in self.connections[session_id]:
            self.connections[session_id].remove(websocket)
            
            # Clean up empty connection sets
            if not self.connections[session_id]:
                del self.connections[session_id]
                
        # Remove user mapping
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
            
        # Broadcast leave event
        await self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="user_left",
            actor_id=user_id,
            timestamp=datetime.utcnow(),
            data={"user_id": user_id}
        ))
        
    # Agent Handoff Protocol
    async def initiate_handoff(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        context: Dict[str, Any],
        reason: str
    ) -> AgentHandoff:
        """Initiate agent handoff"""
        handoff = AgentHandoff(
            handoff_id=str(uuid.uuid4()),
            session_id=session_id,
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
            reason=reason,
            timestamp=datetime.utcnow()
        )
        
        # Store handoff in Redis
        await self.redis_client.setex(
            f"handoff:{handoff.handoff_id}",
            300,  # 5 minute TTL
            json.dumps(self._handoff_to_dict(handoff))
        )
        
        # Broadcast handoff request
        await self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="handoff_requested",
            actor_id=from_agent,
            timestamp=datetime.utcnow(),
            data=self._handoff_to_dict(handoff)
        ))
        
        return handoff
        
    async def accept_handoff(self, handoff_id: str, agent_id: str):
        """Accept agent handoff"""
        # Retrieve handoff from Redis
        handoff_data = await self.redis_client.get(f"handoff:{handoff_id}")
        if not handoff_data:
            raise ValueError(f"Handoff {handoff_id} not found or expired")
            
        handoff_dict = json.loads(handoff_data)
        
        # Verify the accepting agent
        if handoff_dict["to_agent"] != agent_id:
            raise ValueError(f"Agent {agent_id} is not the target of this handoff")
            
        # Update handoff status
        handoff_dict["accepted"] = True
        await self.redis_client.setex(
            f"handoff:{handoff_id}",
            300,
            json.dumps(handoff_dict)
        )
        
        # Update agent sessions
        session_id = handoff_dict["session_id"]
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Remove from_agent and add to_agent
            if handoff_dict["from_agent"] in session.agents:
                session.agents.remove(handoff_dict["from_agent"])
            if agent_id not in session.agents:
                session.agents.append(agent_id)
                
            await self._save_session_to_redis(session)
            
            # Update agent mappings
            if handoff_dict["from_agent"] in self.agent_sessions:
                del self.agent_sessions[handoff_dict["from_agent"]]
            self.agent_sessions[agent_id] = session_id
            
        # Broadcast handoff acceptance
        await self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="handoff_accepted",
            actor_id=agent_id,
            timestamp=datetime.utcnow(),
            data=handoff_dict
        ))
        
    # Shared Context Management
    async def update_shared_context(
        self,
        session_id: str,
        actor_id: str,
        updates: Dict[str, Any]
    ):
        """Update shared context for collaboration"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        
        # Apply updates with conflict resolution
        for key, value in updates.items():
            if key in session.shared_context:
                # Simple last-write-wins for now
                # TODO: Implement more sophisticated conflict resolution
                logger.warning(f"Overwriting shared context key: {key}")
                
            session.shared_context[key] = value
            
        await self._save_session_to_redis(session)
        
        # Broadcast context update
        await self._broadcast_event(CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="context_updated",
            actor_id=actor_id,
            timestamp=datetime.utcnow(),
            data={"updates": updates}
        ))
        
    # Live Progress Tracking
    async def report_progress(
        self,
        session_id: str,
        agent_id: str,
        task_id: str,
        progress: float,
        status: str,
        details: Dict[str, Any] = None
    ):
        """Report task progress in real-time"""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="progress_update",
            actor_id=agent_id,
            timestamp=datetime.utcnow(),
            data={
                "task_id": task_id,
                "progress": progress,
                "status": status,
                "details": details or {}
            }
        )
        
        await self._broadcast_event(event)
        
        # Store progress in Redis for persistence
        progress_key = f"progress:{session_id}:{task_id}"
        await self.redis_client.setex(
            progress_key,
            3600,  # 1 hour TTL
            json.dumps(event.data)
        )
        
    # Collaborative Editing
    async def submit_edit(
        self,
        session_id: str,
        user_id: str,
        document_id: str,
        edit_type: str,
        position: Dict[str, int],
        content: str
    ):
        """Submit a collaborative edit"""
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type="edit_submitted",
            actor_id=user_id,
            timestamp=datetime.utcnow(),
            data={
                "document_id": document_id,
                "edit_type": edit_type,
                "position": position,
                "content": content
            }
        )
        
        # Apply Operational Transformation if needed
        # TODO: Implement OT for conflict-free collaborative editing
        
        await self._broadcast_event(event)
        
    # Helper Methods
    async def _broadcast_event(self, event: CollaborationEvent):
        """Broadcast event to all session participants"""
        # Publish to Redis channel
        await self.redis_client.publish(
            f"session:{event.session_id}",
            json.dumps(self._event_to_dict(event))
        )
        
        # Send to WebSocket connections
        if event.session_id in self.connections:
            message = json.dumps({
                "type": "collaboration_event",
                "event": self._event_to_dict(event)
            })
            
            disconnected = set()
            for websocket in self.connections[event.session_id]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send to websocket: {e}")
                    disconnected.add(websocket)
                    
            # Remove disconnected websockets
            self.connections[event.session_id] -= disconnected
            
    async def _save_session_to_redis(self, session: CollaborationSession):
        """Save session state to Redis"""
        session_key = f"session:{session.session_id}"
        await self.redis_client.setex(
            session_key,
            86400,  # 24 hour TTL
            json.dumps(self._session_to_dict(session))
        )
        
    def _session_to_dict(self, session: CollaborationSession) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": session.session_id,
            "name": session.name,
            "participants": session.participants,
            "agents": session.agents,
            "created_at": session.created_at.isoformat(),
            "status": session.status.value,
            "metadata": session.metadata,
            "shared_context": session.shared_context
        }
        
    def _event_to_dict(self, event: CollaborationEvent) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "event_type": event.event_type,
            "actor_id": event.actor_id,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data
        }
        
    def _handoff_to_dict(self, handoff: AgentHandoff) -> Dict[str, Any]:
        """Convert handoff to dictionary"""
        return {
            "handoff_id": handoff.handoff_id,
            "session_id": handoff.session_id,
            "from_agent": handoff.from_agent,
            "to_agent": handoff.to_agent,
            "context": handoff.context,
            "reason": handoff.reason,
            "timestamp": handoff.timestamp.isoformat(),
            "accepted": handoff.accepted
        }

# WebSocket Handler for Real-time Collaboration
class CollaborationWebSocketHandler:
    """Handles WebSocket connections for collaboration"""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.manager = collaboration_manager
        
    async def handle_connection(self, websocket: WebSocket, user_id: str):
        """Handle a WebSocket connection"""
        await websocket.accept()
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await self._handle_message(websocket, user_id, message)
                
        except WebSocketDisconnect:
            # Handle disconnection
            session_id = self.manager.user_sessions.get(user_id)
            if session_id:
                await self.manager.leave_session(session_id, user_id, websocket)
                
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}")
            await websocket.close()
            
    async def _handle_message(self, websocket: WebSocket, user_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        msg_type = message.get("type")
        
        if msg_type == "join_session":
            session_id = message["session_id"]
            await self.manager.join_session(session_id, user_id, websocket)
            
        elif msg_type == "leave_session":
            session_id = message["session_id"]
            await self.manager.leave_session(session_id, user_id, websocket)
            
        elif msg_type == "update_context":
            session_id = message["session_id"]
            updates = message["updates"]
            await self.manager.update_shared_context(session_id, user_id, updates)
            
        elif msg_type == "submit_edit":
            await self.manager.submit_edit(
                session_id=message["session_id"],
                user_id=user_id,
                document_id=message["document_id"],
                edit_type=message["edit_type"],
                position=message["position"],
                content=message["content"]
            )
            
        elif msg_type == "report_progress":
            # For agents reporting progress
            await self.manager.report_progress(
                session_id=message["session_id"],
                agent_id=user_id,  # Could be agent_id
                task_id=message["task_id"],
                progress=message["progress"],
                status=message["status"],
                details=message.get("details")
            )
            
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            
# Integration with existing API server
def setup_collaboration_routes(app, collaboration_manager: CollaborationManager):
    """Setup collaboration routes on FastAPI app"""
    
    @app.post("/api/v1/collaboration/sessions")
    async def create_collaboration_session(
        name: str,
        creator_id: str,
        participants: List[str] = None,
        agents: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Create a new collaboration session"""
        session = await collaboration_manager.create_session(
            name=name,
            creator_id=creator_id,
            participants=participants,
            agents=agents,
            metadata=metadata
        )
        return {"session": collaboration_manager._session_to_dict(session)}
        
    @app.websocket("/ws/collaboration/{user_id}")
    async def collaboration_websocket(websocket: WebSocket, user_id: str):
        """WebSocket endpoint for collaboration"""
        handler = CollaborationWebSocketHandler(collaboration_manager)
        await handler.handle_connection(websocket, user_id)
        
    @app.post("/api/v1/collaboration/handoff")
    async def initiate_agent_handoff(
        session_id: str,
        from_agent: str,
        to_agent: str,
        context: Dict[str, Any],
        reason: str
    ):
        """Initiate agent handoff"""
        handoff = await collaboration_manager.initiate_handoff(
            session_id=session_id,
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
            reason=reason
        )
        return {"handoff": collaboration_manager._handoff_to_dict(handoff)}
        
    @app.post("/api/v1/collaboration/handoff/{handoff_id}/accept")
    async def accept_agent_handoff(handoff_id: str, agent_id: str):
        """Accept agent handoff"""
        await collaboration_manager.accept_handoff(handoff_id, agent_id)
        return {"status": "accepted"} 