"""
Supabase implementations of repository interfaces with circuit breaker patterns and monitoring.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from supabase import create_client, Client

from src.core.entities.message import Message, MessageType, MessageStatus
from src.core.interfaces.message_repository import MessageRepository
from src.shared.exceptions import InfrastructureException
from src.infrastructure.monitoring import (
    track_database_operation, record_error, update_db_connections,
    time_function, track_async_operation
)
from src.infrastructure.resilience import (
    circuit_breaker, get_db_circuit_breaker, DB_CIRCUIT_BREAKER_CONFIG
)

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Singleton Supabase client with circuit breaker protection."""
    _instance: Optional[Client] = None
    _connection_count = 0

    @classmethod
    def get_client(cls, url: str, key: str) -> Client:
        """Get or create Supabase client."""
        if cls._instance is None:
            cls._instance = create_client(url, key)
            cls._connection_count += 1
            update_db_connections("supabase", cls._connection_count)
        return cls._instance

    @classmethod
    def close_connection(cls):
        """Close the database connection."""
        if cls._instance:
            cls._connection_count = max(0, cls._connection_count - 1)
            update_db_connections("supabase", cls._connection_count)
            cls._instance = None

class SupabaseMessageRepository(MessageRepository):
    """Supabase implementation of message repository with monitoring and resilience."""
    
    def __init__(self, client: Client):
        self.client = client
        self.table = "messages"
        self._circuit_breaker = None

    async def _get_circuit_breaker(self):
        """Get circuit breaker for database operations."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await get_db_circuit_breaker()
        return self._circuit_breaker

    @track_database_operation("save", "messages")
    @time_function("message_save", {"operation": "save", "table": "messages"})
    async def save(self, message: Message) -> Message:
        """Save message to database with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _save_operation():
                data = {
                    "id": str(message.id),
                    "content": message.content,
                    "message_type": message.message_type.value,
                    "session_id": str(message.session_id) if message.session_id else None,
                    "user_id": str(message.user_id) if message.user_id else None,
                    "agent_id": str(message.agent_id) if message.agent_id else None,
                    "parent_message_id": str(message.parent_message_id) if message.parent_message_id else None,
                    "context": json.dumps(message.context),
                    "metadata": json.dumps(message.metadata),
                    "status": message.status.value,
                    "processing_time": message.processing_time,
                    "error_message": message.error_message,
                    "created_at": message.created_at.isoformat(),
                    "updated_at": message.updated_at.isoformat()
                }
                result = self.client.table(self.table).upsert(data).execute()
                if result.data:
                    return message
                else:
                    raise InfrastructureException("Failed to save message")
                    
            return await breaker.call(_save_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error saving message: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_id", "messages")
    @time_function("message_find_by_id", {"operation": "find_by_id", "table": "messages"})
    async def find_by_id(self, message_id: UUID) -> Optional[Message]:
        """Find message by ID with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("id", str(message_id)).execute()
                if result.data and len(result.data) > 0:
                    return self._to_entity(result.data[0])
                return None
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error finding message: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_session", "messages")
    @time_function("message_find_by_session", {"operation": "find_by_session", "table": "messages"})
    async def find_by_session(self, session_id: UUID) -> List[Message]:
        """Find messages by session with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("session_id", str(session_id)).order("created_at").execute()
                return [self._to_entity(data) for data in result.data]
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error finding messages by session: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_type", "messages")
    @time_function("message_find_by_type", {"operation": "find_by_type", "table": "messages"})
    async def find_by_type(self, message_type: MessageType) -> List[Message]:
        """Find messages by type with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("message_type", message_type.value).execute()
                return [self._to_entity(data) for data in result.data]
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error finding messages by type: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("delete", "messages")
    @time_function("message_delete", {"operation": "delete", "table": "messages"})
    async def delete(self, message_id: UUID) -> bool:
        """Delete message with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _delete_operation():
                result = self.client.table(self.table).delete().eq("id", str(message_id)).execute()
                return len(result.data) > 0
                
            return await breaker.call(_delete_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error deleting message: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("get_statistics", "messages")
    @time_function("message_statistics", {"operation": "get_statistics", "table": "messages"})
    async def get_statistics(self) -> dict:
        """Get message statistics with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _statistics_operation():
                total_result = self.client.table(self.table).select("id", count="exact").execute()
                total_count = total_result.count if total_result else 0
                
                type_counts = {}
                for msg_type in MessageType:
                    result = self.client.table(self.table).select("id", count="exact").eq("message_type", msg_type.value).execute()
                    type_counts[msg_type.value] = result.count if result else 0
                    
                status_counts = {}
                for status in MessageStatus:
                    result = self.client.table(self.table).select("id", count="exact").eq("status", status.value).execute()
                    status_counts[status.value] = result.count if result else 0
                    
                return {
                    "total_messages": total_count,
                    "by_type": type_counts,
                    "by_status": status_counts
                }
                
            return await breaker.call(_statistics_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error getting statistics: {}", extra={"str_e_": str(e)})
            return {"error": str(e)}

    def _to_entity(self, data: Dict[str, Any]) -> Message:
        """Convert database record to entity."""
        try:
            message = Message(
                content=data["content"],
                message_type=MessageType(data["message_type"])
            )
            message.id = UUID(data["id"])
            message.session_id = UUID(data["session_id"]) if data.get("session_id") else None
            message.user_id = UUID(data["user_id"]) if data.get("user_id") else None
            message.agent_id = UUID(data["agent_id"]) if data.get("agent_id") else None
            message.parent_message_id = UUID(data["parent_message_id"]) if data.get("parent_message_id") else None
            message.context = json.loads(data.get("context", "{}"))
            message.metadata = json.loads(data.get("metadata", "{}"))
            message.status = MessageStatus(data["status"])
            message.processing_time = data.get("processing_time")
            message.error_message = data.get("error_message")
            message.created_at = datetime.fromisoformat(data["created_at"])
            message.updated_at = datetime.fromisoformat(data["updated_at"])
            return message
        except Exception as e:
            record_error(type(e).__name__, "message_repository", "error")
            logger.error("Error converting data to entity: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Data conversion error: {str(e)}")

# Import missing classes for the rest of the file
from src.core.entities.tool import Tool, ToolType
from src.core.entities.session import Session
from src.core.entities.agent import Agent, AgentType
from src.core.interfaces.tool_repository import ToolRepository
from src.core.interfaces.session_repository import SessionRepository
from src.core.interfaces.agent_repository import AgentRepository

class SupabaseToolRepository(ToolRepository):
    """Supabase implementation of tool repository with monitoring and resilience."""
    
    def __init__(self, client: Client):
        self.client = client
        self.table = "tools"
        self.metrics_table = "tool_reliability_metrics"
        self._circuit_breaker = None

    async def _get_circuit_breaker(self):
        """Get circuit breaker for database operations."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await get_db_circuit_breaker()
        return self._circuit_breaker

    @track_database_operation("save", "tools")
    @time_function("tool_save", {"operation": "save", "table": "tools"})
    async def save(self, tool: Tool) -> Tool:
        """Save tool to database with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _save_operation():
                data = {
                    "id": str(tool.id),
                    "name": tool.name,
                    "description": tool.description,
                    "tool_type": tool.tool_type.value,
                    "parameters_schema": json.dumps(tool.parameters_schema),
                    "return_schema": json.dumps(tool.return_schema),
                    "status": tool.status.value,
                    "is_enabled": tool.is_enabled,
                    "is_public": tool.is_public,
                    "version": tool.version,
                    "author": tool.author,
                    "tags": tool.tags,
                    "documentation": tool.documentation,
                    "created_at": tool.created_at.isoformat(),
                    "updated_at": tool.updated_at.isoformat()
                }
                result = self.client.table(self.table).upsert(data).execute()
                if result.data:
                    await self._update_metrics(tool)
                    return tool
                else:
                    raise InfrastructureException("Failed to save tool")
                    
            return await breaker.call(_save_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error saving tool: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_id", "tools")
    @time_function("tool_find_by_id", {"operation": "find_by_id", "table": "tools"})
    async def find_by_id(self, tool_id: UUID) -> Optional[Tool]:
        """Find tool by ID with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("id", str(tool_id)).execute()
                if result.data and len(result.data) > 0:
                    tool = self._to_entity(result.data[0])
                    await self._load_metrics(tool)
                    return tool
                return None
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error finding tool: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_name", "tools")
    @time_function("tool_find_by_name", {"operation": "find_by_name", "table": "tools"})
    async def find_by_name(self, name: str) -> Optional[Tool]:
        """Find tool by name with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("name", name).execute()
                if result.data and len(result.data) > 0:
                    tool = self._to_entity(result.data[0])
                    await self._load_metrics(tool)
                    return tool
                return None
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error finding tool by name: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_type", "tools")
    @time_function("tool_find_by_type", {"operation": "find_by_type", "table": "tools"})
    async def find_by_type(self, tool_type: ToolType) -> List[Tool]:
        """Find tools by type with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("tool_type", tool_type.value).execute()
                tools = [self._to_entity(data) for data in result.data]
                for tool in tools:
                    await self._load_metrics(tool)
                return tools
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error finding tools by type: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("delete", "tools")
    @time_function("tool_delete", {"operation": "delete", "table": "tools"})
    async def delete(self, tool_id: UUID) -> bool:
        """Delete tool with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _delete_operation():
                result = self.client.table(self.table).delete().eq("id", str(tool_id)).execute()
                return len(result.data) > 0
                
            return await breaker.call(_delete_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error deleting tool: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("get_statistics", "tools")
    @time_function("tool_statistics", {"operation": "get_statistics", "table": "tools"})
    async def get_statistics(self) -> dict:
        """Get tool statistics with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _statistics_operation():
                total_result = self.client.table(self.table).select("id", count="exact").execute()
                total_count = total_result.count if total_result else 0
                
                type_counts = {}
                for tool_type in ToolType:
                    result = self.client.table(self.table).select("id", count="exact").eq("tool_type", tool_type.value).execute()
                    type_counts[tool_type.value] = result.count if result else 0
                    
                enabled_result = self.client.table(self.table).select("id", count="exact").eq("is_enabled", True).execute()
                enabled_count = enabled_result.count if enabled_result else 0
                
                return {
                    "total_tools": total_count,
                    "by_type": type_counts,
                    "enabled_tools": enabled_count,
                    "disabled_tools": total_count - enabled_count
                }
                
            return await breaker.call(_statistics_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error getting statistics: {}", extra={"str_e_": str(e)})
            return {"error": str(e)}

    async def _update_metrics(self, tool: Tool) -> None:
        """Update tool reliability metrics."""
        try:
            async with track_async_operation("update_tool_metrics", {"tool_id": str(tool.id)}):
                # This would update reliability metrics for the tool
                pass
        except Exception as e:
            logger.warning("Failed to update tool metrics: {}", extra={"str_e_": str(e)})

    async def _load_metrics(self, tool: Tool) -> None:
        """Load tool reliability metrics."""
        try:
            async with track_async_operation("load_tool_metrics", {"tool_id": str(tool.id)}):
                # This would load reliability metrics for the tool
                pass
        except Exception as e:
            logger.warning("Failed to load tool metrics: {}", extra={"str_e_": str(e)})

    def _to_entity(self, data: Dict[str, Any]) -> Tool:
        """Convert database record to entity."""
        try:
            tool = Tool(
                name=data["name"],
                description=data["description"],
                tool_type=ToolType(data["tool_type"])
            )
            tool.id = UUID(data["id"])
            tool.parameters_schema = json.loads(data.get("parameters_schema", "{}"))
            tool.return_schema = json.loads(data.get("return_schema", "{}"))
            tool.status = data.get("status", "active")
            tool.is_enabled = data.get("is_enabled", True)
            tool.is_public = data.get("is_public", False)
            tool.version = data.get("version", "1.0.0")
            tool.author = data.get("author", "")
            tool.tags = data.get("tags", [])
            tool.documentation = data.get("documentation", "")
            tool.created_at = datetime.fromisoformat(data["created_at"])
            tool.updated_at = datetime.fromisoformat(data["updated_at"])
            return tool
        except Exception as e:
            record_error(type(e).__name__, "tool_repository", "error")
            logger.error("Error converting data to entity: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Data conversion error: {str(e)}")

class SupabaseSessionRepository(SessionRepository):
    """Supabase implementation of session repository with monitoring and resilience."""
    
    def __init__(self, client: Client):
        self.client = client
        self.table = "sessions"
        self._circuit_breaker = None

    async def _get_circuit_breaker(self):
        """Get circuit breaker for database operations."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await get_db_circuit_breaker()
        return self._circuit_breaker

    @track_database_operation("save", "sessions")
    @time_function("session_save", {"operation": "save", "table": "sessions"})
    async def save(self, session: Session) -> Session:
        """Save session to database with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _save_operation():
                data = {
                    "id": str(session.id),
                    "user_id": str(session.user_id) if session.user_id else None,
                    "agent_id": str(session.agent_id) if session.agent_id else None,
                    "status": session.status.value,
                    "context": json.dumps(session.context),
                    "metadata": json.dumps(session.metadata),
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat()
                }
                result = self.client.table(self.table).upsert(data).execute()
                if result.data:
                    return session
                else:
                    raise InfrastructureException("Failed to save session")
                    
            return await breaker.call(_save_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "session_repository", "error")
            logger.error("Error saving session: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_id", "sessions")
    @time_function("session_find_by_id", {"operation": "find_by_id", "table": "sessions"})
    async def find_by_id(self, session_id: UUID) -> Optional[Session]:
        """Find session by ID with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("id", str(session_id)).execute()
                if result.data and len(result.data) > 0:
                    return self._to_entity(result.data[0])
                return None
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "session_repository", "error")
            logger.error("Error finding session: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_active", "sessions")
    @time_function("session_find_active", {"operation": "find_active", "table": "sessions"})
    async def find_active(self) -> List[Session]:
        """Find active sessions with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("status", "active").execute()
                return [self._to_entity(data) for data in result.data]
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "session_repository", "error")
            logger.error("Error finding active sessions: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("delete", "sessions")
    @time_function("session_delete", {"operation": "delete", "table": "sessions"})
    async def delete(self, session_id: UUID) -> bool:
        """Delete session with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _delete_operation():
                result = self.client.table(self.table).delete().eq("id", str(session_id)).execute()
                return len(result.data) > 0
                
            return await breaker.call(_delete_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "session_repository", "error")
            logger.error("Error deleting session: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("get_statistics", "sessions")
    @time_function("session_statistics", {"operation": "get_statistics", "table": "sessions"})
    async def get_statistics(self) -> dict:
        """Get session statistics with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _statistics_operation():
                total_result = self.client.table(self.table).select("id", count="exact").execute()
                total_count = total_result.count if total_result else 0
                
                active_result = self.client.table(self.table).select("id", count="exact").eq("status", "active").execute()
                active_count = active_result.count if active_result else 0
                
                completed_result = self.client.table(self.table).select("id", count="exact").eq("status", "completed").execute()
                completed_count = completed_result.count if completed_result else 0
                
                return {
                    "total_sessions": total_count,
                    "active_sessions": active_count,
                    "completed_sessions": completed_count,
                    "terminated_sessions": total_count - active_count - completed_count
                }
                
            return await breaker.call(_statistics_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "session_repository", "error")
            logger.error("Error getting statistics: {}", extra={"str_e_": str(e)})
            return {"error": str(e)}

    def _to_entity(self, data: Dict[str, Any]) -> Session:
        """Convert database record to entity."""
        try:
            session = Session()
            session.id = UUID(data["id"])
            session.user_id = UUID(data["user_id"]) if data.get("user_id") else None
            session.agent_id = UUID(data["agent_id"]) if data.get("agent_id") else None
            session.status = data.get("status", "active")
            session.context = json.loads(data.get("context", "{}"))
            session.metadata = json.loads(data.get("metadata", "{}"))
            session.start_time = datetime.fromisoformat(data["start_time"])
            session.end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
            session.created_at = datetime.fromisoformat(data["created_at"])
            session.updated_at = datetime.fromisoformat(data["updated_at"])
            return session
        except Exception as e:
            record_error(type(e).__name__, "session_repository", "error")
            logger.error("Error converting data to entity: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Data conversion error: {str(e)}")

class SupabaseAgentRepository(AgentRepository):
    """Supabase implementation of agent repository with monitoring and resilience."""
    
    def __init__(self, client: Client):
        self.client = client
        self.table = "agents"
        self._circuit_breaker = None

    async def _get_circuit_breaker(self):
        """Get circuit breaker for database operations."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await get_db_circuit_breaker()
        return self._circuit_breaker

    @track_database_operation("save", "agents")
    @time_function("agent_save", {"operation": "save", "table": "agents"})
    async def save(self, agent: Agent) -> Agent:
        """Save agent to database with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _save_operation():
                data = {
                    "id": str(agent.id),
                    "name": agent.name,
                    "description": agent.description,
                    "agent_type": agent.agent_type.value,
                    "capabilities": json.dumps(agent.capabilities),
                    "status": agent.status.value,
                    "is_available": agent.is_available,
                    "performance_metrics": json.dumps(agent.performance_metrics),
                    "metadata": json.dumps(agent.metadata),
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat()
                }
                result = self.client.table(self.table).upsert(data).execute()
                if result.data:
                    return agent
                else:
                    raise InfrastructureException("Failed to save agent")
                    
            return await breaker.call(_save_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error saving agent: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_id", "agents")
    @time_function("agent_find_by_id", {"operation": "find_by_id", "table": "agents"})
    async def find_by_id(self, agent_id: UUID) -> Optional[Agent]:
        """Find agent by ID with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("id", str(agent_id)).execute()
                if result.data and len(result.data) > 0:
                    return self._to_entity(result.data[0])
                return None
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error finding agent: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_by_type", "agents")
    @time_function("agent_find_by_type", {"operation": "find_by_type", "table": "agents"})
    async def find_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Find agents by type with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("agent_type", agent_type.value).execute()
                return [self._to_entity(data) for data in result.data]
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error finding agents by type: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("find_available", "agents")
    @time_function("agent_find_available", {"operation": "find_available", "table": "agents"})
    async def find_available(self) -> List[Agent]:
        """Find available agents with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _find_operation():
                result = self.client.table(self.table).select("*").eq("is_available", True).execute()
                return [self._to_entity(data) for data in result.data]
                
            return await breaker.call(_find_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error finding available agents: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("delete", "agents")
    @time_function("agent_delete", {"operation": "delete", "table": "agents"})
    async def delete(self, agent_id: UUID) -> bool:
        """Delete agent with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _delete_operation():
                result = self.client.table(self.table).delete().eq("id", str(agent_id)).execute()
                return len(result.data) > 0
                
            return await breaker.call(_delete_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error deleting agent: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("update_performance_metrics", "agents")
    @time_function("agent_update_metrics", {"operation": "update_performance_metrics", "table": "agents"})
    async def update_performance_metrics(self, agent_id: UUID, metrics: dict) -> bool:
        """Update agent performance metrics with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _update_operation():
                data = {"performance_metrics": json.dumps(metrics), "updated_at": datetime.now().isoformat()}
                result = self.client.table(self.table).update(data).eq("id", str(agent_id)).execute()
                return len(result.data) > 0
                
            return await breaker.call(_update_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error updating performance metrics: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Database error: {str(e)}")

    @track_database_operation("get_statistics", "agents")
    @time_function("agent_statistics", {"operation": "get_statistics", "table": "agents"})
    async def get_statistics(self) -> dict:
        """Get agent statistics with circuit breaker protection."""
        try:
            breaker = await self._get_circuit_breaker()
            
            async def _statistics_operation():
                total_result = self.client.table(self.table).select("id", count="exact").execute()
                total_count = total_result.count if total_result else 0
                
                type_counts = {}
                for agent_type in AgentType:
                    result = self.client.table(self.table).select("id", count="exact").eq("agent_type", agent_type.value).execute()
                    type_counts[agent_type.value] = result.count if result else 0
                    
                available_result = self.client.table(self.table).select("id", count="exact").eq("is_available", True).execute()
                available_count = available_result.count if available_result else 0
                
                return {
                    "total_agents": total_count,
                    "by_type": type_counts,
                    "available_agents": available_count,
                    "unavailable_agents": total_count - available_count
                }
                
            return await breaker.call(_statistics_operation)
            
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error getting statistics: {}", extra={"str_e_": str(e)})
            return {"error": str(e)}

    def _to_entity(self, data: Dict[str, Any]) -> Agent:
        """Convert database record to entity."""
        try:
            agent = Agent(
                name=data["name"],
                description=data["description"],
                agent_type=AgentType(data["agent_type"])
            )
            agent.id = UUID(data["id"])
            agent.capabilities = json.loads(data.get("capabilities", "[]"))
            agent.status = data.get("status", "active")
            agent.is_available = data.get("is_available", True)
            agent.performance_metrics = json.loads(data.get("performance_metrics", "{}"))
            agent.metadata = json.loads(data.get("metadata", "{}"))
            agent.created_at = datetime.fromisoformat(data["created_at"])
            agent.updated_at = datetime.fromisoformat(data["updated_at"])
            return agent
        except Exception as e:
            record_error(type(e).__name__, "agent_repository", "error")
            logger.error("Error converting data to entity: {}", extra={"str_e_": str(e)})
            raise InfrastructureException(f"Data conversion error: {str(e)}") 