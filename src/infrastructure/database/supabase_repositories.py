"""
Supabase implementations of repository interfaces.
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

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Singleton Supabase client."""
    _instance: Optional[Client] = None

    @classmethod
    def get_client(cls, url: str, key: str) -> Client:
        """Get or create Supabase client."""
        if cls._instance is None:
            cls._instance = create_client(url, key)
        return cls._instance

class SupabaseMessageRepository(MessageRepository):
    """Supabase implementation of message repository."""
    def __init__(self, client: Client):
        self.client = client
        self.table = "messages"

    async def save(self, message: Message) -> Message:
        """Save message to database."""
        try:
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
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_id(self, message_id: UUID) -> Optional[Message]:
        """Find message by ID."""
        try:
            result = self.client.table(self.table).select("*").eq("id", str(message_id)).execute()
            if result.data and len(result.data) > 0:
                return self._to_entity(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error finding message: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_session(self, session_id: UUID) -> List[Message]:
        """Find messages by session."""
        try:
            result = self.client.table(self.table).select("*").eq("session_id", str(session_id)).order("created_at").execute()
            return [self._to_entity(data) for data in result.data]
        except Exception as e:
            logger.error(f"Error finding messages by session: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_type(self, message_type: MessageType) -> List[Message]:
        """Find messages by type."""
        try:
            result = self.client.table(self.table).select("*").eq("message_type", message_type.value).execute()
            return [self._to_entity(data) for data in result.data]
        except Exception as e:
            logger.error(f"Error finding messages by type: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def delete(self, message_id: UUID) -> bool:
        """Delete message."""
        try:
            result = self.client.table(self.table).delete().eq("id", str(message_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error deleting message: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def get_statistics(self) -> dict:
        """Get message statistics."""
        try:
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
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

    def _to_entity(self, data: Dict[str, Any]) -> Message:
        """Convert database record to entity."""
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

class SupabaseToolRepository(ToolRepository):
    """Supabase implementation of tool repository."""
    def __init__(self, client: Client):
        self.client = client
        self.table = "tools"
        self.metrics_table = "tool_reliability_metrics"

    async def save(self, tool: Tool) -> Tool:
        """Save tool to database."""
        try:
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
        except Exception as e:
            logger.error(f"Error saving tool: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_id(self, tool_id: UUID) -> Optional[Tool]:
        """Find tool by ID."""
        try:
            result = self.client.table(self.table).select("*").eq("id", str(tool_id)).execute()
            if result.data and len(result.data) > 0:
                tool = self._to_entity(result.data[0])
                await self._load_metrics(tool)
                return tool
            return None
        except Exception as e:
            logger.error(f"Error finding tool: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_name(self, name: str) -> Optional[Tool]:
        """Find tool by name."""
        try:
            result = self.client.table(self.table).select("*").eq("name", name).execute()
            if result.data and len(result.data) > 0:
                tool = self._to_entity(result.data[0])
                await self._load_metrics(tool)
                return tool
            return None
        except Exception as e:
            logger.error(f"Error finding tool by name: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_type(self, tool_type: ToolType) -> List[Tool]:
        """Find tools by type."""
        try:
            result = self.client.table(self.table).select("*").eq("tool_type", tool_type.value).execute()
            tools = [self._to_entity(data) for data in result.data]
            for tool in tools:
                await self._load_metrics(tool)
            return tools
        except Exception as e:
            logger.error(f"Error finding tools by type: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def delete(self, tool_id: UUID) -> bool:
        """Delete tool."""
        try:
            result = self.client.table(self.table).delete().eq("id", str(tool_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error deleting tool: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def get_statistics(self) -> dict:
        """Get tool statistics."""
        try:
            total_result = self.client.table(self.table).select("id", count="exact").execute()
            enabled_result = self.client.table(self.table).select("id", count="exact").eq("is_enabled", True).execute()
            type_counts = {}
            for tool_type in ToolType:
                result = self.client.table(self.table).select("id", count="exact").eq("tool_type", tool_type.value).execute()
                type_counts[tool_type.value] = result.count if result else 0
            metrics_result = self.client.table(self.metrics_table).select("*").execute()
            total_calls = sum(m.get("total_calls", 0) for m in metrics_result.data)
            total_success = sum(m.get("success_count", 0) for m in metrics_result.data)
            avg_latency = sum(m.get("average_latency_ms", 0) for m in metrics_result.data) / max(1, len(metrics_result.data))
            return {
                "total_tools": total_result.count if total_result else 0,
                "enabled_tools": enabled_result.count if enabled_result else 0,
                "by_type": type_counts,
                "reliability": {
                    "total_calls": total_calls,
                    "success_rate": total_success / max(1, total_calls),
                    "avg_latency_ms": avg_latency
                }
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

    async def _update_metrics(self, tool: Tool) -> None:
        """Update tool metrics in database."""
        try:
            data = {
                "tool_name": tool.name,
                "success_count": tool.successful_executions,
                "failure_count": tool.failed_executions,
                "total_calls": tool.total_executions,
                "average_latency_ms": tool.average_execution_time * 1000,
                "last_used_at": tool.last_executed_at.isoformat() if tool.last_executed_at else None
            }
            self.client.table(self.metrics_table).upsert(data).execute()
        except Exception as e:
            logger.error(f"Error updating tool metrics: {str(e)}")

    async def _load_metrics(self, tool: Tool) -> None:
        """Load tool metrics from database."""
        try:
            result = self.client.table(self.metrics_table).select("*").eq("tool_name", tool.name).execute()
            if result.data and len(result.data) > 0:
                metrics = result.data[0]
                tool.successful_executions = metrics.get("success_count", 0)
                tool.failed_executions = metrics.get("failure_count", 0)
                tool.total_executions = metrics.get("total_calls", 0)
                tool.average_execution_time = metrics.get("average_latency_ms", 0) / 1000
                if metrics.get("last_used_at"):
                    tool.last_executed_at = datetime.fromisoformat(metrics["last_used_at"])
        except Exception as e:
            logger.error(f"Error loading tool metrics: {str(e)}")

    def _to_entity(self, data: Dict[str, Any]) -> Tool:
        """Convert database record to entity."""
        tool = Tool(
            name=data["name"],
            description=data["description"],
            tool_type=ToolType(data["tool_type"]),
            function=lambda **kwargs: None  # Placeholder
        )
        tool.id = UUID(data["id"])
        tool.parameters_schema = json.loads(data.get("parameters_schema", "{}"))
        tool.return_schema = json.loads(data.get("return_schema", "{}"))
        tool.status = ToolStatus(data["status"])
        tool.is_enabled = data["is_enabled"]
        tool.is_public = data["is_public"]
        tool.version = data.get("version", "1.0.0")
        tool.author = data.get("author")
        tool.tags = data.get("tags", [])
        tool.documentation = data.get("documentation")
        tool.created_at = datetime.fromisoformat(data["created_at"])
        tool.updated_at = datetime.fromisoformat(data["updated_at"])
        return tool

class SupabaseSessionRepository(SessionRepository):
    """Supabase implementation of session repository."""
    def __init__(self, client: Client):
        self.client = client
        self.table = "user_sessions"

    async def save(self, session: Session) -> Session:
        """Save session to database."""
        try:
            data = {
                "session_id": str(session.id),
                "user_id": str(session.user_id) if session.user_id else None,
                "title": session.title,
                "status": session.status.value,
                "total_queries": session.total_messages,
                "successful_queries": session.successful_messages,
                "failed_queries": session.failed_messages,
                "conversation_history": json.dumps([msg.to_dict() for msg in session.messages]),
                "metadata": json.dumps(session.metadata),
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "last_active_at": session.last_message_at.isoformat() if session.last_message_at else None
            }
            result = self.client.table(self.table).upsert(data).execute()
            if result.data:
                return session
            else:
                raise InfrastructureException("Failed to save session")
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_id(self, session_id: UUID) -> Optional[Session]:
        """Find session by ID."""
        try:
            result = self.client.table(self.table).select("*").eq("session_id", str(session_id)).execute()
            if result.data and len(result.data) > 0:
                return self._to_entity(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error finding session: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_active(self) -> List[Session]:
        """Find active sessions."""
        try:
            result = self.client.table(self.table).select("*").eq("status", SessionStatus.ACTIVE.value).execute()
            return [self._to_entity(data) for data in result.data]
        except Exception as e:
            logger.error(f"Error finding active sessions: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def delete(self, session_id: UUID) -> bool:
        """Delete session."""
        try:
            result = self.client.table(self.table).delete().eq("session_id", str(session_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def get_statistics(self) -> dict:
        """Get session statistics."""
        try:
            total_result = self.client.table(self.table).select("session_id", count="exact").execute()
            active_result = self.client.table(self.table).select("session_id", count="exact").eq("status", SessionStatus.ACTIVE.value).execute()
            stats_result = self.client.table(self.table).select("total_queries", "successful_queries", "failed_queries").execute()
            total_queries = sum(s.get("total_queries", 0) for s in stats_result.data)
            successful_queries = sum(s.get("successful_queries", 0) for s in stats_result.data)
            failed_queries = sum(s.get("failed_queries", 0) for s in stats_result.data)
            return {
                "total_sessions": total_result.count if total_result else 0,
                "active_sessions": active_result.count if active_result else 0,
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": successful_queries / max(1, total_queries)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

    def _to_entity(self, data: Dict[str, Any]) -> Session:
        """Convert database record to entity."""
        session = Session(
            user_id=UUID(data["user_id"]) if data.get("user_id") else None,
            title=data.get("title")
        )
        session.id = UUID(data["session_id"])
        session.status = SessionStatus(data["status"])
        session.total_messages = data.get("total_queries", 0)
        session.successful_messages = data.get("successful_queries", 0)
        session.failed_messages = data.get("failed_queries", 0)
        session.metadata = json.loads(data.get("metadata", "{}"))
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("last_active_at"):
            session.last_message_at = datetime.fromisoformat(data["last_active_at"])
        return session

class SupabaseAgentRepository(AgentRepository):
    """Supabase implementation of agent repository."""
    def __init__(self, client: Client):
        self.client = client
        self.table = "agents"

    async def save(self, agent: Agent) -> Agent:
        """Save agent to database."""
        try:
            data = {
                "id": str(agent.id),
                "name": agent.name,
                "agent_type": agent.agent_type.value,
                "state": agent.state.value,
                "current_task": agent.current_task,
                "total_requests": agent.total_requests,
                "successful_requests": agent.successful_requests,
                "failed_requests": agent.failed_requests,
                "average_response_time": agent.average_response_time,
                "created_at": agent.created_at.isoformat(),
                "last_active": agent.last_active.isoformat()
            }
            result = self.client.table(self.table).upsert(data).execute()
            if result.data:
                return agent
            else:
                raise InfrastructureException("Failed to save agent")
        except Exception as e:
            logger.error(f"Error saving agent: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_id(self, agent_id: UUID) -> Optional[Agent]:
        """Find agent by ID."""
        try:
            result = self.client.table(self.table).select("*").eq("id", str(agent_id)).execute()
            if result.data and len(result.data) > 0:
                return self._to_entity(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error finding agent: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Find agents by type."""
        try:
            result = self.client.table(self.table).select("*").eq("agent_type", agent_type.value).execute()
            return [self._to_entity(data) for data in result.data]
        except Exception as e:
            logger.error(f"Error finding agents by type: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def find_available(self) -> List[Agent]:
        """Find available agents."""
        try:
            result = self.client.table(self.table).select("*").in_("state", [AgentState.IDLE.value, AgentState.COMPLETED.value]).execute()
            return [self._to_entity(data) for data in result.data]
        except Exception as e:
            logger.error(f"Error finding available agents: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def delete(self, agent_id: UUID) -> bool:
        """Delete agent."""
        try:
            result = self.client.table(self.table).delete().eq("id", str(agent_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error deleting agent: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def update_performance_metrics(self, agent_id: UUID, metrics: dict) -> bool:
        """Update agent performance metrics."""
        try:
            result = self.client.table(self.table).update(metrics).eq("id", str(agent_id)).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise InfrastructureException(f"Database error: {str(e)}")

    async def get_statistics(self) -> dict:
        """Get agent statistics."""
        try:
            total_result = self.client.table(self.table).select("id", count="exact").execute()
            type_counts = {}
            for agent_type in AgentType:
                result = self.client.table(self.table).select("id", count="exact").eq("agent_type", agent_type.value).execute()
                type_counts[agent_type.value] = result.count if result else 0
            state_counts = {}
            for state in AgentState:
                result = self.client.table(self.table).select("id", count="exact").eq("state", state.value).execute()
                state_counts[state.value] = result.count if result else 0
            return {
                "total_agents": total_result.count if total_result else 0,
                "by_type": type_counts,
                "by_state": state_counts
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

    def _to_entity(self, data: Dict[str, Any]) -> Agent:
        """Convert database record to entity."""
        agent = Agent(
            name=data["name"],
            agent_type=AgentType(data["agent_type"])
        )
        agent.id = UUID(data["id"])
        agent.state = AgentState(data["state"])
        agent.current_task = data.get("current_task")
        agent.total_requests = data.get("total_requests", 0)
        agent.successful_requests = data.get("successful_requests", 0)
        agent.failed_requests = data.get("failed_requests", 0)
        agent.average_response_time = data.get("average_response_time", 0.0)
        agent.created_at = datetime.fromisoformat(data["created_at"])
        agent.last_active = datetime.fromisoformat(data["last_active"])
        return agent 