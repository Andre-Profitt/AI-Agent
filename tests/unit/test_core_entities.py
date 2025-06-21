from tests.load_test import data
from tests.load_test import user
from tests.performance.performance_test import agent_data

from src.api_server import message
from src.collaboration.realtime_collaboration import session
from src.collaboration.realtime_collaboration import session_id
from src.core.entities.agent import Agent
from src.database.models import tool
from src.database.models import user_id

from src.agents.advanced_agent_fsm import AgentStatus

from src.agents.advanced_agent_fsm import AgentType

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.tools.base_tool import ToolType
from requests import Session
from src.api.auth import User
from src.collaboration.realtime_collaboration import SessionStatus
from src.core.entities.message import MessageStatus
from src.core.entities.tool import ToolResult
from src.gaia_components.adaptive_tool_system import Tool
from src.gaia_components.adaptive_tool_system import ToolStatus
from src.gaia_components.adaptive_tool_system import ToolType
from src.gaia_components.multi_agent_orchestrator import Agent
from src.infrastructure.agents.agent_factory import AgentType
from src.unified_architecture.communication import MessageType
from src.unified_architecture.enhanced_platform import AgentStatus
from uuid import UUID
# TODO: Fix undefined variables: UUID, agent_data, data, datetime, message, message_data, result, session, session_id, timezone, user, user_id, uuid4
from tests.test_gaia_agent import agent

from src.core.entities.agent import AgentType
from src.database.models import Session
from src.tools.base_tool import tool

# TODO: Fix undefined variables: agent, agent_data, data, message, message_data, result, session, session_id, timezone, tool, user, user_id

"""

from datetime import timezone
from langchain.tools import Tool
from sqlalchemy.orm import Session
Unit tests for core entities
"""

from uuid import uuid4, UUID
from datetime import datetime, timezone

from src.core.entities.agent import Agent, AgentType, AgentStatus
from src.core.entities.message import Message, MessageType, MessageStatus
from src.core.entities.tool import Tool, ToolType, ToolStatus, ToolResult
from src.core.entities.session import Session, SessionStatus
from src.core.entities.user import User

class TestAgent:
    """Test Agent entity"""

    def test_agent_creation(self):
        """Test basic agent creation"""
        agent = Agent(
            name="Test Agent",
            agent_type=AgentType.REASONING,
            description="A test agent"
        )

        assert agent.name == "Test Agent"
        assert agent.agent_type == AgentType.REASONING
        assert agent.description == "A test agent"
        assert agent.status == AgentStatus.ACTIVE
        assert isinstance(agent.id, UUID)
        assert isinstance(agent.created_at, datetime)
        assert isinstance(agent.updated_at, datetime)

    def test_agent_serialization(self):
        """Test agent serialization to dict"""
        agent = Agent(
            name="Test Agent",
            agent_type=AgentType.REASONING,
            description="A test agent"
        )

        data = agent.to_dict()

        assert data["name"] == "Test Agent"
        assert data["agent_type"] == AgentType.REASONING.value
        assert data["description"] == "A test agent"
        assert data["status"] == AgentStatus.ACTIVE.value
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_agent_from_dict(self):
        """Test agent creation from dict"""
        agent_data = {
            "id": str(uuid4()),
            "name": "Test Agent",
            "agent_type": AgentType.REASONING.value,
            "description": "A test agent",
            "status": AgentStatus.ACTIVE.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        agent = Agent.from_dict(agent_data)

        assert agent.name == "Test Agent"
        assert agent.agent_type == AgentType.REASONING
        assert agent.description == "A test agent"
        assert agent.status == AgentStatus.ACTIVE

class TestMessage:
    """Test Message entity"""

    def test_message_creation(self):
        """Test basic message creation"""
        session_id = uuid4()
        message = Message(
            content="Hello, world!",
            message_type=MessageType.USER,
            session_id=session_id
        )

        assert message.content == "Hello, world!"
        assert message.message_type == MessageType.USER
        assert message.session_id == session_id
        assert message.status == MessageStatus.PENDING
        assert isinstance(message.id, UUID)
        assert isinstance(message.created_at, datetime)

    def test_message_serialization(self):
        """Test message serialization to dict"""
        session_id = uuid4()
        message = Message(
            content="Hello, world!",
            message_type=MessageType.USER,
            session_id=session_id
        )

        data = message.to_dict()

        assert data["content"] == "Hello, world!"
        assert data["message_type"] == MessageType.USER.value
        assert data["session_id"] == str(session_id)
        assert data["status"] == MessageStatus.PENDING.value
        assert "id" in data
        assert "created_at" in data

    def test_message_from_dict(self):
        """Test message creation from dict"""
        session_id = uuid4()
        message_data = {
            "id": str(uuid4()),
            "content": "Hello, world!",
            "message_type": MessageType.USER.value,
            "session_id": str(session_id),
            "status": MessageStatus.PENDING.value,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        message = Message.from_dict(message_data)

        assert message.content == "Hello, world!"
        assert message.message_type == MessageType.USER
        assert message.session_id == session_id
        assert message.status == MessageStatus.PENDING

class TestTool:
    """Test Tool entity"""

    def test_tool_creation(self):
        """Test basic tool creation"""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.SEARCH,
            parameters={"query": "string"}
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.tool_type == ToolType.SEARCH
        assert tool.parameters == {"query": "string"}
        assert tool.status == ToolStatus.ENABLED
        assert isinstance(tool.id, UUID)
        assert isinstance(tool.created_at, datetime)

    def test_tool_execution(self):
        """Test tool execution"""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.SEARCH,
            parameters={"query": "string"}
        )

        # Mock execution
        result = tool.execute({"query": "test query"})

        assert isinstance(result, ToolResult)
        assert result.success is False  # Default mock behavior
        assert result.output is None
        assert result.error_message is not None

    def test_tool_serialization(self):
        """Test tool serialization to dict"""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.SEARCH,
            parameters={"query": "string"}
        )

        data = tool.to_dict()

        assert data["name"] == "test_tool"
        assert data["description"] == "A test tool"
        assert data["tool_type"] == ToolType.SEARCH.value
        assert data["parameters"] == {"query": "string"}
        assert data["status"] == ToolStatus.ENABLED.value
        assert "id" in data
        assert "created_at" in data

class TestSession:
    """Test Session entity"""

    def test_session_creation(self):
        """Test basic session creation"""
        user_id = uuid4()
        session = Session(
            user_id=user_id,
            title="Test Session"
        )

        assert session.user_id == user_id
        assert session.title == "Test Session"
        assert session.status == SessionStatus.ACTIVE
        assert session.is_active is True
        assert isinstance(session.id, UUID)
        assert isinstance(session.created_at, datetime)

    def test_session_serialization(self):
        """Test session serialization to dict"""
        user_id = uuid4()
        session = Session(
            user_id=user_id,
            title="Test Session"
        )

        data = session.to_dict()

        assert data["user_id"] == str(user_id)
        assert data["title"] == "Test Session"
        assert data["status"] == SessionStatus.ACTIVE.value
        assert "id" in data
        assert "created_at" in data

    def test_session_closure(self):
        """Test session closure"""
        session = Session(
            user_id=uuid4(),
            title="Test Session"
        )

        assert session.is_active is True
        session.close()
        assert session.is_active is False
        assert session.status == SessionStatus.CLOSED

class TestUser:
    """Test User entity"""

    def test_user_creation(self):
        """Test basic user creation"""
        user = User(
            username="testuser",
            email="test@example.com"
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert isinstance(user.id, UUID)
        assert isinstance(user.created_at, datetime)

    def test_user_serialization(self):
        """Test user serialization to dict"""
        user = User(
            username="testuser",
            email="test@example.com"
        )

        data = user.to_dict()

        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data

    def test_user_deactivation(self):
        """Test user deactivation"""
        user = User(
            username="testuser",
            email="test@example.com"
        )

        assert user.is_active is True
        user.deactivate()
        assert user.is_active is False