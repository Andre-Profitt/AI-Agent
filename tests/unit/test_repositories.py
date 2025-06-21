from performance_dashboard import stats

from src.collaboration.realtime_collaboration import session_id
from src.core.entities.tool import search_tools
from src.core.use_cases.manage_session import saved_session
from src.core.use_cases.process_message import saved_message
from src.database_extended import deleted

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.tools.base_tool import ToolType
from requests import Session
from src.api.auth import User
from src.core.entities.message import Message
from src.gaia_components.adaptive_tool_system import Tool
from src.gaia_components.adaptive_tool_system import ToolType
from src.unified_architecture.communication import MessageType
from uuid import uuid4
# TODO: Fix undefined variables: active_sessions, agent_messages, all_users, calculator_tools, deleted, found_message, found_session, found_tool, found_user, m, message1, message2, message3, s, saved_message, saved_session, saved_tool, saved_user, search_tools, session1, session2, session3, session_id, session_messages, stats, t, tool1, tool2, tool3, u, user1, user2, user3, user_messages, uuid4
from src.database.models import Session

# TODO: Fix undefined variables: active_sessions, agent_messages, all_users, calculator_tools, deleted, found_message, found_session, found_tool, found_user, m, message1, message2, message3, s, saved_message, saved_session, saved_tool, saved_user, search_tools, session1, session2, session3, session_id, session_messages, stats, t, tool1, tool2, tool3, u, user1, user2, user3, user_messages

"""

from langchain.tools import Tool
from sqlalchemy.orm import Session
Unit tests for repository implementations
"""

import pytest

from src.infrastructure.database.in_memory_message_repository import InMemoryMessageRepository
from src.infrastructure.database.in_memory_session_repository import InMemorySessionRepository
from src.infrastructure.database.in_memory_tool_repository import InMemoryToolRepository
from src.infrastructure.database.in_memory_user_repository import InMemoryUserRepository

from src.core.entities.user import User

class TestInMemoryMessageRepository:
    """Test InMemoryMessageRepository"""

    @pytest.fixture
    def repository(self):
        return InMemoryMessageRepository()

    @pytest.fixture
    def sample_message(self):
        return Message(
            content="Test message",
            message_type=MessageType.USER,
            session_id=uuid4()
        )

    async def test_save_message(self, repository, sample_message):
        """Test saving a message"""
        saved_message = await repository.save(sample_message)

        assert saved_message.id == sample_message.id
        assert saved_message.content == "Test message"
        assert saved_message.message_type == MessageType.USER

    async def test_find_by_id(self, repository, sample_message):
        """Test finding message by ID"""
        await repository.save(sample_message)

        found_message = await repository.find_by_id(sample_message.id)

        assert found_message is not None
        assert found_message.id == sample_message.id
        assert found_message.content == sample_message.content

    async def test_find_by_id_not_found(self, repository):
        """Test finding message by non-existent ID"""
        found_message = await repository.find_by_id(uuid4())

        assert found_message is None

    async def test_find_by_session(self, repository):
        """Test finding messages by session ID"""
        session_id = uuid4()
        message1 = Message(content="Message 1", message_type=MessageType.USER, session_id=session_id)
        message2 = Message(content="Message 2", message_type=MessageType.AGENT, session_id=session_id)
        message3 = Message(content="Message 3", message_type=MessageType.USER, session_id=uuid4())

        await repository.save(message1)
        await repository.save(message2)
        await repository.save(message3)

        session_messages = await repository.find_by_session(session_id)

        assert len(session_messages) == 2
        assert any(m.id == message1.id for m in session_messages)
        assert any(m.id == message2.id for m in session_messages)
        assert not any(m.id == message3.id for m in session_messages)

    async def test_find_by_type(self, repository):
        """Test finding messages by type"""
        message1 = Message(content="User message", message_type=MessageType.USER, session_id=uuid4())
        message2 = Message(content="Agent message", message_type=MessageType.AGENT, session_id=uuid4())
        message3 = Message(content="Another user message", message_type=MessageType.USER, session_id=uuid4())

        await repository.save(message1)
        await repository.save(message2)
        await repository.save(message3)

        user_messages = await repository.find_by_type(MessageType.USER)
        agent_messages = await repository.find_by_type(MessageType.AGENT)

        assert len(user_messages) == 2
        assert len(agent_messages) == 1
        assert all(m.message_type == MessageType.USER for m in user_messages)
        assert all(m.message_type == MessageType.AGENT for m in agent_messages)

    async def test_delete_message(self, repository, sample_message):
        """Test deleting a message"""
        await repository.save(sample_message)

        # Verify message exists
        found_message = await repository.find_by_id(sample_message.id)
        assert found_message is not None

        # Delete message
        deleted = await repository.delete(sample_message.id)
        assert deleted is True

        # Verify message is gone
        found_message = await repository.find_by_id(sample_message.id)
        assert found_message is None

    async def test_delete_nonexistent_message(self, repository):
        """Test deleting a non-existent message"""
        deleted = await repository.delete(uuid4())
        assert deleted is False

    async def test_get_statistics(self, repository):
        """Test getting repository statistics"""
        # Add some test messages
        message1 = Message(content="User message", message_type=MessageType.USER, session_id=uuid4())
        message2 = Message(content="Agent message", message_type=MessageType.AGENT, session_id=uuid4())
        message3 = Message(content="Another user message", message_type=MessageType.USER, session_id=uuid4())

        await repository.save(message1)
        await repository.save(message2)
        await repository.save(message3)

        stats = await repository.get_statistics()

        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 2
        assert stats["agent_messages"] == 1

class TestInMemorySessionRepository:
    """Test InMemorySessionRepository"""

    @pytest.fixture
    def repository(self):
        return InMemorySessionRepository()

    @pytest.fixture
    def sample_session(self):
        return Session(
            user_id=uuid4(),
            title="Test Session"
        )

    async def test_save_session(self, repository, sample_session):
        """Test saving a session"""
        saved_session = await repository.save(sample_session)

        assert saved_session.id == sample_session.id
        assert saved_session.title == "Test Session"
        assert saved_session.is_active is True

    async def test_find_by_id(self, repository, sample_session):
        """Test finding session by ID"""
        await repository.save(sample_session)

        found_session = await repository.find_by_id(sample_session.id)

        assert found_session is not None
        assert found_session.id == sample_session.id
        assert found_session.title == sample_session.title

    async def test_find_by_id_not_found(self, repository):
        """Test finding session by non-existent ID"""
        found_session = await repository.find_by_id(uuid4())

        assert found_session is None

    async def test_find_active(self, repository):
        """Test finding active sessions"""
        session1 = Session(user_id=uuid4(), title="Active Session 1")
        session2 = Session(user_id=uuid4(), title="Active Session 2")
        session3 = Session(user_id=uuid4(), title="Closed Session")
        session3.close()

        await repository.save(session1)
        await repository.save(session2)
        await repository.save(session3)

        active_sessions = await repository.find_active()

        assert len(active_sessions) == 2
        assert all(s.is_active for s in active_sessions)
        assert not any(s.id == session3.id for s in active_sessions)

    async def test_delete_session(self, repository, sample_session):
        """Test deleting a session"""
        await repository.save(sample_session)

        # Verify session exists
        found_session = await repository.find_by_id(sample_session.id)
        assert found_session is not None

        # Delete session
        deleted = await repository.delete(sample_session.id)
        assert deleted is True

        # Verify session is gone
        found_session = await repository.find_by_id(sample_session.id)
        assert found_session is None

    async def test_get_statistics(self, repository):
        """Test getting repository statistics"""
        session1 = Session(user_id=uuid4(), title="Active Session 1")
        session2 = Session(user_id=uuid4(), title="Active Session 2")
        session3 = Session(user_id=uuid4(), title="Closed Session")
        session3.close()

        await repository.save(session1)
        await repository.save(session2)
        await repository.save(session3)

        stats = await repository.get_statistics()

        assert stats["total_sessions"] == 3
        assert stats["active_sessions"] == 2

class TestInMemoryToolRepository:
    """Test InMemoryToolRepository"""

    @pytest.fixture
    def repository(self):
        return InMemoryToolRepository()

    @pytest.fixture
    def sample_tool(self):
        return Tool(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.SEARCH,
            parameters={"query": "string"}
        )

    async def test_save_tool(self, repository, sample_tool):
        """Test saving a tool"""
        saved_tool = await repository.save(sample_tool)

        assert saved_tool.id == sample_tool.id
        assert saved_tool.name == "test_tool"
        assert saved_tool.tool_type == ToolType.SEARCH

    async def test_find_by_id(self, repository, sample_tool):
        """Test finding tool by ID"""
        await repository.save(sample_tool)

        found_tool = await repository.find_by_id(sample_tool.id)

        assert found_tool is not None
        assert found_tool.id == sample_tool.id
        assert found_tool.name == sample_tool.name

    async def test_find_by_name(self, repository, sample_tool):
        """Test finding tool by name"""
        await repository.save(sample_tool)

        found_tool = await repository.find_by_name("test_tool")

        assert found_tool is not None
        assert found_tool.name == "test_tool"
        assert found_tool.id == sample_tool.id

    async def test_find_by_name_not_found(self, repository):
        """Test finding tool by non-existent name"""
        found_tool = await repository.find_by_name("nonexistent_tool")

        assert found_tool is None

    async def test_find_by_type(self, repository):
        """Test finding tools by type"""
        tool1 = Tool(name="search_tool", description="Search tool", tool_type=ToolType.SEARCH, parameters={})
        tool2 = Tool(name="calculator_tool", description="Calculator tool", tool_type=ToolType.CALCULATOR, parameters={})
        tool3 = Tool(name="another_search", description="Another search tool", tool_type=ToolType.SEARCH, parameters={})

        await repository.save(tool1)
        await repository.save(tool2)
        await repository.save(tool3)

        search_tools = await repository.find_by_type(ToolType.SEARCH)
        calculator_tools = await repository.find_by_type(ToolType.CALCULATOR)

        assert len(search_tools) == 2
        assert len(calculator_tools) == 1
        assert all(t.tool_type == ToolType.SEARCH for t in search_tools)
        assert all(t.tool_type == ToolType.CALCULATOR for t in calculator_tools)

    async def test_delete_tool(self, repository, sample_tool):
        """Test deleting a tool"""
        await repository.save(sample_tool)

        # Verify tool exists
        found_tool = await repository.find_by_id(sample_tool.id)
        assert found_tool is not None

        # Delete tool
        deleted = await repository.delete(sample_tool.id)
        assert deleted is True

        # Verify tool is gone
        found_tool = await repository.find_by_id(sample_tool.id)
        assert found_tool is None

        # Verify tool is also gone from name index
        found_tool = await repository.find_by_name("test_tool")
        assert found_tool is None

    async def test_get_statistics(self, repository):
        """Test getting repository statistics"""
        tool1 = Tool(name="enabled_tool", description="Enabled tool", tool_type=ToolType.SEARCH, parameters={})
        tool2 = Tool(name="disabled_tool", description="Disabled tool", tool_type=ToolType.CALCULATOR, parameters={})
        tool2.disable()

        await repository.save(tool1)
        await repository.save(tool2)

        stats = await repository.get_statistics()

        assert stats["total_tools"] == 2
        assert stats["enabled_tools"] == 1

class TestInMemoryUserRepository:
    """Test InMemoryUserRepository"""

    @pytest.fixture
    def repository(self):
        return InMemoryUserRepository()

    @pytest.fixture
    def sample_user(self):
        return User(
            username="testuser",
            email="test@example.com"
        )

    async def test_save_user(self, repository, sample_user):
        """Test saving a user"""
        saved_user = await repository.save(sample_user)

        assert saved_user.id == sample_user.id
        assert saved_user.username == "testuser"
        assert saved_user.email == "test@example.com"

    async def test_find_by_id(self, repository, sample_user):
        """Test finding user by ID"""
        await repository.save(sample_user)

        found_user = await repository.find_by_id(sample_user.id)

        assert found_user is not None
        assert found_user.id == sample_user.id
        assert found_user.username == sample_user.username

    async def test_find_by_email(self, repository, sample_user):
        """Test finding user by email"""
        await repository.save(sample_user)

        found_user = await repository.find_by_email("test@example.com")

        assert found_user is not None
        assert found_user.email == "test@example.com"
        assert found_user.id == sample_user.id

    async def test_find_by_email_not_found(self, repository):
        """Test finding user by non-existent email"""
        found_user = await repository.find_by_email("nonexistent@example.com")

        assert found_user is None

    async def test_find_all(self, repository):
        """Test finding all users"""
        user1 = User(username="user1", email="user1@example.com")
        user2 = User(username="user2", email="user2@example.com")
        user3 = User(username="user3", email="user3@example.com")

        await repository.save(user1)
        await repository.save(user2)
        await repository.save(user3)

        all_users = await repository.find_all()

        assert len(all_users) == 3
        assert all(isinstance(u, User) for u in all_users)

    async def test_delete_user(self, repository, sample_user):
        """Test deleting a user"""
        await repository.save(sample_user)

        # Verify user exists
        found_user = await repository.find_by_id(sample_user.id)
        assert found_user is not None

        # Delete user
        deleted = await repository.delete(sample_user.id)
        assert deleted is True

        # Verify user is gone
        found_user = await repository.find_by_id(sample_user.id)
        assert found_user is None

        # Verify user is also gone from email index
        found_user = await repository.find_by_email("test@example.com")
        assert found_user is None

    async def test_get_statistics(self, repository):
        """Test getting repository statistics"""
        user1 = User(username="user1", email="user1@example.com")
        user2 = User(username="user2", email="user2@example.com")
        user3 = User(username="user3", email=None)  # No email

        await repository.save(user1)
        await repository.save(user2)
        await repository.save(user3)

        stats = await repository.get_statistics()

        assert stats["total_users"] == 3
        assert stats["users_with_email"] == 2