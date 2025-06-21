from app import msg
from setup_environment import value

from src.api_server import message
from src.core.monitoring import key
from src.infrastructure.events.event_bus import processing_time
from src.tools_introspection import error_message
from src.utils.tools_introspection import field

from src.agents.advanced_agent_fsm import Agent
from uuid import uuid4
# TODO: Fix undefined variables: Any, Dict, Enum, List, Optional, UUID, dataclass, datetime, error_message, field, key, message, message_type, msg, processing_time, tag, uuid4, value
# TODO: Fix undefined variables: error_message, key, message, message_type, msg, processing_time, self, tag, value

"""
Core Message entity representing communication in the AI Agent system.
"""

from typing import Optional
from dataclasses import field
from typing import Any
from typing import List

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID, uuid4

from src.shared.exceptions import ValidationException

class MessageType(str, Enum):
    """Types of messages in the system."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"

class MessageStatus(str, Enum):
    """Status of message processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Message:
    """
    Core Message entity representing a communication unit.

    This entity encapsulates all message-related business logic,
    including validation, status management, and metadata handling.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)

    # Content
    content: str = field(default="")
    message_type: MessageType = field(default=MessageType.USER)

    # Relationships
    session_id: Optional[UUID] = field(default=None)
    user_id: Optional[UUID] = field(default=None)
    agent_id: Optional[UUID] = field(default=None)
    parent_message_id: Optional[UUID] = field(default=None)

    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Status and processing
    status: MessageStatus = field(default=MessageStatus.PENDING)
    processing_time: Optional[float] = field(default=None)
    error_message: Optional[str] = field(default=None)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.content.strip():
            raise ValidationException("Message content cannot be empty")

        if len(self.content) > 10000:  # 10KB limit
            raise ValidationException("Message content too long (max 10KB)")

    def start_processing(self) -> None:
        """Mark message as being processed."""
        if self.status != MessageStatus.PENDING:
            raise ValidationException(f"Cannot start processing message in {self.status} status")

        self.status = MessageStatus.PROCESSING
        self.updated_at = datetime.now()

    def complete_processing(self, processing_time: float) -> None:
        """Mark message as completed."""
        if self.status != MessageStatus.PROCESSING:
            raise ValidationException(f"Cannot complete message in {self.status} status")

        self.status = MessageStatus.COMPLETED
        self.processing_time = processing_time
        self.updated_at = datetime.now()

    def fail_processing(self, error_message: str) -> None:
        """Mark message as failed."""
        if self.status not in [MessageStatus.PENDING, MessageStatus.PROCESSING]:
            raise ValidationException(f"Cannot fail message in {self.status} status")

        self.status = MessageStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now()

    def cancel_processing(self) -> None:
        """Cancel message processing."""
        if self.status not in [MessageStatus.PENDING, MessageStatus.PROCESSING]:
            raise ValidationException(f"Cannot cancel message in {self.status} status")

        self.status = MessageStatus.CANCELLED
        self.updated_at = datetime.now()

    def add_context(self, key: str, value: Any) -> None:
        """Add context information to the message."""
        self.context[key] = value
        self.updated_at = datetime.now()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the message."""
        self.metadata[key] = value
        self.updated_at = datetime.now()

    @property
    def is_user_message(self) -> bool:
        """Check if this is a user message."""
        return self.message_type == MessageType.USER

    @property
    def is_agent_message(self) -> bool:
        """Check if this is an agent message."""
        return self.message_type == MessageType.AGENT

    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message."""
        return self.message_type == MessageType.SYSTEM

    @property
    def is_tool_message(self) -> bool:
        """Check if this is a tool message."""
        return self.message_type == MessageType.TOOL

    @property
    def is_error_message(self) -> bool:
        """Check if this is an error message."""
        return self.message_type == MessageType.ERROR

    @property
    def is_processed(self) -> bool:
        """Check if message has been processed."""
        return self.status in [MessageStatus.COMPLETED, MessageStatus.FAILED, MessageStatus.CANCELLED]

    @property
    def is_pending(self) -> bool:
        """Check if message is pending processing."""
        return self.status == MessageStatus.PENDING

    @property
    def is_processing(self) -> bool:
        """Check if message is being processed."""
        return self.status == MessageStatus.PROCESSING

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": str(self.id),
            "content": self.content,
            "message_type": self.message_type.value,
            "session_id": str(self.session_id) if self.session_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "parent_message_id": str(self.parent_message_id) if self.parent_message_id else None,
            "context": self.context,
            "metadata": self.metadata,
            "status": self.status.value,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_user_message": self.is_user_message,
            "is_agent_message": self.is_agent_message,
            "is_system_message": self.is_system_message,
            "is_tool_message": self.is_tool_message,
            "is_error_message": self.is_error_message,
            "is_processed": self.is_processed,
            "is_pending": self.is_pending,
            "is_processing": self.is_processing
        }

@dataclass
class Conversation:
    """
    Conversation entity representing a sequence of related messages.

    This entity manages conversation context, threading, and metadata.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)

    # Messages
    messages: List[Message] = field(default_factory=list)

    # Metadata
    title: Optional[str] = field(default=None)
    summary: Optional[str] = field(default=None)
    tags: List[str] = field(default_factory=list)

    # Status
    is_active: bool = field(default=True)
    is_archived: bool = field(default=False)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_message_at: Optional[datetime] = field(default=None)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        if not isinstance(message, Message):
            raise ValidationException("Can only add Message objects to conversation")

        # Set conversation context
        message.session_id = self.session_id

        # Add to messages list
        self.messages.append(message)

        # Update timestamps
        self.updated_at = datetime.now()
        self.last_message_at = datetime.now()

        # Update title if this is the first message
        if len(self.messages) == 1 and not self.title:
            self.title = message.content[:100] + "..." if len(message.content) > 100 else message.content

    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """Get all messages of a specific type."""
        return [msg for msg in self.messages if msg.message_type == message_type]

    def get_user_messages(self) -> List[Message]:
        """Get all user messages."""
        return self.get_messages_by_type(MessageType.USER)

    def get_agent_messages(self) -> List[Message]:
        """Get all agent messages."""
        return self.get_messages_by_type(MessageType.AGENT)

    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)

    def archive(self) -> None:
        """Archive the conversation."""
        self.is_active = False
        self.is_archived = True
        self.updated_at = datetime.now()

    def reactivate(self) -> None:
        """Reactivate the conversation."""
        self.is_active = True
        self.is_archived = False
        self.updated_at = datetime.now()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the conversation."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the conversation."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary representation."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "messages": [msg.to_dict() for msg in self.messages],
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "message_count": self.get_message_count()
        }