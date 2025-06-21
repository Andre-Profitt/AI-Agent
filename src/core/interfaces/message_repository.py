"""
from abc import abstractmethod
from src.unified_architecture.communication import MessageType
# TODO: Fix undefined variables: ABC, List, Optional, UUID, abstractmethod

Message repository interface defining the contract for message persistence.
"""

from typing import Optional

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from src.core.entities.message import Message, MessageType

class MessageRepository(ABC):
    """
    Abstract interface for message persistence operations.
    """
    @abstractmethod
    async def save(self, message: Message) -> Message:
        pass

    @abstractmethod
    async def find_by_id(self, message_id: UUID) -> Optional[Message]:
        pass

    @abstractmethod
    async def find_by_session(self, session_id: UUID) -> List[Message]:
        pass

    @abstractmethod
    async def find_by_type(self, message_type: MessageType) -> List[Message]:
        pass

    @abstractmethod
    async def delete(self, message_id: UUID) -> bool:
        pass

    @abstractmethod
    async def get_statistics(self) -> dict:
        pass
