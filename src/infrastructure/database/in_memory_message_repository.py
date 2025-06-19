"""
In-memory implementation of the MessageRepository interface.
"""

from typing import List, Optional, Dict
from uuid import UUID
import asyncio

from src.core.entities.message import Message, MessageType
from src.core.interfaces.message_repository import MessageRepository
from typing import Optional, Dict, Any, List, Union, Tuple

class InMemoryMessageRepository(MessageRepository):
    def __init__(self) -> None:
        self._messages: Dict[UUID, Message] = {}

    async def save(self, message: Message) -> Message:
        self._messages[message.id] = message
        return message

    async def find_by_id(self, message_id: UUID) -> Optional[Message]:
        return self._messages.get(message_id)

    async def find_by_session(self, session_id: UUID) -> List[Message]:
        return [msg for msg in self._messages.values() if msg.session_id == session_id]

    async def find_by_type(self, message_type: MessageType) -> List[Message]:
        return [msg for msg in self._messages.values() if msg.message_type == message_type]

    async def delete(self, message_id: UUID) -> bool:
        if message_id in self._messages:
            del self._messages[message_id]
            return True
        return False

    async def get_statistics(self) -> dict:
        return {
            "total_messages": len(self._messages),
            "user_messages": len([m for m in self._messages.values() if m.message_type == MessageType.USER]),
            "agent_messages": len([m for m in self._messages.values() if m.message_type == MessageType.AGENT]),
        } 