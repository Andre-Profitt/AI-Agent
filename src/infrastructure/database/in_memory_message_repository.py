from app import msg

from src.api_server import message
from src.collaboration.realtime_collaboration import session_id
from src.database.supabase_manager import message_id

"""
from typing import Optional
from src.unified_architecture.communication import MessageType
# TODO: Fix undefined variables: Dict, List, Optional, UUID, m, message, message_id, message_type, msg, session_id
# TODO: Fix undefined variables: m, message, message_id, message_type, msg, self, session_id

In-memory implementation of the MessageRepository interface.
"""

from typing import Dict

from typing import List, Optional, Dict
from uuid import UUID

from src.core.entities.message import Message, MessageType
from src.core.interfaces.message_repository import MessageRepository

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
