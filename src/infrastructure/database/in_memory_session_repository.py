"""
In-memory implementation of the SessionRepository interface.
"""

from typing import List, Optional, Dict
from uuid import UUID
import asyncio

from src.core.entities.session import Session
from src.core.interfaces.session_repository import SessionRepository
from typing import Optional, Dict, Any, List, Union, Tuple

class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._sessions: Dict[UUID, Session] = {}

    async def save(self, session: Session) -> Session:
        self._sessions[session.id] = session
        return session

    async def find_by_id(self, session_id: UUID) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def find_active(self) -> List[Session]:
        return [s for s in self._sessions.values() if s.is_active]

    async def delete(self, session_id: UUID) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def get_statistics(self) -> dict:
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len([s for s in self._sessions.values() if s.is_active]),
        } 