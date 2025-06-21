from src.collaboration.realtime_collaboration import session
from src.collaboration.realtime_collaboration import session_id

"""
from typing import Optional
from requests import Session
# TODO: Fix undefined variables: Dict, List, Optional, UUID, s, session, session_id
from src.database.models import Session

# TODO: Fix undefined variables: s, self, session, session_id

from sqlalchemy.orm import Session
In-memory implementation of the SessionRepository interface.
"""

from typing import Dict

from typing import List, Optional, Dict
from uuid import UUID

from src.core.entities.session import Session
from src.core.interfaces.session_repository import SessionRepository

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