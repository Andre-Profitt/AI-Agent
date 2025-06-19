"""
Session repository interface defining the contract for session persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from src.core.entities.session import Session

class SessionRepository(ABC):
    """
    Abstract interface for session persistence operations.
    """
    @abstractmethod
    async def save(self, session: Session) -> Session:
        pass

    @abstractmethod
    async def find_by_id(self, session_id: UUID) -> Optional[Session]:
        pass

    @abstractmethod
    async def find_active(self) -> List[Session]:
        pass

    @abstractmethod
    async def delete(self, session_id: UUID) -> bool:
        pass

    @abstractmethod
    async def get_statistics(self) -> dict:
        pass 