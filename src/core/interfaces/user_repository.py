"""
from abc import abstractmethod
from src.api.auth import User
# TODO: Fix undefined variables: ABC, List, Optional, UUID, abstractmethod

User repository interface defining the contract for user persistence.
"""

from typing import Optional

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from src.core.entities.user import User

class UserRepository(ABC):
    """
    Abstract interface for user persistence operations.
    """
    @abstractmethod
    async def save(self, user: User) -> User:
        pass

    @abstractmethod
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        pass

    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    async def find_all(self) -> List[User]:
        pass

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        pass

    @abstractmethod
    async def get_statistics(self) -> dict:
        pass
