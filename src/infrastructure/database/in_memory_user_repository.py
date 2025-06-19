"""
In-memory implementation of the UserRepository interface.
"""

from typing import List, Optional, Dict
from uuid import UUID
import asyncio

from src.core.entities.user import User
from src.core.interfaces.user_repository import UserRepository

class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self._users: Dict[UUID, User] = {}
        self._users_by_email: Dict[str, User] = {}

    async def save(self, user: User) -> User:
        self._users[user.id] = user
        if user.email:
            self._users_by_email[user.email] = user
        return user

    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        return self._users.get(user_id)

    async def find_by_email(self, email: str) -> Optional[User]:
        return self._users_by_email.get(email)

    async def find_all(self) -> List[User]:
        return list(self._users.values())

    async def delete(self, user_id: UUID) -> bool:
        user = self._users.get(user_id)
        if user:
            del self._users[user_id]
            if user.email in self._users_by_email:
                del self._users_by_email[user.email]
            return True
        return False

    async def get_statistics(self) -> dict:
        return {
            "total_users": len(self._users),
            "users_with_email": len(self._users_by_email),
        } 