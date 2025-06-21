from migrations.env import config
from tests.load_test import credentials
from tests.load_test import registration
from tests.load_test import token_response
from tests.load_test import user

from src.api.auth import access_token
from src.api.auth import admin_user
from src.api.auth import basic_user
from src.api.auth import hashed
from src.api.auth import hashed_password
from src.api.auth import new_refresh_token
from src.api.auth import payload
from src.api.auth import premium_user
from src.api.auth import refresh_token
from src.api.auth import salt
from src.api.auth import secret_key
from src.api_server import token
from src.database.models import role
from src.database.models import user_id
from src.database.models import username
from src.infrastructure.config.configuration_service import secrets

from datetime import timedelta
# TODO: Fix undefined variables: Any, Dict, Enum, List, Optional, access_token, admin_user, authorization, basic_user, config, credentials, dataclass, datetime, e, hashed, hashed_password, logging, new_refresh_token, new_role, os, password, payload, perm, permission, premium_user, refresh_token, registration, role, salt_rounds, secret_key, timedelta, token, token_response, u, user, user_id, username
import salt

from pydantic import Field

# TODO: Fix undefined variables: BaseModel, Field, access_token, admin_user, authorization, basic_user, bcrypt, config, credentials, e, hashed, hashed_password, jwt, new_refresh_token, new_role, password, payload, perm, permission, premium_user, refresh_token, registration, role, salt, salt_rounds, secret_key, secrets, self, token, token_response, u, user, user_id, username
"""
Authentication System for GAIA API
JWT-based authentication with role-based access control
"""

from typing import Dict
from typing import Any
from typing import List

import jwt
import bcrypt

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
from pydantic import BaseModel, Field
import secrets

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for access control"""
    ANONYMOUS = "anonymous"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    ENTERPRISE = "enterprise"

class Permission(Enum):
    """System permissions"""
    READ_QUERIES = "read_queries"
    WRITE_QUERIES = "write_queries"
    EXECUTE_TOOLS = "execute_tools"
    ACCESS_ADVANCED_FEATURES = "access_advanced_features"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    ADMIN_ACCESS = "admin_access"

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class UserCredentials(BaseModel):
    """User login credentials"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)

class UserRegistration(BaseModel):
    """User registration data"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.USER

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    role: str

class AuthConfig:
    """Authentication configuration"""
    def __init__(self):
        self.secret_key = self._get_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.password_salt_rounds = 12

    def _get_secret_key(self) -> str:
        """Get secret key from environment or generate one"""
        import os
        secret_key = os.getenv("JWT_SECRET_KEY")
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
            logger.warning("JWT_SECRET_KEY not set, using generated key")
        return secret_key

class PasswordManager:
    """Password hashing and verification"""

    @staticmethod
    def hash_password(self, password: str, salt_rounds: int = 12) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt(salt_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

class TokenManager:
    """JWT token management"""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.blacklisted_tokens = set()

    def create_access_token(self, user: User) -> str:
        """Create access token for user"""
        payload = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes),
            "iat": datetime.utcnow()
        }

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def create_refresh_token(self, user: User) -> str:
        """Create refresh token for user"""
        payload = {
            "sub": user.id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days),
            "iat": datetime.utcnow()
        }

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])

            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise jwt.InvalidTokenError("Token has been blacklisted")

            return payload
        except jwt.ExpiredSignatureError:
            raise jwt.ExpiredSignatureError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f"Invalid token: {e}")

    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)

    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        return token in self.blacklisted_tokens

class UserManager:
    """User management and storage"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.password_hashes: Dict[str, str] = {}
        self._initialize_default_users()

    def _initialize_default_users(self):
        """Initialize default users for testing"""
        # Admin user
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@gaia.ai",
            role=UserRole.ADMIN,
            permissions=[
                Permission.READ_QUERIES,
                Permission.WRITE_QUERIES,
                Permission.EXECUTE_TOOLS,
                Permission.ACCESS_ADVANCED_FEATURES,
                Permission.MANAGE_USERS,
                Permission.VIEW_ANALYTICS,
                Permission.ADMIN_ACCESS
            ]
        )
        self.users["admin"] = admin_user
        self.password_hashes["admin"] = PasswordManager.hash_password("admin123")

        # Premium user
        premium_user = User(
            id="premium",
            username="premium",
            email="premium@example.com",
            role=UserRole.PREMIUM,
            permissions=[
                Permission.READ_QUERIES,
                Permission.WRITE_QUERIES,
                Permission.EXECUTE_TOOLS,
                Permission.ACCESS_ADVANCED_FEATURES,
                Permission.VIEW_ANALYTICS
            ]
        )
        self.users["premium"] = premium_user
        self.password_hashes["premium"] = PasswordManager.hash_password("premium123")

        # Basic user
        basic_user = User(
            id="user",
            username="user",
            email="user@example.com",
            role=UserRole.USER,
            permissions=[
                Permission.READ_QUERIES,
                Permission.WRITE_QUERIES,
                Permission.EXECUTE_TOOLS
            ]
        )
        self.users["user"] = basic_user
        self.password_hashes["user"] = PasswordManager.hash_password("user123")

    def create_user(self, registration: UserRegistration) -> User:
        """Create a new user"""
        # Check if username or email already exists
        for user in self.users.values():
            if user.username == registration.username:
                raise ValueError("Username already exists")
            if user.email == registration.email:
                raise ValueError("Email already exists")

        # Create new user
        user_id = f"user_{len(self.users) + 1}"
        user = User(
            id=user_id,
            username=registration.username,
            email=registration.email,
            role=registration.role,
            permissions=self._get_default_permissions(registration.role)
        )

        # Hash password
        hashed_password = PasswordManager.hash_password(registration.password)

        # Store user and password
        self.users[user_id] = user
        self.password_hashes[user_id] = hashed_password

        logger.info(f"Created new user: {user.username}")
        return user

    def _get_default_permissions(self, role: UserRole) -> List[Permission]:
        """Get default permissions for role"""
        if role == UserRole.ADMIN:
            return list(Permission)
        elif role == UserRole.PREMIUM:
            return [
                Permission.READ_QUERIES,
                Permission.WRITE_QUERIES,
                Permission.EXECUTE_TOOLS,
                Permission.ACCESS_ADVANCED_FEATURES,
                Permission.VIEW_ANALYTICS
            ]
        elif role == UserRole.ENTERPRISE:
            return [
                Permission.READ_QUERIES,
                Permission.WRITE_QUERIES,
                Permission.EXECUTE_TOOLS,
                Permission.ACCESS_ADVANCED_FEATURES,
                Permission.VIEW_ANALYTICS
            ]
        else:  # USER
            return [
                Permission.READ_QUERIES,
                Permission.WRITE_QUERIES,
                Permission.EXECUTE_TOOLS
            ]

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            return None

        # Check if user is active
        if not user.is_active:
            return None

        # Verify password
        hashed_password = self.password_hashes.get(user.id)
        if not hashed_password:
            return None

        if not PasswordManager.verify_password(password, hashed_password):
            return None

        # Update last login
        user.last_login = datetime.utcnow()

        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def update_user_role(self, user_id: str, new_role: UserRole) -> bool:
        """Update user role"""
        user = self.users.get(user_id)
        if not user:
            return False

        user.role = new_role
        user.permissions = self._get_default_permissions(new_role)
        return True

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user"""
        user = self.users.get(user_id)
        if not user:
            return False

        user.is_active = False
        return True

class AuthService:
    """Main authentication service"""

    def __init__(self):
        self.config = AuthConfig()
        self.token_manager = TokenManager(self.config)
        self.user_manager = UserManager()

    def register_user(self, registration: UserRegistration) -> User:
        """Register a new user"""
        return self.user_manager.create_user(registration)

    def login_user(self, credentials: UserCredentials) -> TokenResponse:
        """Login user and return tokens"""
        user = self.user_manager.authenticate_user(credentials.username, credentials.password)

        if not user:
            raise ValueError("Invalid username or password")

        # Create tokens
        access_token = self.token_manager.create_access_token(user)
        refresh_token = self.token_manager.create_refresh_token(user)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60,
            user_id=user.id,
            username=user.username,
            role=user.role.value
        )

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        try:
            payload = self.token_manager.decode_token(refresh_token)

            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")

            user_id = payload.get("sub")
            user = self.user_manager.get_user_by_id(user_id)

            if not user or not user.is_active:
                raise ValueError("User not found or inactive")

            # Create new tokens
            access_token = self.token_manager.create_access_token(user)
            new_refresh_token = self.token_manager.create_refresh_token(user)

            # Blacklist old refresh token
            self.token_manager.blacklist_token(refresh_token)

            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=self.config.access_token_expire_minutes * 60,
                user_id=user.id,
                username=user.username,
                role=user.role.value
            )

        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid refresh token: {e}")

    def logout_user(self, access_token: str, refresh_token: str):
        """Logout user by blacklisting tokens"""
        self.token_manager.blacklist_token(access_token)
        self.token_manager.blacklist_token(refresh_token)

    def get_current_user(self, token: str) -> Optional[User]:
        """Get current user from token"""
        try:
            payload = self.token_manager.decode_token(token)

            if payload.get("type") != "access":
                return None

            user_id = payload.get("sub")
            user = self.user_manager.get_user_by_id(user_id)

            if not user or not user.is_active:
                return None

            return user

        except jwt.InvalidTokenError:
            return None

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(self, func):
            def wrapper(*args, **kwargs):
                # This would be used in FastAPI dependency injection
                # The actual implementation depends on how you integrate with FastAPI
                pass
            return wrapper
        return decorator

# Global auth service instance
auth_service = AuthService()

# FastAPI dependency functions
def get_current_user(token: str = None) -> Optional[User]:
    """FastAPI dependency to get current user"""
    if not token:
        return None
    return auth_service.get_current_user(token)

def require_auth(self, user: User = None) -> User:
    """FastAPI dependency to require authentication"""
    if not user:
        raise ValueError("Authentication required")
    return user

def require_permission(self, permission: Permission):
    """FastAPI dependency to require specific permission"""
    def dependency(self, user: User = None) -> User:
        if not user:
            raise ValueError("Authentication required")

        if not auth_service.has_permission(user, permission):
            raise ValueError(f"Permission {permission.value} required")

        return user

    return dependency

# Utility functions for API integration
def extract_token_from_header(authorization: str) -> Optional[str]:
    """Extract token from Authorization header"""
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    return authorization[7:]  # Remove "Bearer " prefix

def create_auth_headers(self, user: User) -> Dict[str, str]:
    """Create authentication headers for testing"""
    token_response = auth_service.login_user(
        UserCredentials(username=user.username, password="password")
    )
    return {"Authorization": f"Bearer {token_response.access_token}"}
