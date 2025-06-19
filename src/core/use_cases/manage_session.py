"""
Use case for managing user sessions.
"""

from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
import logging
from datetime import datetime, timedelta

from src.core.entities.session import Session, SessionState
from src.core.entities.message import Message, MessageType
from src.core.interfaces.session_repository import SessionRepository
from src.core.interfaces.message_repository import MessageRepository
from src.core.interfaces.logging_service import LoggingService
from src.shared.exceptions import DomainException, ValidationException


class ManageSessionUseCase:
    """
    Use case for managing user sessions.
    
    This use case handles session creation, updates, deletion,
    and message history management.
    """
    
    def __init__(
        self,
        session_repository: SessionRepository,
        message_repository: MessageRepository,
        logging_service: LoggingService
    ):
        self.session_repository = session_repository
        self.message_repository = message_repository
        self.logging_service = logging_service
        self.logger = logging.getLogger(__name__)
    
    async def create_session(
        self,
        user_id: Optional[UUID] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            title: Optional session title
            metadata: Optional session metadata
            
        Returns:
            Dictionary containing the created session information
        """
        try:
            # Create session entity
            session = Session(
                user_id=user_id,
                title=title or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                metadata=metadata or {},
                state=SessionState.ACTIVE
            )
            
            # Save session
            saved_session = await self.session_repository.save(session)
            
            # Log creation
            await self.logging_service.log_info(
                "session_created",
                f"Created session {saved_session.id}",
                {"session_id": str(saved_session.id), "user_id": str(user_id) if user_id else None}
            )
            
            return {
                "success": True,
                "session_id": str(saved_session.id),
                "title": saved_session.title,
                "state": saved_session.state.value,
                "created_at": saved_session.created_at.isoformat() if saved_session.created_at else None
            }
            
        except Exception as e:
            self.logger.error("Failed to create session: {}", extra={"str_e_": str(e)})
            await self.logging_service.log_error(
                "session_creation_failed",
                str(e),
                {"user_id": str(user_id) if user_id else None}
            )
            return {"success": False, "error": str(e)}
    
    async def get_session(self, session_id: UUID) -> Dict[str, Any]:
        """
        Get session information.
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            Dictionary containing session information
        """
        try:
            session = await self.session_repository.find_by_id(session_id)
            if not session:
                return {"success": False, "error": f"Session {session_id} not found"}
            
            return {
                "success": True,
                "session": {
                    "id": str(session.id),
                    "title": session.title,
                    "state": session.state.value,
                    "metadata": session.metadata,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to get session {}: {}", extra={"session_id": session_id, "str_e_": str(e)})
            return {"success": False, "error": str(e)}
    
    async def update_session(
        self,
        session_id: UUID,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        state: Optional[SessionState] = None
    ) -> Dict[str, Any]:
        """
        Update an existing session.
        
        Args:
            session_id: ID of the session to update
            title: New session title
            metadata: New session metadata
            state: New session state
            
        Returns:
            Dictionary containing the update result
        """
        try:
            # Find session
            session = await self.session_repository.find_by_id(session_id)
            if not session:
                raise DomainException(f"Session {session_id} not found")
            
            # Update fields
            if title is not None:
                session.title = title
            
            if metadata is not None:
                session.metadata.update(metadata)
            
            if state is not None:
                session.state = state
            
            # Update last activity
            session.last_activity = datetime.utcnow()
            
            # Save updated session
            updated_session = await self.session_repository.save(session)
            
            # Log update
            await self.logging_service.log_info(
                "session_updated",
                f"Updated session {session_id}",
                {"session_id": str(session_id)}
            )
            
            return {
                "success": True,
                "session_id": str(updated_session.id),
                "title": updated_session.title,
                "state": updated_session.state.value
            }
            
        except Exception as e:
            self.logger.error("Failed to update session {}: {}", extra={"session_id": session_id, "str_e_": str(e)})
            await self.logging_service.log_error(
                "session_update_failed",
                str(e),
                {"session_id": str(session_id)}
            )
            return {"success": False, "error": str(e)}
    
    async def delete_session(self, session_id: UUID) -> Dict[str, Any]:
        """
        Delete a session and all its messages.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            Dictionary containing the deletion result
        """
        try:
            # Check if session exists
            session = await self.session_repository.find_by_id(session_id)
            if not session:
                raise DomainException(f"Session {session_id} not found")
            
            # Delete all messages in the session
            messages = await self.message_repository.find_by_session_id(session_id)
            for message in messages:
                await self.message_repository.delete(message.id)
            
            # Delete session
            success = await self.session_repository.delete(session_id)
            if not success:
                raise DomainException(f"Failed to delete session {session_id}")
            
            # Log deletion
            await self.logging_service.log_info(
                "session_deleted",
                f"Deleted session {session_id} and {len(messages)} messages",
                {"session_id": str(session_id), "messages_deleted": len(messages)}
            )
            
            return {"success": True, "session_id": str(session_id)}
            
        except Exception as e:
            self.logger.error("Failed to delete session {}: {}", extra={"session_id": session_id, "str_e_": str(e)})
            await self.logging_service.log_error(
                "session_deletion_failed",
                str(e),
                {"session_id": str(session_id)}
            )
            return {"success": False, "error": str(e)}
    
    async def list_sessions(
        self,
        user_id: Optional[UUID] = None,
        state: Optional[SessionState] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List sessions with optional filtering.
        
        Args:
            user_id: Optional user filter
            state: Optional state filter
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            Dictionary containing the list of sessions
        """
        try:
            sessions = await self.session_repository.find_all(
                user_id=user_id,
                state=state,
                limit=limit,
                offset=offset
            )
            
            session_list = []
            for session in sessions:
                session_list.append({
                    "id": str(session.id),
                    "title": session.title,
                    "state": session.state.value,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None
                })
            
            return {
                "success": True,
                "sessions": session_list,
                "count": len(session_list)
            }
            
        except Exception as e:
            self.logger.error("Failed to list sessions: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}
    
    async def get_session_messages(
        self,
        session_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get messages for a session.
        
        Args:
            session_id: ID of the session
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            Dictionary containing the session messages
        """
        try:
            # Check if session exists
            session = await self.session_repository.find_by_id(session_id)
            if not session:
                return {"success": False, "error": f"Session {session_id} not found"}
            
            # Get messages
            messages = await self.message_repository.find_by_session_id(
                session_id, limit=limit, offset=offset
            )
            
            message_list = []
            for message in messages:
                message_list.append({
                    "id": str(message.id),
                    "content": message.content,
                    "message_type": message.message_type.value,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                    "context": message.context
                })
            
            return {
                "success": True,
                "session_id": str(session_id),
                "messages": message_list,
                "count": len(message_list)
            }
            
        except Exception as e:
            self.logger.error("Failed to get session messages {}: {}", extra={"session_id": session_id, "str_e_": str(e)})
            return {"success": False, "error": str(e)}
    
    async def cleanup_expired_sessions(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up expired sessions.
        
        Args:
            max_age_hours: Maximum age in hours before session is considered expired
            
        Returns:
            Dictionary containing cleanup results
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            # Find expired sessions
            expired_sessions = await self.session_repository.find_expired(cutoff_time)
            
            deleted_count = 0
            for session in expired_sessions:
                try:
                    # Delete session and its messages
                    await self.delete_session(session.id)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning("Failed to delete expired session {}: {}", extra={"session_id": session.id, "str_e_": str(e)})
            
            # Log cleanup
            await self.logging_service.log_info(
                "sessions_cleaned_up",
                f"Cleaned up {deleted_count} expired sessions",
                {"deleted_count": deleted_count, "max_age_hours": max_age_hours}
            )
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "max_age_hours": max_age_hours
            }
            
        except Exception as e:
            self.logger.error("Failed to cleanup expired sessions: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get session repository statistics.
        
        Returns:
            Dictionary containing session statistics
        """
        try:
            stats = await self.session_repository.get_statistics()
            return {"success": True, "statistics": stats}
            
        except Exception as e:
            self.logger.error("Failed to get session statistics: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)} 