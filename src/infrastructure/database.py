from migrations.env import url
from setup_environment import components

from src.collaboration.realtime_collaboration import session_id
from src.core.langgraph_compatibility import loop
from src.core.monitoring import key
from src.database_extended import interaction
from src.infrastructure.database import client
from src.infrastructure.database import log_entry

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Client, Optional, assistant_response, client, components, create_client, e, initialize_supabase_enhanced, interaction, key, log_entry, logging, loop, os, record, session_id, url, user_message
# TODO: Fix undefined variables: Client, assistant_response, client, components, create_client, e, initialize_supabase_enhanced, interaction, key, log_entry, loop, record, self, session_id, url, user_message

"""
Database utilities for the AI Agent system
Provides vector store access and database connection management
"""

from typing import Any

import os
import logging
from typing import Optional, Any
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Global client instance
_supabase_client: Optional[Client] = None
_vector_store: Optional[Any] = None

def get_supabase_client() -> Client:
    """Get or create Supabase client"""
    global _supabase_client

    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

        _supabase_client = create_client(url, key)
        logger.info("Supabase client initialized")

    return _supabase_client

def get_vector_store() -> Any:
    """Get vector store instance"""
    global _vector_store

    if _vector_store is None:
        try:
            # Try to get from enhanced database
            from .database_enhanced import initialize_supabase_enhanced
            import asyncio

            # Initialize enhanced vector store
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                components = loop.run_until_complete(initialize_supabase_enhanced())
                _vector_store = components.get('vector_store')
            finally:
                loop.close()

        except Exception as e:
            logger.warning("Could not initialize enhanced vector store: {}", extra={"e": e})
            _vector_store = None

    return _vector_store

class SupabaseLogHandler(logging.Handler):
    """Custom log handler for Supabase"""

    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client

    def emit(self, record) -> Any:
        try:
            # Create log entry
            log_entry = {
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'timestamp': record.created
            }

            # Insert into logs table
            self.client.table('logs').insert(log_entry).execute()

        except Exception as e:
            # Don't let logging errors crash the application
            logger.info("Log handler error: {}", extra={"e": e})

    def log_interaction(self, session_id: str, user_message: str, assistant_response: str) -> Any:
        """Log user interaction"""
        try:
            interaction = {
                'session_id': session_id,
                'user_message': user_message,
                'assistant_response': assistant_response,
                'timestamp': 'now()'
            }

            self.client.table('interactions').insert(interaction).execute()

        except Exception as e:
            logger.error("Failed to log interaction: {}", extra={"e": e})

def create_tables() -> Any:
    """Create necessary database tables"""
    try:
        client = get_supabase_client()

        # Create logs table
        client.rpc('create_logs_table').execute()

        # Create interactions table
        client.rpc('create_interactions_table').execute()

        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error("Failed to create tables: {}", extra={"e": e})

def health_check() -> dict:
    """Check database health"""
    try:
        client = get_supabase_client()

        # Test connection
        response = client.table('logs').select('count').limit(1).execute()

        return {
            'status': 'healthy',
            'connection': 'ok',
            'tables': 'accessible'
        }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }