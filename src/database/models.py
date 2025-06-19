"""
Database Models for GAIA System
SQLAlchemy models for all GAIA components
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

# Association tables for many-to-many relationships
user_permissions = Table(
    'user_permissions',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id')),
    Column('permission_id', String, ForeignKey('permissions.id'))
)

agent_tools = Table(
    'agent_tools',
    Base.metadata,
    Column('agent_id', String, ForeignKey('agents.id')),
    Column('tool_id', String, ForeignKey('tools.id'))
)

class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    
    # Relationships
    permissions = relationship("Permission", secondary=user_permissions, back_populates="users")
    sessions = relationship("Session", back_populates="user")
    queries = relationship("Query", back_populates="user")

class Permission(Base):
    """Permission model"""
    __tablename__ = 'permissions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    users = relationship("User", secondary=user_permissions, back_populates="permissions")

class Session(Base):
    """User session model"""
    __tablename__ = 'sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")

class Query(Base):
    """Query model for tracking user queries"""
    __tablename__ = 'queries'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    success = Column(Boolean, default=True)
    execution_time = Column(Float)
    confidence = Column(Float)
    verification_level = Column(String(20), default='basic')
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="queries")
    tool_calls = relationship("ToolCall", back_populates="query")

class ToolCall(Base):
    """Tool call model for tracking tool executions"""
    __tablename__ = 'tool_calls'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, ForeignKey('queries.id'), nullable=False)
    tool_name = Column(String(100), nullable=False)
    tool_input = Column(JSON)
    tool_output = Column(JSON)
    success = Column(Boolean, default=True)
    execution_time = Column(Float)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    query = relationship("Query", back_populates="tool_calls")

class Agent(Base):
    """Agent model"""
    __tablename__ = 'agents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)
    model_name = Column(String(100))
    configuration = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime)
    
    # Relationships
    tools = relationship("Tool", secondary=agent_tools, back_populates="agents")
    executions = relationship("AgentExecution", back_populates="agent")

class Tool(Base):
    """Tool model"""
    __tablename__ = 'tools'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    tool_type = Column(String(50))
    parameters = Column(JSON)
    is_active = Column(Boolean, default=True)
    reliability_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime)
    
    # Relationships
    agents = relationship("Agent", secondary=agent_tools, back_populates="tools")
    executions = relationship("ToolExecution", back_populates="tool")

class AgentExecution(Base):
    """Agent execution model"""
    __tablename__ = 'agent_executions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('agents.id'), nullable=False)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    success = Column(Boolean, default=True)
    execution_time = Column(Float)
    confidence = Column(Float)
    reasoning_path = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="executions")

class ToolExecution(Base):
    """Tool execution model"""
    __tablename__ = 'tool_executions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tool_id = Column(String, ForeignKey('tools.id'), nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    success = Column(Boolean, default=True)
    execution_time = Column(Float)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    tool = relationship("Tool", back_populates="executions")

class Memory(Base):
    """Memory model for GAIA memory system"""
    __tablename__ = 'memories'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_type = Column(String(50), nullable=False)  # episodic, semantic, working
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    priority = Column(Integer, default=0)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)

class VectorStore(Base):
    """Vector store model"""
    __tablename__ = 'vector_stores'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, nullable=False)
    provider = Column(String(50), nullable=False)  # chroma, pinecone, etc.
    configuration = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now())

class Embedding(Base):
    """Embedding model"""
    __tablename__ = 'embeddings'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    vector_store_id = Column(String, ForeignKey('vector_stores.id'), nullable=False)
    text = Column(Text, nullable=False)
    embedding_vector = Column(JSON)  # Store as JSON array
    metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    vector_store = relationship("VectorStore")

class PerformanceMetric(Base):
    """Performance metrics model"""
    __tablename__ = 'performance_metrics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50))  # counter, gauge, histogram
    labels = Column(JSON)
    timestamp = Column(DateTime, default=func.now())

class SystemHealth(Base):
    """System health model"""
    __tablename__ = 'system_health'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    component = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # healthy, degraded, down
    message = Column(Text)
    last_check = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())

class Backup(Base):
    """Backup model"""
    __tablename__ = 'backups'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    backup_id = Column(String(100), unique=True, nullable=False)
    backup_type = Column(String(50), nullable=False)
    storage_backend = Column(String(50), nullable=False)
    size_bytes = Column(Integer)
    checksum = Column(String(64))
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)

class AuditLog(Base):
    """Audit log model"""
    __tablename__ = 'audit_logs'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User")

# Indexes for better performance
from sqlalchemy import Index

# User indexes
Index('idx_users_username', User.username)
Index('idx_users_email', User.email)
Index('idx_users_role', User.role)

# Query indexes
Index('idx_queries_user_id', Query.user_id)
Index('idx_queries_created_at', Query.created_at)
Index('idx_queries_success', Query.success)

# Tool call indexes
Index('idx_tool_calls_query_id', ToolCall.query_id)
Index('idx_tool_calls_tool_name', ToolCall.tool_name)
Index('idx_tool_calls_created_at', ToolCall.created_at)

# Memory indexes
Index('idx_memories_type', Memory.memory_type)
Index('idx_memories_priority', Memory.priority)
Index('idx_memories_last_accessed', Memory.last_accessed)

# Performance metrics indexes
Index('idx_performance_metrics_name', PerformanceMetric.metric_name)
Index('idx_performance_metrics_timestamp', PerformanceMetric.timestamp)

# Audit log indexes
Index('idx_audit_logs_user_id', AuditLog.user_id)
Index('idx_audit_logs_action', AuditLog.action)
Index('idx_audit_logs_created_at', AuditLog.created_at) 