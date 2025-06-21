from src.tools.base_tool import Tool

from src.tools.base_tool import BaseTool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import BaseAgent

"""
from typing import Awaitable
from typing import TypeVar
# TODO: Fix undefined variables: Awaitable, runtime_checkable

Type protocols and interfaces for dependency injection
"""

from typing import Callable
from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from typing import Union

from typing import Protocol, Optional, Dict, Any, List, Callable, Union, Awaitable, TypeVar, runtime_checkable
from abc import abstractmethod
import logging
from dataclasses import dataclass

# Type variables for generics
T = TypeVar('T')
TEntity = TypeVar('TEntity')
TQuery = TypeVar('TQuery')
TResult = TypeVar('TResult')

@runtime_checkable
class ConfigurationService(Protocol):
    """Protocol for configuration services"""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        ...

    @abstractmethod
    async def reload_config(self) -> bool:
        """Reload configuration from source"""
        ...

    @abstractmethod
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get specific configuration value"""
        ...

    @abstractmethod
    async def is_configured_safe(self) -> bool:
        """Check if configuration is valid with protection"""
        ...

@runtime_checkable
class LoggingService(Protocol):
    """Protocol for logging services"""

    @abstractmethod
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message at specified level"""
        ...

    @abstractmethod
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""
        ...

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        ...

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        ...

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        ...

    @abstractmethod
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message"""
        ...

@runtime_checkable
class DatabaseClient(Protocol):
    """Protocol for database clients"""

    @abstractmethod
    async def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        ...

    @abstractmethod
    async def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a statement and return affected rows or result"""
        ...

    @abstractmethod
    async def transaction(self) -> 'TransactionContext':
        """Start a database transaction"""
        ...

    @abstractmethod
    async def table(self, name: str) -> 'TableClient':
        """Get a table client"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close database connection"""
        ...

@runtime_checkable
class TableClient(Protocol):
    """Protocol for database table operations"""

    @abstractmethod
    def select(self, *columns: str) -> 'QueryBuilder':
        """Start a select query"""
        ...

    @abstractmethod
    def insert(self, data: Dict[str, Any]) -> Awaitable[Dict[str, Any]]:
        """Insert data into table"""
        ...

    @abstractmethod
    def update(self, data: Dict[str, Any]) -> 'QueryBuilder':
        """Update data in table"""
        ...

    @abstractmethod
    def delete(self) -> 'QueryBuilder':
        """Delete from table"""
        ...

@runtime_checkable
class QueryBuilder(Protocol):
    """Protocol for query building"""

    @abstractmethod
    def eq(self, column: str, value: Any) -> 'QueryBuilder':
        """Add equality condition"""
        ...

    @abstractmethod
    def neq(self, column: str, value: Any) -> 'QueryBuilder':
        """Add not equal condition"""
        ...

    @abstractmethod
    def gt(self, column: str, value: Any) -> 'QueryBuilder':
        """Add greater than condition"""
        ...

    @abstractmethod
    def gte(self, column: str, value: Any) -> 'QueryBuilder':
        """Add greater than or equal condition"""
        ...

    @abstractmethod
    def lt(self, column: str, value: Any) -> 'QueryBuilder':
        """Add less than condition"""
        ...

    @abstractmethod
    def lte(self, column: str, value: Any) -> 'QueryBuilder':
        """Add less than or equal condition"""
        ...

    @abstractmethod
    def order(self, column: str, ascending: bool = True) -> 'QueryBuilder':
        """Add order by clause"""
        ...

    @abstractmethod
    def limit(self, count: int) -> 'QueryBuilder':
        """Add limit clause"""
        ...

    @abstractmethod
    def execute(self) -> Awaitable[Union[List[Dict[str, Any]], Dict[str, Any], int]]:
        """Execute the query"""
        ...

@runtime_checkable
class TransactionContext(Protocol):
    """Protocol for database transactions"""

    @abstractmethod
    async def __aenter__(self) -> 'TransactionContext':
        """Enter transaction context"""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit transaction context"""
        ...

    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction"""
        ...

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction"""
        ...

@runtime_checkable
class CacheClient(Protocol):
    """Protocol for cache clients"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL in seconds"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        ...

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on existing key"""
        ...

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        ...

@runtime_checkable
class Repository(Protocol[TEntity]):
    """Base protocol for repositories"""

    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[TEntity]:
        """Find entity by ID"""
        ...

    @abstractmethod
    async def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[TEntity]:
        """Find all entities with pagination"""
        ...

    @abstractmethod
    async def save(self, entity: TEntity) -> TEntity:
        """Save entity (create or update)"""
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID"""
        ...

    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if entity exists"""
        ...

@runtime_checkable
class MessageQueue(Protocol):
    """Protocol for message queues"""

    @abstractmethod
    async def publish(self, topic: str, message: Any, headers: Optional[Dict[str, str]] = None) -> bool:
        """Publish a message to a topic"""
        ...

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]) -> str:
        """Subscribe to a topic with a handler"""
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic"""
        ...

    @abstractmethod
    async def acknowledge(self, message_id: str) -> bool:
        """Acknowledge message processing"""
        ...

@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for metrics collection"""

    @abstractmethod
    def increment(self, metric: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        ...

    @abstractmethod
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        ...

    @abstractmethod
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        ...

    @abstractmethod
    def timing(self, metric: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        ...

@runtime_checkable
class EventBus(Protocol):
    """Protocol for event bus"""

    @abstractmethod
    async def emit(self, event_type: str, data: Any) -> None:
        """Emit an event"""
        ...

    @abstractmethod
    def on(self, event_type: str, handler: Callable[[Any], Awaitable[None]]) -> str:
        """Register an event handler"""
        ...

    @abstractmethod
    def off(self, handler_id: str) -> bool:
        """Unregister an event handler"""
        ...

@runtime_checkable
class CircuitBreaker(Protocol):
    """Protocol for circuit breaker"""

    @abstractmethod
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Call function through circuit breaker"""
        ...

    @abstractmethod
    def is_open(self) -> bool:
        """Check if circuit is open"""
        ...

    @abstractmethod
    def is_closed(self) -> bool:
        """Check if circuit is closed"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset circuit breaker state"""
        ...

# Agent and Tool protocols
@runtime_checkable
class BaseTool(Protocol):
    """Protocol for agent tools"""

    name: str
    description: str

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given input"""
        ...

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get tool input schema"""
        ...

@runtime_checkable
class BaseAgent(Protocol):
    """Protocol for agents"""

    name: str
    agent_id: str

    @abstractmethod
    async def execute(self, task: Any) -> Any:
        """Execute a task"""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent"""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the agent"""
        ...

# Service container types
@dataclass
class ServiceDescriptor:
    """Descriptor for a service in DI container"""
    service_type: type
    implementation_type: type
    lifetime: str  # "singleton", "scoped", "transient"
    factory: Optional[Callable[[], Any]] = None

class ServiceContainer(Protocol):
    """Protocol for dependency injection container"""

    @abstractmethod
    def register(self, service_type: type, implementation: Union[type, Callable[[], Any]],
                 lifetime: str = "transient") -> None:
        """Register a service"""
        ...

    @abstractmethod
    def resolve(self, service_type: type) -> Any:
        """Resolve a service"""
        ...

    @abstractmethod
    def create_scope(self) -> 'ServiceScope':
        """Create a new scope"""
        ...

class ServiceScope(Protocol):
    """Protocol for DI scope"""

    @abstractmethod
    def resolve(self, service_type: type) -> Any:
        """Resolve a service in this scope"""
        ...

    @abstractmethod
    def dispose(self) -> None:
        """Dispose of this scope"""
        ...

# Type aliases for common patterns
MiddlewareFunc = Callable[[Dict[str, Any], Callable], Awaitable[Dict[str, Any]]]
HandlerFunc = Callable[[Any], Awaitable[Any]]
ValidatorFunc = Callable[[Any], bool]
TransformFunc = Callable[[Any], Any]
PredicateFunc = Callable[[Any], bool]

# Database types
ConnectionString = str
DatabaseConnection = DatabaseClient
SupabaseClient = DatabaseClient

# Component types for clean architecture
class UseCase(Protocol[TQuery, TResult]):
    """Protocol for use cases"""

    @abstractmethod
    async def execute(self, query: TQuery) -> TResult:
        """Execute the use case"""
        ...

class Command(Protocol[TResult]):
    """Protocol for commands"""

    @abstractmethod
    async def execute(self) -> TResult:
        """Execute the command"""
        ...

class Query(Protocol[TResult]):
    """Protocol for queries"""

    @abstractmethod
    async def execute(self) -> TResult:
        """Execute the query"""
        ...

# Export all protocols and types
__all__ = [
    # Core protocols
    'ConfigurationService',
    'LoggingService',
    'DatabaseClient',
    'CacheClient',
    'Repository',
    'MessageQueue',
    'MetricsCollector',
    'EventBus',
    'CircuitBreaker',

    # Database protocols
    'TableClient',
    'QueryBuilder',
    'TransactionContext',

    # Agent protocols
    'BaseTool',
    'BaseAgent',

    # DI protocols
    'ServiceContainer',
    'ServiceScope',
    'ServiceDescriptor',

    # Architecture protocols
    'UseCase',
    'Command',
    'Query',

    # Type aliases
    'MiddlewareFunc',
    'HandlerFunc',
    'ValidatorFunc',
    'TransformFunc',
    'PredicateFunc',
    'ConnectionString',
    'DatabaseConnection',
    'SupabaseClient',
]