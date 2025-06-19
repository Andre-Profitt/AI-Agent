"""
Dependency injection container for managing service dependencies.
"""

from typing import Dict, Any, Type, Optional, Callable
import logging
from functools import wraps

from src.shared.exceptions import InfrastructureException
from src.infrastructure.database.in_memory_message_repository import InMemoryMessageRepository
from src.infrastructure.database.in_memory_session_repository import InMemorySessionRepository
from src.infrastructure.database.in_memory_tool_repository import InMemoryToolRepository
from src.infrastructure.database.in_memory_user_repository import InMemoryUserRepository
from src.infrastructure.logging.logging_service import LoggingService
from src.infrastructure.config.configuration_service import ConfigurationService
from src.application.agents.agent_executor import AgentExecutor
from src.application.tools.tool_executor import ToolExecutor
from src.shared.types.config import LoggingConfig


class Container:
    """
    Simple dependency injection container.
    
    This container manages the creation and lifecycle of services,
    ensuring proper dependency resolution and singleton management.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, service_name: str, factory: Callable, singleton: bool = True) -> None:
        """
        Register a service with the container.
        
        Args:
            service_name: Name of the service
            factory: Factory function to create the service
            singleton: Whether the service should be a singleton
        """
        self._factories[service_name] = factory
        if not singleton:
            self._services[service_name] = None  # Mark as non-singleton
    
    def register_instance(self, service_name: str, instance: Any) -> None:
        """
        Register an existing instance with the container.
        
        Args:
            service_name: Name of the service
            instance: The service instance
        """
        self._singletons[service_name] = instance
    
    def resolve(self, service_name: str) -> Any:
        """
        Resolve a service from the container.
        
        Args:
            service_name: Name of the service to resolve
            
        Returns:
            The resolved service instance
            
        Raises:
            InfrastructureException: If service cannot be resolved
        """
        # Check if we have a singleton instance
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check if we have a factory
        if service_name not in self._factories:
            raise InfrastructureException(f"Service '{service_name}' not registered")
        
        # Create new instance
        factory = self._factories[service_name]
        try:
            instance = factory(self)
            
            # Store as singleton if configured
            if service_name not in self._services or self._services[service_name] is not None:
                self._singletons[service_name] = instance
            
            return instance
            
        except Exception as e:
            self.logger.error(f"Failed to resolve service '{service_name}': {str(e)}")
            raise InfrastructureException(f"Failed to resolve service '{service_name}': {str(e)}")
    
    def resolve_all(self, service_names: list[str]) -> Dict[str, Any]:
        """
        Resolve multiple services at once.
        
        Args:
            service_names: List of service names to resolve
            
        Returns:
            Dictionary mapping service names to instances
        """
        return {name: self.resolve(name) for name in service_names}
    
    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._factories or service_name in self._singletons
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
    
    def get_registered_services(self) -> list[str]:
        """Get list of all registered service names."""
        return list(set(self._factories.keys()) | set(self._singletons.keys()))


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def inject(*service_names: str):
    """
    Decorator to inject dependencies into a function or method.
    
    Args:
        *service_names: Names of services to inject
        
    Returns:
        Decorated function with injected dependencies
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            dependencies = container.resolve_all(service_names)
            
            # Inject dependencies as keyword arguments
            kwargs.update(dependencies)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls: Type) -> Type:
    """
    Decorator to make a class a singleton.
    
    Args:
        cls: The class to make a singleton
        
    Returns:
        Singleton class
    """
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def setup_container():
    container = get_container()
    # Register repositories
    container.register('message_repository', lambda c: InMemoryMessageRepository())
    container.register('session_repository', lambda c: InMemorySessionRepository())
    container.register('tool_repository', lambda c: InMemoryToolRepository())
    container.register('user_repository', lambda c: InMemoryUserRepository())
    # Register services
    container.register('logging_service', lambda c: LoggingService(LoggingConfig()), singleton=True)
    container.register('configuration_service', lambda c: ConfigurationService(), singleton=True)
    container.register('agent_executor', lambda c: AgentExecutor(), singleton=True)
    container.register('tool_executor', lambda c: ToolExecutor(), singleton=True)
    return container 