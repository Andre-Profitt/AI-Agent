"""Enhanced Dependency Injection Container with Agent Factory Integration"""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from src.shared.exceptions import InfrastructureException

T = TypeVar('T')

class DIContainer:
    """Enhanced Dependency Injection Container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_singleton(self, name: str, factory: Callable[[], T]) -> None:
        """Register a singleton service"""
        if name in self._singletons:
            raise InfrastructureException(f"Singleton {name} already registered")
        
        self._factories[name] = factory
        self.logger.debug(f"Registered singleton: {name}")
    
    def register_transient(self, name: str, factory: Callable[[], T]) -> None:
        """Register a transient service"""
        self._factories[name] = factory
        self.logger.debug(f"Registered transient: {name}")
    
    def register_instance(self, name: str, instance: T) -> None:
        """Register an existing instance"""
        self._services[name] = instance
        self._singletons[name] = instance
        self.logger.debug(f"Registered instance: {name}")
    
    def resolve(self, name: str) -> Any:
        """Resolve a service by name"""
        # Check if instance already exists
        if name in self._services:
            return self._services[name]
        
        # Check if it's a singleton that needs to be created
        if name in self._singletons:
            return self._singletons[name]
        
        # Check if factory exists
        if name in self._factories:
            # Create instance
            instance = self._factories[name]()
            
            # If it's meant to be a singleton, store it
            if name in self._factories and name not in self._services:
                self._singletons[name] = instance
                self._services[name] = instance
            
            return instance
        
        raise InfrastructureException(f"Service {name} not registered")
    
    def has(self, name: str) -> bool:
        """Check if a service is registered"""
        return name in self._services or name in self._factories
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
    
    def get_registered_services(self) -> list[str]:
        """Get list of all registered service names."""
        return list(set(self._factories.keys()) | set(self._singletons.keys()))


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get the global container instance"""
    global _container
    if _container is None:
        _container = DIContainer()
        setup_container()
    return _container


def setup_container() -> None:
    """Setup all container dependencies including Agent Factory"""
    container = get_container()
    
    # Import all required modules
    from src.infrastructure.config.configuration_service import ConfigurationService
    from src.infrastructure.logging.logging_service import LoggingService
    from src.application.agents.agent_factory import AgentFactory
    from src.application.tools.tool_factory import ToolFactory
    
    # Repositories
    from src.infrastructure.database.in_memory_agent_repository import InMemoryAgentRepository
    from src.infrastructure.database.in_memory_message_repository import InMemoryMessageRepository
    from src.infrastructure.database.in_memory_tool_repository import InMemoryToolRepository
    from src.infrastructure.database.in_memory_session_repository import InMemorySessionRepository
    
    # Use cases
    from src.core.use_cases.process_message import ProcessMessageUseCase
    from src.core.use_cases.manage_agent import ManageAgentUseCase
    from src.core.use_cases.execute_tool import ExecuteToolUseCase
    from src.core.use_cases.manage_session import ManageSessionUseCase
    
    # Executors
    from src.application.agents.agent_executor import AgentExecutor
    from src.application.tools.tool_executor import ToolExecutor
    
    # Register configuration as singleton
    container.register_singleton("configuration_service", ConfigurationService)
    
    # Register logging as singleton
    container.register_singleton("logging_service", lambda: LoggingService(
        container.resolve("configuration_service").config.logging_config
    ))
    
    # Register repositories as singletons
    container.register_singleton("agent_repository", InMemoryAgentRepository)
    container.register_singleton("message_repository", InMemoryMessageRepository)
    container.register_singleton("tool_repository", InMemoryToolRepository)
    container.register_singleton("session_repository", InMemorySessionRepository)
    
    # Register factories as singletons
    container.register_singleton("tool_factory", ToolFactory)
    container.register_singleton("agent_factory", lambda: AgentFactory(
        tool_factory=container.resolve("tool_factory")
    ))
    
    # Register executors
    container.register_singleton("agent_executor", lambda: AgentExecutor())
    container.register_singleton("tool_executor", lambda: ToolExecutor())
    
    # Register use cases as transient (new instance each time)
    container.register_transient("process_message_use_case", lambda: ProcessMessageUseCase(
        agent_repository=container.resolve("agent_repository"),
        message_repository=container.resolve("message_repository"),
        agent_executor=container.resolve("agent_executor"),
        logging_service=container.resolve("logging_service"),
        config=container.resolve("configuration_service").config.agent_config
    ))
    
    container.register_transient("manage_agent_use_case", lambda: ManageAgentUseCase(
        agent_repository=container.resolve("agent_repository"),
        agent_factory=container.resolve("agent_factory"),
        logging_service=container.resolve("logging_service")
    ))
    
    container.register_transient("execute_tool_use_case", lambda: ExecuteToolUseCase(
        tool_repository=container.resolve("tool_repository"),
        tool_executor=container.resolve("tool_executor"),
        logging_service=container.resolve("logging_service")
    ))
    
    container.register_transient("manage_session_use_case", lambda: ManageSessionUseCase(
        session_repository=container.resolve("session_repository"),
        message_repository=container.resolve("message_repository"),
        logging_service=container.resolve("logging_service")
    ))
    
    container.logger.info("Dependency container configured successfully")


# Legacy compatibility functions
def inject(*service_names: str):
    """
    Decorator to inject dependencies into a function or method.
    
    Args:
        *service_names: Names of services to inject
        
    Returns:
        Decorated function with injected dependencies
    """
    def decorator(func: Callable) -> Callable:
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = get_container()
            dependencies = {name: container.resolve(name) for name in service_names}
            
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