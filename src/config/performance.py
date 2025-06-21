"""
Performance configuration and tuning
Central configuration for all performance optimizations
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import timedelta
import os

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    
    # Caching
    enable_caching: bool = True
    cache_memory_size: int = 1000
    cache_memory_ttl: int = 300  # 5 minutes
    cache_redis_ttl: int = 3600  # 1 hour
    cache_disk_size_mb: int = 100
    
    # Connection pooling
    enable_connection_pooling: bool = True
    pool_min_size: int = 2
    pool_max_size: int = 10
    pool_idle_timeout: float = 300.0
    
    # Resource limits
    enable_resource_limits: bool = True
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_concurrent_requests: int = 100
    max_tokens_per_minute: int = 10000
    
    # Request handling
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Batch processing
    enable_batching: bool = True
    batch_size: int = 50
    batch_timeout: float = 1.0
    
    # Async settings
    max_workers: int = 10
    event_loop_policy: str = "uvloop"  # uvloop, asyncio
    
    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_interval: float = 60.0
    slow_query_threshold: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_caching=os.getenv("PERF_ENABLE_CACHING", "true").lower() == "true",
            cache_memory_size=int(os.getenv("PERF_CACHE_MEMORY_SIZE", "1000")),
            cache_memory_ttl=int(os.getenv("PERF_CACHE_MEMORY_TTL", "300")),
            cache_redis_ttl=int(os.getenv("PERF_CACHE_REDIS_TTL", "3600")),
            
            enable_connection_pooling=os.getenv("PERF_ENABLE_POOLING", "true").lower() == "true",
            pool_max_size=int(os.getenv("PERF_POOL_MAX_SIZE", "10")),
            
            enable_resource_limits=os.getenv("PERF_ENABLE_LIMITS", "true").lower() == "true",
            max_memory_mb=int(os.getenv("PERF_MAX_MEMORY_MB", "1024")),
            max_concurrent_requests=int(os.getenv("PERF_MAX_CONCURRENT", "100")),
            
            request_timeout=float(os.getenv("PERF_REQUEST_TIMEOUT", "30.0")),
            
            enable_batching=os.getenv("PERF_ENABLE_BATCHING", "true").lower() == "true",
            batch_size=int(os.getenv("PERF_BATCH_SIZE", "50")),
            
            enable_performance_monitoring=os.getenv("PERF_ENABLE_MONITORING", "true").lower() == "true",
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "caching": {
                "enabled": self.enable_caching,
                "memory_size": self.cache_memory_size,
                "memory_ttl": self.cache_memory_ttl,
                "redis_ttl": self.cache_redis_ttl,
                "disk_size_mb": self.cache_disk_size_mb
            },
            "connection_pooling": {
                "enabled": self.enable_connection_pooling,
                "min_size": self.pool_min_size,
                "max_size": self.pool_max_size,
                "idle_timeout": self.pool_idle_timeout
            },
            "resource_limits": {
                "enabled": self.enable_resource_limits,
                "max_memory_mb": self.max_memory_mb,
                "max_cpu_percent": self.max_cpu_percent,
                "max_concurrent_requests": self.max_concurrent_requests,
                "max_tokens_per_minute": self.max_tokens_per_minute
            },
            "request_handling": {
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.circuit_breaker_timeout
            },
            "batch_processing": {
                "enabled": self.enable_batching,
                "size": self.batch_size,
                "timeout": self.batch_timeout
            },
            "async_settings": {
                "max_workers": self.max_workers,
                "event_loop_policy": self.event_loop_policy
            },
            "monitoring": {
                "enabled": self.enable_performance_monitoring,
                "metrics_interval": self.metrics_interval,
                "slow_query_threshold": self.slow_query_threshold
            }
        }

# Performance tuning utilities
class PerformanceTuner:
    """Automatic performance tuning based on system resources"""
    
    @staticmethod
    def auto_tune() -> PerformanceConfig:
        """Automatically tune performance settings based on system"""
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        
        config = PerformanceConfig()
        
        # Tune based on CPU
        config.max_workers = min(cpu_count * 2, 20)
        config.pool_max_size = min(cpu_count * 5, 50)
        
        # Tune based on memory
        if memory_gb >= 16:
            config.cache_memory_size = 5000
            config.max_memory_mb = 4096
            config.batch_size = 100
        elif memory_gb >= 8:
            config.cache_memory_size = 2000
            config.max_memory_mb = 2048
            config.batch_size = 50
        else:
            config.cache_memory_size = 500
            config.max_memory_mb = 512
            config.batch_size = 20
            
        # Use uvloop if available
        try:
            import uvloop
            config.event_loop_policy = "uvloop"
        except ImportError:
            config.event_loop_policy = "asyncio"
            
        return config

# Global performance configuration
_perf_config = None

def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration"""
    global _perf_config
    if not _perf_config:
        # Try to load from environment, otherwise auto-tune
        if os.getenv("PERF_AUTO_TUNE", "true").lower() == "true":
            _perf_config = PerformanceTuner.auto_tune()
        else:
            _perf_config = PerformanceConfig.from_env()
    return _perf_config

# Apply performance settings
async def apply_performance_settings():
    """Apply all performance settings"""
    config = get_performance_config()
    
    # Set event loop policy
    if config.event_loop_policy == "uvloop":
        try:
            import uvloop
            uvloop.install()
            logger.info("Installed uvloop event loop")
        except ImportError:
            pass
            
    # Initialize caching if enabled
    if config.enable_caching:
        from src.utils.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        logger.info("Initialized cache manager")
        
    # Initialize connection pools if enabled
    if config.enable_connection_pooling:
        from src.utils.connection_pool import init_default_pools
        await init_default_pools({
            "database_url": os.getenv("DATABASE_URL"),
            "redis_url": os.getenv("REDIS_URL"),
            "mongodb_uri": os.getenv("MONGODB_URI")
        })
        logger.info("Initialized connection pools")
        
    # Set resource limits if enabled
    if config.enable_resource_limits:
        from src.utils.resource_limiter import set_process_limits
        set_process_limits()
        logger.info("Set resource limits")
        
    logger.info("Applied performance settings")
