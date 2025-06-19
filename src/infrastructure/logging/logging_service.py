"""
LoggingService implementation for the AI Agent system.
"""

import logging
from typing import Optional, Any, Dict
from src.shared.types.config import LoggingConfig
from src.shared.types.di_types import (
    ConfigurationService, DatabaseClient, CacheClient, LoggingService
from typing import Optional, Dict, Any, List, Union, Tuple

class LoggingService:
    def __init__(self, config: LoggingConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("ai_agent")
        self._configure_logger()


    async def _get_safe_config_value(self, key: str) -> str:
        """Safely get configuration value with error handling"""
        try:
            parts = key.split('_')
            if len(parts) == 2:
                service, attr = parts
                config_obj = getattr(self.config, service, None)
                if config_obj:
                    return getattr(config_obj, attr, "")
            
            # Direct attribute access
            return getattr(self.config, key, "")
        except Exception as e:
            logger.error("Config access failed", extra={"key": key, "error": str(e)})
            return ""
    def _configure_logger(self) -> Any:
        self.logger.setLevel(await self._get_safe_config_value("level_value"))
        formatter = logging.Formatter(self.config.format, self.config.date_format)
        if self.config.enable_file_logging:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    async def initialize(self) -> Any:
        # Placeholder for async initialization if needed
        pass

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> Any:
        self.logger.info(msg, extra=extra)

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> Any:
        self.logger.warning(msg, extra=extra)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> Any:
        self.logger.error(msg, extra=extra)

    async def log_interaction(self, **kwargs) -> Any:
        self.logger.info("Interaction: {}", extra={"kwargs": kwargs})

    async def log_error(self, error_type: str, message: str, context: Optional[dict] = None) -> Any:
        self.logger.error("{}: {} | Context: {}", extra={"error_type": error_type, "message": message, "context": context}) 