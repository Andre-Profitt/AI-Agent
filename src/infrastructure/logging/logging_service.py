"""
LoggingService implementation for the AI Agent system.
"""

import logging
from typing import Optional, Any, Dict
from src.shared.types.config import LoggingConfig

class LoggingService:
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = logging.getLogger("ai_agent")
        self._configure_logger()

    def _configure_logger(self):
        self.logger.setLevel(self.config.level.value)
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

    async def initialize(self):
        # Placeholder for async initialization if needed
        pass

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.info(msg, extra=extra)

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.warning(msg, extra=extra)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.error(msg, extra=extra)

    async def log_interaction(self, **kwargs):
        self.logger.info(f"Interaction: {kwargs}")

    async def log_error(self, error_type: str, message: str, context: Optional[dict] = None):
        self.logger.error(f"{error_type}: {message} | Context: {context}") 