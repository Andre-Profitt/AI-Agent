"""Structured logging utilities"""

import logging
import sys
from typing import Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name or __name__)