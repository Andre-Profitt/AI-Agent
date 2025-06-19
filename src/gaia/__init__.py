"""
GAIA (General AI Agent) Optimization Package

This package contains all GAIA-specific optimizations, tools, caching, metrics, and testing utilities.
"""

from .tools.gaia_specialized import (
    gaia_chess_analyzer,
    gaia_music_search, 
    gaia_country_code_lookup,
    gaia_mathematical_calculator
)

from .caching.gaia_cache import (
    GAIAResponseCache,
    GAIAQuestionCache,
    GAIAErrorCache,
    response_cache,
    question_cache,
    error_cache
)

from .metrics.gaia_metrics import (
    GAIAMetrics,
    gaia_metrics
)

from .testing.gaia_test_patterns import (
    GAIATestPattern,
    GAIATestPatterns
)

__all__ = [
    # Tools
    'gaia_chess_analyzer',
    'gaia_music_search',
    'gaia_country_code_lookup', 
    'gaia_mathematical_calculator',
    
    # Caching
    'GAIAResponseCache',
    'GAIAQuestionCache',
    'GAIAErrorCache',
    'response_cache',
    'question_cache',
    'error_cache',
    
    # Metrics
    'GAIAMetrics',
    'gaia_metrics',
    
    # Testing
    'GAIATestPattern',
    'GAIATestPatterns'
] 