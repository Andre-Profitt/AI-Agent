"""
Tools package for the AI Agent.
Contains base tool implementations and specialized tools.
"""

import logging

logger = logging.getLogger(__name__)

# Import base tools with fallbacks
try:
    from .base_tool import BaseTool
    from .file_reader import file_reader
    from .advanced_file_reader import advanced_file_reader
    from .semantic_search_tool import semantic_search_tool
    from .python_interpreter import python_interpreter
    from .tavily_search import tavily_search, tavily_search_backoff
    from .weather import get_weather
    from .audio_transcriber import audio_transcriber
    BASE_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Base tools not available: {e}")
    BASE_TOOLS_AVAILABLE = False
    # Create dummy imports
    BaseTool = None
    file_reader = None
    advanced_file_reader = None
    semantic_search_tool = None
    python_interpreter = None
    tavily_search = None
    tavily_search_backoff = None
    get_weather = None
    audio_transcriber = None

# Import enhanced tools
try:
    from ..tools_enhanced import get_enhanced_tools, gaia_video_analyzer, chess_logic_tool, web_researcher, abstract_reasoning_tool
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced tools not available")
    ENHANCED_TOOLS_AVAILABLE = False
    get_enhanced_tools = None
    gaia_video_analyzer = None
    chess_logic_tool = None
    web_researcher = None
    abstract_reasoning_tool = None

# Import production tools
try:
    from ..tools_production import get_production_tools, video_analyzer_production, chess_analyzer_production, install_stockfish
    PRODUCTION_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("Production tools not available")
    PRODUCTION_TOOLS_AVAILABLE = False
    get_production_tools = None
    video_analyzer_production = None
    chess_analyzer_production = None
    install_stockfish = None

# Import interactive tools
try:
    from ..tools_interactive import get_interactive_tools, set_clarification_callback, get_pending_clarifications
    INTERACTIVE_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("Interactive tools not available")
    INTERACTIVE_TOOLS_AVAILABLE = False
    get_interactive_tools = None
    set_clarification_callback = None
    get_pending_clarifications = None

# Import introspection tools
try:
    from ..tools_introspection import register_tools, get_tool_schema, analyze_tool_error, get_available_tools
    INTROSPECTION_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("Introspection tools not available")
    INTROSPECTION_TOOLS_AVAILABLE = False
    register_tools = None
    get_tool_schema = None
    analyze_tool_error = None
    get_available_tools = None

# Tools package for GAIA agent

__all__ = [
    'BaseTool',
    'file_reader',
    'advanced_file_reader',
    'semantic_search_tool',
    'python_interpreter',
    'tavily_search',
    'tavily_search_backoff',
    'get_weather',
    'audio_transcriber',
    # Enhanced tools
    'get_enhanced_tools',
    'gaia_video_analyzer',
    'chess_logic_tool',
    'web_researcher',
    'abstract_reasoning_tool',
    # Production tools
    'get_production_tools',
    'video_analyzer_production',
    'chess_analyzer_production',
    'install_stockfish',
    # Interactive tools
    'get_interactive_tools',
    'set_clarification_callback',
    'get_pending_clarifications',
    # Introspection tools
    'register_tools',
    'get_tool_schema',
    'analyze_tool_error',
    'get_available_tools',
    # Availability flags
    'BASE_TOOLS_AVAILABLE',
    'ENHANCED_TOOLS_AVAILABLE',
    'PRODUCTION_TOOLS_AVAILABLE',
    'INTERACTIVE_TOOLS_AVAILABLE',
    'INTROSPECTION_TOOLS_AVAILABLE'
] 