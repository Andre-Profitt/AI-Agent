import os
import logging
import json
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import requests
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Whisper model (loaded once)
WHISPER_MODEL = None

def get_whisper_model():
    """Lazy load the Whisper model"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logger.info("Loading Whisper model...")
        try:
            import whisper
            WHISPER_MODEL = whisper.load_model("base")
        except ImportError:
            logger.warning("Whisper not available - install openai-whisper")
            return None
    return WHISPER_MODEL

@tool
def video_analyzer_production(video_url: str) -> str:
    """
    Production video analyzer that downloads and transcribes videos using yt-dlp and Whisper.
    Handles various video platforms including YouTube and googleusercontent.com URLs.
    
    Args:
        video_url (str): The video URL to analyze
        
    Returns:
        str: JSON string containing video metadata and transcript
    """
    try:
        logger.info(f"Production video analyzer called with URL: {video_url}")
        
        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            # Download video/audio
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(video_url, download=True)
                        title = info.get('title', 'Unknown')
                        duration = info.get('duration', 0)
                        
                        # Find the downloaded audio file
                        audio_files = list(Path(temp_dir).glob("*.mp3"))
                        if not audio_files:
                            return json.dumps({
                                "error": "Failed to download audio from video"
                            })
                        
                        audio_file = str(audio_files[0])
                        
                        # Transcribe with Whisper
                        logger.info("Transcribing audio with Whisper...")
                        model = get_whisper_model()
                        if model:
                            result = model.transcribe(audio_file)
                            
                            return json.dumps({
                                "metadata": {
                                    "title": title,
                                    "duration": duration,
                                    "url": video_url
                                },
                                "transcript": result["text"],
                                "source": "whisper_transcription"
                            }, indent=2)
                        else:
                            return json.dumps({
                                "error": "Whisper model not available"
                            })
                        
                    except Exception as e:
                        logger.error(f"yt-dlp download error: {e}")
                        return json.dumps({
                            "error": f"Failed to download video: {str(e)}"
                        })
            except ImportError:
                return json.dumps({
                    "error": "yt-dlp not available - install yt-dlp for video analysis"
                })
                    
    except Exception as e:
        logger.error(f"Error in production video analyzer: {e}")
        return json.dumps({"error": f"Failed to analyze video: {str(e)}"})

@tool
def chess_analyzer_production(fen_string: str, analysis_time_seconds: float = 3.0) -> str:
    """
    Production chess analyzer using Stockfish engine for grandmaster-level analysis.
    
    Args:
        fen_string (str): The chess position in Forsyth-Edwards Notation
        analysis_time_seconds (float): Time to spend on analysis
        
    Returns:
        str: The best move in algebraic notation with evaluation
    """
    try:
        logger.info(f"Production chess analyzer called with FEN: {fen_string}")
        
        # Validate FEN
        try:
            import chess
            board = chess.Board(fen_string)
        except ImportError:
            return "Error: python-chess not available - install python-chess"
        except ValueError as e:
            return f"Error: Invalid FEN string - {str(e)}"
        
        # Try to find Stockfish engine
        stockfish_paths = [
            "/usr/local/bin/stockfish",  # Homebrew on macOS
            "/usr/bin/stockfish",         # Linux
            "stockfish",                  # In PATH
            "/opt/homebrew/bin/stockfish" # Apple Silicon Homebrew
        ]
        
        engine_path = None
        for path in stockfish_paths:
            if os.path.exists(path) or subprocess.run(["which", path], capture_output=True).returncode == 0:
                engine_path = path
                break
        
        if not engine_path:
            # If Stockfish not found, provide instructions
            return ("Error: Stockfish engine not found. Please install it:\n"
                   "macOS: brew install stockfish\n"
                   "Linux: sudo apt-get install stockfish\n"
                   "Then retry the analysis.")
        
        # Analyze with Stockfish
        try:
            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                result = engine.analyse(board, chess.engine.Limit(time=analysis_time_seconds))
                
                best_move = result["pv"][0] if "pv" in result and result["pv"] else None
                evaluation = result.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                
                if best_move:
                    # Format evaluation
                    if evaluation.is_mate():
                        eval_str = f"Mate in {evaluation.mate()}"
                    else:
                        # Convert centipawns to pawns
                        cp_value = evaluation.white().score()
                        pawn_value = cp_value / 100 if cp_value is not None else 0
                        eval_str = f"{pawn_value:+.2f}"
                    
                    # Get move in SAN notation
                    san_move = board.san(best_move)
                    
                    return f"Best move: {san_move} (evaluation: {eval_str})\nUCI: {best_move.uci()}"
                else:
                    return "Error: No valid moves in this position"
        except Exception as e:
            return f"Error analyzing with Stockfish: {str(e)}"
                
    except Exception as e:
        logger.error(f"Error in production chess analyzer: {e}")
        return f"Error analyzing chess position: {str(e)}"

@tool
def install_stockfish() -> str:
    """
    Helper tool to install Stockfish chess engine on the system.
    """
    try:
        system = os.uname().sysname.lower()
        
        if system == "darwin":  # macOS
            # Check if Homebrew is installed
            brew_check = subprocess.run(["which", "brew"], capture_output=True)
            if brew_check.returncode != 0:
                return "Error: Homebrew not installed. Please install from https://brew.sh first"
            
            # Install Stockfish
            result = subprocess.run(["brew", "install", "stockfish"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return "Successfully installed Stockfish via Homebrew"
            else:
                return f"Error installing Stockfish: {result.stderr}"
                
        elif system == "linux":
            # Try apt-get
            result = subprocess.run(["sudo", "apt-get", "install", "-y", "stockfish"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return "Successfully installed Stockfish via apt-get"
            else:
                return f"Error installing Stockfish: {result.stderr}"
        else:
            return f"Unsupported system: {system}. Please install Stockfish manually."
            
    except Exception as e:
        logger.error(f"Error installing Stockfish: {e}")
        return f"Error installing Stockfish: {str(e)}"

@tool
def image_analyzer_chess(image_path: str) -> str:
    """
    Specialized image analyzer for chess positions that converts board images to FEN notation.
    
    Args:
        image_path (str): Path to the chess position image
        
    Returns:
        str: FEN notation of the chess position
    """
    try:
        logger.info(f"Chess image analyzer called with: {image_path}")
        
        # This would integrate with a vision model to analyze chess positions
        # For now, return a placeholder
        return ("Chess position analysis would be performed here.\n"
                "The image would be analyzed to extract piece positions and convert to FEN notation.")
                
    except Exception as e:
        logger.error(f"Error in chess image analyzer: {e}")
        return f"Error analyzing chess image: {str(e)}"

@tool
def music_discography_tool(artist_name: str) -> str:
    """
    Tool to search for music discography information.
    
    Args:
        artist_name (str): Name of the artist to search for
        
    Returns:
        str: Discography information
    """
    try:
        logger.info(f"Music discography tool called for: {artist_name}")
        
        # This would integrate with music APIs like Spotify, Last.fm, or MusicBrainz
        # For now, use web search as fallback
        search_query = f"{artist_name} discography albums songs"
        
        # Use Tavily search if available
        try:
            from src.tools import tavily_search_backoff
            return tavily_search_backoff(search_query)
        except ImportError:
            return f"Discography search for {artist_name} would be performed here."
            
    except Exception as e:
        logger.error(f"Error in music discography tool: {e}")
        return f"Error searching discography: {str(e)}"

@tool
def sports_data_tool(query: str) -> str:
    """
    Tool to search for sports statistics and data.
    
    Args:
        query (str): Sports-related query (e.g., "Olympic 2020 participation statistics")
        
    Returns:
        str: Sports data and statistics
    """
    try:
        logger.info(f"Sports data tool called with: {query}")
        
        # This would integrate with sports APIs
        # For now, use web search as fallback
        try:
            from src.tools import tavily_search_backoff
            return tavily_search_backoff(query)
        except ImportError:
            return f"Sports data search for '{query}' would be performed here."
            
    except Exception as e:
        logger.error(f"Error in sports data tool: {e}")
        return f"Error searching sports data: {str(e)}"

@tool
def text_reversal_tool(text: str) -> str:
    """
    Tool to reverse text, useful for solving puzzles and riddles.
    
    Args:
        text (str): Text to reverse
        
    Returns:
        str: Reversed text
    """
    try:
        logger.info(f"Text reversal tool called with: {text[:50]}...")
        
        # Simple text reversal
        reversed_text = text[::-1]
        
        return f"Original: {text}\nReversed: {reversed_text}"
        
    except Exception as e:
        logger.error(f"Error in text reversal tool: {e}")
        return f"Error reversing text: {str(e)}"

@tool
def mathematical_calculator(expression: str) -> str:
    """
    Advanced mathematical calculator that can handle complex expressions.
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        str: Result of the calculation
    """
    try:
        logger.info(f"Mathematical calculator called with: {expression}")
        
        # Use Python's eval with safe math functions
        import math
        
        # Define safe math functions
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        return f"Expression: {expression}\nResult: {result}"
        
    except Exception as e:
        logger.error(f"Error in mathematical calculator: {e}")
        return f"Error calculating expression: {str(e)}"

def get_production_tools() -> List[tool]:
    """Get all production tools"""
    return [
        video_analyzer_production,
        chess_analyzer_production,
        install_stockfish,
        image_analyzer_chess,
        music_discography_tool,
        sports_data_tool,
        text_reversal_tool,
        mathematical_calculator
    ]

class ToolsProduction:
    """Production tools class for importing"""
    
    def __init__(self):
        self.tools = get_production_tools()
    
    def get_tools(self) -> List[tool]:
        """Get all production tools"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[tool]:
        """Get a specific tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None 