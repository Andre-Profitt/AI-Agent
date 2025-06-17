import os
import logging
import json
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import chess
import chess.engine
import yt_dlp
import whisper
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
        WHISPER_MODEL = whisper.load_model("base")
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
                    
                except yt_dlp.utils.DownloadError as e:
                    logger.error(f"yt-dlp download error: {e}")
                    return json.dumps({
                        "error": f"Failed to download video: {str(e)}"
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
            board = chess.Board(fen_string)
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
        return f"Error installing Stockfish: {str(e)}"

# Image analyzer with chess position detection
@tool
def image_analyzer_chess(image_path: str) -> str:
    """
    Analyzes chess board images and converts them to FEN notation.
    This is a placeholder that would use computer vision in production.
    
    Args:
        image_path (str): Path to the chess board image
        
    Returns:
        str: FEN notation of the detected position
    """
    try:
        logger.info(f"Chess image analyzer called with: {image_path}")
        
        # In production, this would use:
        # 1. OpenCV for board detection
        # 2. CNN for piece recognition
        # 3. FEN generation from detected pieces
        
        # For now, return a message indicating manual FEN input is needed
        return ("Chess board image detected. In production, this would use computer vision "
                "to detect the position. For now, please manually input the FEN notation "
                "of the position shown in the image.")
        
    except Exception as e:
        logger.error(f"Error in chess image analyzer: {e}")
        return f"Error analyzing chess image: {str(e)}" 