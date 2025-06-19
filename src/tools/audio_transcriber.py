"""
Audio transcriber tool implementation.
"""

import os
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class AudioTranscriberInput(BaseModel):
    """Input schema for audio transcriber tool."""
    audio_file: str = Field(description="Path to the audio file to transcribe")
    language: str = Field(default="en", description="Language code for transcription")

@tool
def audio_transcriber(audio_file: str, language: str = "en") -> str:
    """
    Transcribe audio file to text.
    
    Args:
        audio_file (str): Path to the audio file to transcribe
        language (str): Language code for transcription
        
    Returns:
        str: Transcribed text or error message
    """
    try:
        if not os.path.exists(audio_file):
            return f"Error: Audio file not found: {audio_file}"
            
        # Try to use OpenAI Whisper if available
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_file, language=language)
            return result["text"]
        except ImportError:
            # Fallback to a mock response
            return f"Mock transcription for {audio_file} in {language}. Audio transcription requires OpenAI Whisper to be installed."
                
    except Exception as e:
        return f"Error transcribing audio: {str(e)}" 