"""
Advanced file reader tool with additional features.
"""

import os
from typing import Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class AdvancedFileReaderInput(BaseModel):
    """Input schema for advanced file reader tool."""
    filename: str = Field(description="Path to the file to read")
    lines: int = Field(default=-1, description="Number of lines to read (-1 for all)")
    encoding: str = Field(default="utf-8", description="File encoding")
    start_line: int = Field(default=1, description="Starting line number (1-based)")

@tool
def advanced_file_reader(
    filename: str, 
    lines: int = -1,
    encoding: str = "utf-8",
    start_line: int = 1
) -> str:
    """
    Advanced file reader with support for encoding and line ranges.
    
    Args:
        filename (str): Path to the file to read
        lines (int): Number of lines to read (-1 for all)
        encoding (str): File encoding
        start_line (int): Starting line number (1-based)
        
    Returns:
        str: File contents or error message
    """
    try:
        if not os.path.exists(filename):
            return f"Error: File not found: {filename}"
            
        with open(filename, 'r', encoding=encoding) as f:
            # Skip to start line
            for _ in range(start_line - 1):
                f.readline()
                
            if lines == -1:
                return f.read()
            else:
                return ''.join(f.readline() for _ in range(lines))
                
    except Exception as e:
        return f"Error reading file: {str(e)}" 