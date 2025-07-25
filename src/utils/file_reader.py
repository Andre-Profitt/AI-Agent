# TODO: Fix undefined variables: BaseModel, Field, e, f, filename, lines, tool
"""
File reader tool implementation.
"""
from agent import lines
from benchmarks.cot_performance import filename

from src.database.models import tool


import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field
# TODO: Fix undefined variables: e, f, filename, lines, os
from pydantic import Field

from src.tools.base_tool import tool


class FileReaderInput(BaseModel):
    """Input schema for file reader tool."""
    filename: str = Field(description="Path to the file to read")
    lines: int = Field(default=-1, description="Number of lines to read (-1 for all)")

@tool
def file_reader(filename: str, lines: int = -1) -> str:
    """
    Read contents of a file.

    Args:
        filename (str): Path to the file to read
        lines (int): Number of lines to read (-1 for all)

    Returns:
        str: File contents or error message
    """
    try:
        if not os.path.exists(filename):
            return f"Error: File not found: {filename}"

        with open(filename, 'r', encoding='utf-8') as f:
            if lines == -1:
                return f.read()
            else:
                return ''.join(f.readline() for _ in range(lines))

    except Exception as e:
        return f"Error reading file: {str(e)}"
