"""
Python interpreter tool implementation.
"""

import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class PythonInterpreterInput(BaseModel):
    """Input schema for Python interpreter tool."""
    code: str = Field(description="Python code to execute")

@tool
def python_interpreter(code: str) -> str:
    """
    Execute Python code and return the output.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Execution output or error message
    """
    try:
        # Capture stdout and stderr
        stdout = StringIO()
        stderr = StringIO()
        
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # Execute code
            exec(code, {'__builtins__': __builtins__})
            
        # Get output
        output = stdout.getvalue()
        error = stderr.getvalue()
        
        if error:
            return f"Error: {error}"
        return output or "Code executed successfully"
                
    except Exception as e:
        return f"Error executing Python code: {str(e)}" 