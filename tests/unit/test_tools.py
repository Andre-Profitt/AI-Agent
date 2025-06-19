"""
Unit tests for tool functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools import (
import logging

logger = logging.getLogger(__name__)

    web_researcher,
    semantic_search_tool,
    python_interpreter,
    tavily_search,
    file_reader,
    get_enhanced_tools,
    get_production_tools,
    get_interactive_tools,
    get_tool_schema,
    get_available_tools
)
import sys


class TestWebResearcher:
    """Test web researcher tool"""
    
    @patch('src.tools.WikipediaAPIWrapper')
    @patch('src.tools.WEB_SCRAPING_AVAILABLE', True)
    def test_web_researcher_success(self, mock_wikipedia):
        """Test successful web research"""
        # Mock the Wikipedia API
        mock_api = Mock()
        mock_api.run.return_value = "Test article content"
        mock_wikipedia.return_value = mock_api
        
        result = web_researcher("Python programming")
        
        assert "Test article content" in result
        mock_api.run.assert_called_once_with("Python programming")
    
    @patch('src.tools.WikipediaAPIWrapper')
    def test_web_researcher_error(self, mock_wikipedia):
        """Test web researcher error handling"""
        # Mock an error
        mock_api = Mock()
        mock_api.run.side_effect = Exception("API error")
        mock_wikipedia.return_value = mock_api
        
        result = web_researcher("Test query")
        
        assert "error" in result.lower()


class TestSemanticSearchTool:
    """Test semantic search tool"""
    
    @patch('src.tools.SemanticSearchEngine')
    def test_semantic_search_success(self, mock_engine):
        """Test successful semantic search"""
        # Mock the search engine
        mock_search = Mock()
        mock_search.search.return_value = [
            {"content": "Result 1", "score": 0.9},
            {"content": "Result 2", "score": 0.8}
        ]
        mock_engine.return_value = mock_search
        
        # Call as a StructuredTool: pass all args as a dict
        result = semantic_search_tool({
            "query": "test query",
            "filename": "test.csv",
            "top_k": 2
        })
        
        assert "Result 1" in result
        assert "Result 2" in result
    
    def test_semantic_search_invalid_params(self):
        """Test semantic search with invalid parameters"""
        result = semantic_search_tool("", filename="test.csv")
        assert "error" in result.lower() or "invalid" in result.lower()


class TestPythonInterpreter:
    """Test Python interpreter tool"""
    
    def test_python_interpreter_simple(self):
        """Test simple Python code execution"""
        code = "result = 2 + 2\nlogger.info("Value", extra={"value": result})"
        result = python_interpreter(code)
        
        assert "4" in result
    
    def test_python_interpreter_imports(self):
        """Test Python with imports"""
        code = "import math\nresult = math.sqrt(16)\nlogger.info("Value", extra={"value": result})"
        result = python_interpreter(code)
        
        assert "4.0" in result
    
    def test_python_interpreter_error(self):
        """Test Python interpreter error handling"""
        code = "1/0"  # Division by zero
        result = python_interpreter(code)
        
        assert "error" in result.lower() or "zerodivision" in result.lower()
    
    def test_python_interpreter_timeout(self):
        """Test Python interpreter timeout"""
        code = "import time\ntime.sleep(100)"  # Should timeout
        result = python_interpreter(code)
        
        # Should contain timeout or error message
        assert any(word in result.lower() for word in ["timeout", "error", "exceeded"])


class TestTavilySearch:
    """Test Tavily search tool"""
    
    @patch('src.tools.os.getenv')
    @patch('src.tools.TavilySearchAPIWrapper')
    def test_tavily_search_success(self, mock_tavily, mock_getenv):
        """Test successful Tavily search"""
        # Mock environment variable
        mock_getenv.return_value = "test-api-key"
        
        # Mock Tavily API
        mock_api = Mock()
        mock_api.search.return_value = "Search results"
        mock_tavily.return_value = mock_api
        
        result = tavily_search("test query", max_results=3)
        
        assert "Search results" in result
        mock_api.search.assert_called_once()
    
    @patch('src.tools.os.getenv')
    def test_tavily_search_no_api_key(self, mock_getenv):
        """Test Tavily search without API key"""
        # No API key
        mock_getenv.return_value = None
        
        result = tavily_search("test query")
        
        assert "api key" in result.lower() or "not available" in result.lower()


class TestFileReader:
    """Test file reader tool"""
    
    @patch('builtins.open', create=True)
    def test_file_reader_text_file(self, mock_open):
        """Test reading a text file"""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = "File content"
        
        result = file_reader("test.txt")
        
        assert "File content" in result
        mock_open.assert_called_once_with("test.txt", 'r', encoding='utf-8')
    
    @patch('builtins.open', create=True)
    def test_file_reader_with_lines(self, mock_open):
        """Test reading specific lines from a file"""
        # Mock file with multiple lines
        mock_file = MagicMock()
        mock_file.readlines.return_value = [
            "Line 1\n",
            "Line 2\n", 
            "Line 3\n",
            "Line 4\n",
            "Line 5\n"
        ]
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = file_reader("test.txt", lines=3)
        
        # Should only contain first 3 lines
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" not in result
    
    def test_file_reader_nonexistent_file(self):
        """Test reading non-existent file"""
        result = file_reader("/nonexistent/file.txt")
        
        assert "error" in result.lower() or "not found" in result.lower()
    
    @patch('src.tools.PyPDF2')
    @patch('builtins.open', create=True)
    def test_file_reader_pdf(self, mock_open, mock_pypdf):
        """Test reading PDF file"""
        # Mock PDF reader
        mock_pdf_reader = Mock()
        mock_pdf_reader.pages = [Mock(extract_text=lambda: "Page 1 content")]
        mock_pypdf.PdfReader.return_value = mock_pdf_reader
        
        result = file_reader("test.pdf")
        
        assert "Page 1 content" in result 

def test_enhanced_tools_discoverable():
    tools = get_enhanced_tools()
    assert len(tools) > 0
    for tool in tools:
        assert hasattr(tool, 'name')
        schema = get_tool_schema(tool.name)
        assert 'name' in schema

def test_production_tools_discoverable():
    tools = get_production_tools()
    assert len(tools) > 0
    for tool in tools:
        assert hasattr(tool, 'name')
        schema = get_tool_schema(tool.name)
        assert 'name' in schema

def test_interactive_tools_discoverable():
    tools = get_interactive_tools()
    assert len(tools) > 0
    for tool in tools:
        assert hasattr(tool, 'name')
        schema = get_tool_schema(tool.name)
        assert 'name' in schema

def test_introspection_lists_tools():
    available = get_available_tools()
    assert isinstance(available, list)
    assert len(available) > 0 