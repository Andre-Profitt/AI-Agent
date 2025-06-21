"""
Tests for Tool System
Tests tool execution, registry, and error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.core.entities.tool import Tool, ToolResult
from src.application.tools.tool_executor import ToolExecutor
from src.tools.registry import ToolRegistry
from src.core.exceptions import ToolError


class TestToolSystem:
    """Test tool functionality"""
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool"""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.parameters = {
            "input": {"type": "string", "required": True}
        }
        tool.execute = AsyncMock(return_value=ToolResult(
            tool_name="test_tool",
            success=True,
            data="Test result"
        ))
        return tool
        
    @pytest.fixture
    def tool_executor(self, mock_tool):
        """Create tool executor"""
        return ToolExecutor([mock_tool])
        
    @pytest.mark.asyncio
    async def test_tool_execution_success(self, tool_executor):
        """Test successful tool execution"""
        result = await tool_executor.execute("test_tool", {"input": "test"})
        
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.data == "Test result"
        
    @pytest.mark.asyncio
    async def test_tool_execution_failure(self, tool_executor, mock_tool):
        """Test tool execution failure"""
        mock_tool.execute.side_effect = Exception("Tool error")
        
        result = await tool_executor.execute("test_tool", {"input": "test"})
        
        assert result.success is False
        assert result.error == "Tool error"
        
    @pytest.mark.asyncio
    async def test_tool_not_found(self, tool_executor):
        """Test execution of non-existent tool"""
        with pytest.raises(ToolError):
            await tool_executor.execute("non_existent", {})
            
    @pytest.mark.asyncio
    async def test_invalid_parameters(self, tool_executor):
        """Test tool execution with invalid parameters"""
        # Missing required parameter
        with pytest.raises(ToolError):
            await tool_executor.execute("test_tool", {})
            
    def test_tool_registry(self, mock_tool):
        """Test tool registry functionality"""
        registry = ToolRegistry()
        
        # Register tool
        registry.register(mock_tool)
        assert registry.get("test_tool") == mock_tool
        
        # List tools
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        
        # Unregister tool
        registry.unregister("test_tool")
        assert registry.get("test_tool") is None
        
    @pytest.mark.asyncio
    async def test_tool_timeout(self, mock_tool):
        """Test tool execution timeout"""
        async def slow_execute(*args, **kwargs):
            import asyncio
            await asyncio.sleep(5)
            return ToolResult(tool_name="test_tool", success=True)
            
        mock_tool.execute = slow_execute
        executor = ToolExecutor([mock_tool], timeout=1.0)
        
        result = await executor.execute("test_tool", {"input": "test"})
        
        assert result.success is False
        assert "timeout" in result.error.lower()
        
    @pytest.mark.asyncio
    async def test_tool_retry(self, mock_tool):
        """Test tool retry mechanism"""
        call_count = 0
        
        async def flaky_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return ToolResult(tool_name="test_tool", success=True, data="Success")
            
        mock_tool.execute = flaky_execute
        executor = ToolExecutor([mock_tool], retry_attempts=3)
        
        result = await executor.execute("test_tool", {"input": "test"})
        
        assert result.success is True
        assert call_count == 3
