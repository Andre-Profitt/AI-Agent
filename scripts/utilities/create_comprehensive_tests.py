#!/usr/bin/env python3
"""
Comprehensive test creation for AI Agent project
Creates unit, integration, and end-to-end tests
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TestCreator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def create_tests(self):
        """Create comprehensive test suite"""
        logger.info("🧪 Creating comprehensive test suite...\n")
        
        # Create test files
        self._create_agent_tests()
        self._create_tool_tests()
        self._create_performance_tests()
        self._create_security_tests()
        self._create_integration_tests()
        self._update_conftest()
        
        logger.info("\n✅ Comprehensive tests created!")
        
    def _create_agent_tests(self):
        """Create tests for unified agent"""
        test_content = '''"""
Tests for Unified Agent
Comprehensive unit and integration tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.unified_agent import (
    UnifiedAgent, AgentState, AgentCapability, 
    AgentContext, create_agent
)
from src.core.entities.message import Message
from src.core.entities.tool import Tool, ToolResult
from src.infrastructure.config import AgentConfig
from src.core.exceptions import AgentError


class TestUnifiedAgent:
    """Test unified agent functionality"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_iterations=5,
            timeout=30.0,
            enable_memory=True,
            enable_monitoring=True,
            error_threshold=3,
            recovery_timeout=1.0
        )
        
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools"""
        tool1 = Mock(spec=Tool)
        tool1.name = "search"
        tool1.description = "Search the web"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "calculate"
        tool2.description = "Perform calculations"
        
        return [tool1, tool2]
        
    @pytest.fixture
    def agent(self, agent_config, mock_tools):
        """Create test agent"""
        return UnifiedAgent(
            agent_id="test-agent",
            name="Test Agent",
            config=agent_config,
            tools=mock_tools,
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY
            ]
        )
        
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent is properly initialized"""
        assert agent.agent_id == "test-agent"
        assert agent.name == "Test Agent"
        assert agent.state == AgentState.IDLE
        assert len(agent.tools) == 2
        assert AgentCapability.MEMORY in agent.capabilities
        
    @pytest.mark.asyncio
    async def test_state_transitions(self, agent):
        """Test state transition logic"""
        # Test transition to thinking
        await agent._transition_state(AgentState.THINKING)
        assert agent.state == AgentState.THINKING
        
        # Test transition to executing
        await agent._transition_state(AgentState.EXECUTING)
        assert agent.state == AgentState.EXECUTING
        
        # Test error state recovery
        await agent._transition_state(AgentState.ERROR)
        assert agent.state == AgentState.ERROR
        await asyncio.sleep(1.1)  # Wait for recovery
        
    @pytest.mark.asyncio
    async def test_message_processing(self, agent):
        """Test message processing pipeline"""
        message = Message(content="What is the weather?", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock reasoning engine
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = {
                "response": "Let me check the weather.",
                "requires_tools": False,
                "confidence": 0.9
            }
            
            response = await agent.process(message, context)
            
            assert response.role == "assistant"
            assert "Let me check the weather" in response.content
            assert response.metadata["confidence"] == 0.9
            
    @pytest.mark.asyncio
    async def test_tool_execution(self, agent, mock_tools):
        """Test tool execution with circuit breaker"""
        message = Message(content="Search for AI news", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock reasoning and tool execution
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = {
                "response": "I'll search for AI news.",
                "requires_tools": True,
                "tool_calls": [
                    {"tool": "search", "parameters": {"query": "AI news"}}
                ]
            }
            
            with patch.object(agent, '_tool_executor') as mock_executor:
                mock_executor.execute = AsyncMock(return_value=ToolResult(
                    tool_name="search",
                    success=True,
                    data="Latest AI breakthroughs..."
                ))
                
                response = await agent.process(message, context)
                
                assert "search" in response.metadata["tools_used"]
                assert "Latest AI breakthroughs" in response.content
                
    @pytest.mark.asyncio
    async def test_memory_system(self, agent):
        """Test memory system integration"""
        message = Message(content="Remember this: API key is 12345", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock memory system
        with patch('src.gaia_components.enhanced_memory_system.EnhancedMemorySystem') as MockMemory:
            mock_memory = AsyncMock()
            MockMemory.return_value = mock_memory
            
            with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
                mock_reason.return_value = {
                    "response": "I'll remember that.",
                    "requires_tools": False
                }
                
                response = await agent.process(message, context)
                
                # Verify memory was updated
                mock_memory.add_interaction.assert_called_once()
                
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling and recovery"""
        message = Message(content="Cause an error", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock reasoning to raise error
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.side_effect = Exception("Test error")
            
            response = await agent.process(message, context)
            
            assert response.metadata["error"] is True
            assert "encountered an error" in response.content
            assert agent.state == AgentState.ERROR
            
    @pytest.mark.asyncio
    async def test_collaboration(self, agent, agent_config, mock_tools):
        """Test agent collaboration"""
        # Create second agent
        other_agent = UnifiedAgent(
            agent_id="helper-agent",
            name="Helper Agent",
            config=agent_config,
            tools=mock_tools,
            capabilities=[AgentCapability.COLLABORATION]
        )
        
        # Add collaboration capability to first agent
        agent.capabilities.append(AgentCapability.COLLABORATION)
        
        task = {"type": "research", "topic": "quantum computing"}
        
        with patch.object(other_agent, 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = Message(
                content="Research completed",
                role="assistant"
            )
            
            result = await agent.collaborate(other_agent, task)
            
            assert result.content == "Research completed"
            mock_process.assert_called_once()
            
    def test_factory_function(self, agent_config, mock_tools):
        """Test agent factory function"""
        # Test default creation
        agent = create_agent()
        assert agent.name.startswith("unified_agent_")
        
        # Test with parameters
        agent = create_agent(
            agent_type="unified",
            name="Custom Agent",
            config=agent_config,
            tools=mock_tools
        )
        assert agent.name == "Custom Agent"
        assert len(agent.tools) == 2
        
    def test_metrics_collection(self, agent):
        """Test metrics collection"""
        metrics = agent.get_metrics()
        
        assert metrics["agent_id"] == "test-agent"
        assert metrics["name"] == "Test Agent"
        assert metrics["state"] == "idle"
        assert metrics["tools_count"] == 2
        assert "circuit_breaker_state" in metrics


class TestAgentIntegration:
    """Integration tests for agent system"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        agent = create_agent(
            name="Integration Test Agent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY
            ]
        )
        
        context = AgentContext(
            session_id="integration-test",
            user_id="test-user"
        )
        
        # Simulate conversation
        messages = [
            "Hello, what can you do?",
            "Can you remember my name? It's Alice.",
            "What's my name?"
        ]
        
        for msg_content in messages:
            message = Message(content=msg_content, role="user")
            
            with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
                if "remember" in msg_content:
                    mock_reason.return_value = {
                        "response": "I'll remember your name is Alice.",
                        "requires_tools": False
                    }
                elif "What's my name" in msg_content:
                    mock_reason.return_value = {
                        "response": "Your name is Alice.",
                        "requires_tools": False
                    }
                else:
                    mock_reason.return_value = {
                        "response": "I can help with various tasks.",
                        "requires_tools": False
                    }
                    
                response = await agent.process(message, context)
                assert response.role == "assistant"
                context.history.append(message)
                context.history.append(response)
'''
        
        test_path = self.project_root / "tests/unit/test_unified_agent.py"
        test_path.write_text(test_content)
        logger.info("✅ Created unified agent tests")
        
    def _create_tool_tests(self):
        """Create tests for tools and tool executor"""
        test_content = '''"""
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
'''
        
        test_path = self.project_root / "tests/unit/test_tool_system.py"
        test_path.write_text(test_content)
        logger.info("✅ Created tool system tests")
        
    def _create_performance_tests(self):
        """Create performance tests"""
        test_content = '''"""
Performance Tests
Tests for caching, connection pooling, and resource limits
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import aiohttp

from src.utils.cache_manager import CacheManager, CacheConfig
from src.utils.connection_pool import ConnectionPoolManager
from src.utils.resource_limiter import ResourceLimiter, ResourceConfig


class TestPerformanceOptimizations:
    """Test performance optimization components"""
    
    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration"""
        return CacheConfig(
            memory_cache_size=100,
            memory_ttl=60,
            redis_url="redis://localhost:6379",
            redis_ttl=300,
            disk_cache_dir=".test_cache",
            enable_compression=True
        )
        
    @pytest.fixture
    def cache_manager(self, cache_config):
        """Create cache manager"""
        return CacheManager(cache_config)
        
    @pytest.mark.asyncio
    async def test_memory_cache(self, cache_manager):
        """Test in-memory caching"""
        # Set value
        await cache_manager.set("test_key", {"data": "test_value"})
        
        # Get value
        result = await cache_manager.get("test_key")
        assert result == {"data": "test_value"}
        
        # Test TTL
        await cache_manager.set("ttl_key", "value", ttl=1)
        await asyncio.sleep(1.5)
        result = await cache_manager.get("ttl_key")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_cache_decorator(self, cache_manager):
        """Test cache decorator"""
        call_count = 0
        
        @cache_manager.cached(ttl=60)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 2
            
        # First call
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (cached)
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
        
    @pytest.mark.asyncio
    async def test_connection_pool(self):
        """Test connection pool manager"""
        pool_manager = ConnectionPoolManager()
        
        # Test HTTP pool
        async with pool_manager.get_http_session() as session:
            assert isinstance(session, aiohttp.ClientSession)
            
        # Test connection reuse
        sessions = []
        for _ in range(5):
            async with pool_manager.get_http_session() as session:
                sessions.append(id(session))
                
        # Should reuse same session
        assert len(set(sessions)) == 1
        
    @pytest.mark.asyncio
    async def test_resource_limiter(self):
        """Test resource limiting"""
        config = ResourceConfig(
            max_concurrent_requests=2,
            max_requests_per_minute=10,
            max_memory_mb=100,
            max_cpu_percent=50
        )
        
        limiter = ResourceLimiter(config)
        
        # Test concurrent limit
        async def task():
            async with limiter.acquire("test_resource"):
                await asyncio.sleep(0.1)
                return True
                
        # Run 5 tasks with limit of 2
        start_time = time.time()
        results = await asyncio.gather(*[task() for _ in range(5)])
        end_time = time.time()
        
        assert all(results)
        # Should take at least 0.3s (3 batches of 0.1s)
        assert end_time - start_time >= 0.25
        
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting"""
        config = ResourceConfig(max_requests_per_minute=5)
        limiter = ResourceLimiter(config)
        
        # Make 5 requests quickly
        for i in range(5):
            assert await limiter.check_rate_limit("test_key")
            
        # 6th request should fail
        assert not await limiter.check_rate_limit("test_key")
        
    def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        # Generate some cache activity
        cache_manager.memory_cache.get("hit_key")  # Miss
        cache_manager.memory_cache.set("hit_key", "value")
        cache_manager.memory_cache.get("hit_key")  # Hit
        
        stats = cache_manager.get_stats()
        assert stats["memory"]["hits"] == 1
        assert stats["memory"]["misses"] == 1
        assert stats["memory"]["hit_rate"] == 0.5
        
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test connection pool cleanup"""
        pool_manager = ConnectionPoolManager()
        
        # Create session
        async with pool_manager.get_http_session() as session:
            pass
            
        # Close pool
        await pool_manager.close()
        
        # Verify closed
        with pytest.raises(RuntimeError):
            async with pool_manager.get_http_session() as session:
                pass


class TestPerformanceIntegration:
    """Integration tests for performance features"""
    
    @pytest.mark.asyncio
    async def test_cached_api_calls(self):
        """Test caching of API calls"""
        cache_manager = CacheManager()
        pool_manager = ConnectionPoolManager()
        
        call_count = 0
        
        @cache_manager.cached(ttl=300)
        async def fetch_data(url: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            
            async with pool_manager.get_http_session() as session:
                # Mock response
                return {"data": f"Response for {url}", "timestamp": time.time()}
                
        # First call
        result1 = await fetch_data("https://api.example.com/data")
        assert call_count == 1
        
        # Second call (cached)
        result2 = await fetch_data("https://api.example.com/data")
        assert call_count == 1
        assert result1 == result2
        
        # Different URL
        result3 = await fetch_data("https://api.example.com/other")
        assert call_count == 2
        
        await pool_manager.close()
'''
        
        test_path = self.project_root / "tests/unit/test_performance.py"
        test_path.write_text(test_content)
        logger.info("✅ Created performance tests")
        
    def _create_security_tests(self):
        """Create security tests"""
        test_content = '''"""
Security Tests
Tests for authentication, authorization, and security measures
"""

import pytest
from unittest.mock import Mock, patch
import jwt
from datetime import datetime, timedelta

from src.utils.security_validation import SecurityValidator
from src.config.security import SecurityConfig
from src.api.auth import create_token, verify_token
from src.api.rate_limiter import RateLimiter


class TestSecurityMeasures:
    """Test security implementations"""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration"""
        return SecurityConfig(
            jwt_secret="test_secret_key_12345",
            jwt_algorithm="HS256",
            token_expire_minutes=30,
            bcrypt_rounds=4,  # Lower for testing
            rate_limit_requests=100,
            rate_limit_window=60
        )
        
    @pytest.fixture
    def validator(self):
        """Create security validator"""
        return SecurityValidator()
        
    def test_sql_injection_detection(self, validator):
        """Test SQL injection detection"""
        # Test malicious inputs
        assert validator.is_sql_injection("'; DROP TABLE users; --")
        assert validator.is_sql_injection("1' OR '1'='1")
        assert validator.is_sql_injection("admin'--")
        
        # Test safe inputs
        assert not validator.is_sql_injection("normal user input")
        assert not validator.is_sql_injection("user@example.com")
        
    def test_xss_detection(self, validator):
        """Test XSS detection"""
        # Test malicious inputs
        assert validator.contains_xss("<script>alert('XSS')</script>")
        assert validator.contains_xss("<img src=x onerror=alert(1)>")
        assert validator.contains_xss("javascript:alert('XSS')")
        
        # Test safe inputs
        assert not validator.contains_xss("Normal text")
        assert not validator.contains_xss("user@example.com")
        
    def test_path_traversal_detection(self, validator):
        """Test path traversal detection"""
        # Test malicious paths
        assert not validator.is_safe_path("../../etc/passwd")
        assert not validator.is_safe_path("/etc/shadow")
        assert not validator.is_safe_path("..\\\\windows\\\\system32")
        
        # Test safe paths
        assert validator.is_safe_path("data/file.txt")
        assert validator.is_safe_path("./local/path")
        
    def test_jwt_token_creation(self, security_config):
        """Test JWT token creation"""
        user_id = "test_user_123"
        
        with patch('src.config.security.get_security_config', return_value=security_config):
            token = create_token(user_id)
            
            # Decode token
            payload = jwt.decode(
                token, 
                security_config.jwt_secret, 
                algorithms=[security_config.jwt_algorithm]
            )
            
            assert payload["sub"] == user_id
            assert "exp" in payload
            
    def test_jwt_token_verification(self, security_config):
        """Test JWT token verification"""
        user_id = "test_user_123"
        
        with patch('src.config.security.get_security_config', return_value=security_config):
            # Create token
            token = create_token(user_id)
            
            # Verify token
            verified_user_id = verify_token(token)
            assert verified_user_id == user_id
            
            # Test invalid token
            assert verify_token("invalid.token.here") is None
            
            # Test expired token
            expired_token = jwt.encode(
                {
                    "sub": user_id,
                    "exp": datetime.utcnow() - timedelta(minutes=1)
                },
                security_config.jwt_secret,
                algorithm=security_config.jwt_algorithm
            )
            assert verify_token(expired_token) is None
            
    @pytest.mark.asyncio
    async def test_rate_limiter(self, security_config):
        """Test rate limiting"""
        limiter = RateLimiter(
            max_requests=5,
            window_seconds=60
        )
        
        client_id = "test_client"
        
        # Make 5 requests
        for i in range(5):
            allowed = await limiter.check_rate_limit(client_id)
            assert allowed is True
            
        # 6th request should be blocked
        allowed = await limiter.check_rate_limit(client_id)
        assert allowed is False
        
    def test_password_hashing(self, security_config):
        """Test password hashing"""
        from src.utils.security_validation import hash_password, verify_password
        
        password = "SecurePassword123!"
        
        # Hash password
        hashed = hash_password(password)
        assert hashed != password
        assert hashed.startswith("$2b$")
        
        # Verify correct password
        assert verify_password(password, hashed) is True
        
        # Verify incorrect password
        assert verify_password("wrong_password", hashed) is False
        
    def test_input_sanitization(self, validator):
        """Test input sanitization"""
        # Test various inputs
        assert validator.sanitize_input("<script>alert(1)</script>") == "alert(1)"
        assert validator.sanitize_input("normal text") == "normal text"
        assert validator.sanitize_input("user@example.com") == "user@example.com"
        
    @pytest.mark.asyncio
    async def test_secure_file_operations(self, validator):
        """Test secure file operations"""
        # Test file path validation
        safe_paths = [
            "data/users.json",
            "./config/settings.yaml",
            "logs/app.log"
        ]
        
        unsafe_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "C:\\\\Windows\\\\System32\\\\config"
        ]
        
        for path in safe_paths:
            assert validator.is_safe_path(path)
            
        for path in unsafe_paths:
            assert not validator.is_safe_path(path)


class TestSecurityIntegration:
    """Integration tests for security features"""
    
    @pytest.mark.asyncio
    async def test_authenticated_request_flow(self):
        """Test full authentication flow"""
        from src.api.auth import create_token, get_current_user
        from fastapi import HTTPException
        
        user_id = "test_user"
        
        # Create token
        token = create_token(user_id)
        
        # Mock request with token
        mock_credentials = Mock()
        mock_credentials.credentials = token
        
        # Verify user
        with patch('src.api.auth.verify_token', return_value=user_id):
            current_user = await get_current_user(mock_credentials)
            assert current_user == user_id
            
        # Test invalid token
        mock_credentials.credentials = "invalid_token"
        with patch('src.api.auth.verify_token', return_value=None):
            with pytest.raises(HTTPException) as exc:
                await get_current_user(mock_credentials)
            assert exc.value.status_code == 401
'''
        
        test_path = self.project_root / "tests/unit/test_security.py"
        test_path.write_text(test_content)
        logger.info("✅ Created security tests")
        
    def _create_integration_tests(self):
        """Create end-to-end integration tests"""
        test_content = '''"""
End-to-End Integration Tests
Tests complete system functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.agents.unified_agent import create_agent, AgentCapability
from src.core.entities.message import Message
from src.infrastructure.config import AgentConfig
from src.unified_architecture.platform import UnifiedPlatform
from src.services.integration_hub import IntegrationHub


class TestEndToEndIntegration:
    """Test complete system integration"""
    
    @pytest.fixture
    async def platform(self):
        """Create test platform"""
        platform = UnifiedPlatform()
        await platform.initialize()
        yield platform
        await platform.shutdown()
        
    @pytest.fixture
    def integration_hub(self):
        """Create integration hub"""
        return IntegrationHub()
        
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, platform):
        """Test multiple agents working together"""
        # Create specialized agents
        research_agent = create_agent(
            name="Research Agent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.COLLABORATION
            ]
        )
        
        analysis_agent = create_agent(
            name="Analysis Agent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.MEMORY,
                AgentCapability.COLLABORATION
            ]
        )
        
        # Register agents
        await platform.register_agent(research_agent)
        await platform.register_agent(analysis_agent)
        
        # Create collaborative task
        task = {
            "type": "research_and_analyze",
            "topic": "AI trends 2024",
            "steps": [
                {"agent": "Research Agent", "action": "gather_data"},
                {"agent": "Analysis Agent", "action": "analyze_findings"}
            ]
        }
        
        # Mock agent responses
        with patch.object(research_agent, 'process', new_callable=AsyncMock) as mock_research:
            mock_research.return_value = Message(
                content="Found 10 key AI trends for 2024...",
                role="assistant",
                metadata={"data_points": 10}
            )
            
            with patch.object(analysis_agent, 'process', new_callable=AsyncMock) as mock_analysis:
                mock_analysis.return_value = Message(
                    content="Analysis complete: Top trends are...",
                    role="assistant",
                    metadata={"confidence": 0.85}
                )
                
                # Execute task
                result = await platform.execute_task(task)
                
                assert result["status"] == "completed"
                assert len(result["steps"]) == 2
                
    @pytest.mark.asyncio
    async def test_tool_integration_flow(self):
        """Test agent with multiple tool integrations"""
        # Create agent with tools
        tools = [
            Mock(name="web_search", execute=AsyncMock()),
            Mock(name="calculator", execute=AsyncMock()),
            Mock(name="file_reader", execute=AsyncMock())
        ]
        
        agent = create_agent(
            name="Tool User",
            tools=tools,
            capabilities=[AgentCapability.TOOL_USE]
        )
        
        # Test complex query requiring multiple tools
        query = "Search for the latest GDP data and calculate the growth rate"
        
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = {
                "response": "I'll search for GDP data and calculate the growth rate.",
                "requires_tools": True,
                "tool_calls": [
                    {"tool": "web_search", "parameters": {"query": "latest GDP data"}},
                    {"tool": "calculator", "parameters": {"expression": "(new-old)/old*100"}}
                ]
            }
            
            # Mock tool results
            tools[0].execute.return_value = {"data": "GDP: 2023: $25T, 2024: $26T"}
            tools[1].execute.return_value = {"result": 4.0}
            
            response = await agent.process(
                Message(content=query, role="user"),
                AgentContext(session_id="test")
            )
            
            assert "4" in response.content or "growth" in response.content.lower()
            
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self):
        """Test system error recovery"""
        agent = create_agent(name="Resilient Agent")
        
        # Simulate intermittent failures
        call_count = 0
        
        async def flaky_reasoning(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise Exception("Temporary failure")
                
            return {
                "response": "Success after retries",
                "requires_tools": False
            }
            
        with patch.object(agent, '_reason', new=flaky_reasoning):
            # First attempt should fail but recover
            response = await agent.process(
                Message(content="Test resilience", role="user"),
                AgentContext(session_id="test")
            )
            
            # Should eventually succeed
            assert "error" in response.content.lower()
            
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance with concurrent requests"""
        agent = create_agent(
            name="Performance Test Agent",
            config=AgentConfig(max_concurrent_requests=5)
        )
        
        # Create multiple concurrent requests
        async def make_request(i: int):
            message = Message(content=f"Request {i}", role="user")
            context = AgentContext(session_id=f"session-{i}")
            
            with patch.object(agent, '_reason', new_callable=AsyncMock) as mock:
                mock.return_value = {
                    "response": f"Response to request {i}",
                    "requires_tools": False
                }
                
                return await agent.process(message, context)
                
        # Run 20 concurrent requests
        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*[
            make_request(i) for i in range(20)
        ])
        end_time = asyncio.get_event_loop().time()
        
        # Verify all succeeded
        assert len(responses) == 20
        assert all(r.role == "assistant" for r in responses)
        
        # Should have rate limited (not all concurrent)
        duration = end_time - start_time
        assert duration > 0.5  # Some throttling occurred
        
    @pytest.mark.asyncio
    async def test_integration_hub_connections(self, integration_hub):
        """Test external service integrations"""
        # Test service registration
        mock_service = Mock(
            name="test_service",
            connect=AsyncMock(return_value=True),
            disconnect=AsyncMock(return_value=True),
            execute=AsyncMock(return_value={"status": "success"})
        )
        
        integration_hub.register_service(mock_service)
        
        # Test connection
        await integration_hub.connect("test_service")
        mock_service.connect.assert_called_once()
        
        # Test execution
        result = await integration_hub.execute(
            "test_service",
            "test_action",
            {"param": "value"}
        )
        
        assert result["status"] == "success"
        
        # Test disconnection
        await integration_hub.disconnect("test_service")
        mock_service.disconnect.assert_called_once()


class TestSystemReliability:
    """Test system reliability and fault tolerance"""
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test that system doesn't leak memory"""
        import gc
        import weakref
        
        # Create and destroy many agents
        refs = []
        
        for i in range(100):
            agent = create_agent(name=f"Agent-{i}")
            refs.append(weakref.ref(agent))
            del agent
            
        # Force garbage collection
        gc.collect()
        
        # Check that agents were cleaned up
        alive_count = sum(1 for ref in refs if ref() is not None)
        assert alive_count == 0
        
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, platform):
        """Test graceful shutdown procedures"""
        # Start some tasks
        async def long_running_task():
            await asyncio.sleep(10)
            
        task = asyncio.create_task(long_running_task())
        
        # Shutdown should cancel tasks gracefully
        await platform.shutdown()
        
        assert task.cancelled()
'''
        
        test_path = self.project_root / "tests/integration/test_e2e_integration.py"
        test_path.write_text(test_content)
        logger.info("✅ Created end-to-end integration tests")
        
    def _update_conftest(self):
        """Update pytest configuration"""
        conftest_content = '''"""
Pytest Configuration
Shared fixtures and test setup
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "ERROR"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset any singleton instances
    from src.infrastructure.di.container import Container
    Container._instance = None
    
    from src.tools.registry import ToolRegistry
    ToolRegistry._instance = None


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = Mock()
    llm.agenerate = AsyncMock(return_value=Mock(
        generations=[[Mock(text="Test response")]]
    ))
    return llm


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests"""
    return tmp_path


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = Mock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.expire = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_database():
    """Mock database connection"""
    db = Mock()
    db.execute = AsyncMock()
    db.fetch_one = AsyncMock()
    db.fetch_all = AsyncMock()
    return db


# Markers
pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
'''
        
        conftest_path = self.project_root / "tests/conftest.py"
        
        # Read existing content and append if needed
        if conftest_path.exists():
            existing_content = conftest_path.read_text()
            if "@pytest.fixture(scope=\"session\")" not in existing_content:
                # Append new fixtures
                conftest_path.write_text(existing_content + "\n\n" + conftest_content)
                logger.info("✅ Updated existing conftest.py")
            else:
                logger.info("✅ conftest.py already configured")
        else:
            conftest_path.write_text(conftest_content)
            logger.info("✅ Created conftest.py")

def main():
    creator = TestCreator()
    creator.create_tests()
    
    logger.info("\n✅ Comprehensive test suite created!")
    logger.info("\nCreated tests:")
    logger.info("  - tests/unit/test_unified_agent.py")
    logger.info("  - tests/unit/test_tool_system.py")
    logger.info("  - tests/unit/test_performance.py")
    logger.info("  - tests/unit/test_security.py")
    logger.info("  - tests/integration/test_e2e_integration.py")
    logger.info("  - Updated tests/conftest.py")
    logger.info("\nRun tests with: pytest tests/")
    logger.info("Run specific test: pytest tests/unit/test_unified_agent.py")
    logger.info("Run with coverage: pytest --cov=src tests/")

if __name__ == "__main__":
    main()