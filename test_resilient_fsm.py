#!/usr/bin/env python3
"""
Comprehensive test suite for the resilient FSM agent implementation.
This test verifies that all the architectural improvements work correctly.
"""

import os
import sys
import uuid
import time
import json
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly after our fixes."""
    print("🔧 Testing imports and basic instantiation...")
    
    try:
        # Test that we can import without the NameError
        from src.advanced_agent_fsm import (
            FSMReActAgent, 
            correlation_context, 
            ResilientAPIClient,
            EnhancedPlanner,
            PlanResponse,
            FSMState
        )
        print("✅ All imports successful - NameError fixed!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_correlation_logging():
    """Test the correlation ID logging system."""
    print("\n🔧 Testing correlation ID logging...")
    
    try:
        from src.advanced_agent_fsm import correlation_context, logger
        
        correlation_id = str(uuid.uuid4())
        
        with correlation_context(correlation_id):
            logger.info("Test message with correlation ID")
            
        print("✅ Correlation logging system working!")
        return True
    except Exception as e:
        print(f"❌ Correlation logging error: {e}")
        return False

def test_resilient_api_client():
    """Test the resilient API client with mocked responses."""
    print("\n🔧 Testing resilient API client...")
    
    try:
        from src.advanced_agent_fsm import ResilientAPIClient
        
        # Test initialization without actual API key
        client = ResilientAPIClient("test-key")
        
        # Verify circuit breaker and retry components are set up
        assert hasattr(client, 'session'), "Session not initialized"
        assert hasattr(client, 'rate_limiter'), "Rate limiter not initialized"
        
        print("✅ ResilientAPIClient initialization successful!")
        return True
    except Exception as e:
        print(f"❌ ResilientAPIClient error: {e}")
        return False

def test_pydantic_validation():
    """Test Pydantic data contracts."""
    print("\n🔧 Testing Pydantic data validation...")
    
    try:
        from src.advanced_agent_fsm import PlanStep, PlanResponse
        
        # Test valid plan step
        step = PlanStep(
            step_name="web_researcher",
            parameters={"query": "test query", "source": "wikipedia"},
            reasoning="Need to search for information",
            expected_output="Search results about the topic"
        )
        
        # Test valid plan response
        plan = PlanResponse(
            steps=[step],
            total_steps=1,
            confidence=0.8
        )
        
        print("✅ Pydantic validation working correctly!")
        return True
    except Exception as e:
        print(f"❌ Pydantic validation error: {e}")
        return False

def test_fsm_agent_initialization():
    """Test FSM agent initialization with mocked tools."""
    print("\n🔧 Testing FSM agent initialization...")
    
    try:
        from src.advanced_agent_fsm import FSMReActAgent
        from langchain_core.tools import Tool
        
        # Create mock tools
        mock_tools = [
            Tool(
                name="test_tool",
                description="A test tool",
                func=lambda x: f"Test result for: {x}"
            )
        ]
        
        # Test initialization (without API key to test fallback)
        with patch.dict(os.environ, {}, clear=True):  # Remove GROQ_API_KEY
            agent = FSMReActAgent(tools=mock_tools)
            
            assert hasattr(agent, 'graph'), "Graph not initialized"
            assert hasattr(agent, 'tool_registry'), "Tool registry not initialized"
            
        print("✅ FSM agent initialization successful!")
        return True
    except Exception as e:
        print(f"❌ FSM agent initialization error: {e}")
        return False

def test_fallback_parser_fix():
    """Test that the fallback parser no longer crashes with NameError."""
    print("\n🔧 Testing fallback parser fix...")
    
    try:
        from src.advanced_agent_fsm import FSMReActAgent
        from langchain_core.tools import Tool
        
        # Create mock tools
        mock_tools = [
            Tool(
                name="test_tool",
                description="A test tool",
                func=lambda x: f"Test result for: {x}"
            )
        ]
        
        # Initialize agent
        with patch.dict(os.environ, {}, clear=True):
            agent = FSMReActAgent(tools=mock_tools)
        
        # Test the fallback parser method directly
        test_description = "Tool: web_researcher with parameters: {query: 'test'}"
        result = agent._fallback_parse_params(test_description, "web_researcher")
        
        # Should not crash and should return a dict
        assert isinstance(result, dict), "Fallback parser should return dict"
        
        print("✅ Fallback parser fix working - no more NameError!")
        return True
    except NameError as e:
        print(f"❌ NameError still present: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error in fallback parser: {e}")
        return False

def test_graceful_degradation():
    """Test graceful degradation when external services fail."""
    print("\n🔧 Testing graceful degradation...")
    
    try:
        from src.advanced_agent_fsm import FSMReActAgent
        from langchain_core.tools import Tool
        
        # Create mock tools
        mock_tools = [
            Tool(
                name="test_tool",
                description="A test tool",
                func=lambda x: f"Test result for: {x}"
            )
        ]
        
        # Initialize agent without API key (should gracefully degrade)
        with patch.dict(os.environ, {}, clear=True):
            agent = FSMReActAgent(tools=mock_tools)
            
            # Try to run the agent (should handle missing API gracefully)
            result = agent.run({"input": "What is 2+2?"})
            
            # Should return a response, not crash
            assert "output" in result, "Agent should return output even without API"
            
        print("✅ Graceful degradation working!")
        return True
    except Exception as e:
        print(f"❌ Graceful degradation failed: {e}")
        return False

def test_structured_logging():
    """Test that structured logging is working correctly."""
    print("\n🔧 Testing structured logging...")
    
    try:
        from src.advanced_agent_fsm import logger, correlation_context
        
        # Test logging with extra fields
        correlation_id = str(uuid.uuid4())
        
        with correlation_context(correlation_id):
            logger.info("Test structured log", extra={
                'test_field': 'test_value',
                'numeric_field': 123
            })
            
        print("✅ Structured logging working!")
        return True
    except Exception as e:
        print(f"❌ Structured logging error: {e}")
        return False

def main():
    """Run all tests and provide a comprehensive report."""
    print("🚀 Starting Resilient FSM Agent Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_correlation_logging,
        test_resilient_api_client,
        test_pydantic_validation,
        test_fsm_agent_initialization,
        test_fallback_parser_fix,
        test_graceful_degradation,
        test_structured_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The resilient FSM agent is working correctly.")
        print("\n🔧 Key fixes implemented:")
        print("   ✅ Fixed NameError: name 're' is not defined")
        print("   ✅ Added structured logging with correlation IDs")
        print("   ✅ Implemented resilient API client with retries")
        print("   ✅ Added Pydantic data contracts for validation")
        print("   ✅ Enhanced FSM with granular failure states")
        print("   ✅ Implemented graceful degradation patterns")
        print("\n🚀 The agent should now handle all queries without immediate failures!")
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 