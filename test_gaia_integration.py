#!/usr/bin/env python3
"""
Quick test script to verify GAIA integration is working correctly.
Run this before attempting full GAIA evaluation.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work."""
    logger.info("\n🔍 Testing imports...")
    
    required_imports = [
        ("langchain_core.messages", "HumanMessage"),
        ("langgraph.graph", "Graph"),
        ("gradio", "Blocks"),
        ("pandas", "DataFrame"),
        ("requests", "get"),
    ]
    
    failed_imports = []
    
    for module_name, component in required_imports:
        try:
            module = __import__(module_name, fromlist=[component])
            getattr(module, component)
            logger.info("✅ {}.{}", extra={"module_name": module_name, "component": component})
        except ImportError as e:
            logger.info("❌ {}.{} - {}", extra={"module_name": module_name, "component": component, "e": e})
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def test_environment():
    """Test environment variables."""
    logger.info("\n🔍 Testing environment variables...")
    
    required_vars = ["GROQ_API_KEY", "OPENAI_API_KEY"]
    optional_vars = ["TAVILY_API_KEY", "SPACE_ID", "SPACE_HOST"]
    
    has_required = False
    
    # Check if at least one required API key is present
    for var in required_vars:
        if os.getenv(var):
            logger.info("✅ {} is set", extra={"var": var})
            has_required = True
        else:
            logger.info("⚠️  {} is not set", extra={"var": var})
    
    if not has_required:
        logger.info("❌ At least one API key (GROQ_API_KEY or OPENAI_API_KEY) must be set!")
        return False
    
    # Check optional variables
    for var in optional_vars:
        if os.getenv(var):
            logger.info("✅ {} is set (optional)", extra={"var": var})
        else:
            logger.info("ℹ️  {} is not set (optional)", extra={"var": var})
    
    return True

def test_agent_wrapper():
    """Test the agent wrapper."""
    logger.info("\n🔍 Testing agent wrapper...")
    
    try:
        from agent import build_graph
        logger.info("✅ Successfully imported build_graph from agent.py")
        
        # Try to build the graph
        graph = build_graph()
        logger.info("✅ Successfully built agent graph")
        
        # Test with a simple question
        from langchain_core.messages import HumanMessage
        
        test_questions = [
            "What is 2+2?",
            "What is the capital of France?",
            "List the files in the current directory."
        ]
        
        for question in test_questions:
            logger.info("\n📝 Testing question: '{}'", extra={"question": question})
            try:
                result = graph.invoke({"messages": [HumanMessage(content=question)]})
                answer = result['messages'][-1].content
                logger.info("✅ Got answer: {}...", extra={"answer_": answer[})
                
                # Check if answer is in GAIA format
                if "<<<" in answer and ">>>" in answer:
                    logger.info("✅ Answer is in correct GAIA format")
                else:
                    logger.info("⚠️  Answer may not be in GAIA format (missing <<<>>>)")
                    
            except Exception as e:
                logger.info("❌ Error processing question: {}", extra={"e": e})
                logger.exception("Detailed error:")
        
        return True
        
    except ImportError as e:
        logger.info("❌ Could not import agent.py: {}", extra={"e": e})
        return False
    except Exception as e:
        logger.info("❌ Error testing agent: {}", extra={"e": e})
        logger.exception("Detailed error:")
        return False

def test_gaia_api():
    """Test connection to GAIA API."""
    logger.info("\n🔍 Testing GAIA API connection...")
    
    try:
        import requests
        
        api_url = "https://agents-course-unit4-scoring.hf.space"
        questions_url = f"{api_url}/questions"
        
        logger.info("📡 Connecting to {}", extra={"questions_url": questions_url})
        
        response = requests.get(questions_url, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Successfully connected to GAIA API")
            
            questions = response.json()
            logger.info("✅ Retrieved {} questions from GAIA", extra={"len_questions_": len(questions)})
            
            if questions:
                # Show sample question structure
                sample = questions[0]
                logger.info("\n📋 Sample question structure:")
                logger.info("   - task_id: {}", extra={"sample_get__task_id____N_A__": sample.get('task_id', 'N/A')})
                logger.info("   - question: {}...", extra={"sample_get__question____N_A___": sample.get('question', 'N/A')[})
            
            return True
        else:
            logger.info("❌ GAIA API returned status code: {}", extra={"response_status_code": response.status_code})
            return False
            
    except requests.exceptions.Timeout:
        logger.info("⚠️  GAIA API request timed out (this might be normal)")
        return True  # Don't fail the test for timeout
    except Exception as e:
        logger.info("❌ Error connecting to GAIA API: {}", extra={"e": e})
        return False

def test_gradio_interface():
    """Test that Gradio interface can be created."""
    logger.info("\n🔍 Testing Gradio interface...")
    
    try:
        import gradio as gr
        
        # Try to create a simple interface
        with gr.Blocks() as demo:
            gr.Markdown("# Test Interface")
        
        logger.info("✅ Gradio interface can be created")
        return True
        
    except Exception as e:
        logger.info("❌ Error creating Gradio interface: {}", extra={"e": e})
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("="*60)
    logger.info("🚀 GAIA Integration Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Agent Wrapper", test_agent_wrapper),
        ("GAIA API", test_gaia_api),
        ("Gradio Interface", test_gradio_interface),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.info("\n❌ Unexpected error in {}: {}", extra={"test_name": test_name, "e": e})
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    logger.info("📊 Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info("{} {}", extra={"test_name": test_name, "status": status})
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Your GAIA integration is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run 'python app.py' to start the interface")
        logger.info("2. Test with individual questions in the 'Test Agent' tab")
        logger.info("3. Login to HuggingFace and run full GAIA evaluation")
    else:
        logger.info("\n⚠️  Some tests failed. Please fix the issues above before running GAIA evaluation.")
        logger.info("\nCommon fixes:")
        logger.info("1. Install missing dependencies: pip install -r requirements.txt")
        logger.info("2. Set environment variables in .env file")
        logger.info("3. Ensure agent.py is in the root directory")
    
    return all_passed

if __name__ == "__main__":
    # Load environment variables if .env exists
    if os.path.exists(".env"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("✅ Loaded .env file")
        except ImportError:
            logger.info("⚠️  python-dotenv not installed, skipping .env file")
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 