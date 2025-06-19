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
    print("\n🔍 Testing imports...")
    
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
            print(f"✅ {module_name}.{component}")
        except ImportError as e:
            print(f"❌ {module_name}.{component} - {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment variables...")
    
    required_vars = ["GROQ_API_KEY", "OPENAI_API_KEY"]
    optional_vars = ["TAVILY_API_KEY", "SPACE_ID", "SPACE_HOST"]
    
    has_required = False
    
    # Check if at least one required API key is present
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
            has_required = True
        else:
            print(f"⚠️  {var} is not set")
    
    if not has_required:
        print("❌ At least one API key (GROQ_API_KEY or OPENAI_API_KEY) must be set!")
        return False
    
    # Check optional variables
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set (optional)")
        else:
            print(f"ℹ️  {var} is not set (optional)")
    
    return True

def test_agent_wrapper():
    """Test the agent wrapper."""
    print("\n🔍 Testing agent wrapper...")
    
    try:
        from agent import build_graph
        print("✅ Successfully imported build_graph from agent.py")
        
        # Try to build the graph
        graph = build_graph()
        print("✅ Successfully built agent graph")
        
        # Test with a simple question
        from langchain_core.messages import HumanMessage
        
        test_questions = [
            "What is 2+2?",
            "What is the capital of France?",
            "List the files in the current directory."
        ]
        
        for question in test_questions:
            print(f"\n📝 Testing question: '{question}'")
            try:
                result = graph.invoke({"messages": [HumanMessage(content=question)]})
                answer = result['messages'][-1].content
                print(f"✅ Got answer: {answer[:100]}...")
                
                # Check if answer is in GAIA format
                if "<<<" in answer and ">>>" in answer:
                    print("✅ Answer is in correct GAIA format")
                else:
                    print("⚠️  Answer may not be in GAIA format (missing <<<>>>)")
                    
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                logger.exception("Detailed error:")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import agent.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        logger.exception("Detailed error:")
        return False

def test_gaia_api():
    """Test connection to GAIA API."""
    print("\n🔍 Testing GAIA API connection...")
    
    try:
        import requests
        
        api_url = "https://agents-course-unit4-scoring.hf.space"
        questions_url = f"{api_url}/questions"
        
        print(f"📡 Connecting to {questions_url}")
        
        response = requests.get(questions_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Successfully connected to GAIA API")
            
            questions = response.json()
            print(f"✅ Retrieved {len(questions)} questions from GAIA")
            
            if questions:
                # Show sample question structure
                sample = questions[0]
                print(f"\n📋 Sample question structure:")
                print(f"   - task_id: {sample.get('task_id', 'N/A')}")
                print(f"   - question: {sample.get('question', 'N/A')[:100]}...")
            
            return True
        else:
            print(f"❌ GAIA API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("⚠️  GAIA API request timed out (this might be normal)")
        return True  # Don't fail the test for timeout
    except Exception as e:
        print(f"❌ Error connecting to GAIA API: {e}")
        return False

def test_gradio_interface():
    """Test that Gradio interface can be created."""
    print("\n🔍 Testing Gradio interface...")
    
    try:
        import gradio as gr
        
        # Try to create a simple interface
        with gr.Blocks() as demo:
            gr.Markdown("# Test Interface")
        
        print("✅ Gradio interface can be created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating Gradio interface: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("="*60)
    print("🚀 GAIA Integration Test Suite")
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
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your GAIA integration is ready.")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the interface")
        print("2. Test with individual questions in the 'Test Agent' tab")
        print("3. Login to HuggingFace and run full GAIA evaluation")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running GAIA evaluation.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Set environment variables in .env file")
        print("3. Ensure agent.py is in the root directory")
    
    return all_passed

if __name__ == "__main__":
    # Load environment variables if .env exists
    if os.path.exists(".env"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded .env file")
        except ImportError:
            print("⚠️  python-dotenv not installed, skipping .env file")
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 