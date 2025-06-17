#!/usr/bin/env python3
"""
Test script for FSM Agent implementation
Tests key improvements: no recursion errors, tool compatibility, state passing
"""

import logging
from src.advanced_agent_fsm import FSMReActAgent
from src.tools_enhanced import get_enhanced_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fsm_agent():
    """Test the FSM agent with various GAIA-style questions."""
    
    # Initialize agent with enhanced tools
    logger.info("Initializing FSM Agent...")
    tools = get_enhanced_tools()
    agent = FSMReActAgent(tools=tools, model_preference="balanced")
    
    # Test cases covering different scenarios
    test_questions = [
        # Simple factual question
        {
            "query": "What is the capital of France?",
            "expected_type": "string",
            "description": "Simple factual query"
        },
        # Counting question (tests structured output)
        {
            "query": "How many days are there in a leap year?",
            "expected_type": "integer",
            "description": "Counting question with integer answer"
        },
        # Logic puzzle (tests abstract reasoning tool)
        {
            "query": "If you reverse the word 'hello', what do you get?",
            "expected_type": "string",
            "description": "Text reversal logic puzzle"
        },
        # Multi-step question (tests state passing)
        {
            "query": "First find the largest planet in our solar system, then tell me how many moons it has.",
            "expected_type": "integer",
            "description": "Multi-step reasoning with state passing"
        },
        # Mock video analysis (tests GAIA video tool)
        {
            "query": "In the video at googleusercontent.com/costa-rica/birds, what is the highest bird count mentioned?",
            "expected_type": "integer",
            "description": "Video analysis with mock GAIA tool"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i}: {test_case['description']}")
        logger.info(f"Query: {test_case['query']}")
        
        try:
            # Run the agent
            result = agent.run({"input": test_case["query"]})
            
            # Extract answer
            answer = result.get("output", "No answer")
            confidence = result.get("confidence", 0.0)
            steps = result.get("total_steps", 0)
            
            logger.info(f"Answer: {answer}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Steps taken: {steps}")
            
            # Check if answer type matches expected
            if test_case["expected_type"] == "integer":
                try:
                    int(answer)
                    type_match = True
                except:
                    type_match = False
            else:
                type_match = isinstance(answer, str) and answer != "Error"
            
            results.append({
                "test": i,
                "description": test_case["description"],
                "success": "Error" not in answer and type_match,
                "answer": answer,
                "confidence": confidence,
                "steps": steps
            })
            
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append({
                "test": i,
                "description": test_case["description"],
                "success": False,
                "answer": f"Exception: {str(e)}",
                "confidence": 0.0,
                "steps": 0
            })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Failed: {total_tests - successful_tests}")
    logger.info(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    logger.info(f"\nDetailed Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} Test {result['test']}: {result['description']}")
        logger.info(f"   Answer: {result['answer']}")
        logger.info(f"   Confidence: {result['confidence']:.2f}, Steps: {result['steps']}")
    
    return results

if __name__ == "__main__":
    logger.info("Starting FSM Agent Tests...")
    results = test_fsm_agent()
    logger.info("\nTests completed!") 