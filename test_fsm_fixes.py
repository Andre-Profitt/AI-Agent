"""
Test script to verify FSM agent fixes for GAIA benchmark failures.
Tests the 5 key fixes implemented:
1. Tool-input contract enforcement
2. Parameter translation layer
3. Stub tools for missing references
4. Guard rails before synthesis/verification
5. Tool reliability initialization
"""

import logging
from src.advanced_agent_fsm import FSMReActAgent
from src.tools_enhanced import get_enhanced_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fsm_fixes():
    """Test the FSM agent with queries that were previously failing."""
    
    # Initialize tools and agent
    tools = get_enhanced_tools()
    fsm_agent = FSMReActAgent(tools=tools, model_preference="fast")
    
    # Test cases that were failing in the logs
    test_queries = [
        # Test 1: File reader with wrong parameter (was using 'query' instead of 'filename')
        {
            "query": "Read the attached Excel file",
            "expected_fix": "Parameter translation should convert 'query' to 'filename'"
        },
        # Test 2: Video analyzer that doesn't exist (should use stub)
        {
            "query": "Analyze the video at https://example.com/video.mp4",
            "expected_fix": "Stub tool should handle video_analyzer reference"
        },
        # Test 3: Simple question to test no tool outputs guard rail
        {
            "query": "What is 2 + 2?",
            "expected_fix": "Should complete without requiring tool outputs"
        },
        # Test 4: Chess position analysis (FEN string parameter)
        {
            "query": "Analyze this chess position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "expected_fix": "Should translate 'query' to 'fen_string' for chess tool"
        },
        # Test 5: Web search with generic query parameter
        {
            "query": "Who won the Nobel Prize in Physics in 2023?",
            "expected_fix": "Web researcher should work with 'query' parameter"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST {i}: {test_case['query'][:50]}...")
        logger.info(f"Expected fix: {test_case['expected_fix']}")
        logger.info(f"{'='*60}")
        
        try:
            result = fsm_agent.run({"input": test_case["query"]})
            
            # Check if we got a valid response
            if isinstance(result, dict) and "output" in result:
                answer = result["output"]
                success = not answer.startswith("Error") and not answer.startswith("I encountered an error")
                
                results.append({
                    "test": i,
                    "query": test_case["query"][:50] + "...",
                    "success": success,
                    "answer": answer[:100] + "..." if len(answer) > 100 else answer,
                    "steps": result.get("total_steps", 0)
                })
                
                logger.info(f"✅ Test {i} completed: {'SUCCESS' if success else 'FAILED'}")
                logger.info(f"Answer: {answer[:100]}...")
                logger.info(f"Total steps: {result.get('total_steps', 0)}")
            else:
                results.append({
                    "test": i,
                    "query": test_case["query"][:50] + "...",
                    "success": False,
                    "answer": "Invalid result format",
                    "steps": 0
                })
                logger.error(f"❌ Test {i} failed: Invalid result format")
                
        except Exception as e:
            results.append({
                "test": i,
                "query": test_case["query"][:50] + "...",
                "success": False,
                "answer": f"Exception: {str(e)[:100]}",
                "steps": 0
            })
            logger.error(f"❌ Test {i} failed with exception: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)
    
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Failed: {total_tests - successful_tests}")
    logger.info(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    logger.info("\nDetailed Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} Test {result['test']}: {result['query']}")
        logger.info(f"   Answer: {result['answer']}")
        logger.info(f"   Steps: {result['steps']}")
    
    return results

def test_specific_gaia_failures():
    """Test specific GAIA benchmark queries that were failing."""
    
    tools = get_enhanced_tools()
    fsm_agent = FSMReActAgent(tools=tools, model_preference="balanced")
    
    # Real GAIA queries from the logs
    gaia_queries = [
        "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
        "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species",
        ".rewsna eht sa 'tfel' drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI",
        "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("TESTING REAL GAIA QUERIES")
    logger.info("="*60)
    
    for i, query in enumerate(gaia_queries, 1):
        logger.info(f"\nGAIA Query {i}: {query[:60]}...")
        
        try:
            result = fsm_agent.run({"input": query})
            answer = result.get("output", "No output")
            logger.info(f"Result: {answer[:100]}...")
            logger.info(f"Steps taken: {result.get('total_steps', 0)}")
        except Exception as e:
            logger.error(f"Failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting FSM Agent Fix Tests...")
    logger.info("Testing fixes for:")
    logger.info("1. Tool-input contract enforcement")
    logger.info("2. Parameter translation layer")
    logger.info("3. Stub tools for missing references")
    logger.info("4. Guard rails before synthesis/verification")
    logger.info("5. Tool reliability initialization")
    
    # Run basic tests
    test_fsm_fixes()
    
    # Run GAIA-specific tests
    test_specific_gaia_failures()
    
    logger.info("\n✅ Test script completed!") 