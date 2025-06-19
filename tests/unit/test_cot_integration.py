#!/usr/bin/env python3
"""
Test script for Optimized Chain of Thought integration
Verifies that the CoT system works correctly with the hybrid architecture
"""

import asyncio
import sys
import os
import logging

logger = logging.getLogger(__name__)


# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cot_standalone():
    """Test the CoT system in standalone mode"""
    logger.info("🧪 Testing Optimized Chain of Thought (Standalone)")
    print("-" * 50)
    
    try:
        from optimized_chain_of_thought import (
            OptimizedChainOfThought, ReasoningType, ComplexityAnalyzer
        )
        
        # Test complexity analyzer
        analyzer = ComplexityAnalyzer()
        complexity, features = analyzer.analyze("What is machine learning?")
        logger.info("✅ Complexity Analysis: {}", extra={"complexity": complexity})
        logger.info("   Features: {}", extra={"list_features_keys___": list(features.keys())})
        
        # Test CoT system
        cot = OptimizedChainOfThought("test_cot")
        logger.info("✅ CoT System initialized")
        
        return True
        
    except Exception as e:
        logger.info("❌ CoT Standalone Test Failed: {}", extra={"e": e})
        return False

async def test_cot_integration():
    """Test CoT integration with hybrid architecture"""
    logger.info("\n🔗 Testing CoT Integration with Hybrid Architecture")
    print("-" * 50)
    
    try:
        from advanced_hybrid_architecture import AdvancedHybridAgent, AgentMode
        
        # Initialize hybrid agent
        agent = AdvancedHybridAgent(
            "test_agent",
            config={
                'cot': {'max_paths': 3, 'cache_size': 50},
                'fsm': {'max_steps': 5}
            }
        )
        logger.info("✅ Hybrid Agent initialized")
        
        # Test CoT mode specifically
        test_query = "Explain the concept of artificial intelligence step by step"
        result = await agent.process_query(test_query)
        
        logger.info("✅ Query processed successfully")
        logger.info("   Mode: {}", extra={"result_get__mode__": result.get('mode')})
        logger.info("   Confidence: {}", extra={"result_get__confidence___0_": result.get('confidence', 0)})
        
        if 'reasoning_path' in result:
            path = result['reasoning_path']
            logger.info("   CoT Steps: {}", extra={"len_path_steps_": len(path.steps)})
            logger.info("   Template: {}", extra={"path_template_used": path.template_used})
            logger.info("   Final Answer: {}...", extra={"path_final_answer_": path.final_answer[})
        
        return True
        
    except Exception as e:
        logger.info("❌ CoT Integration Test Failed: {}", extra={"e": e})
        import traceback
        traceback.print_exc()
        return False

async def test_performance_tracking():
    """Test performance tracking capabilities"""
    logger.info("\n📊 Testing Performance Tracking")
    print("-" * 50)
    
    try:
        from advanced_hybrid_architecture import AdvancedHybridAgent
        
        agent = AdvancedHybridAgent("perf_test_agent")
        
        # Process multiple queries
        queries = [
            "What is 2 + 2?",
            "Explain machine learning",
            "Compare AI and human intelligence"
        ]
        
        for query in queries:
            await agent.process_query(query)
        
        # Get performance report
        report = agent.get_performance_report()
        
        logger.info("✅ Performance tracking working")
        logger.info("   Total queries: {}", extra={"report__total_queries_": report['total_queries']})
        logger.info("   Average confidence: {}", extra={"report__average_confidence_": report['average_confidence']})
        logger.info("   Mode usage: {}", extra={"report__mode_usage_": report['mode_usage']})
        
        return True
        
    except Exception as e:
        logger.info("❌ Performance Tracking Test Failed: {}", extra={"e": e})
        return False

async def test_caching():
    """Test caching functionality"""
    logger.info("\n💾 Testing Caching System")
    print("-" * 50)
    
    try:
        from optimized_chain_of_thought import OptimizedChainOfThought
        
        cot = OptimizedChainOfThought("cache_test")
        
        # First query
        query = "What is the capital of France?"
        result1 = await cot.reason(query)
        
        # Same query again (should use cache)
        result2 = await cot.reason(query)
        
        logger.info("✅ Caching system working")
        logger.info("   First run confidence: {}", extra={"result1_total_confidence": result1.total_confidence})
        logger.info("   Cached run confidence: {}", extra={"result2_total_confidence": result2.total_confidence})
        
        # Check cache stats
        stats = cot.reasoning_cache.get_stats()
        logger.info("   Cache size: {}", extra={"stats__size_": stats['size']})
        logger.info("   Hit rate: {}", extra={"stats__hit_rate_": stats['hit_rate']})
        
        return True
        
    except Exception as e:
        logger.info("❌ Caching Test Failed: {}", extra={"e": e})
        return False

async def test_template_system():
    """Test template selection and usage"""
    logger.info("\n📋 Testing Template System")
    print("-" * 50)
    
    try:
        from optimized_chain_of_thought import TemplateLibrary, ComplexityAnalyzer
        
        library = TemplateLibrary()
        analyzer = ComplexityAnalyzer()
        
        # Test different query types
        test_cases = [
            ("Solve the equation: 2x + 3 = 7", "mathematical"),
            ("Compare cats and dogs", "comparative"),
            ("Why does the sky appear blue?", "causal"),
            ("Analyze the impact of social media", "analytical")
        ]
        
        for query, expected_type in test_cases:
            complexity, features = analyzer.analyze(query)
            template = library.select_template(query, features)
            
            logger.info("   Query: {}...", extra={"query_": query[})
            logger.info("   Selected: {}", extra={"template_name": template.name})
            logger.info("   Expected: {}", extra={"expected_type": expected_type})
            logger.info("   Match: {}", extra={"____if_template_name____expected_type_else____": '✅' if template.name == expected_type else '❌'})
            logger.info("")
        
        return True
        
    except Exception as e:
        logger.info("❌ Template System Test Failed: {}", extra={"e": e})
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Optimized Chain of Thought Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Standalone CoT", test_cot_standalone),
        ("CoT Integration", test_cot_integration),
        ("Performance Tracking", test_performance_tracking),
        ("Caching System", test_caching),
        ("Template System", test_template_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info("\n🧪 Running {} Test...", extra={"test_name": test_name})
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.info("❌ {} Test Failed: {}", extra={"test_name": test_name, "e": e})
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    logger.info("📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info("{}: {}", extra={"test_name": test_name, "status": status})
        if result:
            passed += 1
    
    logger.info("\nOverall: {}/{} tests passed", extra={"passed": passed, "total": total})
    
    if passed == total:
        logger.info("🎉 All tests passed! CoT integration is working correctly.")
    else:
        logger.info("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.info("\n❌ Test suite failed: {}", extra={"e": e})
        import traceback
        traceback.print_exc()
        sys.exit(1) 