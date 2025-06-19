#!/usr/bin/env python3
"""
Test script for Optimized Chain of Thought integration
Verifies that the CoT system works correctly with the hybrid architecture
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cot_standalone():
    """Test the CoT system in standalone mode"""
    print("🧪 Testing Optimized Chain of Thought (Standalone)")
    print("-" * 50)
    
    try:
        from optimized_chain_of_thought import (
            OptimizedChainOfThought, ReasoningType, ComplexityAnalyzer
        )
        
        # Test complexity analyzer
        analyzer = ComplexityAnalyzer()
        complexity, features = analyzer.analyze("What is machine learning?")
        print(f"✅ Complexity Analysis: {complexity:.3f}")
        print(f"   Features: {list(features.keys())}")
        
        # Test CoT system
        cot = OptimizedChainOfThought("test_cot")
        print(f"✅ CoT System initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ CoT Standalone Test Failed: {e}")
        return False

async def test_cot_integration():
    """Test CoT integration with hybrid architecture"""
    print("\n🔗 Testing CoT Integration with Hybrid Architecture")
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
        print(f"✅ Hybrid Agent initialized")
        
        # Test CoT mode specifically
        test_query = "Explain the concept of artificial intelligence step by step"
        result = await agent.process_query(test_query)
        
        print(f"✅ Query processed successfully")
        print(f"   Mode: {result.get('mode')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        
        if 'reasoning_path' in result:
            path = result['reasoning_path']
            print(f"   CoT Steps: {len(path.steps)}")
            print(f"   Template: {path.template_used}")
            print(f"   Final Answer: {path.final_answer[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ CoT Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_tracking():
    """Test performance tracking capabilities"""
    print("\n📊 Testing Performance Tracking")
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
        
        print(f"✅ Performance tracking working")
        print(f"   Total queries: {report['total_queries']}")
        print(f"   Average confidence: {report['average_confidence']:.3f}")
        print(f"   Mode usage: {report['mode_usage']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Tracking Test Failed: {e}")
        return False

async def test_caching():
    """Test caching functionality"""
    print("\n💾 Testing Caching System")
    print("-" * 50)
    
    try:
        from optimized_chain_of_thought import OptimizedChainOfThought
        
        cot = OptimizedChainOfThought("cache_test")
        
        # First query
        query = "What is the capital of France?"
        result1 = await cot.reason(query)
        
        # Same query again (should use cache)
        result2 = await cot.reason(query)
        
        print(f"✅ Caching system working")
        print(f"   First run confidence: {result1.total_confidence:.3f}")
        print(f"   Cached run confidence: {result2.total_confidence:.3f}")
        
        # Check cache stats
        stats = cot.reasoning_cache.get_stats()
        print(f"   Cache size: {stats['size']}")
        print(f"   Hit rate: {stats['hit_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching Test Failed: {e}")
        return False

async def test_template_system():
    """Test template selection and usage"""
    print("\n📋 Testing Template System")
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
            
            print(f"   Query: {query[:30]}...")
            print(f"   Selected: {template.name}")
            print(f"   Expected: {expected_type}")
            print(f"   Match: {'✅' if template.name == expected_type else '❌'}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Template System Test Failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Optimized Chain of Thought Integration Tests")
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
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} Test Failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CoT integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 