#!/usr/bin/env python3
"""
Demo script for the Enhanced Advanced Hybrid AI Agent Architecture
Showcasing FSM, ReAct, Chain of Thought, and Multi-Agent capabilities
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from advanced_hybrid_architecture import AdvancedHybridAgent, AgentMode
from optimized_chain_of_thought import OptimizedChainOfThought, ReasoningType

async def main():
    """Main demo function"""
    print("🚀 Enhanced Advanced Hybrid AI Agent Architecture Demo")
    print("=" * 70)
    print("This demo showcases the integration of:")
    print("• Finite State Machine (FSM) with ReAct")
    print("• Optimized Chain of Thought (CoT) reasoning")
    print("• Multi-agent collaboration")
    print("• Adaptive mode selection")
    print("• Performance optimization")
    print("• Emergent behavior detection")
    print("=" * 70)
    
    # Initialize the enhanced hybrid agent
    print("\n📋 Initializing Enhanced Hybrid Agent...")
    agent = AdvancedHybridAgent(
        "demo_agent",
        config={
            'fsm': {
                'max_steps': 15,
                'reflection_enabled': True,
                'parallel_processing': True
            },
            'cot': {
                'max_paths': 5,
                'cache_size': 200,
                'cache_ttl': 24,
                'metacognitive_reflection': True
            },
            'multi_agent': {
                'researcher_enabled': True,
                'executor_enabled': True,
                'synthesizer_enabled': True
            }
        }
    )
    
    print("✅ Agent initialized successfully!")
    
    # Test different types of queries
    test_queries = [
        {
            'query': "What is the current weather in New York?",
            'expected_mode': AgentMode.FSM_REACT,
            'description': "Simple factual query - should use FSM for efficiency"
        },
        {
            'query': "Explain the concept of machine learning and its applications in healthcare",
            'expected_mode': AgentMode.CHAIN_OF_THOUGHT,
            'description': "Complex analytical query - should use CoT for deep reasoning"
        },
        {
            'query': "Compare and contrast the economic systems of capitalism and socialism, then analyze their impact on innovation",
            'expected_mode': AgentMode.HYBRID_ADAPTIVE,
            'description': "Multi-faceted complex query - should use hybrid approach"
        },
        {
            'query': "Solve the equation: 3x^2 + 7x - 2 = 0 and explain each step",
            'expected_mode': AgentMode.CHAIN_OF_THOUGHT,
            'description': "Mathematical problem - should use CoT with mathematical template"
        },
        {
            'query': "Analyze the potential long-term impacts of artificial intelligence on employment, education, and society",
            'expected_mode': AgentMode.MULTI_AGENT,
            'description': "Complex multi-domain analysis - should use multi-agent collaboration"
        }
    ]
    
    print(f"\n🧪 Testing {len(test_queries)} different query types...")
    print("-" * 70)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Mode: {test_case['expected_mode'].name}")
        
        # Process the query
        start_time = asyncio.get_event_loop().time()
        result = await agent.process_query(test_case['query'])
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Display results
        print(f"✅ Actual Mode: {result.get('mode', 'unknown')}")
        print(f"🎯 Confidence: {result.get('confidence', 0):.3f}")
        print(f"⏱️  Execution Time: {execution_time:.3f}s")
        
        # Show mode-specific details
        if result.get('mode') == 'chain_of_thought':
            reasoning_path = result.get('reasoning_path')
            if reasoning_path:
                print(f"🧠 CoT Steps: {len(reasoning_path.steps)}")
                print(f"📋 Template: {reasoning_path.template_used}")
                print(f"🔍 Reasoning Types: {[step.reasoning_type.name for step in reasoning_path.steps[:3]]}")
                
                # Show key insights
                insights = result.get('insights', {})
                if insights:
                    print(f"💡 Key Thoughts: {insights.get('key_thoughts', [])[:2]}")
        
        elif result.get('mode') == 'fsm_react':
            steps = result.get('steps', [])
            tools_used = result.get('tools_used', [])
            print(f"⚙️  FSM Steps: {len(steps)}")
            print(f"🔧 Tools Used: {tools_used}")
        
        elif result.get('mode') == 'hybrid':
            print(f"🔄 Hybrid Synthesis: {result.get('answer', '')[:100]}...")
            print(f"📊 Secondary Answer: {result.get('secondary_answer', '')[:50]}...")
        
        elif result.get('mode') == 'multi_agent':
            research = result.get('research', {})
            execution = result.get('execution', {})
            synthesis = result.get('synthesis', {})
            print(f"🔬 Research Confidence: {research.get('confidence', 0):.3f}")
            print(f"⚡ Execution Confidence: {execution.get('confidence', 0):.3f}")
            print(f"🎯 Synthesis Confidence: {synthesis.get('confidence', 0):.3f}")
        
        # Show emergent insights if any
        if 'emergent_insights' in result:
            insights = result['emergent_insights']
            print(f"🌟 Emergent Insights: {insights}")
        
        print("-" * 50)
    
    # Performance Analysis
    print(f"\n📊 Performance Analysis")
    print("=" * 50)
    
    report = agent.get_performance_report()
    
    print(f"📈 Total Queries: {report['total_queries']}")
    print(f"🎯 Average Confidence: {report['average_confidence']:.3f}")
    print(f"⏱️  Average Execution Time: {report['average_execution_time']:.3f}s")
    
    print(f"\n📋 Mode Usage:")
    for mode, count in report['mode_usage'].items():
        percentage = (count / report['total_queries']) * 100
        print(f"  {mode}: {count} queries ({percentage:.1f}%)")
    
    # CoT Performance Details
    if 'cot_performance' in report:
        cot_perf = report['cot_performance']
        print(f"\n🧠 Chain of Thought Performance:")
        print(f"  Cache Hit Rate: {cot_perf.get('cache_hit_rate', 0):.3f}")
        print(f"  Average Confidence: {cot_perf.get('average_confidence', 0):.3f}")
        print(f"  Templates Used: {cot_perf.get('templates_usage', {})}")
    
    # Reasoning History
    print(f"\n📚 Recent Reasoning History:")
    print("-" * 50)
    history = agent.get_reasoning_history()
    for entry in history[-5:]:  # Show last 5 entries
        print(f"  {entry['mode']}: {entry['query'][:40]}... (conf: {entry['confidence']:.2f})")
    
    # Advanced Features Demo
    print(f"\n🚀 Advanced Features Demo")
    print("=" * 50)
    
    # Test parallel reasoning
    print("\n🔄 Testing Parallel Reasoning...")
    parallel_result = await agent.process_query(
        "Analyze the benefits and risks of renewable energy sources"
    )
    print(f"Parallel Mode: {parallel_result.get('mode')}")
    print(f"Best Confidence: {parallel_result.get('confidence', 0):.3f}")
    
    # Test caching
    print("\n💾 Testing Cache Performance...")
    cache_query = "What is machine learning?"
    start_time = asyncio.get_event_loop().time()
    result1 = await agent.process_query(cache_query)
    time1 = asyncio.get_event_loop().time() - start_time
    
    start_time = asyncio.get_event_loop().time()
    result2 = await agent.process_query(cache_query)
    time2 = asyncio.get_event_loop().time() - start_time
    
    print(f"First run: {time1:.3f}s")
    print(f"Cached run: {time2:.3f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    
    print(f"\n🎉 Demo completed successfully!")
    print("The enhanced hybrid architecture demonstrates:")
    print("• Intelligent mode selection based on query complexity")
    print("• Optimized Chain of Thought with multiple reasoning paths")
    print("• Multi-agent collaboration for complex tasks")
    print("• Performance optimization through caching")
    print("• Emergent behavior detection and analysis")
    print("• Comprehensive performance tracking and reporting")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc() 