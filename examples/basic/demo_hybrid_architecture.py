#!/usr/bin/env python3
"""
Demo script for the Enhanced Advanced Hybrid AI Agent Architecture
Showcasing FSM, ReAct, Chain of Thought, and Multi-Agent capabilities
"""

import asyncio
import sys
import os
import logging

logger = logging.getLogger(__name__)


# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from advanced_hybrid_architecture import AdvancedHybridAgent, AgentMode
from optimized_chain_of_thought import OptimizedChainOfThought, ReasoningType

async def main():
    """Main demo function"""
    logger.info("🚀 Enhanced Advanced Hybrid AI Agent Architecture Demo")
    print("=" * 70)
    logger.info("This demo showcases the integration of:")
    logger.info("• Finite State Machine (FSM) with ReAct")
    logger.info("• Optimized Chain of Thought (CoT) reasoning")
    logger.info("• Multi-agent collaboration")
    logger.info("• Adaptive mode selection")
    logger.info("• Performance optimization")
    logger.info("• Emergent behavior detection")
    print("=" * 70)
    
    # Initialize the enhanced hybrid agent
    logger.info("\n📋 Initializing Enhanced Hybrid Agent...")
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
    
    logger.info("✅ Agent initialized successfully!")
    
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
    
    logger.info("\n🧪 Testing {} different query types...", extra={"len_test_queries_": len(test_queries)})
    print("-" * 70)
    
    for i, test_case in enumerate(test_queries, 1):
        logger.info("\n📝 Test Case {}: {}", extra={"i": i, "test_case__description_": test_case['description']})
        logger.info("Query: {}", extra={"test_case__query_": test_case['query']})
        logger.info("Expected Mode: {}", extra={"test_case__expected_mode__name": test_case['expected_mode'].name})
        
        # Process the query
        start_time = asyncio.get_event_loop().time()
        result = await agent.process_query(test_case['query'])
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Display results
        logger.info("✅ Actual Mode: {}", extra={"result_get__mode____unknown__": result.get('mode', 'unknown')})
        logger.info("🎯 Confidence: {}", extra={"result_get__confidence___0_": result.get('confidence', 0)})
        logger.info("⏱️  Execution Time: {}s", extra={"execution_time": execution_time})
        
        # Show mode-specific details
        if result.get('mode') == 'chain_of_thought':
            reasoning_path = result.get('reasoning_path')
            if reasoning_path:
                logger.info("🧠 CoT Steps: {}", extra={"len_reasoning_path_steps_": len(reasoning_path.steps)})
                logger.info("📋 Template: {}", extra={"reasoning_path_template_used": reasoning_path.template_used})
                logger.info("🔍 Reasoning Types: {}", extra={"_step_reasoning_type_name_for_step_in_reasoning_path_steps_": [step.reasoning_type.name for step in reasoning_path.steps[})
                
                # Show key insights
                insights = result.get('insights', {})
                if insights:
                    logger.info("💡 Key Thoughts: {}", extra={"insights_get__key_thoughts______": insights.get('key_thoughts', [])[})
        
        elif result.get('mode') == 'fsm_react':
            steps = result.get('steps', [])
            tools_used = result.get('tools_used', [])
            logger.info("⚙️  FSM Steps: {}", extra={"len_steps_": len(steps)})
            logger.info("🔧 Tools Used: {}", extra={"tools_used": tools_used})
        
        elif result.get('mode') == 'hybrid':
            logger.info("🔄 Hybrid Synthesis: {}...", extra={"result_get__answer_______": result.get('answer', '')[})
            logger.info("📊 Secondary Answer: {}...", extra={"result_get__secondary_answer_______": result.get('secondary_answer', '')[})
        
        elif result.get('mode') == 'multi_agent':
            research = result.get('research', {})
            execution = result.get('execution', {})
            synthesis = result.get('synthesis', {})
            logger.info("🔬 Research Confidence: {}", extra={"research_get__confidence___0_": research.get('confidence', 0)})
            logger.info("⚡ Execution Confidence: {}", extra={"execution_get__confidence___0_": execution.get('confidence', 0)})
            logger.info("🎯 Synthesis Confidence: {}", extra={"synthesis_get__confidence___0_": synthesis.get('confidence', 0)})
        
        # Show emergent insights if any
        if 'emergent_insights' in result:
            insights = result['emergent_insights']
            logger.info("🌟 Emergent Insights: {}", extra={"insights": insights})
        
        print("-" * 50)
    
    # Performance Analysis
    logger.info("\n📊 Performance Analysis")
    print("=" * 50)
    
    report = agent.get_performance_report()
    
    logger.info("📈 Total Queries: {}", extra={"report__total_queries_": report['total_queries']})
    logger.info("🎯 Average Confidence: {}", extra={"report__average_confidence_": report['average_confidence']})
    logger.info("⏱️  Average Execution Time: {}s", extra={"report__average_execution_time_": report['average_execution_time']})
    
    logger.info("\n📋 Mode Usage:")
    for mode, count in report['mode_usage'].items():
        percentage = (count / report['total_queries']) * 100
        logger.info("  {}: {} queries ({}%)", extra={"mode": mode, "count": count, "percentage": percentage})
    
    # CoT Performance Details
    if 'cot_performance' in report:
        cot_perf = report['cot_performance']
        logger.info("\n🧠 Chain of Thought Performance:")
        logger.info("  Cache Hit Rate: {}", extra={"cot_perf_get__cache_hit_rate___0_": cot_perf.get('cache_hit_rate', 0)})
        logger.info("  Average Confidence: {}", extra={"cot_perf_get__average_confidence___0_": cot_perf.get('average_confidence', 0)})
        logger.info("  Templates Used: {cot_perf.get('templates_usage', {})}")
    
    # Reasoning History
    logger.info("\n📚 Recent Reasoning History:")
    print("-" * 50)
    history = agent.get_reasoning_history()
    for entry in history[-5:]:  # Show last 5 entries
        logger.info("  {}: {}... (conf: {})", extra={"entry__mode_": entry['mode'], "entry__query__": entry['query'][, "entry__confidence_": entry['confidence']})
    
    # Advanced Features Demo
    logger.info("\n🚀 Advanced Features Demo")
    print("=" * 50)
    
    # Test parallel reasoning
    logger.info("\n🔄 Testing Parallel Reasoning...")
    parallel_result = await agent.process_query(
        "Analyze the benefits and risks of renewable energy sources"
    )
    logger.info("Parallel Mode: {}", extra={"parallel_result_get__mode__": parallel_result.get('mode')})
    logger.info("Best Confidence: {}", extra={"parallel_result_get__confidence___0_": parallel_result.get('confidence', 0)})
    
    # Test caching
    logger.info("\n💾 Testing Cache Performance...")
    cache_query = "What is machine learning?"
    start_time = asyncio.get_event_loop().time()
    result1 = await agent.process_query(cache_query)
    time1 = asyncio.get_event_loop().time() - start_time
    
    start_time = asyncio.get_event_loop().time()
    result2 = await agent.process_query(cache_query)
    time2 = asyncio.get_event_loop().time() - start_time
    
    logger.info("First run: {}s", extra={"time1": time1})
    logger.info("Cached run: {}s", extra={"time2": time2})
    logger.info("Speedup: {}x", extra={"time1_time2": time1/time2})
    
    logger.info("\n🎉 Demo completed successfully!")
    logger.info("The enhanced hybrid architecture demonstrates:")
    logger.info("• Intelligent mode selection based on query complexity")
    logger.info("• Optimized Chain of Thought with multiple reasoning paths")
    logger.info("• Multi-agent collaboration for complex tasks")
    logger.info("• Performance optimization through caching")
    logger.info("• Emergent behavior detection and analysis")
    logger.info("• Comprehensive performance tracking and reporting")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.info("\n❌ Error during demo: {}", extra={"e": e})
        import traceback
        traceback.print_exc() 