"""
GAIA-Enhanced FSMReActAgent Usage Example
Demonstrates how to use the GAIA-enhanced agent with various scenarios
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.advanced_agent_fsm import FSMReActAgent
from tools.base_tool import BaseTool
from config.settings import DataQualityLevel, ReasoningType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockSearchTool(BaseTool):
    """Mock search tool for demonstration"""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the web for information"
    
    def __call__(self, query: str, **kwargs):
        """Mock search implementation"""
        return f"Search results for: {query} - Found relevant information about the topic."

class MockCalculatorTool(BaseTool):
    """Mock calculator tool for demonstration"""
    
    def __init__(self):
        self.name = "calculator"
        self.description = "Perform mathematical calculations"
    
    def __call__(self, expression: str, **kwargs):
        """Mock calculation implementation"""
        try:
            # Simple evaluation for demonstration
            result = eval(expression)
            return f"Calculation result: {expression} = {result}"
        except Exception as e:
            return f"Calculation error: {e}"

class MockAnalyzerTool(BaseTool):
    """Mock analyzer tool for demonstration"""
    
    def __init__(self):
        self.name = "data_analyzer"
        self.description = "Analyze data and extract insights"
    
    def __call__(self, data: str, **kwargs):
        """Mock analysis implementation"""
        return f"Analysis of '{data}': Key insights and patterns identified."

def create_gaia_agent():
    """Create a GAIA-enhanced FSMReActAgent with mock tools"""
    
    # Create mock tools
    tools = [
        MockSearchTool(),
        MockCalculatorTool(),
        MockAnalyzerTool()
    ]
    
    # Create GAIA-enhanced agent
    agent = FSMReActAgent(
        tools=tools,
        model_name="llama-3.3-70b-versatile",
        quality_level=DataQualityLevel.THOROUGH,
        reasoning_type=ReasoningType.LAYERED,
        model_preference="balanced",
        use_crew=False
    )
    
    return agent

async def example_1_basic_query(agent):
    """Example 1: Basic query processing"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Query Processing")
    print("="*60)
    
    query = "What is the capital of France?"
    
    print(f"Query: {query}")
    print("Processing...")
    
    try:
        result = await agent.run(query)
        
        print(f"\nResult: {result.get('final_answer', 'No answer')}")
        print(f"Success: {result.get('success', False)}")
        print(f"Confidence: {result.get('confidence', 0.0)}")
        
        if result.get('tool_calls'):
            print(f"\nTools used: {len(result['tool_calls'])}")
            for i, call in enumerate(result['tool_calls']):
                print(f"  {i+1}. {call.get('tool', 'unknown')}: {call.get('success', False)}")
        
    except Exception as e:
        print(f"Error: {e}")

async def example_2_complex_calculation(agent):
    """Example 2: Complex calculation with reasoning"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Complex Calculation")
    print("="*60)
    
    query = "Calculate the area of a circle with radius 5 meters"
    
    print(f"Query: {query}")
    print("Processing...")
    
    try:
        result = await agent.run(query)
        
        print(f"\nResult: {result.get('final_answer', 'No answer')}")
        print(f"Success: {result.get('success', False)}")
        print(f"Confidence: {result.get('confidence', 0.0)}")
        
        if result.get('reasoning_path'):
            print(f"\nReasoning steps: {len(result['reasoning_path'].steps)}")
            for i, step in enumerate(result['reasoning_path'].steps):
                print(f"  {i+1}. {step.step_type}: {step.content[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")

async def example_3_multi_agent_coordination(agent):
    """Example 3: Multi-agent coordination for complex task"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-Agent Coordination")
    print("="*60)
    
    query = "Research the latest developments in renewable energy and provide a comprehensive analysis"
    
    print(f"Query: {query}")
    print("Processing with multi-agent coordination...")
    
    try:
        # Use multi-agent system if available
        if agent.multi_agent:
            result = agent.multi_agent.process_gaia_query(query)
            print(f"\nMulti-agent result: {result}")
        else:
            # Fallback to regular processing
            result = await agent.run(query)
            print(f"\nResult: {result.get('final_answer', 'No answer')}")
        
        print(f"Success: {result.get('success', False)}")
        
    except Exception as e:
        print(f"Error: {e}")

async def example_4_memory_integration(agent):
    """Example 4: Memory system integration"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Memory System Integration")
    print("="*60)
    
    # First query to create memory
    query1 = "What is machine learning?"
    print(f"First query: {query1}")
    
    try:
        result1 = await agent.run(query1)
        print(f"First result: {result1.get('final_answer', 'No answer')[:100]}...")
        
        # Second query that should benefit from memory
        query2 = "Tell me more about the applications of machine learning"
        print(f"\nSecond query: {query2}")
        
        result2 = await agent.run(query2)
        print(f"Second result: {result2.get('final_answer', 'No answer')[:100]}...")
        
        # Check memory statistics
        if agent.memory_system:
            stats = agent.memory_system.get_memory_statistics()
            print(f"\nMemory statistics: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")

async def example_5_adaptive_tool_selection(agent):
    """Example 5: Adaptive tool selection"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Adaptive Tool Selection")
    print("="*60)
    
    queries = [
        "Search for information about climate change",
        "Calculate 15 * 23 + 7",
        "Analyze the trends in global temperature data"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        try:
            result = await agent.run(query)
            
            print(f"Result: {result.get('final_answer', 'No answer')[:100]}...")
            
            if result.get('tool_calls'):
                print("Tools selected:")
                for call in result['tool_calls']:
                    print(f"  - {call.get('tool', 'unknown')} (confidence: {call.get('confidence', 0.0):.2f})")
            
        except Exception as e:
            print(f"Error: {e}")

async def example_6_error_recovery(agent):
    """Example 6: Error recovery and resilience"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Error Recovery and Resilience")
    print("="*60)
    
    # Query that might cause issues
    query = "Perform a complex calculation that might fail: 1/0"
    
    print(f"Query: {query}")
    print("Testing error recovery...")
    
    try:
        result = await agent.run(query)
        
        print(f"Result: {result.get('final_answer', 'No answer')}")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('errors'):
            print(f"Errors encountered: {len(result['errors'])}")
            for error in result['errors']:
                print(f"  - {error}")
        
        # Check if recovery was attempted
        if result.get('recovery_attempts', 0) > 0:
            print(f"Recovery attempts: {result['recovery_attempts']}")
        
    except Exception as e:
        print(f"Error: {e}")

async def example_7_performance_monitoring(agent):
    """Example 7: Performance monitoring and statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Monitoring")
    print("="*60)
    
    # Run several queries to generate performance data
    queries = [
        "What is artificial intelligence?",
        "Calculate the Fibonacci sequence up to 10",
        "Analyze the benefits of renewable energy"
    ]
    
    for query in queries:
        try:
            await agent.run(query)
        except Exception as e:
            print(f"Error processing '{query}': {e}")
    
    # Get performance statistics
    if hasattr(agent, 'performance_monitor'):
        stats = agent.performance_monitor.get_health_status()
        print(f"\nPerformance statistics:")
        print(f"  Success rate: {stats.get('success_rate', 0.0):.2f}")
        print(f"  Average response time: {stats.get('avg_response_time', 0.0):.2f}s")
        print(f"  Total operations: {stats.get('total_operations', 0)}")
    
    # Get system statistics
    if hasattr(agent, 'adaptive_tools'):
        tool_stats = agent.adaptive_tools.get_system_statistics()
        print(f"\nTool system statistics:")
        print(f"  Total tools: {tool_stats.get('total_tools', 0)}")
        print(f"  Available tools: {tool_stats.get('available_tools', 0)}")
    
    if hasattr(agent, 'memory_system'):
        memory_stats = agent.memory_system.get_memory_statistics()
        print(f"\nMemory system statistics:")
        print(f"  Total memories: {memory_stats.get('total_memories', 0)}")
        print(f"  Episodic memories: {memory_stats.get('episodic_count', 0)}")
        print(f"  Semantic memories: {memory_stats.get('semantic_count', 0)}")

async def main():
    """Main function to run all examples"""
    print("GAIA-Enhanced FSMReActAgent Usage Examples")
    print("="*60)
    
    # Create agent
    print("Initializing GAIA-enhanced agent...")
    agent = create_gaia_agent()
    print("Agent initialized successfully!")
    
    # Run examples
    await example_1_basic_query(agent)
    await example_2_complex_calculation(agent)
    await example_3_multi_agent_coordination(agent)
    await example_4_memory_integration(agent)
    await example_5_adaptive_tool_selection(agent)
    await example_6_error_recovery(agent)
    await example_7_performance_monitoring(agent)
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 