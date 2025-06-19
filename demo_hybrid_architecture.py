#!/usr/bin/env python3
"""
Demonstration of Advanced AI Agent Architecture
Shows how to use the hybrid FSM, ReAct, and Chain of Thought system
"""

import asyncio
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.advanced_hybrid_architecture import (
    AdvancedHybridSystem, HybridAgent, ProbabilisticFSM, 
    AgentState, Transition, Tool, ReasoningStep
)
from src.tools.semantic_search_tool import SemanticSearchTool
from src.tools.python_interpreter import PythonInterpreter
from src.tools.weather import WeatherTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoTools:
    """Demo tools for testing the architecture"""
    
    @staticmethod
    def calculator_tool(expression: str) -> dict:
        """Simple calculator tool"""
        try:
            result = eval(expression)  # Note: eval is unsafe in production
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}
    
    @staticmethod
    def text_analyzer_tool(text: str) -> dict:
        """Simple text analysis tool"""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    @staticmethod
    def mock_search_tool(query: str) -> dict:
        """Mock search tool"""
        return {
            "results": [
                f"Result 1 for: {query}",
                f"Result 2 for: {query}",
                f"Result 3 for: {query}"
            ],
            "query": query
        }

class CustomTool:
    """Custom tool wrapper for demo tools"""
    
    def __init__(self, name: str, function, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    def run(self, **kwargs):
        return self.function(**kwargs)

async def demo_basic_hybrid_agent():
    """Demonstrate basic hybrid agent functionality"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Hybrid Agent")
    print("="*60)
    
    # Create demo tools
    tools = [
        CustomTool("calculator", DemoTools.calculator_tool, "Perform mathematical calculations"),
        CustomTool("text_analyzer", DemoTools.text_analyzer_tool, "Analyze text statistics"),
        CustomTool("search", DemoTools.mock_search_tool, "Search for information")
    ]
    
    # Create hybrid agent
    agent = HybridAgent("demo_agent", tools)
    
    # Test different task types
    tasks = [
        {
            "type": "reasoning",
            "query": "What is the complexity of analyzing hybrid AI architectures?"
        },
        {
            "type": "tool_use",
            "query": "Calculate 15 * 23 and analyze the text 'Hello World'",
            "context": {"require_calculation": True, "require_analysis": True}
        },
        {
            "type": "state_based",
            "states": [
                {"name": "start", "data": {"step": 1}},
                {"name": "process", "data": {"step": 2}},
                {"name": "complete", "data": {"step": 3}}
            ]
        }
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}: {task['type']} ---")
        print(f"Query: {task.get('query', 'State-based task')}")
        
        result = await agent.execute_task(task)
        print(f"Mode used: {agent.current_mode}")
        print(f"Result: {result}")
        
        # Show performance metrics
        print(f"Performance by mode: {agent.mode_performance}")

async def demo_multi_agent_system():
    """Demonstrate multi-agent collaboration"""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Agent System")
    print("="*60)
    
    # Create the hybrid system
    system = AdvancedHybridSystem()
    
    # Create different types of tools
    general_tools = [
        CustomTool("search", DemoTools.mock_search_tool, "Search for information"),
        CustomTool("calculator", DemoTools.calculator_tool, "Perform calculations")
    ]
    
    analysis_tools = [
        CustomTool("text_analyzer", DemoTools.text_analyzer_tool, "Analyze text"),
        CustomTool("calculator", DemoTools.calculator_tool, "Perform calculations")
    ]
    
    # Create specialized agents
    general_agent = system.create_agent("general_agent", general_tools, ["general", "search", "calculation"])
    analysis_agent = system.create_agent("analysis_agent", analysis_tools, ["reasoning", "analysis"])
    
    # Complex collaborative task
    complex_task = {
        "type": "complex",
        "query": "Analyze the benefits of hybrid AI architectures and calculate some statistics",
        "subtasks": [
            {
                "type": "reasoning",
                "query": "What are the key benefits of hybrid AI architectures?",
                "required_capability": "reasoning"
            },
            {
                "type": "tool_use",
                "query": "Calculate the average word length in 'hybrid artificial intelligence'",
                "required_capability": "calculation"
            },
            {
                "type": "analysis",
                "query": "Analyze the complexity of multi-agent systems",
                "required_capability": "analysis"
            }
        ]
    }
    
    print("Executing complex collaborative task...")
    result = await system.execute_complex_task(complex_task)
    print(f"Collaborative result: {result}")
    
    # Show system health
    health = system.get_system_health()
    print(f"\nSystem Health Summary:")
    print(f"- Resource usage: {health['resource_usage']}")
    print(f"- Behavior patterns observed: {health['behavior_patterns']}")
    print(f"- Cache size: {health['cache_stats']['size']}/{health['cache_stats']['max_size']}")

async def demo_fsm_learning():
    """Demonstrate FSM learning capabilities"""
    print("\n" + "="*60)
    print("DEMO 3: FSM Learning")
    print("="*60)
    
    # Create a probabilistic FSM
    fsm = ProbabilisticFSM("learning_fsm")
    
    # Define states
    states = [
        AgentState("idle", {"energy": 100}),
        AgentState("working", {"task": None}),
        AgentState("resting", {"recovery": 0}),
        AgentState("completed", {"result": None})
    ]
    
    for state in states:
        fsm.add_state(state)
    
    # Define transitions with conditions
    def has_energy(state):
        return state.data.get("energy", 0) > 30
    
    def is_tired(state):
        return state.data.get("energy", 0) < 50
    
    def work_complete(state):
        return state.data.get("task") == "done"
    
    def rested(state):
        return state.data.get("recovery", 0) > 5
    
    transitions = [
        Transition("idle", "working", has_energy, probability=0.8),
        Transition("idle", "resting", is_tired, probability=0.2),
        Transition("working", "completed", work_complete, probability=0.9),
        Transition("working", "resting", is_tired, probability=0.1),
        Transition("resting", "idle", rested, probability=0.7),
        Transition("completed", "idle", lambda s: True, probability=1.0)
    ]
    
    for transition in transitions:
        fsm.add_transition(transition)
    
    # Set initial state
    fsm.set_initial_state("idle")
    
    print("Running FSM with learning...")
    print("Initial state:", fsm.current_state.name)
    
    # Run FSM for several steps
    for step in range(10):
        if fsm.step():
            print(f"Step {step + 1}: {fsm.current_state.name} (energy: {fsm.current_state.data.get('energy', 0)})")
            
            # Simulate state changes
            if fsm.current_state.name == "working":
                fsm.current_state.data["energy"] = max(0, fsm.current_state.data.get("energy", 100) - 20)
                fsm.current_state.data["task"] = "done" if step > 5 else "in_progress"
            elif fsm.current_state.name == "resting":
                fsm.current_state.data["energy"] = min(100, fsm.current_state.data.get("energy", 0) + 30)
                fsm.current_state.data["recovery"] = fsm.current_state.data.get("recovery", 0) + 1
            elif fsm.current_state.name == "idle":
                fsm.current_state.data["energy"] = min(100, fsm.current_state.data.get("energy", 0) + 10)
        else:
            print(f"Step {step + 1}: No valid transitions")
            break
    
    print(f"\nLearned transition probabilities: {fsm.learned_transitions}")

async def demo_chain_of_thought():
    """Demonstrate Chain of Thought reasoning"""
    print("\n" + "="*60)
    print("DEMO 4: Chain of Thought Reasoning")
    print("="*60)
    
    from src.advanced_hybrid_architecture import ChainOfThought, ComplexityAnalyzer, TemplateLibrary
    
    # Create CoT components
    cot = ChainOfThought("demo_cot")
    complexity_analyzer = ComplexityAnalyzer()
    template_library = TemplateLibrary()
    
    # Test queries of different complexity
    queries = [
        "What is 2+2?",
        "Analyze the benefits of using hybrid AI architectures in modern applications",
        "Compare and contrast different approaches to multi-agent systems",
        "Calculate the complexity of implementing a hierarchical FSM with learning capabilities"
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        
        # Analyze complexity
        complexity = complexity_analyzer.analyze(query)
        print(f"Complexity score: {complexity:.3f}")
        
        # Select template
        template = template_library.select_template(query)
        print(f"Selected template: {template}")
        
        # Execute reasoning
        steps = cot.reason(query)
        print(f"Reasoning steps: {len(steps)}")
        
        for i, step in enumerate(steps[:3]):  # Show first 3 steps
            print(f"  Step {i+1}: {step.thought[:80]}... (confidence: {step.confidence:.2f})")

async def demo_performance_optimization():
    """Demonstrate performance optimization features"""
    print("\n" + "="*60)
    print("DEMO 5: Performance Optimization")
    print("="*60)
    
    from src.advanced_hybrid_architecture import PerformanceOptimizer, ResultCache, TaskPredictor, ResourceMonitor
    
    # Create optimization components
    optimizer = PerformanceOptimizer()
    cache = ResultCache(max_size=5)
    predictor = TaskPredictor()
    monitor = ResourceMonitor()
    
    # Create a demo agent
    tools = [CustomTool("calculator", DemoTools.calculator_tool, "Calculate")]
    agent = HybridAgent("optimized_agent", tools)
    
    # Record some task sequences
    task_sequences = [
        [
            {"type": "calculation", "query": "2+2"},
            {"type": "calculation", "query": "3+3"},
            {"type": "calculation", "query": "4+4"}
        ],
        [
            {"type": "analysis", "query": "Analyze text"},
            {"type": "calculation", "query": "5+5"},
            {"type": "analysis", "query": "Analyze data"}
        ]
    ]
    
    for sequence in task_sequences:
        predictor.record_sequence(sequence)
    
    # Test caching
    test_task = {"type": "calculation", "query": "10+10"}
    test_result = {"result": 20}
    
    print("Testing caching...")
    cache.store(test_task, test_result)
    cached_result = cache.get(test_task)
    print(f"Cached result: {cached_result}")
    
    # Test prediction
    current_task = {"type": "calculation", "query": "1+1"}
    predictions = predictor.predict_next_tasks(current_task)
    print(f"Predicted next tasks: {len(predictions)}")
    
    # Test resource monitoring
    monitor.record_usage("cpu", 75.5)
    monitor.record_usage("memory", 512.0)
    monitor.record_usage("execution_time", 2.3)
    
    usage_summary = monitor.get_usage_summary()
    print(f"Resource usage summary: {usage_summary}")

async def demo_emergent_behavior():
    """Demonstrate emergent behavior detection"""
    print("\n" + "="*60)
    print("DEMO 6: Emergent Behavior")
    print("="*60)
    
    from src.advanced_hybrid_architecture import EmergentBehaviorEngine, BehaviorPattern
    
    # Create emergent behavior engine
    engine = EmergentBehaviorEngine()
    
    # Create demo agents
    tools = [CustomTool("calculator", DemoTools.calculator_tool, "Calculate")]
    agent1 = HybridAgent("agent_1", tools)
    agent2 = HybridAgent("agent_2", tools)
    
    # Simulate behavior patterns
    tasks = [
        {"type": "calculation", "query": "Simple math"},
        {"type": "analysis", "query": "Complex analysis"},
        {"type": "reasoning", "query": "Logical reasoning"}
    ]
    
    print("Simulating agent behaviors...")
    
    # Simulate successful patterns
    for i in range(50):
        task = tasks[i % len(tasks)]
        success = i < 40  # 80% success rate for first agent
        
        engine.observe_behavior(agent1, task, {"result": "success"}, success)
        
        # Second agent with different pattern
        success2 = i < 30  # 60% success rate for second agent
        engine.observe_behavior(agent2, task, {"result": "success"}, success2)
    
    print(f"Total behavior patterns recorded: {len(engine.behavior_patterns)}")
    
    # Analyze patterns
    engine.analyze_patterns()
    
    # Test behavior evolution
    original_behavior = {"parameters": {"threshold": 0.5, "timeout": 10}}
    evolved_behavior = engine.evolve_behavior(agent1, original_behavior)
    print(f"Original behavior: {original_behavior}")
    print(f"Evolved behavior: {evolved_behavior}")

async def main():
    """Run all demonstrations"""
    print("Advanced AI Agent Architecture Demonstration")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_basic_hybrid_agent()
        await demo_multi_agent_system()
        await demo_fsm_learning()
        await demo_chain_of_thought()
        await demo_performance_optimization()
        await demo_emergent_behavior()
        
        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 