#!/usr/bin/env python3
"""
Simplified Demonstration of Advanced AI Agent Architecture
Shows core concepts without external dependencies
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import numpy as np
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================
# Core Data Structures
# =============================

@dataclass
class AgentState:
    """Represents the current state of an agent"""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class Transition:
    """Represents a state transition"""
    from_state: str
    to_state: str
    condition: Callable
    probability: float = 1.0
    action: Optional[Callable] = None
    
@dataclass
class ReasoningStep:
    """Represents a step in chain of thought reasoning"""
    step_id: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 1.0

# =============================
# Enhanced FSM Implementation
# =============================

class ProbabilisticFSM:
    """Enhanced FSM with probabilistic transitions and learning capabilities"""
    
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, AgentState] = {}
        self.transitions: List[Transition] = []
        self.current_state: Optional[AgentState] = None
        self.state_history: deque = deque(maxlen=100)
        self.transition_history: deque = deque(maxlen=1000)
        self.learned_transitions: Dict[tuple, float] = {}
        
    def add_state(self, state: AgentState):
        """Add a state to the FSM"""
        self.states[state.name] = state
        
    def add_transition(self, transition: Transition):
        """Add a transition between states"""
        self.transitions.append(transition)
        
    def set_initial_state(self, state_name: str):
        """Set the initial state"""
        if state_name not in self.states:
            raise ValueError(f"State {state_name} not found")
        self.current_state = self.states[state_name]
        self.state_history.append(self.current_state)
        
    def evaluate_transitions(self) -> List[tuple]:
        """Evaluate all possible transitions from current state"""
        if not self.current_state:
            return []
            
        possible_transitions = []
        for transition in self.transitions:
            if transition.from_state == self.current_state.name:
                if transition.condition(self.current_state):
                    # Apply learned probabilities
                    key = (transition.from_state, transition.to_state)
                    learned_prob = self.learned_transitions.get(key, transition.probability)
                    final_prob = 0.7 * transition.probability + 0.3 * learned_prob
                    possible_transitions.append((transition, final_prob))
                    
        return sorted(possible_transitions, key=lambda x: x[1], reverse=True)
        
    def execute_transition(self, transition: Transition):
        """Execute a state transition"""
        if transition.action:
            transition.action(self.current_state)
            
        self.current_state = self.states[transition.to_state]
        self.state_history.append(self.current_state)
        self.transition_history.append((transition, time.time()))
        
        # Update learned probabilities
        self.update_learning(transition)
        
    def update_learning(self, transition: Transition):
        """Update learned transition probabilities based on success"""
        key = (transition.from_state, transition.to_state)
        current = self.learned_transitions.get(key, transition.probability)
        # Simple exponential moving average
        self.learned_transitions[key] = 0.9 * current + 0.1
        
    def step(self) -> bool:
        """Execute one step of the FSM"""
        transitions = self.evaluate_transitions()
        if not transitions:
            return False
            
        # Probabilistic selection
        if len(transitions) == 1:
            selected = transitions[0][0]
        else:
            probs = [t[1] for t in transitions]
            probs = np.array(probs) / sum(probs)
            idx = np.random.choice(len(transitions), p=probs)
            selected = transitions[idx][0]
            
        self.execute_transition(selected)
        return True

# =============================
# Chain of Thought Implementation
# =============================

class ChainOfThought:
    """Optimized CoT with adaptive depth and caching"""
    
    def __init__(self, name: str):
        self.name = name
        self.reasoning_cache: Dict[str, List[ReasoningStep]] = {}
        self.complexity_analyzer = ComplexityAnalyzer()
        self.template_library = TemplateLibrary()
        
    def analyze_complexity(self, query: str) -> float:
        """Analyze query complexity to determine reasoning depth"""
        return self.complexity_analyzer.analyze(query)
        
    def get_cached_reasoning(self, query: str) -> Optional[List[ReasoningStep]]:
        """Check if we have cached reasoning for similar queries"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.reasoning_cache.get(query_hash)
        
    def reason(self, query: str, max_depth: Optional[int] = None) -> List[ReasoningStep]:
        """Execute chain of thought reasoning"""
        # Check cache first
        cached = self.get_cached_reasoning(query)
        if cached:
            logger.info("Using cached reasoning")
            return cached
            
        # Determine reasoning depth based on complexity
        complexity = self.analyze_complexity(query)
        if max_depth is None:
            max_depth = min(int(complexity * 10), 20)
            
        # Select appropriate template
        template = self.template_library.select_template(query)
        
        # Execute reasoning
        steps = self.execute_reasoning(query, template, max_depth)
        
        # Cache successful reasoning
        if steps and steps[-1].confidence > 0.8:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.reasoning_cache[query_hash] = steps
            
        return steps
        
    def execute_reasoning(self, query: str, template: str, 
                         max_depth: int) -> List[ReasoningStep]:
        """Execute the actual reasoning process"""
        steps = []
        current_thought = query
        
        for i in range(max_depth):
            # Generate next thought based on template
            next_thought = f"{template} Step {i+1}: Analyzing '{current_thought}'"
            
            step = ReasoningStep(
                step_id=i,
                thought=next_thought,
                confidence=0.9 - (i * 0.05)  # Decreasing confidence with depth
            )
            steps.append(step)
            
            # Check if we've reached a conclusion
            if "therefore" in next_thought.lower() or i == max_depth - 1:
                break
                
            current_thought = next_thought
            
        return steps

class ComplexityAnalyzer:
    """Analyzes query complexity"""
    
    def analyze(self, query: str) -> float:
        """Return complexity score between 0 and 1"""
        # Simplified implementation
        factors = {
            'length': len(query) / 500,
            'questions': query.count('?') / 5,
            'conjunctions': sum(query.count(w) for w in ['and', 'or', 'but']) / 10,
            'technical_terms': sum(query.count(w) for w in ['calculate', 'analyze', 'evaluate']) / 5
        }
        
        complexity = min(sum(factors.values()) / len(factors), 1.0)
        return complexity

class TemplateLibrary:
    """Library of reasoning templates"""
    
    def __init__(self):
        self.templates = {
            'mathematical': "Let me solve this step by step using mathematical principles.",
            'analytical': "I'll analyze this by breaking it down into components.",
            'comparative': "I'll compare and contrast the different aspects.",
            'default': "Let me think through this systematically."
        }
        
    def select_template(self, query: str) -> str:
        """Select appropriate template based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['calculate', 'solve', 'compute']):
            return self.templates['mathematical']
        elif any(word in query_lower for word in ['analyze', 'examine', 'investigate']):
            return self.templates['analytical']
        elif any(word in query_lower for word in ['compare', 'contrast', 'difference']):
            return self.templates['comparative']
        else:
            return self.templates['default']

# =============================
# Simple Tool System
# =============================

class SimpleTool:
    """Simple tool implementation"""
    
    def __init__(self, name: str, function: Callable, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    def run(self, **kwargs):
        return self.function(**kwargs)

class DemoTools:
    """Demo tools for testing"""
    
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

# =============================
# ReAct Implementation
# =============================

class ReActAgent:
    """Simple ReAct agent implementation"""
    
    def __init__(self, name: str, tools: List[SimpleTool], max_steps: int = 10):
        self.name = name
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.reasoning_history: List[ReasoningStep] = []
        self.tool_usage_stats: Dict[str, int] = defaultdict(int)
        
    def think(self, observation: str, context: Dict[str, Any]) -> str:
        """Generate a thought based on observation"""
        thought = f"Analyzing: {observation}. Context indicates we need to use available tools."
        return thought
        
    def act(self, thought: str, context: Dict[str, Any]) -> tuple:
        """Decide on an action based on thought"""
        available_tools = list(self.tools.keys())
        if available_tools:
            selected_tool = available_tools[0]  # Simple selection
            self.tool_usage_stats[selected_tool] += 1
            return selected_tool, {}
        return "no_action", None
        
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return observation"""
        if tool_name not in self.tools:
            return f"Tool {tool_name} not found"
            
        tool = self.tools[tool_name]
        try:
            result = tool.run(**args)
            return json.dumps(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
            
    async def reasoning_path(self, query: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Execute a single reasoning path"""
        steps = []
        observation = query
        
        for step_num in range(self.max_steps):
            # Think
            thought = self.think(observation, context)
            
            # Act
            action, args = self.act(thought, context)
            
            # Execute
            if action != "no_action":
                observation = self.execute_tool(action, args or {})
            else:
                observation = "No action taken"
                
            step = ReasoningStep(
                step_id=step_num,
                thought=thought,
                action=action,
                observation=observation,
                confidence=0.8 + 0.2 * np.random.random()  # Simulated confidence
            )
            steps.append(step)
            
            # Check if we have a final answer
            if "final_answer" in observation.lower():
                break
                
        return steps

# =============================
# Unified Hybrid Agent
# =============================

class HybridAgent:
    """Unified agent combining FSM, ReAct, and CoT"""
    
    def __init__(self, name: str, tools: List[SimpleTool] = None):
        self.name = name
        self.fsm = ProbabilisticFSM(f"{name}_fsm")
        self.react = ReActAgent(f"{name}_react", tools or [])
        self.cot = ChainOfThought(f"{name}_cot")
        
        self.current_mode = "fsm"
        self.mode_performance: Dict[str, float] = {
            "fsm": 0.5,
            "react": 0.5,
            "cot": 0.5
        }
        
    def select_mode(self, task: Dict[str, Any]) -> str:
        """Select the best mode for the current task"""
        task_type = task.get("type", "unknown")
        
        # Simple heuristics for mode selection
        if task_type == "navigation" or task_type == "state_based":
            return "fsm"
        elif task_type == "tool_use" or task_type == "external_interaction":
            return "react"
        elif task_type == "reasoning" or task_type == "analysis":
            return "cot"
        else:
            # Select based on past performance
            return max(self.mode_performance.items(), key=lambda x: x[1])[0]
            
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task using the appropriate mode"""
        mode = self.select_mode(task)
        self.current_mode = mode
        
        start_time = time.time()
        result = None
        success = False
        
        try:
            if mode == "fsm":
                result = await self.execute_fsm_task(task)
            elif mode == "react":
                result = await self.execute_react_task(task)
            elif mode == "cot":
                result = await self.execute_cot_task(task)
                
            success = result is not None
            
        except Exception as e:
            logger.error(f"Error executing task in {mode} mode: {str(e)}")
            
        # Update performance metrics
        execution_time = time.time() - start_time
        self.update_performance(mode, success, execution_time)
        
        return result
        
    async def execute_fsm_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using FSM"""
        # Setup FSM for the task
        states = task.get("states", [])
        for state_data in states:
            state = AgentState(name=state_data["name"], data=state_data.get("data", {}))
            self.fsm.add_state(state)
            
        # Run FSM
        self.fsm.set_initial_state(states[0]["name"])
        
        while self.fsm.step():
            await asyncio.sleep(0.1)  # Simulate processing
            
        return {"final_state": self.fsm.current_state.name}
        
    async def execute_react_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using ReAct"""
        query = task.get("query", "")
        context = task.get("context", {})
        
        # Execute reasoning path
        steps = await self.react.reasoning_path(query, context)
        
        return {"reasoning_path": steps}
        
    async def execute_cot_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using Chain of Thought"""
        query = task.get("query", "")
        
        # Execute reasoning
        steps = self.cot.reason(query)
        
        return {"reasoning_steps": steps}
        
    def update_performance(self, mode: str, success: bool, execution_time: float):
        """Update performance metrics for mode selection"""
        # Simple exponential moving average
        alpha = 0.1
        performance_score = (1.0 if success else 0.0) * (1.0 / (1.0 + execution_time))
        
        self.mode_performance[mode] = (
            (1 - alpha) * self.mode_performance[mode] + 
            alpha * performance_score
        )

# =============================
# Demonstration Functions
# =============================

async def demo_basic_hybrid_agent():
    """Demonstrate basic hybrid agent functionality"""
    print("\n" + "="*60)
    logger.info("DEMO 1: Basic Hybrid Agent")
    print("="*60)
    
    # Create demo tools
    tools = [
        SimpleTool("calculator", DemoTools.calculator_tool, "Perform mathematical calculations"),
        SimpleTool("text_analyzer", DemoTools.text_analyzer_tool, "Analyze text statistics"),
        SimpleTool("search", DemoTools.mock_search_tool, "Search for information")
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
        logger.info("\n--- Task {}: {} ---", extra={"i": i, "task__type_": task['type']})
        logger.info("Query: {}", extra={"task_get__query____State_based_task__": task.get('query', 'State-based task')})
        
        result = await agent.execute_task(task)
        logger.info("Mode used: {}", extra={"agent_current_mode": agent.current_mode})
        logger.info("Result: {}", extra={"result": result})
        
        # Show performance metrics
        logger.info("Performance by mode: {}", extra={"agent_mode_performance": agent.mode_performance})

async def demo_fsm_learning():
    """Demonstrate FSM learning capabilities"""
    print("\n" + "="*60)
    logger.info("DEMO 2: FSM Learning")
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
    
    logger.info("Running FSM with learning...")
    logger.info("Initial state:", extra={"data": fsm.current_state.name})
    
    # Run FSM for several steps
    for step in range(10):
        if fsm.step():
            logger.info("Step {}: {} (energy: {})", extra={"step___1": step + 1, "fsm_current_state_name": fsm.current_state.name, "fsm_current_state_data_get__energy___0_": fsm.current_state.data.get('energy', 0)})
            
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
            logger.info("Step {}: No valid transitions", extra={"step___1": step + 1})
            break
    
    logger.info("\nLearned transition probabilities: {}", extra={"fsm_learned_transitions": fsm.learned_transitions})

async def demo_chain_of_thought():
    """Demonstrate Chain of Thought reasoning"""
    print("\n" + "="*60)
    logger.info("DEMO 3: Chain of Thought Reasoning")
    print("="*60)
    
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
        logger.info("\n--- Query: {} ---", extra={"query": query})
        
        # Analyze complexity
        complexity = complexity_analyzer.analyze(query)
        logger.info("Complexity score: {}", extra={"complexity": complexity})
        
        # Select template
        template = template_library.select_template(query)
        logger.info("Selected template: {}", extra={"template": template})
        
        # Execute reasoning
        steps = cot.reason(query)
        logger.info("Reasoning steps: {}", extra={"len_steps_": len(steps)})
        
        for i, step in enumerate(steps[:3]):  # Show first 3 steps
            logger.info("  Step {}: {}... (confidence: {})", extra={"i_1": i+1, "step_thought_": step.thought[, "step_confidence": step.confidence})

async def main():
    """Run all demonstrations"""
    logger.info("Advanced AI Agent Architecture - Simplified Demonstration")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_basic_hybrid_agent()
        await demo_fsm_learning()
        await demo_chain_of_thought()
        
        print("\n" + "="*60)
        logger.info("All demonstrations completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        logger.info("Error: {}", extra={"str_e_": str(e)})

if __name__ == "__main__":
    asyncio.run(main()) 