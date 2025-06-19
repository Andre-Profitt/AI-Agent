"""
Advanced AI Agent Architecture Implementation
A comprehensive framework combining FSM, ReAct, and Chain of Thought approaches
Integrated with existing codebase structure
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import operator
from datetime import datetime
from contextlib import contextmanager

# Import existing codebase components
from src.advanced_agent_fsm import (
    FSMReActAgent, EnhancedAgentState, ResilientAPIClient, 
    PlanResponse, PlanStep, ValidationResult, validate_user_prompt
)
from src.tools.base_tool import BaseTool
from src.reasoning.reasoning_path import ReasoningPath, ReasoningType
from src.errors.error_category import ErrorCategory
from src.core.services.data_quality import DataQualityLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    
@dataclass
class Tool:
    """Represents an external tool that agents can use"""
    name: str
    description: str
    function: Callable
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

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
        self.learned_transitions: Dict[Tuple[str, str], float] = {}
        
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
        
    def evaluate_transitions(self) -> List[Tuple[Transition, float]]:
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
# Hierarchical FSM
# =============================

class HierarchicalFSM(ProbabilisticFSM):
    """Hierarchical FSM with parent-child relationships"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.child_fsms: Dict[str, ProbabilisticFSM] = {}
        self.parent_fsm: Optional[HierarchicalFSM] = None
        
    def add_child_fsm(self, state_name: str, child_fsm: ProbabilisticFSM):
        """Add a child FSM to a state"""
        if state_name not in self.states:
            raise ValueError(f"State {state_name} not found")
        self.child_fsms[state_name] = child_fsm
        if isinstance(child_fsm, HierarchicalFSM):
            child_fsm.parent_fsm = self
            
    def step(self) -> bool:
        """Execute one step, including child FSMs"""
        # Check if current state has a child FSM
        if self.current_state and self.current_state.name in self.child_fsms:
            child_fsm = self.child_fsms[self.current_state.name]
            child_result = child_fsm.step()
            
            # If child FSM completed, transition in parent
            if not child_result:
                return super().step()
            return True
        else:
            return super().step()

# =============================
# Advanced ReAct Implementation
# =============================

class ReActAgent:
    """Enhanced ReAct agent with parallel reasoning and dynamic tool discovery"""
    
    def __init__(self, name: str, tools: List[BaseTool], max_steps: int = 10):
        self.name = name
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.reasoning_history: List[ReasoningStep] = []
        self.tool_usage_stats: Dict[str, int] = defaultdict(int)
        self.discovered_tools: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def think(self, observation: str, context: Dict[str, Any]) -> str:
        """Generate a thought based on observation"""
        # Simplified - in practice, this would call an LLM
        thought = f"Analyzing: {observation}. Context indicates we need to use available tools."
        return thought
        
    def act(self, thought: str, context: Dict[str, Any]) -> Tuple[str, Any]:
        """Decide on an action based on thought"""
        # Simplified tool selection - in practice, use LLM
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
            
    async def parallel_reasoning(self, query: str, context: Dict[str, Any], 
                               num_paths: int = 3) -> List[List[ReasoningStep]]:
        """Execute multiple reasoning paths in parallel"""
        tasks = []
        for i in range(num_paths):
            task = asyncio.create_task(self.reasoning_path(query, context.copy(), i))
            tasks.append(task)
            
        paths = await asyncio.gather(*tasks)
        return paths
        
    async def reasoning_path(self, query: str, context: Dict[str, Any], 
                           path_id: int) -> List[ReasoningStep]:
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
        
    def discover_tool(self, tool: BaseTool):
        """Dynamically discover and add a new tool"""
        self.tools[tool.name] = tool
        self.discovered_tools.add(tool.name)
        logger.info(f"Discovered new tool: {tool.name}")

# =============================
# Optimized Chain of Thought
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
# Unified Hybrid Architecture
# =============================

class HybridAgent:
    """Unified agent combining FSM, ReAct, and CoT with existing FSMReActAgent integration"""
    
    def __init__(self, name: str, tools: List[BaseTool] = None):
        self.name = name
        self.fsm = HierarchicalFSM(f"{name}_fsm")
        self.react = ReActAgent(f"{name}_react", tools or [])
        self.cot = ChainOfThought(f"{name}_cot")
        
        # Integrate with existing FSMReActAgent
        self.fsm_react_agent = FSMReActAgent(
            tools=tools or [],
            model_name="llama-3.3-70b-versatile",
            quality_level=DataQualityLevel.THOROUGH,
            reasoning_type=ReasoningType.LAYERED
        )
        
        self.current_mode = "fsm"
        self.mode_performance: Dict[str, float] = {
            "fsm": 0.5,
            "react": 0.5,
            "cot": 0.5,
            "fsm_react": 0.5
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
        elif task_type == "complex" or task_type == "gaia":
            return "fsm_react"  # Use existing FSMReActAgent for complex tasks
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
            elif mode == "fsm_react":
                result = await self.execute_fsm_react_task(task)
                
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
        
        # Add tools for the task
        tools = task.get("tools", [])
        for tool_data in tools:
            tool = Tool(**tool_data)
            self.react.discover_tool(tool)
            
        # Execute parallel reasoning
        paths = await self.react.parallel_reasoning(query, context)
        
        # Select best path based on confidence
        best_path = max(paths, key=lambda p: sum(s.confidence for s in p))
        
        return {"reasoning_path": best_path}
        
    async def execute_cot_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using Chain of Thought"""
        query = task.get("query", "")
        
        # Execute reasoning
        steps = self.cot.reason(query)
        
        return {"reasoning_steps": steps}
        
    async def execute_fsm_react_task(self, task: Dict[str, Any]) -> Any:
        """Execute task using existing FSMReActAgent"""
        query = task.get("query", "")
        
        # Validate input
        validation_result = validate_user_prompt(query)
        if not validation_result.is_valid:
            return {"error": "Invalid input", "details": validation_result.validation_errors}
        
        # Create initial state
        initial_state = {
            "query": query,
            "input_query": validation_result,
            "plan": "",
            "master_plan": [],
            "validated_plan": None,
            "tool_calls": [],
            "step_outputs": {},
            "final_answer": "",
            "current_fsm_state": "PLANNING",
            "stagnation_counter": 0,
            "max_stagnation": 5,
            "retry_count": 0,
            "max_retries": 3,
            "failure_history": [],
            "circuit_breaker_status": "closed",
            "last_api_error": None,
            "verification_level": "thorough",
            "confidence": 1.0,
            "cross_validation_sources": [],
            "messages": [],
            "errors": [],
            "step_count": 0,
            "start_time": time.time(),
            "end_time": 0.0,
            "tool_reliability": {},
            "tool_preferences": {},
            "turn_count": 0,
            "action_history": [],
            "stagnation_score": 0,
            "error_log": [],
            "error_counts": {},
            "remaining_loops": 15,
            "last_state_hash": "",
            "force_termination": False,
            "tool_errors": [],
            "recovery_attempts": 0,
            "fallback_level": 0,
            "draft_answer": "",
            "reflection_passed": False,
            "reflection_issues": [],
            "requires_human_approval": False,
            "approval_request": None,
            "execution_paused": False
        }
        
        # Execute using FSMReActAgent
        result = await self.fsm_react_agent.run(initial_state)
        
        return result
        
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
# Multi-Agent Collaboration
# =============================

class AgentRegistry:
    """Registry for agent discovery and management"""
    
    def __init__(self):
        self.agents: Dict[str, HybridAgent] = {}
        self.capabilities: Dict[str, List[str]] = defaultdict(list)
        
    def register_agent(self, agent: HybridAgent, capabilities: List[str]):
        """Register an agent with its capabilities"""
        self.agents[agent.name] = agent
        for capability in capabilities:
            self.capabilities[capability].append(agent.name)
            
    def find_agents_by_capability(self, capability: str) -> List[HybridAgent]:
        """Find agents with a specific capability"""
        agent_names = self.capabilities.get(capability, [])
        return [self.agents[name] for name in agent_names if name in self.agents]

class MultiAgentSystem:
    """Orchestrates collaboration between multiple agents"""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.shared_memory = SharedMemory()
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        
    def add_agent(self, agent: HybridAgent, capabilities: List[str]):
        """Add an agent to the system"""
        self.registry.register_agent(agent, capabilities)
        
    async def distribute_task(self, task: Dict[str, Any]) -> Any:
        """Distribute a task to appropriate agents"""
        required_capability = task.get("required_capability", "general")
        agents = self.registry.find_agents_by_capability(required_capability)
        
        if not agents:
            logger.warning(f"No agents found for capability: {required_capability}")
            return None
            
        # Simple round-robin distribution
        agent = agents[0]  # In practice, use more sophisticated selection
        
        # Share task context
        task_id = task.get("id", str(time.time()))
        self.shared_memory.store(f"task_{task_id}_context", task.get("context", {}))
        
        # Execute task
        result = await agent.execute_task(task)
        
        # Store result in shared memory
        self.shared_memory.store(f"task_{task_id}_result", result)
        
        return result
        
    async def collaborate_on_task(self, complex_task: Dict[str, Any]) -> Any:
        """Multiple agents collaborate on a complex task"""
        subtasks = complex_task.get("subtasks", [])
        results = []
        
        # Execute subtasks in parallel
        tasks = []
        for subtask in subtasks:
            task = asyncio.create_task(self.distribute_task(subtask))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        return self.aggregate_results(results)
        
    def aggregate_results(self, results: List[Any]) -> Any:
        """Aggregate results from multiple agents"""
        # Simple aggregation - in practice, use more sophisticated methods
        return {
            "aggregated_results": results,
            "summary": f"Completed {len(results)} subtasks"
        }

class SharedMemory:
    """Shared memory for inter-agent communication"""
    
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.access_log: List[Tuple[str, str, float]] = []
        
    def store(self, key: str, value: Any):
        """Store a value in shared memory"""
        self.memory[key] = value
        self.access_log.append(("store", key, time.time()))
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from shared memory"""
        self.access_log.append(("retrieve", key, time.time()))
        return self.memory.get(key)
        
    def search(self, pattern: str) -> Dict[str, Any]:
        """Search for keys matching a pattern"""
        results = {}
        for key, value in self.memory.items():
            if pattern in key:
                results[key] = value
        return results

# =============================
# Emergent Behavior System
# =============================

class EmergentBehaviorEngine:
    """Enables agents to discover and evolve new behaviors"""
    
    def __init__(self):
        self.behavior_patterns: List[BehaviorPattern] = []
        self.success_threshold = 0.8
        self.mutation_rate = 0.1
        
    def observe_behavior(self, agent: HybridAgent, task: Dict[str, Any], 
                        result: Any, success: bool):
        """Observe and record agent behavior"""
        pattern = BehaviorPattern(
            agent_name=agent.name,
            task_type=task.get("type", "unknown"),
            mode_used=agent.current_mode,
            success=success,
            timestamp=time.time()
        )
        self.behavior_patterns.append(pattern)
        
        # Check for emergent patterns
        self.analyze_patterns()
        
    def analyze_patterns(self):
        """Analyze behavior patterns for emergent behaviors"""
        if len(self.behavior_patterns) < 100:
            return
            
        # Group by task type and mode
        pattern_groups = defaultdict(list)
        for pattern in self.behavior_patterns[-1000:]:  # Last 1000 patterns
            key = (pattern.task_type, pattern.mode_used)
            pattern_groups[key].append(pattern)
            
        # Find successful patterns
        for key, patterns in pattern_groups.items():
            success_rate = sum(p.success for p in patterns) / len(patterns)
            if success_rate > self.success_threshold:
                logger.info(f"Successful pattern found: {key} with {success_rate:.2%} success rate")
                
    def evolve_behavior(self, agent: HybridAgent, behavior: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve a behavior through mutation"""
        evolved = behavior.copy()
        
        # Simple mutation
        if np.random.random() < self.mutation_rate:
            # Mutate some aspect of the behavior
            if "parameters" in evolved:
                for param, value in evolved["parameters"].items():
                    if isinstance(value, (int, float)):
                        evolved["parameters"][param] = value * (1 + np.random.normal(0, 0.1))
                        
        return evolved

@dataclass
class BehaviorPattern:
    """Represents an observed behavior pattern"""
    agent_name: str
    task_type: str
    mode_used: str
    success: bool
    timestamp: float

# =============================
# Performance Optimization
# =============================

class PerformanceOptimizer:
    """Optimizes agent performance through caching and prediction"""
    
    def __init__(self):
        self.result_cache = ResultCache()
        self.predictor = TaskPredictor()
        self.resource_monitor = ResourceMonitor()
        
    def optimize_execution(self, agent: HybridAgent, task: Dict[str, Any]) -> Any:
        """Optimize task execution"""
        # Check cache first
        cached_result = self.result_cache.get(task)
        if cached_result:
            logger.info("Using cached result")
            return cached_result
            
        # Predict next likely tasks
        predictions = self.predictor.predict_next_tasks(task)
        
        # Precompute likely tasks
        for predicted_task in predictions[:3]:  # Top 3 predictions
            asyncio.create_task(self.precompute_task(agent, predicted_task))
            
        return None
        
    async def precompute_task(self, agent: HybridAgent, task: Dict[str, Any]):
        """Precompute a predicted task"""
        result = await agent.execute_task(task)
        self.result_cache.store(task, result)

class ResultCache:
    """Caches task results"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
        
    def get(self, task: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for a task"""
        key = self._task_key(task)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def store(self, task: Dict[str, Any], result: Any):
        """Store a result in cache"""
        key = self._task_key(task)
        
        # Evict old entries if needed
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = result
        self.access_times[key] = time.time()
        
    def _task_key(self, task: Dict[str, Any]) -> str:
        """Generate a unique key for a task"""
        # Simple hash of task dict
        task_str = json.dumps(task, sort_keys=True)
        return hashlib.md5(task_str.encode()).hexdigest()

class TaskPredictor:
    """Predicts likely next tasks"""
    
    def __init__(self):
        self.task_sequences: List[List[Dict[str, Any]]] = []
        
    def record_sequence(self, tasks: List[Dict[str, Any]]):
        """Record a sequence of tasks"""
        self.task_sequences.append(tasks)
        
    def predict_next_tasks(self, current_task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely next tasks based on patterns"""
        predictions = []
        
        # Find similar tasks in history
        for sequence in self.task_sequences:
            for i, task in enumerate(sequence[:-1]):
                if self._tasks_similar(task, current_task):
                    next_task = sequence[i + 1]
                    predictions.append(next_task)
                    
        # Return most common predictions
        # Simplified - in practice, use more sophisticated prediction
        return predictions[:5]
        
    def _tasks_similar(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """Check if two tasks are similar"""
        return task1.get("type") == task2.get("type")

class ResourceMonitor:
    """Monitors and optimizes resource usage"""
    
    def __init__(self):
        self.usage_stats: Dict[str, List[float]] = defaultdict(list)
        
    def record_usage(self, resource: str, amount: float):
        """Record resource usage"""
        self.usage_stats[resource].append(amount)
        
    def get_usage_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of resource usage"""
        summary = {}
        for resource, usage_list in self.usage_stats.items():
            if usage_list:
                summary[resource] = {
                    "mean": np.mean(usage_list),
                    "max": np.max(usage_list),
                    "min": np.min(usage_list),
                    "total": np.sum(usage_list)
                }
        return summary

# =============================
# Integration with Existing Codebase
# =============================

class AdvancedHybridSystem:
    """Main system that integrates all components with existing codebase"""
    
    def __init__(self):
        self.multi_agent_system = MultiAgentSystem()
        self.emergent_behavior_engine = EmergentBehaviorEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.resource_monitor = ResourceMonitor()
        
    def create_agent(self, name: str, tools: List[BaseTool], capabilities: List[str]) -> HybridAgent:
        """Create and register a new hybrid agent"""
        agent = HybridAgent(name, tools)
        self.multi_agent_system.add_agent(agent, capabilities)
        return agent
        
    async def execute_complex_task(self, task: Dict[str, Any]) -> Any:
        """Execute a complex task using the hybrid system"""
        # Monitor resource usage
        start_time = time.time()
        
        # Execute task
        result = await self.multi_agent_system.collaborate_on_task(task)
        
        # Record resource usage
        execution_time = time.time() - start_time
        self.resource_monitor.record_usage("execution_time", execution_time)
        
        # Observe behavior for emergent patterns
        if "agents" in result:
            for agent_result in result["agents"]:
                self.emergent_behavior_engine.observe_behavior(
                    agent_result["agent"],
                    task,
                    agent_result["result"],
                    agent_result.get("success", True)
                )
                
        return result
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        return {
            "resource_usage": self.resource_monitor.get_usage_summary(),
            "agent_performance": {
                agent.name: agent.mode_performance 
                for agent in self.multi_agent_system.registry.agents.values()
            },
            "behavior_patterns": len(self.emergent_behavior_engine.behavior_patterns),
            "cache_stats": {
                "size": len(self.performance_optimizer.result_cache.cache),
                "max_size": self.performance_optimizer.result_cache.max_size
            }
        }

# =============================
# Example Usage and Integration
# =============================

async def main():
    """Example usage of the advanced hybrid architecture"""
    
    # Create the hybrid system
    system = AdvancedHybridSystem()
    
    # Create tools (using existing BaseTool structure)
    from src.tools.semantic_search_tool import SemanticSearchTool
    from src.tools.python_interpreter import PythonInterpreter
    
    tools = [
        SemanticSearchTool(),
        PythonInterpreter()
    ]
    
    # Create agents with different capabilities
    general_agent = system.create_agent("general_agent", tools, ["general", "search", "calculation"])
    reasoning_agent = system.create_agent("reasoning_agent", tools, ["reasoning", "analysis"])
    
    # Example tasks
    tasks = [
        {
            "type": "complex",
            "query": "What are the benefits of using hybrid AI architectures?",
            "subtasks": [
                {"type": "reasoning", "query": "Analyze the problem", "required_capability": "reasoning"},
                {"type": "tool_use", "query": "Search for solutions", "required_capability": "search"}
            ]
        },
        {
            "type": "gaia",
            "query": "How many birds are in the video?",
            "required_capability": "general"
        }
    ]
    
    # Execute tasks
    for task in tasks:
        print(f"\nExecuting task: {task['type']}")
        result = await system.execute_complex_task(task)
        print(f"Result: {result}")
        
    # Show system health
    health = system.get_system_health()
    print(f"\nSystem Health: {health}")

if __name__ == "__main__":
    asyncio.run(main()) 