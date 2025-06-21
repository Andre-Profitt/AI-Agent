from agent import path
from agent import query
from agent import tools
from app import history
from benchmarks.cot_performance import cot_system
from benchmarks.cot_performance import filename
from benchmarks.cot_performance import insights
from benchmarks.cot_performance import total
from examples.basic.simple_hybrid_demo import mode
from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import agents
from examples.parallel_execution_example import results
from fix_security_issues import patterns
from fix_security_issues import report
from migrations.env import config
from setup_environment import value

from src.agents.advanced_hybrid_architecture import base_metrics
from src.agents.advanced_hybrid_architecture import best_result
from src.agents.advanced_hybrid_architecture import combined_answer
from src.agents.advanced_hybrid_architecture import combined_confidence
from src.agents.advanced_hybrid_architecture import confidence_values
from src.agents.advanced_hybrid_architecture import cot_metrics
from src.agents.advanced_hybrid_architecture import cot_result
from src.agents.advanced_hybrid_architecture import cot_task
from src.agents.advanced_hybrid_architecture import cot_weight
from src.agents.advanced_hybrid_architecture import early_confidence
from src.agents.advanced_hybrid_architecture import emergent_insights
from src.agents.advanced_hybrid_architecture import execution_result
from src.agents.advanced_hybrid_architecture import fsm_result
from src.agents.advanced_hybrid_architecture import fsm_task
from src.agents.advanced_hybrid_architecture import fsm_weight
from src.agents.advanced_hybrid_architecture import hybrid_result
from src.agents.advanced_hybrid_architecture import late_confidence
from src.agents.advanced_hybrid_architecture import mode_counts
from src.agents.advanced_hybrid_architecture import mode_sequence
from src.agents.advanced_hybrid_architecture import preferences
from src.agents.advanced_hybrid_architecture import primary_answer
from src.agents.advanced_hybrid_architecture import recent_states
from src.agents.advanced_hybrid_architecture import research_result
from src.agents.advanced_hybrid_architecture import secondary_answer
from src.agents.advanced_hybrid_architecture import selected_mode
from src.agents.advanced_hybrid_architecture import synthesis_result
from src.agents.advanced_hybrid_architecture import test_queries
from src.agents.advanced_hybrid_architecture import total_weight
from src.agents.enhanced_fsm import state
from src.api_server import fsm_agent
from src.core.entities.agent import Agent
from src.core.monitoring import key
from src.core.optimized_chain_of_thought import features
from src.core.optimized_chain_of_thought import n
from src.core.optimized_chain_of_thought import step
from src.database.models import reasoning_path
from src.gaia_components.performance_optimization import entry
from src.gaia_components.production_vector_store import count
from src.meta_cognition import complexity
from src.meta_cognition import confidence
from src.tools import python_interpreter
from src.tools_introspection import code
from src.tools_introspection import improvements
from src.tools_introspection import name
from src.unified_architecture.conflict_resolution import states
from src.utils.semantic_search_tool import semantic_search_tool
from src.utils.tools_introspection import field
from src.utils.weather import get_weather

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Dict, Enum, List, Optional, agents, auto, base_metrics, best_result, code, combined_answer, combined_confidence, complexity, complexity_score, confidence, confidence_values, config, context, cot_metrics, cot_result, cot_system, cot_task, cot_weight, count, dataclass, defaultdict, e, early_confidence, emergent_insights, entry, execution_result, execution_time, features, field, filename, fsm_agent, fsm_result, fsm_task, fsm_weight, history, hybrid_result, i, improvements, insights, k, key, late_confidence, location, logging, mode, mode_counts, mode_sequence, n, name, path, patterns, preferences, primary_answer, query, r, reasoning_path, recent_states, report, research_result, result, results, s, secondary_answer, selected_mode, start_time, state, state_history, states, step, synthesis_result, tasks, test_queries, time, tools, top_k, total, total_weight, units, v, value
from concurrent.futures import ThreadPoolExecutor
from tests.test_gaia_agent import agent

from src.utils.base_tool import get_weather
from src.utils.base_tool import semantic_search_tool
from src.utilsthon_interpreter import python_interpreter


"""
from collections import defaultdict
from typing import Dict
from enum import auto
from src.core.optimized_chain_of_thought import ComplexityAnalyzer
from src.core.optimized_chain_of_thought import OptimizedChainOfThought
from src.reasoning.reasoning_path import ReasoningPath
from src.reasoning.reasoning_path import ReasoningType
from src.shared.types.di_types import BaseTool
# TODO: Fix undefined variables: ThreadPoolExecutor, agent, agents, base_metrics, best_result, code, combined_answer, combined_confidence, complexity, complexity_score, confidence, confidence_values, config, context, cot_metrics, cot_result, cot_system, cot_task, cot_weight, count, e, early_confidence, emergent_insights, entry, execution_result, execution_time, features, filename, fsm_agent, fsm_result, fsm_task, fsm_weight, get_weather, history, hybrid_result, i, improvements, insights, k, key, late_confidence, location, mode, mode_counts, mode_sequence, n, name, path, patterns, preferences, primary_answer, python_interpreter, query, r, reasoning_path, recent_states, report, research_result, result, results, s, secondary_answer, selected_mode, self, semantic_search_tool, start_time, state, state_history, states, step, synthesis_result, tasks, test_queries, tools, top_k, total, total_weight, units, v, value

from langchain.tools import BaseTool
Advanced Hybrid AI Agent Architecture
Combining FSM, ReAct, and Chain of Thought approaches with advanced features
"""

from dataclasses import field
from typing import List
from typing import Optional
from typing import Any

import asyncio

import logging
import time

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

# Import existing components with relative imports
try:
    from .advanced_agent_fsm import FSMReActAgent
    from .tools.base_tool import BaseTool
    from .tools.semantic_search_tool import semantic_search_tool
    from .tools.python_interpreter import python_interpreter
    from .tools.weather import get_weather
except ImportError:
    # Fallback for direct execution
    from src.agents.advanced_agent_fsm import FSMReActAgent
    from src.tools.base_tool import BaseTool
    from tools.semantic_search_tool import semantic_search_tool
    from tools.python_interpreter import python_interpreter
    from tools.weather import get_weather

# Import the new Optimized Chain of Thought system
try:
    from .optimized_chain_of_thought import (
        OptimizedChainOfThought, ReasoningPath, ReasoningStep, ReasoningType,
        ComplexityAnalyzer, TemplateLibrary, ReasoningCache, MultiPathReasoning,
        MetacognitiveLayer
    )
except ImportError:
    # Fallback for direct execution
    from optimized_chain_of_thought import (
        OptimizedChainOfThought, ReasoningPath, ReasoningStep, ReasoningType,
        ComplexityAnalyzer, TemplateLibrary, ReasoningCache, MultiPathReasoning,
        MetacognitiveLayer
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# Enhanced Data Structures
# =============================

class AgentMode(Enum):
    """Enhanced agent operation modes"""
    FSM_REACT = auto()           # Finite State Machine with ReAct
    CHAIN_OF_THOUGHT = auto()    # Optimized Chain of Thought
    HYBRID_ADAPTIVE = auto()     # Adaptive mode selection
    MULTI_AGENT = auto()         # Multi-agent collaboration
    EMERGENT_BEHAVIOR = auto()   # Emergent behavior detection
    PROBABILISTIC_FSM = auto()   # Probabilistic FSM
    HIERARCHICAL_FSM = auto()    # Hierarchical FSM
    PARALLEL_REASONING = auto()  # Parallel reasoning paths

@dataclass
class AgentState:
    """Enhanced agent state with CoT integration"""
    mode: AgentMode
    current_query: str
    reasoning_path: Optional[ReasoningPath] = None
    fsm_state: Optional[str] = None
    confidence: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class PerformanceMetrics:
    """Enhanced performance tracking"""
    total_queries: int = 0
    mode_usage: Dict[AgentMode, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    cache_hit_rate: float = 0.0
    cot_performance: Dict[str, Any] = field(default_factory=dict)
    fsm_performance: Dict[str, Any] = field(default_factory=dict)

# =============================
# Enhanced Hybrid Agent
# =============================

class AdvancedHybridAgent:
    """Advanced hybrid agent with CoT integration"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

        # Initialize core components
        try:
            self.fsm_agent = FSMReActAgent(
                tools=self._initialize_tools()
            )
        except Exception as e:
            logger.warning("FSM agent initialization failed: {}", extra={"e": e})
            self.fsm_agent = None

        # Initialize Optimized Chain of Thought system
        self.cot_system = OptimizedChainOfThought(
            name=f"{name}_cot",
            config=self.config.get('cot', {
                'max_paths': 5,
                'cache_size': 1000,
                'cache_ttl': 24
            })
        )

        # Initialize enhanced components
        self.complexity_analyzer = ComplexityAnalyzer()
        self.mode_selector = AdaptiveModeSelector()
        self.performance_tracker = PerformanceTracker()
        self.multi_agent_coordinator = MultiAgentCoordinator()
        self.emergent_behavior_detector = EmergentBehaviorDetector()

        # State management
        self.current_state = AgentState(mode=AgentMode.HYBRID_ADAPTIVE, current_query="")
        self.state_history: List[AgentState] = []

        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}

        logger.info("Advanced Hybrid Agent '{}' initialized successfully", extra={"name": name})

    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize tools for the agent"""
        tools = []
        try:
            tools.append(SemanticSearchTool())
        except Exception as e:
            logger.warning("SemanticSearchTool not available: {}", extra={"e": e})

        try:
            tools.append(PythonInterpreter())
        except Exception as e:
            logger.warning("PythonInterpreter not available: {}", extra={"e": e})

        try:
            tools.append(WeatherTool())
        except Exception as e:
            logger.warning("WeatherTool not available: {}", extra={"e": e})

        return tools

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main query processing method with enhanced capabilities"""
        start_time = time.time()

        # Update state
        self.current_state.current_query = query
        self.current_state.timestamp = time.time()

        # Analyze query complexity
        complexity_score, features = self.complexity_analyzer.analyze(query)
        logger.info("Query complexity: {}", extra={"complexity_score": complexity_score})

        # Select optimal mode
        selected_mode = self.mode_selector.select_mode(
            query, complexity_score, features, self.current_state
        )
        self.current_state.mode = selected_mode

        # Process based on selected mode
        if selected_mode == AgentMode.CHAIN_OF_THOUGHT:
            result = await self._process_with_cot(query, context, complexity_score)
        elif selected_mode == AgentMode.FSM_REACT and self.fsm_agent:
            result = await self._process_with_fsm(query, context)
        elif selected_mode == AgentMode.HYBRID_ADAPTIVE:
            result = await self._process_hybrid(query, context, complexity_score)
        elif selected_mode == AgentMode.MULTI_AGENT:
            result = await self._process_multi_agent(query, context)
        elif selected_mode == AgentMode.PARALLEL_REASONING:
            result = await self._process_parallel(query, context, complexity_score)
        else:
            # Fallback to CoT if FSM is not available
            result = await self._process_with_cot(query, context, complexity_score)

        # Update performance metrics
        execution_time = time.time() - start_time
        self.performance_tracker.update_metrics(
            selected_mode, result.get('confidence', 0.0), execution_time
        )

        # Store state
        self.current_state.reasoning_path = result.get('reasoning_path')
        self.current_state.confidence = result.get('confidence', 0.0)
        self.current_state.execution_time = execution_time
        self.state_history.append(self.current_state)

        # Detect emergent behavior
        emergent_insights = self.emergent_behavior_detector.analyze(
            self.state_history, result
        )
        if emergent_insights:
            result['emergent_insights'] = emergent_insights

        return result

    async def _process_with_cot(self, query: str, context: Optional[Dict[str, Any]],
                              complexity: float) -> Dict[str, Any]:
        """Process query using Optimized Chain of Thought"""
        logger.info("Processing with Optimized Chain of Thought")

        # Execute CoT reasoning
        reasoning_path = await self.cot_system.reason(query, context)

        # Extract insights from reasoning path
        insights = self._extract_cot_insights(reasoning_path)

        return {
            'mode': 'chain_of_thought',
            'reasoning_path': reasoning_path,
            'answer': reasoning_path.final_answer,
            'confidence': reasoning_path.total_confidence,
            'insights': insights,
            'steps_count': len(reasoning_path.steps),
            'template_used': reasoning_path.template_used,
            'complexity_score': reasoning_path.complexity_score
        }

    async def _process_with_fsm(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process query using FSM ReAct agent"""
        logger.info("Processing with FSM ReAct")

        if not self.fsm_agent:
            # Fallback to CoT if FSM is not available
            return await self._process_with_cot(query, context, 0.5)

        # Execute FSM reasoning
        result = await self.fsm_agent.run(query)

        return {
            'mode': 'fsm_react',
            'answer': result.get('answer', ''),
            'confidence': result.get('confidence', 0.7),
            'steps': result.get('steps', []),
            'tools_used': result.get('tools_used', [])
        }

    async def _process_hybrid(self, query: str, context: Optional[Dict[str, Any]],
                            complexity: float) -> Dict[str, Any]:
        """Process query using hybrid approach"""
        logger.info("Processing with hybrid approach")

        # Start both approaches in parallel
        cot_task = asyncio.create_task(self._process_with_cot(query, context, complexity))

        if self.fsm_agent:
            fsm_task = asyncio.create_task(self._process_with_fsm(query, context))
            # Wait for both to complete
            cot_result, fsm_result = await asyncio.gather(cot_task, fsm_task)
        else:
            # Only CoT available
            cot_result = await cot_task
            fsm_result = {'answer': '', 'confidence': 0.0, 'steps': [], 'tools_used': []}

        # Synthesize results
        hybrid_result = self._synthesize_hybrid_results(cot_result, fsm_result)

        return hybrid_result

    async def _process_multi_agent(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process query using multi-agent collaboration"""
        logger.info("Processing with multi-agent collaboration")

        # Coordinate multiple agents
        result = await self.multi_agent_coordinator.coordinate(
            query, context, self.fsm_agent, self.cot_system
        )

        return result

    async def _process_parallel(self, query: str, context: Optional[Dict[str, Any]],
                              complexity: float) -> Dict[str, Any]:
        """Process query using parallel reasoning paths"""
        logger.info("Processing with parallel reasoning")

        # Execute multiple reasoning approaches in parallel
        tasks = [
            asyncio.create_task(self._process_with_cot(query, context, complexity)),
            asyncio.create_task(self._process_with_cot(query, context, complexity))  # Second CoT path
        ]

        if self.fsm_agent:
            tasks.append(asyncio.create_task(self._process_with_fsm(query, context)))

        results = await asyncio.gather(*tasks)

        # Select best result or synthesize
        best_result = max(results, key=lambda r: r.get('confidence', 0))

        return {
            'mode': 'parallel_reasoning',
            'best_result': best_result,
            'all_results': results,
            'confidence': best_result.get('confidence', 0)
        }

    def _extract_cot_insights(self, reasoning_path: ReasoningPath) -> Dict[str, Any]:
        """Extract insights from Chain of Thought reasoning path"""
        insights = {
            'reasoning_types_used': list(set(step.reasoning_type for step in reasoning_path.steps)),
            'confidence_progression': [step.confidence for step in reasoning_path.steps],
            'key_thoughts': [step.thought for step in reasoning_path.steps[-3:]],  # Last 3 steps
            'reflection_steps': [
                step for step in reasoning_path.steps
                if step.reasoning_type == ReasoningType.METACOGNITIVE
            ]
        }

        return insights

    def _synthesize_hybrid_results(self, cot_result: Dict[str, Any],
                                 fsm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from CoT and FSM approaches"""
        # Weight the results based on confidence
        cot_weight = cot_result.get('confidence', 0)
        fsm_weight = fsm_result.get('confidence', 0)

        total_weight = cot_weight + fsm_weight
        if total_weight == 0:
            total_weight = 1

        # Combine answers
        if cot_weight > fsm_weight:
            primary_answer = cot_result.get('answer', '')
            secondary_answer = fsm_result.get('answer', '')
        else:
            primary_answer = fsm_result.get('answer', '')
            secondary_answer = cot_result.get('answer', '')

        # Calculate combined confidence
        combined_confidence = (cot_weight + fsm_weight) / 2

        return {
            'mode': 'hybrid',
            'answer': primary_answer,
            'secondary_answer': secondary_answer,
            'confidence': combined_confidence,
            'cot_insights': cot_result.get('insights', {}),
            'fsm_steps': fsm_result.get('steps', []),
            'reasoning_path': cot_result.get('reasoning_path')
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        base_metrics = self.performance_tracker.get_metrics()
        cot_metrics = self.cot_system.get_performance_report()

        return {
            'agent_name': self.name,
            'total_queries': base_metrics.total_queries,
            'mode_usage': base_metrics.mode_usage,
            'average_confidence': base_metrics.average_confidence,
            'average_execution_time': base_metrics.average_execution_time,
            'cot_performance': cot_metrics,
            'current_state': {
                'mode': self.current_state.mode.name,
                'confidence': self.current_state.confidence,
                'last_query': self.current_state.current_query
            }
        }

    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get reasoning history"""
        history = []
        for state in self.state_history[-10:]:  # Last 10 states
            history.append({
                'timestamp': state.timestamp,
                'mode': state.mode.name,
                'query': state.current_query,
                'confidence': state.confidence,
                'execution_time': state.execution_time
            })
        return history

# =============================
# Enhanced Supporting Classes
# =============================

class AdaptiveModeSelector:
    """Enhanced mode selection with CoT integration"""

    def __init__(self):
        self.mode_weights = {
            AgentMode.CHAIN_OF_THOUGHT: 0.3,
            AgentMode.FSM_REACT: 0.3,
            AgentMode.HYBRID_ADAPTIVE: 0.2,
            AgentMode.MULTI_AGENT: 0.1,
            AgentMode.PARALLEL_REASONING: 0.1
        }

        self.complexity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def select_mode(self, query: str, complexity: float, features: Dict[str, float],
                   current_state: AgentState) -> AgentMode:
        """Select optimal mode based on query characteristics"""

        # Complexity-based selection
        if complexity > self.complexity_thresholds['high']:
            # High complexity: prefer CoT or hybrid
            if 'analytical' in features and features['analytical'] > 0.5:
                return AgentMode.CHAIN_OF_THOUGHT
            else:
                return AgentMode.HYBRID_ADAPTIVE
        elif complexity > self.complexity_thresholds['medium']:
            # Medium complexity: prefer hybrid or FSM
            return AgentMode.HYBRID_ADAPTIVE
        else:
            # Low complexity: prefer FSM for efficiency
            return AgentMode.FSM_REACT

class PerformanceTracker:
    """Enhanced performance tracking with CoT metrics"""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.mode_performance = defaultdict(list)

    def update_metrics(self, mode: AgentMode, confidence: float, execution_time: float):
        """Update performance metrics"""
        self.metrics.total_queries += 1
        self.metrics.mode_usage[mode] = self.metrics.mode_usage.get(mode, 0) + 1

        # Update running averages
        n = self.metrics.total_queries
        self.metrics.average_confidence = (
            (self.metrics.average_confidence * (n - 1) + confidence) / n
        )
        self.metrics.average_execution_time = (
            (self.metrics.average_execution_time * (n - 1) + execution_time) / n
        )

        # Track mode-specific performance
        self.mode_performance[mode].append({
            'confidence': confidence,
            'execution_time': execution_time,
            'timestamp': time.time()
        })

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics"""
        return self.metrics

class MultiAgentCoordinator:
    """Coordinate multiple agents for complex tasks"""

    async def coordinate(self, query: str, context: Optional[Dict[str, Any]],
                        fsm_agent, cot_system: OptimizedChainOfThought) -> Dict[str, Any]:
        """Coordinate multiple agents"""

        # Create specialized agents
        agents = {
            'researcher': self._create_researcher_agent(cot_system),
            'executor': self._create_executor_agent(fsm_agent),
            'synthesizer': self._create_synthesizer_agent()
        }

        # Execute in sequence
        research_result = await agents['researcher'].process_query(query, context)
        execution_result = await agents['executor'].process_query(query, context)
        synthesis_result = await agents['synthesizer'].process_query(
            query, context, research_result, execution_result
        )

        return {
            'mode': 'multi_agent',
            'research': research_result,
            'execution': execution_result,
            'synthesis': synthesis_result,
            'final_answer': synthesis_result.get('answer', ''),
            'confidence': synthesis_result.get('confidence', 0)
        }

    def _create_researcher_agent(self, cot_system: OptimizedChainOfThought):
        """Create research-focused agent"""
        return cot_system  # Use CoT for research

    def _create_executor_agent(self, fsm_agent):
        """Create execution-focused agent"""
        return fsm_agent  # Use FSM for execution

    def _create_synthesizer_agent(self):
        """Create synthesis-focused agent"""
        # Simple synthesis agent
        class SynthesisAgent:
            async def process_query(self, query: str, context: Optional[Dict[str, Any]],
                                  research_result: Dict[str, Any],
                                  execution_result: Dict[str, Any]) -> Dict[str, Any]:
                # Synthesize results
                combined_answer = f"Research: {research_result.get('answer', '')} | Execution: {execution_result.get('answer', '')}"
                combined_confidence = (research_result.get('confidence', 0) + execution_result.get('confidence', 0)) / 2

                return {
                    'answer': combined_answer,
                    'confidence': combined_confidence,
                    'synthesis_method': 'simple_combination'
                }

        return SynthesisAgent()

class EmergentBehaviorDetector:
    """Detect emergent behaviors and patterns"""

    def __init__(self):
        self.pattern_window = 10
        self.confidence_threshold = 0.8

    def analyze(self, state_history: List[AgentState], current_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze for emergent behaviors"""
        if len(state_history) < self.pattern_window:
            return None

        recent_states = state_history[-self.pattern_window:]

        # Detect patterns
        patterns = self._detect_patterns(recent_states)

        # Detect performance improvements
        improvements = self._detect_improvements(recent_states)

        # Detect mode preferences
        preferences = self._detect_preferences(recent_states)

        if patterns or improvements or preferences:
            return {
                'patterns': patterns,
                'improvements': improvements,
                'preferences': preferences,
                'timestamp': time.time()
            }

        return None

    def _detect_patterns(self, states: List[AgentState]) -> List[Dict[str, Any]]:
        """Detect recurring patterns"""
        patterns = []

        # Mode switching patterns
        mode_sequence = [state.mode for state in states]
        if len(set(mode_sequence)) > 1:
            patterns.append({
                'type': 'mode_switching',
                'sequence': [mode.name for mode in mode_sequence]
            })

        # Confidence patterns
        confidence_values = [state.confidence for state in states]
        if max(confidence_values) - min(confidence_values) > 0.3:
            patterns.append({
                'type': 'confidence_variation',
                'range': f"{min(confidence_values):.2f} - {max(confidence_values):.2f}"
            })

        return patterns

    def _detect_improvements(self, states: List[AgentState]) -> List[Dict[str, Any]]:
        """Detect performance improvements"""
        improvements = []

        # Check for confidence improvements
        early_confidence = np.mean([s.confidence for s in states[:5]])
        late_confidence = np.mean([s.confidence for s in states[-5:]])

        if late_confidence > early_confidence + 0.1:
            improvements.append({
                'type': 'confidence_improvement',
                'improvement': late_confidence - early_confidence
            })

        return improvements

    def _detect_preferences(self, states: List[AgentState]) -> Dict[str, Any]:
        """Detect mode preferences"""
        mode_counts = defaultdict(int)
        for state in states:
            mode_counts[state.mode.name] += 1

        total = len(states)
        preferences = {
            mode: count / total
            for mode, count in mode_counts.items()
        }

        return preferences

# =============================
# Example Usage
# =============================

async def demo_hybrid_architecture():
    """Demonstrate the enhanced hybrid architecture with CoT integration"""

    logger.info("=== Advanced Hybrid AI Agent Architecture Demo ===\n")

    # Create hybrid agent
    agent = AdvancedHybridAgent(
        "demo_agent",
        config={
            'fsm': {'max_steps': 10},
            'cot': {
                'max_paths': 3,
                'cache_size': 100,
                'cache_ttl': 12
            }
        }
    )

    # Test queries of varying complexity
    test_queries = [
        "What is the weather like today?",
        "Explain the concept of machine learning in simple terms.",
        "Compare and contrast supervised and unsupervised learning approaches.",
        "Analyze the potential impact of quantum computing on cryptography.",
        "Solve the equation: 2x^2 + 3x - 5 = 0"
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info("Query {}: {}", extra={"i": i, "query": query})
        logger.info("-" * 60)

        # Process query
        result = await agent.process_query(query)

        logger.info("Mode: {}", extra={"result_get__mode____unknown__": result.get('mode', 'unknown')})
        logger.info("Confidence: {}", extra={"result_get__confidence___0_": result.get('confidence', 0)})
        logger.info("Answer: {}...", extra={"result_get__answer____No_answer___": result.get('answer', 'No answer')[:50]})

        if 'reasoning_path' in result:
            path = result['reasoning_path']
            logger.info("CoT Steps: {}", extra={"len_path_steps_": len(path.steps)})
            logger.info("Template: {}", extra={"path_template_used": path.template_used})

        if 'emergent_insights' in result:
            logger.info("Emergent Insights: {}", extra={"result__emergent_insights_": result['emergent_insights']})

        logger.info("\n" + str("="*80 + "\n"))

    # Show performance report
    logger.info("=== Performance Report ===")
    report = agent.get_performance_report()
    for key, value in report.items():
        if isinstance(value, dict):
            logger.info("{}:", extra={"key": key})
            for k, v in value.items():
                logger.info("  {}: {}", extra={"k": k, "v": v})
        else:
            logger.info("{}: {}", extra={"key": key, "value": value})

    # Show reasoning history
    logger.info("\n=== Recent Reasoning History ===")
    history = agent.get_reasoning_history()
    for entry in history:
        logger.info("{}: {} - {}... (conf: {})", extra={"entry__timestamp_": entry['timestamp'], "entry__mode_": entry['mode'], "entry__query__": entry['query'], "entry__confidence_": entry['confidence']})

# Create a wrapper class for semantic search tool
class SemanticSearchTool(BaseTool):
    def __init__(self):
        super().__init__("semantic_search", "Perform semantic search on knowledge base")

    def execute(self, query: str, filename: str, top_k: int = 3) -> str:
        return semantic_search_tool(query, filename, top_k)

    def _run(self, query: str, filename: str, top_k: int = 3) -> str:
        return semantic_search_tool(query, filename, top_k)

# Create a wrapper class for python interpreter tool
class PythonInterpreter(BaseTool):
    def __init__(self):
        super().__init__("python_interpreter", "Execute Python code")

    def execute(self, code: str) -> str:
        return python_interpreter(code)

    def _run(self, code: str) -> str:
        return python_interpreter(code)

# Create a wrapper class for weather tool
class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__("weather", "Get weather information for a location")

    def execute(self, location: str, units: str = "metric") -> str:
        return get_weather(location, units)

    def _run(self, location: str, units: str = "metric") -> str:
        return get_weather(location, units)

if __name__ == "__main__":
    asyncio.run(demo_hybrid_architecture())