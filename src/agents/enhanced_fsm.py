from agent import answer
from agent import path
from agent import query
from agent import response
from agent import tools
from app import health_status
from benchmarks.cot_performance import current
from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import start_time
from examples.parallel_execution_example import agents
from setup_environment import value
from tests.load_test import success

from src.agents.enhanced_fsm import active
from src.agents.enhanced_fsm import active_states
from src.agents.enhanced_fsm import child
from src.agents.enhanced_fsm import common_length
from src.agents.enhanced_fsm import episodic
from src.agents.enhanced_fsm import first_child
from src.agents.enhanced_fsm import found
from src.agents.enhanced_fsm import initial
from src.agents.enhanced_fsm import memory_count
from src.agents.enhanced_fsm import multi_agent_triggers
from src.agents.enhanced_fsm import parent
from src.agents.enhanced_fsm import recommended_tools
from src.agents.enhanced_fsm import required_agents
from src.agents.enhanced_fsm import seen
from src.agents.enhanced_fsm import selected_reasoning
from src.agents.enhanced_fsm import semantic
from src.agents.enhanced_fsm import should_use_multi_agent
from src.agents.enhanced_fsm import source
from src.agents.enhanced_fsm import source_path
from src.agents.enhanced_fsm import state
from src.agents.enhanced_fsm import state_names
from src.agents.enhanced_fsm import target
from src.agents.enhanced_fsm import target_path
from src.agents.enhanced_fsm import target_state
from src.agents.enhanced_fsm import tools_used
from src.agents.enhanced_fsm import transition
from src.agents.enhanced_fsm import unique_tools
from src.agents.migrated_enhanced_fsm_agent import max_iterations
from src.collaboration.realtime_collaboration import session_id
from src.core.monitoring import key
from src.core.optimized_chain_of_thought import reasoning_type
from src.database.models import action
from src.database.models import model_name
from src.database.models import priority
from src.database.models import vector_store
from src.gaia_components.enhanced_memory_system import memories
from src.infrastructure.workflow.workflow_engine import initial_state
from src.meta_cognition import confidence
from src.meta_cognition import query_lower
from src.tools_introspection import name
from src.unified_architecture.conflict_resolution import states
from src.utils.knowledge_utils import word
from src.workflow.workflow_automation import condition

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import FSMReActAgent

from src.gaia_components.enhanced_memory_system import MemoryType
# TODO: Fix undefined variables: action, active, active_states, agents, answer, child, common_length, condition, confidence, context, current, enable_adaptive_tools, enable_memory, enable_monitoring, enable_multi_agent, episodic, execution_time, first_child, found, from_state, health_status, i, initial, initial_state, key, kwargs, max_iterations, memories, memory_count, metrics, model_name, multi_agent_triggers, name, on_enter, on_exit, on_update, parent, parent_name, path, priority, quality_level, query, query_lower, reasoning_type, recommended_tools, required_agents, response, result, s, seen, selected_reasoning, semantic, session_id, should_use_multi_agent, source, source_path, start_time, state, state_name, state_names, state_type, states, success, t, target, target_path, target_state, temperature, to_state, tools, tools_used, transition, trigger, unique_tools, use_multi_agent, value, vector_store, word, action, active, active_states, agents, answer, child, common_length, condition, confidence, context, current, enable_adaptive_tools, enable_memory, enable_monitoring, enable_multi_agent, episodic, execution_time, first_child, found, from_state, health_status, i, initial, initial_state, key, kwargs, max_iterations, memories, memory_count, metrics, model_name, multi_agent_triggers, name, on_enter, on_exit, on_update, parent, parent_name, path, priority, quality_level, query, query_lower, reasoning_type, recommended_tools, required_agents, response, result, s, seen, selected_reasoning, self, semantic, session_id, should_use_multi_agent, source, source_path, start_time, state, state_name, state_names, state_type, states, success, t, target, target_path, target_state, temperature, to_state, tools, tools_used, transition, trigger, unique_tools, use_multi_agent, value, vector_store, word
from langchain_core.messages import HumanMessage

from src.tools.base_tool import tool


"""
from typing import List
from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine
from src.reasoning.reasoning_path import ReasoningType
from src.shared.types.di_types import BaseTool
from src.unified_architecture.shared_memory import MemoryType
# TODO: Fix undefined variables: action, active, active_states, agents, answer, child, common_length, condition, confidence, context, current, enable_adaptive_tools, enable_memory, enable_monitoring, enable_multi_agent, episodic, execution_time, first_child, found, from_state, health_status, i, initial, initial_state, key, kwargs, max_iterations, memories, memory_count, metrics, model_name, multi_agent_triggers, name, on_enter, on_exit, on_update, parent, parent_name, path, priority, quality_level, query, query_lower, reasoning_type, recommended_tools, required_agents, response, result, s, seen, selected_reasoning, semantic, session_id, should_use_multi_agent, source, source_path, start_time, state, state_name, state_names, state_type, states, success, t, target, target_path, target_state, temperature, to_state, tools, tools_used, transition, trigger, unique_tools, use_multi_agent, value, vector_store, word, action, active, active_states, agents, answer, child, common_length, condition, confidence, context, current, enable_adaptive_tools, enable_memory, enable_monitoring, enable_multi_agent, episodic, execution_time, first_child, found, from_state, health_status, i, initial, initial_state, key, kwargs, max_iterations, memories, memory_count, metrics, model_name, multi_agent_triggers, name, on_enter, on_exit, on_update, parent, parent_name, path, priority, quality_level, query, query_lower, reasoning_type, recommended_tools, required_agents, response, result, s, seen, selected_reasoning, self, semantic, session_id, should_use_multi_agent, source, source_path, start_time, state, state_name, state_names, state_type, states, success, t, target, target_path, target_state, temperature, to_state, tools, tools_used, transition, trigger, unique_tools, use_multi_agent, value, vector_store, word

from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from unittest.mock import call
Enhanced FSM Agent for Production GAIA System
Includes both EnhancedFSMAgent and HierarchicalFSM implementations
"""

from dataclasses import field
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Any
from typing import Callable

import logging

from datetime import datetime
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.tools.base_tool import BaseTool

from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryType
from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem
from src.gaia_components.multi_agent_orchestrator import MultiAgentOrchestrator
from src.utils.data_quality import DataQualityLevel
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from math import e
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import call
import logging
import time

from langchain.schema import HumanMessage

from src.tools.base_tool import tool

logger = logging.getLogger(__name__)

# ============================================
# HIERARCHICAL FSM COMPONENTS
# ============================================

class StateType(str, Enum):
    """Types of states in hierarchical FSM"""
    ATOMIC = "atomic"          # Basic state with no substates
    COMPOSITE = "composite"    # State containing substates
    PARALLEL = "parallel"      # State with parallel substates
    HISTORY = "history"        # State that remembers last substate

@dataclass
class StateTransition:
    """Represents a transition between states"""
    from_state: str
    to_state: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    priority: int = 0

    def can_transition(self, context: Dict[str, Any]) -> bool:
        """Check if transition condition is met"""
        if self.condition is None:
            return True
        return self.condition(context)

    def execute_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transition action"""
        if self.action is None:
            return context
        return self.action(context)

class HierarchicalState:
    """State in a hierarchical FSM"""

    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.ATOMIC,
        parent: Optional['HierarchicalState'] = None,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        on_update: Optional[Callable] = None
    ):
        self.name = name
        self.state_type = state_type
        self.parent = parent
        self.children: Dict[str, 'HierarchicalState'] = {}
        self.transitions: List[StateTransition] = []
        self.is_active = False

        # State callbacks
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.on_update = on_update

        # History for composite states
        self.last_active_child: Optional[str] = None

        # Add to parent if specified
        if parent:
            parent.add_child(self)

    def add_child(self, child: 'HierarchicalState'):
        """Add a child state"""
        if self.state_type == StateType.ATOMIC:
            raise ValueError(f"Cannot add child to atomic state {self.name}")
        child.parent = self
        self.children[child.name] = child

    def add_transition(self, transition: StateTransition):
        """Add a transition from this state"""
        self.transitions.append(transition)
        # Sort by priority
        self.transitions.sort(key=lambda t: t.priority, reverse=True)

    def get_path(self) -> List[str]:
        """Get full path from root to this state"""
        path = []
        current = self
        while current:
            path.insert(0, current.name)
            current = current.parent
        return path

    def enter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enter this state"""
        self.is_active = True
        if self.on_enter:
            context = self.on_enter(context)

        # Enter default child if composite
        if self.state_type in [StateType.COMPOSITE, StateType.PARALLEL]:
            if self.state_type == StateType.PARALLEL:
                # Enter all children for parallel states
                for child in self.children.values():
                    context = child.enter(context)
            elif self.children:
                # Enter first child or history for composite
                if self.state_type == StateType.HISTORY and self.last_active_child:
                    child = self.children.get(self.last_active_child)
                    if child:
                        context = child.enter(context)
                else:
                    # Enter first child
                    first_child = next(iter(self.children.values()))
                    context = first_child.enter(context)

        return context

    def exit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Exit this state"""
        # Exit all active children first
        for child in self.children.values():
            if child.is_active:
                context = child.exit(context)
                if self.state_type == StateType.HISTORY:
                    self.last_active_child = child.name

        if self.on_exit:
            context = self.on_exit(context)

        self.is_active = False
        return context

    def update(self, context: Dict[str, Any]) -> Tuple[Optional[StateTransition], Dict[str, Any]]:
        """Update state and check for transitions"""
        if not self.is_active:
            return None, context

        # Update callback
        if self.on_update:
            context = self.on_update(context)

        # Update active children
        if self.state_type == StateType.PARALLEL:
            # Update all active children
            for child in self.children.values():
                if child.is_active:
                    transition, context = child.update(context)
                    if transition:
                        # Handle child transition
                        context = self._handle_child_transition(child, transition, context)
        else:
            # Update single active child
            for child in self.children.values():
                if child.is_active:
                    transition, context = child.update(context)
                    if transition:
                        # Handle child transition
                        context = self._handle_child_transition(child, transition, context)
                    break

        # Check own transitions
        for transition in self.transitions:
            if transition.can_transition(context):
                return transition, context

        return None, context

    def _handle_child_transition(
        self,
        child: 'HierarchicalState',
        transition: StateTransition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle transition from child state"""
        # Check if transition target is within this state
        target_state = self._find_child_by_name(transition.to_state)

        if target_state:
            # Internal transition
            context = child.exit(context)
            context = transition.execute_action(context)
            context = target_state.enter(context)
        else:
            # External transition - will be handled by parent
            pass

        return context

    def _find_child_by_name(self, name: str) -> Optional['HierarchicalState']:
        """Find child state by name (recursive)"""
        if name in self.children:
            return self.children[name]

        for child in self.children.values():
            found = child._find_child_by_name(name)
            if found:
                return found

        return None

    def get_active_states(self) -> List['HierarchicalState']:
        """Get all active states in the hierarchy"""
        active = []
        if self.is_active:
            active.append(self)
            for child in self.children.values():
                active.extend(child.get_active_states())
        return active

class HierarchicalFSM:
    """
    Hierarchical Finite State Machine
    Supports composite states, parallel states, and history
    """

    def __init__(self, name: str = "HierarchicalFSM"):
        self.name = name
        self.root_state = HierarchicalState("ROOT", StateType.COMPOSITE)
        self.current_context: Dict[str, Any] = {}
        self.state_history: List[Tuple[datetime, List[str]]] = []
        self.transition_history: List[Tuple[datetime, StateTransition]] = []
        self.is_running = False

        logger.info(f"HierarchicalFSM '{name}' initialized")

    def add_state(
        self,
        state: HierarchicalState,
        parent_name: Optional[str] = None
    ) -> 'HierarchicalFSM':
        """Add a state to the FSM"""
        if parent_name:
            parent = self.find_state(parent_name)
            if not parent:
                raise ValueError(f"Parent state '{parent_name}' not found")
            parent.add_child(state)
        else:
            self.root_state.add_child(state)

        return self

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        condition: Optional[Callable] = None,
        action: Optional[Callable] = None,
        priority: int = 0
    ) -> 'HierarchicalFSM':
        """Add a transition between states"""
        source = self.find_state(from_state)
        if not source:
            raise ValueError(f"Source state '{from_state}' not found")

        # Verify target exists
        target = self.find_state(to_state)
        if not target:
            raise ValueError(f"Target state '{to_state}' not found")

        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            condition=condition,
            action=action,
            priority=priority
        )

        source.add_transition(transition)
        return self

    def find_state(self, name: str) -> Optional[HierarchicalState]:
        """Find state by name in the hierarchy"""
        return self.root_state._find_child_by_name(name)

    def start(self, initial_state: str, context: Optional[Dict[str, Any]] = None):
        """Start the FSM"""
        self.current_context = context or {}
        self.is_running = True

        # Enter root
        self.current_context = self.root_state.enter(self.current_context)

        # Enter initial state
        initial = self.find_state(initial_state)
        if not initial:
            raise ValueError(f"Initial state '{initial_state}' not found")

        self.current_context = initial.enter(self.current_context)
        self._record_state_change()

        logger.info(f"HierarchicalFSM '{self.name}' started in state '{initial_state}'")

    def update(self) -> bool:
        """
        Update the FSM, process transitions

        Returns:
            True if FSM is still running, False if stopped
        """
        if not self.is_running:
            return False

        # Update from root
        transition, self.current_context = self.root_state.update(self.current_context)

        # Handle transition if any
        if transition:
            self._execute_transition(transition)

        return self.is_running

    def stop(self):
        """Stop the FSM"""
        # Exit all active states
        self.current_context = self.root_state.exit(self.current_context)
        self.is_running = False
        logger.info(f"HierarchicalFSM '{self.name}' stopped")

    def _execute_transition(self, transition: StateTransition):
        """Execute a state transition"""
        source = self.find_state(transition.from_state)
        target = self.find_state(transition.to_state)

        if not source or not target:
            logger.error(f"Invalid transition: {transition.from_state} -> {transition.to_state}")
            return

        # Find common ancestor
        source_path = source.get_path()
        target_path = target.get_path()

        # Find divergence point
        common_length = 0
        for i in range(min(len(source_path), len(target_path))):
            if source_path[i] == target_path[i]:
                common_length = i + 1
            else:
                break

        # Exit states from source up to common ancestor
        current = source
        while current and current.name not in target_path[:common_length]:
            self.current_context = current.exit(self.current_context)
            current = current.parent

        # Execute transition action
        self.current_context = transition.execute_action(self.current_context)

        # Enter states from common ancestor down to target
        for i in range(common_length, len(target_path)):
            state = self.find_state(target_path[i])
            if state:
                self.current_context = state.enter(self.current_context)

        # Record transition
        self.transition_history.append((datetime.now(), transition))
        self._record_state_change()

        logger.info(f"Transition: {transition.from_state} -> {transition.to_state}")

    def _record_state_change(self):
        """Record current active states"""
        active_states = self.get_active_states()
        state_names = [s.name for s in active_states]
        self.state_history.append((datetime.now(), state_names))

    def get_active_states(self) -> List[HierarchicalState]:
        """Get all currently active states"""
        return self.root_state.get_active_states()

    def get_active_state_names(self) -> List[str]:
        """Get names of all active states"""
        return [s.name for s in self.get_active_states()]

    def is_in_state(self, state_name: str) -> bool:
        """Check if FSM is in a particular state"""
        state = self.find_state(state_name)
        return state is not None and state.is_active

    def get_context(self) -> Dict[str, Any]:
        """Get current context"""
        return self.current_context.copy()

    def set_context_value(self, key: str, value: Any):
        """Set a value in the context"""
        self.current_context[key] = value

    def get_state_metrics(self) -> Dict[str, Any]:
        """Get metrics about state usage"""
        metrics = {
            "total_transitions": len(self.transition_history),
            "unique_states_visited": len(set(
                state for _, states in self.state_history for state in states
            )),
            "current_active_states": len(self.get_active_states()),
            "state_visit_counts": defaultdict(int)
        }

        # Count state visits
        for _, states in self.state_history:
            for state in states:
                metrics["state_visit_counts"][state] += 1

        return dict(metrics)

# ============================================
# ENHANCED FSM AGENT
# ============================================

@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_execution_time: float = 0.0
    average_confidence: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    reasoning_type_usage: Dict[str, int] = field(default_factory=dict)

    def update(self, success: bool, execution_time: float, tools_used: List[str],
               confidence: float, reasoning_type: str):
        """Update metrics with query results"""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        self.total_execution_time += execution_time

        # Update running average of confidence
        self.average_confidence = (
            (self.average_confidence * (self.total_queries - 1) + confidence)
            / self.total_queries
        )

        # Track tool usage
        for tool in tools_used:
            self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1

        # Track reasoning type usage
        self.reasoning_type_usage[reasoning_type] = (
            self.reasoning_type_usage.get(reasoning_type, 0) + 1
        )

    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100

    def get_average_execution_time(self) -> float:
        """Get average execution time per query"""
        if self.total_queries == 0:
            return 0.0
        return self.total_execution_time / self.total_queries

class EnhancedFSMAgent:
    """
    Enhanced FSM Agent with production features:
    - Advanced reasoning integration
    - Memory system with persistence
    - Adaptive tool selection
    - Multi-agent coordination
    - Performance monitoring
    - Error recovery
    """

    def __init__(
        self,
        tools: List[BaseTool],
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_iterations: int = 20,
        quality_level: DataQualityLevel = DataQualityLevel.THOROUGH,
        reasoning_type: ReasoningType = ReasoningType.LAYERED,
        enable_memory: bool = True,
        enable_adaptive_tools: bool = True,
        enable_multi_agent: bool = False,
        enable_monitoring: bool = True,
        vector_store: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize Enhanced FSM Agent
        """
        # Initialize base FSM agent
        self.base_agent = FSMReActAgent(
            tools=tools,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            **kwargs
        )

        # Configuration
        self.quality_level = quality_level
        self.reasoning_type = reasoning_type
        self.enable_memory = enable_memory
        self.enable_adaptive_tools = enable_adaptive_tools
        self.enable_multi_agent = enable_multi_agent
        self.enable_monitoring = enable_monitoring

        # Initialize components
        self._initialize_components(tools, vector_store)

        # Metrics
        self.metrics = AgentMetrics() if enable_monitoring else None

        # Session management
        self.current_session_id = None
        self.session_history = []

        logger.info(
            f"EnhancedFSMAgent initialized with {len(tools)} tools, "
            f"model: {model_name}, quality: {quality_level.value}"
        )

    def _initialize_components(self, tools: List[BaseTool], vector_store: Optional[Any]):
        """Initialize enhanced components"""

        # Advanced reasoning engine
        self.reasoning_engine = AdvancedReasoningEngine(
            enable_meta_reasoning=True,
            confidence_threshold=0.7
        ) if self.reasoning_type != ReasoningType.SIMPLE else None

        # Memory system
        if self.enable_memory:
            self.memory_system = EnhancedMemorySystem(
                vector_store=vector_store,
                enable_emotional=True,
                memory_consolidation_threshold=100
            )
        else:
            self.memory_system = None

        # Adaptive tool system
        if self.enable_adaptive_tools:
            self.tool_system = AdaptiveToolSystem(
                tools=tools,
                enable_ml_selection=True,
                failure_threshold=3
            )
        else:
            self.tool_system = None

        # Multi-agent orchestrator
        if self.enable_multi_agent:
            self.orchestrator = MultiAgentOrchestrator(
                tools=tools,
                max_agents=5,
                enable_specialized_agents=True
            )
        else:
            self.orchestrator = None

        # Inject components into base agent if compatible
        if hasattr(self.base_agent, 'memory_system'):
            self.base_agent.memory_system = self.memory_system
        if hasattr(self.base_agent, 'reasoning_engine'):
            self.base_agent.reasoning_engine = self.reasoning_engine

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        use_multi_agent: Optional[bool] = None,
        reasoning_type: Optional[ReasoningType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the enhanced agent on a query
        """
        start_time = time.time()
        self.current_session_id = session_id or f"session_{int(time.time())}"

        # Determine if we should use multi-agent
        should_use_multi_agent = (
            use_multi_agent if use_multi_agent is not None
            else (self.enable_multi_agent and self._should_use_multi_agent(query))
        )

        # Select reasoning type
        selected_reasoning = reasoning_type or self.reasoning_type

        try:
            # Load relevant memories if enabled
            if self.enable_memory and self.memory_system:
                memories = await self._load_relevant_memories(query)
                context = context or {}
                context['memories'] = memories

            # Execute with multi-agent if needed
            if should_use_multi_agent and self.orchestrator:
                logger.info("Using multi-agent orchestration")
                result = await self._run_multi_agent(query, context)
            else:
                # Use adaptive tool selection if enabled
                if self.enable_adaptive_tools and self.tool_system:
                    recommended_tools = await self.tool_system.recommend_tools(query)
                    logger.info(f"Recommended tools: {[t.name for t in recommended_tools]}")

                # Run base agent with enhanced context
                result = await self.base_agent.arun(
                    {"messages": [HumanMessage(content=query)]},
                    context=context,
                    reasoning_type=selected_reasoning,
                    **kwargs
                )

            # Process result
            success = bool(result.get('final_answer'))
            answer = result.get('final_answer', 'Unable to generate answer')

            # Extract metadata
            execution_time = time.time() - start_time
            tools_used = self._extract_tools_used(result)
            confidence = result.get('confidence', 0.5)

            # Save to memory if enabled
            if self.enable_memory and self.memory_system and success:
                await self._save_to_memory(query, answer, result)

            # Update metrics if monitoring enabled
            if self.enable_monitoring and self.metrics:
                self.metrics.update(
                    success=success,
                    execution_time=execution_time,
                    tools_used=tools_used,
                    confidence=confidence,
                    reasoning_type=selected_reasoning.value
                )

            # Prepare response
            response = {
                'answer': answer,
                'success': success,
                'execution_time': execution_time,
                'confidence': confidence,
                'tools_used': tools_used,
                'reasoning_type': selected_reasoning.value,
                'session_id': self.current_session_id,
                'used_multi_agent': should_use_multi_agent,
                'metadata': result.get('metadata', {})
            }

            # Add reasoning path if available
            if 'reasoning_path' in result:
                response['reasoning_path'] = result['reasoning_path']

            # Add performance metrics if monitoring
            if self.enable_monitoring:
                response['metrics'] = {
                    'success_rate': self.metrics.get_success_rate(),
                    'avg_execution_time': self.metrics.get_average_execution_time(),
                    'avg_confidence': self.metrics.average_confidence
                }

            return response

        except Exception as e:
            logger.error(f"Error in EnhancedFSMAgent.run: {str(e)}", exc_info=True)

            # Update failure metrics
            if self.enable_monitoring and self.metrics:
                self.metrics.update(
                    success=False,
                    execution_time=time.time() - start_time,
                    tools_used=[],
                    confidence=0.0,
                    reasoning_type=selected_reasoning.value
                )

            return {
                'answer': f"Error processing query: {str(e)}",
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'session_id': self.current_session_id
            }

    def _should_use_multi_agent(self, query: str) -> bool:
        """Determine if query requires multi-agent coordination"""
        # Simple heuristic - can be enhanced with ML
        multi_agent_triggers = [
            'analyze', 'compare', 'research', 'investigate',
            'comprehensive', 'detailed', 'multiple', 'various',
            'step by step', 'complex', 'elaborate'
        ]

        query_lower = query.lower()
        return any(trigger in query_lower for trigger in multi_agent_triggers)

    async def _run_multi_agent(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run query through multi-agent orchestrator"""
        if not self.orchestrator:
            raise ValueError("Multi-agent orchestrator not initialized")

        # Determine required agents based on query
        required_agents = self._determine_required_agents(query)

        # Run orchestration
        result = await self.orchestrator.coordinate(
            query=query,
            context=context,
            required_agents=required_agents
        )

        return result

    def _determine_required_agents(self, query: str) -> List[str]:
        """Determine which specialized agents are needed"""
        # Simple heuristic - can be enhanced
        agents = ['researcher']  # Always include researcher

        query_lower = query.lower()

        if any(word in query_lower for word in ['analyze', 'evaluate', 'assess']):
            agents.append('analyst')

        if any(word in query_lower for word in ['create', 'generate', 'write']):
            agents.append('creator')

        if any(word in query_lower for word in ['verify', 'validate', 'check']):
            agents.append('validator')

        agents.append('synthesizer')  # Always include synthesizer

        return agents

    async def _load_relevant_memories(self, query: str) -> List[Dict[str, Any]]:
        """Load relevant memories for the query"""
        if not self.memory_system:
            return []

        try:
            # Search episodic memory
            episodic = await self.memory_system.search_memories(
                query=query,
                memory_type=MemoryType.EPISODIC,
                top_k=3
            )

            # Search semantic memory
            semantic = await self.memory_system.search_memories(
                query=query,
                memory_type=MemoryType.SEMANTIC,
                top_k=3
            )

            return episodic + semantic

        except Exception as e:
            logger.warning(f"Error loading memories: {e}")
            return []

    async def _save_to_memory(self, query: str, answer: str, result: Dict[str, Any]):
        """Save interaction to memory"""
        if not self.memory_system:
            return

        try:
            # Save to episodic memory
            await self.memory_system.add_memory(
                content=f"Query: {query}\nAnswer: {answer}",
                memory_type=MemoryType.EPISODIC,
                metadata={
                    'session_id': self.current_session_id,
                    'confidence': result.get('confidence', 0.5),
                    'tools_used': result.get('tools_used', []),
                    'timestamp': datetime.now().isoformat()
                }
            )

            # Extract and save key facts to semantic memory
            if result.get('confidence', 0) > 0.7:
                await self.memory_system.add_memory(
                    content=answer,
                    memory_type=MemoryType.SEMANTIC,
                    metadata={
                        'source_query': query,
                        'confidence': result.get('confidence'),
                        'timestamp': datetime.now().isoformat()
                    }
                )

        except Exception as e:
            logger.warning(f"Error saving to memory: {e}")

    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
        """Extract list of tools used from result"""
        tools_used = []

        # Check different possible locations for tool usage info
        if 'tool_calls' in result:
            for call in result['tool_calls']:
                if isinstance(call, dict) and 'tool_name' in call:
                    tools_used.append(call['tool_name'])

        if 'metadata' in result and 'tools_used' in result['metadata']:
            tools_used.extend(result['metadata']['tools_used'])

        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in tools_used:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)

        return unique_tools

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics"""
        if not self.metrics:
            return None

        return {
            'total_queries': self.metrics.total_queries,
            'success_rate': self.metrics.get_success_rate(),
            'average_execution_time': self.metrics.get_average_execution_time(),
            'average_confidence': self.metrics.average_confidence,
            'tool_usage': dict(self.metrics.tool_usage),
            'reasoning_type_usage': dict(self.metrics.reasoning_type_usage)
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        if self.metrics:
            self.metrics = AgentMetrics()
            logger.info("Performance metrics reset")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }

        # Check base agent
        health_status['components']['base_agent'] = {
            'status': 'healthy' if self.base_agent else 'not_initialized'
        }

        # Check memory system
        if self.enable_memory and self.memory_system:
            try:
                memory_count = len(self.memory_system.episodic_memory.memories)
                health_status['components']['memory_system'] = {
                    'status': 'healthy',
                    'memory_count': memory_count
                }
            except Exception as e:
                health_status['components']['memory_system'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['status'] = 'degraded'

        # Check tool system
        if self.enable_adaptive_tools and self.tool_system:
            health_status['components']['tool_system'] = {
                'status': 'healthy',
                'tools_available': len(self.tool_system.tools)
            }

        # Check orchestrator
        if self.enable_multi_agent and self.orchestrator:
            health_status['components']['orchestrator'] = {
                'status': 'healthy',
                'agents_available': len(self.orchestrator.agents)
            }

        # Add metrics if available
        if self.metrics:
            health_status['metrics'] = self.get_metrics()

        return health_status