from agent import query
from agent import tools
from benchmarks.cot_performance import analysis
from benchmarks.cot_performance import filename

from src.agents.enhanced_fsm import active_states
from src.agents.migrated_enhanced_fsm_agent import error_handling
from src.agents.migrated_enhanced_fsm_agent import iteration
from src.agents.migrated_enhanced_fsm_agent import max_iterations
from src.agents.migrated_enhanced_fsm_agent import optimization
from src.agents.migrated_enhanced_fsm_agent import planning
from src.agents.migrated_enhanced_fsm_agent import processing
from src.agents.migrated_enhanced_fsm_agent import still_running
from src.agents.migrated_enhanced_fsm_agent import synthesis
from src.agents.migrated_enhanced_fsm_agent import validation
from src.api_server import execution
from src.database.models import tool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import FSMReActAgent

from src.agents.advanced_agent_fsm import MigratedEnhancedFSMAgent
# TODO: Fix undefined variables: Any, Dict, List, Optional, active_states, analysis, context, datetime, e, enable_discovery, enable_hierarchical, enable_metrics, enable_probabilistic, error_handling, execution, f, filename, fsm_name, iteration, logging, max_iterations, optimization, planning, processing, query, result, still_running, synthesis, time, tools, validation
from src.tools.base_tool import tool


"""
from typing import Dict
from src.agents.enhanced_fsm import HierarchicalFSM
from src.agents.enhanced_fsm import HierarchicalState
from src.agents.enhanced_fsm import StateTransition
from src.agents.enhanced_fsm import StateType
# TODO: Fix undefined variables: active_states, analysis, context, e, enable_discovery, enable_hierarchical, enable_metrics, enable_probabilistic, error_handling, execution, f, filename, fsm_name, iteration, max_iterations, optimization, planning, processing, query, result, self, still_running, synthesis, tool, tools, validation

Migrated Enhanced FSM Agent
===========================

This module provides the MigratedEnhancedFSMAgent class that integrates
the Enhanced FSM with the existing AI Agent system while maintaining
backward compatibility with the current FSMReActAgent.
"""

from typing import Optional
from typing import Any
from typing import List

import logging
import time

from datetime import datetime

from .enhanced_fsm import (
    HierarchicalFSM,
    HierarchicalState,
    StateTransition,
    StateType
)

logger = logging.getLogger(__name__)

class MigratedEnhancedFSMAgent:
    """
    Enhanced FSM Agent that integrates with the existing AI Agent system.

    This agent provides:
    - Hierarchical FSM with composite and atomic states
    - Probabilistic transitions with context-aware learning
    - Dynamic state discovery engine
    - Comprehensive metrics and monitoring
    - Backward compatibility with existing FSMReActAgent
    """

    def __init__(
        self,
        tools: List[Any] = None,
        enable_hierarchical: bool = True,
        enable_probabilistic: bool = True,
        enable_discovery: bool = True,
        enable_metrics: bool = True,
        fsm_name: str = "EnhancedFSMAgent"
    ):
        """
        Initialize the Migrated Enhanced FSM Agent.

        Args:
            tools: List of tools available to the agent
            enable_hierarchical: Enable hierarchical states
            enable_probabilistic: Enable probabilistic transitions
            enable_discovery: Enable state discovery engine
            enable_metrics: Enable comprehensive metrics
            fsm_name: Name for the FSM instance
        """
        self.tools = tools or []
        self.enable_hierarchical = enable_hierarchical
        self.enable_probabilistic = enable_probabilistic
        self.enable_discovery = enable_discovery
        self.enable_metrics = enable_metrics
        self.fsm_name = fsm_name

        # Initialize the Enhanced FSM
        self.fsm = HierarchicalFSM(fsm_name)

        # Build the hierarchical FSM
        self._build_hierarchical_fsm()

        # Set up probabilistic transitions
        if enable_probabilistic:
            self._setup_probabilistic_transitions()

        # Initialize state discovery engine (simplified)
        if enable_discovery:
            self.discovery_engine = None  # Simplified for now
        else:
            self.discovery_engine = None

        # Context and state tracking
        self.current_context: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []

        logger.info("MigratedEnhancedFSMAgent initialized with {} tools", extra={"len_self_tools_": len(self.tools)})

    def _build_hierarchical_fsm(self):
        """Build the hierarchical FSM structure"""

        # Create atomic states using HierarchicalState
        planning = HierarchicalState("PLANNING", StateType.ATOMIC)
        execution = HierarchicalState("EXECUTION", StateType.ATOMIC)
        synthesis = HierarchicalState("SYNTHESIS", StateType.ATOMIC)
        error_handling = HierarchicalState("ERROR_HANDLING", StateType.ATOMIC)

        # Set up state actions using callbacks
        planning.on_enter = self._planning_action
        execution.on_enter = self._execution_action
        synthesis.on_enter = self._synthesis_action
        error_handling.on_enter = self._error_handling_action

        # Add atomic states
        self.fsm.add_state(planning)
        self.fsm.add_state(execution)
        self.fsm.add_state(synthesis)
        self.fsm.add_state(error_handling)

        # Create composite states if hierarchical mode is enabled
        if self.enable_hierarchical:
            processing = HierarchicalState("PROCESSING", StateType.COMPOSITE)

            # Create substates
            analysis = HierarchicalState("ANALYSIS", StateType.ATOMIC)
            validation = HierarchicalState("VALIDATION", StateType.ATOMIC)
            optimization = HierarchicalState("OPTIMIZATION", StateType.ATOMIC)

            # Set up substate actions
            analysis.on_enter = self._analysis_action
            validation.on_enter = self._validation_action
            optimization.on_enter = self._optimization_action

            # Add substates to composite state
            processing.add_child(analysis)
            processing.add_child(validation)
            processing.add_child(optimization)

            # Add composite state
            self.fsm.add_state(processing)

        logger.info("Built hierarchical FSM with {} states", extra={"len_self_fsm_states_": len(self.fsm.root_state.children)})

    def _setup_probabilistic_transitions(self):
        """Set up probabilistic transitions with context modifiers"""

        # Basic workflow transitions using StateTransition
        plan_to_next = StateTransition("PLANNING", "PROCESSING" if self.enable_hierarchical else "EXECUTION", priority=9)
        next_to_exec = StateTransition("PROCESSING" if self.enable_hierarchical else "PLANNING", "EXECUTION", priority=8)
        exec_to_synth = StateTransition("EXECUTION", "SYNTHESIS", priority=9)

        # Error handling transitions
        plan_to_error = StateTransition("PLANNING", "ERROR_HANDLING", priority=1)
        next_to_error = StateTransition("PROCESSING" if self.enable_hierarchical else "PLANNING", "ERROR_HANDLING", priority=2)
        exec_to_error = StateTransition("EXECUTION", "ERROR_HANDLING", priority=1)
        error_to_plan = StateTransition("ERROR_HANDLING", "PLANNING", priority=7)

        # Add transitions to FSM
        self.fsm.add_transition("PLANNING", "PROCESSING" if self.enable_hierarchical else "EXECUTION", priority=9)
        self.fsm.add_transition("PROCESSING" if self.enable_hierarchical else "PLANNING", "EXECUTION", priority=8)
        self.fsm.add_transition("EXECUTION", "SYNTHESIS", priority=9)
        self.fsm.add_transition("PLANNING", "ERROR_HANDLING", priority=1)
        self.fsm.add_transition("PROCESSING" if self.enable_hierarchical else "PLANNING", "ERROR_HANDLING", priority=2)
        self.fsm.add_transition("EXECUTION", "ERROR_HANDLING", priority=1)
        self.fsm.add_transition("ERROR_HANDLING", "PLANNING", priority=7)

        logger.info("Set up probabilistic transitions")

    def _planning_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the PLANNING state"""
        logger.info("Executing PLANNING action")

        # Simulate planning process
        context['plan'] = {
            'steps': ['analyze', 'execute', 'synthesize'],
            'estimated_time': 5.0,
            'confidence': context.get('confidence', 0.8)
        }

        # Update context with planning results
        context['planning_complete'] = True
        context['planning_time'] = time.time()

        return context

    def _analysis_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the ANALYSIS substate"""
        logger.info("Executing ANALYSIS action")

        # Simulate analysis process
        context['analysis'] = {
            'tools_needed': [tool.name for tool in self.tools[:2]],
            'complexity': 'medium',
            'estimated_duration': 2.0
        }

        context['analysis_complete'] = True
        return context

    def _validation_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the VALIDATION substate"""
        logger.info("Executing VALIDATION action")

        # Simulate validation process
        context['validation_passed'] = context.get('confidence', 0.8) > 0.5
        context['validation_time'] = time.time()

        return context

    def _optimization_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the OPTIMIZATION substate"""
        logger.info("Executing OPTIMIZATION action")

        # Simulate optimization process
        context['optimization'] = {
            'improvements': ['reduced_complexity', 'better_tool_selection'],
            'performance_gain': 0.15
        }

        context['optimization_complete'] = True
        return context

    def _execution_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the EXECUTION state"""
        logger.info("Executing EXECUTION action")

        # Simulate execution process
        context['execution'] = {
            'tools_used': [tool.name for tool in self.tools],
            'results': ['result1', 'result2'],
            'execution_time': 3.0
        }

        context['execution_success'] = True
        context['execution_complete'] = True

        return context

    def _synthesis_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the SYNTHESIS state"""
        logger.info("Executing SYNTHESIS action")

        # Simulate synthesis process
        context['synthesis'] = {
            'final_answer': "Synthesized result based on execution",
            'confidence': context.get('confidence', 0.8),
            'sources': context.get('execution', {}).get('results', [])
        }

        context['synthesis_complete'] = True
        context['task_complete'] = True

        return context

    def _error_handling_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action for the ERROR_HANDLING state"""
        logger.info("Executing ERROR_HANDLING action")

        # Simulate error handling process
        context['error_handling'] = {
            'error_type': 'execution_failure',
            'recovery_strategy': 'retry_with_different_approach',
            'retry_count': context.get('retry_count', 0) + 1
        }

        context['error_handled'] = True

        return context

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the Enhanced FSM Agent with the given query.

        Args:
            query: The user query to process
            context: Optional context dictionary

        Returns:
            Dictionary containing the result and metadata
        """
        # Initialize context
        if context is None:
            context = {}

        # Set up initial context
        self.current_context = {
            'query': query,
            'start_time': datetime.now(),
            'confidence': 0.8,
            'retry_count': 0,
            'tools_available': len(self.tools),
            **context
        }

        # Start the FSM
        self.fsm.start("PLANNING", self.current_context)

        # Execute the FSM workflow
        try:
            # Main workflow loop
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations and not self.current_context.get('task_complete', False):
                iteration += 1

                # Update FSM (this will trigger state transitions)
                still_running = self.fsm.update()

                if not still_running:
                    logger.warning("FSM stopped running, ending workflow")
                    break

                # Record execution
                self.execution_history.append({
                    'iteration': iteration,
                    'active_states': self.fsm.get_active_state_names(),
                    'context': self.current_context.copy(),
                    'timestamp': datetime.now()
                })

                # Check if we've completed the task
                if self.current_context.get('task_complete', False):
                    break

            # Prepare result
            result = {
                'success': self.current_context.get('task_complete', False),
                'final_states': self.fsm.get_active_state_names(),
                'iterations': iteration,
                'result': self.current_context.get('synthesis', {}).get('final_answer', 'No result generated'),
                'confidence': self.current_context.get('confidence', 0.0),
                'execution_time': (datetime.now() - self.current_context['start_time']).total_seconds(),
                'context': self.current_context
            }

            # Add metrics if enabled
            if self.enable_metrics:
                result['metrics'] = self.fsm.get_state_metrics()

            return result

        except Exception as e:
            logger.error("Error during FSM execution: {}", extra={"e": e})

            # Handle error in FSM
            if not self.fsm.is_in_state("ERROR_HANDLING"):
                self.fsm.add_transition(self.fsm.get_active_state_names()[0], "ERROR_HANDLING", priority=10)
                self.fsm.update()

            return {
                'success': False,
                'error': str(e),
                'final_states': self.fsm.get_active_state_names(),
                'context': self.current_context
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about the agent's performance"""
        if not self.enable_metrics:
            return {'metrics_disabled': True}

        return self.fsm.get_state_metrics()

    def visualize_current_state(self) -> str:
        """Get a visualization of the current FSM state"""
        active_states = self.fsm.get_active_state_names()
        return f"Active states: {', '.join(active_states)}"

    def save_visualization(self, filename: str):
        """Save a graphical visualization of the FSM"""
        # Simplified visualization - just save state info
        with open(filename, 'w') as f:
            f.write(f"FSM: {self.fsm.name}\n")
            f.write(f"Active states: {', '.join(self.fsm.get_active_state_names())}\n")
            f.write(f"Total transitions: {len(self.fsm.transition_history)}\n")

    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered patterns"""
        if not self.enable_discovery or not self.discovery_engine:
            return {'discovery_disabled': True}

        # Simplified discovery statistics
        return {
            'patterns_discovered': 0,
            'discovery_enabled': self.enable_discovery,
            'engine_available': self.discovery_engine is not None
        }

    def reset(self):
        """Reset the agent's state and history"""
        self.current_context = {}
        self.execution_history = []

        # Reset FSM if it's been started
        if self.fsm.is_running:
            # Create a new FSM instance
            self.fsm = HierarchicalFSM(self.fsm_name)
            self._build_hierarchical_fsm()
            if self.enable_probabilistic:
                self._setup_probabilistic_transitions()

        logger.info("Agent state reset")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history"""
        return self.execution_history.copy()

    def get_current_context(self) -> Dict[str, Any]:
        """Get the current context"""
        return self.current_context.copy()