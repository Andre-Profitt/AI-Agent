"""
Migrated Enhanced FSM Agent
===========================

This module provides the MigratedEnhancedFSMAgent class that integrates
the Enhanced FSM with the existing AI Agent system while maintaining
backward compatibility with the current FSMReActAgent.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .enhanced_fsm import (
    HierarchicalFSM, 
    AtomicState, 
    CompositeState,
    ProbabilisticTransition,
    StateDiscoveryEngine
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
        
        # Initialize state discovery engine
        if enable_discovery:
            self.discovery_engine = StateDiscoveryEngine()
        else:
            self.discovery_engine = None
        
        # Context and state tracking
        self.current_context: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"MigratedEnhancedFSMAgent initialized with {len(self.tools)} tools")
    
    def _build_hierarchical_fsm(self):
        """Build the hierarchical FSM structure"""
        
        # Create atomic states
        planning = AtomicState("PLANNING")
        execution = AtomicState("EXECUTION")
        synthesis = AtomicState("SYNTHESIS")
        error_handling = AtomicState("ERROR_HANDLING")
        
        # Set up state actions
        planning.action = self._planning_action
        execution.action = self._execution_action
        synthesis.action = self._synthesis_action
        error_handling.action = self._error_handling_action
        
        # Add atomic states
        self.fsm.add_state(planning)
        self.fsm.add_state(execution)
        self.fsm.add_state(synthesis)
        self.fsm.add_state(error_handling)
        
        # Create composite states if hierarchical mode is enabled
        if self.enable_hierarchical:
            processing = CompositeState("PROCESSING")
            
            # Create substates
            analysis = AtomicState("ANALYSIS")
            validation = AtomicState("VALIDATION")
            optimization = AtomicState("OPTIMIZATION")
            
            # Set up substate actions
            analysis.action = self._analysis_action
            validation.action = self._validation_action
            optimization.action = self._optimization_action
            
            # Add substates to composite state
            processing.add_substate(analysis)
            processing.add_substate(validation)
            processing.add_substate(optimization)
            
            # Add composite state
            self.fsm.add_state(processing)
        
        logger.info(f"Built hierarchical FSM with {len(self.fsm.states)} states")
    
    def _setup_probabilistic_transitions(self):
        """Set up probabilistic transitions with context modifiers"""
        
        # Basic workflow transitions
        plan_to_next = ProbabilisticTransition("PLANNING", "PROCESSING" if self.enable_hierarchical else "EXECUTION", 0.9)
        next_to_exec = ProbabilisticTransition("PROCESSING" if self.enable_hierarchical else "PLANNING", "EXECUTION", 0.8)
        exec_to_synth = ProbabilisticTransition("EXECUTION", "SYNTHESIS", 0.9)
        
        # Error handling transitions
        plan_to_error = ProbabilisticTransition("PLANNING", "ERROR_HANDLING", 0.1)
        next_to_error = ProbabilisticTransition("PROCESSING" if self.enable_hierarchical else "PLANNING", "ERROR_HANDLING", 0.2)
        exec_to_error = ProbabilisticTransition("EXECUTION", "ERROR_HANDLING", 0.1)
        error_to_plan = ProbabilisticTransition("ERROR_HANDLING", "PLANNING", 0.7)
        
        # Add context modifiers
        plan_to_next.add_context_modifier("confidence<0.5", 0.3)
        plan_to_error.add_context_modifier("confidence<0.5", 0.8)
        
        next_to_exec.add_context_modifier("validation_passed==True", 0.95)
        next_to_error.add_context_modifier("validation_passed==False", 0.9)
        
        exec_to_synth.add_context_modifier("execution_success==True", 0.95)
        exec_to_error.add_context_modifier("execution_success==False", 0.8)
        
        # Add transitions to FSM
        self.fsm.add_transition(plan_to_next)
        self.fsm.add_transition(next_to_exec)
        self.fsm.add_transition(exec_to_synth)
        self.fsm.add_transition(plan_to_error)
        self.fsm.add_transition(next_to_error)
        self.fsm.add_transition(exec_to_error)
        self.fsm.add_transition(error_to_plan)
        
        logger.info(f"Set up {len(self.fsm.transitions)} probabilistic transitions")
    
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
            # Execute initial state
            self.current_context = self.fsm.execute_current_state(self.current_context)
            
            # Main workflow loop
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations and not self.current_context.get('task_complete', False):
                iteration += 1
                
                # Get available transitions
                available_transitions = self.fsm.get_available_transitions(self.current_context)
                
                if not available_transitions:
                    logger.warning("No available transitions, ending workflow")
                    break
                
                # Select best transition (highest probability)
                best_transition = max(available_transitions, key=lambda t: t['probability'])
                
                # Attempt transition
                success = self.fsm.transition_to(best_transition['to_state'], self.current_context)
                
                if success:
                    # Execute the new state
                    self.current_context = self.fsm.execute_current_state(self.current_context)
                    
                    # Record execution
                    self.execution_history.append({
                        'iteration': iteration,
                        'state': self.fsm.current_state.name,
                        'context': self.current_context.copy(),
                        'timestamp': datetime.now()
                    })
                    
                    # Analyze context for state discovery
                    if self.enable_discovery and self.discovery_engine:
                        pattern = self.discovery_engine.analyze_context(self.current_context)
                        if pattern:
                            logger.info(f"Discovered new pattern: {pattern.name}")
                
                else:
                    logger.warning(f"Transition to {best_transition['to_state']} failed")
                    break
            
            # Prepare result
            result = {
                'success': self.current_context.get('task_complete', False),
                'final_state': self.fsm.current_state.name if self.fsm.current_state else None,
                'iterations': iteration,
                'result': self.current_context.get('synthesis', {}).get('final_answer', 'No result generated'),
                'confidence': self.current_context.get('confidence', 0.0),
                'execution_time': (datetime.now() - self.current_context['start_time']).total_seconds(),
                'context': self.current_context
            }
            
            # Add metrics if enabled
            if self.enable_metrics:
                result['metrics'] = self.fsm.export_metrics()
            
            return result
            
        except Exception as e:
            logger.error(f"Error during FSM execution: {e}")
            
            # Handle error in FSM
            if self.fsm.current_state and self.fsm.current_state.name != "ERROR_HANDLING":
                self.fsm.transition_to("ERROR_HANDLING", self.current_context)
                self.current_context = self.fsm.execute_current_state(self.current_context)
            
            return {
                'success': False,
                'error': str(e),
                'final_state': self.fsm.current_state.name if self.fsm.current_state else None,
                'context': self.current_context
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about the agent's performance"""
        if not self.enable_metrics:
            return {'metrics_disabled': True}
        
        return self.fsm.export_metrics()
    
    def visualize_current_state(self) -> str:
        """Get a visualization of the current FSM state"""
        return self.fsm.visualize()
    
    def save_visualization(self, filename: str):
        """Save a graphical visualization of the FSM"""
        self.fsm.save_visualization(filename)
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered patterns"""
        if not self.enable_discovery or not self.discovery_engine:
            return {'discovery_disabled': True}
        
        return self.discovery_engine.get_pattern_statistics()
    
    def reset(self):
        """Reset the agent's state and history"""
        self.current_context = {}
        self.execution_history = []
        
        # Reset FSM if it's been started
        if self.fsm.started:
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