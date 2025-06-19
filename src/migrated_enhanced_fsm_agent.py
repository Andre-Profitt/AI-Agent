"""
Practical Integration Example: Migrating Existing FSMReActAgent to Enhanced FSM
This shows how to integrate the enhanced FSM features with your existing codebase
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Import existing components from your codebase
from src.advanced_agent_fsm import (
    FSMReActAgent, 
    FSMState, 
    EnhancedAgentState,
    ToolCall,
    correlation_context
)
from src.tools_enhanced import get_enhanced_tools
from src.errors.error_handler import ErrorHandler, ErrorCategory

# Import the new enhanced FSM components
from src.enhanced_fsm import (
    HierarchicalFSM,
    CompositeState,
    AtomicState,
    ProbabilisticTransition,
    StateDiscoveryEngine
)

logger = logging.getLogger(__name__)


class MigratedEnhancedFSMAgent(FSMReActAgent):
    """
    Enhanced version of FSMReActAgent with hierarchical states,
    probabilistic transitions, and dynamic state discovery.
    
    This class shows how to integrate new features while maintaining
    backward compatibility with existing code.
    """
    
    def __init__(
        self, 
        tools: List[Any],
        enable_hierarchical: bool = True,
        enable_probabilistic: bool = True,
        enable_discovery: bool = True,
        **kwargs
    ):
        # Initialize parent class
        super().__init__(tools, **kwargs)
        
        # New enhanced features
        self.enable_hierarchical = enable_hierarchical
        self.enable_probabilistic = enable_probabilistic
        self.enable_discovery = enable_discovery
        
        # Initialize enhanced components
        if self.enable_hierarchical:
            self.hfsm = self._build_hierarchical_fsm()
        
        if self.enable_discovery:
            self.state_discovery = StateDiscoveryEngine(similarity_threshold=0.85)
            
        # Track additional metrics
        self.state_metrics = {
            'transition_count': 0,
            'discovered_states': [],
            'probability_history': []
        }
    
    def _build_hierarchical_fsm(self) -> HierarchicalFSM:
        """Build hierarchical FSM that mirrors existing FSMState enum"""
        hfsm = HierarchicalFSM("EnhancedAgentFSM")
        
        # === PLANNING PHASE ===
        planning_phase = CompositeState("PLANNING_PHASE")
        
        # Planning sub-states
        planning = AtomicState("PLANNING")
        planning.action = self._do_planning
        planning_phase.add_child(planning)
        
        awaiting_plan = AtomicState("AWAITING_PLAN_RESPONSE")
        awaiting_plan.action = self._await_plan_response
        planning_phase.add_child(awaiting_plan)
        
        validating_plan = AtomicState("VALIDATING_PLAN")
        validating_plan.action = self._validate_plan
        planning_phase.add_child(validating_plan)
        
        # === EXECUTION PHASE ===
        execution_phase = CompositeState("EXECUTION_PHASE")
        
        tool_execution = AtomicState("TOOL_EXECUTION")
        tool_execution.action = self._execute_tools
        execution_phase.add_child(tool_execution)
        
        # === SYNTHESIS PHASE ===
        synthesis_phase = CompositeState("SYNTHESIS_PHASE")
        
        synthesizing = AtomicState("SYNTHESIZING")
        synthesizing.action = self._synthesize_results
        synthesis_phase.add_child(synthesizing)
        
        verifying = AtomicState("VERIFYING")
        verifying.action = self._verify_results
        synthesis_phase.add_child(verifying)
        
        # === FAILURE HANDLING ===
        failure_phase = CompositeState("FAILURE_PHASE")
        
        transient_failure = AtomicState("TRANSIENT_API_FAILURE")
        transient_failure.action = self._handle_transient_failure
        failure_phase.add_child(transient_failure)
        
        permanent_failure = AtomicState("PERMANENT_API_FAILURE")
        permanent_failure.action = self._handle_permanent_failure
        failure_phase.add_child(permanent_failure)
        
        # Add all phases to HFSM
        hfsm.add_state(planning_phase)
        hfsm.add_state(execution_phase)
        hfsm.add_state(synthesis_phase)
        hfsm.add_state(failure_phase)
        
        # Setup probabilistic transitions
        self._setup_probabilistic_transitions(hfsm)
        
        return hfsm
    
    def _setup_probabilistic_transitions(self, hfsm: HierarchicalFSM):
        """Configure probabilistic transitions between states"""
        
        # Planning -> Execution (high probability when plan is good)
        plan_to_exec = ProbabilisticTransition(
            "VALIDATING_PLAN", 
            "TOOL_EXECUTION",
            base_probability=0.85
        )
        plan_to_exec.add_context_modifier("plan_confidence<0.5", 0.5)
        plan_to_exec.add_context_modifier("retry_count>2", 0.6)
        
        # Execution -> Synthesis (depends on tool success)
        exec_to_synth = ProbabilisticTransition(
            "TOOL_EXECUTION",
            "SYNTHESIZING", 
            base_probability=0.9
        )
        exec_to_synth.add_context_modifier("tool_errors>0", 0.3)
        exec_to_synth.add_context_modifier("incomplete_results", 0.4)
        
        # Synthesis -> Verification (high probability)
        synth_to_verify = ProbabilisticTransition(
            "SYNTHESIZING",
            "VERIFYING",
            base_probability=0.95
        )
        
        # Failure recovery transitions
        transient_to_retry = ProbabilisticTransition(
            "TRANSIENT_API_FAILURE",
            "PLANNING",
            base_probability=0.7
        )
        transient_to_retry.add_context_modifier("retry_count>3", 0.2)
        
        # Register all transitions
        for transition in [plan_to_exec, exec_to_synth, synth_to_verify, transient_to_retry]:
            hfsm.add_transition(transition)
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced run method that uses hierarchical FSM if enabled,
        otherwise falls back to original implementation
        """
        if not self.enable_hierarchical:
            # Use original implementation
            return super().run(inputs)
        
        # Enhanced implementation with HFSM
        correlation_id = inputs.get("correlation_id", str(datetime.now().timestamp()))
        
        with correlation_context(correlation_id):
            logger.info("Starting enhanced FSM execution", extra={
                'correlation_id': correlation_id,
                'hierarchical': self.enable_hierarchical,
                'probabilistic': self.enable_probabilistic,
                'discovery': self.enable_discovery
            })
            
            # Initialize context
            context = self._initialize_context(inputs)
            
            # Start HFSM
            self.hfsm.start("PLANNING", context)
            
            # Main execution loop
            max_iterations = 50
            iteration = 0
            
            while iteration < max_iterations and not self._is_complete(context):
                iteration += 1
                
                # Check for state discovery
                if self.enable_discovery:
                    self._check_state_discovery(context)
                
                # Execute current state
                current_state = self.hfsm.current_state
                if current_state and hasattr(current_state, 'action') and current_state.action:
                    try:
                        current_state.action(context)
                    except Exception as e:
                        logger.error(f"Error in state {current_state.name}: {str(e)}")
                        context['last_error'] = str(e)
                        self._handle_state_error(current_state, context)
                
                # Determine next transition
                next_state = self._determine_next_state(context)
                if next_state:
                    success = self.hfsm.transition_to(next_state)
                    if success:
                        self.state_metrics['transition_count'] += 1
                        logger.info(f"Transitioned to {next_state}")
                    else:
                        logger.warning(f"Failed to transition to {next_state}")
                else:
                    # No valid transition, check if stuck
                    if self._is_stuck(context):
                        logger.warning("FSM appears stuck, attempting recovery")
                        self._attempt_recovery(context)
            
            # Prepare final result
            result = self._prepare_result(context)
            
            # Log execution summary
            logger.info("Enhanced FSM execution complete", extra={
                'iterations': iteration,
                'final_state': self.hfsm.current_state.name if self.hfsm.current_state else "UNKNOWN",
                'transition_count': self.state_metrics['transition_count'],
                'discovered_states': len(self.state_metrics['discovered_states'])
            })
            
            return result
    
    def _initialize_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize execution context"""
        return {
            'correlation_id': inputs.get('correlation_id', str(datetime.now().timestamp())),
            'query': inputs.get('input', ''),
            'tools': self.tools,
            'tool_results': [],
            'errors': [],
            'retry_count': 0,
            'start_time': datetime.now(),
            'plan': None,
            'final_answer': None,
            'confidence': 0.0,
            'state_history': []
        }
    
    def _check_state_discovery(self, context: Dict[str, Any]):
        """Check for new state patterns"""
        discovery_context = {
            'recent_tools': [t.tool_name for t in context.get('tool_results', [])[-5:]],
            'error_types': list(set(e.get('type', 'unknown') for e in context.get('errors', []))),
            'data_stats': {
                'result_count': len(context.get('tool_results', [])),
                'error_rate': len(context.get('errors', [])) / max(1, len(context.get('tool_results', [])))
            },
            'metrics': {
                'execution_time': (datetime.now() - context['start_time']).total_seconds(),
                'confidence': context.get('confidence', 0.0)
            }
        }
        
        discovered = self.state_discovery.analyze_context(discovery_context)
        if discovered and discovered not in self.state_metrics['discovered_states']:
            logger.info(f"Discovered new state pattern: {discovered}")
            self.state_metrics['discovered_states'].append(discovered)
            
            # Create and integrate new state
            new_state = AtomicState(discovered)
            new_state.action = lambda ctx: self._handle_discovered_state(ctx, discovered)
            
            # Add to appropriate parent (for now, add to execution phase)
            execution_phase = self.hfsm._find_state("EXECUTION_PHASE")
            if execution_phase:
                execution_phase.add_child(new_state)
    
    def _determine_next_state(self, context: Dict[str, Any]) -> Optional[str]:
        """Determine next state using probabilistic or deterministic logic"""
        current = self.hfsm.current_state
        if not current:
            return None
        
        if self.enable_probabilistic:
            # Use probabilistic transitions
            return self._get_probabilistic_next_state(current, context)
        else:
            # Use deterministic transitions (original logic)
            return self._get_deterministic_next_state(current, context)
    
    def _get_probabilistic_next_state(self, current: Any, context: Dict[str, Any]) -> Optional[str]:
        """Get next state based on probabilities"""
        # Get all possible transitions from current state
        possible_transitions = []
        
        # This is simplified - in practice, you'd query the HFSM for valid transitions
        if current.name == "PLANNING":
            possible_transitions = [
                ("AWAITING_PLAN_RESPONSE", 0.9),
                ("TRANSIENT_API_FAILURE", 0.1)
            ]
        elif current.name == "VALIDATING_PLAN":
            plan_confidence = context.get('confidence', 0.5)
            if plan_confidence > 0.7:
                possible_transitions = [
                    ("TOOL_EXECUTION", 0.85),
                    ("PLANNING", 0.15)  # Re-plan
                ]
            else:
                possible_transitions = [
                    ("TOOL_EXECUTION", 0.4),
                    ("PLANNING", 0.6)  # Re-plan
                ]
        
        if not possible_transitions:
            return None
        
        # Sample based on probabilities
        states, probs = zip(*possible_transitions)
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        
        next_state = np.random.choice(states, p=probs)
        
        # Log probability decision
        self.state_metrics['probability_history'].append({
            'from': current.name,
            'to': next_state,
            'probabilities': dict(possible_transitions),
            'timestamp': datetime.now()
        })
        
        return next_state
    
    def _get_deterministic_next_state(self, current: Any, context: Dict[str, Any]) -> Optional[str]:
        """Original deterministic state transition logic"""
        # Map current state to next state based on context
        if current.name == "PLANNING" and context.get('plan'):
            return "AWAITING_PLAN_RESPONSE"
        elif current.name == "AWAITING_PLAN_RESPONSE" and context.get('plan_response'):
            return "VALIDATING_PLAN"
        elif current.name == "VALIDATING_PLAN" and context.get('plan_valid'):
            return "TOOL_EXECUTION"
        elif current.name == "TOOL_EXECUTION" and context.get('tools_complete'):
            return "SYNTHESIZING"
        elif current.name == "SYNTHESIZING" and context.get('synthesis_complete'):
            return "VERIFYING"
        elif current.name == "VERIFYING" and context.get('verification_complete'):
            return "FINISHED"
        
        return None
    
    def _is_complete(self, context: Dict[str, Any]) -> bool:
        """Check if execution is complete"""
        return (
            context.get('final_answer') is not None or 
            context.get('fatal_error') is not None or
            self.hfsm.current_state.name == "FINISHED"
        )
    
    def _is_stuck(self, context: Dict[str, Any]) -> bool:
        """Detect if FSM is stuck in a state"""
        history = context.get('state_history', [])
        if len(history) >= 5:
            # Check if last 5 states are the same
            last_states = [h['state'] for h in history[-5:]]
            return len(set(last_states)) == 1
        return False
    
    def _attempt_recovery(self, context: Dict[str, Any]):
        """Attempt to recover from stuck state"""
        logger.warning("Attempting recovery from stuck state")
        
        # Force transition to planning to restart
        self.hfsm.transition_to("PLANNING")
        context['retry_count'] = context.get('retry_count', 0) + 1
        
        # Clear intermediate results to start fresh
        context['plan'] = None
        context['tool_results'] = []
    
    def _prepare_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final result with enhanced metrics"""
        base_result = {
            'output': context.get('final_answer', 'No answer generated'),
            'correlation_id': context['correlation_id'],
            'execution_time': (datetime.now() - context['start_time']).total_seconds(),
            'success': context.get('final_answer') is not None
        }
        
        # Add enhanced metrics
        if self.enable_hierarchical:
            base_result['state_transitions'] = self.hfsm.transition_log
            base_result['final_state'] = self.hfsm.current_state.name if self.hfsm.current_state else "UNKNOWN"
        
        if self.enable_discovery:
            base_result['discovered_states'] = self.state_metrics['discovered_states']
        
        if self.enable_probabilistic:
            base_result['probability_decisions'] = self.state_metrics['probability_history']
        
        return base_result
    
    # === State Action Methods ===
    
    def _do_planning(self, context: Dict[str, Any]):
        """Planning state action"""
        logger.info("Executing planning phase")
        # Your existing planning logic
        context['plan'] = self._create_plan(context['query'])
    
    def _await_plan_response(self, context: Dict[str, Any]):
        """Await plan response action"""
        logger.info("Awaiting plan response")
        # Your existing logic
        context['plan_response'] = True
    
    def _validate_plan(self, context: Dict[str, Any]):
        """Validate plan action"""
        logger.info("Validating plan")
        # Your existing validation logic
        context['plan_valid'] = True
        context['confidence'] = 0.8  # Example confidence
    
    def _execute_tools(self, context: Dict[str, Any]):
        """Execute tools action"""
        logger.info("Executing tools")
        # Your existing tool execution logic
        context['tools_complete'] = True
    
    def _synthesize_results(self, context: Dict[str, Any]):
        """Synthesize results action"""
        logger.info("Synthesizing results")
        # Your existing synthesis logic
        context['synthesis_complete'] = True
    
    def _verify_results(self, context: Dict[str, Any]):
        """Verify results action"""
        logger.info("Verifying results")
        # Your existing verification logic
        context['verification_complete'] = True
        context['final_answer'] = "Example answer"
    
    def _handle_transient_failure(self, context: Dict[str, Any]):
        """Handle transient failure"""
        logger.warning("Handling transient failure")
        context['retry_count'] = context.get('retry_count', 0) + 1
    
    def _handle_permanent_failure(self, context: Dict[str, Any]):
        """Handle permanent failure"""
        logger.error("Handling permanent failure")
        context['fatal_error'] = context.get('last_error', 'Unknown error')
    
    def _handle_discovered_state(self, context: Dict[str, Any], state_name: str):
        """Handle dynamically discovered state"""
        logger.info(f"Executing discovered state: {state_name}")
        # Implement logic for discovered states
        pass
    
    def _handle_state_error(self, state: Any, context: Dict[str, Any]):
        """Handle errors during state execution"""
        error_info = {
            'state': state.name,
            'error': context.get('last_error', 'Unknown error'),
            'timestamp': datetime.now(),
            'type': 'execution_error'
        }
        context.setdefault('errors', []).append(error_info)
        
        # Transition to appropriate failure state
        if context.get('retry_count', 0) < 3:
            self.hfsm.transition_to("TRANSIENT_API_FAILURE")
        else:
            self.hfsm.transition_to("PERMANENT_API_FAILURE")
    
    def visualize_current_state(self) -> str:
        """Get visual representation of current FSM state"""
        if self.enable_hierarchical:
            return self.hfsm.visualize()
        else:
            return f"Current State: {self.current_fsm_state}"
    
    def _create_plan(self, query: str) -> str:
        """Create a plan for the query (placeholder implementation)"""
        return f"Plan for: {query}"


# === USAGE EXAMPLE ===

def example_migration():
    """Example showing how to use the migrated agent"""
    
    # Get your existing tools
    tools = get_enhanced_tools()
    
    # Create enhanced agent with all features enabled
    agent = MigratedEnhancedFSMAgent(
        tools=tools,
        enable_hierarchical=True,
        enable_probabilistic=True,
        enable_discovery=True,
        model_preference="balanced"
    )
    
    # Run a query
    result = agent.run({
        "input": "What is the weather in Tokyo and how does it compare to New York?",
        "correlation_id": "test-123"
    })
    
    # Analyze results
    print(f"Answer: {result['output']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"State transitions: {len(result.get('state_transitions', []))}")
    print(f"Discovered states: {result.get('discovered_states', [])}")
    
    # Visualize final state
    print("\nFinal FSM State:")
    print(agent.visualize_current_state())
    
    # Analyze probability decisions
    if result.get('probability_decisions'):
        print("\nProbability-based decisions:")
        for decision in result['probability_decisions'][-5:]:  # Last 5 decisions
            print(f"  {decision['from']} -> {decision['to']} "
                  f"(probabilities: {decision['probabilities']})")


if __name__ == "__main__":
    # Run the example
    example_migration() 