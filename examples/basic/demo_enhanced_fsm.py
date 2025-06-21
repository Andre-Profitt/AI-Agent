# TODO: Fix undefined variables: actual_prob, context, current_marker, e, end_time, exec_to_synth, execution, from_state, fsm, initial_state, lines, log, metrics, name, old_state, plan_to_exec, planning, probability, self, start_time, state, state_metrics, state_name, success_rate, synthesis, t, target_state, to_state, traceback, transition
#!/usr/bin/env python3
"""
Enhanced FSM Demo
================

A simple working demo that shows the Enhanced FSM in action.
"""
from agent import lines
from examples.basic.demo_enhanced_fsm import actual_prob
from examples.basic.demo_enhanced_fsm import current_marker
from examples.basic.demo_enhanced_fsm import fsm
from examples.basic.demo_enhanced_fsm import old_state
from examples.basic.demo_enhanced_fsm import plan_to_exec
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import start_time
from examples.parallel_execution_example import end_time

from src.agents.enhanced_fsm import state
from src.agents.enhanced_fsm import target_state
from src.agents.enhanced_fsm import transition
from src.agents.migrated_enhanced_fsm_agent import exec_to_synth
from src.agents.migrated_enhanced_fsm_agent import planning
from src.agents.migrated_enhanced_fsm_agent import synthesis
from src.api_server import execution
from src.database_extended import success_rate
from src.infrastructure.workflow.workflow_engine import initial_state
from src.shared.types.di_types import log
from src.tools_introspection import name


from typing import Optional
from typing import Any
from typing import List

import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
# TODO: Fix undefined variables: Any, Dict, List, Optional, actual_prob, context, current_marker, datetime, e, end_time, exec_to_synth, execution, from_state, fsm, initial_state, lines, log, logging, metrics, name, old_state, plan_to_exec, planning, probability, random, start_time, state, state_metrics, state_name, success_rate, synthesis, sys, t, target_state, time, to_state, transition

logger = logging.getLogger(__name__)

class State:
    """Simple state implementation"""

    def __init__(self, name: str):
        self.name = name
        self.entry_count = 0
        self.exit_count = 0
        self.total_time = 0.0

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the state"""
        start_time = time.time()
        self.entry_count += 1

        logger.info("    Executing state: {}", extra={"self_name": self.name})

        # Simulate some work
        time.sleep(0.1)

        end_time = time.time()
        self.total_time += (end_time - start_time)
        self.exit_count += 1

        return context

class Transition:
    """Simple transition implementation"""

    def __init__(self, from_state: str, to_state: str, probability: float = 1.0):
        self.from_state = from_state
        self.to_state = to_state
        self.probability = probability
        self.usage_count = 0
        self.success_count = 0

    def should_transition(self, context: Dict[str, Any]) -> bool:
        """Decide whether to take this transition"""
        # Apply context modifiers
        actual_prob = self.probability

        if 'confidence' in context and context['confidence'] < 0.5:
            actual_prob *= 0.5

        if 'errors' in context and context['errors'] > 0:
            actual_prob *= 0.7

        # Record usage
        self.usage_count += 1

        # Decide
        if random.random() <= actual_prob:
            self.success_count += 1
            return True
        return False

class SimpleFSM:
    """Simple FSM implementation"""

    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: List[Transition] = []
        self.current_state: Optional[State] = None
        self.context: Dict[str, Any] = {}
        self.transition_log: List[Dict[str, Any]] = []

    def add_state(self, state: State):
        """Add a state"""
        self.states[state.name] = state

    def add_transition(self, transition: Transition):
        """Add a transition"""
        self.transitions.append(transition)

    def start(self, initial_state: str, context: Dict[str, Any]):
        """Start the FSM"""
        if initial_state not in self.states:
            raise ValueError(f"State '{initial_state}' not found")

        self.current_state = self.states[initial_state]
        self.context = context.copy()
        logger.info("🚀 FSM '{}' started in state '{}'", extra={"self_name": self.name, "initial_state": initial_state})

    def transition_to(self, target_state: str, context: Dict[str, Any]) -> bool:
        """Transition to a target state"""
        if target_state not in self.states:
            logger.info("❌ Target state '{}' not found", extra={"target_state": target_state})
            return False

        # Find transition
        transition = None
        for t in self.transitions:
            if t.from_state == self.current_state.name and t.to_state == target_state:
                transition = t
                break

        if not transition:
            logger.info("❌ No transition from '{}' to '{}'", extra={"self_current_state_name": self.current_state.name, "target_state": target_state})
            return False

        # Check if we should transition
        if transition.should_transition(context):
            old_state = self.current_state.name
            self.current_state = self.states[target_state]
            self.context.update(context)

            # Log transition
            self.transition_log.append({
                'timestamp': datetime.now(),
                'from_state': old_state,
                'to_state': target_state,
                'probability': transition.probability
            })

            logger.info("✅ Transitioned: {} -> {}", extra={"old_state": old_state, "target_state": target_state})
            return True
        else:
            logger.info("❌ Transition rejected: {} -> {}", extra={"self_current_state_name": self.current_state.name, "target_state": target_state})
            return False

    def execute_current_state(self, context: Dict[str, Any]):
        """Execute the current state"""
        if not self.current_state:
            raise RuntimeError("No current state")

        self.context.update(context)
        return self.current_state.execute(self.context)

    def get_metrics(self) -> Dict[str, Any]:
        """Get FSM metrics"""
        return {
            'fsm_name': self.name,
            'total_states': len(self.states),
            'total_transitions': len(self.transitions),
            'current_state': self.current_state.name if self.current_state else None,
            'state_metrics': {
                name: {
                    'entry_count': state.entry_count,
                    'exit_count': state.exit_count,
                    'total_time': state.total_time,
                    'avg_time': state.total_time / max(1, state.exit_count)
                }
                for name, state in self.states.items()
            },
            'transition_log': [
                {
                    'timestamp': log['timestamp'].isoformat(),
                    'from_state': log['from_state'],
                    'to_state': log['to_state'],
                    'probability': log['probability']
                }
                for log in self.transition_log
            ]
        }

    def visualize(self) -> str:
        """Generate a simple visualization"""
        lines = [f"FSM: {self.name}", "=" * (len(self.name) + 5)]

        # States
        lines.append("\nStates:")
        for name, state in self.states.items():
            current_marker = " (CURRENT)" if self.current_state and self.current_state.name == name else ""
            lines.append(f"  {name}{current_marker}")

        # Transitions
        lines.append("\nTransitions:")
        for transition in self.transitions:
            success_rate = transition.success_count / max(1, transition.usage_count)
            lines.append(f"  {transition.from_state} -> {transition.to_state} (p={transition.probability:.2f}, success={success_rate:.1%})")

        return "\n".join(lines)

def main():
    """Main demo function"""
    logger.info("🚀 Enhanced FSM Demo")
    print("=" * 50)

    try:
        # Step 1: Create FSM
        logger.info("\n📋 Step 1: Creating FSM")
        fsm = SimpleFSM("DemoFSM")

        # Create states
        planning = State("PLANNING")
        execution = State("EXECUTION")
        synthesis = State("SYNTHESIS")

        fsm.add_state(planning)
        fsm.add_state(execution)
        fsm.add_state(synthesis)

        logger.info("   Created {} states: {}", extra={"len_fsm_states_": len(fsm.states), "list_fsm_states_keys___": list(fsm.states.keys())})

        # Step 2: Add transitions
        logger.info("\n🔄 Step 2: Adding Transitions")

        plan_to_exec = Transition("PLANNING", "EXECUTION", 0.9)
        exec_to_synth = Transition("EXECUTION", "SYNTHESIS", 0.8)

        fsm.add_transition(plan_to_exec)
        fsm.add_transition(exec_to_synth)

        logger.info("   Added {} transitions", extra={"len_fsm_transitions_": len(fsm.transitions)})

        # Step 3: Test execution
        logger.info("\n▶️  Step 3: Testing Execution")

        # Create context
        context = {
            "query": "What is 2+2?",
            "confidence": 0.8,
            "errors": 0
        }

        # Start FSM
        fsm.start("PLANNING", context)

        # Execute current state
        fsm.execute_current_state(context)

        # Try transitions
        logger.info("\n   Testing transitions...")

        # Transition to execution
        success = fsm.transition_to("EXECUTION", context)
        if success:
            fsm.execute_current_state(context)

        # Transition to synthesis
        success = fsm.transition_to("SYNTHESIS", context)
        if success:
            fsm.execute_current_state(context)

        # Step 4: Show results
        logger.info("\n📊 Step 4: Results")

        # Show visualization
        logger.info("\nFSM Visualization:")
        print(fsm.visualize())

        # Show metrics
        metrics = fsm.get_metrics()
        logger.info("\nMetrics Summary:")
        logger.info("  FSM Name: {}", extra={"metrics__fsm_name_": metrics['fsm_name']})
        logger.info("  Total States: {}", extra={"metrics__total_states_": metrics['total_states']})
        logger.info("  Total Transitions: {}", extra={"metrics__total_transitions_": metrics['total_transitions']})
        logger.info("  Current State: {}", extra={"metrics__current_state_": metrics['current_state']})
        logger.info("  Transition Log Entries: {}", extra={"len_metrics__transition_log__": len(metrics['transition_log'])})

        logger.info("\nState Metrics:")
        for state_name, state_metrics in metrics['state_metrics'].items():
            logger.info("  {}: {} entries, {}s avg time", extra={"state_name": state_name, "state_metrics__entry_count_": state_metrics['entry_count'], "state_metrics__avg_time_": state_metrics['avg_time']})

        logger.info("\n🎉 Enhanced FSM Demo Completed Successfully!")
        print("=" * 50)

        return True

    except Exception as e:
        logger.info("❌ Error: {}", extra={"e": e})
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
