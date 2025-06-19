#!/usr/bin/env python3
"""
Enhanced FSM Demo
================

A simple working demo that shows the Enhanced FSM in action.
"""

import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

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
        
        print(f"    Executing state: {self.name}")
        
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
        print(f"🚀 FSM '{self.name}' started in state '{initial_state}'")
    
    def transition_to(self, target_state: str, context: Dict[str, Any]) -> bool:
        """Transition to a target state"""
        if target_state not in self.states:
            print(f"❌ Target state '{target_state}' not found")
            return False
        
        # Find transition
        transition = None
        for t in self.transitions:
            if t.from_state == self.current_state.name and t.to_state == target_state:
                transition = t
                break
        
        if not transition:
            print(f"❌ No transition from '{self.current_state.name}' to '{target_state}'")
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
            
            print(f"✅ Transitioned: {old_state} -> {target_state}")
            return True
        else:
            print(f"❌ Transition rejected: {self.current_state.name} -> {target_state}")
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
    print("🚀 Enhanced FSM Demo")
    print("=" * 50)
    
    try:
        # Step 1: Create FSM
        print("\n📋 Step 1: Creating FSM")
        fsm = SimpleFSM("DemoFSM")
        
        # Create states
        planning = State("PLANNING")
        execution = State("EXECUTION")
        synthesis = State("SYNTHESIS")
        
        fsm.add_state(planning)
        fsm.add_state(execution)
        fsm.add_state(synthesis)
        
        print(f"   Created {len(fsm.states)} states: {list(fsm.states.keys())}")
        
        # Step 2: Add transitions
        print("\n🔄 Step 2: Adding Transitions")
        
        plan_to_exec = Transition("PLANNING", "EXECUTION", 0.9)
        exec_to_synth = Transition("EXECUTION", "SYNTHESIS", 0.8)
        
        fsm.add_transition(plan_to_exec)
        fsm.add_transition(exec_to_synth)
        
        print(f"   Added {len(fsm.transitions)} transitions")
        
        # Step 3: Test execution
        print("\n▶️  Step 3: Testing Execution")
        
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
        print("\n   Testing transitions...")
        
        # Transition to execution
        success = fsm.transition_to("EXECUTION", context)
        if success:
            fsm.execute_current_state(context)
        
        # Transition to synthesis
        success = fsm.transition_to("SYNTHESIS", context)
        if success:
            fsm.execute_current_state(context)
        
        # Step 4: Show results
        print("\n📊 Step 4: Results")
        
        # Show visualization
        print("\nFSM Visualization:")
        print(fsm.visualize())
        
        # Show metrics
        metrics = fsm.get_metrics()
        print(f"\nMetrics Summary:")
        print(f"  FSM Name: {metrics['fsm_name']}")
        print(f"  Total States: {metrics['total_states']}")
        print(f"  Total Transitions: {metrics['total_transitions']}")
        print(f"  Current State: {metrics['current_state']}")
        print(f"  Transition Log Entries: {len(metrics['transition_log'])}")
        
        print("\nState Metrics:")
        for state_name, state_metrics in metrics['state_metrics'].items():
            print(f"  {state_name}: {state_metrics['entry_count']} entries, {state_metrics['avg_time']:.3f}s avg time")
        
        print("\n🎉 Enhanced FSM Demo Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 