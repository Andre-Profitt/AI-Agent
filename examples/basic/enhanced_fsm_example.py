#!/usr/bin/env python3
"""
Enhanced FSM Example
====================

A simple working example that demonstrates the Enhanced FSM features.
This can be run immediately to test the implementation.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main example function"""
    print("🚀 Enhanced FSM Example")
    print("=" * 50)
    
    try:
        # Import enhanced FSM components
        from src.enhanced_fsm import (
            HierarchicalFSM, 
            AtomicState, 
            CompositeState,
            ProbabilisticTransition,
            StateDiscoveryEngine
        )
        
        print("✅ Successfully imported Enhanced FSM components")
        
        # Step 1: Create a simple FSM
        print("\n📋 Step 1: Creating Simple FSM")
        hfsm = HierarchicalFSM("ExampleFSM")
        
        # Create atomic states
        planning = AtomicState("PLANNING")
        execution = AtomicState("EXECUTION")
        synthesis = AtomicState("SYNTHESIS")
        
        # Add states to FSM
        hfsm.add_state(planning)
        hfsm.add_state(execution)
        hfsm.add_state(synthesis)
        
        print(f"   Created {len(hfsm.states)} states: {list(hfsm.states.keys())}")
        
        # Step 2: Add probabilistic transitions
        print("\n🔄 Step 2: Adding Probabilistic Transitions")
        
        # Create transitions
        plan_to_exec = ProbabilisticTransition("PLANNING", "EXECUTION", 0.9)
        exec_to_synth = ProbabilisticTransition("EXECUTION", "SYNTHESIS", 0.8)
        
        # Add context modifiers
        plan_to_exec.add_context_modifier("confidence<0.5", 0.5)
        exec_to_synth.add_context_modifier("errors>0", 0.3)
        
        # Add transitions to FSM
        hfsm.add_transition(plan_to_exec)
        hfsm.add_transition(exec_to_synth)
        
        print(f"   Added {len(hfsm.transitions)} transitions")
        
        # Step 3: Test FSM execution
        print("\n▶️  Step 3: Testing FSM Execution")
        
        # Create context
        context = {
            "query": "What is 2+2?",
            "confidence": 0.8,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        # Start FSM
        hfsm.start("PLANNING", context)
        print(f"   Started FSM in state: {hfsm.current_state.name}")
        
        # Get available transitions
        available = hfsm.get_available_transitions(context)
        print(f"   Available transitions: {len(available)}")
        for transition in available:
            print(f"     -> {transition['to_state']} (p={transition['probability']:.3f})")
        
        # Execute transitions
        print("\n   Executing transitions...")
        
        # Transition to execution
        success = hfsm.transition_to("EXECUTION", context)
        print(f"     PLANNING -> EXECUTION: {'✅' if success else '❌'}")
        
        # Transition to synthesis
        success = hfsm.transition_to("SYNTHESIS", context)
        print(f"     EXECUTION -> SYNTHESIS: {'✅' if success else '❌'}")
        
        print(f"   Final state: {hfsm.current_state.name}")
        
        # Step 4: Test state discovery
        print("\n🔍 Step 4: Testing State Discovery")
        
        discovery = StateDiscoveryEngine(similarity_threshold=0.8, min_pattern_frequency=2)
        
        # Test contexts
        test_contexts = [
            {
                'recent_tools': ['calculator', 'search'],
                'error_types': ['timeout'],
                'data_stats': {'result_count': 3, 'error_rate': 0.1},
                'metrics': {'execution_time': 2.0, 'confidence': 0.9}
            },
            {
                'recent_tools': ['calculator', 'search'],
                'error_types': ['timeout'],
                'data_stats': {'result_count': 4, 'error_rate': 0.15},
                'metrics': {'execution_time': 2.2, 'confidence': 0.85}
            }
        ]
        
        for i, test_context in enumerate(test_contexts):
            print(f"   Analyzing context {i+1}...")
            pattern = discovery.analyze_context(test_context)
            if pattern:
                print(f"     ✅ Discovered pattern: {pattern.name}")
            else:
                print(f"     ⚠️  No new pattern discovered")
        
        # Step 5: Test metrics and visualization
        print("\n📊 Step 5: Testing Metrics and Visualization")
        
        # Get state metrics
        metrics = hfsm.get_state_metrics()
        print("   State Metrics:")
        for state_name, state_metrics in metrics.items():
            success_rate = state_metrics.success_count / max(1, state_metrics.exit_count)
            print(f"     {state_name}: {success_rate:.1%} success rate, {state_metrics.avg_time:.3f}s avg time")
        
        # Generate visualization
        print("\n   FSM Visualization:")
        visualization = hfsm.visualize()
        print(visualization)
        
        # Export comprehensive metrics
        export_data = hfsm.export_metrics()
        print(f"\n   Export Summary:")
        print(f"     FSM Name: {export_data['fsm_name']}")
        print(f"     Total States: {export_data['total_states']}")
        print(f"     Total Transitions: {export_data['total_transitions']}")
        print(f"     Transition Log Entries: {len(export_data['transition_log'])}")
        
        print("\n🎉 Enhanced FSM Example Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install numpy scikit-learn matplotlib networkx")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Detailed error information:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
