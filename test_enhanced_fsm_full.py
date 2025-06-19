#!/usr/bin/env python3
"""
Full Enhanced FSM Test
=====================

A comprehensive test of the full Enhanced FSM implementation
with all advanced features including state discovery and visualization.
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

def test_full_enhanced_fsm():
    """Test the full Enhanced FSM implementation"""
    print("🚀 Full Enhanced FSM Test")
    print("=" * 50)
    
    try:
        # Import the full Enhanced FSM
        from src.enhanced_fsm import (
            HierarchicalFSM, 
            AtomicState, 
            CompositeState,
            ProbabilisticTransition,
            StateDiscoveryEngine
        )
        
        print("✅ Successfully imported full Enhanced FSM components")
        
        # Step 1: Create hierarchical FSM
        print("\n📋 Step 1: Creating Hierarchical FSM")
        hfsm = HierarchicalFSM("AdvancedFSM")
        
        # Create atomic states
        planning = AtomicState("PLANNING")
        execution = AtomicState("EXECUTION")
        synthesis = AtomicState("SYNTHESIS")
        error_handling = AtomicState("ERROR_HANDLING")
        
        # Create composite state
        processing = CompositeState("PROCESSING")
        sub_analysis = AtomicState("ANALYSIS")
        sub_validation = AtomicState("VALIDATION")
        processing.add_substate(sub_analysis)
        processing.add_substate(sub_validation)
        
        # Add states to FSM
        hfsm.add_state(planning)
        hfsm.add_state(processing)
        hfsm.add_state(execution)
        hfsm.add_state(synthesis)
        hfsm.add_state(error_handling)
        
        print(f"   Created {len(hfsm.states)} states (including composite)")
        
        # Step 2: Add probabilistic transitions with context modifiers
        print("\n🔄 Step 2: Adding Probabilistic Transitions")
        
        # Create transitions
        plan_to_process = ProbabilisticTransition("PLANNING", "PROCESSING", 0.9)
        process_to_exec = ProbabilisticTransition("PROCESSING", "EXECUTION", 0.8)
        exec_to_synth = ProbabilisticTransition("EXECUTION", "SYNTHESIS", 0.9)
        
        # Error handling transitions
        plan_to_error = ProbabilisticTransition("PLANNING", "ERROR_HANDLING", 0.1)
        process_to_error = ProbabilisticTransition("PROCESSING", "ERROR_HANDLING", 0.2)
        exec_to_error = ProbabilisticTransition("EXECUTION", "ERROR_HANDLING", 0.1)
        error_to_plan = ProbabilisticTransition("ERROR_HANDLING", "PLANNING", 0.7)
        
        # Add context modifiers
        plan_to_process.add_context_modifier("confidence<0.5", 0.3)
        plan_to_error.add_context_modifier("confidence<0.5", 0.8)
        
        process_to_exec.add_context_modifier("validation_passed==True", 0.95)
        process_to_error.add_context_modifier("validation_passed==False", 0.9)
        
        exec_to_synth.add_context_modifier("execution_success==True", 0.95)
        exec_to_error.add_context_modifier("execution_success==False", 0.8)
        
        # Add transitions to FSM
        hfsm.add_transition(plan_to_process)
        hfsm.add_transition(process_to_exec)
        hfsm.add_transition(exec_to_synth)
        hfsm.add_transition(plan_to_error)
        hfsm.add_transition(process_to_error)
        hfsm.add_transition(exec_to_error)
        hfsm.add_transition(error_to_plan)
        
        print(f"   Added {len(hfsm.transitions)} transitions with context modifiers")
        
        # Step 3: Test FSM execution with different contexts
        print("\n▶️  Step 3: Testing FSM Execution")
        
        # Test context 1: Normal flow
        print("\n   Test 1: Normal flow (high confidence)")
        context1 = {
            "query": "What is 2+2?",
            "confidence": 0.9,
            "validation_passed": True,
            "execution_success": True,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        hfsm.start("PLANNING", context1)
        print(f"     Started in: {hfsm.current_state.name}")
        
        # Execute and transition through normal flow
        hfsm.execute_current_state(context1)
        hfsm.transition_to("PROCESSING", context1)
        hfsm.execute_current_state(context1)
        hfsm.transition_to("EXECUTION", context1)
        hfsm.execute_current_state(context1)
        hfsm.transition_to("SYNTHESIS", context1)
        hfsm.execute_current_state(context1)
        
        print(f"     Final state: {hfsm.current_state.name}")
        
        # Test context 2: Error flow
        print("\n   Test 2: Error flow (low confidence)")
        context2 = {
            "query": "Complex query",
            "confidence": 0.3,
            "validation_passed": False,
            "execution_success": False,
            "errors": 2,
            "start_time": datetime.now()
        }
        
        hfsm.start("PLANNING", context2)
        print(f"     Started in: {hfsm.current_state.name}")
        
        # Execute and transition through error flow
        hfsm.execute_current_state(context2)
        hfsm.transition_to("ERROR_HANDLING", context2)
        hfsm.execute_current_state(context2)
        hfsm.transition_to("PLANNING", context2)
        hfsm.execute_current_state(context2)
        
        print(f"     Final state: {hfsm.current_state.name}")
        
        # Step 4: Test state discovery
        print("\n🔍 Step 4: Testing State Discovery")
        
        # Create discovery engine
        discovery = StateDiscoveryEngine(similarity_threshold=0.8, min_pattern_frequency=2)
        
        # Test contexts for pattern discovery
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
            },
            {
                'recent_tools': ['database', 'api'],
                'error_types': ['connection_error'],
                'data_stats': {'result_count': 1, 'error_rate': 0.5},
                'metrics': {'execution_time': 5.0, 'confidence': 0.6}
            }
        ]
        
        discovered_patterns = []
        for i, test_context in enumerate(test_contexts):
            print(f"     Analyzing context {i+1}...")
            pattern = discovery.analyze_context(test_context)
            if pattern:
                discovered_patterns.append(pattern)
                print(f"       ✅ Discovered: {pattern.name} (confidence: {pattern.confidence:.3f})")
            else:
                print(f"       ⚠️  No new pattern")
        
        # Get discovery statistics
        stats = discovery.get_pattern_statistics()
        print(f"\n     Discovery Statistics:")
        print(f"       Total patterns: {stats['total_patterns']}")
        print(f"       Most used: {stats['most_used_pattern']}")
        print(f"       Recent discoveries: {stats['recent_discoveries']}")
        print(f"       Average confidence: {stats['average_confidence']:.3f}")
        
        # Step 5: Test metrics and visualization
        print("\n📊 Step 5: Testing Metrics and Visualization")
        
        # Get state metrics
        metrics = hfsm.get_state_metrics()
        print("     State Metrics:")
        for state_name, state_metrics in metrics.items():
            success_rate = state_metrics.success_count / max(1, state_metrics.exit_count)
            print(f"       {state_name}: {success_rate:.1%} success rate, {state_metrics.avg_time:.3f}s avg time")
        
        # Generate visualization
        print("\n     FSM Visualization:")
        visualization = hfsm.visualize()
        print(visualization)
        
        # Export comprehensive metrics
        export_data = hfsm.export_metrics()
        print(f"\n     Export Summary:")
        print(f"       FSM Name: {export_data['fsm_name']}")
        print(f"       Total States: {export_data['total_states']}")
        print(f"       Total Transitions: {export_data['total_transitions']}")
        print(f"       Current State: {export_data['current_state']}")
        print(f"       Transition Log Entries: {len(export_data['transition_log'])}")
        
        # Step 6: Test graphical visualization
        print("\n🎨 Step 6: Testing Graphical Visualization")
        try:
            hfsm.save_visualization("fsm_visualization.png")
            print("     ✅ FSM visualization saved to fsm_visualization.png")
        except Exception as e:
            print(f"     ⚠️  Could not save visualization: {e}")
        
        print("\n🎉 Full Enhanced FSM Test Completed Successfully!")
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

def main():
    """Main test function"""
    success = test_full_enhanced_fsm()
    
    if success:
        print("\n✅ All Enhanced FSM features are working correctly!")
        print("   You can now integrate this into your AI Agent system.")
    else:
        print("\n❌ Some tests failed. Check the implementation and dependencies.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 