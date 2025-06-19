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
    logger.info("🚀 Full Enhanced FSM Test")
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
        
        logger.info("✅ Successfully imported full Enhanced FSM components")
        
        # Step 1: Create hierarchical FSM
        logger.info("\n📋 Step 1: Creating Hierarchical FSM")
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
        
        logger.info("   Created {} states (including composite)", extra={"len_hfsm_states_": len(hfsm.states)})
        
        # Step 2: Add probabilistic transitions with context modifiers
        logger.info("\n🔄 Step 2: Adding Probabilistic Transitions")
        
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
        
        logger.info("   Added {} transitions with context modifiers", extra={"len_hfsm_transitions_": len(hfsm.transitions)})
        
        # Step 3: Test FSM execution with different contexts
        logger.info("\n▶️  Step 3: Testing FSM Execution")
        
        # Test context 1: Normal flow
        logger.info("\n   Test 1: Normal flow (high confidence)")
        context1 = {
            "query": "What is 2+2?",
            "confidence": 0.9,
            "validation_passed": True,
            "execution_success": True,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        hfsm.start("PLANNING", context1)
        logger.info("     Started in: {}", extra={"hfsm_current_state_name": hfsm.current_state.name})
        
        # Execute and transition through normal flow
        hfsm.execute_current_state(context1)
        hfsm.transition_to("PROCESSING", context1)
        hfsm.execute_current_state(context1)
        hfsm.transition_to("EXECUTION", context1)
        hfsm.execute_current_state(context1)
        hfsm.transition_to("SYNTHESIS", context1)
        hfsm.execute_current_state(context1)
        
        logger.info("     Final state: {}", extra={"hfsm_current_state_name": hfsm.current_state.name})
        
        # Test context 2: Error flow
        logger.info("\n   Test 2: Error flow (low confidence)")
        context2 = {
            "query": "Complex query",
            "confidence": 0.3,
            "validation_passed": False,
            "execution_success": False,
            "errors": 2,
            "start_time": datetime.now()
        }
        
        hfsm.start("PLANNING", context2)
        logger.info("     Started in: {}", extra={"hfsm_current_state_name": hfsm.current_state.name})
        
        # Execute and transition through error flow
        hfsm.execute_current_state(context2)
        hfsm.transition_to("ERROR_HANDLING", context2)
        hfsm.execute_current_state(context2)
        hfsm.transition_to("PLANNING", context2)
        hfsm.execute_current_state(context2)
        
        logger.info("     Final state: {}", extra={"hfsm_current_state_name": hfsm.current_state.name})
        
        # Step 4: Test state discovery
        logger.info("\n🔍 Step 4: Testing State Discovery")
        
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
            logger.info("     Analyzing context {}...", extra={"i_1": i+1})
            pattern = discovery.analyze_context(test_context)
            if pattern:
                discovered_patterns.append(pattern)
                logger.info("       ✅ Discovered: {} (confidence: {})", extra={"pattern_name": pattern.name, "pattern_confidence": pattern.confidence})
            else:
                logger.info("       ⚠️  No new pattern")
        
        # Get discovery statistics
        stats = discovery.get_pattern_statistics()
        logger.info("\n     Discovery Statistics:")
        logger.info("       Total patterns: {}", extra={"stats__total_patterns_": stats['total_patterns']})
        logger.info("       Most used: {}", extra={"stats__most_used_pattern_": stats['most_used_pattern']})
        logger.info("       Recent discoveries: {}", extra={"stats__recent_discoveries_": stats['recent_discoveries']})
        logger.info("       Average confidence: {}", extra={"stats__average_confidence_": stats['average_confidence']})
        
        # Step 5: Test metrics and visualization
        logger.info("\n📊 Step 5: Testing Metrics and Visualization")
        
        # Get state metrics
        metrics = hfsm.get_state_metrics()
        logger.info("     State Metrics:")
        for state_name, state_metrics in metrics.items():
            success_rate = state_metrics.success_count / max(1, state_metrics.exit_count)
            logger.info("       {}: {} success rate, {}s avg time", extra={"state_name": state_name, "success_rate": success_rate, "state_metrics_avg_time": state_metrics.avg_time})
        
        # Generate visualization
        logger.info("\n     FSM Visualization:")
        visualization = hfsm.visualize()
        logger.info("Value", extra={"value": visualization})
        
        # Export comprehensive metrics
        export_data = hfsm.export_metrics()
        logger.info("\n     Export Summary:")
        logger.info("       FSM Name: {}", extra={"export_data__fsm_name_": export_data['fsm_name']})
        logger.info("       Total States: {}", extra={"export_data__total_states_": export_data['total_states']})
        logger.info("       Total Transitions: {}", extra={"export_data__total_transitions_": export_data['total_transitions']})
        logger.info("       Current State: {}", extra={"export_data__current_state_": export_data['current_state']})
        logger.info("       Transition Log Entries: {}", extra={"len_export_data__transition_log__": len(export_data['transition_log'])})
        
        # Step 6: Test graphical visualization
        logger.info("\n🎨 Step 6: Testing Graphical Visualization")
        try:
            hfsm.save_visualization("fsm_visualization.png")
            logger.info("     ✅ FSM visualization saved to fsm_visualization.png")
        except Exception as e:
            logger.info("     ⚠️  Could not save visualization: {}", extra={"e": e})
        
        logger.info("\n🎉 Full Enhanced FSM Test Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        logger.info("❌ Import Error: {}", extra={"e": e})
        logger.info("   Make sure all dependencies are installed:")
        logger.info("   pip install numpy scikit-learn matplotlib networkx")
        return False
        
    except Exception as e:
        logger.info("❌ Error: {}", extra={"e": e})
        logger.exception("Detailed error information:")
        return False

def main():
    """Main test function"""
    success = test_full_enhanced_fsm()
    
    if success:
        logger.info("\n✅ All Enhanced FSM features are working correctly!")
        logger.info("   You can now integrate this into your AI Agent system.")
    else:
        logger.info("\n❌ Some tests failed. Check the implementation and dependencies.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 