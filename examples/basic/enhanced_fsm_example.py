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
    logger.info("🚀 Enhanced FSM Example")
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
        
        logger.info("✅ Successfully imported Enhanced FSM components")
        
        # Step 1: Create a simple FSM
        logger.info("\n📋 Step 1: Creating Simple FSM")
        hfsm = HierarchicalFSM("ExampleFSM")
        
        # Create atomic states
        planning = AtomicState("PLANNING")
        execution = AtomicState("EXECUTION")
        synthesis = AtomicState("SYNTHESIS")
        
        # Add states to FSM
        hfsm.add_state(planning)
        hfsm.add_state(execution)
        hfsm.add_state(synthesis)
        
        logger.info("   Created {} states: {}", extra={"len_hfsm_states_": len(hfsm.states), "list_hfsm_states_keys___": list(hfsm.states.keys())})
        
        # Step 2: Add probabilistic transitions
        logger.info("\n🔄 Step 2: Adding Probabilistic Transitions")
        
        # Create transitions
        plan_to_exec = ProbabilisticTransition("PLANNING", "EXECUTION", 0.9)
        exec_to_synth = ProbabilisticTransition("EXECUTION", "SYNTHESIS", 0.8)
        
        # Add context modifiers
        plan_to_exec.add_context_modifier("confidence<0.5", 0.5)
        exec_to_synth.add_context_modifier("errors>0", 0.3)
        
        # Add transitions to FSM
        hfsm.add_transition(plan_to_exec)
        hfsm.add_transition(exec_to_synth)
        
        logger.info("   Added {} transitions", extra={"len_hfsm_transitions_": len(hfsm.transitions)})
        
        # Step 3: Test FSM execution
        logger.info("\n▶️  Step 3: Testing FSM Execution")
        
        # Create context
        context = {
            "query": "What is 2+2?",
            "confidence": 0.8,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        # Start FSM
        hfsm.start("PLANNING", context)
        logger.info("   Started FSM in state: {}", extra={"hfsm_current_state_name": hfsm.current_state.name})
        
        # Get available transitions
        available = hfsm.get_available_transitions(context)
        logger.info("   Available transitions: {}", extra={"len_available_": len(available)})
        for transition in available:
            logger.info("     -> {} (p={})", extra={"transition__to_state_": transition['to_state'], "transition__probability_": transition['probability']})
        
        # Execute transitions
        logger.info("\n   Executing transitions...")
        
        # Transition to execution
        success = hfsm.transition_to("EXECUTION", context)
        logger.info("     PLANNING -> EXECUTION: {}", extra={"____if_success_else____": '✅' if success else '❌'})
        
        # Transition to synthesis
        success = hfsm.transition_to("SYNTHESIS", context)
        logger.info("     EXECUTION -> SYNTHESIS: {}", extra={"____if_success_else____": '✅' if success else '❌'})
        
        logger.info("   Final state: {}", extra={"hfsm_current_state_name": hfsm.current_state.name})
        
        # Step 4: Test state discovery
        logger.info("\n🔍 Step 4: Testing State Discovery")
        
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
            logger.info("   Analyzing context {}...", extra={"i_1": i+1})
            pattern = discovery.analyze_context(test_context)
            if pattern:
                logger.info("     ✅ Discovered pattern: {}", extra={"pattern_name": pattern.name})
            else:
                logger.info("     ⚠️  No new pattern discovered")
        
        # Step 5: Test metrics and visualization
        logger.info("\n📊 Step 5: Testing Metrics and Visualization")
        
        # Get state metrics
        metrics = hfsm.get_state_metrics()
        logger.info("   State Metrics:")
        for state_name, state_metrics in metrics.items():
            success_rate = state_metrics.success_count / max(1, state_metrics.exit_count)
            logger.info("     {}: {} success rate, {}s avg time", extra={"state_name": state_name, "success_rate": success_rate, "state_metrics_avg_time": state_metrics.avg_time})
        
        # Generate visualization
        logger.info("\n   FSM Visualization:")
        visualization = hfsm.visualize()
        logger.info("Value", extra={"value": visualization})
        
        # Export comprehensive metrics
        export_data = hfsm.export_metrics()
        logger.info("\n   Export Summary:")
        logger.info("     FSM Name: {}", extra={"export_data__fsm_name_": export_data['fsm_name']})
        logger.info("     Total States: {}", extra={"export_data__total_states_": export_data['total_states']})
        logger.info("     Total Transitions: {}", extra={"export_data__total_transitions_": export_data['total_transitions']})
        logger.info("     Transition Log Entries: {}", extra={"len_export_data__transition_log__": len(export_data['transition_log'])})
        
        logger.info("\n🎉 Enhanced FSM Example Completed Successfully!")
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

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
