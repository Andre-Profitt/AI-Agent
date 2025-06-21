#!/usr/bin/env python3
from app import history
from examples.basic.enhanced_fsm_example import visualization
from examples.enhanced_unified_example import metrics
from examples.gaia_usage_example import result1
from examples.gaia_usage_example import result2

from src.api_server import execution
from src.core.entities.agent import Agent
from src.gaia_components.adaptive_tool_system import mock_tools
from src.infrastructure.monitoring.decorators import error_context
from src.tools_introspection import name

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import MigratedEnhancedFSMAgent
from src.agents.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
# TODO: Fix undefined variables: context, current_context, discovery_stats, error_context, execution, history, history_after_reset, i, metrics, mock_tools, name, result1, result2, result3, visualization, agent, context, current_context, discovery_stats, error_context, execution, history, history_after_reset, i, metrics, mock_tools, name, result1, result2, result3, self, visualization
from tests.test_gaia_agent import agent

# TODO: Fix undefined variables: context, current_context, discovery_stats, error_context, execution, history, history_after_reset, i, metrics, mock_tools, name, result1, result2, result3, visualization, agent, context, current_context, discovery_stats, error_context, execution, history, history_after_reset, i, metrics, mock_tools, name, result1, result2, result3, self, visualization

"""
Test Migrated Enhanced FSM Agent
================================

Test the MigratedEnhancedFSMAgent integration with the existing AI Agent system.
"""

import sys
import os
import logging
from math import e
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_migrated_agent():
    """Test the MigratedEnhancedFSMAgent"""
    logger.info("🚀 Test Migrated Enhanced FSM Agent")
    print("=" * 50)

    try:
        # Import the migrated agent
        from src.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent

        # Create mock tools
        class MockTool:
            def __init__(self, name):
                self.name = name
                self.tool_name = name

        mock_tools = [
            MockTool("search"),
            MockTool("calculator"),
            MockTool("database"),
            MockTool("api_client")
        ]

        # Create agent with all features enabled
        logger.info("\n📋 Creating MigratedEnhancedFSMAgent...")
        agent = MigratedEnhancedFSMAgent(
            tools=mock_tools,
            enable_hierarchical=True,
            enable_probabilistic=True,
            enable_discovery=True,
            enable_metrics=True,
            fsm_name="TestEnhancedAgent"
        )

        logger.info("   ✅ Agent created with {} tools", extra={"len_mock_tools_": len(mock_tools)})
        logger.info("   ✅ Hierarchical FSM: {}", extra={"agent_enable_hierarchical": agent.enable_hierarchical})
        logger.info("   ✅ Probabilistic transitions: {}", extra={"agent_enable_probabilistic": agent.enable_probabilistic})
        logger.info("   ✅ State discovery: {}", extra={"agent_enable_discovery": agent.enable_discovery})
        logger.info("   ✅ Metrics: {}", extra={"agent_enable_metrics": agent.enable_metrics})

        # Test 1: Simple query
        logger.info("\n▶️  Test 1: Simple Query")
        result1 = agent.run("What is 2+2?")

        logger.info("   Success: {}", extra={"result1__success_": result1['success']})
        logger.info("   Final State: {}", extra={"result1__final_state_": result1['final_state']})
        logger.info("   Iterations: {}", extra={"result1__iterations_": result1['iterations']})
        logger.info("   Result: {}", extra={"result1__result_": result1['result']})
        logger.info("   Confidence: {}", extra={"result1__confidence_": result1['confidence']})
        logger.info("   Execution Time: {}s", extra={"result1__execution_time_": result1['execution_time']})

        # Test 2: Complex query with context
        logger.info("\n▶️  Test 2: Complex Query with Context")
        context = {
            'confidence': 0.6,
            'validation_passed': True,
            'execution_success': True
        }

        result2 = agent.run("Analyze the performance of machine learning models", context)

        logger.info("   Success: {}", extra={"result2__success_": result2['success']})
        logger.info("   Final State: {}", extra={"result2__final_state_": result2['final_state']})
        logger.info("   Iterations: {}", extra={"result2__iterations_": result2['iterations']})
        logger.info("   Result: {}", extra={"result2__result_": result2['result']})
        logger.info("   Confidence: {}", extra={"result2__confidence_": result2['confidence']})
        logger.info("   Execution Time: {}s", extra={"result2__execution_time_": result2['execution_time']})

        # Test 3: Error handling
        logger.info("\n▶️  Test 3: Error Handling")
        error_context = {
            'confidence': 0.3,
            'validation_passed': False,
            'execution_success': False
        }

        result3 = agent.run("Complex query that might fail", error_context)

        logger.info("   Success: {}", extra={"result3__success_": result3['success']})
        logger.info("   Final State: {}", extra={"result3__final_state_": result3['final_state']})
        logger.info("   Iterations: {}", extra={"result3__iterations_": result3['iterations']})
        logger.info("   Result: {}", extra={"result3__result_": result3['result']})
        logger.info("   Confidence: {}", extra={"result3__confidence_": result3['confidence']})
        logger.info("   Execution Time: {}s", extra={"result3__execution_time_": result3['execution_time']})

        # Test 4: Metrics and visualization
        logger.info("\n📊 Test 4: Metrics and Visualization")

        # Get metrics
        metrics = agent.get_metrics()
        logger.info("   FSM Name: {}", extra={"metrics__fsm_name_": metrics['fsm_name']})
        logger.info("   Total States: {}", extra={"metrics__total_states_": metrics['total_states']})
        logger.info("   Total Transitions: {}", extra={"metrics__total_transitions_": metrics['total_transitions']})
        logger.info("   Transition Log Entries: {}", extra={"len_metrics__transition_log__": len(metrics['transition_log'])})

        # Get discovery statistics
        discovery_stats = agent.get_discovery_statistics()
        logger.info("   Discovery Patterns: {}", extra={"discovery_stats_get__total_patterns___0_": discovery_stats.get('total_patterns', 0)})
        logger.info("   Most Used Pattern: {}", extra={"discovery_stats_get__most_used_pattern____None__": discovery_stats.get('most_used_pattern', 'None')})

        # Show visualization
        logger.info("\n   FSM Visualization:")
        visualization = agent.visualize_current_state()
        logger.info("Value", extra={"value": visualization})

        # Test 5: Execution history
        logger.info("\n📜 Test 5: Execution History")
        history = agent.get_execution_history()
        logger.info("   Total executions: {}", extra={"len_history_": len(history)})

        for i, execution in enumerate(history[:3]):  # Show first 3
            logger.info("     Execution {}: {} (iteration {})", extra={"i_1": i+1, "execution__state_": execution['state'], "execution__iteration_": execution['iteration']})

        # Test 6: Save visualization
        logger.info("\n🎨 Test 6: Save Visualization")
        try:
            agent.save_visualization("migrated_agent_visualization.png")
            logger.info("   ✅ Visualization saved to migrated_agent_visualization.png")
        except Exception as e:
            logger.info("   ⚠️  Could not save visualization: {}", extra={"e": e})

        # Test 7: Reset functionality
        logger.info("\n🔄 Test 7: Reset Functionality")
        agent.reset()

        # Verify reset
        current_context = agent.get_current_context()
        history_after_reset = agent.get_execution_history()

        logger.info("   Context after reset: {} items", extra={"len_current_context_": len(current_context)})
        logger.info("   History after reset: {} items", extra={"len_history_after_reset_": len(history_after_reset)})

        logger.info("\n🎉 Migrated Enhanced FSM Agent Test Completed Successfully!")
        print("=" * 50)

        return True

    except ImportError as e:
        logger.info("❌ Import Error: {}", extra={"e": e})
        return False

    except Exception as e:
        logger.info("❌ Error: {}", extra={"e": e})
        logger.exception("Detailed error information:")
        return False

def main():
    """Main test function"""
    success = test_migrated_agent()

    if success:
        logger.info("\n✅ MigratedEnhancedFSMAgent is working correctly!")
        logger.info("   You can now integrate this into your existing AI Agent system.")
        logger.info("\n   Next steps:")
        logger.info("   1. Update your app.py to use MigratedEnhancedFSMAgent")
        logger.info("   2. Configure the agent with your existing tools")
        logger.info("   3. Test with real queries")
        logger.info("   4. Monitor metrics and discovered patterns")
    else:
        logger.info("\n❌ Some tests failed. Check the implementation.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)