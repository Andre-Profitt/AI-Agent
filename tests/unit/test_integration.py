#!/usr/bin/env python3
from examples.enhanced_unified_example import metrics

from src.core.entities.agent import Agent
from src.gaia_components.adaptive_tool_system import mock_tools
from src.tools_introspection import name

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import FSMReActAgent

from src.agents.advanced_agent_fsm import MigratedEnhancedFSMAgent
from src.agents.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
# TODO: Fix undefined variables: compatibility_success, discovery_stats, e, integration_success, logging, metrics, mock_tools, name, os, result, sys
from tests.test_gaia_agent import agent

# TODO: Fix undefined variables: agent, compatibility_success, discovery_stats, e, integration_success, metrics, mock_tools, name, result, self

"""
Integration Test for Enhanced FSM
=================================

Test the integration of Enhanced FSM with the existing AI Agent system.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_fsm_integration():
    """Test the Enhanced FSM integration"""
    logger.info("🚀 Enhanced FSM Integration Test")
    print("=" * 50)

    try:
        # Test 1: Import Enhanced FSM
        logger.info("\n📋 Test 1: Import Enhanced FSM")
        from src.enhanced_fsm import HierarchicalFSM, AtomicState, ProbabilisticTransition
        logger.info("   ✅ Enhanced FSM imported successfully")

        # Test 2: Import Migrated Agent
        logger.info("\n📋 Test 2: Import Migrated Agent")
        from src.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
        logger.info("   ✅ MigratedEnhancedFSMAgent imported successfully")

        # Test 3: Create Enhanced Agent
        logger.info("\n📋 Test 3: Create Enhanced Agent")

        # Create mock tools
        class MockTool:
            def __init__(self, name):
                self.name = name
                self.tool_name = name

        mock_tools = [MockTool("search"), MockTool("calculator")]

        # Create agent
        agent = MigratedEnhancedFSMAgent(
            tools=mock_tools,
            enable_hierarchical=True,
            enable_probabilistic=True,
            enable_discovery=True,
            enable_metrics=True
        )
        logger.info("   ✅ Enhanced Agent created successfully")

        # Test 4: Run Enhanced Agent
        logger.info("\n📋 Test 4: Run Enhanced Agent")
        result = agent.run("What is 2+2?")

        logger.info("   Success: {}", extra={"result__success_": result['success']})
        logger.info("   Final State: {}", extra={"result__final_state_": result['final_state']})
        logger.info("   Result: {}", extra={"result__result_": result['result']})
        logger.info("   Execution Time: {}s", extra={"result__execution_time_": result['execution_time']})

        # Test 5: Get Metrics
        logger.info("\n📋 Test 5: Get Metrics")
        metrics = agent.get_metrics()
        logger.info("   FSM Name: {}", extra={"metrics__fsm_name_": metrics['fsm_name']})
        logger.info("   Total States: {}", extra={"metrics__total_states_": metrics['total_states']})
        logger.info("   Total Transitions: {}", extra={"metrics__total_transitions_": metrics['total_transitions']})

        # Test 6: Get Discovery Statistics
        logger.info("\n📋 Test 6: Get Discovery Statistics")
        discovery_stats = agent.get_discovery_statistics()
        logger.info("   Total Patterns: {}", extra={"discovery_stats_get__total_patterns___0_": discovery_stats.get('total_patterns', 0)})
        logger.info("   Most Used Pattern: {}", extra={"discovery_stats_get__most_used_pattern____None__": discovery_stats.get('most_used_pattern', 'None')})

        # Test 7: Visualization
        logger.info("\n📋 Test 7: Visualization")
        visualization = agent.visualize_current_state()
        logger.info("   ✅ Visualization generated successfully")

        logger.info("\n🎉 Enhanced FSM Integration Test Completed Successfully!")
        print("=" * 50)

        return True

    except ImportError as e:
        logger.info("❌ Import Error: {}", extra={"e": e})
        return False

    except Exception as e:
        logger.info("❌ Error: {}", extra={"e": e})
        logger.exception("Detailed error information:")
        return False

def test_backward_compatibility():
    """Test backward compatibility with existing system"""
    logger.info("\n🔄 Backward Compatibility Test")
    print("=" * 50)

    try:
        # Test 1: Import original FSM
        logger.info("\n📋 Test 1: Import Original FSM")
        from src.advanced_agent_fsm import FSMReActAgent
        logger.info("   ✅ Original FSM imported successfully")

        # Test 2: Create original agent
        logger.info("\n📋 Test 2: Create Original Agent")

        # Create mock tools
        class MockTool:
            def __init__(self, name):
                self.name = name
                self.tool_name = name

        mock_tools = [MockTool("search"), MockTool("calculator")]

        # Create original agent
        original_agent = FSMReActAgent(
            tools=mock_tools,
            model_name="gpt-3.5-turbo",
            model_preference="balanced"
        )
        logger.info("   ✅ Original Agent created successfully")

        logger.info("\n🎉 Backward Compatibility Test Completed Successfully!")
        print("=" * 50)

        return True

    except Exception as e:
        logger.info("❌ Error: {}", extra={"e": e})
        logger.exception("Detailed error information:")
        return False

def main():
    """Main test function"""
    logger.info("Enhanced FSM Integration Tests")
    print("=" * 60)

    # Run integration test
    integration_success = test_enhanced_fsm_integration()

    # Run backward compatibility test
    compatibility_success = test_backward_compatibility()

    # Summary
    logger.info("\n{}", extra={"____60": '='*60})
    logger.info("Test Results:")
    logger.info("  Enhanced FSM Integration: {}", extra={"___PASSED__if_integration_success_else____FAILED_": '✅ PASSED' if integration_success else '❌ FAILED'})
    logger.info("  Backward Compatibility: {}", extra={"___PASSED__if_compatibility_success_else____FAILED_": '✅ PASSED' if compatibility_success else '❌ FAILED'})

    if integration_success and compatibility_success:
        logger.info("\n🎉 All tests passed! Enhanced FSM is ready for integration.")
        logger.info("\nNext steps:")
        logger.info("  1. Use app_enhanced_fsm.py for the enhanced version")
        logger.info("  2. The original app.py still works for backward compatibility")
        logger.info("  3. Monitor metrics and discovered patterns")
        logger.info("  4. Use visualization tools for debugging")
    else:
        logger.info("\n⚠️  Some tests failed. Check the implementation.")

    return integration_success and compatibility_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)