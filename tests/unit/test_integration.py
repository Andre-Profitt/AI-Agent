#!/usr/bin/env python3
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
    print("🚀 Enhanced FSM Integration Test")
    print("=" * 50)
    
    try:
        # Test 1: Import Enhanced FSM
        print("\n📋 Test 1: Import Enhanced FSM")
        from src.enhanced_fsm import HierarchicalFSM, AtomicState, ProbabilisticTransition
        print("   ✅ Enhanced FSM imported successfully")
        
        # Test 2: Import Migrated Agent
        print("\n📋 Test 2: Import Migrated Agent")
        from src.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
        print("   ✅ MigratedEnhancedFSMAgent imported successfully")
        
        # Test 3: Create Enhanced Agent
        print("\n📋 Test 3: Create Enhanced Agent")
        
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
        print("   ✅ Enhanced Agent created successfully")
        
        # Test 4: Run Enhanced Agent
        print("\n📋 Test 4: Run Enhanced Agent")
        result = agent.run("What is 2+2?")
        
        print(f"   Success: {result['success']}")
        print(f"   Final State: {result['final_state']}")
        print(f"   Result: {result['result']}")
        print(f"   Execution Time: {result['execution_time']:.3f}s")
        
        # Test 5: Get Metrics
        print("\n📋 Test 5: Get Metrics")
        metrics = agent.get_metrics()
        print(f"   FSM Name: {metrics['fsm_name']}")
        print(f"   Total States: {metrics['total_states']}")
        print(f"   Total Transitions: {metrics['total_transitions']}")
        
        # Test 6: Get Discovery Statistics
        print("\n📋 Test 6: Get Discovery Statistics")
        discovery_stats = agent.get_discovery_statistics()
        print(f"   Total Patterns: {discovery_stats.get('total_patterns', 0)}")
        print(f"   Most Used Pattern: {discovery_stats.get('most_used_pattern', 'None')}")
        
        # Test 7: Visualization
        print("\n📋 Test 7: Visualization")
        visualization = agent.visualize_current_state()
        print("   ✅ Visualization generated successfully")
        
        print("\n🎉 Enhanced FSM Integration Test Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Detailed error information:")
        return False

def test_backward_compatibility():
    """Test backward compatibility with existing system"""
    print("\n🔄 Backward Compatibility Test")
    print("=" * 50)
    
    try:
        # Test 1: Import original FSM
        print("\n📋 Test 1: Import Original FSM")
        from src.advanced_agent_fsm import FSMReActAgent
        print("   ✅ Original FSM imported successfully")
        
        # Test 2: Create original agent
        print("\n📋 Test 2: Create Original Agent")
        
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
        print("   ✅ Original Agent created successfully")
        
        print("\n🎉 Backward Compatibility Test Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Detailed error information:")
        return False

def main():
    """Main test function"""
    print("Enhanced FSM Integration Tests")
    print("=" * 60)
    
    # Run integration test
    integration_success = test_enhanced_fsm_integration()
    
    # Run backward compatibility test
    compatibility_success = test_backward_compatibility()
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"  Enhanced FSM Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"  Backward Compatibility: {'✅ PASSED' if compatibility_success else '❌ FAILED'}")
    
    if integration_success and compatibility_success:
        print("\n🎉 All tests passed! Enhanced FSM is ready for integration.")
        print("\nNext steps:")
        print("  1. Use app_enhanced_fsm.py for the enhanced version")
        print("  2. The original app.py still works for backward compatibility")
        print("  3. Monitor metrics and discovered patterns")
        print("  4. Use visualization tools for debugging")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")
    
    return integration_success and compatibility_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 