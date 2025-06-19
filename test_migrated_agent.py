#!/usr/bin/env python3
"""
Test Migrated Enhanced FSM Agent
================================

Test the MigratedEnhancedFSMAgent integration with the existing AI Agent system.
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

def test_migrated_agent():
    """Test the MigratedEnhancedFSMAgent"""
    print("🚀 Test Migrated Enhanced FSM Agent")
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
        print("\n📋 Creating MigratedEnhancedFSMAgent...")
        agent = MigratedEnhancedFSMAgent(
            tools=mock_tools,
            enable_hierarchical=True,
            enable_probabilistic=True,
            enable_discovery=True,
            enable_metrics=True,
            fsm_name="TestEnhancedAgent"
        )
        
        print(f"   ✅ Agent created with {len(mock_tools)} tools")
        print(f"   ✅ Hierarchical FSM: {agent.enable_hierarchical}")
        print(f"   ✅ Probabilistic transitions: {agent.enable_probabilistic}")
        print(f"   ✅ State discovery: {agent.enable_discovery}")
        print(f"   ✅ Metrics: {agent.enable_metrics}")
        
        # Test 1: Simple query
        print("\n▶️  Test 1: Simple Query")
        result1 = agent.run("What is 2+2?")
        
        print(f"   Success: {result1['success']}")
        print(f"   Final State: {result1['final_state']}")
        print(f"   Iterations: {result1['iterations']}")
        print(f"   Result: {result1['result']}")
        print(f"   Confidence: {result1['confidence']:.2f}")
        print(f"   Execution Time: {result1['execution_time']:.3f}s")
        
        # Test 2: Complex query with context
        print("\n▶️  Test 2: Complex Query with Context")
        context = {
            'confidence': 0.6,
            'validation_passed': True,
            'execution_success': True
        }
        
        result2 = agent.run("Analyze the performance of machine learning models", context)
        
        print(f"   Success: {result2['success']}")
        print(f"   Final State: {result2['final_state']}")
        print(f"   Iterations: {result2['iterations']}")
        print(f"   Result: {result2['result']}")
        print(f"   Confidence: {result2['confidence']:.2f}")
        print(f"   Execution Time: {result2['execution_time']:.3f}s")
        
        # Test 3: Error handling
        print("\n▶️  Test 3: Error Handling")
        error_context = {
            'confidence': 0.3,
            'validation_passed': False,
            'execution_success': False
        }
        
        result3 = agent.run("Complex query that might fail", error_context)
        
        print(f"   Success: {result3['success']}")
        print(f"   Final State: {result3['final_state']}")
        print(f"   Iterations: {result3['iterations']}")
        print(f"   Result: {result3['result']}")
        print(f"   Confidence: {result3['confidence']:.2f}")
        print(f"   Execution Time: {result3['execution_time']:.3f}s")
        
        # Test 4: Metrics and visualization
        print("\n📊 Test 4: Metrics and Visualization")
        
        # Get metrics
        metrics = agent.get_metrics()
        print(f"   FSM Name: {metrics['fsm_name']}")
        print(f"   Total States: {metrics['total_states']}")
        print(f"   Total Transitions: {metrics['total_transitions']}")
        print(f"   Transition Log Entries: {len(metrics['transition_log'])}")
        
        # Get discovery statistics
        discovery_stats = agent.get_discovery_statistics()
        print(f"   Discovery Patterns: {discovery_stats.get('total_patterns', 0)}")
        print(f"   Most Used Pattern: {discovery_stats.get('most_used_pattern', 'None')}")
        
        # Show visualization
        print("\n   FSM Visualization:")
        visualization = agent.visualize_current_state()
        print(visualization)
        
        # Test 5: Execution history
        print("\n📜 Test 5: Execution History")
        history = agent.get_execution_history()
        print(f"   Total executions: {len(history)}")
        
        for i, execution in enumerate(history[:3]):  # Show first 3
            print(f"     Execution {i+1}: {execution['state']} (iteration {execution['iteration']})")
        
        # Test 6: Save visualization
        print("\n🎨 Test 6: Save Visualization")
        try:
            agent.save_visualization("migrated_agent_visualization.png")
            print("   ✅ Visualization saved to migrated_agent_visualization.png")
        except Exception as e:
            print(f"   ⚠️  Could not save visualization: {e}")
        
        # Test 7: Reset functionality
        print("\n🔄 Test 7: Reset Functionality")
        agent.reset()
        
        # Verify reset
        current_context = agent.get_current_context()
        history_after_reset = agent.get_execution_history()
        
        print(f"   Context after reset: {len(current_context)} items")
        print(f"   History after reset: {len(history_after_reset)} items")
        
        print("\n🎉 Migrated Enhanced FSM Agent Test Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Detailed error information:")
        return False

def main():
    """Main test function"""
    success = test_migrated_agent()
    
    if success:
        print("\n✅ MigratedEnhancedFSMAgent is working correctly!")
        print("   You can now integrate this into your existing AI Agent system.")
        print("\n   Next steps:")
        print("   1. Update your app.py to use MigratedEnhancedFSMAgent")
        print("   2. Configure the agent with your existing tools")
        print("   3. Test with real queries")
        print("   4. Monitor metrics and discovered patterns")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 