#!/usr/bin/env python3
"""
Verify that the full AI Agent pipeline works end-to-end
This script tests all major components and their integration
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineVerifier:
    """Verifies the complete AI Agent pipeline"""
    
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
        
    async def verify_imports(self) -> bool:
        """Verify all required imports work"""
        print("\n🔍 Verifying imports...")
        
        try:
            # Core imports
            from src.agents.unified_agent import UnifiedAgent, AgentContext, create_agent
            from src.infrastructure.agent_config import AgentConfig
            from src.core.entities.base_message import Message
            
            # Subsystem imports
            from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningType
            from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryPriority
            from src.gaia_components.tool_executor import create_production_tool_executor
            
            # Utils imports
            from src.utils.structured_logging import get_logger
            
            self.results["passed"].append("✅ All imports successful")
            return True
            
        except ImportError as e:
            self.results["failed"].append(f"❌ Import error: {str(e)}")
            return False
            
    async def verify_reasoning_engine(self) -> bool:
        """Verify reasoning engine functionality"""
        print("\n🧠 Verifying Reasoning Engine...")
        
        try:
            from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningType
            
            engine = AdvancedReasoningEngine()
            
            # Test basic reasoning
            result = await engine.reason(
                query="What is 2+2?",
                context={"test": True},
                reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
            )
            
            if result and "response" in result and result["confidence"] > 0:
                self.results["passed"].append("✅ Reasoning engine working")
                return True
            else:
                self.results["failed"].append("❌ Reasoning engine failed to produce valid result")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"❌ Reasoning engine error: {str(e)}")
            return False
            
    async def verify_memory_system(self) -> bool:
        """Verify memory system functionality"""
        print("\n🧩 Verifying Memory System...")
        
        try:
            from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryPriority
            
            memory = EnhancedMemorySystem(persist_directory="./test_memory")
            
            # Test memory storage
            episode_id = memory.store_episodic(
                content="Test episodic memory",
                event_type="test"
            )
            
            semantic_id = memory.store_semantic(
                content="Test semantic memory",
                concepts=["test", "memory"]
            )
            
            working_id = memory.store_working(
                content="Test working memory",
                priority=MemoryPriority.HIGH
            )
            
            # Test retrieval
            memories = memory.search_memories("test")
            
            if len(memories) >= 3:
                self.results["passed"].append("✅ Memory system working")
                return True
            else:
                self.results["failed"].append("❌ Memory system failed to store/retrieve")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"❌ Memory system error: {str(e)}")
            return False
            
    async def verify_tool_executor(self) -> bool:
        """Verify tool executor functionality"""
        print("\n🔧 Verifying Tool Executor...")
        
        try:
            from src.gaia_components.tool_executor import create_production_tool_executor
            
            executor = create_production_tool_executor()
            
            async with executor:
                # Test calculator
                result = await executor.execute(
                    "calculator",
                    expression="2 + 2"
                )
                
                if result.status.value == "success" and result.result["result"] == 4.0:
                    self.results["passed"].append("✅ Tool executor working")
                    return True
                else:
                    self.results["failed"].append("❌ Tool executor calculation failed")
                    return False
                    
        except Exception as e:
            self.results["failed"].append(f"❌ Tool executor error: {str(e)}")
            return False
            
    async def verify_unified_agent(self) -> bool:
        """Verify unified agent functionality"""
        print("\n🤖 Verifying Unified Agent...")
        
        try:
            from src.agents.unified_agent import create_agent, AgentContext
            from src.infrastructure.agent_config import AgentConfig
            from src.core.entities.base_message import Message
            
            # Create agent
            config = AgentConfig(
                model_name="gpt-4",
                temperature=0.5,
                enable_memory=True
            )
            
            agent = create_agent(
                agent_type="unified",
                name="Verification Agent",
                config=config
            )
            
            # Create context
            context = AgentContext(
                session_id="verify_session",
                metadata={"verification": True}
            )
            
            # Test message processing
            message = Message(
                content="Hello, this is a test",
                role="user"
            )
            
            response = await agent.process(message, context)
            
            if response and response.content and response.role == "assistant":
                self.results["passed"].append("✅ Unified agent working")
                return True
            else:
                self.results["failed"].append("❌ Unified agent failed to process message")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"❌ Unified agent error: {str(e)}")
            return False
            
    async def verify_multi_agent_collaboration(self) -> bool:
        """Verify multi-agent collaboration"""
        print("\n🤝 Verifying Multi-Agent Collaboration...")
        
        try:
            from src.agents.unified_agent import create_agent
            from src.infrastructure.agent_config import AgentConfig
            
            config = AgentConfig(enable_memory=True)
            
            # Create two agents
            agent1 = create_agent(
                agent_type="unified",
                name="Agent 1",
                config=config,
                capabilities=['reasoning', 'memory', 'collaboration']
            )
            
            agent2 = create_agent(
                agent_type="unified",
                name="Agent 2", 
                config=config,
                capabilities=['reasoning', 'memory', 'collaboration']
            )
            
            # Test collaboration
            task = {"action": "collaborate", "data": "test"}
            result = await agent1.collaborate(agent2, task)
            
            if result:
                self.results["passed"].append("✅ Multi-agent collaboration working")
                return True
            else:
                self.results["failed"].append("❌ Multi-agent collaboration failed")
                return False
                
        except Exception as e:
            self.results["failed"].append(f"❌ Multi-agent collaboration error: {str(e)}")
            return False
            
    async def verify_end_to_end_pipeline(self) -> bool:
        """Verify complete end-to-end pipeline"""
        print("\n🚀 Verifying End-to-End Pipeline...")
        
        try:
            from src.agents.unified_agent import create_agent, AgentContext
            from src.infrastructure.agent_config import AgentConfig
            from src.core.entities.base_message import Message
            from src.gaia_components.tool_executor import create_production_tool_executor
            
            # Create complete system
            config = AgentConfig(
                model_name="gpt-4",
                temperature=0.5,
                max_iterations=10,
                enable_memory=True
            )
            
            agent = create_agent(
                agent_type="unified",
                name="Pipeline Test Agent",
                config=config,
                capabilities=['reasoning', 'tool_use', 'memory']
            )
            
            context = AgentContext(
                session_id="pipeline_test",
                metadata={"test": True}
            )
            
            tool_executor = create_production_tool_executor()
            
            async with tool_executor:
                # Test 1: Simple reasoning
                msg1 = Message(content="What is the capital of France?", role="user")
                response1 = await agent.process(msg1, context)
                
                if not response1:
                    self.results["failed"].append("❌ Pipeline failed at simple reasoning")
                    return False
                    
                # Test 2: Tool usage
                msg2 = Message(content="Calculate 15 * 15", role="user")
                response2 = await agent.process(msg2, context)
                
                if not response2:
                    self.results["failed"].append("❌ Pipeline failed at tool usage")
                    return False
                    
                # Test 3: Memory recall
                msg3 = Message(content="What did I ask you first?", role="user")
                response3 = await agent.process(msg3, context)
                
                if not response3:
                    self.results["failed"].append("❌ Pipeline failed at memory recall")
                    return False
                    
            self.results["passed"].append("✅ End-to-end pipeline working")
            return True
            
        except Exception as e:
            self.results["failed"].append(f"❌ End-to-end pipeline error: {str(e)}")
            return False
            
    def generate_report(self):
        """Generate verification report"""
        print("\n" + "="*60)
        print("📊 VERIFICATION REPORT")
        print("="*60)
        
        total_tests = len(self.results["passed"]) + len(self.results["failed"])
        passed = len(self.results["passed"])
        
        print(f"\n✅ Passed: {passed}/{total_tests}")
        for item in self.results["passed"]:
            print(f"   {item}")
            
        if self.results["failed"]:
            print(f"\n❌ Failed: {len(self.results['failed'])}/{total_tests}")
            for item in self.results["failed"]:
                print(f"   {item}")
                
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings: {len(self.results['warnings'])}")
            for item in self.results["warnings"]:
                print(f"   {item}")
                
        print("\n" + "="*60)
        
        if passed == total_tests:
            print("🎉 ALL TESTS PASSED! Your AI Agent pipeline is working perfectly!")
        else:
            print("⚠️  Some tests failed. Please check the errors above.")
            
        return passed == total_tests
        
    async def run_verification(self):
        """Run all verification tests"""
        print("🔬 Starting AI Agent Pipeline Verification...")
        print("="*60)
        
        tests = [
            self.verify_imports,
            self.verify_reasoning_engine,
            self.verify_memory_system,
            self.verify_tool_executor,
            self.verify_unified_agent,
            self.verify_multi_agent_collaboration,
            self.verify_end_to_end_pipeline
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                self.results["failed"].append(f"❌ Test {test.__name__} crashed: {str(e)}")
                
        return self.generate_report()


async def main():
    """Main entry point"""
    verifier = PipelineVerifier()
    success = await verifier.run_verification()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())