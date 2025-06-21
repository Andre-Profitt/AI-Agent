#!/usr/bin/env python3
"""
🚀 Super AI Agent Demo - Showcasing Advanced Capabilities
This demo shows off the powerful features of our enhanced AI Agent system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import our super agent components
from src.agents.unified_agent import UnifiedAgent, AgentContext, create_agent
from src.infrastructure.agent_config import AgentConfig
from src.core.entities.base_message import Message
from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningType
from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryPriority
from src.gaia_components.tool_executor import create_production_tool_executor
from src.utils.structured_logging import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)

class SuperAgentDemo:
    """Demo class to showcase the super powerful AI Agent capabilities"""
    
    def __init__(self):
        self.agents = {}
        self.tool_executor = None
        self.memory_system = EnhancedMemorySystem()
        self.reasoning_engine = AdvancedReasoningEngine()
        
    async def setup(self):
        """Set up the demo environment"""
        logger.info("🚀 Setting up Super AI Agent Demo...")
        
        # Create tool executor with built-in tools
        self.tool_executor = create_production_tool_executor(max_workers=10)
        
        # Create specialized agents
        self.agents['research'] = await self._create_research_agent()
        self.agents['analyst'] = await self._create_analyst_agent()
        self.agents['creative'] = await self._create_creative_agent()
        self.agents['coordinator'] = await self._create_coordinator_agent()
        
        logger.info("✅ Super AI Agent Demo setup complete!")
        
    async def _create_research_agent(self) -> UnifiedAgent:
        """Create a research-focused agent"""
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.3,
            max_iterations=20,
            enable_memory=True
        )
        
        agent = create_agent(
            agent_type="unified",
            name="Research Agent",
            config=config,
            capabilities=['reasoning', 'tool_use', 'memory']
        )
        
        return agent
        
    async def _create_analyst_agent(self) -> UnifiedAgent:
        """Create a data analysis agent"""
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.2,
            max_iterations=15,
            enable_memory=True
        )
        
        agent = create_agent(
            agent_type="unified",
            name="Data Analyst Agent",
            config=config,
            capabilities=['reasoning', 'tool_use', 'memory']
        )
        
        return agent
        
    async def _create_creative_agent(self) -> UnifiedAgent:
        """Create a creative writing agent"""
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.8,
            max_iterations=10,
            enable_memory=True
        )
        
        agent = create_agent(
            agent_type="unified",
            name="Creative Agent",
            config=config,
            capabilities=['reasoning', 'memory', 'learning']
        )
        
        return agent
        
    async def _create_coordinator_agent(self) -> UnifiedAgent:
        """Create a coordinator agent for multi-agent collaboration"""
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.5,
            max_iterations=25,
            enable_memory=True
        )
        
        agent = create_agent(
            agent_type="unified",
            name="Coordinator Agent",
            config=config,
            capabilities=['reasoning', 'tool_use', 'memory', 'collaboration']
        )
        
        return agent
        
    async def demo_advanced_reasoning(self):
        """Demo: Advanced Reasoning Capabilities"""
        print("\n" + "="*60)
        print("🧠 DEMO 1: Advanced Reasoning Engine")
        print("="*60)
        
        queries = [
            {
                "query": "Why do trees lose their leaves in autumn?",
                "type": ReasoningType.DEDUCTIVE
            },
            {
                "query": "Based on the pattern 2, 4, 8, 16, what comes next?",
                "type": ReasoningType.INDUCTIVE
            },
            {
                "query": "The car won't start and the lights are dim. What's the most likely cause?",
                "type": ReasoningType.ABDUCTIVE
            },
            {
                "query": "Learning a new language is like building a house. Explain this analogy.",
                "type": ReasoningType.ANALOGY
            }
        ]
        
        for q in queries:
            print(f"\n📝 Query: {q['query']}")
            print(f"🔍 Reasoning Type: {q['type'].value}")
            
            result = await self.reasoning_engine.reason(
                query=q['query'],
                context={"demo": True},
                reasoning_type=q['type']
            )
            
            print(f"💡 Conclusion: {result['response']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"🔄 Reasoning Steps:")
            for step in result['reasoning_path'][:3]:  # Show first 3 steps
                print(f"   - {step['step']}")
                
    async def demo_memory_system(self):
        """Demo: Enhanced Memory System"""
        print("\n" + "="*60)
        print("🧩 DEMO 2: Enhanced Memory System")
        print("="*60)
        
        # Store different types of memories
        print("\n📝 Storing memories...")
        
        # Episodic memory
        episode_id = self.memory_system.store_episodic(
            content="User asked about quantum computing applications",
            event_type="user_question",
            metadata={"topic": "quantum_computing", "timestamp": datetime.now()}
        )
        print(f"✅ Stored episodic memory: {episode_id[:8]}...")
        
        # Semantic memory
        semantic_id = self.memory_system.store_semantic(
            content="Quantum computing uses qubits which can be in superposition",
            concepts=["quantum", "computing", "qubits", "superposition"],
            metadata={"source": "knowledge_base"}
        )
        print(f"✅ Stored semantic memory: {semantic_id[:8]}...")
        
        # Working memory
        working_id = self.memory_system.store_working(
            content="Current task: Explain quantum computing to the user",
            priority=MemoryPriority.HIGH,
            metadata={"task_type": "explanation"}
        )
        print(f"✅ Stored working memory: {working_id[:8]}...")
        
        # Retrieve memories
        print("\n🔍 Retrieving memories...")
        
        # Search by concept
        quantum_memories = self.memory_system.retrieve_semantic_by_concept("quantum")
        print(f"Found {len(quantum_memories)} memories related to 'quantum'")
        
        # Get memory statistics
        stats = self.memory_system.get_memory_statistics()
        print(f"\n📊 Memory Statistics:")
        print(f"   - Episodic memories: {stats['episodic_count']}")
        print(f"   - Semantic memories: {stats['semantic_count']}")
        print(f"   - Working memories: {stats['working_count']}")
        print(f"   - Total memories: {stats['total_memories']}")
        
    async def demo_tool_execution(self):
        """Demo: Tool Execution System"""
        print("\n" + "="*60)
        print("🔧 DEMO 3: Production Tool Executor")
        print("="*60)
        
        async with self.tool_executor:
            # Calculator demo
            print("\n🧮 Calculator Tool:")
            calc_result = await self.tool_executor.execute(
                "calculator",
                expression="(2 ** 10) + sqrt(16) - log10(100)"
            )
            if calc_result.status.value == "success":
                print(f"   Result: {calc_result.result['result']}")
                print(f"   Execution time: {calc_result.execution_time:.3f}s")
                
            # Text processor demo
            print("\n📝 Text Processor Tool:")
            text = "The quick brown fox jumps over the lazy dog. This is a demo of our text processing capabilities."
            text_result = await self.tool_executor.execute(
                "text_processor",
                text=text,
                operation="extract_keywords"
            )
            if text_result.status.value == "success":
                print("   Top keywords:")
                for kw in text_result.result['keywords'][:5]:
                    print(f"   - {kw['word']}: {kw['frequency']} occurrences")
                    
            # Data analyzer demo
            print("\n📊 Data Analyzer Tool:")
            sample_data = [
                {"name": "Alice", "age": 25, "score": 85},
                {"name": "Bob", "age": 30, "score": 92},
                {"name": "Charlie", "age": 35, "score": 78},
                {"name": "Diana", "age": 28, "score": 95}
            ]
            data_result = await self.tool_executor.execute(
                "data_analyzer",
                data=sample_data,
                analysis_type="basic"
            )
            if data_result.status.value == "success":
                print(f"   Shape: {data_result.result['shape']}")
                print(f"   Columns: {data_result.result['columns']}")
                
            # Show execution statistics
            stats = self.tool_executor.get_execution_stats()
            print("\n📈 Execution Statistics:")
            for tool_name, tool_stats in stats.items():
                print(f"   {tool_name}:")
                print(f"     - Total calls: {tool_stats['total_calls']}")
                print(f"     - Success rate: {tool_stats['successful_calls']/tool_stats['total_calls']:.1%}")
                print(f"     - Avg time: {tool_stats['avg_time']:.3f}s")
                
    async def demo_multi_agent_collaboration(self):
        """Demo: Multi-Agent Collaboration"""
        print("\n" + "="*60)
        print("🤝 DEMO 4: Multi-Agent Collaboration")
        print("="*60)
        
        # Create a complex task that requires multiple agents
        task = {
            "goal": "Research and analyze climate change data, then create a summary report",
            "subtasks": [
                "Research recent climate change studies",
                "Analyze temperature trend data",
                "Create an engaging summary"
            ]
        }
        
        context = AgentContext(
            session_id="demo_collaboration",
            metadata={"task": task}
        )
        
        print(f"\n📋 Task: {task['goal']}")
        print("\n👥 Agent Collaboration:")
        
        # Research phase
        print("\n1️⃣ Research Agent working...")
        research_msg = Message(
            content="Find recent climate change studies and key findings",
            role="user"
        )
        research_result = await self.agents['research'].process(research_msg, context)
        print(f"   ✅ Research complete: Found relevant studies")
        
        # Analysis phase
        print("\n2️⃣ Analyst Agent working...")
        analysis_msg = Message(
            content="Analyze temperature trends from the research data",
            role="user",
            metadata={"research_data": research_result.content}
        )
        analysis_result = await self.agents['analyst'].process(analysis_msg, context)
        print(f"   ✅ Analysis complete: Identified key trends")
        
        # Creative phase
        print("\n3️⃣ Creative Agent working...")
        creative_msg = Message(
            content="Create an engaging summary of the climate findings",
            role="user",
            metadata={
                "research": research_result.content,
                "analysis": analysis_result.content
            }
        )
        creative_result = await self.agents['creative'].process(creative_msg, context)
        print(f"   ✅ Creative summary complete")
        
        # Coordination
        print("\n4️⃣ Coordinator Agent synthesizing...")
        coord_msg = Message(
            content="Synthesize all findings into a final report",
            role="user",
            metadata={
                "all_results": {
                    "research": research_result.content,
                    "analysis": analysis_result.content,
                    "creative": creative_result.content
                }
            }
        )
        final_result = await self.agents['coordinator'].process(coord_msg, context)
        print(f"   ✅ Final report ready!")
        
        # Show agent metrics
        print("\n📊 Agent Performance Metrics:")
        for name, agent in self.agents.items():
            metrics = agent.get_metrics()
            print(f"   {name}: State={metrics['state']}, Tools={metrics['tools_count']}")
            
    async def demo_adaptive_behavior(self):
        """Demo: Adaptive Agent Behavior"""
        print("\n" + "="*60)
        print("🔄 DEMO 5: Adaptive Agent Behavior")
        print("="*60)
        
        agent = self.agents['coordinator']
        context = AgentContext(
            session_id="demo_adaptive",
            metadata={"adaptive_demo": True}
        )
        
        # Test different query complexities
        queries = [
            {"content": "What is 2+2?", "expected": "simple"},
            {"content": "Explain the theory of relativity in simple terms", "expected": "complex"},
            {"content": "Why did the Roman Empire fall?", "expected": "historical"},
            {"content": "Compare machine learning vs deep learning", "expected": "technical"}
        ]
        
        print("\n🧪 Testing adaptive responses to different query types:")
        
        for q in queries:
            msg = Message(content=q['content'], role="user")
            
            print(f"\n❓ Query: {q['content']}")
            print(f"   Expected type: {q['expected']}")
            
            # Process with agent
            result = await agent.process(msg, context)
            
            # The agent should adapt its reasoning and tool use based on query
            print(f"   🤖 Agent response preview: {result.content[:100]}...")
            print(f"   📊 Confidence: {result.metadata.get('confidence', 0.5):.2%}")
            if result.metadata.get('reasoning_path'):
                print(f"   🔍 Reasoning type used: {result.metadata.get('reasoning_path')[0]['step']}")
                
    async def run_all_demos(self):
        """Run all demos"""
        await self.setup()
        
        print("\n" + "🌟"*30)
        print("🚀 SUPER AI AGENT DEMONSTRATION")
        print("🌟"*30)
        
        demos = [
            self.demo_advanced_reasoning,
            self.demo_memory_system,
            self.demo_tool_execution,
            self.demo_multi_agent_collaboration,
            self.demo_adaptive_behavior
        ]
        
        for demo in demos:
            try:
                await demo()
                await asyncio.sleep(1)  # Brief pause between demos
            except Exception as e:
                logger.error(f"Error in demo {demo.__name__}: {e}")
                
        print("\n" + "="*60)
        print("✨ Super AI Agent Demo Complete!")
        print("="*60)
        
        # Final statistics
        print("\n📊 Final System Statistics:")
        print(f"   - Total agents created: {len(self.agents)}")
        print(f"   - Memory system entries: {self.memory_system.get_memory_statistics()['total_memories']}")
        print(f"   - Tool executions: {sum(s['total_calls'] for s in self.tool_executor.get_execution_stats().values())}")
        
async def main():
    """Main entry point"""
    demo = SuperAgentDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    # Run the super agent demo
    asyncio.run(main())