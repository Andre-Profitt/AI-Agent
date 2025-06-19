"""
Enhanced Unified Architecture Example

This example demonstrates how to integrate the existing FSM agents
with the enhanced unified architecture for advanced multi-agent collaboration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from src.adapters.fsm_unified_adapter import UnifiedArchitectureBridge
from src.agents.advanced_agent_fsm import FSMReActAgent
from src.tools import (
    file_reader, advanced_file_reader, web_researcher,
    semantic_search_tool, python_interpreter, tavily_search_backoff,
    get_weather, PythonREPLTool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_fsm_agents() -> Dict[str, FSMReActAgent]:
    """Create multiple FSM agents with different tool sets"""
    
    # Agent 1: General purpose with all tools
    general_tools = [
        file_reader, advanced_file_reader, web_researcher,
        semantic_search_tool, python_interpreter, tavily_search_backoff,
        get_weather, PythonREPLTool
    ]
    general_tools = [tool for tool in general_tools if tool is not None]
    
    general_agent = FSMReActAgent(tools=general_tools)
    
    # Agent 2: Specialized for web research
    research_tools = [
        web_researcher, tavily_search_backoff, semantic_search_tool
    ]
    research_tools = [tool for tool in research_tools if tool is not None]
    
    research_agent = FSMReActAgent(tools=research_tools)
    
    # Agent 3: Specialized for computation
    computation_tools = [
        python_interpreter, PythonREPLTool
    ]
    computation_tools = [tool for tool in computation_tools if tool is not None]
    
    computation_agent = FSMReActAgent(tools=computation_tools)
    
    return {
        "general": general_agent,
        "research": research_agent,
        "computation": computation_agent
    }

async def demonstrate_enhanced_architecture():
    """Demonstrate the enhanced unified architecture capabilities"""
    
    logger.info("=== Enhanced Unified Architecture Demo ===")
    
    # Create bridge
    bridge = UnifiedArchitectureBridge()
    await bridge.initialize_platform()
    
    try:
        # Create FSM agents
        logger.info("Creating FSM agents...")
        fsm_agents = await create_fsm_agents()
        
        # Register agents with unified architecture
        agent_configs = [
            {
                "key": "general",
                "agent": fsm_agents["general"],
                "id": "fsm-general-001",
                "name": "General Purpose FSM Agent",
                "tags": ["general", "reasoning", "tools"]
            },
            {
                "key": "research",
                "agent": fsm_agents["research"],
                "id": "fsm-research-001",
                "name": "Research FSM Agent",
                "tags": ["research", "web", "search"]
            },
            {
                "key": "computation",
                "agent": fsm_agents["computation"],
                "id": "fsm-computation-001",
                "name": "Computation FSM Agent",
                "tags": ["computation", "python", "math"]
            }
        ]
        
        registered_agents = []
        for config in agent_configs:
            success = await bridge.register_fsm_agent(
                config["agent"],
                config["id"],
                config["name"],
                config["tags"]
            )
            
            if success:
                registered_agents.append(config["id"])
                logger.info(f"Registered {config['name']}")
            else:
                logger.error(f"Failed to register {config['name']}")
        
        if not registered_agents:
            logger.error("No agents registered successfully")
            return
        
        # Demonstrate different types of tasks
        tasks = [
            {
                "query": "What is the current weather in San Francisco?",
                "type": "weather_query",
                "priority": 3,
                "expected_agent": "fsm-general-001"
            },
            {
                "query": "Search for recent developments in artificial intelligence",
                "type": "research_query",
                "priority": 4,
                "expected_agent": "fsm-research-001"
            },
            {
                "query": "Calculate the factorial of 10",
                "type": "computation_query",
                "priority": 2,
                "expected_agent": "fsm-computation-001"
            },
            {
                "query": "What is the population of Tokyo and how does it compare to New York?",
                "type": "complex_query",
                "priority": 5,
                "expected_agent": "fsm-general-001"
            }
        ]
        
        logger.info("\n=== Executing Tasks ===")
        
        for i, task_config in enumerate(tasks, 1):
            logger.info(f"\nTask {i}: {task_config['query']}")
            
            # Create unified task
            task = await bridge.create_task_from_query(
                task_config["query"],
                task_config["type"],
                task_config["priority"]
            )
            
            # Submit task
            start_time = datetime.now()
            result = await bridge.submit_task(task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info(f"Success: {result.success}")
            logger.info(f"Agent used: {result.agent_id}")
            
            if result.success:
                if isinstance(result.result, dict):
                    output = result.result.get("output", str(result.result))
                    final_answer = result.result.get("final_answer", "")
                    
                    logger.info(f"Output: {output[:200]}...")
                    if final_answer:
                        logger.info(f"Final answer: {final_answer}")
                else:
                    logger.info(f"Result: {str(result.result)[:200]}...")
            else:
                logger.error(f"Error: {result.error}")
            
            # Get agent metrics
            metrics = await bridge.get_agent_metrics(result.agent_id)
            if metrics:
                logger.info(f"Agent metrics - Success rate: {metrics.get('success_rate', 0):.2%}")
        
        # Demonstrate performance tracking
        logger.info("\n=== Performance Summary ===")
        
        for agent_id in registered_agents:
            metrics = await bridge.get_agent_metrics(agent_id)
            if metrics:
                logger.info(f"\n{agent_id}:")
                logger.info(f"  Total tasks: {metrics.get('total_tasks', 0)}")
                logger.info(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
                logger.info(f"  Avg execution time: {metrics.get('avg_execution_time', 0):.2f}s")
                
                task_breakdown = metrics.get('task_breakdown', {})
                if task_breakdown:
                    logger.info("  Task breakdown:")
                    for task_type, stats in task_breakdown.items():
                        logger.info(f"    {task_type}: {stats['count']} tasks, "
                                  f"{stats['success_rate']:.2%} success rate")
        
        logger.info("\n=== Demo Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        await bridge.shutdown()

async def demonstrate_agent_health_monitoring():
    """Demonstrate agent health monitoring capabilities"""
    
    logger.info("\n=== Agent Health Monitoring Demo ===")
    
    bridge = UnifiedArchitectureBridge()
    await bridge.initialize_platform()
    
    try:
        # Create and register a single agent for health monitoring
        fsm_agents = await create_fsm_agents()
        general_agent = fsm_agents["general"]
        
        success = await bridge.register_fsm_agent(
            general_agent,
            "health-monitor-agent",
            "Health Monitor Agent",
            ["monitoring", "health"]
        )
        
        if success:
            # Get health check
            adapter = bridge.adapters["health-monitor-agent"]
            health = await adapter.health_check()
            
            logger.info("Agent Health Check:")
            logger.info(f"  Healthy: {health['healthy']}")
            logger.info(f"  Status: {health['status']}")
            logger.info(f"  Agent Type: {health['agent_type']}")
            logger.info(f"  Capabilities: {health['capabilities']}")
            logger.info(f"  Tools Available: {health['tools_available']}")
            
            # Simulate some work
            task = await bridge.create_task_from_query(
                "What is 2 + 2?",
                "simple_math",
                1
            )
            
            result = await bridge.submit_task(task)
            logger.info(f"Simple task result: {result.success}")
            
            # Check health again
            health_after = await adapter.health_check()
            logger.info(f"Health after task: {health_after['healthy']}")
        
    finally:
        await bridge.shutdown()

if __name__ == "__main__":
    # Run the main demonstration
    asyncio.run(demonstrate_enhanced_architecture())
    
    # Run health monitoring demonstration
    asyncio.run(demonstrate_agent_health_monitoring()) 