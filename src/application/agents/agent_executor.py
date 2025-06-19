"""
Agent executor implementation for executing AI agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime

from src.core.interfaces.agent_executor import AgentExecutor
from src.core.entities.agent import Agent, AgentType
from src.core.entities.message import Message
from src.shared.exceptions import DomainException
from typing import Optional, Dict, Any, List, Union, Tuple


class AgentExecutorImpl(AgentExecutor):
    """
    Implementation of the agent executor interface.
    
    This class handles the execution of different types of agents
    and manages their lifecycle during processing.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._active_executions: Dict[UUID, Dict[str, Any]] = {}
    
    async def execute(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """
        Execute an agent with a given message.
        
        Args:
            agent: The agent to execute
            message: The message to process
            
        Returns:
            Dictionary containing the execution result
        """
        execution_id = uuid4()
        start_time = datetime.now()
        
        try:
            # Register execution
            self._active_executions[execution_id] = {
                "agent_id": agent.id,
                "message_id": message.id,
                "start_time": start_time,
                "status": "running"
            }
            
            self.logger.info("Starting execution {} for agent {}", extra={"execution_id": execution_id, "agent_id": agent.id})
            
            # Validate agent
            validation_result = await self.validate_agent(agent)
            if not validation_result.get("valid", False):
                raise DomainException(f"Agent validation failed: {validation_result.get('errors', [])}")
            
            # Execute based on agent type
            if agent.agent_type == AgentType.FSM_REACT:
                result = await self._execute_fsm_react_agent(agent, message)
            elif agent.agent_type == AgentType.NEXT_GEN:
                result = await self._execute_next_gen_agent(agent, message)
            elif agent.agent_type == AgentType.CREW:
                result = await self._execute_crew_agent(agent, message)
            elif agent.agent_type == AgentType.SPECIALIZED:
                result = await self._execute_specialized_agent(agent, message)
            else:
                raise DomainException(f"Unsupported agent type: {agent.agent_type}")
            
            # Update execution status
            execution_time = (datetime.now() - start_time).total_seconds()
            self._active_executions[execution_id]["status"] = "completed"
            self._active_executions[execution_id]["execution_time"] = execution_time
            
            # Add execution metadata
            result["execution_id"] = str(execution_id)
            result["execution_time"] = execution_time
            
            self.logger.info("Execution {} completed successfully in {}s", extra={"execution_id": execution_id, "execution_time": execution_time})
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._active_executions[execution_id]["status"] = "failed"
            self._active_executions[execution_id]["error"] = str(e)
            self._active_executions[execution_id]["execution_time"] = execution_time
            
            self.logger.error("Execution {} failed: {}", extra={"execution_id": execution_id, "str_e_": str(e)})
            raise DomainException(f"Agent execution failed: {str(e)}")
    
    async def validate_agent(self, agent: Agent) -> Dict[str, Any]:
        """
        Validate an agent before execution.
        
        Args:
            agent: The agent to validate
            
        Returns:
            Dictionary containing validation result
        """
        errors = []
        warnings = []
        
        # Check if agent is available
        if not agent.is_available:
            errors.append("Agent is not available for execution")
        
        # Check agent configuration
        if not agent.config:
            warnings.append("Agent has no configuration")
        
        # Check model configuration
        if not agent.model_config:
            warnings.append("Agent has no model configuration")
        
        # Validate agent type
        if agent.agent_type not in AgentType:
            errors.append(f"Invalid agent type: {agent.agent_type}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def get_agent_capabilities(self, agent: Agent) -> Dict[str, Any]:
        """
        Get agent capabilities and supported operations.
        
        Args:
            agent: The agent to query
            
        Returns:
            Dictionary containing agent capabilities
        """
        capabilities = {
            "agent_type": agent.agent_type.value,
            "supported_operations": [],
            "tools_available": [],
            "model_info": {}
        }
        
        # Add capabilities based on agent type
        if agent.agent_type == AgentType.FSM_REACT:
            capabilities["supported_operations"] = [
                "text_processing",
                "tool_execution",
                "state_management",
                "reasoning"
            ]
        elif agent.agent_type == AgentType.NEXT_GEN:
            capabilities["supported_operations"] = [
                "advanced_reasoning",
                "parallel_processing",
                "multi_modal_processing",
                "learning"
            ]
        elif agent.agent_type == AgentType.CREW:
            capabilities["supported_operations"] = [
                "multi_agent_coordination",
                "task_delegation",
                "collaborative_reasoning",
                "workflow_management"
            ]
        elif agent.agent_type == AgentType.SPECIALIZED:
            capabilities["supported_operations"] = [
                "domain_specific_processing",
                "expert_knowledge",
                "specialized_tools"
            ]
        
        return capabilities
    
    async def cancel_execution(self, execution_id: UUID) -> bool:
        """
        Cancel a running execution.
        
        Args:
            execution_id: The execution to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        if execution_id not in self._active_executions:
            return False
        
        execution = self._active_executions[execution_id]
        if execution["status"] == "running":
            execution["status"] = "cancelled"
            execution["end_time"] = datetime.now()
            self.logger.info("Execution {} cancelled", extra={"execution_id": execution_id})
            return True
        
        return False
    
    async def get_execution_status(self, execution_id: UUID) -> Dict[str, Any]:
        """
        Get the status of an execution.
        
        Args:
            execution_id: The execution to query
            
        Returns:
            Dictionary containing execution status
        """
        if execution_id not in self._active_executions:
            return {"error": "Execution not found"}
        
        execution = self._active_executions[execution_id]
        status = {
            "execution_id": str(execution_id),
            "status": execution["status"],
            "agent_id": str(execution["agent_id"]),
            "message_id": str(execution["message_id"]),
            "start_time": execution["start_time"].isoformat()
        }
        
        if "execution_time" in execution:
            status["execution_time"] = execution["execution_time"]
        
        if "error" in execution:
            status["error"] = execution["error"]
        
        return status
    
    async def _execute_fsm_react_agent(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """Execute an FSM React agent."""
        # This would integrate with the existing FSM agent implementation
        # For now, return a mock response
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "response": f"FSM React agent processed: {message.content}",
            "confidence": 0.85,
            "tools_used": ["text_processor", "reasoning_engine"],
            "metadata": {
                "agent_type": "fsm_react",
                "processing_steps": 3
            }
        }
    
    async def _execute_next_gen_agent(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """Execute a Next Gen agent."""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        return {
            "response": f"Next Gen agent processed: {message.content}",
            "confidence": 0.92,
            "tools_used": ["advanced_reasoning", "parallel_processor"],
            "metadata": {
                "agent_type": "next_gen",
                "processing_steps": 5
            }
        }
    
    async def _execute_crew_agent(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """Execute a Crew agent."""
        await asyncio.sleep(0.3)  # Simulate processing time
        
        return {
            "response": f"Crew agent processed: {message.content}",
            "confidence": 0.88,
            "tools_used": ["coordinator", "researcher", "executor"],
            "metadata": {
                "agent_type": "crew",
                "crew_size": 3,
                "processing_steps": 7
            }
        }
    
    async def _execute_specialized_agent(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """Execute a Specialized agent."""
        await asyncio.sleep(0.15)  # Simulate processing time
        
        return {
            "response": f"Specialized agent processed: {message.content}",
            "confidence": 0.95,
            "tools_used": ["domain_expert", "specialized_tool"],
            "metadata": {
                "agent_type": "specialized",
                "domain": "expert",
                "processing_steps": 2
            }
        }


# Alias for backward compatibility
AgentExecutor = AgentExecutorImpl 