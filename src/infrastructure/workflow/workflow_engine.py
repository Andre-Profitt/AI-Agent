"""
Workflow Orchestration Engine
Provides workflow management using LangGraph for complex agent workflows
Enhanced with circuit breaker protection and FSM integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import circuit breaker protection
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    circuit_breaker
)

# Import FSM agent for integration
from src.agents.advanced_agent_fsm import FSMReActAgent, FSMState

logger = logging.getLogger(__name__)

# =============================
# Workflow Types
# =============================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class WorkflowType(Enum):
    """Types of workflows"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"

@dataclass
class WorkflowStep:
    """Represents a step in a workflow"""
    step_id: str
    name: str
    description: str
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_count: int = 3
    retry_delay: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Expression for conditional execution
    parallel: bool = False

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep]
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================
# Workflow State
# =============================

@dataclass
class WorkflowState:
    """State passed between workflow steps"""
    execution_id: str
    workflow_id: str
    current_step: str
    step_results: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_history: List[str] = field(default_factory=list)
    retry_count: Dict[str, int] = field(default_factory=dict)

# =============================
# Workflow Engine
# =============================

class WorkflowEngine:
    """Main workflow orchestration engine"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.agents: Dict[str, Any] = {}  # Agent registry
        self.tools: Dict[str, BaseTool] = {}  # Tool registry
        self._lock = asyncio.Lock()
        
    async def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Register a new workflow definition"""
        async with self._lock:
            self.workflows[workflow.workflow_id] = workflow
            logger.info("Registered workflow: {} ({})", extra={"workflow_name": workflow.name, "workflow_workflow_id": workflow.workflow_id})
            return True
            
    async def unregister_workflow(self, workflow_id: str) -> bool:
        """Unregister a workflow definition"""
        async with self._lock:
            if workflow_id in self.workflows:
                del self.workflows[workflow_id]
                logger.info("Unregistered workflow: {}", extra={"workflow_id": workflow_id})
                return True
            return False
            
    async def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent for workflow execution"""
        self.agents[agent_id] = agent
        
    async def register_tool(self, tool_name: str, tool: BaseTool) -> None:
        """Register a tool for workflow execution"""
        self.tools[tool_name] = tool
        
    @circuit_breaker("workflow_execution", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=120))
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any],
                             execution_id: Optional[str] = None) -> WorkflowExecution:
        """Execute a workflow with circuit breaker protection"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        execution_id = execution_id or str(uuid.uuid4())
        
        # Create execution instance
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            input_data=input_data
        )
        
        self.executions[execution_id] = execution
        
        try:
            # Update status to running
            execution.status = WorkflowStatus.RUNNING
            
            # Create workflow graph
            graph = await self._create_workflow_graph(workflow)
            
            # Execute workflow
            initial_state = WorkflowState(
                execution_id=execution_id,
                workflow_id=workflow_id,
                current_step="start",
                input_data=input_data
            )
            
            # Run the workflow
            final_state = await self._run_workflow(graph, initial_state, workflow)
            
            # Update execution with results
            execution.output_data = final_state.output_data
            execution.step_results = final_state.step_results
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            
            logger.info("Workflow execution completed with circuit breaker protection: {}", extra={"execution_id": execution_id})
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            logger.error("Workflow execution failed: {}, error: {}", extra={"execution_id": execution_id, "e": e})
            
        return execution
        
    async def _create_workflow_graph(self, workflow: WorkflowDefinition) -> StateGraph:
        """Create LangGraph state graph from workflow definition"""
        workflow_graph = StateGraph(WorkflowState)
        
        # Add nodes for each step
        for step in workflow.steps:
            workflow_graph.add_node(step.step_id, self._create_step_node(step))
            
        # Add edges based on workflow type
        if workflow.workflow_type == WorkflowType.SEQUENTIAL:
            await self._add_sequential_edges(workflow_graph, workflow.steps)
        elif workflow.workflow_type == WorkflowType.PARALLEL:
            await self._add_parallel_edges(workflow_graph, workflow.steps)
        elif workflow.workflow_type == WorkflowType.CONDITIONAL:
            await self._add_conditional_edges(workflow_graph, workflow.steps)
        elif workflow.workflow_type == WorkflowType.LOOP:
            await self._add_loop_edges(workflow_graph, workflow.steps)
            
        # Set entry point
        if workflow.steps:
            workflow_graph.set_entry_point(workflow.steps[0].step_id)
            
        return workflow_graph.compile()
        
    def _create_step_node(self, step: WorkflowStep) -> Callable:
        """Create a node function for a workflow step"""
        async def step_node(state: WorkflowState) -> WorkflowState:
            try:
                logger.info("Executing step: {} ({})", extra={"step_name": step.name, "step_step_id": step.step_id})
                
                # Update current step
                state.current_step = step.step_id
                state.step_history.append(step.step_id)
                
                # Prepare input for the step
                step_input = self._prepare_step_input(step, state)
                
                # Execute the step
                if step.agent_id:
                    result = await self._execute_agent_step(step, step_input)
                elif step.tool_name:
                    result = await self._execute_tool_step(step, step_input)
                else:
                    result = await self._execute_custom_step(step, step_input)
                    
                # Process output
                self._process_step_output(step, result, state)
                
                # Update step results
                state.step_results[step.step_id] = {
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error("Step execution failed: {}, error: {}", extra={"step_step_id": step.step_id, "e": e})
                
                # Handle retries
                retry_count = state.retry_count.get(step.step_id, 0)
                if retry_count < step.retry_count:
                    state.retry_count[step.step_id] = retry_count + 1
                    await asyncio.sleep(step.retry_delay * (retry_count + 1))
                    # Retry the step
                    return await step_node(state)
                else:
                    # Max retries exceeded
                    state.error = str(e)
                    state.step_results[step.step_id] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    
            return state
            
        return step_node
        
    def _prepare_step_input(self, step: WorkflowStep, state: WorkflowState) -> Dict[str, Any]:
        """Prepare input data for a workflow step"""
        step_input = {}
        
        for input_key, source_path in step.input_mapping.items():
            if source_path.startswith("input."):
                # Map from workflow input
                key = source_path.split(".", 1)[1]
                step_input[input_key] = state.input_data.get(key)
            elif source_path.startswith("output."):
                # Map from previous step output
                step_id, output_key = source_path.split(".", 2)[1:]
                if step_id in state.step_results:
                    step_input[input_key] = state.step_results[step_id].get("result", {}).get(output_key)
            else:
                # Direct value
                step_input[input_key] = source_path
                
        return step_input
        
    @circuit_breaker("agent_execution", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60))
    async def _execute_agent_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent step with circuit breaker protection"""
        agent_id = step.agent_id
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        
        # Handle FSM agent specifically
        if isinstance(agent, FSMReActAgent):
            # Use FSM agent's run method
            result = await agent.run(step_input.get('query', ''))
            return {'output': result, 'agent_type': 'fsm'}
        else:
            # Handle other agent types
            result = await agent.arun(step_input)
            return {'output': result, 'agent_type': 'standard'}
        
    @circuit_breaker("tool_execution", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60))
    async def _execute_tool_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool step with circuit breaker protection"""
        tool_name = step.tool_name
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        tool = self.tools[tool_name]
        
        try:
            # Execute tool with proper error handling
            result = await tool.ainvoke(step_input)
            return {'output': result, 'tool_name': tool_name, 'success': True}
        except Exception as e:
            logger.error("Tool execution failed: {}, error: {}", extra={"tool_name": tool_name, "e": e})
            return {'output': None, 'tool_name': tool_name, 'success': False, 'error': str(e)}
        
    async def _execute_custom_step(self, step: WorkflowStep, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom step (placeholder for custom logic)"""
        # This would be implemented based on custom step types
        return {
            "step_type": "custom",
            "input": step_input,
            "result": {"status": "completed"}
        }
        
    def _process_step_output(self, step: WorkflowStep, result: Dict[str, Any], state: WorkflowState):
        """Process output from a workflow step"""
        for output_key, target_path in step.output_mapping.items():
            if target_path.startswith("output."):
                # Map to workflow output
                key = target_path.split(".", 1)[1]
                state.output_data[key] = result.get(output_key)
            else:
                # Store in step results
                state.step_results[step.step_id][output_key] = result.get(output_key)
                
    async def _add_sequential_edges(self, graph: StateGraph, steps: List[WorkflowStep]):
        """Add edges for sequential workflow"""
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            graph.add_edge(current_step.step_id, next_step.step_id)
            
        # Add final edge to END
        if steps:
            graph.add_edge(steps[-1].step_id, END)
            
    async def _add_parallel_edges(self, graph: StateGraph, steps: List[WorkflowStep]):
        """Add edges for parallel workflow"""
        # All steps can run in parallel
        for step in steps:
            graph.add_edge(step.step_id, END)
            
    async def _add_conditional_edges(self, graph: StateGraph, steps: List[WorkflowStep]):
        """Add edges for conditional workflow"""
        # This would implement conditional routing based on step conditions
        for step in steps:
            if step.condition:
                # Add conditional edge
                graph.add_conditional_edges(
                    step.step_id,
                    self._create_condition_function(step.condition)
                )
            else:
                graph.add_edge(step.step_id, END)
                
    async def _add_loop_edges(self, graph: StateGraph, steps: List[WorkflowStep]):
        """Add edges for loop workflow"""
        # This would implement loop logic
        for step in steps:
            graph.add_edge(step.step_id, step.step_id)  # Loop back to same step
            
    def _create_condition_function(self, condition: str) -> Callable:
        """Create a condition function for conditional routing"""
        def condition_func(state: WorkflowState) -> str:
            # Simple condition evaluation (would be more sophisticated in practice)
            try:
                # Evaluate condition against state
                return "continue" if eval(condition, {"state": state}) else "end"
            except:
                return "end"
        return condition_func
        
    async def _run_workflow(self, graph: StateGraph, initial_state: WorkflowState,
                          workflow: WorkflowDefinition) -> WorkflowState:
        """Run the workflow graph"""
        # Execute the graph
        final_state = await graph.ainvoke(initial_state)
        
        # Check for timeout
        if workflow.timeout:
            execution_time = (datetime.now() - initial_state.start_time).total_seconds()
            if execution_time > workflow.timeout:
                raise TimeoutError(f"Workflow execution timed out after {workflow.timeout} seconds")
                
        return final_state
        
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get the status of a workflow execution"""
        return self.executions.get(execution_id)
        
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.CANCELLED
                execution.end_time = datetime.now()
                return True
        return False
        
    async def get_workflow_definitions(self) -> List[WorkflowDefinition]:
        """Get all workflow definitions"""
        return list(self.workflows.values())
        
    async def get_execution_history(self, workflow_id: Optional[str] = None) -> List[WorkflowExecution]:
        """Get execution history"""
        executions = list(self.executions.values())
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        return executions

# =============================
# Workflow Builder
# =============================

class WorkflowBuilder:
    """Builder for creating workflow definitions"""
    
    def __init__(self, name: str, description: str = ""):
        self.workflow_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.workflow_type = WorkflowType.SEQUENTIAL
        self.steps: List[WorkflowStep] = []
        self.input_schema: Dict[str, Any] = {}
        self.output_schema: Dict[str, Any] = {}
        self.timeout: Optional[float] = None
        self.max_retries: int = 3
        self.metadata: Dict[str, Any] = {}
        
    def set_type(self, workflow_type: WorkflowType) -> 'WorkflowBuilder':
        """Set workflow type"""
        self.workflow_type = workflow_type
        return self
        
    def add_step(self, step: WorkflowStep) -> 'WorkflowBuilder':
        """Add a step to the workflow"""
        self.steps.append(step)
        return self
        
    def add_agent_step(self, name: str, agent_id: str, description: str = "",
                      input_mapping: Optional[Dict[str, str]] = None,
                      output_mapping: Optional[Dict[str, str]] = None) -> 'WorkflowBuilder':
        """Add an agent step"""
        step = WorkflowStep(
            step_id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_id=agent_id,
            input_mapping=input_mapping or {},
            output_mapping=output_mapping or {}
        )
        return self.add_step(step)
        
    def add_tool_step(self, name: str, tool_name: str, description: str = "",
                     input_mapping: Optional[Dict[str, str]] = None,
                     output_mapping: Optional[Dict[str, str]] = None) -> 'WorkflowBuilder':
        """Add a tool step"""
        step = WorkflowStep(
            step_id=str(uuid.uuid4()),
            name=name,
            description=description,
            tool_name=tool_name,
            input_mapping=input_mapping or {},
            output_mapping=output_mapping or {}
        )
        return self.add_step(step)
        
    def set_input_schema(self, schema: Dict[str, Any]) -> 'WorkflowBuilder':
        """Set input schema"""
        self.input_schema = schema
        return self
        
    def set_output_schema(self, schema: Dict[str, Any]) -> 'WorkflowBuilder':
        """Set output schema"""
        self.output_schema = schema
        return self
        
    def set_timeout(self, timeout: float) -> 'WorkflowBuilder':
        """Set workflow timeout"""
        self.timeout = timeout
        return self
        
    def set_max_retries(self, max_retries: int) -> 'WorkflowBuilder':
        """Set maximum retries"""
        self.max_retries = max_retries
        return self
        
    def add_metadata(self, key: str, value: Any) -> 'WorkflowBuilder':
        """Add metadata"""
        self.metadata[key] = value
        return self
        
    def build(self) -> WorkflowDefinition:
        """Build the workflow definition"""
        return WorkflowDefinition(
            workflow_id=self.workflow_id,
            name=self.name,
            description=self.description,
            workflow_type=self.workflow_type,
            steps=self.steps,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            timeout=self.timeout,
            max_retries=self.max_retries,
            metadata=self.metadata
        )

# =============================
# Global Workflow Engine
# =============================

# Global workflow engine instance
workflow_engine = WorkflowEngine()

# =============================
# Utility Functions
# =============================

async def register_workflow(workflow: WorkflowDefinition) -> bool:
    """Register a workflow with the global engine"""
    return await workflow_engine.register_workflow(workflow)

async def execute_workflow(workflow_id: str, input_data: Dict[str, Any],
                          execution_id: Optional[str] = None) -> WorkflowExecution:
    """Execute a workflow using the global engine"""
    return await workflow_engine.execute_workflow(workflow_id, input_data, execution_id)

async def get_execution_status(execution_id: str) -> Optional[WorkflowExecution]:
    """Get execution status from the global engine"""
    return await workflow_engine.get_execution_status(execution_id)

def create_workflow_builder(name: str, description: str = "") -> WorkflowBuilder:
    """Create a new workflow builder"""
    return WorkflowBuilder(name, description)

# =============================
# Agent Orchestrator for FSM Integration
# =============================

class AgentOrchestrator:
    """Orchestrator that integrates with existing FSM agent for workflow management"""
    
    def __init__(self, workflow_engine: WorkflowEngine = None):
        self.workflow_engine = workflow_engine or WorkflowEngine()
        self.fsm_agents: Dict[str, FSMReActAgent] = {}
        self.workflow_definitions: Dict[str, Dict[str, Any]] = {}
        
    async def register_fsm_agent(self, agent_id: str, agent: FSMReActAgent) -> None:
        """Register an FSM agent for orchestration"""
        self.fsm_agents[agent_id] = agent
        await self.workflow_engine.register_agent(agent_id, agent)
        logger.info("Registered FSM agent: {}", extra={"agent_id": agent_id})
    
    async def create_workflow_from_fsm(self, workflow_id: str, fsm_agent: FSMReActAgent,
                                     workflow_steps: List[Dict[str, Any]]) -> str:
        """Create a workflow definition from FSM agent and steps"""
        
        # Convert steps to WorkflowStep objects
        steps = []
        for i, step_data in enumerate(workflow_steps):
            step = WorkflowStep(
                step_id=f"step_{i}",
                name=step_data.get('name', f'Step {i}'),
                description=step_data.get('description', ''),
                agent_id=step_data.get('agent_id', 'fsm_agent'),
                tool_name=step_data.get('tool_name'),
                input_mapping=step_data.get('input_mapping', {}),
                output_mapping=step_data.get('output_mapping', {}),
                timeout=step_data.get('timeout'),
                retry_count=step_data.get('retry_count', 3),
                dependencies=step_data.get('dependencies', [])
            )
            steps.append(step)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=f"FSM Workflow {workflow_id}",
            description="Workflow created from FSM agent",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=steps,
            timeout=300,  # 5 minutes default
            max_retries=3
        )
        
        # Register workflow
        await self.workflow_engine.register_workflow(workflow)
        
        # Store workflow definition
        self.workflow_definitions[workflow_id] = {
            'workflow': workflow,
            'fsm_agent': fsm_agent,
            'steps': workflow_steps
        }
        
        logger.info("Created FSM workflow: {}", extra={"workflow_id": workflow_id})
        return workflow_id
    
    @circuit_breaker("orchestrator_execution", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=120))
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any],
                             execution_id: Optional[str] = None) -> WorkflowExecution:
        """Execute a workflow with circuit breaker protection"""
        return await self.workflow_engine.execute_workflow(workflow_id, input_data, execution_id)
    
    async def execute_fsm_workflow(self, query: str, workflow_steps: List[Dict[str, Any]],
                                 fsm_agent: FSMReActAgent = None) -> Dict[str, Any]:
        """Execute a workflow using FSM agent directly"""
        
        if fsm_agent is None:
            # Use default FSM agent if available
            if 'default' in self.fsm_agents:
                fsm_agent = self.fsm_agents['default']
            else:
                raise ValueError("No FSM agent provided and no default agent available")
        
        try:
            # Execute FSM agent with workflow steps
            result = await fsm_agent.run(query)
            
            # Process workflow steps if needed
            workflow_results = []
            for step in workflow_steps:
                step_result = await self._execute_workflow_step(step, result, fsm_agent)
                workflow_results.append(step_result)
            
            return {
                'fsm_result': result,
                'workflow_results': workflow_results,
                'success': True
            }
            
        except Exception as e:
            logger.error("FSM workflow execution failed: {}", extra={"e": e})
            return {
                'fsm_result': None,
                'workflow_results': [],
                'success': False,
                'error': str(e)
            }
    
    async def _execute_workflow_step(self, step: Dict[str, Any], fsm_result: Any,
                                   fsm_agent: FSMReActAgent) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_type = step.get('type', 'tool')
        
        if step_type == 'tool':
            # Execute tool using FSM agent's tools
            tool_name = step.get('tool_name')
            if tool_name and hasattr(fsm_agent, 'tools'):
                for tool in fsm_agent.tools:
                    if tool.name == tool_name:
                        try:
                            result = await tool.ainvoke(step.get('params', {}))
                            return {'step_type': 'tool', 'tool_name': tool_name, 'result': result}
                        except Exception as e:
                            return {'step_type': 'tool', 'tool_name': tool_name, 'error': str(e)}
        
        elif step_type == 'agent':
            # Execute sub-agent
            agent_id = step.get('agent_id')
            if agent_id in self.fsm_agents:
                try:
                    result = await self.fsm_agents[agent_id].run(step.get('query', ''))
                    return {'step_type': 'agent', 'agent_id': agent_id, 'result': result}
                except Exception as e:
                    return {'step_type': 'agent', 'agent_id': agent_id, 'error': str(e)}
        
        return {'step_type': step_type, 'error': 'Unknown step type'}
    
    async def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return await self.workflow_engine.get_execution_status(execution_id)
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow"""
        return await self.workflow_engine.cancel_execution(execution_id)
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow IDs"""
        return list(self.workflow_definitions.keys())
    
    def get_fsm_agents(self) -> List[str]:
        """Get list of registered FSM agent IDs"""
        return list(self.fsm_agents.keys()) 