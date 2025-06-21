from agent import workflow

from src.agents.enhanced_fsm import state
from src.api_server import execution
from src.core.entities.agent import Agent
from src.core.optimized_chain_of_thought import step
from src.core.optimized_chain_of_thought import steps
from src.gaia_components.multi_agent_orchestrator import execute_workflow
from src.infrastructure.workflow.workflow_engine import create_workflow_builder
from src.infrastructure.workflow.workflow_engine import workflow_engine
from src.unified_architecture.enhanced_platform import agent1
from src.unified_architecture.enhanced_platform import agent2

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
from src.infrastructure.workflow.workflow_engine import WorkflowBuilder
from src.infrastructure.workflow.workflow_engine import WorkflowDefinition
from src.infrastructure.workflow.workflow_engine import WorkflowEngine
from src.infrastructure.workflow.workflow_engine import WorkflowExecution
from src.infrastructure.workflow.workflow_engine import WorkflowState
from src.infrastructure.workflow.workflow_engine import WorkflowStatus
from src.infrastructure.workflow.workflow_engine import WorkflowStep
from src.infrastructure.workflow.workflow_engine import WorkflowType
from unittest.mock import Mock
# TODO: Fix undefined variables: agent1, agent2, all_executions, builder, call_count, definitions, error_agent, execute_workflow, execution, execution1, execution2, execution3, failing_agent, get_execution_status, input_schema, output_schema, register_workflow, result, slow_agent, state, step, step_id, steps, tool1, w, workflow, workflow1, workflow1_executions, workflow2, workflow_engine, workflow_ids, agent, agent1, agent2, all_executions, builder, call_count, create_workflow_builder, definitions, error_agent, execute_workflow, execution, execution1, execution2, execution3, failing_agent, get_execution_status, input_schema, output_schema, register_workflow, result, slow_agent, state, step, step_id, steps, tool1, w, workflow, workflow1, workflow1_executions, workflow2, workflow_engine, workflow_ids
from tests.test_gaia_agent import agent
from unittest.mock import AsyncMock

from src.infrastructure.workflow.workflow_engine import create_workflow_builder
from src.tools.base_tool import tool

# TODO: Fix undefined variables: agent1, agent2, all_executions, builder, call_count, definitions, error_agent, execute_workflow, execution, execution1, execution2, execution3, failing_agent, get_execution_status, input_schema, output_schema, register_workflow, result, slow_agent, state, step, step_id, steps, tool1, w, workflow, workflow1, workflow1_executions, workflow2, workflow_engine, workflow_ids, agent, agent1, agent2, all_executions, builder, call_count, create_workflow_builder, definitions, error_agent, execute_workflow, execution, execution1, execution2, execution3, failing_agent, get_execution_status, input_schema, output_schema, register_workflow, result, slow_agent, state, step, step_id, steps, tool1, w, workflow, workflow1, workflow1_executions, workflow2, workflow_engine, workflow_ids

"""
Unit tests for workflow orchestration engine
"""

import pytest
import asyncio

from datetime import datetime

from src.infrastructure.workflow.workflow_engine import (
from datetime import datetime
from math import e
from unittest.mock import AsyncMock

from src.tools.base_tool import tool

    WorkflowEngine,
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStep,
    WorkflowStatus,
    WorkflowType,
    WorkflowState,
    register_workflow,
    execute_workflow,
    get_execution_status,
    create_workflow_builder,
    workflow_engine
)

class TestWorkflowStep:
    """Test workflow step functionality"""

    def test_workflow_step_creation(self):
        """Test creating a workflow step"""
        step = WorkflowStep(
            step_id="step_1",
            name="Test Step",
            description="A test step",
            agent_id="agent_1",
            input_mapping={"input": "input.data"},
            output_mapping={"output": "output.result"},
            timeout=30.0,
            retry_count=3,
            retry_delay=1.0,
            dependencies=["step_0"],
            condition="state.step_results['step_0']['status'] == 'success'",
            parallel=False
        )

        assert step.step_id == "step_1"
        assert step.name == "Test Step"
        assert step.description == "A test step"
        assert step.agent_id == "agent_1"
        assert step.input_mapping == {"input": "input.data"}
        assert step.output_mapping == {"output": "output.result"}
        assert step.timeout == 30.0
        assert step.retry_count == 3
        assert step.retry_delay == 1.0
        assert step.dependencies == ["step_0"]
        assert step.condition == "state.step_results['step_0']['status'] == 'success'"
        assert step.parallel is False

class TestWorkflowDefinition:
    """Test workflow definition functionality"""

    def test_workflow_definition_creation(self):
        """Test creating a workflow definition"""
        steps = [
            WorkflowStep(
                step_id="step_1",
                name="Step 1",
                description="First step",
                agent_id="agent_1"
            ),
            WorkflowStep(
                step_id="step_2",
                name="Step 2",
                description="Second step",
                agent_id="agent_2"
            )
        ]

        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=steps,
            input_schema={"data": "string"},
            output_schema={"result": "string"},
            timeout=60.0,
            max_retries=3,
            metadata={"version": "1.0"}
        )

        assert workflow.workflow_id == "workflow_1"
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow"
        assert workflow.workflow_type == WorkflowType.SEQUENTIAL
        assert len(workflow.steps) == 2
        assert workflow.input_schema == {"data": "string"}
        assert workflow.output_schema == {"result": "string"}
        assert workflow.timeout == 60.0
        assert workflow.max_retries == 3
        assert workflow.metadata == {"version": "1.0"}

class TestWorkflowExecution:
    """Test workflow execution functionality"""

    def test_workflow_execution_creation(self):
        """Test creating a workflow execution"""
        execution = WorkflowExecution(
            execution_id="exec_1",
            workflow_id="workflow_1",
            status=WorkflowStatus.RUNNING,
            input_data={"data": "test"},
            output_data={"result": "success"},
            step_results={"step_1": {"status": "success"}},
            error_message=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            metadata={"duration": 10.0}
        )

        assert execution.execution_id == "exec_1"
        assert execution.workflow_id == "workflow_1"
        assert execution.status == WorkflowStatus.RUNNING
        assert execution.input_data == {"data": "test"}
        assert execution.output_data == {"result": "success"}
        assert execution.step_results == {"step_1": {"status": "success"}}
        assert execution.error_message is None
        assert execution.metadata == {"duration": 10.0}

class TestWorkflowState:
    """Test workflow state functionality"""

    def test_workflow_state_creation(self):
        """Test creating a workflow state"""
        state = WorkflowState(
            execution_id="exec_1",
            workflow_id="workflow_1",
            current_step="step_1",
            step_results={"step_1": {"status": "success"}},
            input_data={"data": "test"},
            output_data={"result": "success"},
            error=None,
            metadata={"progress": 50},
            step_history=["step_1"],
            retry_count={"step_1": 0}
        )

        assert state.execution_id == "exec_1"
        assert state.workflow_id == "workflow_1"
        assert state.current_step == "step_1"
        assert state.step_results == {"step_1": {"status": "success"}}
        assert state.input_data == {"data": "test"}
        assert state.output_data == {"result": "success"}
        assert state.error is None
        assert state.metadata == {"progress": 50}
        assert state.step_history == ["step_1"]
        assert state.retry_count == {"step_1": 0}

class TestWorkflowBuilder:
    """Test workflow builder functionality"""

    def test_workflow_builder_creation(self):
        """Test creating a workflow builder"""
        builder = WorkflowBuilder("Test Workflow", "A test workflow")

        assert builder.name == "Test Workflow"
        assert builder.description == "A test workflow"
        assert builder.workflow_type == WorkflowType.SEQUENTIAL
        assert len(builder.steps) == 0

    def test_set_type(self):
        """Test setting workflow type"""
        builder = WorkflowBuilder("Test Workflow")
        builder.set_type(WorkflowType.PARALLEL)

        assert builder.workflow_type == WorkflowType.PARALLEL

    def test_add_step(self):
        """Test adding a step"""
        builder = WorkflowBuilder("Test Workflow")
        step = WorkflowStep(
            step_id="step_1",
            name="Test Step",
            description="A test step",
            agent_id="agent_1"
        )

        builder.add_step(step)

        assert len(builder.steps) == 1
        assert builder.steps[0] == step

    def test_add_agent_step(self):
        """Test adding an agent step"""
        builder = WorkflowBuilder("Test Workflow")
        builder.add_agent_step(
            name="Agent Step",
            agent_id="agent_1",
            description="An agent step",
            input_mapping={"input": "input.data"},
            output_mapping={"output": "output.result"}
        )

        assert len(builder.steps) == 1
        step = builder.steps[0]
        assert step.name == "Agent Step"
        assert step.agent_id == "agent_1"
        assert step.description == "An agent step"
        assert step.input_mapping == {"input": "input.data"}
        assert step.output_mapping == {"output": "output.result"}

    def test_add_tool_step(self):
        """Test adding a tool step"""
        builder = WorkflowBuilder("Test Workflow")
        builder.add_tool_step(
            name="Tool Step",
            tool_name="test_tool",
            description="A tool step",
            input_mapping={"input": "input.data"},
            output_mapping={"output": "output.result"}
        )

        assert len(builder.steps) == 1
        step = builder.steps[0]
        assert step.name == "Tool Step"
        assert step.tool_name == "test_tool"
        assert step.description == "A tool step"

    def test_set_schemas(self):
        """Test setting input and output schemas"""
        builder = WorkflowBuilder("Test Workflow")
        input_schema = {"data": "string"}
        output_schema = {"result": "string"}

        builder.set_input_schema(input_schema)
        builder.set_output_schema(output_schema)

        assert builder.input_schema == input_schema
        assert builder.output_schema == output_schema

    def test_set_timeout_and_retries(self):
        """Test setting timeout and retries"""
        builder = WorkflowBuilder("Test Workflow")
        builder.set_timeout(60.0)
        builder.set_max_retries(5)

        assert builder.timeout == 60.0
        assert builder.max_retries == 5

    def test_add_metadata(self):
        """Test adding metadata"""
        builder = WorkflowBuilder("Test Workflow")
        builder.add_metadata("version", "1.0")
        builder.add_metadata("author", "test")

        assert builder.metadata["version"] == "1.0"
        assert builder.metadata["author"] == "test"

    def test_build(self):
        """Test building a workflow definition"""
        builder = WorkflowBuilder("Test Workflow", "A test workflow")
        builder.set_type(WorkflowType.SEQUENTIAL)
        builder.add_agent_step("Step 1", "agent_1")
        builder.add_tool_step("Step 2", "tool_1")
        builder.set_input_schema({"data": "string"})
        builder.set_output_schema({"result": "string"})
        builder.set_timeout(60.0)
        builder.set_max_retries(3)
        builder.add_metadata("version", "1.0")

        workflow = builder.build()

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow"
        assert workflow.workflow_type == WorkflowType.SEQUENTIAL
        assert len(workflow.steps) == 2
        assert workflow.input_schema == {"data": "string"}
        assert workflow.output_schema == {"result": "string"}
        assert workflow.timeout == 60.0
        assert workflow.max_retries == 3
        assert workflow.metadata["version"] == "1.0"

class TestWorkflowEngine:
    """Test workflow engine functionality"""

    @pytest.fixture
    def engine(self):
        """Create a workflow engine for testing"""
        return WorkflowEngine()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent"""
        agent = AsyncMock()
        agent.ainvoke.return_value.content = "Agent response"
        agent.ainvoke.return_value.additional_kwargs = {}
        return agent

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool"""
        tool = AsyncMock()
        tool.ainvoke.return_value = "Tool result"
        return tool

    @pytest.mark.asyncio
    async def test_register_workflow(self, engine):
        """Test registering a workflow"""
        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[],
            input_schema={},
            output_schema={}
        )

        result = await engine.register_workflow(workflow)

        assert result is True
        assert "workflow_1" in engine.workflows
        assert engine.workflows["workflow_1"] == workflow

    @pytest.mark.asyncio
    async def test_unregister_workflow(self, engine):
        """Test unregistering a workflow"""
        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)
        result = await engine.unregister_workflow("workflow_1")

        assert result is True
        assert "workflow_1" not in engine.workflows

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_workflow(self, engine):
        """Test unregistering a non-existent workflow"""
        result = await engine.unregister_workflow("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_register_agent(self, engine):
        """Test registering an agent"""
        agent = Mock()

        await engine.register_agent("agent_1", agent)

        assert "agent_1" in engine.agents
        assert engine.agents["agent_1"] == agent

    @pytest.mark.asyncio
    async def test_register_tool(self, engine):
        """Test registering a tool"""
        tool = Mock()

        await engine.register_tool("tool_1", tool)

        assert "tool_1" in engine.tools
        assert engine.tools["tool_1"] == tool

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, engine, mock_agent):
        """Test successful workflow execution"""
        # Register agent
        await engine.register_agent("agent_1", mock_agent)

        # Create workflow
        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Test Step",
                    description="A test step",
                    agent_id="agent_1",
                    input_mapping={"message": "input.data"},
                    output_mapping={"response": "output.result"}
                )
            ],
            input_schema={"data": "string"},
            output_schema={"result": "string"}
        )

        await engine.register_workflow(workflow)

        # Execute workflow
        execution = await engine.execute_workflow(
            "workflow_1",
            {"data": "test input"}
        )

        assert execution.workflow_id == "workflow_1"
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.input_data == {"data": "test input"}
        assert execution.output_data == {"result": "Agent response"}
        assert "step_1" in execution.step_results
        assert execution.step_results["step_1"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_workflow_with_tool(self, engine, mock_tool):
        """Test workflow execution with tool"""
        # Register tool
        await engine.register_tool("tool_1", mock_tool)

        # Create workflow
        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Tool Step",
                    description="A tool step",
                    tool_name="tool_1",
                    input_mapping={"input": "input.data"},
                    output_mapping={"result": "output.result"}
                )
            ],
            input_schema={"data": "string"},
            output_schema={"result": "string"}
        )

        await engine.register_workflow(workflow)

        # Execute workflow
        execution = await engine.execute_workflow(
            "workflow_1",
            {"data": "test input"}
        )

        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.output_data == {"result": "Tool result"}

    @pytest.mark.asyncio
    async def test_execute_nonexistent_workflow(self, engine):
        """Test executing a non-existent workflow"""
        with pytest.raises(ValueError, match="Workflow workflow_1 not found"):
            await engine.execute_workflow("workflow_1", {})

    @pytest.mark.asyncio
    async def test_execute_workflow_with_missing_agent(self, engine):
        """Test workflow execution with missing agent"""
        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Test Step",
                    description="A test step",
                    agent_id="missing_agent"
                )
            ],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("workflow_1", {})

        assert execution.status == WorkflowStatus.FAILED
        assert "Agent missing_agent not found" in execution.error_message

    @pytest.mark.asyncio
    async def test_execute_workflow_with_missing_tool(self, engine):
        """Test workflow execution with missing tool"""
        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Tool Step",
                    description="A tool step",
                    tool_name="missing_tool"
                )
            ],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("workflow_1", {})

        assert execution.status == WorkflowStatus.FAILED
        assert "Tool missing_tool not found" in execution.error_message

    @pytest.mark.asyncio
    async def test_execute_workflow_with_agent_error(self, engine):
        """Test workflow execution with agent error"""
        # Create agent that raises an error
        error_agent = AsyncMock()
        error_agent.ainvoke.side_effect = ValueError("Agent error")

        await engine.register_agent("error_agent", error_agent)

        workflow = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Error Step",
                    description="A step that will error",
                    agent_id="error_agent",
                    retry_count=1
                )
            ],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("workflow_1", {})

        assert execution.status == WorkflowStatus.FAILED
        assert "Agent error" in execution.error_message

    @pytest.mark.asyncio
    async def test_get_execution_status(self, engine):
        """Test getting execution status"""
        execution = WorkflowExecution(
            execution_id="exec_1",
            workflow_id="workflow_1",
            status=WorkflowStatus.COMPLETED,
            input_data={},
            output_data={}
        )

        engine.executions["exec_1"] = execution

        result = await engine.get_execution_status("exec_1")

        assert result == execution

    @pytest.mark.asyncio
    async def test_get_nonexistent_execution_status(self, engine):
        """Test getting non-existent execution status"""
        result = await engine.get_execution_status("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_execution(self, engine):
        """Test canceling an execution"""
        execution = WorkflowExecution(
            execution_id="exec_1",
            workflow_id="workflow_1",
            status=WorkflowStatus.RUNNING,
            input_data={},
            output_data={}
        )

        engine.executions["exec_1"] = execution

        result = await engine.cancel_execution("exec_1")

        assert result is True
        assert execution.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_execution(self, engine):
        """Test canceling a completed execution"""
        execution = WorkflowExecution(
            execution_id="exec_1",
            workflow_id="workflow_1",
            status=WorkflowStatus.COMPLETED,
            input_data={},
            output_data={}
        )

        engine.executions["exec_1"] = execution

        result = await engine.cancel_execution("exec_1")

        assert result is False
        assert execution.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_workflow_definitions(self, engine):
        """Test getting workflow definitions"""
        workflow1 = WorkflowDefinition(
            workflow_id="workflow_1",
            name="Workflow 1",
            description="First workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[],
            input_schema={},
            output_schema={}
        )

        workflow2 = WorkflowDefinition(
            workflow_id="workflow_2",
            name="Workflow 2",
            description="Second workflow",
            workflow_type=WorkflowType.PARALLEL,
            steps=[],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow1)
        await engine.register_workflow(workflow2)

        definitions = await engine.get_workflow_definitions()

        assert len(definitions) == 2
        workflow_ids = [w.workflow_id for w in definitions]
        assert "workflow_1" in workflow_ids
        assert "workflow_2" in workflow_ids

    @pytest.mark.asyncio
    async def test_get_execution_history(self, engine):
        """Test getting execution history"""
        execution1 = WorkflowExecution(
            execution_id="exec_1",
            workflow_id="workflow_1",
            status=WorkflowStatus.COMPLETED,
            input_data={},
            output_data={}
        )

        execution2 = WorkflowExecution(
            execution_id="exec_2",
            workflow_id="workflow_1",
            status=WorkflowStatus.FAILED,
            input_data={},
            output_data={}
        )

        execution3 = WorkflowExecution(
            execution_id="exec_3",
            workflow_id="workflow_2",
            status=WorkflowStatus.COMPLETED,
            input_data={},
            output_data={}
        )

        engine.executions["exec_1"] = execution1
        engine.executions["exec_2"] = execution2
        engine.executions["exec_3"] = execution3

        # Get all executions
        all_executions = await engine.get_execution_history()
        assert len(all_executions) == 3

        # Get executions for specific workflow
        workflow1_executions = await engine.get_execution_history("workflow_1")
        assert len(workflow1_executions) == 2
        assert all(e.workflow_id == "workflow_1" for e in workflow1_executions)

class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.asyncio
    async def test_register_workflow_global(self):
        """Test registering workflow with global engine"""
        workflow = WorkflowDefinition(
            workflow_id="global_workflow",
            name="Global Workflow",
            description="A global workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[],
            input_schema={},
            output_schema={}
        )

        result = await register_workflow(workflow)

        assert result is True
        assert "global_workflow" in workflow_engine.workflows

    @pytest.mark.asyncio
    async def test_execute_workflow_global(self):
        """Test executing workflow with global engine"""
        # Register a simple workflow
        workflow = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[],
            input_schema={},
            output_schema={}
        )

        await register_workflow(workflow)

        execution = await execute_workflow("test_workflow", {"data": "test"})

        assert execution.workflow_id == "test_workflow"
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.input_data == {"data": "test"}

    @pytest.mark.asyncio
    async def test_get_execution_status_global(self):
        """Test getting execution status with global engine"""
        execution = WorkflowExecution(
            execution_id="global_exec",
            workflow_id="workflow_1",
            status=WorkflowStatus.COMPLETED,
            input_data={},
            output_data={}
        )

        workflow_engine.executions["global_exec"] = execution

        result = await get_execution_status("global_exec")

        assert result == execution

    def test_create_workflow_builder(self):
        """Test creating workflow builder"""
        builder = create_workflow_builder("Test Builder", "A test builder")

        assert isinstance(builder, WorkflowBuilder)
        assert builder.name == "Test Builder"
        assert builder.description == "A test builder"

class TestWorkflowTypes:
    """Test different workflow types"""

    @pytest.mark.asyncio
    async def test_sequential_workflow(self):
        """Test sequential workflow execution"""
        engine = WorkflowEngine()

        # Create mock agents
        agent1 = AsyncMock()
        agent1.ainvoke.return_value.content = "Agent 1 response"
        agent1.ainvoke.return_value.additional_kwargs = {}

        agent2 = AsyncMock()
        agent2.ainvoke.return_value.content = "Agent 2 response"
        agent2.ainvoke.return_value.additional_kwargs = {}

        await engine.register_agent("agent_1", agent1)
        await engine.register_agent("agent_2", agent2)

        # Create sequential workflow
        workflow = WorkflowDefinition(
            workflow_id="sequential_workflow",
            name="Sequential Workflow",
            description="A sequential workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Step 1",
                    description="First step",
                    agent_id="agent_1"
                ),
                WorkflowStep(
                    step_id="step_2",
                    name="Step 2",
                    description="Second step",
                    agent_id="agent_2"
                )
            ],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("sequential_workflow", {})

        assert execution.status == WorkflowStatus.COMPLETED
        assert "step_1" in execution.step_results
        assert "step_2" in execution.step_results
        assert execution.step_results["step_1"]["status"] == "success"
        assert execution.step_results["step_2"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_parallel_workflow(self):
        """Test parallel workflow execution"""
        engine = WorkflowEngine()

        # Create mock agents
        agent1 = AsyncMock()
        agent1.ainvoke.return_value.content = "Agent 1 response"
        agent1.ainvoke.return_value.additional_kwargs = {}

        agent2 = AsyncMock()
        agent2.ainvoke.return_value.content = "Agent 2 response"
        agent2.ainvoke.return_value.additional_kwargs = {}

        await engine.register_agent("agent_1", agent1)
        await engine.register_agent("agent_2", agent2)

        # Create parallel workflow
        workflow = WorkflowDefinition(
            workflow_id="parallel_workflow",
            name="Parallel Workflow",
            description="A parallel workflow",
            workflow_type=WorkflowType.PARALLEL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Step 1",
                    description="First step",
                    agent_id="agent_1"
                ),
                WorkflowStep(
                    step_id="step_2",
                    name="Step 2",
                    description="Second step",
                    agent_id="agent_2"
                )
            ],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("parallel_workflow", {})

        assert execution.status == WorkflowStatus.COMPLETED
        assert "step_1" in execution.step_results
        assert "step_2" in execution.step_results

class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_workflow_timeout(self):
        """Test workflow timeout"""
        engine = WorkflowEngine()

        # Create slow agent
        slow_agent = AsyncMock()
        slow_agent.ainvoke.side_effect = lambda x: asyncio.sleep(2)

        await engine.register_agent("slow_agent", slow_agent)

        # Create workflow with short timeout
        workflow = WorkflowDefinition(
            workflow_id="timeout_workflow",
            name="Timeout Workflow",
            description="A workflow that will timeout",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Slow Step",
                    description="A slow step",
                    agent_id="slow_agent"
                )
            ],
            input_schema={},
            output_schema={},
            timeout=1.0  # 1 second timeout
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("timeout_workflow", {})

        assert execution.status == WorkflowStatus.FAILED
        assert "timeout" in execution.error_message.lower()

    @pytest.mark.asyncio
    async def test_step_retry_logic(self):
        """Test step retry logic"""
        engine = WorkflowEngine()

        # Create agent that fails twice then succeeds
        failing_agent = AsyncMock()
        call_count = 0

        async def failing_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Agent error {call_count}")
            return type('Response', (), {
                'content': 'Success after retries',
                'additional_kwargs': {}
            })()

        failing_agent.ainvoke.side_effect = failing_func

        await engine.register_agent("failing_agent", failing_agent)

        # Create workflow with retries
        workflow = WorkflowDefinition(
            workflow_id="retry_workflow",
            name="Retry Workflow",
            description="A workflow with retries",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Retry Step",
                    description="A step that will retry",
                    agent_id="failing_agent",
                    retry_count=3,
                    retry_delay=0.1
                )
            ],
            input_schema={},
            output_schema={}
        )

        await engine.register_workflow(workflow)

        execution = await engine.execute_workflow("retry_workflow", {})

        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.step_results["step_1"]["status"] == "success"
        assert call_count == 3  # Should have been called 3 times

class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_complex_workflow_integration(self):
        """Test complex workflow integration"""
        engine = WorkflowEngine()

        # Create multiple agents
        agent1 = AsyncMock()
        agent1.ainvoke.return_value.content = "Agent 1 result"
        agent1.ainvoke.return_value.additional_kwargs = {}

        agent2 = AsyncMock()
        agent2.ainvoke.return_value.content = "Agent 2 result"
        agent2.ainvoke.return_value.additional_kwargs = {}

        tool1 = AsyncMock()
        tool1.ainvoke.return_value = "Tool 1 result"

        await engine.register_agent("agent_1", agent1)
        await engine.register_agent("agent_2", agent2)
        await engine.register_tool("tool_1", tool1)

        # Create complex workflow
        workflow = WorkflowDefinition(
            workflow_id="complex_workflow",
            name="Complex Workflow",
            description="A complex workflow with multiple steps",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    name="Agent Step",
                    description="First agent step",
                    agent_id="agent_1",
                    input_mapping={"message": "input.data"},
                    output_mapping={"agent_result": "output.agent_result"}
                ),
                WorkflowStep(
                    step_id="step_2",
                    name="Tool Step",
                    description="Tool processing step",
                    tool_name="tool_1",
                    input_mapping={"input": "output.agent_result"},
                    output_mapping={"tool_result": "output.tool_result"}
                ),
                WorkflowStep(
                    step_id="step_3",
                    name="Final Agent Step",
                    description="Final agent step",
                    agent_id="agent_2",
                    input_mapping={"message": "output.tool_result"},
                    output_mapping={"final_result": "output.final_result"}
                )
            ],
            input_schema={"data": "string"},
            output_schema={"agent_result": "string", "tool_result": "string", "final_result": "string"}
        )

        await engine.register_workflow(workflow)

        # Execute workflow
        execution = await engine.execute_workflow(
            "complex_workflow",
            {"data": "initial input"}
        )

        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.output_data["agent_result"] == "Agent 1 result"
        assert execution.output_data["tool_result"] == "Tool 1 result"
        assert execution.output_data["final_result"] == "Agent 2 result"

        # Verify all steps were executed
        assert len(execution.step_results) == 3
        for step_id in ["step_1", "step_2", "step_3"]:
            assert step_id in execution.step_results
            assert execution.step_results[step_id]["status"] == "success"