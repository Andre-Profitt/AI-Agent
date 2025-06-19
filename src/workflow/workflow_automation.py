# Workflow Templates & Automation System
# src/workflow/workflow_automation.py

import asyncio
import json
import yaml
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import networkx as nx
from jinja2 import Template
import logging

from src.agents.base import BaseAgent
from src.core.monitoring import MetricsCollector

logger = logging.getLogger(__name__)

# Workflow Models
class StepType(str, Enum):
    AGENT_TASK = "agent_task"
    PARALLEL_AGENTS = "parallel_agents"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    HUMAN_APPROVAL = "human_approval"
    WEBHOOK = "webhook"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"

class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowStep:
    """Single step in a workflow"""
    step_id: str
    name: str
    step_type: StepType
    agent_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None  # seconds
    condition: Optional[str] = None  # For conditional steps
    
@dataclass
class WorkflowTemplate:
    """Reusable workflow template"""
    template_id: str
    name: str
    description: str
    category: str
    steps: List[WorkflowStep]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
@dataclass
class WorkflowExecution:
    """Instance of a running workflow"""
    execution_id: str
    template_id: str
    status: WorkflowStatus
    context: Dict[str, Any]  # Workflow context/state
    step_results: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

# Workflow Engine
class WorkflowEngine:
    """Core workflow execution engine"""
    
    def __init__(
        self,
        agent_registry: Dict[str, BaseAgent],
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.agent_registry = agent_registry
        self.metrics = metrics_collector
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.step_handlers: Dict[StepType, Callable] = {
            StepType.AGENT_TASK: self._handle_agent_task,
            StepType.PARALLEL_AGENTS: self._handle_parallel_agents,
            StepType.CONDITIONAL: self._handle_conditional,
            StepType.LOOP: self._handle_loop,
            StepType.HUMAN_APPROVAL: self._handle_human_approval,
            StepType.WEBHOOK: self._handle_webhook,
            StepType.TRANSFORM: self._handle_transform,
            StepType.AGGREGATE: self._handle_aggregate
        }
        self._load_builtin_templates()
        
    def _load_builtin_templates(self):
        """Load built-in workflow templates"""
        # Research Report Template
        research_report = WorkflowTemplate(
            template_id="research_report",
            name="Research Report",
            description="Comprehensive research report with sources and analysis",
            category="research",
            steps=[
                WorkflowStep(
                    step_id="gather_sources",
                    name="Gather Sources",
                    step_type=StepType.AGENT_TASK,
                    agent_id="research_agent",
                    parameters={
                        "action": "search_and_collect",
                        "max_sources": 10
                    },
                    timeout=300
                ),
                WorkflowStep(
                    step_id="validate_sources",
                    name="Validate Sources",
                    step_type=StepType.AGENT_TASK,
                    agent_id="validator_agent",
                    parameters={
                        "action": "validate_credibility"
                    },
                    dependencies=["gather_sources"],
                    timeout=120
                ),
                WorkflowStep(
                    step_id="synthesize_findings",
                    name="Synthesize Findings",
                    step_type=StepType.AGENT_TASK,
                    agent_id="analysis_agent",
                    parameters={
                        "action": "synthesize",
                        "format": "structured"
                    },
                    dependencies=["validate_sources"],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="create_report",
                    name="Create Report",
                    step_type=StepType.AGENT_TASK,
                    agent_id="writer_agent",
                    parameters={
                        "action": "write_report",
                        "style": "academic"
                    },
                    dependencies=["synthesize_findings"],
                    timeout=600
                ),
                WorkflowStep(
                    step_id="quality_check",
                    name="Quality Check",
                    step_type=StepType.PARALLEL_AGENTS,
                    parameters={
                        "agents": ["review_agent", "fact_checker_agent"],
                        "aggregation": "all_pass"
                    },
                    dependencies=["create_report"],
                    timeout=300
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "scope": {"type": "string"},
                    "deadline": {"type": "string", "format": "date-time"}
                },
                "required": ["topic", "scope"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "report": {"type": "string"},
                    "sources": {"type": "array"},
                    "confidence_score": {"type": "number"}
                }
            }
        )
        self.templates[research_report.template_id] = research_report
        
        # Data Analysis Pipeline Template
        data_pipeline = WorkflowTemplate(
            template_id="data_analysis_pipeline",
            name="Data Analysis Pipeline",
            description="End-to-end data analysis with visualization",
            category="analytics",
            steps=[
                WorkflowStep(
                    step_id="data_ingestion",
                    name="Data Ingestion",
                    step_type=StepType.AGENT_TASK,
                    agent_id="data_agent",
                    parameters={
                        "action": "ingest",
                        "validation": True
                    },
                    timeout=180
                ),
                WorkflowStep(
                    step_id="data_cleaning",
                    name="Data Cleaning",
                    step_type=StepType.AGENT_TASK,
                    agent_id="data_agent",
                    parameters={
                        "action": "clean",
                        "strategies": ["remove_duplicates", "handle_missing", "normalize"]
                    },
                    dependencies=["data_ingestion"],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="exploratory_analysis",
                    name="Exploratory Analysis",
                    step_type=StepType.PARALLEL_AGENTS,
                    parameters={
                        "agents": ["stats_agent", "ml_agent"],
                        "tasks": {
                            "stats_agent": {"action": "descriptive_stats"},
                            "ml_agent": {"action": "feature_analysis"}
                        }
                    },
                    dependencies=["data_cleaning"],
                    timeout=600
                ),
                WorkflowStep(
                    step_id="generate_insights",
                    name="Generate Insights",
                    step_type=StepType.AGENT_TASK,
                    agent_id="insight_agent",
                    parameters={
                        "action": "analyze_patterns",
                        "depth": "comprehensive"
                    },
                    dependencies=["exploratory_analysis"],
                    timeout=400
                ),
                WorkflowStep(
                    step_id="create_visualizations",
                    name="Create Visualizations",
                    step_type=StepType.AGENT_TASK,
                    agent_id="viz_agent",
                    parameters={
                        "action": "create_dashboard",
                        "interactive": True
                    },
                    dependencies=["generate_insights"],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="human_review",
                    name="Human Review",
                    step_type=StepType.HUMAN_APPROVAL,
                    parameters={
                        "approval_type": "insights_validation",
                        "timeout": 3600
                    },
                    dependencies=["create_visualizations"]
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "data_source": {"type": "string"},
                    "analysis_goals": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["data_source"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "insights": {"type": "array"},
                    "visualizations": {"type": "array"},
                    "recommendations": {"type": "array"}
                }
            }
        )
        self.templates[data_pipeline.template_id] = data_pipeline
        
        # Content Creation Workflow
        content_workflow = WorkflowTemplate(
            template_id="content_creation",
            name="Content Creation Workflow",
            description="Multi-stage content creation with SEO optimization",
            category="content",
            steps=[
                WorkflowStep(
                    step_id="topic_research",
                    name="Topic Research",
                    step_type=StepType.AGENT_TASK,
                    agent_id="research_agent",
                    parameters={
                        "action": "analyze_topic",
                        "include_competitors": True
                    },
                    timeout=300
                ),
                WorkflowStep(
                    step_id="outline_creation",
                    name="Create Outline",
                    step_type=StepType.AGENT_TASK,
                    agent_id="planner_agent",
                    parameters={
                        "action": "create_outline",
                        "style": "detailed"
                    },
                    dependencies=["topic_research"],
                    timeout=180
                ),
                WorkflowStep(
                    step_id="content_drafting",
                    name="Draft Content",
                    step_type=StepType.LOOP,
                    parameters={
                        "iterations": 3,
                        "step": {
                            "agent_id": "writer_agent",
                            "action": "write_section",
                            "refinement": True
                        }
                    },
                    dependencies=["outline_creation"],
                    timeout=900
                ),
                WorkflowStep(
                    step_id="seo_optimization",
                    name="SEO Optimization",
                    step_type=StepType.AGENT_TASK,
                    agent_id="seo_agent",
                    parameters={
                        "action": "optimize",
                        "target_keywords": True
                    },
                    dependencies=["content_drafting"],
                    timeout=300
                ),
                WorkflowStep(
                    step_id="final_review",
                    name="Final Review",
                    step_type=StepType.CONDITIONAL,
                    condition: "context.content_type == 'article'",
                    parameters={
                        "true_step": {
                            "agent_id": "editor_agent",
                            "action": "editorial_review"
                        },
                        "false_step": {
                            "agent_id": "reviewer_agent",
                            "action": "basic_review"
                        }
                    },
                    dependencies=["seo_optimization"],
                    timeout=300
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "content_type": {"type": "string", "enum": ["article", "blog", "guide"]},
                    "target_audience": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["topic", "content_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                    "seo_score": {"type": "number"}
                }
            }
        )
        self.templates[content_workflow.template_id] = content_workflow
        
    # Template Management
    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template"""
        self.templates[template.template_id] = template
        logger.info(f"Registered workflow template: {template.name}")
        
    def load_template_from_yaml(self, yaml_path: Path) -> WorkflowTemplate:
        """Load workflow template from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        steps = []
        for step_data in data['steps']:
            step = WorkflowStep(
                step_id=step_data['id'],
                name=step_data['name'],
                step_type=StepType(step_data['type']),
                agent_id=step_data.get('agent_id'),
                parameters=step_data.get('parameters', {}),
                dependencies=step_data.get('dependencies', []),
                retry_config=step_data.get('retry', {}),
                timeout=step_data.get('timeout'),
                condition=step_data.get('condition')
            )
            steps.append(step)
            
        template = WorkflowTemplate(
            template_id=data['id'],
            name=data['name'],
            description=data['description'],
            category=data['category'],
            steps=steps,
            input_schema=data.get('input_schema', {}),
            output_schema=data.get('output_schema', {}),
            metadata=data.get('metadata', {}),
            version=data.get('version', '1.0.0')
        )
        
        self.register_template(template)
        return template
        
    # Workflow Execution
    async def execute_workflow(
        self,
        template_id: str,
        context: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> WorkflowExecution:
        """Execute a workflow from template"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
            
        template = self.templates[template_id]
        execution_id = execution_id or str(uuid.uuid4())
        
        # Validate input against schema
        # TODO: Add JSON schema validation
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            template_id=template_id,
            status=WorkflowStatus.RUNNING,
            context=context,
            started_at=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        
        try:
            # Build execution graph
            graph = self._build_execution_graph(template)
            
            # Execute workflow
            await self._execute_graph(graph, execution)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            
        # Record metrics
        if self.metrics:
            self.metrics.record_workflow_execution(
                template_id=template_id,
                duration=(execution.completed_at - execution.started_at).total_seconds(),
                status=execution.status.value
            )
            
        return execution
        
    def _build_execution_graph(self, template: WorkflowTemplate) -> nx.DiGraph:
        """Build directed graph for workflow execution"""
        graph = nx.DiGraph()
        
        for step in template.steps:
            graph.add_node(step.step_id, step=step)
            
            for dep in step.dependencies:
                graph.add_edge(dep, step.step_id)
                
        # Validate graph (check for cycles)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains cycles")
            
        return graph
        
    async def _execute_graph(self, graph: nx.DiGraph, execution: WorkflowExecution):
        """Execute workflow graph"""
        # Get topological order
        order = list(nx.topological_sort(graph))
        
        for step_id in order:
            step = graph.nodes[step_id]['step']
            execution.current_step = step_id
            
            try:
                # Check if dependencies completed successfully
                if not self._check_dependencies(step, execution):
                    execution.step_results[step_id] = {
                        "status": StepStatus.SKIPPED,
                        "reason": "Dependencies failed"
                    }
                    continue
                    
                # Execute step
                result = await self._execute_step(step, execution)
                execution.step_results[step_id] = result
                
                # Update context with result
                execution.context[f"step_{step_id}_result"] = result.get("output")
                
            except Exception as e:
                logger.error(f"Step {step_id} failed: {e}")
                execution.step_results[step_id] = {
                    "status": StepStatus.FAILED,
                    "error": str(e)
                }
                
                # Check if we should continue on failure
                if not step.retry_config.get("continue_on_failure", False):
                    raise
                    
    def _check_dependencies(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Check if all dependencies completed successfully"""
        for dep_id in step.dependencies:
            dep_result = execution.step_results.get(dep_id, {})
            if dep_result.get("status") != StepStatus.COMPLETED:
                return False
        return True
        
    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a single workflow step"""
        handler = self.step_handlers.get(step.step_type)
        if not handler:
            raise ValueError(f"Unknown step type: {step.step_type}")
            
        # Apply timeout if specified
        if step.timeout:
            return await asyncio.wait_for(
                handler(step, execution),
                timeout=step.timeout
            )
        else:
            return await handler(step, execution)
            
    # Step Handlers
    async def _handle_agent_task(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle agent task execution"""
        agent = self.agent_registry.get(step.agent_id)
        if not agent:
            raise ValueError(f"Agent {step.agent_id} not found")
            
        # Prepare input from context and parameters
        input_data = {
            **step.parameters,
            "context": execution.context
        }
        
        # Execute agent task
        start_time = datetime.utcnow()
        result = await agent.execute(input_data)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "status": StepStatus.COMPLETED,
            "output": result,
            "duration": duration,
            "agent_id": step.agent_id
        }
        
    async def _handle_parallel_agents(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle parallel agent execution"""
        agents = step.parameters.get("agents", [])
        tasks = step.parameters.get("tasks", {})
        aggregation = step.parameters.get("aggregation", "all")
        
        # Create tasks for parallel execution
        agent_tasks = []
        for agent_id in agents:
            agent = self.agent_registry.get(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
                
            task_params = tasks.get(agent_id, step.parameters)
            agent_tasks.append(
                agent.execute({
                    **task_params,
                    "context": execution.context
                })
            )
            
        # Execute in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results based on aggregation strategy
        outputs = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "agent_id": agents[i],
                    "error": str(result)
                })
            else:
                outputs.append({
                    "agent_id": agents[i],
                    "output": result
                })
                
        # Determine overall status
        if aggregation == "all" and errors:
            status = StepStatus.FAILED
        elif aggregation == "any" and outputs:
            status = StepStatus.COMPLETED
        else:
            status = StepStatus.COMPLETED if outputs else StepStatus.FAILED
            
        return {
            "status": status,
            "outputs": outputs,
            "errors": errors,
            "aggregation": aggregation
        }
        
    async def _handle_conditional(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle conditional execution"""
        condition = step.condition
        if not condition:
            raise ValueError("Conditional step requires a condition")
            
        # Evaluate condition using Jinja2
        template = Template(condition)
        condition_result = template.render(context=execution.context)
        
        # Convert to boolean
        is_true = condition_result.lower() in ['true', '1', 'yes']
        
        # Execute appropriate branch
        if is_true:
            branch_config = step.parameters.get("true_step", {})
        else:
            branch_config = step.parameters.get("false_step", {})
            
        if not branch_config:
            return {
                "status": StepStatus.SKIPPED,
                "condition_result": is_true,
                "reason": "No branch configured"
            }
            
        # Create temporary step for branch
        branch_step = WorkflowStep(
            step_id=f"{step.step_id}_branch",
            name=f"{step.name} Branch",
            step_type=StepType.AGENT_TASK,
            agent_id=branch_config.get("agent_id"),
            parameters=branch_config,
            timeout=step.timeout
        )
        
        result = await self._handle_agent_task(branch_step, execution)
        result["condition_result"] = is_true
        
        return result
        
    async def _handle_loop(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle loop execution"""
        iterations = step.parameters.get("iterations", 1)
        loop_step_config = step.parameters.get("step", {})
        
        outputs = []
        
        for i in range(iterations):
            # Update context with loop index
            execution.context["loop_index"] = i
            
            # Create temporary step for iteration
            loop_step = WorkflowStep(
                step_id=f"{step.step_id}_iter_{i}",
                name=f"{step.name} Iteration {i}",
                step_type=StepType.AGENT_TASK,
                agent_id=loop_step_config.get("agent_id"),
                parameters=loop_step_config,
                timeout=step.timeout
            )
            
            result = await self._handle_agent_task(loop_step, execution)
            outputs.append(result.get("output"))
            
            # Check for early termination
            if execution.context.get("break_loop", False):
                break
                
        return {
            "status": StepStatus.COMPLETED,
            "outputs": outputs,
            "iterations_completed": len(outputs)
        }
        
    async def _handle_human_approval(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle human approval step"""
        approval_type = step.parameters.get("approval_type", "generic")
        timeout = step.parameters.get("timeout", 3600)  # 1 hour default
        
        # Create approval request
        approval_id = str(uuid.uuid4())
        approval_request = {
            "id": approval_id,
            "execution_id": execution.execution_id,
            "step_id": step.step_id,
            "type": approval_type,
            "context": execution.context,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=timeout)
        }
        
        # Store approval request (in real implementation, this would go to a database)
        # For now, we'll simulate approval after a short delay
        await asyncio.sleep(2)  # Simulate waiting
        
        # In real implementation, this would wait for actual human approval
        approved = True  # Simulated approval
        
        return {
            "status": StepStatus.COMPLETED if approved else StepStatus.FAILED,
            "approval_id": approval_id,
            "approved": approved,
            "approver": "simulated_user",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def _handle_webhook(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle webhook call"""
        url = step.parameters.get("url")
        method = step.parameters.get("method", "POST")
        headers = step.parameters.get("headers", {})
        body_template = step.parameters.get("body", {})
        
        # Render body template with context
        if isinstance(body_template, str):
            template = Template(body_template)
            body = template.render(context=execution.context)
        else:
            body = json.dumps(body_template)
            
        # Make webhook call
        # In real implementation, use aiohttp or httpx
        # For now, simulate success
        await asyncio.sleep(1)
        
        return {
            "status": StepStatus.COMPLETED,
            "webhook_url": url,
            "response": {
                "status_code": 200,
                "body": {"success": True}
            }
        }
        
    async def _handle_transform(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle data transformation"""
        input_key = step.parameters.get("input_key", "previous_output")
        transform_type = step.parameters.get("transform_type", "custom")
        transform_config = step.parameters.get("config", {})
        
        # Get input data from context
        input_data = execution.context.get(input_key)
        
        # Apply transformation
        if transform_type == "json_path":
            # Extract using JSON path
            # In real implementation, use jsonpath library
            output = input_data  # Simplified
        elif transform_type == "template":
            # Apply Jinja2 template
            template = Template(transform_config.get("template", ""))
            output = template.render(data=input_data, context=execution.context)
        else:
            # Custom transformation
            output = input_data  # Simplified
            
        return {
            "status": StepStatus.COMPLETED,
            "output": output,
            "transform_type": transform_type
        }
        
    async def _handle_aggregate(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle data aggregation"""
        input_keys = step.parameters.get("input_keys", [])
        aggregation_type = step.parameters.get("type", "merge")
        
        # Collect inputs
        inputs = []
        for key in input_keys:
            value = execution.context.get(key)
            if value:
                inputs.append(value)
                
        # Apply aggregation
        if aggregation_type == "merge":
            output = {}
            for inp in inputs:
                if isinstance(inp, dict):
                    output.update(inp)
        elif aggregation_type == "concat":
            output = inputs
        elif aggregation_type == "sum":
            output = sum(inputs) if all(isinstance(x, (int, float)) for x in inputs) else None
        else:
            output = inputs
            
        return {
            "status": StepStatus.COMPLETED,
            "output": output,
            "aggregation_type": aggregation_type,
            "input_count": len(inputs)
        }

# Workflow API Extensions
def setup_workflow_routes(app, workflow_engine: WorkflowEngine):
    """Setup workflow routes on FastAPI app"""
    
    @app.get("/api/v1/workflows/templates")
    async def list_workflow_templates(category: Optional[str] = None):
        """List available workflow templates"""
        templates = workflow_engine.templates.values()
        if category:
            templates = [t for t in templates if t.category == category]
            
        return {
            "templates": [
                {
                    "id": t.template_id,
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "version": t.version,
                    "steps_count": len(t.steps)
                }
                for t in templates
            ]
        }
        
    @app.get("/api/v1/workflows/templates/{template_id}")
    async def get_workflow_template(template_id: str):
        """Get detailed workflow template"""
        template = workflow_engine.templates.get(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
            
        return {
            "template": {
                "id": template.template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "version": template.version,
                "steps": [
                    {
                        "id": s.step_id,
                        "name": s.name,
                        "type": s.step_type.value,
                        "agent_id": s.agent_id,
                        "dependencies": s.dependencies
                    }
                    for s in template.steps
                ],
                "input_schema": template.input_schema,
                "output_schema": template.output_schema
            }
        }
        
    @app.post("/api/v1/workflows/execute")
    async def execute_workflow(
        template_id: str,
        context: Dict[str, Any],
        execution_id: Optional[str] = None
    ):
        """Execute a workflow"""
        try:
            execution = await workflow_engine.execute_workflow(
                template_id=template_id,
                context=context,
                execution_id=execution_id
            )
            
            return {
                "execution": {
                    "id": execution.execution_id,
                    "template_id": execution.template_id,
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "current_step": execution.current_step,
                    "step_results": execution.step_results,
                    "error": execution.error
                }
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.get("/api/v1/workflows/executions/{execution_id}")
    async def get_workflow_execution(execution_id: str):
        """Get workflow execution status"""
        execution = workflow_engine.executions.get(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
            
        return {
            "execution": {
                "id": execution.execution_id,
                "template_id": execution.template_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "current_step": execution.current_step,
                "step_results": execution.step_results,
                "context": execution.context,
                "error": execution.error
            }
        }
        
    @app.post("/api/v1/workflows/templates/upload")
    async def upload_workflow_template(yaml_content: str):
        """Upload a custom workflow template"""
        try:
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                temp_path = Path(f.name)
                
            # Load template
            template = workflow_engine.load_template_from_yaml(temp_path)
            
            # Clean up
            temp_path.unlink()
            
            return {
                "message": "Template uploaded successfully",
                "template_id": template.template_id
            }
            
        except Exception as e:
            logger.error(f"Template upload failed: {e}")
            raise HTTPException(status_code=400, detail=str(e)) 