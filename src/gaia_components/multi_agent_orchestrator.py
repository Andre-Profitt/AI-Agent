from agent import messages
from agent import query
from benchmarks.cot_performance import duration
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import results
from fix_security_issues import content
from tests.load_test import success

from src.agents.crew_enhanced import orchestrator
from src.api_server import message
from src.database.models import agent_id
from src.database.models import agent_type
from src.database.models import parameters
from src.database.models import priority
from src.database.supabase_manager import message_id
from src.database_extended import performance
from src.database_extended import success_rate
from src.gaia_components.advanced_reasoning_engine import overlap
from src.gaia_components.multi_agent_orchestrator import agent_scores
from src.gaia_components.multi_agent_orchestrator import agent_stats
from src.gaia_components.multi_agent_orchestrator import analysis_task
from src.gaia_components.multi_agent_orchestrator import available_agents
from src.gaia_components.multi_agent_orchestrator import avg_response_time
from src.gaia_components.multi_agent_orchestrator import best_agent_id
from src.gaia_components.multi_agent_orchestrator import capability_match
from src.gaia_components.multi_agent_orchestrator import capability_words
from src.gaia_components.multi_agent_orchestrator import completed_tasks
from src.gaia_components.multi_agent_orchestrator import current_load
from src.gaia_components.multi_agent_orchestrator import default_agents
from src.gaia_components.multi_agent_orchestrator import dependencies_satisfied
from src.gaia_components.multi_agent_orchestrator import failed_tasks
from src.gaia_components.multi_agent_orchestrator import gaia_task_id
from src.gaia_components.multi_agent_orchestrator import load_score
from src.gaia_components.multi_agent_orchestrator import perf
from src.gaia_components.multi_agent_orchestrator import performance_score
from src.gaia_components.multi_agent_orchestrator import ready_tasks
from src.gaia_components.multi_agent_orchestrator import research_task
from src.gaia_components.multi_agent_orchestrator import suitable_agents
from src.gaia_components.multi_agent_orchestrator import synthesis_task
from src.gaia_components.multi_agent_orchestrator import task_id
from src.gaia_components.multi_agent_orchestrator import task_stats
from src.gaia_components.multi_agent_orchestrator import task_words
from src.gaia_components.multi_agent_orchestrator import total_tasks
from src.gaia_components.multi_agent_orchestrator import validation_task
from src.gaia_components.multi_agent_orchestrator import workflow_id
from src.gaia_components.multi_agent_orchestrator import workflow_tasks
from src.infrastructure.di.container import dependencies
from src.meta_cognition import score
from src.tools_introspection import description
from src.tools_introspection import error

"""

from collections import deque
from dataclasses import field
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
Multi-Agent Orchestrator for GAIA-enhanced FSMReActAgent
Implements sophisticated multi-agent coordination and specialized agent management
"""

from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import logging
import time
import uuid
# TODO: Fix undefined variables: agent_id, agent_scores, agent_stats, agent_type, analysis_task, available_agents, avg_response_time, best_agent_id, capability, capability_match, capability_words, completed_tasks, content, context, current_load, default_agents, dep_id, dependencies, dependencies_satisfied, description, duration, error, failed_tasks, gaia_task_id, load_score, m, max_agents, message, message_id, message_type, message_types, messages, orchestrator, overlap, parameters, perf, performance, performance_score, priority, query, ready_tasks, recipient_id, research_task, result, results, rule_func, score, sender_id, start_time, success, success_rate, suitable_agents, synthesis_task, task, task_id, task_stats, task_words, tasks, total_tasks, validation_task, workflow_id, workflow_tasks, x
from tests.test_gaia_agent import agent
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from math import e
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
import logging
import time
import uuid


# Avoid circular imports
if TYPE_CHECKING:
    from src.agents.advanced_agent_fsm import Agent, FSMReActAgent

logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    """Types of specialized agents"""
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    ANALYZER = "analyzer"
    CREATOR = "creator"
    OPTIMIZER = "optimizer"

class AgentStatus(str, Enum):
    """Agent status indicators"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"

class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskStatus(str, Enum):
    """Task status indicators"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a task to be executed by an agent"""
    id: str
    description: str
    agent_type: AgentType
    priority: TaskPriority
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assign(self, agent_id: str):
        """Assign task to an agent"""
        self.status = TaskStatus.ASSIGNED
        self.assigned_at = datetime.now()
        self.metadata['assigned_agent'] = agent_id

    def start(self):
        """Start task execution"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self, result: Any):
        """Complete task successfully"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def fail(self, error: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

@dataclass
class Agent:
    """Represents a specialized agent"""
    id: str
    name: str
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None  # Task ID
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None

    def assign_task(self, task_id: str):
        """Assign a task to this agent"""
        self.current_task = task_id
        self.status = AgentStatus.BUSY
        self.last_activity = datetime.now()

    def complete_task(self):
        """Complete current task"""
        self.current_task = None
        self.status = AgentStatus.IDLE
        self.last_activity = datetime.now()

    def fail_task(self):
        """Mark current task as failed"""
        self.current_task = None
        self.status = AgentStatus.ERROR
        self.last_activity = datetime.now()

class TaskScheduler:
    """Intelligent task scheduling and load balancing"""

    def __init__(self):
        self.agent_loads: Dict[str, int] = defaultdict(int)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.task_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def assign_task(self, task: Task, available_agents: List[Agent]) -> Optional[str]:
        """Assign task to the best available agent"""
        if not available_agents:
            return None

        # Filter agents by type
        suitable_agents = [
            agent for agent in available_agents
            if agent.agent_type == task.agent_type and agent.status == AgentStatus.IDLE
        ]

        if not suitable_agents:
            return None

        # Score agents based on multiple factors
        agent_scores = []
        for agent in suitable_agents:
            score = self._calculate_agent_score(agent, task)
            agent_scores.append((agent.id, score))

        # Select agent with highest score
        if agent_scores:
            best_agent_id = max(agent_scores, key=lambda x: x[1])[0]
            self.agent_loads[best_agent_id] += 1
            return best_agent_id

        return None

    def _calculate_agent_score(self, agent: Agent, task: Task) -> float:
        """Calculate score for agent-task assignment"""
        score = 0.0

        # 1. Load factor (prefer less loaded agents)
        current_load = self.agent_loads.get(agent.id, 0)
        load_score = max(0.0, 1.0 - (current_load / 10.0))  # Normalize to 0-1
        score += load_score * 0.3

        # 2. Performance factor
        performance = self.agent_performance.get(agent.id, {})
        success_rate = performance.get('success_rate', 0.5)
        avg_response_time = performance.get('avg_response_time', 10.0)

        performance_score = success_rate * (1.0 - min(avg_response_time / 60.0, 1.0))
        score += performance_score * 0.4

        # 3. Capability match
        capability_match = 0.0
        if task.description:
            # Simple keyword matching (in practice, use semantic similarity)
            task_words = set(task.description.lower().split())
            capability_words = set()
            for capability in agent.capabilities:
                capability_words.update(capability.lower().split())

            if capability_words:
                overlap = len(task_words.intersection(capability_words))
                capability_match = min(overlap / len(task_words), 1.0)

        score += capability_match * 0.3

        return score

    def record_task_completion(self, agent_id: str, task_id: str, success: bool,
                             duration: float):
        """Record task completion for performance tracking"""
        # Update agent load
        self.agent_loads[agent_id] = max(0, self.agent_loads.get(agent_id, 0) - 1)

        # Update performance metrics
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'total_duration': 0.0
            }

        perf = self.agent_performance[agent_id]
        perf['total_tasks'] += 1
        perf['total_duration'] += duration

        if success:
            perf['successful_tasks'] += 1

        # Calculate derived metrics
        perf['success_rate'] = perf['successful_tasks'] / perf['total_tasks']
        perf['avg_response_time'] = perf['total_duration'] / perf['total_tasks']

        # Record in task history
        self.task_history[agent_id].append({
            'task_id': task_id,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only recent history
        if len(self.task_history[agent_id]) > 100:
            self.task_history[agent_id] = self.task_history[agent_id][-100:]

class WorkflowEngine:
    """Manages complex workflows with dependencies"""

    def __init__(self):
        self.workflows: Dict[str, List[Task]] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.completed_tasks: Dict[str, Any] = {}

    def create_workflow(self, workflow_id: str, tasks: List[Task]) -> str:
        """Create a new workflow"""
        self.workflows[workflow_id] = tasks

        # Build dependency graph
        for task in tasks:
            for dep_id in task.dependencies:
                self.dependency_graph[dep_id].append(task.id)

        return workflow_id

    def get_ready_tasks(self, workflow_id: str) -> List[Task]:
        """Get tasks that are ready to execute (dependencies satisfied)"""
        if workflow_id not in self.workflows:
            return []

        ready_tasks = []
        for task in self.workflows[workflow_id]:
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_satisfied = all(
                    dep_id in self.completed_tasks
                    for dep_id in task.dependencies
                )

                if dependencies_satisfied:
                    ready_tasks.append(task)

        return ready_tasks

    def mark_task_completed(self, task_id: str, result: Any):
        """Mark a task as completed"""
        self.completed_tasks[task_id] = result

        # Update task status in all workflows
        for workflow_tasks in self.workflows.values():
            for task in workflow_tasks:
                if task.id == task_id:
                    task.complete(result)
                    break

    def is_workflow_completed(self, workflow_id: str) -> bool:
        """Check if a workflow is completed"""
        if workflow_id not in self.workflows:
            return False

        return all(
            task.status == TaskStatus.COMPLETED
            for task in self.workflows[workflow_id]
        )

    def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow progress information"""
        if workflow_id not in self.workflows:
            return {}

        tasks = self.workflows[workflow_id]
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in tasks if task.status == TaskStatus.FAILED)

        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'pending_tasks': total_tasks - completed_tasks - failed_tasks,
            'progress_percentage': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        }

class CommunicationHub:
    """Manages inter-agent communication and message routing"""

    def __init__(self):
        self.message_queue: deque = deque(maxlen=1000)
        self.agent_subscriptions: Dict[str, List[str]] = defaultdict(list)  # agent_id -> message_types
        self.message_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.routing_rules: Dict[str, Callable] = {}

    def send_message(self, sender_id: str, recipient_id: str, message_type: str,
                    content: Any, priority: TaskPriority = TaskPriority.MEDIUM):
        """Send a message between agents"""
        message = {
            'id': str(uuid.uuid4()),
            'sender_id': sender_id,
            'recipient_id': recipient_id,
            'message_type': message_type,
            'content': content,
            'priority': priority.value,
            'timestamp': datetime.now().isoformat(),
            'delivered': False
        }

        self.message_queue.append(message)
        self.message_history[sender_id].append(message)

        logger.debug(f"Message sent from {sender_id} to {recipient_id}: {message_type}")

    def broadcast_message(self, sender_id: str, message_type: str, content: Any,
                         priority: TaskPriority = TaskPriority.MEDIUM):
        """Broadcast message to all subscribed agents"""
        message = {
            'id': str(uuid.uuid4()),
            'sender_id': sender_id,
            'recipient_id': 'broadcast',
            'message_type': message_type,
            'content': content,
            'priority': priority.value,
            'timestamp': datetime.now().isoformat(),
            'delivered': False
        }

        self.message_queue.append(message)
        self.message_history[sender_id].append(message)

        logger.debug(f"Broadcast message from {sender_id}: {message_type}")

    def get_messages_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for a specific agent"""
        messages = []

        # Get direct messages
        for message in self.message_queue:
            if (message['recipient_id'] == agent_id or
                message['recipient_id'] == 'broadcast') and not message['delivered']:
                messages.append(message)

        # Sort by priority and timestamp
        messages.sort(key=lambda m: (
            TaskPriority(m['priority']).value,
            m['timestamp']
        ))

        return messages

    def mark_message_delivered(self, message_id: str):
        """Mark a message as delivered"""
        for message in self.message_queue:
            if message['id'] == message_id:
                message['delivered'] = True
                break

    def subscribe_agent(self, agent_id: str, message_types: List[str]):
        """Subscribe an agent to specific message types"""
        self.agent_subscriptions[agent_id].extend(message_types)

    def add_routing_rule(self, message_type: str, rule_func: Callable):
        """Add a custom routing rule"""
        self.routing_rules[message_type] = rule_func

class MultiAgentOrchestrator:
    """Main multi-agent orchestrator"""

    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_scheduler = TaskScheduler()
        self.workflow_engine = WorkflowEngine()
        self.communication_hub = CommunicationHub()

        # Performance tracking
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.avg_task_duration = 0.0

        logger.info("Multi-Agent Orchestrator initialized")

    def register_agent(self, agent: Agent) -> bool:
        """Register a new agent"""
        if len(self.agents) >= self.max_agents:
            logger.warning(f"Maximum number of agents ({self.max_agents}) reached")
            return False

        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        return True

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def create_task(self, description: str, agent_type: AgentType,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   parameters: Dict[str, Any] = None,
                   dependencies: List[str] = None) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            description=description,
            agent_type=agent_type,
            priority=priority,
            parameters=parameters or {},
            dependencies=dependencies or []
        )

        self.tasks[task_id] = task
        logger.info(f"Created task: {task_id} ({agent_type.value})")
        return task_id

    def assign_task(self, task_id: str) -> Optional[str]:
        """Assign a task to an available agent"""
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return None

        task = self.tasks[task_id]
        available_agents = list(self.agents.values())

        agent_id = self.task_scheduler.assign_task(task, available_agents)

        if agent_id:
            task.assign(agent_id)
            self.agents[agent_id].assign_task(task_id)
            logger.info(f"Assigned task {task_id} to agent {agent_id}")

        return agent_id

    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task (mock implementation)"""
        if task_id not in self.tasks:
            return {'success': False, 'error': 'Task not found'}

        task = self.tasks[task_id]
        start_time = time.time()

        try:
            task.start()

            # Mock task execution based on agent type
            result = self._mock_task_execution(task)

            task.complete(result)
            duration = time.time() - start_time

            # Update performance tracking
            self.total_tasks_processed += 1
            self.successful_tasks += 1
            self.avg_task_duration = (
                (self.avg_task_duration * (self.total_tasks_processed - 1) + duration) /
                self.total_tasks_processed
            )

            # Update scheduler
            if task.assigned_at:
                agent_id = task.metadata.get('assigned_agent')
                if agent_id:
                    self.task_scheduler.record_task_completion(
                        agent_id, task_id, True, duration
                    )
                    self.agents[agent_id].complete_task()

            # Update workflow engine
            self.workflow_engine.mark_task_completed(task_id, result)

            return {
                'success': True,
                'result': result,
                'duration': duration,
                'task_id': task_id
            }

        except Exception as e:
            task.fail(str(e))
            duration = time.time() - start_time

            # Update performance tracking
            self.total_tasks_processed += 1
            self.failed_tasks += 1

            # Update scheduler
            if task.assigned_at:
                agent_id = task.metadata.get('assigned_agent')
                if agent_id:
                    self.task_scheduler.record_task_completion(
                        agent_id, task_id, False, duration
                    )
                    self.agents[agent_id].fail_task()

            return {
                'success': False,
                'error': str(e),
                'duration': duration,
                'task_id': task_id
            }

    def _mock_task_execution(self, task: Task) -> Any:
        """Mock task execution based on agent type"""
        if task.agent_type == AgentType.RESEARCHER:
            return f"Research results for: {task.description}"
        elif task.agent_type == AgentType.EXECUTOR:
            return f"Execution completed for: {task.description}"
        elif task.agent_type == AgentType.SYNTHESIZER:
            return f"Synthesis result for: {task.description}"
        elif task.agent_type == AgentType.VALIDATOR:
            return f"Validation passed for: {task.description}"
        elif task.agent_type == AgentType.ANALYZER:
            return f"Analysis results for: {task.description}"
        elif task.agent_type == AgentType.CREATOR:
            return f"Created content for: {task.description}"
        elif task.agent_type == AgentType.OPTIMIZER:
            return f"Optimization results for: {task.description}"
        else:
            return f"Task completed: {task.description}"

    def create_workflow(self, tasks: List[Task]) -> str:
        """Create a workflow from a list of tasks"""
        workflow_id = str(uuid.uuid4())
        self.workflow_engine.create_workflow(workflow_id, tasks)

        # Add tasks to main task list
        for task in tasks:
            self.tasks[task.id] = task

        logger.info(f"Created workflow: {workflow_id} with {len(tasks)} tasks")
        return workflow_id

    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a complete workflow"""
        if workflow_id not in self.workflow_engine.workflows:
            return {'success': False, 'error': 'Workflow not found'}

        workflow_tasks = self.workflow_engine.workflows[workflow_id]
        results = {}

        while not self.workflow_engine.is_workflow_completed(workflow_id):
            # Get ready tasks
            ready_tasks = self.workflow_engine.get_ready_tasks(workflow_id)

            if not ready_tasks:
                # Check for deadlock or failed tasks
                failed_tasks = [
                    task for task in workflow_tasks
                    if task.status == TaskStatus.FAILED
                ]
                if failed_tasks:
                    return {
                        'success': False,
                        'error': f'Workflow failed due to {len(failed_tasks)} failed tasks',
                        'failed_tasks': [task.id for task in failed_tasks]
                    }
                break

            # Execute ready tasks
            for task in ready_tasks:
                result = self.execute_task(task.id)
                results[task.id] = result

                if not result['success']:
                    return {
                        'success': False,
                        'error': f'Task {task.id} failed: {result["error"]}',
                        'results': results
                    }

        return {
            'success': True,
            'workflow_id': workflow_id,
            'results': results,
            'progress': self.workflow_engine.get_workflow_progress(workflow_id)
        }

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        agent_stats = {}
        for agent in self.agents.values():
            agent_stats[agent.id] = {
                'name': agent.name,
                'type': agent.agent_type.value,
                'status': agent.status.value,
                'current_task': agent.current_task,
                'last_activity': agent.last_activity.isoformat() if agent.last_activity else None
            }

        task_stats = {
            'total_tasks': len(self.tasks),
            'pending_tasks': sum(1 for task in self.tasks.values() if task.status == TaskStatus.PENDING),
            'in_progress_tasks': sum(1 for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS),
            'completed_tasks': sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED),
            'failed_tasks': sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
        }

        return {
            'agents': agent_stats,
            'tasks': task_stats,
            'performance': {
                'total_tasks_processed': self.total_tasks_processed,
                'successful_tasks': self.successful_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': self.successful_tasks / max(self.total_tasks_processed, 1),
                'avg_task_duration': self.avg_task_duration
            },
            'workflows': {
                'total_workflows': len(self.workflow_engine.workflows),
                'active_workflows': sum(
                    1 for workflow_id in self.workflow_engine.workflows
                    if not self.workflow_engine.is_workflow_completed(workflow_id)
                )
            },
            'communication': {
                'pending_messages': len(self.communication_hub.message_queue),
                'total_agents_subscribed': len(self.communication_hub.agent_subscriptions)
            }
        }

    def create_default_agents(self):
        """Create default specialized agents"""
        default_agents = [
            Agent(
                id="researcher_1",
                name="Research Agent",
                agent_type=AgentType.RESEARCHER,
                capabilities=["web_search", "data_collection", "information_gathering"]
            ),
            Agent(
                id="executor_1",
                name="Execution Agent",
                agent_type=AgentType.EXECUTOR,
                capabilities=["task_execution", "tool_operation", "action_performance"]
            ),
            Agent(
                id="synthesizer_1",
                name="Synthesis Agent",
                agent_type=AgentType.SYNTHESIZER,
                capabilities=["information_synthesis", "report_generation", "insight_extraction"]
            ),
            Agent(
                id="validator_1",
                name="Validation Agent",
                agent_type=AgentType.VALIDATOR,
                capabilities=["fact_checking", "quality_assurance", "verification"]
            ),
            Agent(
                id="analyzer_1",
                name="Analysis Agent",
                agent_type=AgentType.ANALYZER,
                capabilities=["data_analysis", "pattern_recognition", "trend_identification"]
            )
        ]

        for agent in default_agents:
            self.register_agent(agent)

        logger.info(f"Created {len(default_agents)} default agents")

class MultiAgentGAIASystem:
    """GAIA-specific multi-agent system integration"""

    def __init__(self, orchestrator: MultiAgentOrchestrator):
        self.orchestrator = orchestrator
        self.gaia_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_mappings: Dict[str, List[str]] = defaultdict(list)  # GAIA task -> agent tasks

    def process_gaia_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a GAIA query using multi-agent coordination"""
        gaia_task_id = str(uuid.uuid4())
        self.gaia_tasks[gaia_task_id] = {
            'query': query,
            'context': context or {},
            'status': 'processing',
            'created_at': datetime.now(),
            'results': {}
        }

        # Create workflow for GAIA query processing
        workflow_tasks = self._create_gaia_workflow(query, context)
        workflow_id = self.orchestrator.create_workflow(workflow_tasks)

        # Execute workflow
        result = self.orchestrator.execute_workflow(workflow_id)

        # Update GAIA task
        self.gaia_tasks[gaia_task_id]['status'] = 'completed' if result['success'] else 'failed'
        self.gaia_tasks[gaia_task_id]['results'] = result

        return {
            'gaia_task_id': gaia_task_id,
            'workflow_id': workflow_id,
            'success': result['success'],
            'result': result.get('results', {}),
            'error': result.get('error')
        }

    def _create_gaia_workflow(self, query: str, context: Dict[str, Any]) -> List[Task]:
        """Create a workflow for processing a GAIA query"""
        tasks = []

        # Research phase
        research_task = Task(
            id=str(uuid.uuid4()),
            description=f"Research information for: {query}",
            agent_type=AgentType.RESEARCHER,
            priority=TaskPriority.HIGH,
            parameters={'query': query, 'context': context}
        )
        tasks.append(research_task)

        # Analysis phase (depends on research)
        analysis_task = Task(
            id=str(uuid.uuid4()),
            description=f"Analyze research results for: {query}",
            agent_type=AgentType.ANALYZER,
            priority=TaskPriority.HIGH,
            parameters={'query': query},
            dependencies=[research_task.id]
        )
        tasks.append(analysis_task)

        # Synthesis phase (depends on analysis)
        synthesis_task = Task(
            id=str(uuid.uuid4()),
            description=f"Synthesize final answer for: {query}",
            agent_type=AgentType.SYNTHESIZER,
            priority=TaskPriority.CRITICAL,
            parameters={'query': query},
            dependencies=[analysis_task.id]
        )
        tasks.append(synthesis_task)

        # Validation phase (depends on synthesis)
        validation_task = Task(
            id=str(uuid.uuid4()),
            description=f"Validate final answer for: {query}",
            agent_type=AgentType.VALIDATOR,
            priority=TaskPriority.HIGH,
            parameters={'query': query},
            dependencies=[synthesis_task.id]
        )
        tasks.append(validation_task)

        return tasks

    def get_gaia_task_status(self, gaia_task_id: str) -> Dict[str, Any]:
        """Get status of a GAIA task"""
        if gaia_task_id not in self.gaia_tasks:
            return {'error': 'GAIA task not found'}

        task = self.gaia_tasks[gaia_task_id]
        return {
            'gaia_task_id': gaia_task_id,
            'query': task['query'],
            'status': task['status'],
            'created_at': task['created_at'].isoformat(),
            'results': task['results']
        }