"""
Multi-Agent Platform API Server
FastAPI server providing REST APIs and WebSocket support for the unified architecture
"""

import asyncio
import json
import logging
import time
import uuid
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import aioredis
from fastapi import (
    APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, 
    Query, WebSocket, WebSocketDisconnect, WebSocketException, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from src.unified_architecture import (
    MultiAgentPlatform, Agent, Task, Resource, Conflict, 
    PerformanceMetrics, WorkflowDefinition, WorkflowExecution
)
from src.unified_architecture.platform import PlatformConfig
from src.unified_architecture.enhanced_platform import AgentMetadata
from src.core.entities.agent import AgentType
from src.application.agents.agent_factory import AgentFactory
from src.infrastructure.database.supabase_repositories import (
    SupabaseAgentRepository, SupabaseTaskRepository, 
    SupabaseSessionRepository, SupabaseToolRepository
)
from src.infrastructure.monitoring.metrics import MetricsCollector
from src.infrastructure.circuit_breaker.circuit_breaker import CircuitBreakerRegistry
from src.infrastructure.agents.concrete_agents import (
    FSMReactAgentImpl, NextGenAgentImpl, CrewAgentImpl, SpecializedAgentImpl
)
from src.infrastructure.di.container import get_container

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Redis connection for WebSocket management
redis_client: Optional[aioredis.Redis] = None

# Global platform instance
platform: Optional[MultiAgentPlatform] = None

# Metrics collector
metrics_collector = MetricsCollector()

# Circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()

# Pydantic models for API requests/responses
class AgentRegistrationRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Agent resources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentResponse(BaseModel):
    id: str
    name: str
    capabilities: List[str]
    resources: Dict[str, Any]
    metadata: Dict[str, Any]
    status: str
    created_at: datetime
    last_heartbeat: Optional[datetime] = None

class TaskSubmissionRequest(BaseModel):
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    priority: int = Field(default=1, ge=1, le=10, description="Task priority (1-10)")
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Required resources")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class TaskResponse(BaseModel):
    id: str
    title: str
    description: str
    priority: int
    required_capabilities: List[str]
    resources: Dict[str, Any]
    metadata: Dict[str, Any]
    status: str
    assigned_agent: Optional[str] = None
    created_at: datetime
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class ResourceRequest(BaseModel):
    name: str = Field(..., description="Resource name")
    type: str = Field(..., description="Resource type")
    capacity: Dict[str, Any] = Field(..., description="Resource capacity")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ResourceResponse(BaseModel):
    id: str
    name: str
    type: str
    capacity: Dict[str, Any]
    metadata: Dict[str, Any]
    status: str
    created_at: datetime

class ConflictReportRequest(BaseModel):
    agent_id: str = Field(..., description="Agent ID involved in conflict")
    task_id: str = Field(..., description="Task ID involved in conflict")
    conflict_type: str = Field(..., description="Type of conflict")
    description: str = Field(..., description="Conflict description")
    severity: str = Field(..., description="Conflict severity (low, medium, high, critical)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ConflictResponse(BaseModel):
    id: str
    agent_id: str
    task_id: str
    conflict_type: str
    description: str
    severity: str
    metadata: Dict[str, Any]
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None

class WorkflowDefinitionRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class WorkflowExecutionRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class PerformanceMetricsResponse(BaseModel):
    agent_id: str
    metrics: Dict[str, Any]
    timestamp: datetime

class SystemHealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    metrics: Dict[str, Any]
    timestamp: datetime

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Dependency injection
async def get_platform() -> MultiAgentPlatform:
    if platform is None:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    return platform

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token"""
    # In production, implement proper JWT verification
    token = credentials.credentials
    # For now, accept any non-empty token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.utcnow(),
            "subscriptions": set()
        }
        logger.info(f"WebSocket client {client_id} connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            del self.connection_metadata[websocket]
            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str, exclude: Optional[WebSocket] = None):
        disconnected = []
        for connection in self.active_connections:
            if connection != exclude:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "active_connections": len(self.active_connections),
            "connections": [
                {
                    "client_id": metadata.get("client_id"),
                    "connected_at": metadata.get("connected_at"),
                    "subscriptions": list(metadata.get("subscriptions", set()))
                }
                for metadata in self.connection_metadata.values()
            ]
        }

manager = ConnectionManager()

# API Routes
router = APIRouter(prefix="/api/v1")

@router.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """System health check endpoint"""
    try:
        components = {
            "platform": "healthy" if platform else "unhealthy",
            "redis": "healthy" if redis_client else "unhealthy",
            "database": "healthy"  # Add actual DB health check
        }
        
        metrics = {
            "active_agents": len(platform.agents) if platform else 0,
            "active_tasks": len(platform.tasks) if platform else 0,
            "active_connections": manager.get_connection_info()["active_connections"]
        }
        
        return SystemHealthResponse(
            status="healthy" if all(v == "healthy" for v in components.values()) else "degraded",
            components=components,
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

# Global variables
agent_factory = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with proper platform initialization"""
    global platform, redis_client, agent_factory
    
    # Initialize DI container
    container = get_container()
    
    # Initialize Redis connection
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        global redis_client
        redis_client = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize platform with configuration
    try:
        # Create platform configuration
        platform_config = PlatformConfig(
            max_concurrent_tasks=100,
            task_timeout=300,
            heartbeat_interval=30,
            cleanup_interval=3600,
            enable_marketplace=True,
            enable_dashboard=True,
            enable_performance_tracking=True,
            enable_conflict_resolution=True,
            storage_backend="redis" if redis_client else "memory",
            log_level="INFO"
        )
        
        # Create and initialize platform
        global platform
        platform = MultiAgentPlatform(platform_config)
        
        # Start platform services
        await platform.start()
        
        # Initialize agent factory
        global agent_factory
        agent_factory = AgentFactory()
        
        # Pre-create some default agents
        await _initialize_default_agents(platform, agent_factory)
        
        logger.info("Multi-Agent Platform initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize platform: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    try:
        # Shutdown platform
        if platform:
            await platform.stop()
            logger.info("Platform stopped")
        
        # Shutdown all agents
        if agent_factory:
            await agent_factory.shutdown_all()
            logger.info("All agents shut down")
        
        # Close Redis connection
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
            
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Agent Management
@router.post("/agents", response_model=AgentResponse)
async def register_agent(
    request: AgentRegistrationRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Register a new agent"""
    try:
        agent = Agent(
            id=str(uuid.uuid4()),
            name=request.name,
            capabilities=request.capabilities,
            resources=request.resources,
            metadata=request.metadata
        )
        
        platform.register_agent(agent)
        
        # Broadcast agent registration
        await manager.broadcast(
            WebSocketMessage(
                type="agent_registered",
                data={"agent_id": agent.id, "name": agent.name}
            ).json()
        )
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            capabilities=agent.capabilities,
            resources=agent.resources,
            metadata=agent.metadata,
            status=agent.status,
            created_at=datetime.utcnow(),
            last_heartbeat=None
        )
    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/agents", response_model=List[AgentResponse])
async def list_agents(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token),
    status_filter: Optional[str] = Query(None, description="Filter by agent status"),
    capability_filter: Optional[str] = Query(None, description="Filter by capability")
):
    """List all agents with optional filtering"""
    try:
        agents = platform.agents.values()
        
        if status_filter:
            agents = [a for a in agents if a.status == status_filter]
        
        if capability_filter:
            agents = [a for a in agents if capability_filter in a.capabilities]
        
        return [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                capabilities=agent.capabilities,
                resources=agent.resources,
                metadata=agent.metadata,
                status=agent.status,
                created_at=datetime.utcnow(),  # In production, store actual creation time
                last_heartbeat=None
            )
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Get agent details by ID"""
    try:
        agent = platform.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            capabilities=agent.capabilities,
            resources=agent.resources,
            metadata=agent.metadata,
            status=agent.status,
            created_at=datetime.utcnow(),
            last_heartbeat=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agents/{agent_id}")
async def deregister_agent(
    agent_id: str,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Deregister an agent"""
    try:
        if agent_id not in platform.agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        platform.deregister_agent(agent_id)
        
        # Broadcast agent deregistration
        await manager.broadcast(
            WebSocketMessage(
                type="agent_deregistered",
                data={"agent_id": agent_id}
            ).json()
        )
        
        return {"message": "Agent deregistered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deregister agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Task Management
@router.post("/tasks", response_model=TaskResponse)
async def submit_task(
    request: TaskSubmissionRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Submit a new task"""
    try:
        task = Task(
            id=str(uuid.uuid4()),
            title=request.title,
            description=request.description,
            priority=request.priority,
            required_capabilities=request.required_capabilities,
            resources=request.resources,
            metadata=request.metadata,
            deadline=request.deadline
        )
        
        platform.submit_task(task)
        
        # Broadcast task submission
        await manager.broadcast(
            WebSocketMessage(
                type="task_submitted",
                data={"task_id": task.id, "title": task.title}
            ).json()
        )
        
        return TaskResponse(
            id=task.id,
            title=task.title,
            description=task.description,
            priority=task.priority,
            required_capabilities=task.required_capabilities,
            resources=task.resources,
            metadata=task.metadata,
            status=task.status,
            assigned_agent=task.assigned_agent,
            created_at=datetime.utcnow(),
            deadline=task.deadline,
            completed_at=task.completed_at
        )
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token),
    status_filter: Optional[str] = Query(None, description="Filter by task status"),
    priority_filter: Optional[int] = Query(None, description="Filter by priority")
):
    """List all tasks with optional filtering"""
    try:
        tasks = platform.tasks.values()
        
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        if priority_filter:
            tasks = [t for t in tasks if t.priority == priority_filter]
        
        return [
            TaskResponse(
                id=task.id,
                title=task.title,
                description=task.description,
                priority=task.priority,
                required_capabilities=task.required_capabilities,
                resources=task.resources,
                metadata=task.metadata,
                status=task.status,
                assigned_agent=task.assigned_agent,
                created_at=datetime.utcnow(),
                deadline=task.deadline,
                completed_at=task.completed_at
            )
            for task in tasks
        ]
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Get task details by ID"""
    try:
        task = platform.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskResponse(
            id=task.id,
            title=task.title,
            description=task.description,
            priority=task.priority,
            required_capabilities=task.required_capabilities,
            resources=task.resources,
            metadata=task.metadata,
            status=task.status,
            assigned_agent=task.assigned_agent,
            created_at=datetime.utcnow(),
            deadline=task.deadline,
            completed_at=task.completed_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Resource Management
@router.post("/resources", response_model=ResourceResponse)
async def register_resource(
    request: ResourceRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Register a new resource"""
    try:
        resource = Resource(
            id=str(uuid.uuid4()),
            name=request.name,
            type=request.type,
            capacity=request.capacity,
            metadata=request.metadata
        )
        
        platform.register_resource(resource)
        
        return ResourceResponse(
            id=resource.id,
            name=resource.name,
            type=resource.type,
            capacity=resource.capacity,
            metadata=resource.metadata,
            status=resource.status,
            created_at=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Resource registration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/resources", response_model=List[ResourceResponse])
async def list_resources(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token),
    type_filter: Optional[str] = Query(None, description="Filter by resource type")
):
    """List all resources with optional filtering"""
    try:
        resources = platform.resources.values()
        
        if type_filter:
            resources = [r for r in resources if r.type == type_filter]
        
        return [
            ResourceResponse(
                id=resource.id,
                name=resource.name,
                type=resource.type,
                capacity=resource.capacity,
                metadata=resource.metadata,
                status=resource.status,
                created_at=datetime.utcnow()
            )
            for resource in resources
        ]
    except Exception as e:
        logger.error(f"Failed to list resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Conflict Management
@router.post("/conflicts", response_model=ConflictResponse)
async def report_conflict(
    request: ConflictReportRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Report a conflict"""
    try:
        conflict = Conflict(
            id=str(uuid.uuid4()),
            agent_id=request.agent_id,
            task_id=request.task_id,
            conflict_type=request.conflict_type,
            description=request.description,
            severity=request.severity,
            metadata=request.metadata
        )
        
        platform.report_conflict(conflict)
        
        # Broadcast conflict report
        await manager.broadcast(
            WebSocketMessage(
                type="conflict_reported",
                data={
                    "conflict_id": conflict.id,
                    "agent_id": conflict.agent_id,
                    "task_id": conflict.task_id,
                    "severity": conflict.severity
                }
            ).json()
        )
        
        return ConflictResponse(
            id=conflict.id,
            agent_id=conflict.agent_id,
            task_id=conflict.task_id,
            conflict_type=conflict.conflict_type,
            description=conflict.description,
            severity=conflict.severity,
            metadata=conflict.metadata,
            status=conflict.status,
            created_at=datetime.utcnow(),
            resolved_at=conflict.resolved_at
        )
    except Exception as e:
        logger.error(f"Conflict reporting failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/conflicts", response_model=List[ConflictResponse])
async def list_conflicts(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token),
    status_filter: Optional[str] = Query(None, description="Filter by conflict status"),
    severity_filter: Optional[str] = Query(None, description="Filter by severity")
):
    """List all conflicts with optional filtering"""
    try:
        conflicts = platform.conflicts.values()
        
        if status_filter:
            conflicts = [c for c in conflicts if c.status == status_filter]
        
        if severity_filter:
            conflicts = [c for c in conflicts if c.severity == severity_filter]
        
        return [
            ConflictResponse(
                id=conflict.id,
                agent_id=conflict.agent_id,
                task_id=conflict.task_id,
                conflict_type=conflict.conflict_type,
                description=conflict.description,
                severity=conflict.severity,
                metadata=conflict.metadata,
                status=conflict.status,
                created_at=datetime.utcnow(),
                resolved_at=conflict.resolved_at
            )
            for conflict in conflicts
        ]
    except Exception as e:
        logger.error(f"Failed to list conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Workflow Management
@router.post("/workflows", response_model=Dict[str, Any])
async def create_workflow(
    request: WorkflowDefinitionRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Create a new workflow definition"""
    try:
        workflow = WorkflowDefinition(
            id=str(uuid.uuid4()),
            name=request.name,
            description=request.description,
            steps=request.steps,
            metadata=request.metadata
        )
        
        platform.create_workflow(workflow)
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Workflow creation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/workflows/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecutionRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Execute a workflow"""
    try:
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            parameters=request.parameters,
            metadata=request.metadata
        )
        
        platform.execute_workflow(execution)
        
        return {
            "execution_id": execution.id,
            "workflow_id": workflow_id,
            "status": execution.status
        }
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Performance Monitoring
@router.get("/metrics/performance", response_model=List[PerformanceMetricsResponse])
async def get_performance_metrics(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    time_range: Optional[str] = Query("1h", description="Time range for metrics")
):
    """Get performance metrics"""
    try:
        metrics = []
        for agent in platform.agents.values():
            if agent_id and agent.id != agent_id:
                continue
            
            agent_metrics = platform.get_agent_metrics(agent.id)
            if agent_metrics:
                metrics.append(PerformanceMetricsResponse(
                    agent_id=agent.id,
                    metrics=agent_metrics,
                    timestamp=datetime.utcnow()
                ))
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle subscription requests
            if message.get("type") == "subscribe":
                subscriptions = message.get("subscriptions", [])
                manager.connection_metadata[websocket]["subscriptions"].update(subscriptions)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "subscriptions": list(subscriptions)
                    }),
                    websocket
                )
            
            # Handle unsubscribe requests
            elif message.get("type") == "unsubscribe":
                subscriptions = message.get("subscriptions", [])
                manager.connection_metadata[websocket]["subscriptions"].difference_update(subscriptions)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "unsubscription_confirmed",
                        "subscriptions": list(subscriptions)
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Dashboard endpoints
@router.get("/dashboard/summary")
async def get_dashboard_summary(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    """Get dashboard summary data"""
    try:
        return {
            "agents": {
                "total": len(platform.agents),
                "active": len([a for a in platform.agents.values() if a.status == "active"]),
                "idle": len([a for a in platform.agents.values() if a.status == "idle"]),
                "busy": len([a for a in platform.agents.values() if a.status == "busy"])
            },
            "tasks": {
                "total": len(platform.tasks),
                "pending": len([t for t in platform.tasks.values() if t.status == "pending"]),
                "in_progress": len([t for t in platform.tasks.values() if t.status == "in_progress"]),
                "completed": len([t for t in platform.tasks.values() if t.status == "completed"]),
                "failed": len([t for t in platform.tasks.values() if t.status == "failed"])
            },
            "resources": {
                "total": len(platform.resources),
                "available": len([r for r in platform.resources.values() if r.status == "available"]),
                "in_use": len([r for r in platform.resources.values() if r.status == "in_use"])
            },
            "conflicts": {
                "total": len(platform.conflicts),
                "open": len([c for c in platform.conflicts.values() if c.status == "open"]),
                "resolved": len([c for c in platform.conflicts.values() if c.status == "resolved"])
            },
            "websocket_connections": manager.get_connection_info()["active_connections"]
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/activity")
async def get_dashboard_activity(
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token),
    limit: int = Query(50, description="Number of recent activities to return")
):
    """Get recent activity for dashboard"""
    try:
        # In a real implementation, you would store and retrieve activity logs
        # For now, return a mock activity feed
        activities = []
        
        # Add recent agent activities
        for agent in list(platform.agents.values())[:10]:
            activities.append({
                "type": "agent_activity",
                "timestamp": datetime.utcnow(),
                "data": {
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "status": agent.status
                }
            })
        
        # Add recent task activities
        for task in list(platform.tasks.values())[:10]:
            activities.append({
                "type": "task_activity",
                "timestamp": datetime.utcnow(),
                "data": {
                    "task_id": task.id,
                    "task_title": task.title,
                    "status": task.status
                }
            })
        
        # Sort by timestamp and return limited results
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return activities[:limit]
    
    except Exception as e:
        logger.error(f"Failed to get dashboard activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global platform, redis_client
    
    # Initialize Redis connection
    try:
        redis_client = aioredis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize platform with proper configuration
    try:
        # Configure platform
        platform_config = PlatformConfig(
            max_concurrent_tasks=100,
            enable_marketplace=True,
            enable_dashboard=True,
            enable_performance_tracking=True,
            enable_conflict_resolution=True,
            storage_backend="redis" if redis_client else "memory",
            log_level="INFO"
        )
        
        # Create and initialize platform
        platform = MultiAgentPlatform(platform_config)
        await platform.start()
        
        # Register default agents
        await _register_default_agents(platform)
        
        logger.info("Multi-Agent Platform initialized and started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize platform: {e}")
        raise
    
    yield
    
    # Cleanup
    if platform:
        await platform.stop()
        logger.info("Platform stopped")
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

async def _register_default_agents(platform: MultiAgentPlatform):
    """Register default agents with the platform"""
    try:
        # Create and register FSM React Agent
        fsm_agent = FSMReactAgentImpl()
        await fsm_agent.initialize()
        await platform.register_agent(fsm_agent, fsm_agent.metadata)
        logger.info("FSM React Agent registered")
        
        # Create and register Next Gen Agent
        next_gen_agent = NextGenAgentImpl()
        await next_gen_agent.initialize()
        await platform.register_agent(next_gen_agent, next_gen_agent.metadata)
        logger.info("Next Gen Agent registered")
        
        # Create and register Crew Agent
        crew_agent = CrewAgentImpl()
        await crew_agent.initialize()
        await platform.register_agent(crew_agent, crew_agent.metadata)
        logger.info("Crew Agent registered")
        
        # Create and register specialized agents for common domains
        domains = ["data_analysis", "code_generation", "research", "creative"]
        for domain in domains:
            specialized_agent = SpecializedAgentImpl(domain)
            await specialized_agent.initialize()
            await platform.register_agent(specialized_agent, specialized_agent.metadata)
            logger.info(f"Specialized {domain} Agent registered")
        
        logger.info(f"Total agents registered: {len(platform.agents)}")
        
    except Exception as e:
        logger.error(f"Failed to register default agents: {e}")
        raise

# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Platform API",
    description="REST API and WebSocket server for the unified multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent Platform API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 