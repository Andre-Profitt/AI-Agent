"""
Multi-Agent Platform API Server and Deployment Configuration
Complete FastAPI implementation with WebSocket support for real-time monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Set
import asyncio
import json
import uuid
import os
from datetime import datetime
import logging

# Import from the unified architecture and Enhanced FSM
from src.enhanced_fsm import HierarchicalFSM, AtomicState, ProbabilisticTransition
from src.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# Platform Core Classes
# =============================

class AgentCapability:
    """Agent capabilities enumeration"""
    REASONING = "REASONING"
    COLLABORATION = "COLLABORATION"
    EXECUTION = "EXECUTION"
    ANALYSIS = "ANALYSIS"
    SYNTHESIS = "SYNTHESIS"
    ERROR_HANDLING = "ERROR_HANDLING"

class AgentStatus:
    """Agent status enumeration"""
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    IDLE = "IDLE"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"

class AgentMetadata:
    """Agent metadata"""
    def __init__(self, agent_id: str, name: str, version: str, 
                 capabilities: List[str], tags: List[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.version = version
        self.capabilities = capabilities
        self.tags = tags or []
        self.status = AgentStatus.AVAILABLE
        self.reliability_score = 1.0
        self.last_seen = datetime.now()

class UnifiedTask:
    """Unified task representation"""
    def __init__(self, task_id: str, task_type: str, priority: int, 
                 payload: Dict[str, Any], required_capabilities: List[str],
                 deadline: Optional[datetime] = None, dependencies: List[str] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.payload = payload
        self.required_capabilities = required_capabilities
        self.deadline = deadline
        self.dependencies = dependencies or []
        self.status = "PENDING"
        self.created_at = datetime.now()
        self.completed_at = None
        self.result = None

class ConflictType:
    """Conflict types"""
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    TASK_CONFLICT = "TASK_CONFLICT"
    AGENT_CONFLICT = "AGENT_CONFLICT"
    DATA_CONFLICT = "DATA_CONFLICT"

class Conflict:
    """Conflict representation"""
    def __init__(self, conflict_id: str, conflict_type: str, involved_agents: List[str],
                 description: str, context: Dict[str, Any]):
        self.conflict_id = conflict_id
        self.conflict_type = conflict_type
        self.involved_agents = involved_agents
        self.description = description
        self.context = context
        self.reported_at = datetime.now()
        self.resolved = False
        self.resolution = None

class MarketplaceListing:
    """Marketplace listing"""
    def __init__(self, agent_id: str, metadata: AgentMetadata, description: str,
                 pricing: Dict[str, float], keywords: List[str]):
        self.agent_id = agent_id
        self.metadata = metadata
        self.description = description
        self.pricing = pricing
        self.keywords = keywords
        self.ratings = []
        self.total_usage = 0
        self.average_rating = 0.0
    
    def add_rating(self, rating: float, review: str, user_id: str):
        """Add a rating to the listing"""
        self.ratings.append({
            'rating': rating,
            'review': review,
            'user_id': user_id,
            'timestamp': datetime.now()
        })
        self.average_rating = sum(r['rating'] for r in self.ratings) / len(self.ratings)

class AgentRegistry:
    """Agent registry for managing agent registrations"""
    def __init__(self):
        self.agents: Dict[str, AgentMetadata] = {}
        self.agent_instances: Dict[str, MigratedEnhancedFSMAgent] = {}
    
    async def register(self, agent_id: str, metadata: AgentMetadata, 
                      agent_instance: MigratedEnhancedFSMAgent) -> bool:
        """Register an agent"""
        self.agents[agent_id] = metadata
        self.agent_instances[agent_id] = agent_instance
        return True
    
    async def unregister(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.agent_instances:
                del self.agent_instances[agent_id]
            return True
        return False
    
    async def discover(self, capabilities: List[str] = None, tags: List[str] = None,
                      status: str = None) -> List[AgentMetadata]:
        """Discover agents based on criteria"""
        agents = list(self.agents.values())
        
        if capabilities:
            agents = [a for a in agents if any(cap in a.capabilities for cap in capabilities)]
        
        if tags:
            agents = [a for a in agents if any(tag in a.tags for tag in tags)]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents

class TaskManager:
    """Task manager for handling task execution"""
    def __init__(self):
        self.tasks: Dict[str, UnifiedTask] = {}
        self.task_queue: List[str] = []
    
    async def submit_task(self, task: UnifiedTask) -> str:
        """Submit a task for execution"""
        self.tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        return task.task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result
        }

class ResourceManager:
    """Resource manager for tracking resource utilization"""
    def __init__(self):
        self.allocated_resources: Dict[str, Dict[str, float]] = {}
        self.total_resources = {
            "cpu_cores": 100.0,
            "memory_mb": 102400.0,
            "gpu_memory_mb": 51200.0
        }
    
    async def allocate_resources(self, agent_id: str, resources: Dict[str, float]) -> bool:
        """Allocate resources to an agent"""
        # Check if resources are available
        for resource, amount in resources.items():
            if resource in self.total_resources:
                allocated = sum(
                    agent_resources.get(resource, 0) 
                    for agent_resources in self.allocated_resources.values()
                )
                if allocated + amount > self.total_resources[resource]:
                    return False
        
        self.allocated_resources[agent_id] = resources
        return True
    
    async def release_resources(self, agent_id: str):
        """Release resources from an agent"""
        if agent_id in self.allocated_resources:
            del self.allocated_resources[agent_id]
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        utilization = {}
        
        for resource, total in self.total_resources.items():
            allocated = sum(
                agent_resources.get(resource, 0) 
                for agent_resources in self.allocated_resources.values()
            )
            utilization[resource] = (allocated / total) * 100
        
        return utilization

class Marketplace:
    """Marketplace for agent discovery and rating"""
    def __init__(self):
        self.listings: Dict[str, MarketplaceListing] = {}
    
    async def publish_agent(self, agent_id: str, description: str,
                           pricing: Dict[str, float], keywords: List[str]) -> bool:
        """Publish an agent to marketplace"""
        # This would typically fetch agent metadata from registry
        metadata = AgentMetadata(agent_id, f"Agent_{agent_id}", "1.0.0", 
                               [AgentCapability.REASONING], ["marketplace"])
        
        listing = MarketplaceListing(agent_id, metadata, description, pricing, keywords)
        self.listings[agent_id] = listing
        return True
    
    async def search_agents(self, query: str, min_rating: float = 0.0,
                           max_price: Optional[float] = None) -> List[MarketplaceListing]:
        """Search for agents in marketplace"""
        results = []
        
        for listing in self.listings.values():
            # Simple search implementation
            if (query.lower() in listing.description.lower() or
                query.lower() in listing.metadata.name.lower() or
                any(query.lower() in keyword.lower() for keyword in listing.keywords)):
                
                if listing.average_rating >= min_rating:
                    if max_price is None or listing.pricing.get("per_task", 0) <= max_price:
                        results.append(listing)
        
        return results

class ConflictResolver:
    """Conflict resolution system"""
    def __init__(self):
        self.conflicts: Dict[str, Conflict] = {}
    
    async def report_conflict(self, conflict: Conflict) -> str:
        """Report a conflict for resolution"""
        self.conflicts[conflict.conflict_id] = conflict
        
        # Simple auto-resolution logic
        if conflict.conflict_type == ConflictType.RESOURCE_CONFLICT:
            conflict.resolved = True
            conflict.resolution = "Resources reallocated automatically"
        elif conflict.conflict_type == ConflictType.TASK_CONFLICT:
            conflict.resolved = True
            conflict.resolution = "Task priority adjusted"
        
        return conflict.conflict_id

class Dashboard:
    """Dashboard for system monitoring"""
    def __init__(self, agent_registry: AgentRegistry, task_manager: TaskManager):
        self.agent_registry = agent_registry
        self.task_manager = task_manager
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide overview"""
        agents = list(self.agent_registry.agents.values())
        
        # Calculate agent breakdown by status
        status_breakdown = {}
        for agent in agents:
            status = agent.status
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
        
        # Calculate performance metrics
        completed_tasks = sum(1 for task in self.task_manager.tasks.values() 
                            if task.status == "COMPLETED")
        total_tasks = len(self.task_manager.tasks)
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return {
            "total_agents": len(agents),
            "active_agents": sum(1 for a in agents if a.status == AgentStatus.AVAILABLE),
            "agent_breakdown": {
                "by_status": status_breakdown
            },
            "performance_summary": {
                "total_tasks_completed": completed_tasks,
                "total_tasks": total_tasks,
                "overall_success_rate": success_rate
            }
        }
    
    async def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed agent information"""
        if agent_id not in self.agent_registry.agents:
            return None
        
        agent = self.agent_registry.agents[agent_id]
        agent_instance = self.agent_registry.agent_instances.get(agent_id)
        
        details = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "version": agent.version,
            "capabilities": agent.capabilities,
            "tags": agent.tags,
            "status": agent.status,
            "reliability_score": agent.reliability_score,
            "last_seen": agent.last_seen.isoformat()
        }
        
        if agent_instance:
            # Add Enhanced FSM metrics
            fsm_metrics = agent_instance.get_metrics()
            details["fsm_metrics"] = fsm_metrics
        
        return details

class MultiAgentPlatform:
    """Main platform orchestrator"""
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url
        self.agent_registry = AgentRegistry()
        self.task_manager = TaskManager()
        self.resource_manager = ResourceManager()
        self.marketplace = Marketplace()
        self.conflict_resolver = ConflictResolver()
        self.dashboard = Dashboard(self.agent_registry, self.task_manager)
    
    async def initialize(self):
        """Initialize the platform"""
        logger.info("Initializing Multi-Agent Platform")
        # Initialize Redis connection if needed
        # Initialize other components
    
    async def shutdown(self):
        """Shutdown the platform"""
        logger.info("Shutting down Multi-Agent Platform")
        # Cleanup resources
    
    async def register_agent(self, agent_instance: MigratedEnhancedFSMAgent, 
                           metadata: AgentMetadata, resources: Dict[str, float]) -> bool:
        """Register an agent with the platform"""
        # Allocate resources
        if not await self.resource_manager.allocate_resources(metadata.agent_id, resources):
            return False
        
        # Register agent
        return await self.agent_registry.register(metadata.agent_id, metadata, agent_instance)
    
    async def submit_task(self, task: UnifiedTask) -> str:
        """Submit a task to the platform"""
        return await self.task_manager.submit_task(task)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return await self.task_manager.get_task_status(task_id)

# =============================
# API Models
# =============================

class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration"""
    name: str
    version: str
    capabilities: List[str]
    tags: List[str] = Field(default_factory=list)
    resources: Dict[str, float] = Field(default_factory=dict)
    description: Optional[str] = None

class TaskSubmissionRequest(BaseModel):
    """Request model for task submission"""
    task_type: str
    priority: int = Field(default=5, ge=1, le=10)
    payload: Dict[str, Any]
    required_capabilities: List[str]
    deadline: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)

class MarketplaceListingRequest(BaseModel):
    """Request model for marketplace listing"""
    agent_id: str
    description: str
    pricing: Dict[str, float]
    keywords: List[str]

class ConflictReportRequest(BaseModel):
    """Request model for conflict reporting"""
    conflict_type: str
    involved_agents: List[str]
    description: str
    context: Dict[str, Any]

class AgentSearchRequest(BaseModel):
    """Request model for agent search"""
    query: str
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    max_price: Optional[float] = None
    capabilities: Optional[List[str]] = None
    tags: Optional[List[str]] = None

# =============================
# API Server
# =============================

app = FastAPI(
    title="Multi-Agent Collaboration Platform API",
    description="API for managing and orchestrating AI agents with Enhanced FSM",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global platform instance
platform: Optional[MultiAgentPlatform] = None

# WebSocket connections for real-time updates
websocket_connections: Set[WebSocket] = set()

# =============================
# Lifecycle Events
# =============================

@app.on_event("startup")
async def startup_event():
    """Initialize the platform on startup"""
    global platform
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    platform = MultiAgentPlatform(redis_url=redis_url)
    await platform.initialize()
    
    logger.info("Multi-Agent Platform API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global platform
    
    if platform:
        await platform.shutdown()
    
    # Close all WebSocket connections
    for websocket in websocket_connections.copy():
        await websocket.close()
    
    logger.info("Multi-Agent Platform API shut down")

# =============================
# Authentication
# =============================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    # Simplified - implement proper authentication
    if credentials.credentials != "valid_token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# =============================
# Agent Management Endpoints
# =============================

@app.post("/api/v2/agents/register", response_model=Dict[str, str])
async def register_agent(
    request: AgentRegistrationRequest,
    token: str = Depends(verify_token)
):
    """Register a new agent with the platform"""
    try:
        # Create Enhanced FSM agent instance
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Create mock tools for the agent
        class MockTool:
            def __init__(self, name):
                self.name = name
                self.tool_name = name
        
        mock_tools = [MockTool("search"), MockTool("calculator"), MockTool("database")]
        
        agent = MigratedEnhancedFSMAgent(
            tools=mock_tools,
            enable_hierarchical=True,
            enable_probabilistic=True,
            enable_discovery=True,
            enable_metrics=True,
            fsm_name=f"Agent_{agent_id}"
        )
        
        # Create metadata
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=request.name,
            version=request.version,
            capabilities=request.capabilities,
            tags=request.tags
        )
        
        # Register with platform
        success = await platform.register_agent(agent, metadata, request.resources)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register agent")
        
        # Broadcast update
        await broadcast_update({
            "event": "agent_registered",
            "agent_id": agent_id,
            "name": request.name
        })
        
        return {
            "agent_id": agent_id,
            "status": "registered",
            "message": f"Agent {request.name} successfully registered"
        }
        
    except Exception as e:
        logger.error(f"Agent registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/agents", response_model=List[Dict[str, Any]])
async def list_agents(
    status: Optional[str] = None,
    capability: Optional[str] = None,
    tag: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """List all registered agents with optional filters"""
    try:
        # Parse filters
        status_filter = status if status else None
        capability_filter = [capability] if capability else None
        tag_filter = [tag] if tag else None
        
        # Discover agents
        agents = await platform.agent_registry.discover(
            capabilities=capability_filter,
            tags=tag_filter,
            status=status_filter
        )
        
        # Format response
        return [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "version": agent.version,
                "capabilities": agent.capabilities,
                "tags": agent.tags,
                "status": agent.status,
                "reliability_score": agent.reliability_score,
                "last_seen": agent.last_seen.isoformat()
            }
            for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_details(
    agent_id: str,
    token: str = Depends(verify_token)
):
    """Get detailed information about a specific agent"""
    try:
        details = await platform.dashboard.get_agent_details(agent_id)
        
        if not details:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v2/agents/{agent_id}")
async def unregister_agent(
    agent_id: str,
    token: str = Depends(verify_token)
):
    """Unregister an agent from the platform"""
    try:
        success = await platform.agent_registry.unregister(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Release resources
        await platform.resource_manager.release_resources(agent_id)
        
        # Broadcast update
        await broadcast_update({
            "event": "agent_unregistered",
            "agent_id": agent_id
        })
        
        return {"status": "unregistered", "agent_id": agent_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unregister agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# Task Management Endpoints
# =============================

@app.post("/api/v2/tasks/submit", response_model=Dict[str, str])
async def submit_task(
    request: TaskSubmissionRequest,
    token: str = Depends(verify_token)
):
    """Submit a task to the platform"""
    try:
        # Create task
        task = UnifiedTask(
            task_id=str(uuid.uuid4()),
            task_type=request.task_type,
            priority=request.priority,
            payload=request.payload,
            required_capabilities=request.required_capabilities,
            deadline=request.deadline,
            dependencies=request.dependencies
        )
        
        # Submit to platform
        task_id = await platform.submit_task(task)
        
        # Broadcast update
        await broadcast_update({
            "event": "task_submitted",
            "task_id": task_id,
            "task_type": request.task_type
        })
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Task successfully submitted for execution"
        }
        
    except Exception as e:
        logger.error(f"Task submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_status(
    task_id: str,
    token: str = Depends(verify_token)
):
    """Get the status of a submitted task"""
    try:
        status = await platform.get_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/tasks/batch", response_model=List[Dict[str, str]])
async def submit_batch_tasks(
    tasks: List[TaskSubmissionRequest],
    token: str = Depends(verify_token)
):
    """Submit multiple tasks as a batch"""
    try:
        results = []
        
        for task_request in tasks:
            task = UnifiedTask(
                task_id=str(uuid.uuid4()),
                task_type=task_request.task_type,
                priority=task_request.priority,
                payload=task_request.payload,
                required_capabilities=task_request.required_capabilities,
                deadline=task_request.deadline,
                dependencies=task_request.dependencies
            )
            
            task_id = await platform.submit_task(task)
            results.append({
                "task_id": task_id,
                "status": "submitted"
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch task submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# Marketplace Endpoints
# =============================

@app.post("/api/v2/marketplace/publish", response_model=Dict[str, str])
async def publish_to_marketplace(
    request: MarketplaceListingRequest,
    token: str = Depends(verify_token)
):
    """Publish an agent to the marketplace"""
    try:
        success = await platform.marketplace.publish_agent(
            request.agent_id,
            request.description,
            request.pricing,
            request.keywords
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to publish agent")
        
        return {
            "status": "published",
            "agent_id": request.agent_id,
            "message": "Agent successfully published to marketplace"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Marketplace publishing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/marketplace/search", response_model=List[Dict[str, Any]])
async def search_marketplace(
    request: AgentSearchRequest,
    token: str = Depends(verify_token)
):
    """Search for agents in the marketplace"""
    try:
        listings = await platform.marketplace.search_agents(
            request.query,
            request.min_rating,
            request.max_price
        )
        
        return [
            {
                "agent_id": listing.agent_id,
                "name": listing.metadata.name,
                "description": listing.description,
                "capabilities": listing.metadata.capabilities,
                "pricing": listing.pricing,
                "average_rating": listing.average_rating,
                "total_usage": listing.total_usage
            }
            for listing in listings
        ]
        
    except Exception as e:
        logger.error(f"Marketplace search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/marketplace/rate/{agent_id}")
async def rate_agent(
    agent_id: str,
    rating: float = Field(..., ge=1.0, le=5.0),
    review: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Rate an agent in the marketplace"""
    try:
        listing = platform.marketplace.listings.get(agent_id)
        
        if not listing:
            raise HTTPException(status_code=404, detail="Agent not found in marketplace")
        
        listing.add_rating(rating, review, "anonymous")  # Would use actual user ID
        
        return {
            "status": "rated",
            "agent_id": agent_id,
            "new_average_rating": listing.average_rating
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent rating failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# Monitoring & Analytics Endpoints
# =============================

@app.get("/api/v2/dashboard/overview", response_model=Dict[str, Any])
async def get_system_overview(token: str = Depends(verify_token)):
    """Get system-wide metrics and overview"""
    try:
        overview = await platform.dashboard.get_system_overview()
        return overview
        
    except Exception as e:
        logger.error(f"Failed to get system overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/resources/utilization", response_model=Dict[str, float])
async def get_resource_utilization(token: str = Depends(verify_token)):
    """Get current resource utilization"""
    try:
        utilization = platform.resource_manager.get_resource_utilization()
        
        return {
            resource: percentage
            for resource, percentage in utilization.items()
        }
        
    except Exception as e:
        logger.error(f"Failed to get resource utilization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/conflicts/report", response_model=Dict[str, str])
async def report_conflict(
    request: ConflictReportRequest,
    token: str = Depends(verify_token)
):
    """Report a conflict for resolution"""
    try:
        conflict = Conflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type=request.conflict_type,
            involved_agents=request.involved_agents,
            description=request.description,
            context=request.context
        )
        
        conflict_id = await platform.conflict_resolver.report_conflict(conflict)
        
        return {
            "conflict_id": conflict_id,
            "status": "reported",
            "resolution": conflict.resolution if conflict.resolved else None
        }
        
    except Exception as e:
        logger.error(f"Conflict reporting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# WebSocket Endpoints
# =============================

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        # Send initial system overview
        overview = await platform.dashboard.get_system_overview()
        await websocket.send_json({"type": "overview", "data": overview})
        
        # Keep connection alive and send periodic updates
        while True:
            # Wait for messages or send periodic updates
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client messages if needed
                
            except asyncio.TimeoutError:
                # Send periodic update
                overview = await platform.dashboard.get_system_overview()
                await websocket.send_json({"type": "overview", "data": overview})
                
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        websocket_connections.discard(websocket)

async def broadcast_update(update: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    disconnected = set()
    
    for websocket in websocket_connections:
        try:
            await websocket.send_json({"type": "update", "data": update})
        except:
            disconnected.add(websocket)
    
    # Remove disconnected clients
    websocket_connections.difference_update(disconnected)

# =============================
# HTML Dashboard
# =============================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the web dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Agent Collaboration Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #2196F3;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            .chart-container {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .status-indicator {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 5px;
            }
            .status-active { background-color: #4CAF50; }
            .status-idle { background-color: #FFC107; }
            .status-offline { background-color: #F44336; }
            #connectionStatus {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            .connected { background-color: #4CAF50; color: white; }
            .disconnected { background-color: #F44336; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Multi-Agent Collaboration Platform</h1>
            
            <div id="connectionStatus" class="disconnected">Disconnected</div>
            
            <div class="metrics-grid" id="metricsGrid">
                <div class="metric-card">
                    <div class="metric-value" id="totalAgents">0</div>
                    <div class="metric-label">Total Agents</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="activeAgents">0</div>
                    <div class="metric-label">Active Agents</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="tasksCompleted">0</div>
                    <div class="metric-label">Tasks Completed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="successRate">0%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Agent Status Distribution</h3>
                <canvas id="statusChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Resource Utilization</h3>
                <canvas id="resourceChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Real-time Events</h3>
                <div id="eventLog" style="max-height: 200px; overflow-y: auto;">
                    <!-- Events will be added here -->
                </div>
            </div>
        </div>
        
        <script>
            // WebSocket connection
            let ws = null;
            const wsUrl = `ws://${window.location.host}/ws/dashboard`;
            
            // Charts
            let statusChart = null;
            let resourceChart = null;
            
            function connectWebSocket() {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    document.getElementById('connectionStatus').className = 'connected';
                    document.getElementById('connectionStatus').textContent = 'Connected';
                };
                
                ws.onclose = () => {
                    document.getElementById('connectionStatus').className = 'disconnected';
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    // Reconnect after 3 seconds
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'overview') {
                        updateDashboard(message.data);
                    } else if (message.type === 'update') {
                        addEvent(message.data);
                    }
                };
            }
            
            function updateDashboard(data) {
                // Update metrics
                document.getElementById('totalAgents').textContent = data.total_agents;
                document.getElementById('activeAgents').textContent = data.active_agents;
                document.getElementById('tasksCompleted').textContent = 
                    data.performance_summary.total_tasks_completed;
                document.getElementById('successRate').textContent = 
                    (data.performance_summary.overall_success_rate * 100).toFixed(1) + '%';
                
                // Update status chart
                updateStatusChart(data.agent_breakdown.by_status);
                
                // Update resource chart (would need additional API call)
                fetchResourceUtilization();
            }
            
            function updateStatusChart(statusData) {
                const ctx = document.getElementById('statusChart').getContext('2d');
                
                if (statusChart) {
                    statusChart.destroy();
                }
                
                statusChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: Object.keys(statusData),
                        datasets: [{
                            data: Object.values(statusData),
                            backgroundColor: [
                                '#4CAF50',
                                '#2196F3',
                                '#FFC107',
                                '#F44336',
                                '#9C27B0'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }
            
            async function fetchResourceUtilization() {
                try {
                    const response = await fetch('/api/v2/resources/utilization', {
                        headers: {
                            'Authorization': 'Bearer valid_token'
                        }
                    });
                    const data = await response.json();
                    updateResourceChart(data);
                } catch (error) {
                    console.error('Failed to fetch resource utilization:', error);
                }
            }
            
            function updateResourceChart(resourceData) {
                const ctx = document.getElementById('resourceChart').getContext('2d');
                
                if (resourceChart) {
                    resourceChart.destroy();
                }
                
                resourceChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(resourceData),
                        datasets: [{
                            label: 'Utilization %',
                            data: Object.values(resourceData),
                            backgroundColor: '#2196F3'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
            }
            
            function addEvent(event) {
                const eventLog = document.getElementById('eventLog');
                const eventElement = document.createElement('div');
                eventElement.style.padding = '10px';
                eventElement.style.borderBottom = '1px solid #eee';
                
                const timestamp = new Date().toLocaleTimeString();
                eventElement.innerHTML = `
                    <strong>${timestamp}</strong> - 
                    ${event.event}: ${JSON.stringify(event)}
                `;
                
                eventLog.insertBefore(eventElement, eventLog.firstChild);
                
                // Keep only last 50 events
                while (eventLog.children.length > 50) {
                    eventLog.removeChild(eventLog.lastChild);
                }
            }
            
            // Initialize
            connectWebSocket();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 