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

# Import from the unified architecture
from src.unified_architecture.core import MultiAgentPlatform, UnifiedTask, AgentMetadata, AgentCapability, AgentStatus
from src.unified_architecture.registry import AgentRegistry
from src.unified_architecture.marketplace import Marketplace, MarketplaceListing
from src.unified_architecture.conflict_resolution import ConflictResolver, Conflict, ConflictType
from src.unified_architecture.dashboard import Dashboard
from src.unified_architecture.resource_management import ResourceManager
from src.unified_architecture.communication import CommunicationHub
from src.unified_architecture.orchestration import TaskOrchestrator
from src.unified_architecture.performance import PerformanceMonitor
from src.unified_architecture.shared_memory import SharedMemoryManager
from src.unified_architecture.state_management import StateManager
from src.unified_architecture.task_distribution import TaskDistributor
from src.unified_architecture.platform import Platform

# Example agent for testing
class ExampleUnifiedAgent:
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.AVAILABLE
        self.capabilities = [AgentCapability.REASONING, AgentCapability.COLLABORATION]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    description="API for managing and orchestrating AI agents",
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
        # Create agent instance (simplified - would create based on type)
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        agent = ExampleUnifiedAgent(agent_id, request.name)
        
        # Create metadata
        capabilities = [AgentCapability[cap] for cap in request.capabilities]
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=request.name,
            version=request.version,
            capabilities=capabilities,
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
        status_filter = AgentStatus[status] if status else None
        capability_filter = [AgentCapability[capability]] if capability else None
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
                "capabilities": [c.name for c in agent.capabilities],
                "tags": agent.tags,
                "status": agent.status.name,
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
            required_capabilities=[AgentCapability[cap] for cap in request.required_capabilities],
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
                required_capabilities=[AgentCapability[cap] for cap in task_request.required_capabilities],
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
                "capabilities": [c.name for c in listing.metadata.capabilities],
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
            resource.name: percentage
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
            conflict_type=ConflictType[request.conflict_type],
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
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 