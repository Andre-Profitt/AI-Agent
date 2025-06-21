#!/usr/bin/env python3
"""Simple test server to verify the SaaS UI works"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
from datetime import datetime

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
mock_agents = [
    {
        "id": "1",
        "name": "Research Assistant",
        "type": "research",
        "description": "Specialized in web research and data gathering",
        "capabilities": ["Web Search", "Data Analysis", "Report Generation"],
        "status": "online",
        "metrics": {
            "tasksCompleted": 127,
            "successRate": 95,
            "avgResponseTime": 1200
        }
    },
    {
        "id": "2",
        "name": "Code Assistant",
        "type": "coding",
        "description": "Expert in software development and debugging",
        "capabilities": ["Code Generation", "Bug Fixing", "Code Review", "Refactoring"],
        "status": "online",
        "metrics": {
            "tasksCompleted": 89,
            "successRate": 92,
            "avgResponseTime": 800
        }
    },
    {
        "id": "3",
        "name": "Creative Writer",
        "type": "creative",
        "description": "Helps with creative writing and content generation",
        "capabilities": ["Story Writing", "Content Creation", "Editing"],
        "status": "busy",
        "metrics": {
            "tasksCompleted": 56,
            "successRate": 88,
            "avgResponseTime": 1500
        }
    }
]

@app.get("/")
async def root():
    return {"message": "AI Agent Test Server Running"}

@app.get("/api/v1/agents")
async def get_agents():
    return mock_agents

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = next((a for a in mock_agents if a["id"] == agent_id), None)
    if agent:
        return agent
    return {"error": "Agent not found"}

@app.post("/api/v1/messages")
async def send_message(data: dict):
    # Mock response
    return {
        "id": str(datetime.now().timestamp()),
        "role": "assistant",
        "content": f"This is a mock response from {data.get('agentId', 'the agent')}. Your message was: {data.get('content', '')}",
        "timestamp": datetime.now().isoformat(),
        "agentId": data.get("agentId"),
        "metadata": {
            "tools": ["mock_tool"],
            "confidence": 0.95
        }
    }

@app.get("/api/v1/health")
async def get_health():
    return {
        "status": "healthy",
        "uptime": 3600,
        "activeAgents": len([a for a in mock_agents if a["status"] == "online"]),
        "tasksInProgress": 5,
        "memoryUsage": 45,
        "cpuUsage": 32,
        "latency": 120
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Mock assistant response
            await asyncio.sleep(1)  # Simulate processing
            
            response = {
                "id": str(datetime.now().timestamp()),
                "role": "assistant",
                "content": f"I received your message: '{message.get('content', '')}'. This is a test response!",
                "timestamp": datetime.now().isoformat(),
                "agentId": message.get("agentId")
            }
            
            await websocket.send_json({
                "type": "message",
                "data": response
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    print("🚀 Starting AI Agent Test Server on http://localhost:8000")
    print("📚 API Docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)