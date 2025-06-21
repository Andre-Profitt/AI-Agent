# Import Hierarchy Structure

## Layer 1: Core (No Dependencies)
- `src/core/entities/` - Base entities (Agent, Tool, Message)
- `src/core/exceptions.py` - Custom exceptions
- `src/shared/types/` - Type definitions

## Layer 2: Infrastructure (Depends on Core)
- `src/infrastructure/config/` - Configuration management
- `src/infrastructure/database/` - Database repositories
- `src/infrastructure/resilience/` - Circuit breakers, retry logic

## Layer 3: Application (Depends on Core + Infrastructure)
- `src/application/agents/` - Agent implementations
- `src/application/tools/` - Tool implementations
- `src/application/executors/` - Execution logic

## Layer 4: Services (Depends on All Lower Layers)
- `src/services/` - Business logic services
- `src/api/` - API endpoints
- `src/workflow/` - Workflow orchestration

## Import Rules:
1. **Never import from higher layers to lower layers**
2. **Use TYPE_CHECKING for circular dependencies**
3. **Prefer interfaces over concrete implementations**
4. **Group imports: stdlib, third-party, local**

## Common Import Patterns:

### For Agents:
```python
from typing import Dict, List, Optional, Any
from src.core.entities.agent import Agent, AgentCapability
from src.core.interfaces.agent_executor import IAgentExecutor
from src.infrastructure.config import AgentConfig
```

### For Tools:
```python
from typing import Dict, Any
from src.core.entities.tool import Tool, ToolResult
from src.core.interfaces.tool_executor import IToolExecutor
```

### For API Endpoints:
```python
from fastapi import APIRouter, Depends, HTTPException
from src.core.use_cases.execute_tool import ExecuteToolUseCase
from src.api.dependencies import get_current_user
```
