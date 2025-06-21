# VS Code Auto-Import Guide for AI-Agent Project

## File Analysis Summary

Based on the analysis of your codebase, here are the files with the most undefined variables that need auto-importing:

### 1. **multiagent_api_deployment.py** (Most undefined variables: ~60+)
- Missing imports: `Any`, `Dict`, `List`, `Optional`, `Set`, `datetime`, `logging`, `os`, `uuid`, `asyncio`
- Missing FastAPI components: `FastAPI`, `HTTPException`, `WebSocket`, `WebSocketDisconnect`
- Missing Pydantic: `BaseModel`, `Field`
- Many undefined agent-related variables

### 2. **multi_agent_orchestrator.py** (Undefined variables: ~40+)
- Missing imports: `Any`, `Callable`, `Dict`, `Enum`, `List`, `Optional`
- Missing standard library: `dataclass`, `datetime`, `defaultdict`, `deque`, `logging`, `time`, `uuid`
- Several undefined orchestration-related variables

### 3. **enhanced_fsm.py** (Undefined variables: ~35+)
- Missing imports: `Any`, `Callable`, `Dict`, `Enum`, `List`, `Optional`, `Tuple`
- Missing: `dataclass`, `datetime`, `defaultdict`, `logging`, `time`
- Missing LangChain: `HumanMessage`
- Several state machine related variables

### 4. **workflow_automation.py** (Undefined variables: ~30+)
- Missing imports: `Any`, `Callable`, `Dict`, `Enum`, `List`, `Optional`, `Path`
- Missing: `dataclass`, `datetime`, `timedelta`, `logging`, `json`, `yaml`
- Missing third-party: `Template` (from Jinja2), `networkx`

## Step-by-Step Guide for VS Code Auto-Import

### 1. **Enable Auto-Import Features in VS Code**

First, ensure these settings are enabled in VS Code:

```json
// In VS Code settings.json (Cmd+Shift+P > "Preferences: Open Settings (JSON)")
{
    "python.analysis.autoImportCompletions": true,
    "python.analysis.indexing": true,
    "python.analysis.autoSearchPaths": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### 2. **Install Required Extensions**

Make sure you have these VS Code extensions installed:
- **Python** (by Microsoft)
- **Pylance** (by Microsoft) - Provides better IntelliSense and auto-imports
- **Python Docstring Generator** (optional but helpful)

### 3. **Using Quick Fix (Cmd+.) - Most Effective Method**

This is the fastest way to fix undefined variables:

1. **Open a file** (e.g., `multiagent_api_deployment.py`)
2. **Look for squiggly red underlines** under undefined variables
3. **Click on an undefined variable** (e.g., `Dict`)
4. **Press `Cmd+.` (Mac) or `Ctrl+.` (Windows)** to open Quick Actions
5. **Select the import suggestion** (e.g., "Import 'Dict' from typing")

**Pro tip**: You can fix multiple imports at once:
- Press `Cmd+Shift+P` and search for "Organize Imports"
- This will add all missing imports and remove unused ones

### 4. **Using Auto-Complete While Typing**

When typing a variable name:
1. **Start typing** the name (e.g., `Dict`)
2. **IntelliSense will suggest** imports with a small icon
3. **Select the suggestion** with the import icon
4. VS Code will automatically add the import at the top

### 5. **Fix All Missing Imports in a File**

For files with many undefined variables:

1. **Open the file**
2. **Press `Cmd+Shift+P`** to open Command Palette
3. **Type "Python: Organize Imports"** and select it
4. **OR use the shortcut**: `Shift+Alt+O` (Windows) or `Shift+Option+O` (Mac)

### 6. **Using the Problems Panel**

1. **Open Problems panel**: View > Problems (or `Cmd+Shift+M`)
2. **Filter by current file** using the filter icon
3. **Click on each problem** to jump to it
4. **Use Quick Fix** (`Cmd+.`) on each undefined variable

### 7. **Batch Processing Multiple Files**

For the files with the most issues:

```bash
# 1. Open VS Code in your project directory
code /Users/test/Desktop/ai\ agent/AI-Agent

# 2. Open each problematic file in VS Code
# 3. For each file, run "Organize Imports"
```

### 8. **Common Import Patterns for Your Project**

Based on the analysis, here are the most common imports you'll need:

```python
# Type hints (add to almost every file)
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set

# Dataclasses (for files with @dataclass)
from dataclasses import dataclass, field

# Standard library
import logging
import time
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path

# FastAPI (for API files)
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# LangChain (for agent files)
from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool

# Other third-party
from jinja2 import Template
import networkx as nx
import yaml
```

### 9. **VS Code Shortcuts Summary**

| Action | Mac | Windows/Linux |
|--------|-----|---------------|
| Quick Fix | `Cmd+.` | `Ctrl+.` |
| Organize Imports | `Shift+Option+O` | `Shift+Alt+O` |
| Go to Definition | `F12` | `F12` |
| Peek Definition | `Option+F12` | `Alt+F12` |
| Find All References | `Shift+F12` | `Shift+F12` |
| Command Palette | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| Problems Panel | `Cmd+Shift+M` | `Ctrl+Shift+M` |

### 10. **Recommended Workflow**

1. **Start with the most problematic file**: `multiagent_api_deployment.py`
2. **Open it in VS Code**
3. **Run "Organize Imports"** first to catch obvious imports
4. **Use Quick Fix** (`Cmd+.`) on remaining undefined variables
5. **Save the file** to trigger "organize imports on save"
6. **Check the Problems panel** to ensure all issues are resolved
7. **Repeat for other files** in order of severity

### 11. **Advanced Tips**

1. **Create Import Templates**: Save common import blocks as VS Code snippets
2. **Use Type Stub Files**: For better type hints, install type stubs:
   ```bash
   pip install types-requests types-pyyaml
   ```
3. **Configure Import Sorting**: Add to settings.json:
   ```json
   "python.sortImports.args": [
       "--profile", "black",
       "--line-length", "88"
   ]
   ```

### 12. **Troubleshooting**

If auto-import isn't working:
1. **Reload VS Code Window**: `Cmd+R` in VS Code
2. **Clear Python/Pylance cache**: `Cmd+Shift+P` > "Python: Clear Cache and Reload Window"
3. **Check Python interpreter**: Ensure correct virtual environment is selected
4. **Index workspace**: `Cmd+Shift+P` > "Python: Index Workspace"

## Priority Order for Fixing Files

Based on undefined variable count:
1. `/examples/advanced/multiagent_api_deployment.py` - Fix first (most issues)
2. `/src/gaia_components/multi_agent_orchestrator.py` - Fix second
3. `/src/agents/enhanced_fsm.py` - Fix third
4. `/src/workflow/workflow_automation.py` - Fix fourth

This approach will significantly speed up your development by automatically handling imports!