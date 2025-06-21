# VS Code Auto-Import Guide for AI Agent Project

## Overview
This guide will help you quickly resolve all undefined variables in your AI Agent project using VS Code's auto-import features.

## Files Requiring Auto-Import (by priority)

### 1. Most Undefined Variables (60+)
- `examples/advanced/multiagent_api_deployment.py`

### 2. High Priority (35-40 undefined)
- `src/gaia_components/multi_agent_orchestrator.py`
- `src/agents/enhanced_fsm.py`

### 3. Medium Priority (30+ undefined)
- `src/workflow/workflow_automation.py`

## Step-by-Step Auto-Import Workflow

### Method 1: Organize Imports (Fastest for Multiple Imports)
1. Open a file with undefined variables
2. Press **Shift+Option+O** (Mac) or **Shift+Alt+O** (Windows/Linux)
3. VS Code will automatically:
   - Add all missing imports it can resolve
   - Remove unused imports
   - Sort imports alphabetically

### Method 2: Quick Fix Individual Variables
1. Click on any red-underlined variable
2. Press **Cmd+.** (Mac) or **Ctrl+.** (Windows/Linux)
3. Select the correct import from the dropdown
4. Press Enter to apply

### Method 3: Fix All in File
1. Open the Problems panel: **Cmd+Shift+M**
2. Filter to current file only
3. Click the lightbulb icon next to "undefined variable"
4. Select "Add all missing imports"

### Method 4: Auto-Import While Typing
1. Start typing a class/function name
2. VS Code shows suggestions with import sources
3. Select the one with the import path shown
4. The import is added automatically

## Recommended Workflow for Your Project

### Phase 1: Bulk Import Fix
```bash
# Start with the file with most undefined variables
code examples/advanced/multiagent_api_deployment.py
```

1. Press **Shift+Option+O** to organize imports
2. Save the file (**Cmd+S**)
3. Check remaining undefined variables in Problems panel

### Phase 2: Project-Specific Imports
For your custom imports (like `Agent`, `FSMReActAgent`, etc.):

1. Click on the undefined variable
2. Press **Cmd+.**
3. If VS Code shows suggestions, pick the correct one
4. If no suggestions, type the first few letters of the import path

### Phase 3: Batch Processing
For multiple files:
1. Open VS Code's Command Palette (**Cmd+Shift+P**)
2. Search for "Python: Organize Imports"
3. Run it on each file

## Common Import Patterns in Your Project

### Type Imports
```python
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
```

### Your Project's Core Imports
```python
from src.agents.advanced_agent_fsm import Agent, FSMReActAgent
from src.tools.base_tool import Tool, BaseTool
from src.gaia_components.enhanced_memory_system import MemoryType
from src.infrastructure.config import Config
```

### FastAPI Imports
```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
```

## VS Code Settings Optimization

Your `.vscode/settings.json` is already configured optimally with:
- Pylance language server
- Auto-import completions enabled
- Type checking enabled

## Keyboard Shortcuts Reference

| Action | Mac | Windows/Linux |
|--------|-----|---------------|
| Quick Fix | Cmd+. | Ctrl+. |
| Organize Imports | Shift+Option+O | Shift+Alt+O |
| Problems Panel | Cmd+Shift+M | Ctrl+Shift+M |
| Go to Definition | Cmd+Click | Ctrl+Click |
| Find All References | Shift+F12 | Shift+F12 |
| Command Palette | Cmd+Shift+P | Ctrl+Shift+P |

## Troubleshooting

### If Auto-Import Doesn't Work:
1. **Ensure Python extension is active**: Check bottom-right corner shows Python version
2. **Reload window**: Cmd+R or Ctrl+R
3. **Check import paths**: Your imports use `src.` prefix, ensure VS Code recognizes this
4. **Clear Python cache**: Command Palette → "Python: Clear Cache and Reload Window"

### For Circular Import Issues:
1. Use the Quick Fix option "Add import inside TYPE_CHECKING"
2. This adds:
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from src.module import Class
   ```

## Next Steps

1. Start with `multiagent_api_deployment.py`
2. Use **Shift+Option+O** to organize imports
3. Handle remaining variables with **Cmd+.**
4. Move to next file in priority order
5. Run tests after each file to ensure imports work correctly

## Pro Tips

- **Hover for info**: Hover over any imported symbol to see its source
- **Multi-cursor**: Cmd+D to select multiple instances of same undefined variable
- **Import snippets**: Type `imp` then Tab for import statement template
- **Find symbol**: Cmd+T to search for any class/function in workspace

Remember: VS Code learns from your import choices and improves suggestions over time!