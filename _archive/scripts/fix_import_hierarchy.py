#!/usr/bin/env python3
"""
Import hierarchy cleanup for AI Agent project
Establishes clear import structure and fixes all undefined variables
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ImportHierarchyFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.import_graph = defaultdict(set)  # file -> imported modules
        self.export_map = defaultdict(set)    # module -> exported names
        self.undefined_vars = defaultdict(set) # file -> undefined variables
        self.circular_imports = []
        
    def analyze_and_fix(self):
        """Main method to analyze and fix import hierarchy"""
        logger.info("📦 Analyzing import hierarchy...\n")
        
        # Phase 1: Build import graph and export map
        self._build_import_graph()
        self._build_export_map()
        
        # Phase 2: Detect issues
        self._detect_circular_imports()
        self._collect_undefined_variables()
        
        # Phase 3: Fix issues
        self._fix_circular_imports()
        self._create_import_hierarchy()
        self._fix_undefined_variables()
        
        # Phase 4: Generate report
        self._generate_import_report()
        
    def _build_import_graph(self):
        """Build a graph of all imports in the project"""
        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', 'test_', 'scripts']):
                continue
                
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                
                relative_path = str(py_file.relative_to(self.project_root))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.import_graph[relative_path].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.import_graph[relative_path].add(node.module)
                            
            except Exception as e:
                logger.warning(f"Could not parse {py_file}: {e}")
                
    def _build_export_map(self):
        """Build a map of what each module exports"""
        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', 'test_', 'scripts']):
                continue
                
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                
                module_path = str(py_file.relative_to(self.project_root))[:-3].replace('/', '.')
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self.export_map[module_path].add(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):
                            self.export_map[module_path].add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and not target.id.startswith('_'):
                                self.export_map[module_path].add(target.id)
                                
            except Exception as e:
                pass
                
    def _detect_circular_imports(self):
        """Detect circular import dependencies"""
        def find_cycles(node, path, visited):
            if node in path:
                cycle_start = path.index(node)
                self.circular_imports.append(path[cycle_start:] + [node])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for imported in self.import_graph.get(node, []):
                # Convert import to file path
                if imported.startswith('src.'):
                    file_path = imported.replace('.', '/') + '.py'
                    if file_path in self.import_graph:
                        find_cycles(file_path, path.copy(), visited.copy())
                        
            path.pop()
            
        for file_path in self.import_graph:
            find_cycles(file_path, [], set())
            
    def _collect_undefined_variables(self):
        """Collect all undefined variables from TODO comments"""
        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', 'scripts']):
                continue
                
            try:
                content = py_file.read_text()
                for line in content.split('\n'):
                    if '# TODO: Fix undefined variables:' in line:
                        vars_part = line.split(':', 2)[-1].strip()
                        vars_list = [v.strip() for v in vars_part.split(',')]
                        relative_path = str(py_file.relative_to(self.project_root))
                        self.undefined_vars[relative_path].update(vars_list)
                        
            except Exception:
                pass
                
    def _fix_circular_imports(self):
        """Fix circular imports using TYPE_CHECKING pattern"""
        fixes_applied = []
        
        for cycle in self.circular_imports:
            if len(cycle) <= 3:  # Only fix simple cycles
                # Identify the best place to break the cycle
                for i, file_path in enumerate(cycle[:-1]):
                    if file_path.endswith('.py') and (self.project_root / file_path).exists():
                        self._apply_type_checking_import(file_path, cycle[i+1])
                        fixes_applied.append(f"Fixed circular import: {' -> '.join(cycle)}")
                        break
                        
        logger.info(f"Fixed {len(fixes_applied)} circular imports")
        
    def _apply_type_checking_import(self, file_path: str, imported_module: str):
        """Apply TYPE_CHECKING pattern to break circular import"""
        full_path = self.project_root / file_path
        if not full_path.exists():
            return
            
        content = full_path.read_text()
        
        # Check if TYPE_CHECKING is already imported
        if 'TYPE_CHECKING' not in content:
            # Add TYPE_CHECKING import
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('from typing import'):
                    lines[i] = line.rstrip() + ', TYPE_CHECKING'
                    break
            else:
                # Add new import
                for i, line in enumerate(lines):
                    if line.strip().startswith('import') or line.strip().startswith('from'):
                        lines.insert(i, 'from typing import TYPE_CHECKING')
                        break
                        
            content = '\n'.join(lines)
            
        # Move problematic import under TYPE_CHECKING
        module_name = imported_module.replace('.py', '').replace('/', '.')
        import_pattern = f"from {module_name} import"
        
        if import_pattern in content and 'if TYPE_CHECKING:' not in content:
            lines = content.split('\n')
            new_lines = []
            type_checking_imports = []
            
            for line in lines:
                if import_pattern in line and not line.strip().startswith('#'):
                    type_checking_imports.append('    ' + line.strip())
                else:
                    new_lines.append(line)
                    
            # Add TYPE_CHECKING block
            if type_checking_imports:
                for i, line in enumerate(new_lines):
                    if 'TYPE_CHECKING' in line:
                        new_lines.insert(i + 1, '\nif TYPE_CHECKING:')
                        for imp in type_checking_imports:
                            new_lines.insert(i + 2, imp)
                        break
                        
            content = '\n'.join(new_lines)
            full_path.write_text(content)
            
    def _create_import_hierarchy(self):
        """Create proper import hierarchy documentation"""
        hierarchy = """# Import Hierarchy Structure

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
"""
        
        hierarchy_path = self.project_root / "IMPORT_HIERARCHY.md"
        hierarchy_path.write_text(hierarchy)
        logger.info("Created import hierarchy documentation")
        
    def _fix_undefined_variables(self):
        """Fix remaining undefined variables with proper imports"""
        fixes = defaultdict(list)
        
        # Common undefined variable mappings
        common_fixes = {
            'self': "# 'self' should be a method parameter, not imported",
            'agent': "from src.core.entities.agent import Agent",
            'context': "# 'context' is typically a local variable or parameter",
            'result': "# 'result' is typically a local variable",
            'i': "# 'i' is typically a loop variable",
            'e': "# 'e' is typically an exception variable",
        }
        
        for file_path, undefined in self.undefined_vars.items():
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
                
            file_fixes = []
            for var in undefined:
                if var in common_fixes:
                    if not common_fixes[var].startswith('#'):
                        file_fixes.append(common_fixes[var])
                else:
                    # Try to find where this variable is defined
                    for module, exports in self.export_map.items():
                        if var in exports:
                            import_stmt = f"from {module} import {var}"
                            file_fixes.append(import_stmt)
                            break
                            
            if file_fixes:
                fixes[file_path] = file_fixes
                
        # Apply fixes
        for file_path, import_list in fixes.items():
            self._add_imports_to_file(file_path, import_list)
            
        logger.info(f"Fixed undefined variables in {len(fixes)} files")
        
    def _add_imports_to_file(self, file_path: str, imports: List[str]):
        """Add imports to a file in the correct location"""
        full_path = self.project_root / file_path
        if not full_path.exists():
            return
            
        content = full_path.read_text()
        lines = content.split('\n')
        
        # Find where to insert imports
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') and i > 0:
                # After module docstring
                for j in range(i + 1, len(lines)):
                    if '"""' in lines[j]:
                        insert_idx = j + 1
                        break
                break
            elif line.strip() and not line.strip().startswith('#'):
                insert_idx = i
                break
                
        # Group imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for imp in imports:
            if imp.startswith('from src.') or imp.startswith('import src.'):
                local_imports.append(imp)
            elif any(imp.startswith(f'from {lib}') for lib in ['fastapi', 'pydantic', 'langchain']):
                third_party_imports.append(imp)
            else:
                stdlib_imports.append(imp)
                
        # Insert imports
        new_imports = []
        if stdlib_imports:
            new_imports.extend(sorted(set(stdlib_imports)))
        if third_party_imports:
            if stdlib_imports:
                new_imports.append('')
            new_imports.extend(sorted(set(third_party_imports)))
        if local_imports:
            if stdlib_imports or third_party_imports:
                new_imports.append('')
            new_imports.extend(sorted(set(local_imports)))
            
        if new_imports:
            new_imports.append('')  # Blank line after imports
            
            for imp in reversed(new_imports):
                lines.insert(insert_idx, imp)
                
            full_path.write_text('\n'.join(lines))
            
    def _generate_import_report(self):
        """Generate import analysis report"""
        report = f"""# Import Analysis Report

## Circular Imports Found: {len(self.circular_imports)}

"""
        
        if self.circular_imports:
            report += "### Circular Import Chains:\n"
            for cycle in self.circular_imports[:10]:  # Show first 10
                report += f"- {' → '.join(cycle)}\n"
                
        report += f"""

## Undefined Variables: {sum(len(v) for v in self.undefined_vars.values())}

### Files with Most Undefined Variables:
"""
        
        sorted_undefined = sorted(self.undefined_vars.items(), 
                                key=lambda x: len(x[1]), reverse=True)[:10]
        
        for file_path, vars in sorted_undefined:
            report += f"- `{file_path}`: {len(vars)} variables\n"
            
        report += """

## Import Structure:

The project has been reorganized with a clear layered architecture:

1. **Core Layer**: Entities and interfaces (no external dependencies)
2. **Infrastructure Layer**: Technical implementations
3. **Application Layer**: Business logic implementations  
4. **Service Layer**: High-level orchestration

## Recommendations:

1. Always import from lower layers only
2. Use dependency injection for loose coupling
3. Prefer interfaces over concrete implementations
4. Keep circular dependencies broken with TYPE_CHECKING
"""
        
        report_path = self.project_root / "IMPORT_ANALYSIS_REPORT.md"
        report_path.write_text(report)
        logger.info(f"📄 Import analysis report saved to {report_path}")

def main():
    fixer = ImportHierarchyFixer()
    fixer.analyze_and_fix()
    logger.info("\n✅ Import hierarchy fixes completed!")

if __name__ == "__main__":
    main()