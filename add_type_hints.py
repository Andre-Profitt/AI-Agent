#!/usr/bin/env python3
"""
Add type hints to constructors and methods throughout the codebase
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Union, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Common parameter patterns and their type hints
TYPE_MAPPINGS = {
    # Configuration related
    'config': 'ConfigurationService',
    'configuration': 'ConfigurationService',
    'settings': 'Dict[str, Any]',
    
    # Logging related
    'logger': 'Union[logging.Logger, LoggingService]',
    'log': 'logging.Logger',
    
    # Database related
    'db': 'DatabaseClient',
    'database': 'DatabaseClient',
    'connection': 'DatabaseConnection',
    'client': 'SupabaseClient',
    'supabase': 'SupabaseClient',
    'pool': 'ConnectionPool',
    
    # Cache related
    'cache': 'Optional[CacheClient]',
    'redis': 'Optional[Redis]',
    
    # Repository/Service patterns
    'repository': 'Repository',
    'repo': 'Repository',
    'service': 'Service',
    'manager': 'Manager',
    'handler': 'Handler',
    'executor': 'Executor',
    
    # HTTP/API related
    'session': 'aiohttp.ClientSession',
    'api_key': 'str',
    'token': 'str',
    'url': 'str',
    'endpoint': 'str',
    'headers': 'Dict[str, str]',
    
    # Agent/Tool related
    'tools': 'List[BaseTool]',
    'tool': 'BaseTool',
    'agents': 'List[BaseAgent]',
    'agent': 'BaseAgent',
    'llm': 'LLM',
    'model': 'str',
    
    # Common types
    'timeout': 'Union[int, float]',
    'max_retries': 'int',
    'retry_delay': 'float',
    'batch_size': 'int',
    'limit': 'int',
    'offset': 'int',
    'name': 'str',
    'id': 'str',
    'path': 'Union[str, Path]',
    'data': 'Dict[str, Any]',
    'params': 'Dict[str, Any]',
    'kwargs': 'Dict[str, Any]',
    'args': 'Tuple[Any, ...]',
    
    # Async related
    'loop': 'Optional[asyncio.AbstractEventLoop]',
    'semaphore': 'asyncio.Semaphore',
    'lock': 'asyncio.Lock',
    
    # Monitoring
    'metrics': 'MetricsCollector',
    'tracer': 'Tracer',
    'monitor': 'Monitor',
}

def get_imports_for_file(content: str) -> Set[str]:
    """Determine what imports are needed based on type hints used"""
    imports = set()
    
    # Always need these for type hints
    imports.add("from typing import Optional, Dict, Any, List, Union, Tuple")
    
    if 'logging.Logger' in content:
        imports.add("import logging")
    
    if 'asyncio.' in content:
        imports.add("import asyncio")
    
    if 'Path' in content and 'pathlib' not in content:
        imports.add("from pathlib import Path")
    
    # Add interface imports if used
    if any(term in content for term in ['ConfigurationService', 'DatabaseClient', 'CacheClient', 'LoggingService']):
        imports.add("from src.shared.types.di_types import (")
        imports.add("    ConfigurationService, DatabaseClient, CacheClient, LoggingService")
        imports.add(")")
    
    return imports

def infer_type_from_default(default_value: str) -> str:
    """Infer type from default value"""
    if default_value == 'None':
        return 'Optional[Any]'
    elif default_value == 'True' or default_value == 'False':
        return 'bool'
    elif default_value.startswith('"') or default_value.startswith("'"):
        return 'str'
    elif default_value.isdigit():
        return 'int'
    elif '.' in default_value and default_value.replace('.', '').replace('-', '').isdigit():
        return 'float'
    elif default_value == '[]':
        return 'List[Any]'
    elif default_value == '{}':
        return 'Dict[str, Any]'
    elif default_value.startswith('[') and default_value.endswith(']'):
        return 'List[Any]'
    elif default_value.startswith('{') and default_value.endswith('}'):
        return 'Dict[str, Any]'
    else:
        return 'Any'

def get_type_hint(param_name: str, default_value: str = None, context: str = '') -> str:
    """Get appropriate type hint for parameter"""
    param_lower = param_name.lower()
    
    # Check exact matches first
    if param_name in TYPE_MAPPINGS:
        type_hint = TYPE_MAPPINGS[param_name]
    else:
        # Check if parameter contains known patterns
        type_hint = None
        for pattern, hint in TYPE_MAPPINGS.items():
            if pattern in param_lower:
                type_hint = hint
                break
        
        if not type_hint:
            # Try to infer from default value
            if default_value and default_value != 'None':
                type_hint = infer_type_from_default(default_value)
            else:
                type_hint = 'Any'
    
    # Handle Optional types
    if default_value == 'None' and not type_hint.startswith('Optional['):
        if type_hint == 'Any':
            type_hint = 'Optional[Any]'
        else:
            type_hint = f'Optional[{type_hint}]'
    
    return type_hint

def add_type_hints_to_init(content: str, class_name: str = '') -> Tuple[str, int]:
    """Add type hints to __init__ methods"""
    changes = 0
    
    # Pattern to match __init__ methods
    init_pattern = r'(\s*)def __init__\(self([^)]*)\)(\s*):'
    
    def process_init(match):
        nonlocal changes
        indent = match.group(1)
        params = match.group(2)
        spacing = match.group(3)
        
        if not params.strip():
            changes += 1
            return f'{indent}def __init__(self) -> None{spacing}:'
        
        # Check if already has return type
        if '->' in match.group(0):
            return match.group(0)
        
        # Parse parameters
        param_list = []
        param_parts = params.split(',')
        
        for param in param_parts:
            param = param.strip()
            if not param:
                continue
            
            # Check if already has type hint
            if ':' in param:
                param_list.append(param)
                continue
            
            # Extract parameter name and default
            if '=' in param:
                parts = param.split('=', 1)
                name = parts[0].strip()
                default = parts[1].strip()
            else:
                name = param.strip()
                default = None
            
            # Skip *args and **kwargs for now
            if name.startswith('*'):
                param_list.append(param)
                continue
            
            # Get type hint
            type_hint = get_type_hint(name, default, class_name)
            
            # Reconstruct parameter
            if default is not None:
                param_with_hint = f"{name}: {type_hint} = {default}"
            else:
                param_with_hint = f"{name}: {type_hint}"
            
            param_list.append(param_with_hint)
        
        # Reconstruct init method
        if param_list:
            # Format nicely if long
            if len(', '.join(param_list)) > 80:
                params_str = ',\n' + indent + '        '.join([''] + param_list)
            else:
                params_str = ', '.join(param_list)
        else:
            params_str = ''
        
        changes += 1
        return f'{indent}def __init__(self, {params_str}) -> None{spacing}:'
    
    new_content = re.sub(init_pattern, process_init, content)
    return new_content, changes

def add_type_hints_to_methods(content: str) -> Tuple[str, int]:
    """Add return type hints to methods that don't have them"""
    changes = 0
    
    # Common method patterns and their return types
    method_returns = {
        'get_': 'Any',
        'fetch_': 'Any',
        'load_': 'Any',
        'save': 'bool',
        'delete': 'bool',
        'update': 'bool',
        'create': 'Any',
        'is_': 'bool',
        'has_': 'bool',
        'can_': 'bool',
        'should_': 'bool',
        'validate': 'bool',
        'process': 'Any',
        'execute': 'Any',
        'run': 'Any',
        '_init': 'None',
        'setup': 'None',
        'cleanup': 'None',
        'close': 'None',
        'start': 'None',
        'stop': 'None',
    }
    
    # Pattern for methods without return type
    method_pattern = r'(\s*)(async\s+)?def\s+(\w+)\s*\([^)]*\)(\s*):'
    
    def process_method(match):
        nonlocal changes
        indent = match.group(1)
        async_keyword = match.group(2) or ''
        method_name = match.group(3)
        spacing = match.group(4)
        
        # Skip if already has return type or is __init__
        if '->' in match.group(0) or method_name == '__init__':
            return match.group(0)
        
        # Determine return type
        return_type = 'Any'
        for pattern, ret_type in method_returns.items():
            if method_name.startswith(pattern):
                return_type = ret_type
                break
        
        # Check if it's an async method that might return awaitable
        if async_keyword and return_type == 'Any':
            return_type = 'Any'  # Could be Awaitable[Any] but keeping simple
        
        changes += 1
        # Rebuild the method signature with return type
        full_match = match.group(0)
        before_colon = full_match[:-1]  # Everything except the colon
        return f"{before_colon} -> {return_type}:"
    
    new_content = re.sub(method_pattern, process_method, content)
    return new_content, changes

def process_file(file_path: Path) -> bool:
    """Process a single file to add type hints"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        total_changes = 0
        
        # Skip if file is already well-typed (has many type hints)
        if content.count('->') > 10 and content.count(': ') > 20:
            return False
        
        # Add type hints to __init__ methods
        content, init_changes = add_type_hints_to_init(content)
        total_changes += init_changes
        
        # Add return type hints to methods
        content, method_changes = add_type_hints_to_methods(content)
        total_changes += method_changes
        
        # Add necessary imports if changes were made
        if total_changes > 0:
            needed_imports = get_imports_for_file(content)
            
            # Check which imports are missing
            missing_imports = []
            for imp in needed_imports:
                if imp not in content:
                    missing_imports.append(imp)
            
            if missing_imports:
                # Find where to insert imports
                lines = content.split('\n')
                import_idx = 0
                
                for i, line in enumerate(lines):
                    if line.startswith(('import ', 'from ')):
                        import_idx = i + 1
                    elif import_idx > 0 and line and not line.startswith(('#', 'import', 'from')):
                        break
                
                # Insert missing imports
                for imp in reversed(missing_imports):
                    lines.insert(import_idx, imp)
                
                content = '\n'.join(lines)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        logger.error("Error processing {}: {}", extra={"file_path": file_path, "e": e})
        return False

def add_type_hints():
    """Add type hints to all Python files in the project"""
    logger.info("🏷️  Adding type hints to constructors and methods...")
    
    # Statistics
    files_processed = 0
    files_modified = 0
    
    # Process all Python files in src
    src_files = list(Path('src').rglob('*.py'))
    
    logger.info("Found {} Python files to process", extra={"len_src_files": len(src_files)})
    
    for file_path in src_files:
        # Skip test files and generated files
        if any(skip in str(file_path) for skip in ['__pycache__', 'test_', '_pb2.py']):
            continue
        
        files_processed += 1
        if process_file(file_path):
            files_modified += 1
            logger.info("✅ Added type hints to {}", extra={"file_path": file_path})
    
    logger.info("\n📊 Summary:")
    logger.info("   Files processed: {}", extra={"files_processed": files_processed})
    logger.info("   Files modified: {}", extra={"files_modified": files_modified})
    if files_processed > 0:
        success_rate = files_modified/files_processed*100
        logger.info("   Success rate: {:.1f}%", extra={"success_rate": success_rate})

def verify_type_hints():
    """Verify type hint coverage"""
    logger.info("\n🔍 Verifying type hint coverage...")
    
    total_inits = 0
    typed_inits = 0
    total_methods = 0
    typed_methods = 0
    
    for file_path in Path('src').rglob('*.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count __init__ methods
            inits = re.findall(r'def __init__\([^)]*\)', content)
            total_inits += len(inits)
            
            # Count typed __init__ methods
            typed = re.findall(r'def __init__\([^)]*\) -> ', content)
            typed_inits += len(typed)
            
            # Count all methods
            methods = re.findall(r'def \w+\([^)]*\):', content)
            total_methods += len(methods)
            
            # Count typed methods
            typed_m = re.findall(r'def \w+\([^)]*\) -> ', content)
            typed_methods += len(typed_m)
            
        except Exception:
            pass
    
    if total_inits > 0:
        init_coverage = (typed_inits / total_inits) * 100
        logger.info("   Constructor type coverage: {:.1f}% ({}/{})", extra={
            "init_coverage": init_coverage, "typed_inits": typed_inits, "total_inits": total_inits
        })
    
    if total_methods > 0:
        method_coverage = (typed_methods / total_methods) * 100
        logger.info("   Method type coverage: {:.1f}% ({}/{})", extra={
            "method_coverage": method_coverage, "typed_methods": typed_methods, "total_methods": total_methods
        })
    
    overall = ((typed_inits + typed_methods) / (total_inits + total_methods) * 100) if (total_inits + total_methods) > 0 else 0
    logger.info("   Overall type coverage: {:.1f}%", extra={"overall": overall})
    
    if overall >= 80:
        logger.info("✅ Good type coverage!")
    else:
        logger.info("⚠️  Consider adding more type hints")

if __name__ == "__main__":
    add_type_hints()
    verify_type_hints() 