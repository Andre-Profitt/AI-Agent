"""
Tool Creation System
Allows agents to create their own tools
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import ast
import inspect
import asyncio
import subprocess
import tempfile
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolSpecification:
    """Specification for a new tool"""
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str
    category: str
    requirements: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
@dataclass
class GeneratedTool:
    """A tool created by the agent"""
    spec: ToolSpecification
    implementation: str
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    usage_count: int = 0
    success_rate: float = 0.0

class ToolCreationEngine:
    """
    Advanced tool creation system that:
    - Analyzes task requirements
    - Generates tool specifications
    - Implements tools autonomously
    - Tests and validates tools
    - Optimizes tool performance
    """
    
    def __init__(self):
        self.created_tools = {}
        self.tool_templates = self._load_tool_templates()
        self.implementation_patterns = self._load_patterns()
        self.test_framework = ToolTestFramework()
        
    async def create_tool_for_task(
        self, 
        task_description: str,
        context: Dict[str, Any]
    ) -> Optional[GeneratedTool]:
        """Create a tool to solve specific task"""
        logger.info(f"Creating tool for: {task_description}")
        
        # Analyze task requirements
        requirements = await self._analyze_requirements(task_description, context)
        
        # Generate tool specification
        spec = await self._generate_specification(requirements)
        
        # Implement tool
        implementation = await self._implement_tool(spec, requirements)
        
        # Test tool
        test_results = await self.test_framework.test_tool(implementation, spec)
        
        if test_results["success"]:
            # Optimize if needed
            if test_results["performance_score"] < 0.7:
                implementation = await self._optimize_implementation(
                    implementation, 
                    test_results
                )
                
            # Create tool object
            tool = GeneratedTool(
                spec=spec,
                implementation=implementation,
                test_results=test_results,
                performance_metrics=test_results["metrics"],
                created_at=datetime.utcnow()
            )
            
            # Register tool
            self.created_tools[spec.name] = tool
            
            # Make tool available
            await self._deploy_tool(tool)
            
            return tool
            
        return None
        
    async def _analyze_requirements(
        self, 
        task: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze what the tool needs to do"""
        requirements = {
            "input_types": [],
            "output_type": None,
            "operations": [],
            "constraints": [],
            "external_apis": [],
            "libraries": []
        }
        
        # Parse task description
        # Identify key operations
        if "search" in task.lower():
            requirements["operations"].append("search")
            requirements["external_apis"].append("web_api")
        elif "calculate" in task.lower():
            requirements["operations"].append("computation")
            requirements["libraries"].append("numpy")
        elif "analyze" in task.lower():
            requirements["operations"].append("analysis")
            requirements["libraries"].append("pandas")
            
        # Infer data types
        if "text" in task.lower() or "string" in task.lower():
            requirements["input_types"].append("str")
        elif "number" in task.lower() or "calculate" in task.lower():
            requirements["input_types"].append("float")
            
        # Determine output type
        if "list" in task.lower():
            requirements["output_type"] = "List"
        elif "true" in task.lower() or "false" in task.lower():
            requirements["output_type"] = "bool"
        else:
            requirements["output_type"] = "str"
            
        return requirements
        
    async def _generate_specification(
        self, 
        requirements: Dict[str, Any]
    ) -> ToolSpecification:
        """Generate tool specification from requirements"""
        # Create descriptive name
        operations = requirements["operations"]
        name = f"{'_'.join(operations)}_tool" if operations else "custom_tool"
        
        # Build parameter schema
        parameters = {}
        for i, input_type in enumerate(requirements["input_types"]):
            param_name = f"input_{i+1}" if i > 0 else "input"
            parameters[param_name] = {
                "type": input_type,
                "required": True,
                "description": f"Input parameter of type {input_type}"
            }
            
        # Create specification
        spec = ToolSpecification(
            name=name,
            description=f"Tool for {', '.join(operations) if operations else 'custom operations'}",
            parameters=parameters,
            return_type=requirements["output_type"],
            category="generated",
            requirements=requirements["libraries"]
        )
        
        return spec
        
    async def _implement_tool(
        self, 
        spec: ToolSpecification,
        requirements: Dict[str, Any]
    ) -> str:
        """Generate tool implementation"""
        # Build imports
        imports = ["from typing import *"]
        for lib in requirements["libraries"]:
            imports.append(f"import {lib}")
            
        # Build function signature
        params = []
        for param_name, param_info in spec.parameters.items():
            param_type = param_info["type"]
            params.append(f"{param_name}: {param_type}")
            
        signature = f"async def {spec.name}({', '.join(params)}) -> {spec.return_type}:"
        
        # Build function body based on operations
        body_lines = [f'    """', f'    {spec.description}', f'    """']
        
        for operation in requirements["operations"]:
            if operation == "search":
                body_lines.extend([
                    "    # Perform search operation",
                    "    results = []",
                    "    # Search implementation here",
                    "    return results"
                ])
            elif operation == "computation":
                body_lines.extend([
                    "    # Perform computation",
                    "    import numpy as np",
                    "    result = np.mean([float(x) for x in str(input).split() if x.isdigit()])",
                    "    return result"
                ])
            elif operation == "analysis":
                body_lines.extend([
                    "    # Perform analysis",
                    "    analysis_result = {",
                    "        'length': len(str(input)),",
                    "        'type': type(input).__name__,",
                    "        'summary': str(input)[:100]",
                    "    }",
                    "    return str(analysis_result)"
                ])
            else:
                body_lines.extend([
                    "    # Custom implementation",
                    "    result = str(input)",
                    "    return result"
                ])
                
        # Combine implementation
        implementation = "\n".join(imports) + "\n\n" + signature + "\n" + "\n".join(body_lines)
        
        # Validate syntax
        try:
            ast.parse(implementation)
        except SyntaxError as e:
            logger.error(f"Syntax error in generated tool: {e}")
            # Fix common issues
            implementation = self._fix_syntax_errors(implementation)
            
        return implementation
        
    def _fix_syntax_errors(self, code: str) -> str:
        """Attempt to fix common syntax errors"""
        # Add missing colons
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip().startswith(('if ', 'for ', 'while ', 'def ', 'class ')) and not line.rstrip().endswith(':'):
                line = line.rstrip() + ':'
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
    async def _optimize_implementation(
        self, 
        implementation: str,
        test_results: Dict[str, Any]
    ) -> str:
        """Optimize tool implementation based on test results"""
        optimized = implementation
        
        # Identify bottlenecks
        if test_results["metrics"]["execution_time"] > 1.0:
            # Add caching
            optimized = self._add_caching(optimized)
            
        if test_results["metrics"]["memory_usage"] > 100:  # MB
            # Optimize memory usage
            optimized = self._optimize_memory(optimized)
            
        return optimized
        
    def _add_caching(self, implementation: str) -> str:
        """Add caching to implementation"""
        cache_decorator = """
from functools import lru_cache

@lru_cache(maxsize=128)
"""
        # Insert before async def
        return implementation.replace("async def", cache_decorator + "async def", 1)
        
    async def _deploy_tool(self, tool: GeneratedTool):
        """Deploy tool to make it available"""
        # Save to tools directory
        tool_path = f"src/tools/generated/{tool.spec.name}.py"
        
        # Write implementation
        with open(tool_path, 'w') as f:
            f.write(tool.implementation)
            
        # Register in tool registry
        # This would integrate with existing tool system
        logger.info(f"Deployed tool: {tool.spec.name}")
        
    async def learn_from_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> Optional[GeneratedTool]:
        """Learn to create tools from examples"""
        # Analyze patterns in examples
        patterns = self._extract_patterns_from_examples(examples)
        
        # Infer tool specification
        spec = self._infer_specification(patterns)
        
        # Generate implementation based on patterns
        implementation = await self._generate_from_patterns(spec, patterns)
        
        # Test with examples
        test_results = await self._test_with_examples(implementation, examples)
        
        if test_results["success"]:
            return GeneratedTool(
                spec=spec,
                implementation=implementation,
                test_results=test_results,
                performance_metrics=test_results["metrics"],
                created_at=datetime.utcnow()
            )
            
        return None

class ToolTestFramework:
    """Framework for testing generated tools"""
    
    async def test_tool(
        self, 
        implementation: str,
        spec: ToolSpecification
    ) -> Dict[str, Any]:
        """Comprehensive tool testing"""
        results = {
            "success": False,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "metrics": {},
            "performance_score": 0.0
        }
        
        # Create test environment
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write tool to file
            tool_file = f"{temp_dir}/test_tool.py"
            with open(tool_file, 'w') as f:
                f.write(implementation)
                
            # Run syntax check
            syntax_result = self._check_syntax(tool_file)
            if not syntax_result["valid"]:
                results["errors"].append(f"Syntax error: {syntax_result['error']}")
                return results
                
            # Run type checking
            type_result = await self._check_types(tool_file, spec)
            if not type_result["valid"]:
                results["errors"].append(f"Type error: {type_result['error']}")
                
            # Run functionality tests
            func_results = await self._test_functionality(implementation, spec)
            results["tests_passed"] = func_results["passed"]
            results["tests_failed"] = func_results["failed"]
            
            # Run performance tests
            perf_results = await self._test_performance(implementation, spec)
            results["metrics"] = perf_results
            
            # Calculate overall score
            total_tests = results["tests_passed"] + results["tests_failed"]
            if total_tests > 0:
                results["success"] = results["tests_failed"] == 0
                results["performance_score"] = (
                    results["tests_passed"] / total_tests * 0.7 +
                    min(1.0 / perf_results.get("execution_time", 1.0), 1.0) * 0.3
                )
                
        return results
        
    def _check_syntax(self, file_path: str) -> Dict[str, Any]:
        """Check syntax validity"""
        try:
            with open(file_path, 'r') as f:
                ast.parse(f.read())
            return {"valid": True}
        except SyntaxError as e:
            return {"valid": False, "error": str(e)}
            
    async def _test_functionality(
        self,
        implementation: str,
        spec: ToolSpecification
    ) -> Dict[str, Any]:
        """Test tool functionality"""
        # Create test cases based on specification
        test_cases = self._generate_test_cases(spec)
        
        passed = 0
        failed = 0
        
        # Execute implementation in isolated namespace
        namespace = {}
        exec(implementation, namespace)
        
        tool_func = namespace.get(spec.name)
        if not tool_func:
            return {"passed": 0, "failed": len(test_cases)}
            
        for test_case in test_cases:
            try:
                # Run test
                result = await tool_func(**test_case["input"])
                
                # Check result
                if self._validate_result(result, test_case["expected"], spec.return_type):
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                failed += 1
                
        return {"passed": passed, "failed": failed}
        
    def _generate_test_cases(self, spec: ToolSpecification) -> List[Dict[str, Any]]:
        """Generate test cases for tool"""
        test_cases = []
        
        # Basic test cases based on parameter types
        for param_name, param_info in spec.parameters.items():
            if param_info["type"] == "str":
                test_cases.append({
                    "input": {param_name: "test string"},
                    "expected": {"type": spec.return_type}
                })
            elif param_info["type"] == "int":
                test_cases.append({
                    "input": {param_name: 42},
                    "expected": {"type": spec.return_type}
                })
                
        return test_cases

class ToolEvolution:
    """Evolve and improve tools over time"""
    
    def __init__(self, tool_creation_engine: ToolCreationEngine):
        self.engine = tool_creation_engine
        self.evolution_history = []
        
    async def evolve_tool(
        self, 
        tool: GeneratedTool,
        feedback: Dict[str, Any]
    ) -> Optional[GeneratedTool]:
        """Evolve tool based on usage feedback"""
        # Analyze feedback
        issues = self._analyze_feedback(feedback)
        
        if not issues:
            return None
            
        # Generate improvements
        improvements = await self._generate_improvements(tool, issues)
        
        # Apply improvements
        new_implementation = await self._apply_improvements(
            tool.implementation,
            improvements
        )
        
        # Test evolved version
        test_results = await self.engine.test_framework.test_tool(
            new_implementation,
            tool.spec
        )
        
        if test_results["performance_score"] > tool.test_results["performance_score"]:
            # Create evolved tool
            evolved_tool = GeneratedTool(
                spec=tool.spec,
                implementation=new_implementation,
                test_results=test_results,
                performance_metrics=test_results["metrics"],
                created_at=datetime.utcnow()
            )
            
            # Record evolution
            self.evolution_history.append({
                "original": tool.spec.name,
                "timestamp": datetime.utcnow(),
                "improvements": improvements,
                "performance_gain": test_results["performance_score"] - tool.test_results["performance_score"]
            })
            
            return evolved_tool
            
        return None
