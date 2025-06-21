from agent import tools
from benchmarks.cot_performance import analysis
from benchmarks.cot_performance import recommendations
from examples.parallel_execution_example import tool_name
from fix_import_hierarchy import file_path

from src.application.tools.tool_executor import expected_type
from src.database_extended import common_errors
from src.database_extended import success_rate
from src.database_extended import total_calls
from src.query_classifier import params
from src.templates.template_factory import pattern
from src.tools_introspection import code
from src.tools_introspection import corrected_value
from src.tools_introspection import corrections
from src.tools_introspection import error
from src.tools_introspection import error_message
from src.tools_introspection import error_type
from src.tools_introspection import error_types
from src.tools_introspection import improvements
from src.tools_introspection import tool_error
from src.tools_introspection import tool_errors
from src.tools_introspection import tool_specific

from src.tools.base_tool import Tool

from src.tools.base_tool import BaseTool
# TODO: Fix undefined variables: Any, Dict, List, Optional, analysis, attempted_params, code, common_errors, converter, corrected_value, corrections, dataclass, datetime, default_value, e, error, error_message, error_type, error_types, expected_type, file_path, improvements, introspector, json, logging, max_val, min_val, param_name, param_value, params, pattern_info, pattern_name, recommendations, success_rate, tool_error, tool_errors, tool_name, tool_registry, tool_specific, tools, total_calls, x
import pattern


"""
from typing import Dict
from src.shared.types.di_types import BaseTool
# TODO: Fix undefined variables: analysis, attempted_params, code, common_errors, converter, corrected_value, corrections, default_value, e, error, error_message, error_type, error_types, expected_type, file_path, improvements, introspector, max_val, min_val, param_name, param_value, params, pattern, pattern_info, pattern_name, recommendations, self, success_rate, tool_error, tool_errors, tool_name, tool_registry, tool_specific, tools, total_calls, x

Tools Introspection Module
Provides tools for self-introspection and self-correction of the AI agent
"""

from typing import Optional
from typing import Any
from typing import List

import logging
import json

from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ToolError:
    """Represents a tool execution error"""
    tool_name: str
    error_message: str
    error_type: str
    attempted_params: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any] = None

@dataclass
class ToolAnalysis:
    """Analysis of a tool's behavior"""
    tool_name: str
    success_rate: float
    common_errors: List[str]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]

class ToolIntrospector:
    """Analyzes tool behavior and provides self-correction capabilities"""

    def __init__(self, tool_registry: Dict[str, Any]):
        self.tool_registry = tool_registry
        self.error_history = []
        self.performance_metrics = {}
        self.correction_patterns = self._build_correction_patterns()

    def _build_correction_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for common tool errors and corrections"""
        return {
            "parameter_type_error": {
                "patterns": [
                    "expected.*got.*",
                    "type.*required.*",
                    "must be.*not.*"
                ],
                "corrections": {
                    "int": lambda x: int(float(x)) if isinstance(x, (int, float, str)) else x,
                    "float": lambda x: float(x) if isinstance(x, (int, float, str)) else x,
                    "str": lambda x: str(x) if x is not None else x,
                    "bool": lambda x: bool(x) if x is not None else x,
                    "list": lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else [x],
                    "dict": lambda x: dict(x) if hasattr(x, 'items') else {"value": x}
                }
            },
            "missing_parameter": {
                "patterns": [
                    "missing.*required.*argument",
                    "required.*parameter.*not provided",
                    "argument.*is required"
                ],
                "corrections": {
                    "default_values": {
                        "query": "",
                        "text": "",
                        "url": "",
                        "file_path": "",
                        "limit": 10,
                        "timeout": 30
                    }
                }
            },
            "invalid_value": {
                "patterns": [
                    "invalid.*value",
                    "value.*not allowed",
                    "out of range"
                ],
                "corrections": {
                    "range_limits": {
                        "limit": (1, 100),
                        "timeout": (1, 300),
                        "max_tokens": (1, 4000)
                    }
                }
            }
        }

    def analyze_tool_error(self, tool_name: str, error: Exception,
                          attempted_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze a tool error and suggest corrections

        Args:
            tool_name: Name of the tool that failed
            error: The exception that occurred
            attempted_params: Parameters that were attempted

        Returns:
            Dictionary with analysis and suggested corrections
        """
        error_message = str(error)
        error_type = type(error).__name__

        # Record the error
        tool_error = ToolError(
            tool_name=tool_name,
            error_message=error_message,
            error_type=error_type,
            attempted_params=attempted_params,
            timestamp=datetime.now()
        )
        self.error_history.append(tool_error)

        # Analyze the error
        analysis = {
            "error_type": error_type,
            "error_message": error_message,
            "suggested_corrections": [],
            "confidence": 0.0
        }

        # Check for known error patterns
        for pattern_name, pattern_info in self.correction_patterns.items():
            for pattern in pattern_info["patterns"]:
                if pattern.lower() in error_message.lower():
                    corrections = self._generate_corrections(
                        pattern_name, pattern_info, attempted_params, error_message
                    )
                    analysis["suggested_corrections"].extend(corrections)
                    analysis["confidence"] = max(analysis["confidence"], 0.7)

        # Add tool-specific corrections
        tool_specific = self._get_tool_specific_corrections(tool_name, error_message, attempted_params)
        if tool_specific:
            analysis["suggested_corrections"].extend(tool_specific)
            analysis["confidence"] = max(analysis["confidence"], 0.8)

        logger.info(f"Analyzed tool error for {tool_name}: {error_type}")
        return analysis if analysis["suggested_corrections"] else None

    def _generate_corrections(self, pattern_name: str, pattern_info: Dict[str, Any],
                            attempted_params: Dict[str, Any], error_message: str) -> List[Dict[str, Any]]:
        """Generate corrections based on error pattern"""
        corrections = []

        if pattern_name == "parameter_type_error":
            # Try to identify the parameter and correct its type
            for param_name, param_value in attempted_params.items():
                for expected_type, converter in pattern_info["corrections"].items():
                    if expected_type in error_message.lower():
                        try:
                            corrected_value = converter(param_value)
                            corrections.append({
                                "parameter": param_name,
                                "original_value": param_value,
                                "corrected_value": corrected_value,
                                "reason": f"Type conversion: {type(param_value).__name__} -> {expected_type}",
                                "confidence": 0.8
                            })
                        except (ValueError, TypeError):
                            continue

        elif pattern_name == "missing_parameter":
            # Suggest default values for missing parameters
            for param_name, default_value in pattern_info["corrections"]["default_values"].items():
                if param_name not in attempted_params:
                    corrections.append({
                        "parameter": param_name,
                        "original_value": None,
                        "corrected_value": default_value,
                        "reason": f"Missing required parameter, using default: {default_value}",
                        "confidence": 0.6
                    })

        elif pattern_name == "invalid_value":
            # Suggest value corrections based on limits
            for param_name, param_value in attempted_params.items():
                if param_name in pattern_info["corrections"]["range_limits"]:
                    min_val, max_val = pattern_info["corrections"]["range_limits"][param_name]
                    if isinstance(param_value, (int, float)):
                        if param_value < min_val:
                            corrections.append({
                                "parameter": param_name,
                                "original_value": param_value,
                                "corrected_value": min_val,
                                "reason": f"Value below minimum ({min_val})",
                                "confidence": 0.7
                            })
                        elif param_value > max_val:
                            corrections.append({
                                "parameter": param_name,
                                "original_value": param_value,
                                "corrected_value": max_val,
                                "reason": f"Value above maximum ({max_val})",
                                "confidence": 0.7
                            })

        return corrections

    def _get_tool_specific_corrections(self, tool_name: str, error_message: str,
                                     attempted_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get tool-specific correction suggestions"""
        corrections = []

        # Web search tool corrections
        if "web_search" in tool_name.lower():
            if "query" in attempted_params and not attempted_params["query"].strip():
                corrections.append({
                    "parameter": "query",
                    "original_value": attempted_params["query"],
                    "corrected_value": "general information",
                    "reason": "Empty query provided",
                    "confidence": 0.9
                })

        # File reader tool corrections
        elif "file" in tool_name.lower() or "read" in tool_name.lower():
            if "file_path" in attempted_params:
                file_path = attempted_params["file_path"]
                if not file_path.endswith(('.txt', '.pdf', '.csv', '.json')):
                    corrections.append({
                        "parameter": "file_path",
                        "original_value": file_path,
                        "corrected_value": f"{file_path}.txt",
                        "reason": "File extension may be missing",
                        "confidence": 0.6
                    })

        # Python interpreter corrections
        elif "python" in tool_name.lower() or "code" in tool_name.lower():
            if "code" in attempted_params:
                code = attempted_params["code"]
                if not code.strip():
                    corrections.append({
                        "parameter": "code",
                        "original_value": code,
                        "corrected_value": "print('Hello, World!')",
                        "reason": "Empty code provided",
                        "confidence": 0.8
                    })

        return corrections

    def get_tool_performance_analysis(self, tool_name: str) -> Optional[ToolAnalysis]:
        """Get performance analysis for a specific tool"""
        if tool_name not in self.tool_registry:
            return None

        # Get error history for this tool
        tool_errors = [e for e in self.error_history if e.tool_name == tool_name]

        if not tool_errors:
            return ToolAnalysis(
                tool_name=tool_name,
                success_rate=1.0,
                common_errors=[],
                performance_metrics={},
                recommendations=["Tool has no recorded errors"]
            )

        # Calculate success rate (assuming we have total calls)
        total_calls = len(tool_errors) * 10  # Rough estimate
        success_rate = max(0.0, 1.0 - (len(tool_errors) / total_calls))

        # Analyze common errors
        error_types = {}
        for error in tool_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1

        common_errors = [error_type for error_type, count in
                        sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]]

        # Generate recommendations
        recommendations = []
        if success_rate < 0.8:
            recommendations.append("Consider adding more parameter validation")

        if "TypeError" in common_errors:
            recommendations.append("Add type checking for parameters")

        if "ValueError" in common_errors:
            recommendations.append("Add value range validation")

        return ToolAnalysis(
            tool_name=tool_name,
            success_rate=success_rate,
            common_errors=common_errors,
            performance_metrics={
                "total_errors": len(tool_errors),
                "error_types": error_types
            },
            recommendations=recommendations
        )

    def suggest_tool_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements for all tools"""
        improvements = []

        for tool_name in self.tool_registry:
            analysis = self.get_tool_performance_analysis(tool_name)
            if analysis and analysis.success_rate < 0.9:
                improvements.append({
                    "tool_name": tool_name,
                    "success_rate": analysis.success_rate,
                    "recommendations": analysis.recommendations,
                    "priority": "high" if analysis.success_rate < 0.7 else "medium"
                })

        return sorted(improvements, key=lambda x: x["success_rate"])

# Introspection tools
def get_introspection_tools() -> List[Any]:
    """Get list of introspection tools"""
    tools = []

    try:
        from langchain.tools import BaseTool

        class AnalyzeToolErrorTool(BaseTool):
            name = "analyze_tool_error"
            description = "Analyze a tool error and suggest corrections"

            def __init__(self, introspector: ToolIntrospector):
                super().__init__()
                self.introspector = introspector

            def _run(self, tool_name: str, error_message: str, attempted_params: str) -> str:
                """Analyze a tool error"""
                try:
                    params = json.loads(attempted_params) if isinstance(attempted_params, str) else attempted_params
                except json.JSONDecodeError:
                    params = {"raw_params": attempted_params}

                # Create a mock exception
                class MockError(Exception):
                    pass

                error = MockError(error_message)
                analysis = self.introspector.analyze_tool_error(tool_name, error, params)

                if analysis:
                    return json.dumps(analysis, indent=2)
                else:
                    return "No specific corrections found for this error"

            def _arun(self, tool_name: str, error_message: str, attempted_params: str) -> str:
                return self._run(tool_name, error_message, attempted_params)

        class GetToolPerformanceTool(BaseTool):
            name = "get_tool_performance"
            description = "Get performance analysis for a specific tool"

            def __init__(self, introspector: ToolIntrospector):
                super().__init__()
                self.introspector = introspector

            def _run(self, tool_name: str) -> str:
                """Get tool performance analysis"""
                analysis = self.introspector.get_tool_performance_analysis(tool_name)

                if analysis:
                    return json.dumps({
                        "tool_name": analysis.tool_name,
                        "success_rate": analysis.success_rate,
                        "common_errors": analysis.common_errors,
                        "recommendations": analysis.recommendations
                    }, indent=2)
                else:
                    return f"No performance data available for tool: {tool_name}"

            def _arun(self, tool_name: str) -> str:
                return self._run(tool_name)

        class SuggestImprovementsTool(BaseTool):
            name = "suggest_tool_improvements"
            description = "Suggest improvements for all tools based on error analysis"

            def __init__(self, introspector: ToolIntrospector):
                super().__init__()
                self.introspector = introspector

            def _run(self) -> str:
                """Suggest tool improvements"""
                improvements = self.introspector.suggest_tool_improvements()
                return json.dumps(improvements, indent=2)

            def _arun(self) -> str:
                return self._run()

        # These tools will be instantiated with an introspector when needed
        tools.extend([
            AnalyzeToolErrorTool,
            GetToolPerformanceTool,
            SuggestImprovementsTool
        ])

    except ImportError:
        logger.warning("LangChain BaseTool not available, skipping introspection tools")

    return tools

# Global introspection tools (will be populated when introspector is available)
INTROSPECTION_TOOLS = []