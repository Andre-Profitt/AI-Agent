"""
Enhanced data quality and validation mechanisms for the AI agent.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
import json

logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Levels of data quality validation."""
    BASIC = "basic"      # Minimal validation
    STANDARD = "standard"  # Standard validation
    THOROUGH = "thorough"  # Comprehensive validation
    EXTREME = "extreme"   # Maximum validation

@dataclass
class ValidationResult:
    """Result of a data validation check."""
    is_valid: bool
    issues: List[str]
    confidence: float
    suggestions: List[str]

@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    completeness: float
    accuracy: float
    consistency: float
    relevance: float
    overall_score: float

class DataQualityValidator:
    """Enhanced data quality validation system."""
    
    def __init__(self, quality_level: DataQualityLevel = DataQualityLevel.STANDARD):
        self.quality_level = quality_level
        self.validation_history = []
        self.quality_metrics_history = []
    
    def validate_input(self, input_data: Union[str, Dict[str, Any]]) -> ValidationResult:
        """Validate input data based on quality level."""
        if isinstance(input_data, str):
            return self._validate_text_input(input_data)
        elif isinstance(input_data, dict):
            return self._validate_dict_input(input_data)
        else:
            return ValidationResult(
                is_valid=False,
                issues=["Invalid input type"],
                confidence=0.0,
                suggestions=["Provide input as string or dictionary"]
            )
    
    def _validate_text_input(self, text: str) -> ValidationResult:
        """Validate text input."""
        issues = []
        suggestions = []
        confidence = 1.0
        
        # Basic validation
        if not text or not text.strip():
            issues.append("Empty input")
            suggestions.append("Provide non-empty input")
            confidence = 0.0
        
        # Length validation
        if len(text) < 3:
            issues.append("Input too short")
            suggestions.append("Provide input with at least 3 characters")
            confidence = 0.0
        
        # Character validation
        if not re.search(r'[a-zA-Z0-9]', text):
            issues.append("No alphanumeric characters")
            suggestions.append("Include letters or numbers in input")
            confidence = 0.0
        
        # Standard validation
        if self.quality_level in [DataQualityLevel.STANDARD, DataQualityLevel.THOROUGH, DataQualityLevel.EXTREME]:
            # Check for common issues
            if re.search(r'[<>]', text):
                issues.append("Contains HTML-like tags")
                suggestions.append("Remove HTML tags from input")
                confidence *= 0.8
            
            if re.search(r'[^\x00-\x7F]', text):
                issues.append("Contains non-ASCII characters")
                suggestions.append("Use ASCII characters only")
                confidence *= 0.9
        
        # Thorough validation
        if self.quality_level in [DataQualityLevel.THOROUGH, DataQualityLevel.EXTREME]:
            # Check for potential injection
            if re.search(r'[;|&]', text):
                issues.append("Contains potential command injection characters")
                suggestions.append("Remove special characters")
                confidence *= 0.7
            
            # Check for balanced brackets
            if not self._check_balanced_brackets(text):
                issues.append("Unbalanced brackets")
                suggestions.append("Ensure all brackets are properly closed")
                confidence *= 0.8
        
        # Extreme validation
        if self.quality_level == DataQualityLevel.EXTREME:
            # Check for potential SQL injection
            if re.search(r'(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)', text):
                issues.append("Contains potential SQL injection")
                suggestions.append("Remove SQL keywords")
                confidence *= 0.6
            
            # Check for potential XSS
            if re.search(r'(?i)(script|onerror|onload)', text):
                issues.append("Contains potential XSS")
                suggestions.append("Remove JavaScript-related terms")
                confidence *= 0.6
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=confidence,
            suggestions=suggestions
        )
    
    def _validate_dict_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate dictionary input."""
        issues = []
        suggestions = []
        confidence = 1.0
        
        # Basic validation
        if not data:
            issues.append("Empty dictionary")
            suggestions.append("Provide non-empty dictionary")
            confidence = 0.0
        
        # Standard validation
        if self.quality_level in [DataQualityLevel.STANDARD, DataQualityLevel.THOROUGH, DataQualityLevel.EXTREME]:
            # Check for required fields
            required_fields = ["query", "type"]
            for field in required_fields:
                if field not in data:
                    issues.append(f"Missing required field: {field}")
                    suggestions.append(f"Add {field} field")
                    confidence *= 0.8
            
            # Validate field types
            if "query" in data and not isinstance(data["query"], str):
                issues.append("query field must be string")
                suggestions.append("Convert query to string")
                confidence *= 0.8
        
        # Thorough validation
        if self.quality_level in [DataQualityLevel.THOROUGH, DataQualityLevel.EXTREME]:
            # Validate nested structures
            for key, value in data.items():
                if isinstance(value, dict):
                    nested_result = self._validate_dict_input(value)
                    if not nested_result.is_valid:
                        issues.extend([f"{key}.{issue}" for issue in nested_result.issues])
                        suggestions.extend(nested_result.suggestions)
                        confidence *= nested_result.confidence
        
        # Extreme validation
        if self.quality_level == DataQualityLevel.EXTREME:
            # Check for circular references
            try:
                json.dumps(data)
            except TypeError:
                issues.append("Contains circular references")
                suggestions.append("Remove circular references")
                confidence *= 0.7
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=confidence,
            suggestions=suggestions
        )
    
    def _check_balanced_brackets(self, text: str) -> bool:
        """Check if brackets are properly balanced."""
        brackets = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        stack = []
        
        for char in text:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def assess_quality(self, data: Union[str, Dict[str, Any]]) -> DataQualityMetrics:
        """Assess the quality of data."""
        # Validate input
        validation_result = self.validate_input(data)
        
        # Calculate metrics
        completeness = 1.0 if validation_result.is_valid else 0.0
        accuracy = validation_result.confidence
        consistency = 1.0 if not validation_result.issues else 0.5
        relevance = 1.0 if isinstance(data, str) and len(data) > 10 else 0.5
        
        # Calculate overall score
        overall_score = (completeness + accuracy + consistency + relevance) / 4
        
        metrics = DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            relevance=relevance,
            overall_score=overall_score
        )
        
        # Record metrics
        self.quality_metrics_history.append(metrics)
        
        return metrics
    
    def get_quality_trends(self) -> Dict[str, List[float]]:
        """Get trends in data quality metrics."""
        if not self.quality_metrics_history:
            return {}
        
        return {
            "completeness": [m.completeness for m in self.quality_metrics_history],
            "accuracy": [m.accuracy for m in self.quality_metrics_history],
            "consistency": [m.consistency for m in self.quality_metrics_history],
            "relevance": [m.relevance for m in self.quality_metrics_history],
            "overall_score": [m.overall_score for m in self.quality_metrics_history]
        }
    
    def get_validation_history(self) -> List[ValidationResult]:
        """Get the history of validation results."""
        return self.validation_history.copy() 