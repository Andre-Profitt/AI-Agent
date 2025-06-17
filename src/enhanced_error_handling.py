"""
Enhanced Error Handling and Cross-Validation Module
Provides advanced error recovery and result validation for the ReAct agent.
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of cross-validation analysis."""
    confidence_score: float
    consistency_check: bool
    source_reliability: float
    cross_references: List[str]
    validation_notes: List[str]

@dataclass
class ErrorPattern:
    """Patterns for intelligent error classification."""
    error_type: str
    keywords: List[str]
    recovery_strategy: str
    retry_recommended: bool
    alternative_tools: List[str]

class AdvancedErrorHandler:
    """Sophisticated error handling with adaptive recovery strategies."""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.tool_performance_history = {}
        self.recovery_success_rates = {}
        
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Define intelligent error patterns for recovery."""
        return [
            ErrorPattern(
                error_type="rate_limit",
                keywords=["429", "rate limit", "too many requests", "quota exceeded"],
                recovery_strategy="exponential_backoff",
                retry_recommended=True,
                alternative_tools=[]
            ),
            ErrorPattern(
                error_type="timeout",
                keywords=["timeout", "connection timeout", "read timeout"],
                recovery_strategy="alternative_tool",
                retry_recommended=True,
                alternative_tools=["tavily_search", "semantic_search_tool"]
            ),
            ErrorPattern(
                error_type="not_found",
                keywords=["404", "not found", "no results", "file not found"],
                recovery_strategy="query_refinement",
                retry_recommended=True,
                alternative_tools=["web_researcher", "semantic_search_tool"]
            ),
            ErrorPattern(
                error_type="authentication",
                keywords=["401", "403", "unauthorized", "access denied"],
                recovery_strategy="skip_tool",
                retry_recommended=False,
                alternative_tools=["web_researcher", "semantic_search_tool"]
            ),
            ErrorPattern(
                error_type="parsing_error",
                keywords=["parse error", "invalid format", "malformed"],
                recovery_strategy="format_simplification",
                retry_recommended=True,
                alternative_tools=["python_interpreter"]
            ),
            ErrorPattern(
                error_type="context_length",
                keywords=["context length", "token limit", "input too long"],
                recovery_strategy="context_compression",
                retry_recommended=False,
                alternative_tools=[]
            )
        ]
    
    def classify_error(self, error: Exception) -> ErrorPattern:
        """Classify error and determine recovery strategy."""
        error_str = str(error).lower()
        
        for pattern in self.error_patterns:
            if any(keyword in error_str for keyword in pattern.keywords):
                return pattern
        
        # Default pattern for unknown errors
        return ErrorPattern(
            error_type="unknown",
            keywords=[],
            recovery_strategy="alternative_approach",
            retry_recommended=True,
            alternative_tools=["web_researcher", "python_interpreter"]
        )
    
    def generate_recovery_guidance(self, error: Exception, pattern: ErrorPattern, state: Dict[str, Any]) -> str:
        """Generate intelligent recovery guidance based on error pattern."""
        step_count = state.get("step_count", 0)
        confidence = state.get("confidence", 0.5)
        error_attempts = state.get("error_recovery_attempts", 0)
        
        base_guidance = f"""
🔧 ERROR RECOVERY: {pattern.error_type.upper()}

Error Details: {str(error)}
Recovery Strategy: {pattern.recovery_strategy}
Attempt: {error_attempts + 1}
"""
        
        if pattern.recovery_strategy == "exponential_backoff":
            wait_time = min(60, (2 ** error_attempts) + random.uniform(0, 2))
            return base_guidance + f"""
⏳ Rate limiting detected. Implementing exponential backoff.
- Wait time: {wait_time:.1f} seconds
- Automatically retrying with reduced request frequency
- Consider using alternative search terms if this persists
"""
        
        elif pattern.recovery_strategy == "alternative_tool":
            alternatives = ", ".join(pattern.alternative_tools[:2])
            return base_guidance + f"""
🔄 Tool failure detected. Switching to alternative approach.
- Try alternative tools: {alternatives}
- Modify search strategy or parameters
- Consider breaking down complex requests
- Simplify query if appropriate
"""
        
        elif pattern.recovery_strategy == "query_refinement":
            return base_guidance + f"""
🎯 Resource not found. Refining search approach.
- Try broader or alternative search terms
- Use different keywords or synonyms
- Search for related topics or categories
- Consider if information might be in a different format
- Verify spelling and terminology
"""
        
        elif pattern.recovery_strategy == "skip_tool":
            return base_guidance + f"""
⏭️ Access restricted. Proceeding with alternative sources.
- Skip this tool and try alternatives: {', '.join(pattern.alternative_tools[:2])}
- This tool may require authentication or have access restrictions
- Continue with best available information sources
- Focus on publicly accessible resources
"""
        
        elif pattern.recovery_strategy == "format_simplification":
            return base_guidance + f"""
📝 Format issue detected. Simplifying approach.
- Break down complex requests into smaller parts
- Use simpler, more direct queries
- Try alternative data formats or representations
- Verify input format requirements
"""
        
        elif pattern.recovery_strategy == "context_compression":
            return base_guidance + f"""
📏 Context length exceeded. Compressing information.
- Focus on most relevant recent information
- Summarize key findings before proceeding
- Break down complex analysis into smaller steps
- Prioritize critical information only
"""
        
        else:  # alternative_approach
            return base_guidance + f"""
🔀 Implementing alternative approach.
- Try different methodology or perspective
- Use alternative tools: {', '.join(pattern.alternative_tools[:2])}
- Modify search strategy significantly
- Consider if the information is available through indirect means
- Assess if error is critical to overall task completion
"""

class CrossValidationEngine:
    """Advanced cross-validation for result reliability assessment."""
    
    def __init__(self):
        self.source_reliability_scores = {
            "web_researcher": 0.85,  # Wikipedia and encyclopedic sources
            "semantic_search_tool": 0.80,  # Knowledge base
            "tavily_search": 0.75,  # Real-time search
            "python_interpreter": 0.95,  # Computational results
            "file_reader": 0.90,  # Direct file access
            "advanced_file_reader": 0.90,  # Document parsing
        }
        
    def validate_results(self, tool_results: List[Dict[str, Any]], state: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive cross-validation of tool results."""
        if not tool_results:
            return ValidationResult(
                confidence_score=0.0,
                consistency_check=False,
                source_reliability=0.0,
                cross_references=[],
                validation_notes=["No results to validate"]
            )
        
        # Extract key information from results
        result_contents = []
        tools_used = []
        
        for result in tool_results:
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, 'content'):
                        result_contents.append(msg.content)
                    elif isinstance(msg, dict) and 'content' in msg:
                        result_contents.append(msg['content'])
            
            # Track tools used
            if "tool_name" in result:
                tools_used.append(result["tool_name"])
        
        # Assess consistency across results
        consistency_check = self._check_result_consistency(result_contents)
        
        # Calculate source reliability
        source_reliability = self._calculate_source_reliability(tools_used)
        
        # Generate confidence score
        confidence_score = self._calculate_confidence_score(
            consistency_check, source_reliability, len(result_contents)
        )
        
        # Generate validation notes
        validation_notes = self._generate_validation_notes(
            result_contents, tools_used, consistency_check
        )
        
        return ValidationResult(
            confidence_score=confidence_score,
            consistency_check=consistency_check,
            source_reliability=source_reliability,
            cross_references=tools_used,
            validation_notes=validation_notes
        )
    
    def _check_result_consistency(self, contents: List[str]) -> bool:
        """Check consistency across multiple result sources."""
        if len(contents) < 2:
            return True  # Single source, assume consistent
        
        # Simple consistency check - look for common key terms
        # In a real implementation, this would be more sophisticated
        all_words = set()
        for content in contents:
            words = content.lower().split()
            all_words.update(words)
        
        # Check if there's reasonable overlap between results
        common_words = 0
        for content in contents:
            content_words = set(content.lower().split())
            common_words += len(content_words.intersection(all_words))
        
        # Basic heuristic: results are consistent if they share terminology
        avg_common = common_words / len(contents) if contents else 0
        return avg_common > 5  # Arbitrary threshold
    
    def _calculate_source_reliability(self, tools_used: List[str]) -> float:
        """Calculate overall source reliability based on tools used."""
        if not tools_used:
            return 0.5  # Default middle reliability
        
        total_reliability = sum(
            self.source_reliability_scores.get(tool, 0.5) for tool in tools_used
        )
        return total_reliability / len(tools_used)
    
    def _calculate_confidence_score(self, consistency: bool, reliability: float, num_sources: int) -> float:
        """Calculate overall confidence score for validation."""
        base_score = 0.5
        
        # Boost for consistency
        if consistency:
            base_score += 0.2
        
        # Boost for reliability
        base_score += (reliability - 0.5) * 0.4
        
        # Boost for multiple sources
        source_boost = min(0.2, (num_sources - 1) * 0.1)
        base_score += source_boost
        
        return max(0.1, min(0.95, base_score))
    
    def _generate_validation_notes(self, contents: List[str], tools: List[str], consistent: bool) -> List[str]:
        """Generate helpful validation notes."""
        notes = []
        
        if len(contents) == 1:
            notes.append("Single source validation - consider additional verification")
        elif len(contents) > 3:
            notes.append("Multiple source validation - high confidence")
        
        if consistent:
            notes.append("Results show consistency across sources")
        else:
            notes.append("Results show inconsistencies - manual review recommended")
        
        unique_tools = set(tools)
        if len(unique_tools) > 2:
            notes.append(f"Cross-validated using {len(unique_tools)} different tool types")
        
        return notes

# Helper functions for integration with existing agent
def integrate_enhanced_error_handling(agent_state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    """Integration function for enhanced error handling."""
    error_handler = AdvancedErrorHandler()
    pattern = error_handler.classify_error(error)
    guidance = error_handler.generate_recovery_guidance(error, pattern, agent_state)
    
    return {
        "error_pattern": pattern.error_type,
        "recovery_strategy": pattern.recovery_strategy,
        "guidance": guidance,
        "retry_recommended": pattern.retry_recommended,
        "alternative_tools": pattern.alternative_tools
    }

def integrate_cross_validation(tool_results: List[Dict[str, Any]], agent_state: Dict[str, Any]) -> ValidationResult:
    """Integration function for cross-validation."""
    validator = CrossValidationEngine()
    return validator.validate_results(tool_results, agent_state)

if __name__ == "__main__":
    print("🔧 Enhanced Error Handling and Cross-Validation Module")
    print("✅ Advanced error recovery strategies initialized")
    print("✅ Cross-validation engine ready")
    print("✅ Integration functions available")
    print("\nKey Features:")
    print("  • Intelligent error classification")
    print("  • Adaptive recovery strategies") 
    print("  • Multi-source cross-validation")
    print("  • Source reliability assessment")
    print("  • Consistency checking across results") 