"""
Meta-Cognition Module for Tool Use Decisions
Implements MeCo-inspired self-awareness of agent capabilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from pydantic import BaseModel, Field
import logging
from typing import Optional, Dict, Any, List, Union, Tuple

logger = logging.getLogger(__name__)



class MetaCognitiveScore(BaseModel):
    """Meta-cognitive assessment of capability"""
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in internal knowledge")
    uncertainty_areas: List[str] = Field(default_factory=list, description="Areas of uncertainty")
    recommended_tools: List[str] = Field(default_factory=list, description="Tools that could help")
    reasoning: str = Field(description="Explanation of assessment")


@dataclass 
class KnowledgeDomain:
    """Represents a knowledge domain with confidence levels"""
    name: str
    base_confidence: float
    decay_rate: float = 0.1  # How quickly knowledge becomes outdated
    last_updated: str = "2024-04"  # Training cutoff approximation


class MetaCognition:
    """
    Assesses agent's self-awareness of its capabilities
    Helps decide when to rely on internal knowledge vs external tools
    """
    
    # Knowledge domains with base confidence levels
    KNOWLEDGE_DOMAINS = {
        "general_facts": KnowledgeDomain("general_facts", 0.9),
        "current_events": KnowledgeDomain("current_events", 0.2, decay_rate=0.5),
        "programming": KnowledgeDomain("programming", 0.85),
        "mathematics": KnowledgeDomain("mathematics", 0.9),
        "science": KnowledgeDomain("science", 0.8),
        "literature": KnowledgeDomain("literature", 0.75),
        "real_time_data": KnowledgeDomain("real_time_data", 0.0),  # Always needs tools
        "personal_context": KnowledgeDomain("personal_context", 0.0),  # User-specific
        "local_files": KnowledgeDomain("local_files", 0.0),  # Needs file access
    }
    
    # Patterns that indicate need for external tools
    TOOL_INDICATORS = {
        "web_search": [
            "latest", "current", "today", "recent", "news", "update",
            "2024", "2025", "this year", "last week", "yesterday"
        ],
        "calculator": [
            "calculate", "compute", "solve", "equation", "integral",
            "derivative", "statistics", "probability"
        ],
        "file_access": [
            "file", "document", "read", "analyze", "csv", "pdf", "xlsx",
            "open", "load", "parse"
        ],
        "code_execution": [
            "run", "execute", "simulate", "plot", "visualize", "graph",
            "data analysis", "machine learning", "train model"
        ],
        "database": [
            "database", "query", "sql", "records", "stored", "retrieve"
        ]
    }
    
    def __init__(self) -> None:
        """Initialize meta-cognition module"""
        self.domain_patterns = self._compile_domain_patterns()
    
    def assess_capability(self, query: str, available_tools: List[str]) -> MetaCognitiveScore:
        """
        Assess confidence in answering query with internal knowledge
        
        Args:
            query: User query to assess
            available_tools: List of available tool names
            
        Returns:
            MetaCognitiveScore with confidence and recommendations
        """
        query_lower = query.lower()
        
        # Identify relevant knowledge domains
        domains = self._identify_domains(query_lower)
        
        # Calculate base confidence
        base_confidence = self._calculate_base_confidence(domains)
        
        # Check for tool indicators
        tool_needs = self._identify_tool_needs(query_lower)
        
        # Adjust confidence based on tool needs
        adjusted_confidence = base_confidence
        uncertainty_areas = []
        
        if tool_needs:
            # Reduce confidence if tools are needed
            adjusted_confidence *= 0.5
            uncertainty_areas.extend([f"Requires {tool}" for tool in tool_needs])
        
        # Check temporal sensitivity
        if self._is_temporally_sensitive(query_lower):
            adjusted_confidence *= 0.3
            uncertainty_areas.append("Time-sensitive information")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            domains, base_confidence, adjusted_confidence, tool_needs
        )
        
        # Filter to available tools
        recommended_tools = [tool for tool in tool_needs if tool in available_tools]
        
        return MetaCognitiveScore(
            confidence=adjusted_confidence,
            uncertainty_areas=uncertainty_areas,
            recommended_tools=recommended_tools,
            reasoning=reasoning
        )
    
    def should_use_tools(self, score: MetaCognitiveScore, threshold: float = 0.7) -> bool:
        """
        Decide whether to use tools based on meta-cognitive assessment
        
        Args:
            score: Meta-cognitive assessment
            threshold: Confidence threshold below which tools should be used
            
        Returns:
            True if tools should be used, False otherwise
        """
        return score.confidence < threshold or len(score.recommended_tools) > 0
    
    def _compile_domain_patterns(self) -> Dict[str, List[str]]:
        """Compile patterns for identifying knowledge domains"""
        return {
            "general_facts": ["what is", "who is", "define", "meaning of"],
            "current_events": ["news", "latest", "recent", "happening now"],
            "programming": ["python", "code", "function", "algorithm", "programming"],
            "mathematics": ["math", "calculate", "equation", "formula", "prove"],
            "science": ["physics", "chemistry", "biology", "scientific", "research"],
            "literature": ["book", "author", "novel", "poem", "literary"],
            "real_time_data": ["stock price", "weather", "traffic", "live"],
            "personal_context": ["my", "i", "remember when", "last time"],
            "local_files": ["file", "document", "folder", "directory"]
        }
    
    def _identify_domains(self, query: str) -> List[KnowledgeDomain]:
        """Identify relevant knowledge domains for the query"""
        domains = []
        
        for domain_name, patterns in self.domain_patterns.items():
            if any(pattern in query for pattern in patterns):
                if domain_name in self.KNOWLEDGE_DOMAINS:
                    domains.append(self.KNOWLEDGE_DOMAINS[domain_name])
        
        # Default to general facts if no specific domain identified
        if not domains:
            domains.append(self.KNOWLEDGE_DOMAINS["general_facts"])
        
        return domains
    
    def _calculate_base_confidence(self, domains: List[KnowledgeDomain]) -> float:
        """Calculate base confidence from identified domains"""
        if not domains:
            return 0.5
        
        # Weighted average of domain confidences
        total_confidence = sum(d.base_confidence for d in domains)
        return total_confidence / len(domains)
    
    def _identify_tool_needs(self, query: str) -> List[str]:
        """Identify which tools might be needed"""
        needed_tools = []
        
        for tool, indicators in self.TOOL_INDICATORS.items():
            if any(indicator in query for indicator in indicators):
                needed_tools.append(tool)
        
        return needed_tools
    
    def _is_temporally_sensitive(self, query: str) -> bool:
        """Check if query is about time-sensitive information"""
        temporal_indicators = [
            "today", "yesterday", "tomorrow", "this week", "last week",
            "this month", "this year", "current", "latest", "now",
            "recent", "2024", "2025"
        ]
        return any(indicator in query for indicator in temporal_indicators)
    
    def _generate_reasoning(
        self, 
        domains: List[KnowledgeDomain],
        base_confidence: float,
        adjusted_confidence: float,
        tool_needs: List[str]
    ) -> str:
        """Generate human-readable reasoning for the assessment"""
        domain_names = [d.name for d in domains]
        
        reasoning_parts = [
            f"Query relates to domains: {', '.join(domain_names)}.",
            f"Base confidence: {base_confidence:.2f}."
        ]
        
        if tool_needs:
            reasoning_parts.append(
                f"Tools recommended: {', '.join(tool_needs)} would provide more accurate results."
            )
        
        if adjusted_confidence < base_confidence:
            reasoning_parts.append(
                f"Confidence adjusted to {adjusted_confidence:.2f} due to external data needs."
            )
        
        return " ".join(reasoning_parts)
    
    def get_tool_confidence_boost(self, tool_name: str) -> float:
        """
        Get confidence boost from using a specific tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Confidence boost factor (multiplier)
        """
        tool_boosts = {
            "web_search": 2.0,  # Doubles confidence for current info
            "calculator": 1.5,  # Improves calculation confidence  
            "file_access": 3.0,  # Essential for file-based queries
            "code_execution": 2.5,  # Critical for data analysis
            "database": 2.0  # Important for stored data
        }
        return tool_boosts.get(tool_name, 1.0)


# Integration helper for FSM
class MetaCognitiveRouter:
    """Routes decisions based on meta-cognitive assessment"""
    
    def __init__(self, meta_cognition: MetaCognition, confidence_threshold: float = 0.7) -> None:
        self.meta_cognition = meta_cognition
        self.confidence_threshold = confidence_threshold
    
    def should_enter_tool_loop(
        self, 
        query: str, 
        available_tools: List[str]
    ) -> Tuple[bool, MetaCognitiveScore]:
        """
        Determine if agent should enter tool-using loop
        
        Returns:
            Tuple of (should_use_tools, meta_cognitive_score)
        """
        score = self.meta_cognition.assess_capability(query, available_tools)
        should_use = self.meta_cognition.should_use_tools(score, self.confidence_threshold)
        
        return should_use, score


# Example usage
if __name__ == "__main__":
    mc = MetaCognition()
    router = MetaCognitiveRouter(mc)
    
    test_queries = [
        "What is the capital of France?",
        "What's the latest news about AI developments?",
        "Calculate the integral of x^2 from 0 to 10",
        "Analyze the data in sales.csv and create a visualization"
    ]
    
    available_tools = ["web_search", "calculator", "file_access", "code_execution"]
    
    for query in test_queries:
        should_use, score = router.should_enter_tool_loop(query, available_tools)
        logger.info("\nQuery: {}", extra={"query": query})
        logger.info("Confidence: {}", extra={"score_confidence": score.confidence})
        logger.info("Should use tools: {}", extra={"should_use": should_use})
        logger.info("Reasoning: {}", extra={"score_reasoning": score.reasoning})
        if score.recommended_tools:
            logger.info("Recommended tools: {}", extra={"_____join_score_recommended_tools_": ', '.join(score.recommended_tools)}) 