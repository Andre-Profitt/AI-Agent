from agent import query
from benchmarks.cot_performance import recommendations
from fix_security_issues import matches
from tests.load_test import success

from src.agents.enhanced_fsm import tools_used
from src.core.optimized_chain_of_thought import base_confidence
from src.gaia_components.adaptive_tool_system import available_tools
from src.infrastructure.agents.agent_factory import domain
from src.meta_cognition import base_score
from src.meta_cognition import best_capability
from src.meta_cognition import best_domain
from src.meta_cognition import characteristics
from src.meta_cognition import complexity
from src.meta_cognition import complexity_factor
from src.meta_cognition import confidence
from src.meta_cognition import current_confidence
from src.meta_cognition import domain_keywords
from src.meta_cognition import domain_scores
from src.meta_cognition import estimated_complexity
from src.meta_cognition import keywords
from src.meta_cognition import length_factor
from src.meta_cognition import performance_record
from src.meta_cognition import query_lower
from src.meta_cognition import score
from src.meta_cognition import should_clarify
from src.meta_cognition import should_request_help
from src.meta_cognition import should_use_tools
from src.meta_cognition import suggested_approach
from src.meta_cognition import tool_recommendations

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Dict, Enum, List, Tuple, actual_complexity, available_tools, base_confidence, base_score, best_capability, best_domain, capability, characteristics, complexity, complexity_factor, confidence, current_confidence, dataclass, domain, domain_keywords, domain_match, domain_scores, domain_tools, estimated_complexity, keyword, keywords, length_factor, logging, matches, meta_cognition, performance_record, query, query_lower, recommendations, record, score, should_clarify, should_request_help, should_use_tools, success, suggested_approach, term, tool_recommendations, tools_used, x
from src.tools.base_tool import tool


"""

from typing import Dict
Meta-Cognition Module for AI Agent Self-Awareness
Provides self-awareness capabilities for the AI agent to understand its own capabilities
and make better decisions about tool usage and problem-solving strategies.
"""

from typing import Tuple
from typing import Any
from typing import List

import logging

from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)

class ConfidenceLevel(str, Enum):
    """Confidence levels for meta-cognitive assessments"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class MetaCognitiveScore:
    """Score from meta-cognitive assessment"""
    confidence: float  # 0-1
    reasoning: str
    suggested_approach: str
    tool_recommendations: List[str]
    estimated_complexity: float  # 0-1
    metadata: Dict[str, Any]

class MetaCognition:
    """
    Meta-cognitive system that provides self-awareness to the AI agent.
    Helps the agent understand its capabilities and limitations.
    """
    
    def __init__(self):
        self.capability_assessments = self._initialize_capabilities()
        self.performance_history = []
        self.learning_rate = 0.1
        
    def _initialize_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize capability assessments for different domains"""
        return {
            "factual_questions": {
                "strength": 0.9,
                "confidence": 0.8,
                "tools": ["web_search", "wikipedia"],
                "description": "Strong at answering factual questions with search tools"
            },
            "analytical_reasoning": {
                "strength": 0.7,
                "confidence": 0.6,
                "tools": ["python_repl", "web_search"],
                "description": "Good at analytical tasks with computational tools"
            },
            "creative_generation": {
                "strength": 0.6,
                "confidence": 0.5,
                "tools": ["web_search"],
                "description": "Moderate at creative tasks, better with examples"
            },
            "technical_programming": {
                "strength": 0.8,
                "confidence": 0.7,
                "tools": ["python_repl", "code_analysis"],
                "description": "Strong at programming and technical tasks"
            },
            "research_analysis": {
                "strength": 0.7,
                "confidence": 0.6,
                "tools": ["web_search", "semantic_scholar", "wikipedia"],
                "description": "Good at research tasks with multiple sources"
            },
            "file_processing": {
                "strength": 0.6,
                "confidence": 0.5,
                "tools": ["file_reader", "pdf_reader"],
                "description": "Moderate at file processing tasks"
            }
        }
    
    def assess_query_capability(self, query: str, available_tools: List[str]) -> MetaCognitiveScore:
        """
        Assess the agent's capability to handle a specific query
        
        Args:
            query: The user query to assess
            available_tools: List of available tool names
            
        Returns:
            MetaCognitiveScore with assessment results
        """
        query_lower = query.lower()
        
        # Analyze query characteristics
        characteristics = self._analyze_query_characteristics(query)
        
        # Match to capability domains
        domain_scores = {}
        for domain, capability in self.capability_assessments.items():
            score = self._calculate_domain_match(query_lower, domain, capability)
            domain_scores[domain] = score
        
        # Find best matching domain
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        best_capability = self.capability_assessments[best_domain[0]]
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            best_domain[1], 
            best_capability["confidence"],
            characteristics
        )
        
        # Determine suggested approach
        suggested_approach = self._determine_approach(
            best_domain[0], 
            confidence, 
            characteristics
        )
        
        # Recommend tools
        tool_recommendations = self._recommend_tools(
            best_capability["tools"], 
            available_tools,
            characteristics
        )
        
        # Estimate complexity
        estimated_complexity = self._estimate_complexity(query, characteristics)
        
        return MetaCognitiveScore(
            confidence=confidence,
            reasoning=f"Query best matches {best_domain[0]} domain with {confidence:.2f} confidence",
            suggested_approach=suggested_approach,
            tool_recommendations=tool_recommendations,
            estimated_complexity=estimated_complexity,
            metadata={
                "best_domain": best_domain[0],
                "domain_scores": domain_scores,
                "characteristics": characteristics
            }
        )
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze characteristics of the query"""
        query_lower = query.lower()
        
        return {
            "length": len(query.split()),
            "has_question_mark": "?" in query,
            "has_technical_terms": any(term in query_lower for term in [
                "code", "program", "function", "algorithm", "api", "database"
            ]),
            "has_analytical_terms": any(term in query_lower for term in [
                "analyze", "compare", "evaluate", "assess", "examine"
            ]),
            "has_creative_terms": any(term in query_lower for term in [
                "create", "generate", "write", "design", "brainstorm"
            ]),
            "has_research_terms": any(term in query_lower for term in [
                "research", "investigate", "study", "explore", "find out"
            ]),
            "has_file_terms": any(term in query_lower for term in [
                "file", "document", "pdf", "read", "extract"
            ]),
            "complexity_indicators": sum(1 for term in [
                "complex", "detailed", "comprehensive", "multiple", "various"
            ] if term in query_lower)
        }
    
    def _calculate_domain_match(self, query: str, domain: str, capability: Dict[str, Any]) -> float:
        """Calculate how well a query matches a capability domain"""
        # Simple keyword matching for now
        domain_keywords = {
            "factual_questions": ["what", "who", "when", "where", "which", "define", "meaning"],
            "analytical_reasoning": ["analyze", "compare", "evaluate", "assess", "examine", "why", "how"],
            "creative_generation": ["create", "generate", "write", "design", "brainstorm", "imagine"],
            "technical_programming": ["code", "program", "function", "algorithm", "debug", "python", "api"],
            "research_analysis": ["research", "investigate", "study", "explore", "find out", "sources"],
            "file_processing": ["file", "document", "pdf", "read", "extract", "parse"]
        }
        
        keywords = domain_keywords.get(domain, [])
        matches = sum(1 for keyword in keywords if keyword in query)
        
        # Normalize by number of keywords
        return min(matches / len(keywords) if keywords else 0, 1.0)
    
    def _calculate_overall_confidence(
        self, 
        domain_match: float, 
        base_confidence: float, 
        characteristics: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence based on multiple factors"""
        # Base confidence from domain match and capability
        base_score = (domain_match + base_confidence) / 2
        
        # Adjust for query complexity
        complexity_factor = 1.0 - (characteristics["complexity_indicators"] * 0.1)
        
        # Adjust for query length (very short or very long queries may be harder)
        length_factor = 1.0
        if characteristics["length"] < 3:
            length_factor = 0.8  # Very short queries may be ambiguous
        elif characteristics["length"] > 50:
            length_factor = 0.9  # Very long queries may be complex
        
        return min(base_score * complexity_factor * length_factor, 1.0)
    
    def _determine_approach(self, domain: str, confidence: float, characteristics: Dict[str, Any]) -> str:
        """Determine the suggested approach based on assessment"""
        if confidence > 0.8:
            return "direct_answer"
        elif confidence > 0.6:
            return "tool_assisted"
        elif confidence > 0.4:
            return "clarification_needed"
        else:
            return "human_assistance_recommended"
    
    def _recommend_tools(
        self, 
        domain_tools: List[str], 
        available_tools: List[str],
        characteristics: Dict[str, Any]
    ) -> List[str]:
        """Recommend tools based on domain and available tools"""
        recommendations = []
        
        # Add domain-specific tools that are available
        for tool in domain_tools:
            if tool in available_tools:
                recommendations.append(tool)
        
        # Add general tools based on characteristics
        if characteristics["has_technical_terms"] and "python_repl" in available_tools:
            recommendations.append("python_repl")
        
        if characteristics["has_research_terms"] and "web_search" in available_tools:
            recommendations.append("web_search")
        
        if characteristics["has_file_terms"] and "file_reader" in available_tools:
            recommendations.append("file_reader")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _estimate_complexity(self, query: str, characteristics: Dict[str, Any]) -> float:
        """Estimate the complexity of the query"""
        complexity = 0.5  # Base complexity
        
        # Adjust based on characteristics
        if characteristics["complexity_indicators"] > 0:
            complexity += 0.2
        
        if characteristics["length"] > 20:
            complexity += 0.1
        
        if characteristics["has_technical_terms"]:
            complexity += 0.1
        
        if characteristics["has_analytical_terms"]:
            complexity += 0.1
        
        return min(complexity, 1.0)
    
    def update_performance(self, query: str, success: bool, tools_used: List[str], 
                          actual_complexity: float) -> None:
        """
        Update performance history and learn from results
        
        Args:
            query: The original query
            success: Whether the query was handled successfully
            tools_used: Tools that were actually used
            actual_complexity: Actual complexity experienced
        """
        performance_record = {
            "query": query,
            "success": success,
            "tools_used": tools_used,
            "actual_complexity": actual_complexity,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history (last 100 records)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update capability assessments based on performance
        self._learn_from_performance(performance_record)
    
    def _learn_from_performance(self, record: Dict[str, Any]) -> None:
        """Learn from performance records to improve capability assessments"""
        # Simple learning: adjust confidence based on success rate
        if record["success"]:
            # Increase confidence slightly for successful domains
            for domain in self.capability_assessments:
                if any(tool in record["tools_used"] for tool in self.capability_assessments[domain]["tools"]):
                    current_confidence = self.capability_assessments[domain]["confidence"]
                    self.capability_assessments[domain]["confidence"] = min(
                        current_confidence + self.learning_rate, 1.0
                    )
        else:
            # Decrease confidence slightly for failed attempts
            for domain in self.capability_assessments:
                if any(tool in record["tools_used"] for tool in self.capability_assessments[domain]["tools"]):
                    current_confidence = self.capability_assessments[domain]["confidence"]
                    self.capability_assessments[domain]["confidence"] = max(
                        current_confidence - self.learning_rate, 0.0
                    )

class MetaCognitiveRouter:
    """
    Router that uses meta-cognitive assessment to make decisions about
    tool usage and problem-solving strategies.
    """
    
    def __init__(self, meta_cognition: MetaCognition):
        self.meta_cognition = meta_cognition
        self.decision_thresholds = {
            "tool_usage": 0.6,
            "clarification": 0.4,
            "human_assistance": 0.2
        }
    
    def should_enter_tool_loop(self, query: str, available_tools: List[str]) -> Tuple[bool, MetaCognitiveScore]:
        """
        Determine if the agent should enter a tool-using loop
        
        Args:
            query: The user query
            available_tools: Available tools
            
        Returns:
            Tuple of (should_use_tools, meta_cognitive_score)
        """
        score = self.meta_cognition.assess_query_capability(query, available_tools)
        
        should_use_tools = score.confidence > self.decision_thresholds["tool_usage"]
        
        logger.info(
            f"Meta-cognitive routing decision: confidence={score.confidence:.2f}, "
            f"should_use_tools={should_use_tools}")
        
        
        return should_use_tools, score
    
    def should_request_clarification(self, query: str, available_tools: List[str]) -> Tuple[bool, MetaCognitiveScore]:
        """
        Determine if the agent should request clarification
        
        Args:
            query: The user query
            available_tools: Available tools
            
        Returns:
            Tuple of (should_clarify, meta_cognitive_score)
        """
        score = self.meta_cognition.assess_query_capability(query, available_tools)
        
        should_clarify = (
            score.confidence < self.decision_thresholds["tool_usage"] and
            score.confidence > self.decision_thresholds["clarification"]
        )
        
        return should_clarify, score
    
    def should_request_human_assistance(self, query: str, available_tools: List[str]) -> Tuple[bool, MetaCognitiveScore]:
        """
        Determine if the agent should request human assistance
        
        Args:
            query: The user query
            available_tools: Available tools
            
        Returns:
            Tuple of (should_request_help, meta_cognitive_score)
        """
        score = self.meta_cognition.assess_query_capability(query, available_tools)
        
        should_request_help = score.confidence < self.decision_thresholds["clarification"]
        
        return should_request_help, score
    
    def get_optimal_strategy(self, query: str, available_tools: List[str]) -> Dict[str, Any]:
        """
        Get the optimal strategy for handling a query
        
        Args:
            query: The user query
            available_tools: Available tools
            
        Returns:
            Dictionary with strategy recommendations
        """
        score = self.meta_cognition.assess_query_capability(query, available_tools)
        
        should_use_tools, _ = self.should_enter_tool_loop(query, available_tools)
        should_clarify, _ = self.should_request_clarification(query, available_tools)
        should_request_help, _ = self.should_request_human_assistance(query, available_tools)
        
        return {
            "use_tools": should_use_tools,
            "request_clarification": should_clarify,
            "request_human_assistance": should_request_help,
            "recommended_tools": score.tool_recommendations,
            "suggested_approach": score.suggested_approach,
            "confidence": score.confidence,
            "estimated_complexity": score.estimated_complexity
        } 