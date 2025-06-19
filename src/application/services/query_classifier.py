"""
Query Classification System for Meta-Cognitive Control
Classifies user queries to dynamically adjust agent parameters
"""

import json
import time
from typing import Dict, Literal, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import anthropic
from pydantic import BaseModel, Field

from config import config


class QueryCategory(str, Enum):
    """Categories for query classification"""
    SIMPLE_LOOKUP = "simple_lookup"
    MULTI_STEP_RESEARCH = "multi_step_research"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_GENERATION = "creative_generation"


class QueryClassification(BaseModel):
    """Structured output for query classification"""
    category: QueryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    estimated_complexity: Literal["low", "medium", "high"]
    requires_tools: bool
    requires_code_execution: bool


@dataclass
class OperationalParameters:
    """Dynamic operational parameters based on query classification"""
    model_preference: str
    verification_level: int
    max_reasoning_steps: int
    security_context: Literal["standard", "high"]
    enable_parallelism: bool
    cache_strategy: Literal["aggressive", "moderate", "minimal"]


class QueryClassifier:
    """
    Classifies queries and determines optimal operational parameters
    Uses a fast LLM router for efficient classification
    """
    
    # Configuration map for each query category
    CATEGORY_CONFIG: Dict[QueryCategory, OperationalParameters] = {
        QueryCategory.SIMPLE_LOOKUP: OperationalParameters(
            model_preference="speed",  # Claude 3 Haiku
            verification_level=1,
            max_reasoning_steps=3,
            security_context="standard",
            enable_parallelism=False,
            cache_strategy="aggressive"
        ),
        QueryCategory.MULTI_STEP_RESEARCH: OperationalParameters(
            model_preference="quality",  # Claude 3.5 Sonnet
            verification_level=3,
            max_reasoning_steps=15,
            security_context="standard",
            enable_parallelism=True,
            cache_strategy="moderate"
        ),
        QueryCategory.DATA_ANALYSIS: OperationalParameters(
            model_preference="sonnet",  # Claude 3.5 Sonnet
            verification_level=2,
            max_reasoning_steps=10,
            security_context="high",  # Requires secure sandbox
            enable_parallelism=True,
            cache_strategy="minimal"
        ),
        QueryCategory.CREATIVE_GENERATION: OperationalParameters(
            model_preference="opus",  # Claude 3 Opus
            verification_level=0,
            max_reasoning_steps=5,
            security_context="standard",
            enable_parallelism=False,
            cache_strategy="minimal"
        )
    }
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize the query classifier"""
        # Access the API key through the config object's api attribute
        self.api_key = anthropic_api_key or getattr(config.api, 'ANTHROPIC_API_KEY', None) or config.api.OPENAI_API_KEY
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
    
    def classify_query(self, query: str) -> Tuple[QueryClassification, OperationalParameters]:
        """
        Classify a user query and return classification + operational parameters
        
        Args:
            query: The user's input query
            
        Returns:
            Tuple of (classification, operational_parameters)
        """
        start_time = time.time()
        
        # If no API client, use heuristic classification
        if not self.client:
            classification = self._heuristic_classification(query)
        else:
            classification = self._llm_classification(query)
        
        # Get operational parameters based on classification
        parameters = self.CATEGORY_CONFIG[classification.category]
        
        # Log classification latency
        latency = (time.time() - start_time) * 1000
        print(f"Query classification completed in {latency:.1f}ms")
        
        return classification, parameters
    
    def _llm_classification(self, query: str) -> QueryClassification:
        """Use Claude 3 Haiku for fast, accurate classification"""
        
        classification_prompt = f"""You are a query classifier for an AI agent. Analyze the user query and classify it into ONE of these categories:

1. simple_lookup: Simple, fact-based questions or direct data retrieval that can be answered quickly
2. multi_step_research: Complex queries requiring web searches, data synthesis, and deep reasoning
3. data_analysis: Queries involving code execution, data manipulation, calculations, or visualization
4. creative_generation: Open-ended tasks like writing stories, brainstorming, or generating creative content

Also assess:
- confidence: Your confidence in the classification (0.0 to 1.0)
- reasoning: Brief explanation of why you chose this category
- estimated_complexity: low/medium/high
- requires_tools: Whether external tools (web search, calculator, etc.) are needed
- requires_code_execution: Whether Python code execution is likely needed

User query: "{query}"

Respond with a JSON object containing all fields."""
        
        try:
            # Use Claude 3 Haiku for speed
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": classification_prompt
                }]
            )
            
            # Parse the response
            result = json.loads(response.content[0].text)
            
            # Validate and create classification
            return QueryClassification(
                category=QueryCategory(result["category"]),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                estimated_complexity=result["estimated_complexity"],
                requires_tools=result["requires_tools"],
                requires_code_execution=result["requires_code_execution"]
            )
            
        except Exception as e:
            print(f"LLM classification failed: {e}, falling back to heuristic")
            return self._heuristic_classification(query)
    
    def _heuristic_classification(self, query: str) -> QueryClassification:
        """Fallback heuristic classification when LLM is unavailable"""
        query_lower = query.lower()
        
        # Check for data analysis indicators
        data_keywords = ["analyze", "calculate", "plot", "graph", "data", "csv", "excel", 
                        "statistics", "correlation", "regression", "visualize"]
        if any(keyword in query_lower for keyword in data_keywords):
            return QueryClassification(
                category=QueryCategory.DATA_ANALYSIS,
                confidence=0.7,
                reasoning="Query contains data analysis keywords",
                estimated_complexity="medium",
                requires_tools=True,
                requires_code_execution=True
            )
        
        # Check for creative generation
        creative_keywords = ["write", "create", "generate", "story", "poem", "brainstorm",
                           "imagine", "design", "compose", "invent"]
        if any(keyword in query_lower for keyword in creative_keywords):
            return QueryClassification(
                category=QueryCategory.CREATIVE_GENERATION,
                confidence=0.7,
                reasoning="Query contains creative generation keywords",
                estimated_complexity="medium",
                requires_tools=False,
                requires_code_execution=False
            )
        
        # Check for research indicators
        research_keywords = ["research", "find information", "tell me about", "explain",
                           "how does", "why does", "compare", "investigate"]
        if any(keyword in query_lower for keyword in research_keywords) or "?" in query:
            if len(query.split()) > 10:  # Longer queries likely need research
                return QueryClassification(
                    category=QueryCategory.MULTI_STEP_RESEARCH,
                    confidence=0.6,
                    reasoning="Query appears to require research or investigation",
                    estimated_complexity="high",
                    requires_tools=True,
                    requires_code_execution=False
                )
        
        # Default to simple lookup
        return QueryClassification(
            category=QueryCategory.SIMPLE_LOOKUP,
            confidence=0.5,
            reasoning="Query appears to be a simple factual question",
            estimated_complexity="low",
            requires_tools=False,
            requires_code_execution=False
        )
    
    def get_security_requirements(self, classification: QueryClassification) -> Dict[str, any]:
        """
        Determine security requirements based on classification
        
        Returns:
            Dictionary with security configuration
        """
        if classification.requires_code_execution:
            return {
                "sandbox_required": True,
                "network_access": False,
                "filesystem_access": "read_only",
                "max_execution_time": 10,  # seconds
                "max_memory_mb": 100,
                "allow_imports": ["numpy", "pandas", "matplotlib", "seaborn"]
            }
        else:
            return {
                "sandbox_required": False,
                "network_access": True,
                "filesystem_access": "none",
                "max_execution_time": None,
                "max_memory_mb": None,
                "allow_imports": []
            }


# Example usage
if __name__ == "__main__":
    classifier = QueryClassifier()
    
    test_queries = [
        "What is the capital of France?",
        "Analyze the correlation between temperature and ice cream sales in this CSV file",
        "Research the latest developments in quantum computing and summarize the key breakthroughs",
        "Write a short story about a robot learning to paint"
    ]
    
    for query in test_queries:
        classification, params = classifier.classify_query(query)
        print(f"\nQuery: {query}")
        print(f"Category: {classification.category}")
        print(f"Confidence: {classification.confidence}")
        print(f"Parameters: {params}") 