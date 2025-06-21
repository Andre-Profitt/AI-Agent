from agent import query
from benchmarks.cot_performance import recommendations
from fix_security_issues import patterns

from src.database.models import tool
from src.meta_cognition import complexity
from src.meta_cognition import query_lower
from src.meta_cognition import score
from src.query_classifier import base_tokens
from src.query_classifier import category_multipliers
from src.query_classifier import category_scores
from src.query_classifier import depth_score
from src.query_classifier import doc
from src.query_classifier import entities
from src.query_classifier import entity_score
from src.query_classifier import indicator_score
from src.query_classifier import length_score
from src.query_classifier import multi_part
from src.query_classifier import multiplier
from src.query_classifier import params
from src.query_classifier import question_marks
from src.query_classifier import required_tools
from src.query_classifier import sentence_score
from src.query_classifier import sentences
from src.query_classifier import word_count
from src.templates.template_factory import pattern
from src.tools.registry import category
from src.utils.tools_introspection import field

"""
from typing import Dict
# TODO: Fix undefined variables: Any, Dict, Enum, List, Optional, base_tokens, category, category_multipliers, category_scores, complexity, context, dataclass, defaultdict, depth_score, doc, entities, entity_score, field, file, indicator, indicator_score, indicators, length_score, logging, multi_part, multiplier, params, patterns, query, query_lower, question_marks, re, recommendations, required_tools, score, sentence_score, sentences, word_count, x
import pattern

from src.tools.base_tool import tool

# TODO: Fix undefined variables: base_tokens, category, category_multipliers, category_scores, complexity, context, depth_score, doc, entities, entity_score, file, indicator, indicator_score, indicators, length_score, multi_part, multiplier, params, pattern, patterns, query, query_lower, question_marks, recommendations, required_tools, score, self, sentence_score, sentences, spacy, tool, word_count, x

Query Classifier for Production GAIA System
Classifies queries and determines operational parameters
"""

from typing import Optional
from dataclasses import field
from typing import Any
from typing import List

import re
import logging
from enum import Enum

from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to load spacy model, fallback to simple classification if not available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    logger.warning("SpaCy model not available, using simple classification")
    SPACY_AVAILABLE = False
    nlp = None

class QueryCategory(str, Enum):
    """Categories of queries for routing"""
    FACTUAL = "factual"              # Simple fact-based questions
    ANALYTICAL = "analytical"        # Analysis, comparison, evaluation
    CREATIVE = "creative"            # Generation, brainstorming
    TECHNICAL = "technical"          # Programming, technical questions
    RESEARCH = "research"            # In-depth research queries
    CALCULATION = "calculation"      # Math, calculations
    FILE_ANALYSIS = "file_analysis"  # Document/file processing
    CONVERSATIONAL = "conversational" # Casual chat
    COMPLEX = "complex"              # Multi-step, complex reasoning
    GAIA_BENCHMARK = "gaia_benchmark" # GAIA-specific queries

@dataclass
class OperationalParameters:
    """Parameters for query execution"""
    category: QueryCategory
    complexity_score: float = 0.5  # 0-1, higher = more complex
    estimated_tokens: int = 1000
    recommended_model: str = "llama-3.3-70b-versatile"
    quality_level: str = "BALANCED"  # BASIC, BALANCED, THOROUGH
    reasoning_type: str = "SIMPLE"   # SIMPLE, LAYERED, RECURSIVE
    use_multi_agent: bool = False
    required_tools: List[str] = field(default_factory=list)
    verification_level: str = "basic"  # basic, thorough, exhaustive
    max_iterations: int = 10
    confidence_threshold: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryClassifier:
    """Classifies queries and determines execution parameters"""

    def __init__(self):
        self.classification_patterns = self._build_patterns()
        self.tool_indicators = self._build_tool_indicators()
        self.complexity_indicators = self._build_complexity_indicators()

    def _build_patterns(self) -> Dict[QueryCategory, List[re.Pattern]]:
        """Build regex patterns for query classification"""
        return {
            QueryCategory.FACTUAL: [
                re.compile(r'\b(what|who|when|where|which)\s+(is|are|was|were)\b', re.I),
                re.compile(r'\b(define|meaning of|definition)\b', re.I),
                re.compile(r'\b(capital of|population of|located in)\b', re.I),
            ],
            QueryCategory.ANALYTICAL: [
                re.compile(r'\b(analyze|compare|evaluate|assess|examine)\b', re.I),
                re.compile(r'\b(pros and cons|advantages|disadvantages|differences)\b', re.I),
                re.compile(r'\b(impact|effect|influence|relationship)\b', re.I),
            ],
            QueryCategory.CREATIVE: [
                re.compile(r'\b(create|generate|write|design|brainstorm)\b', re.I),
                re.compile(r'\b(story|poem|script|idea|concept)\b', re.I),
                re.compile(r'\b(imagine|invent|develop)\b', re.I),
            ],
            QueryCategory.TECHNICAL: [
                re.compile(r'\b(code|program|function|algorithm|debug)\b', re.I),
                re.compile(r'\b(python|javascript|java|c\+\+|sql)\b', re.I),
                re.compile(r'\b(api|database|server|frontend|backend)\b', re.I),
            ],
            QueryCategory.RESEARCH: [
                re.compile(r'\b(research|investigate|study|explore|find out)\b', re.I),
                re.compile(r'\b(comprehensive|detailed|in-depth|thorough)\b', re.I),
                re.compile(r'\b(sources|references|citations|evidence)\b', re.I),
            ],
            QueryCategory.CALCULATION: [
                re.compile(r'\b(calculate|compute|solve|math)\b', re.I),
                re.compile(r'\b\d+[\+\-\*/]\d+\b'),  # Basic math operations
                re.compile(r'\b(equation|formula|integral|derivative)\b', re.I),
            ],
            QueryCategory.FILE_ANALYSIS: [
                re.compile(r'\b(file|document|pdf|excel|csv|image)\b', re.I),
                re.compile(r'\b(read|extract|parse|analyze)\s+.*(file|document)\b', re.I),
                re.compile(r'\.(pdf|xlsx|csv|txt|docx|png|jpg)\b', re.I),
            ],
            QueryCategory.GAIA_BENCHMARK: [
                re.compile(r'\bgaia\b', re.I),
                re.compile(r'benchmark.*question', re.I),
                re.compile(r'<<<.*>>>', re.I),  # GAIA answer format
            ]
        }

    def _build_tool_indicators(self) -> Dict[str, List[str]]:
        """Build indicators for required tools"""
        return {
            "web_search_tool": ["search", "find online", "current", "latest", "news"],
            "python_repl_tool": ["calculate", "compute", "solve", "math", "equation"],
            "file_read_tool": ["read file", "open file", "load", "extract from"],
            "semantic_scholar_tool": ["research", "papers", "academic", "scholarly"],
            "wikipedia_tool": ["wikipedia", "wiki", "encyclopedia"],
            "pdf_reader_tool": ["pdf", "document", "read pdf"],
            "audio_transcription_tool": ["audio", "transcribe", "speech", "listen"],
            "image_analysis_tool": ["image", "picture", "photo", "analyze image"],
        }

    def _build_complexity_indicators(self) -> Dict[str, float]:
        """Build complexity scoring indicators"""
        return {
            # Low complexity (0.1-0.3)
            "simple": 0.1, "basic": 0.1, "what is": 0.1, "define": 0.1,

            # Medium complexity (0.4-0.6)
            "explain": 0.4, "describe": 0.4, "how does": 0.5, "why": 0.5,

            # High complexity (0.7-0.9)
            "analyze": 0.7, "compare": 0.7, "evaluate": 0.8, "synthesize": 0.8,
            "comprehensive": 0.9, "detailed": 0.8, "in-depth": 0.9,

            # Multi-step indicators
            "step by step": 0.8, "multiple": 0.7, "various": 0.6,
            "then": 0.6, "after that": 0.6, "finally": 0.6,
        }

    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> OperationalParameters:
        """
        Classify query and determine operational parameters

        Args:
            query: The user query to classify
            context: Optional context (files, previous messages, etc.)

        Returns:
            OperationalParameters with classification and execution settings
        """
        query_lower = query.lower()

        # Start with default parameters
        params = OperationalParameters(
            category=QueryCategory.FACTUAL,
            metadata={"original_query": query}
        )

        # Detect category
        category_scores = defaultdict(float)

        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    category_scores[category] += 1.0

        # Set primary category
        if category_scores:
            params.category = max(category_scores.items(), key=lambda x: x[1])[0]

        # Calculate complexity
        params.complexity_score = self._calculate_complexity(query)

        # Determine required tools
        params.required_tools = self._identify_required_tools(query, context)

        # Set operational parameters based on classification
        self._set_execution_parameters(params, query, context)

        # Check for GAIA-specific handling
        if self._is_gaia_query(query, context):
            params.category = QueryCategory.GAIA_BENCHMARK
            params.quality_level = "THOROUGH"
            params.verification_level = "thorough"

        logger.info(f"Query classified as {params.category} with complexity {params.complexity_score:.2f}")

        return params

    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)"""
        query_lower = query.lower()

        # Base complexity from length
        word_count = len(query.split())
        length_score = min(word_count / 50.0, 0.5)  # Max 0.5 from length

        # Complexity from indicators
        indicator_score = 0.0
        for indicator, score in self.complexity_indicators.items():
            if indicator in query_lower:
                indicator_score = max(indicator_score, score)

        # Question depth (multiple questions, sub-parts)
        question_marks = query.count('?')
        multi_part = len(re.findall(r'\b(and|also|additionally|furthermore)\b', query_lower))
        depth_score = min((question_marks + multi_part) * 0.1, 0.3)

        # Combine scores
        complexity = min(length_score + indicator_score + depth_score, 1.0)

        # Use SpaCy for more sophisticated analysis if available
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(query)
                # Add complexity for entity density
                entities = len(doc.ents)
                entity_score = min(entities * 0.05, 0.2)

                # Add complexity for sentence structure
                sentences = len(list(doc.sents))
                sentence_score = min(sentences * 0.1, 0.3)

                complexity = min(complexity + entity_score + sentence_score, 1.0)
            except:
                pass

        return complexity

    def _identify_required_tools(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Identify tools required for the query"""
        required_tools = []
        query_lower = query.lower()

        # Check tool indicators
        for tool, indicators in self.tool_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    required_tools.append(tool)
                    break

        # Check context for files
        if context and context.get("files"):
            required_tools.append("file_read_tool")
            for file in context["files"]:
                if file.get("name", "").endswith(".pdf"):
                    required_tools.append("pdf_reader_tool")
                elif file.get("name", "").endswith((".png", ".jpg", ".jpeg")):
                    required_tools.append("image_analysis_tool")

        # Always include search for research/analytical queries
        if query_lower in ["research", "analyze", "investigate", "find out"]:
            if "web_search_tool" not in required_tools:
                required_tools.append("web_search_tool")

        return list(set(required_tools))  # Remove duplicates

    def _set_execution_parameters(self, params: OperationalParameters, query: str, context: Optional[Dict[str, Any]]) -> None:
        """Set execution parameters based on classification"""

        # Set quality level based on complexity
        if params.complexity_score < 0.3:
            params.quality_level = "BASIC"
            params.reasoning_type = "SIMPLE"
            params.max_iterations = 5
        elif params.complexity_score < 0.7:
            params.quality_level = "BALANCED"
            params.reasoning_type = "SIMPLE"
            params.max_iterations = 10
        else:
            params.quality_level = "THOROUGH"
            params.reasoning_type = "LAYERED"
            params.max_iterations = 20

        # Category-specific settings
        if params.category == QueryCategory.RESEARCH:
            params.use_multi_agent = True
            params.verification_level = "thorough"
            params.reasoning_type = "LAYERED"

        elif params.category == QueryCategory.ANALYTICAL:
            params.reasoning_type = "LAYERED"
            params.verification_level = "thorough"

        elif params.category == QueryCategory.TECHNICAL:
            params.required_tools.append("python_repl_tool")
            params.confidence_threshold = 0.8

        elif params.category == QueryCategory.CALCULATION:
            params.required_tools.append("python_repl_tool")
            params.verification_level = "thorough"

        elif params.category == QueryCategory.COMPLEX:
            params.use_multi_agent = True
            params.reasoning_type = "RECURSIVE"
            params.max_iterations = 30

        # Token estimation
        params.estimated_tokens = self._estimate_tokens(query, params)

        # Model selection based on complexity
        if params.complexity_score > 0.8 or params.use_multi_agent:
            params.recommended_model = "llama-3.3-70b-versatile"
        elif params.complexity_score < 0.3:
            params.recommended_model = "llama-3.1-8b-instant"
        else:
            params.recommended_model = "llama-3.2-11b-text-preview"

    def _estimate_tokens(self, query: str, params: OperationalParameters) -> int:
        """Estimate tokens needed for response"""
        base_tokens = len(query.split()) * 10  # Rough estimate

        # Add based on category
        category_multipliers = {
            QueryCategory.FACTUAL: 1.5,
            QueryCategory.ANALYTICAL: 3.0,
            QueryCategory.CREATIVE: 4.0,
            QueryCategory.TECHNICAL: 2.5,
            QueryCategory.RESEARCH: 5.0,
            QueryCategory.CALCULATION: 2.0,
            QueryCategory.FILE_ANALYSIS: 3.0,
            QueryCategory.COMPLEX: 5.0,
        }

        multiplier = category_multipliers.get(params.category, 2.0)

        # Adjust for multi-agent
        if params.use_multi_agent:
            multiplier *= 2

        return int(base_tokens * multiplier)

    def _is_gaia_query(self, query: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if this is a GAIA benchmark query"""
        # Check for GAIA indicators
        if "gaia" in query.lower():
            return True

        # Check context
        if context and context.get("source") == "gaia":
            return True

        # Check for GAIA answer format
        if "<<<" in query and ">>>" in query:
            return True

        return False

    def get_tool_recommendations(self, category: QueryCategory) -> List[str]:
        """Get recommended tools for a category"""
        recommendations = {
            QueryCategory.FACTUAL: ["web_search_tool", "wikipedia_tool"],
            QueryCategory.ANALYTICAL: ["web_search_tool", "python_repl_tool"],
            QueryCategory.CREATIVE: ["web_search_tool"],
            QueryCategory.TECHNICAL: ["python_repl_tool", "web_search_tool"],
            QueryCategory.RESEARCH: ["web_search_tool", "semantic_scholar_tool", "wikipedia_tool"],
            QueryCategory.CALCULATION: ["python_repl_tool"],
            QueryCategory.FILE_ANALYSIS: ["file_read_tool", "pdf_reader_tool"],
            QueryCategory.COMPLEX: ["web_search_tool", "python_repl_tool", "wikipedia_tool"],
        }

        return recommendations.get(category, ["web_search_tool"])
