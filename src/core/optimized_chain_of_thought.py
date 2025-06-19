"""
Optimized Chain of Thought (CoT) System
A comprehensive implementation with adaptive depth, multi-path exploration,
caching, and metacognitive capabilities
"""

import asyncio
import hashlib
import json
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from enum import Enum, auto
import pickle
import re
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# Core Data Structures
# =============================

class ReasoningType(Enum):
    """Types of reasoning approaches"""
    DEDUCTIVE = auto()      # General to specific
    INDUCTIVE = auto()      # Specific to general
    ABDUCTIVE = auto()      # Best explanation
    ANALOGICAL = auto()     # Pattern matching
    CAUSAL = auto()         # Cause and effect
    METACOGNITIVE = auto()  # Thinking about thinking

@dataclass
class ReasoningStep:
    """Represents a single step in chain of thought reasoning"""
    step_id: int
    depth: int
    reasoning_type: ReasoningType
    thought: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    sub_steps: List['ReasoningStep'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ReasoningPath:
    """Represents a complete reasoning path"""
    path_id: str
    query: str
    steps: List[ReasoningStep]
    final_answer: Optional[str] = None
    total_confidence: float = 0.0
    execution_time: float = 0.0
    template_used: Optional[str] = None
    complexity_score: float = 0.0

@dataclass
class CachedReasoning:
    """Cached reasoning result"""
    query_hash: str
    reasoning_path: ReasoningPath
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    performance_score: float = 0.0

# =============================
# Complexity Analysis
# =============================

class ComplexityAnalyzer:
    """Advanced complexity analysis for queries"""
    
    def __init__(self):
        self.feature_weights = {
            'length': 0.15,
            'vocabulary_complexity': 0.20,
            'structural_complexity': 0.20,
            'domain_specificity': 0.15,
            'ambiguity': 0.15,
            'multi_step_requirement': 0.15
        }
        
        # Domain-specific term dictionaries
        self.domain_terms = {
            'scientific': ['hypothesis', 'theory', 'experiment', 'analysis', 'correlation'],
            'mathematical': ['calculate', 'solve', 'prove', 'derive', 'equation'],
            'philosophical': ['ethics', 'morality', 'existence', 'consciousness', 'meaning'],
            'technical': ['algorithm', 'optimize', 'implement', 'architecture', 'system'],
            'analytical': ['compare', 'contrast', 'evaluate', 'assess', 'critique']
        }
        
    def analyze(self, query: str) -> Tuple[float, Dict[str, float]]:
        """
        Analyze query complexity and return score with breakdown
        Returns: (overall_score, feature_scores)
        """
        features = {}
        
        # Length complexity
        features['length'] = self._analyze_length(query)
        
        # Vocabulary complexity
        features['vocabulary_complexity'] = self._analyze_vocabulary(query)
        
        # Structural complexity
        features['structural_complexity'] = self._analyze_structure(query)
        
        # Domain specificity
        features['domain_specificity'] = self._analyze_domain_specificity(query)
        
        # Ambiguity level
        features['ambiguity'] = self._analyze_ambiguity(query)
        
        # Multi-step requirement
        features['multi_step_requirement'] = self._analyze_multi_step(query)
        
        # Calculate weighted score
        overall_score = sum(
            features[feature] * self.feature_weights[feature]
            for feature in features
        )
        
        return min(overall_score, 1.0), features
    
    def _analyze_length(self, query: str) -> float:
        """Analyze query length complexity"""
        word_count = len(query.split())
        
        if word_count < 10:
            return 0.2
        elif word_count < 25:
            return 0.4
        elif word_count < 50:
            return 0.6
        elif word_count < 100:
            return 0.8
        else:
            return 1.0
    
    def _analyze_vocabulary(self, query: str) -> float:
        """Analyze vocabulary complexity"""
        words = query.lower().split()
        
        # Check for complex words (length > 7)
        complex_words = [w for w in words if len(w) > 7]
        complexity = len(complex_words) / max(len(words), 1)
        
        # Check for technical terms
        technical_count = sum(
            1 for word in words
            for domain_list in self.domain_terms.values()
            if word in domain_list
        )
        
        technical_ratio = technical_count / max(len(words), 1)
        
        return min((complexity + technical_ratio) / 2, 1.0)
    
    def _analyze_structure(self, query: str) -> float:
        """Analyze structural complexity"""
        # Count clauses (approximated by commas and conjunctions)
        clause_indicators = query.count(',') + sum(
            query.lower().count(conj) for conj in [' and ', ' or ', ' but ', ' if ', ' when ']
        )
        
        # Count questions
        questions = query.count('?')
        
        # Nested parentheses
        nested_level = self._count_nesting(query)
        
        complexity = (clause_indicators * 0.1 + questions * 0.2 + nested_level * 0.3)
        return min(complexity, 1.0)
    
    def _analyze_domain_specificity(self, query: str) -> float:
        """Analyze domain specificity"""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, terms in self.domain_terms.items():
            score = sum(1 for term in terms if term in query_lower)
            domain_scores[domain] = score
            
        max_score = max(domain_scores.values()) if domain_scores else 0
        return min(max_score * 0.2, 1.0)
    
    def _analyze_ambiguity(self, query: str) -> float:
        """Analyze query ambiguity"""
        # Check for vague terms
        vague_terms = ['thing', 'stuff', 'it', 'they', 'some', 'any', 'whatever']
        vague_count = sum(query.lower().count(term) for term in vague_terms)
        
        # Check for unclear references
        pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those']
        pronoun_count = sum(query.lower().count(p) for p in pronouns)
        
        ambiguity = (vague_count * 0.1 + pronoun_count * 0.05)
        return min(ambiguity, 1.0)
    
    def _analyze_multi_step(self, query: str) -> float:
        """Analyze if query requires multiple steps"""
        # Indicators of multi-step reasoning
        multi_step_indicators = [
            'first', 'then', 'after', 'before', 'finally',
            'step', 'process', 'procedure', 'stages',
            'compare', 'analyze', 'evaluate'
        ]
        
        indicator_count = sum(
            1 for indicator in multi_step_indicators
            if indicator in query.lower()
        )
        
        return min(indicator_count * 0.2, 1.0)
    
    def _count_nesting(self, text: str) -> int:
        """Count maximum nesting level of parentheses"""
        max_level = 0
        current_level = 0
        
        for char in text:
            if char == '(':
                current_level += 1
                max_level = max(max_level, current_level)
            elif char == ')':
                current_level = max(0, current_level - 1)
                
        return max_level

# =============================
# Template Library
# =============================

class ReasoningTemplate(ABC):
    """Base class for reasoning templates"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.success_rate = 0.0
        
    @abstractmethod
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate reasoning steps based on template"""
        pass
    
    @abstractmethod
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        """Return applicability score (0-1) for this template"""
        pass

class MathematicalReasoningTemplate(ReasoningTemplate):
    """Template for mathematical reasoning"""
    
    def __init__(self):
        super().__init__(
            "mathematical",
            "Step-by-step mathematical problem solving"
        )
        
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate mathematical reasoning steps"""
        steps = [
            "First, let me identify what we're asked to find",
            "Next, I'll identify the given information",
            "Now, I'll determine which mathematical concepts or formulas apply",
            "Let me set up the problem mathematically",
            "I'll solve step by step",
            "Let me verify the solution",
            "Therefore, the answer is"
        ]
        return steps
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        """Check if mathematical template is applicable"""
        math_keywords = ['calculate', 'solve', 'compute', 'find', 'equation', 
                        'formula', 'prove', 'derive', 'integral', 'derivative']
        
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in math_keywords if keyword in query_lower)
        
        base_score = keyword_matches * 0.2
        domain_boost = features.get('domain_specificity', 0) * 0.3
        
        return min(base_score + domain_boost, 1.0)

class AnalyticalReasoningTemplate(ReasoningTemplate):
    """Template for analytical reasoning"""
    
    def __init__(self):
        super().__init__(
            "analytical",
            "Breaking down complex problems into components"
        )
        
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate analytical reasoning steps"""
        steps = [
            "Let me break this down into key components",
            "First component analysis",
            "Second component analysis",
            "Now I'll examine the relationships between components",
            "Let me consider potential implications",
            "Synthesizing the analysis",
            "Based on this analysis, I conclude"
        ]
        return steps
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        """Check if analytical template is applicable"""
        analytical_keywords = ['analyze', 'examine', 'investigate', 'explore',
                             'evaluate', 'assess', 'consider', 'review']
        
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in analytical_keywords if keyword in query_lower)
        
        complexity_factor = features.get('structural_complexity', 0) * 0.3
        
        return min(keyword_matches * 0.25 + complexity_factor, 1.0)

class ComparativeReasoningTemplate(ReasoningTemplate):
    """Template for comparative reasoning"""
    
    def __init__(self):
        super().__init__(
            "comparative",
            "Comparing and contrasting different elements"
        )
        
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate comparative reasoning steps"""
        steps = [
            "I'll identify the items to be compared",
            "Let me establish the criteria for comparison",
            "Analyzing the first item",
            "Analyzing the second item",
            "Identifying similarities",
            "Identifying differences",
            "Drawing conclusions from the comparison"
        ]
        return steps
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        """Check if comparative template is applicable"""
        comparative_keywords = ['compare', 'contrast', 'difference', 'similar',
                              'versus', 'vs', 'better', 'worse', 'prefer']
        
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in comparative_keywords if keyword in query_lower)
        
        return min(keyword_matches * 0.3, 1.0)

class CausalReasoningTemplate(ReasoningTemplate):
    """Template for causal reasoning"""
    
    def __init__(self):
        super().__init__(
            "causal",
            "Understanding cause and effect relationships"
        )
        
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate causal reasoning steps"""
        steps = [
            "Let me identify the phenomenon or effect in question",
            "I'll examine potential causes",
            "Analyzing the causal chain",
            "Considering alternative explanations",
            "Evaluating the strength of causal relationships",
            "Accounting for confounding factors",
            "Based on the causal analysis"
        ]
        return steps
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        """Check if causal template is applicable"""
        causal_keywords = ['why', 'because', 'cause', 'effect', 'result',
                          'lead to', 'consequence', 'impact', 'influence']
        
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in causal_keywords if keyword in query_lower)
        
        return min(keyword_matches * 0.25, 1.0)

class TemplateLibrary:
    """Library managing reasoning templates"""
    
    def __init__(self):
        self.templates: Dict[str, ReasoningTemplate] = {}
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Initialize default templates"""
        default_templates = [
            MathematicalReasoningTemplate(),
            AnalyticalReasoningTemplate(),
            ComparativeReasoningTemplate(),
            CausalReasoningTemplate()
        ]
        
        for template in default_templates:
            self.add_template(template)
    
    def add_template(self, template: ReasoningTemplate):
        """Add a template to the library"""
        self.templates[template.name] = template
        
    def select_template(self, query: str, features: Dict[str, float]) -> ReasoningTemplate:
        """Select the best template for a query"""
        template_scores = {}
        
        for name, template in self.templates.items():
            score = template.is_applicable(query, features)
            template_scores[name] = score
            
        # Select template with highest score
        if template_scores:
            best_template_name = max(template_scores.items(), key=lambda x: x[1])[0]
            return self.templates[best_template_name]
        
        # Return a default template if none match well
        return self._get_default_template()
    
    def _get_default_template(self) -> ReasoningTemplate:
        """Get default reasoning template"""
        class DefaultTemplate(ReasoningTemplate):
            def __init__(self):
                super().__init__("default", "General purpose reasoning")
                
            def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
                return [
                    "Let me understand the question",
                    "I'll consider the key aspects",
                    "Analyzing the information",
                    "Drawing logical conclusions",
                    "Therefore"
                ]
                
            def is_applicable(self, query: str, features: Dict[str, float]) -> float:
                return 0.5
        
        return DefaultTemplate()

# =============================
# Reasoning Cache
# =============================

class ReasoningCache:
    """Advanced caching system for reasoning patterns"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, CachedReasoning] = {}
        self.performance_threshold = 0.7
        self.similarity_threshold = 0.85
        
    def get(self, query: str) -> Optional[ReasoningPath]:
        """Get cached reasoning for a query"""
        query_hash = self._hash_query(query)
        
        # Direct hit
        if query_hash in self.cache:
            cached = self.cache[query_hash]
            if self._is_valid(cached):
                cached.hit_count += 1
                cached.last_accessed = time.time()
                return cached.reasoning_path
                
        # Check for similar queries
        similar_result = self._find_similar(query)
        if similar_result:
            return similar_result
            
        return None
    
    def store(self, query: str, reasoning_path: ReasoningPath):
        """Store reasoning path in cache"""
        # Only cache high-quality reasoning
        if reasoning_path.total_confidence < self.performance_threshold:
            return
            
        query_hash = self._hash_query(query)
        
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
            
        cached = CachedReasoning(
            query_hash=query_hash,
            reasoning_path=reasoning_path,
            performance_score=reasoning_path.total_confidence
        )
        
        self.cache[query_hash] = cached
        
    def _hash_query(self, query: str) -> str:
        """Generate hash for a query"""
        # Normalize query
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _is_valid(self, cached: CachedReasoning) -> bool:
        """Check if cached entry is still valid"""
        age = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(cached.last_accessed)
        return age < self.ttl
    
    def _find_similar(self, query: str) -> Optional[ReasoningPath]:
        """Find similar cached queries using simple similarity"""
        query_words = set(query.lower().split())
        
        best_match = None
        best_similarity = 0
        
        for cached in self.cache.values():
            if not self._is_valid(cached):
                continue
                
            cached_words = set(cached.reasoning_path.query.lower().split())
            
            # Jaccard similarity
            intersection = len(query_words & cached_words)
            union = len(query_words | cached_words)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached
                    
        if best_match:
            best_match.hit_count += 1
            best_match.last_accessed = time.time()
            return best_match.reasoning_path
            
        return None
    
    def _evict_oldest(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
            
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(c.hit_count for c in self.cache.values())
        avg_performance = np.mean([c.performance_score for c in self.cache.values()]) if self.cache else 0
        
        return {
            "size": len(self.cache),
            "total_hits": total_hits,
            "average_performance": avg_performance,
            "hit_rate": total_hits / max(len(self.cache), 1)
        }

# =============================
# Multi-Path Reasoning Engine
# =============================

class MultiPathReasoning:
    """Engine for exploring multiple reasoning paths in parallel"""
    
    def __init__(self, max_paths: int = 5, max_workers: int = 4):
        self.max_paths = max_paths
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.path_generator = PathGenerator()
        
    async def explore_paths(self, query: str, complexity: float,
                          template_library: TemplateLibrary) -> List[ReasoningPath]:
        """Explore multiple reasoning paths in parallel"""
        # Generate different path configurations
        path_configs = self.path_generator.generate_configurations(
            query, complexity, self.max_paths
        )
        
        # Execute paths in parallel
        tasks = []
        for config in path_configs:
            task = asyncio.create_task(
                self._execute_path(query, config, template_library)
            )
            tasks.append(task)
            
        paths = await asyncio.gather(*tasks)
        
        # Filter out failed paths
        valid_paths = [p for p in paths if p is not None]
        
        # Rank paths by confidence
        valid_paths.sort(key=lambda p: p.total_confidence, reverse=True)
        
        return valid_paths
    
    async def _execute_path(self, query: str, config: Dict[str, Any],
                          template_library: TemplateLibrary) -> Optional[ReasoningPath]:
        """Execute a single reasoning path"""
        try:
            path_id = f"path_{config['id']}_{int(time.time())}"
            start_time = time.time()
            
            # Select template based on config
            template = template_library.templates.get(
                config['template'],
                template_library._get_default_template()
            )
            
            # Generate reasoning steps
            steps = []
            depth = config['max_depth']
            
            for i in range(depth):
                step = await self._generate_step(
                    i, depth, query, template, config
                )
                steps.append(step)
                
                # Early termination if confidence is too low
                if step.confidence < 0.3:
                    break
            
            # Calculate total confidence
            total_confidence = self._calculate_path_confidence(steps)
            
            # Generate final answer
            final_answer = self._synthesize_answer(steps)
            
            path = ReasoningPath(
                path_id=path_id,
                query=query,
                steps=steps,
                final_answer=final_answer,
                total_confidence=total_confidence,
                execution_time=time.time() - start_time,
                template_used=template.name,
                complexity_score=config.get('complexity', 0)
            )
            
            return path
            
        except Exception as e:
            logger.error(f"Path execution failed: {str(e)}")
            return None
    
    async def _generate_step(self, step_num: int, max_depth: int,
                           query: str, template: ReasoningTemplate,
                           config: Dict[str, Any]) -> ReasoningStep:
        """Generate a single reasoning step"""
        # Get template steps
        template_steps = template.generate_steps(query, config)
        
        # Select appropriate step text
        if step_num < len(template_steps):
            thought = template_steps[step_num]
        else:
            thought = "Continuing the analysis..."
        
        # Determine reasoning type based on config
        reasoning_type = config.get('reasoning_types', [ReasoningType.DEDUCTIVE])[
            step_num % len(config.get('reasoning_types', [ReasoningType.DEDUCTIVE]))
        ]
        
        # Calculate step confidence
        # Confidence decreases with depth but varies by reasoning type
        base_confidence = 0.9 - (step_num * 0.1 / max_depth)
        type_modifier = {
            ReasoningType.DEDUCTIVE: 1.0,
            ReasoningType.INDUCTIVE: 0.9,
            ReasoningType.ABDUCTIVE: 0.85,
            ReasoningType.ANALOGICAL: 0.8,
            ReasoningType.CAUSAL: 0.85,
            ReasoningType.METACOGNITIVE: 0.95
        }
        
        confidence = base_confidence * type_modifier.get(reasoning_type, 1.0)
        
        # Add some randomness for diversity
        confidence *= (0.9 + np.random.random() * 0.2)
        
        step = ReasoningStep(
            step_id=step_num,
            depth=step_num,
            reasoning_type=reasoning_type,
            thought=thought,
            confidence=min(confidence, 1.0),
            supporting_evidence=[f"Evidence_{step_num}"],
            metadata={
                "template": template.name,
                "path_config": config['id']
            }
        )
        
        return step
    
    def _calculate_path_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence for a reasoning path"""
        if not steps:
            return 0.0
            
        # Weighted average based on step depth
        weights = [1.0 / (i + 1) for i in range(len(steps))]
        weight_sum = sum(weights)
        
        weighted_confidence = sum(
            step.confidence * weight 
            for step, weight in zip(steps, weights)
        ) / weight_sum
        
        # Penalty for incomplete paths
        completion_ratio = len(steps) / max(len(steps), 5)  # Assume 5 steps is complete
        
        return weighted_confidence * completion_ratio
    
    def _synthesize_answer(self, steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from reasoning steps"""
        # In practice, this would use an LLM to synthesize
        # For now, we'll create a simple summary
        key_thoughts = [step.thought for step in steps[-3:]]  # Last 3 steps
        return f"Based on the reasoning: {' -> '.join(key_thoughts)}"

class PathGenerator:
    """Generates different path configurations for exploration"""
    
    def generate_configurations(self, query: str, complexity: float,
                              num_paths: int) -> List[Dict[str, Any]]:
        """Generate diverse path configurations"""
        configs = []
        
        # Base depth on complexity
        base_depth = int(5 + complexity * 10)
        
        # Generate diverse configurations
        for i in range(num_paths):
            config = {
                'id': i,
                'max_depth': base_depth + np.random.randint(-2, 3),
                'reasoning_types': self._select_reasoning_types(i, complexity),
                'template': self._select_template_preference(i),
                'exploration_strategy': self._select_strategy(i),
                'complexity': complexity
            }
            configs.append(config)
            
        return configs
    
    def _select_reasoning_types(self, path_id: int, complexity: float) -> List[ReasoningType]:
        """Select reasoning types for a path"""
        all_types = list(ReasoningType)
        
        if path_id == 0:
            # First path uses standard deductive reasoning
            return [ReasoningType.DEDUCTIVE]
        elif path_id == 1:
            # Second path mixes deductive and inductive
            return [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]
        elif complexity > 0.7:
            # Complex queries benefit from metacognitive reasoning
            return [ReasoningType.DEDUCTIVE, ReasoningType.METACOGNITIVE, ReasoningType.CAUSAL]
        else:
            # Random selection for diversity
            num_types = np.random.randint(1, 4)
            return list(np.random.choice(all_types, num_types, replace=False))
    
    def _select_template_preference(self, path_id: int) -> str:
        """Select template preference for a path"""
        templates = ['mathematical', 'analytical', 'comparative', 'causal', 'default']
        
        if path_id < len(templates):
            return templates[path_id]
        return np.random.choice(templates)
    
    def _select_strategy(self, path_id: int) -> str:
        """Select exploration strategy"""
        strategies = ['depth_first', 'breadth_first', 'best_first', 'monte_carlo']
        return strategies[path_id % len(strategies)]

# =============================
# Metacognitive Layer
# =============================

class MetacognitiveLayer:
    """Metacognitive reasoning capabilities"""
    
    def __init__(self):
        self.reflection_depth = 2
        self.confidence_threshold = 0.7
        self.uncertainty_markers = [
            'maybe', 'possibly', 'might', 'could be', 'uncertain',
            'not sure', 'approximately', 'roughly'
        ]
        
    async def reflect_on_reasoning(self, reasoning_path: ReasoningPath) -> ReasoningPath:
        """Reflect on and potentially improve reasoning"""
        # Analyze the reasoning quality
        quality_assessment = self._assess_quality(reasoning_path)
        
        if quality_assessment['overall_quality'] < self.confidence_threshold:
            # Attempt to improve reasoning
            improved_path = await self._improve_reasoning(reasoning_path, quality_assessment)
            return improved_path
            
        return reasoning_path
    
    def _assess_quality(self, path: ReasoningPath) -> Dict[str, Any]:
        """Assess the quality of reasoning"""
        assessment = {
            'overall_quality': path.total_confidence,
            'coherence': self._assess_coherence(path),
            'completeness': self._assess_completeness(path),
            'uncertainty_level': self._assess_uncertainty(path),
            'logical_consistency': self._assess_consistency(path)
        }
        
        # Identify weak points
        weak_steps = [
            step for step in path.steps 
            if step.confidence < self.confidence_threshold
        ]
        assessment['weak_steps'] = weak_steps
        
        return assessment
    
    def _assess_coherence(self, path: ReasoningPath) -> float:
        """Assess coherence of reasoning steps"""
        if len(path.steps) < 2:
            return 1.0
            
        # Check if steps follow logically
        # In practice, this would use embeddings or LLM
        # For now, we'll use a simple heuristic
        coherence_scores = []
        
        for i in range(1, len(path.steps)):
            prev_step = path.steps[i-1]
            curr_step = path.steps[i]
            
            # Check if reasoning types are compatible
            if prev_step.reasoning_type == curr_step.reasoning_type:
                coherence_scores.append(0.9)
            else:
                coherence_scores.append(0.7)
                
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _assess_completeness(self, path: ReasoningPath) -> float:
        """Assess if reasoning is complete"""
        # Check if we have a final answer
        if not path.final_answer:
            return 0.5
            
        # Check if all major reasoning types were used for complex queries
        if path.complexity_score > 0.7:
            types_used = set(step.reasoning_type for step in path.steps)
            expected_types = {ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE}
            
            coverage = len(types_used & expected_types) / len(expected_types)
            return coverage
            
        return 1.0
    
    def _assess_uncertainty(self, path: ReasoningPath) -> float:
        """Assess level of uncertainty in reasoning"""
        uncertainty_count = 0
        total_words = 0
        
        for step in path.steps:
            words = step.thought.lower().split()
            total_words += len(words)
            
            for marker in self.uncertainty_markers:
                uncertainty_count += step.thought.lower().count(marker)
                
        return uncertainty_count / max(total_words, 1)
    
    def _assess_consistency(self, path: ReasoningPath) -> float:
        """Assess logical consistency"""
        # Check for contradictions or inconsistencies
        # In practice, this would be more sophisticated
        
        confidence_variance = np.var([step.confidence for step in path.steps])
        
        # Low variance indicates consistent confidence
        consistency = 1.0 - min(confidence_variance, 1.0)
        
        return consistency
    
    async def _improve_reasoning(self, path: ReasoningPath,
                               assessment: Dict[str, Any]) -> ReasoningPath:
        """Attempt to improve reasoning based on assessment"""
        improved_steps = list(path.steps)
        
        # Add metacognitive reflection steps
        for weak_step in assessment['weak_steps']:
            reflection_step = ReasoningStep(
                step_id=weak_step.step_id + 0.5,
                depth=weak_step.depth,
                reasoning_type=ReasoningType.METACOGNITIVE,
                thought=f"Let me reconsider this step more carefully: {weak_step.thought}",
                confidence=weak_step.confidence * 1.2,  # Boost after reflection
                supporting_evidence=weak_step.supporting_evidence + ["Additional reflection"],
                metadata={"reflection": True}
            )
            
            # Insert reflection after weak step
            idx = improved_steps.index(weak_step)
            improved_steps.insert(idx + 1, reflection_step)
        
        # Recalculate confidence
        total_confidence = np.mean([step.confidence for step in improved_steps])
        
        # Create improved path
        improved_path = ReasoningPath(
            path_id=f"{path.path_id}_improved",
            query=path.query,
            steps=improved_steps,
            final_answer=path.final_answer,
            total_confidence=total_confidence,
            execution_time=path.execution_time,
            template_used=path.template_used,
            complexity_score=path.complexity_score
        )
        
        return improved_path

# =============================
# Main Chain of Thought System
# =============================

class OptimizedChainOfThought:
    """Main Chain of Thought system with all optimizations"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
        # Initialize components
        self.complexity_analyzer = ComplexityAnalyzer()
        self.template_library = TemplateLibrary()
        self.reasoning_cache = ReasoningCache(
            max_size=self.config.get('cache_size', 1000),
            ttl_hours=self.config.get('cache_ttl', 24)
        )
        self.multi_path_engine = MultiPathReasoning(
            max_paths=self.config.get('max_paths', 5)
        )
        self.metacognitive_layer = MetacognitiveLayer()
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_confidence': 0,
            'average_execution_time': 0
        }
        
    async def reason(self, query: str, context: Optional[Dict[str, Any]] = None) -> ReasoningPath:
        """Main reasoning method with all optimizations"""
        start_time = time.time()
        self.performance_metrics['total_queries'] += 1
        
        # Check cache first
        cached_result = self.reasoning_cache.get(query)
        if cached_result:
            self.performance_metrics['cache_hits'] += 1
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_result
        
        # Analyze complexity
        complexity_score, features = self.complexity_analyzer.analyze(query)
        logger.info(f"Query complexity: {complexity_score:.2f}")
        
        # Select template
        template = self.template_library.select_template(query, features)
        logger.info(f"Selected template: {template.name}")
        
        # Explore multiple reasoning paths
        paths = await self.multi_path_engine.explore_paths(
            query, complexity_score, self.template_library
        )
        
        if not paths:
            # Fallback to simple reasoning
            path = await self._simple_reasoning(query, template, complexity_score)
        else:
            # Select best path
            path = paths[0]
            
            # Apply metacognitive reflection
            path = await self.metacognitive_layer.reflect_on_reasoning(path)
        
        # Cache successful reasoning
        if path.total_confidence > 0.7:
            self.reasoning_cache.store(query, path)
        
        # Update metrics
        execution_time = time.time() - start_time
        self._update_metrics(path.total_confidence, execution_time)
        
        return path
    
    async def _simple_reasoning(self, query: str, template: ReasoningTemplate,
                              complexity: float) -> ReasoningPath:
        """Fallback simple reasoning"""
        steps = []
        template_steps = template.generate_steps(query, {})
        
        for i, thought in enumerate(template_steps[:5]):  # Limit to 5 steps
            step = ReasoningStep(
                step_id=i,
                depth=i,
                reasoning_type=ReasoningType.DEDUCTIVE,
                thought=thought,
                confidence=0.7 - (i * 0.05)
            )
            steps.append(step)
        
        path = ReasoningPath(
            path_id=f"simple_{int(time.time())}",
            query=query,
            steps=steps,
            final_answer="Based on the analysis above",
            total_confidence=0.6,
            execution_time=0.1,
            template_used=template.name,
            complexity_score=complexity
        )
        
        return path
    
    def _update_metrics(self, confidence: float, execution_time: float):
        """Update performance metrics"""
        n = self.performance_metrics['total_queries']
        
        # Update running averages
        self.performance_metrics['average_confidence'] = (
            (self.performance_metrics['average_confidence'] * (n - 1) + confidence) / n
        )
        
        self.performance_metrics['average_execution_time'] = (
            (self.performance_metrics['average_execution_time'] * (n - 1) + execution_time) / n
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        cache_stats = self.reasoning_cache.get_stats()
        
        return {
            'total_queries': self.performance_metrics['total_queries'],
            'cache_hit_rate': self.performance_metrics['cache_hits'] / 
                            max(self.performance_metrics['total_queries'], 1),
            'average_confidence': self.performance_metrics['average_confidence'],
            'average_execution_time': self.performance_metrics['average_execution_time'],
            'cache_stats': cache_stats,
            'templates_usage': {
                name: template.usage_count 
                for name, template in self.template_library.templates.items()
            }
        }

# =============================
# Example Usage and Testing
# =============================

async def example_usage():
    """Example usage of the Optimized Chain of Thought system"""
    
    # Create CoT system
    cot = OptimizedChainOfThought(
        "advanced_cot",
        config={
            'max_paths': 3,
            'cache_size': 500,
            'cache_ttl': 12
        }
    )
    
    # Example queries of varying complexity
    queries = [
        "What is 2 + 2?",
        "Explain why the sky is blue.",
        "Compare and contrast the economic systems of capitalism and socialism.",
        "Analyze the potential long-term impacts of artificial intelligence on employment.",
        "Solve the equation: 3x^2 + 5x - 2 = 0"
    ]
    
    print("=== Optimized Chain of Thought Examples ===\n")
    
    for query in queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        # Execute reasoning
        result = await cot.reason(query)
        
        print(f"Template used: {result.template_used}")
        print(f"Complexity: {result.complexity_score:.2f}")
        print(f"Confidence: {result.total_confidence:.2f}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"Number of steps: {len(result.steps)}")
        
        print("\nReasoning steps:")
        for step in result.steps:
            print(f"  Step {step.step_id} ({step.reasoning_type.name}): {step.thought}")
            print(f"    Confidence: {step.confidence:.2f}")
        
        print(f"\nFinal answer: {result.final_answer}")
        print("\n" + "="*70 + "\n")
    
    # Test caching
    print("=== Testing Cache ===")
    cached_query = queries[1]  # "Explain why the sky is blue."
    
    print(f"Re-running query: {cached_query}")
    start = time.time()
    result = await cot.reason(cached_query)
    print(f"Execution time (should be faster due to cache): {time.time() - start:.4f}s")
    
    # Show performance report
    print("\n=== Performance Report ===")
    report = cot.get_performance_report()
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(example_usage()) 