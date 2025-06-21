from agent import query
from benchmarks.cot_performance import avg_confidence
from benchmarks.cot_performance import recommendations
from benchmarks.cot_performance import template
from tests.load_test import data

from src.core.optimized_chain_of_thought import domain_boost
from src.core.optimized_chain_of_thought import indicator_count
from src.core.optimized_chain_of_thought import keyword_matches
from src.core.optimized_chain_of_thought import step
from src.core.optimized_chain_of_thought import steps
from src.database.models import reasoning_path
from src.gaia_components.adaptive_tool_system import total_usage
from src.infrastructure.agents.agent_factory import domain
from src.meta_cognition import base_score
from src.meta_cognition import complexity
from src.meta_cognition import query_lower
from src.templates.template_factory import avg_feedback
from src.templates.template_factory import avg_time
from src.templates.template_factory import base_step
from src.templates.template_factory import config_path
from src.templates.template_factory import creative_keywords
from src.templates.template_factory import debug_keywords
from src.templates.template_factory import final_score
from src.templates.template_factory import legal_indicators
from src.templates.template_factory import phase_templates
from src.templates.template_factory import success_adjustment
from src.templates.template_factory import total_templates
from src.tools_introspection import name

"""

from typing import Any
from typing import List
from typing import Optional
Template Factory for Domain-Specific Templates
Creates specialized reasoning templates for different domains
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Import base template class
from src.core.optimized_chain_of_thought import ReasoningTemplate
import logging
# TODO: Fix undefined variables: Any, Dict, List, Optional, avg_confidence, avg_feedback, avg_time, base_score, base_step, complexity, config_path, context, creative_keywords, data, debug_keywords, defaultdict, domain, domain_boost, e, error_occurred, f, final_score, ind, indicator_count, json, keyword, keyword_matches, kw, legal_indicators, logging, name, os, performance_data, phase, phase_templates, query, query_lower, reasoning_path, recommendations, step, steps, success_adjustment, template, template_name, total_templates, total_usage, user_feedback, x
import pattern


logger = logging.getLogger(__name__)



class TemplateFactory:
    """Factory for creating domain-specific templates"""
    
    def __init__(self):
        self.template_registry = {}
        self.domain_patterns = self._load_domain_patterns()
        self.performance_tracker = TemplatePerformanceTracker()
    
    def _load_domain_patterns(self):
        """Load domain-specific patterns from configuration"""
        config_path = 'config/domain_patterns.json'
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.info("Warning: Could not load domain patterns from {}: {}", extra={"config_path": config_path, "e": e})
        
        # Default patterns if config file doesn't exist
        return {
            'scientific': {
                'keywords': ['hypothesis', 'experiment', 'data', 'analysis', 'theory', 'research', 'study', 'evidence'],
                'structure': ['question', 'hypothesis', 'methodology', 'results', 'conclusion'],
                'reasoning_types': ['deductive', 'inductive', 'analytical']
            },
            'business': {
                'keywords': ['revenue', 'profit', 'market', 'strategy', 'business', 'company', 'investment', 'growth'],
                'structure': ['situation', 'analysis', 'recommendations', 'implementation'],
                'reasoning_types': ['analytical', 'strategic', 'practical']
            },
            'technical': {
                'keywords': ['algorithm', 'system', 'architecture', 'performance', 'code', 'implementation', 'design'],
                'structure': ['problem', 'design', 'implementation', 'testing', 'optimization'],
                'reasoning_types': ['logical', 'systematic', 'analytical']
            },
            'legal': {
                'keywords': ['legal', 'law', 'statute', 'regulation', 'court', 'contract', 'liability', 'rights'],
                'structure': ['issue_identification', 'rule_statement', 'application', 'conclusion'],
                'reasoning_types': ['deductive', 'analytical', 'interpretive']
            },
            'creative': {
                'keywords': ['creative', 'artistic', 'design', 'imagine', 'innovative', 'aesthetic', 'expression'],
                'structure': ['inspiration', 'exploration', 'development', 'refinement', 'finalization'],
                'reasoning_types': ['analogical', 'creative', 'intuitive']
            },
            'debug': {
                'keywords': ['error', 'bug', 'issue', 'problem', 'fix', 'debug', 'troubleshoot', 'failed'],
                'structure': ['symptoms', 'diagnosis', 'hypothesis', 'testing', 'solution'],
                'reasoning_types': ['analytical', 'systematic', 'diagnostic']
            }
        }
    
    def create_template(self, domain: str, name: str) -> ReasoningTemplate:
        """Create a domain-specific template"""
        if domain not in self.domain_patterns:
            raise ValueError(f"Unknown domain: {domain}")
        
        pattern = self.domain_patterns[domain]
        
        class DomainTemplate(ReasoningTemplate):
            def __init__(self):
                super().__init__(
                    name=f"{domain}_{name}",
                    description=f"Domain-specific template for {domain}"
                )
                self.domain = domain
                self.pattern = pattern
                self.success_rate = 0.7  # Default success rate
            
            def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
                steps = []
                for phase in self.pattern['structure']:
                    step = self._generate_step_for_phase(phase, query, context)
                    steps.append(step)
                return steps
            
            def _generate_step_for_phase(self, phase: str, query: str, context: Dict[str, Any]) -> str:
                complexity = context.get('complexity', 0.5)
                
                phase_templates = {
                    'question': f"First, let me understand the {self.domain} question: {query[:50]}...",
                    'hypothesis': f"Based on {self.domain} principles, my hypothesis is...",
                    'methodology': f"I'll approach this using {self.domain} methodology...",
                    'results': f"Analyzing the {self.domain} data/information...",
                    'conclusion': f"Based on {self.domain} analysis, I conclude...",
                    'situation': f"Let me assess the current {self.domain} situation...",
                    'analysis': f"Performing detailed {self.domain} analysis...",
                    'recommendations': f"Based on {self.domain} best practices, I recommend...",
                    'implementation': f"For implementation in {self.domain} context...",
                    'problem': f"Defining the {self.domain} problem clearly...",
                    'design': f"Designing a {self.domain} solution...",
                    'testing': f"Testing the {self.domain} approach...",
                    'optimization': f"Optimizing for {self.domain} requirements...",
                    'issue_identification': f"Identifying the key {self.domain} issues...",
                    'rule_statement': f"Examining relevant {self.domain} rules and principles...",
                    'application': f"Applying {self.domain} principles to the facts...",
                    'inspiration': f"Exploring {self.domain} inspiration and ideas...",
                    'exploration': f"Investigating different {self.domain} approaches...",
                    'development': f"Developing the {self.domain} concept...",
                    'refinement': f"Refining the {self.domain} solution...",
                    'finalization': f"Finalizing the {self.domain} outcome...",
                    'symptoms': f"Identifying the symptoms of the {self.domain} problem...",
                    'diagnosis': f"Diagnosing the root cause of the {self.domain} issue...",
                    'solution': f"Implementing the {self.domain} solution..."
                }
                
                base_step = phase_templates.get(phase, f"Continuing {self.domain} analysis...")
                
                # Add complexity-based modifications
                if complexity > 0.7:
                    base_step += f" Given the complexity, I'll need to consider multiple {self.domain} factors."
                elif complexity < 0.3:
                    base_step += f" This appears to be a straightforward {self.domain} question."
                
                return base_step
            
            def is_applicable(self, query: str, features: Dict[str, float]) -> float:
                query_lower = query.lower()
                keyword_matches = sum(
                    1 for keyword in self.pattern['keywords'] 
                    if keyword in query_lower
                )
                
                base_score = keyword_matches * 0.25
                domain_boost = features.get('domain_specificity', 0) * 0.3
                
                # Adjust based on success rate
                success_adjustment = (self.success_rate - 0.5) * 0.2
                
                final_score = min(base_score + domain_boost + success_adjustment, 1.0)
                return max(final_score, 0.0)
        
        template = DomainTemplate()
        self.template_registry[f"{domain}_{name}"] = template
        return template
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return list(self.domain_patterns.keys())
    
    def get_templates_for_domain(self, domain: str) -> List[str]:
        """Get all templates for a specific domain"""
        return [name for name in self.template_registry.keys() if name.startswith(f"{domain}_")]
    
    def optimize_template(self, template_name: str, performance_data: Dict[str, Any]):
        """Optimize template based on performance data"""
        if template_name not in self.template_registry:
            return
        
        template = self.template_registry[template_name]
        
        # Update success rate
        if 'success_rate' in performance_data:
            template.success_rate = performance_data['success_rate']
        
        # Update pattern if provided
        if 'pattern_updates' in performance_data:
            template.pattern.update(performance_data['pattern_updates'])
        
        logger.info("Optimized template '{}' - Success rate: {}", extra={"template_name": template_name, "template_success_rate": template.success_rate})


class TemplatePerformanceTracker:
    """Track and analyze template performance"""
    
    def __init__(self):
        self.performance_data = defaultdict(lambda: {
            'usage_count': 0,
            'total_confidence': 0,
            'total_time': 0,
            'success_rate': 0,
            'feedback_scores': [],
            'error_count': 0
        })
    
    def track_template_performance(self, template_name: str, 
                                 reasoning_path: Any,
                                 user_feedback: Optional[float] = None,
                                 error_occurred: bool = False):
        """Track performance metrics for templates"""
        data = self.performance_data[template_name]
        
        data['usage_count'] += 1
        data['total_confidence'] += reasoning_path.total_confidence
        data['total_time'] += reasoning_path.execution_time
        
        if error_occurred:
            data['error_count'] += 1
        
        if user_feedback is not None:
            data['feedback_scores'].append(user_feedback)
        
        # Calculate success rate (confidence > 0.7)
        if reasoning_path.total_confidence > 0.7:
            data['success_rate'] = (
                (data['success_rate'] * (data['usage_count'] - 1) + 1) / 
                data['usage_count']
            )
        else:
            data['success_rate'] = (
                (data['success_rate'] * (data['usage_count'] - 1)) / 
                data['usage_count']
            )
    
    def get_template_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for template improvements"""
        recommendations = {
            'underperforming': [],
            'overused': [],
            'optimal': [],
            'needs_adjustment': []
        }
        
        for template_name, data in self.performance_data.items():
            if data['usage_count'] < 5:
                continue  # Not enough data
            
            avg_confidence = data['total_confidence'] / data['usage_count']
            avg_time = data['total_time'] / data['usage_count']
            avg_feedback = (
                sum(data['feedback_scores']) / len(data['feedback_scores'])
                if data['feedback_scores'] else None
            )
            
            # Classify template performance
            if avg_confidence < 0.6:
                recommendations['underperforming'].append({
                    'template': template_name,
                    'avg_confidence': avg_confidence,
                    'usage_count': data['usage_count'],
                    'suggestion': 'Consider revising reasoning steps or applicability criteria'
                })
            elif data['usage_count'] > 100 and avg_time > 1.0:
                recommendations['overused'].append({
                    'template': template_name,
                    'usage_count': data['usage_count'],
                    'avg_time': avg_time,
                    'suggestion': 'High usage with slow performance - optimize or cache more aggressively'
                })
            elif avg_confidence > 0.8 and avg_time < 0.5:
                recommendations['optimal'].append({
                    'template': template_name,
                    'metrics': {
                        'confidence': avg_confidence,
                        'time': avg_time,
                        'success_rate': data['success_rate']
                    }
                })
            elif avg_feedback and avg_feedback < 3.0:  # Assuming 1-5 scale
                recommendations['needs_adjustment'].append({
                    'template': template_name,
                    'user_satisfaction': avg_feedback,
                    'suggestion': 'Low user satisfaction - review generated steps and language'
                })
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        total_templates = len(self.performance_data)
        total_usage = sum(data['usage_count'] for data in self.performance_data.values())
        
        if total_usage == 0:
            return {'message': 'No template usage data available'}
        
        avg_confidence = sum(
            data['total_confidence'] for data in self.performance_data.values()
        ) / total_usage
        
        avg_time = sum(
            data['total_time'] for data in self.performance_data.values()
        ) / total_usage
        
        return {
            'total_templates': total_templates,
            'total_usage': total_usage,
            'avg_confidence': avg_confidence,
            'avg_time': avg_time,
            'most_used_template': max(
                self.performance_data.items(),
                key=lambda x: x[1]['usage_count']
            )[0] if self.performance_data else None
        }


# Pre-built specialized templates
class LegalReasoningTemplate(ReasoningTemplate):
    """Specialized template for legal reasoning"""
    
    def __init__(self):
        super().__init__(
            "legal",
            "Legal analysis and argumentation"
        )
        self.legal_framework = {
            'steps': [
                'issue_identification',
                'rule_statement',
                'application',
                'conclusion'
            ],
            'connectors': [
                'Under the law',
                'The relevant statute states',
                'Applying this to the facts',
                'Therefore, legally speaking'
            ]
        }
    
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        steps = []
        
        # IRAC method (Issue, Rule, Application, Conclusion)
        steps.append("First, I'll identify the legal issue at hand")
        steps.append("Next, I'll examine the relevant laws and precedents")
        steps.append("I'll apply the legal principles to the specific facts")
        steps.append("Based on legal analysis, I'll draw conclusions")
        
        # Add additional steps based on complexity
        if context.get('complexity', 0) > 0.7:
            steps.extend([
                "I'll also consider counterarguments and exceptions",
                "Let me examine relevant case law and precedents",
                "I'll analyze potential legal remedies or outcomes"
            ])
        
        return steps
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        legal_indicators = [
            'legal', 'law', 'statute', 'regulation', 'court',
            'contract', 'liability', 'rights', 'obligations',
            'precedent', 'jurisdiction', 'compliance'
        ]
        
        query_lower = query.lower()
        indicator_count = sum(1 for ind in legal_indicators if ind in query_lower)
        
        return min(indicator_count * 0.2, 1.0)


class CreativeReasoningTemplate(ReasoningTemplate):
    """Template for creative and artistic reasoning"""
    
    def __init__(self):
        super().__init__(
            "creative",
            "Creative and artistic exploration"
        )
    
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        return [
            "Let me explore the creative aspects of this question",
            "I'll consider different artistic perspectives and interpretations",
            "Examining the emotional and aesthetic dimensions",
            "Looking at innovative or unconventional approaches",
            "Synthesizing ideas to create something unique",
            "Considering the cultural and contextual influences",
            "Bringing together all creative elements for a final vision"
        ]
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        creative_keywords = [
            'creative', 'artistic', 'design', 'imagine', 'innovative',
            'aesthetic', 'beauty', 'expression', 'inspiration', 'original'
        ]
        
        query_lower = query.lower()
        keyword_matches = sum(1 for kw in creative_keywords if kw in query_lower)
        
        return min(keyword_matches * 0.3, 1.0)


class DebugReasoningTemplate(ReasoningTemplate):
    """Template for debugging and troubleshooting"""
    
    def __init__(self):
        super().__init__(
            "debug",
            "Systematic debugging and troubleshooting"
        )
    
    def generate_steps(self, query: str, context: Dict[str, Any]) -> List[str]:
        return [
            "First, I'll identify the symptoms of the problem",
            "Gathering relevant error messages and logs",
            "Analyzing the system state and recent changes",
            "Formulating hypotheses about potential causes",
            "Testing each hypothesis systematically",
            "Isolating the root cause through elimination",
            "Developing and validating a solution",
            "Implementing preventive measures"
        ]
    
    def is_applicable(self, query: str, features: Dict[str, float]) -> float:
        debug_keywords = [
            'error', 'bug', 'issue', 'problem', 'fix', 'debug',
            'troubleshoot', 'not working', 'failed', 'crash', 'exception'
        ]
        
        query_lower = query.lower()
        keyword_matches = sum(1 for kw in debug_keywords if kw in query_lower)
        
        return min(keyword_matches * 0.25, 1.0)


# Example usage
if __name__ == "__main__":
    # Create template factory
    factory = TemplateFactory()
    
    # Create domain-specific templates
    scientific_template = factory.create_template('scientific', 'research')
    business_template = factory.create_template('business', 'strategy')
    technical_template = factory.create_template('technical', 'development')
    
    # Test template creation
    logger.info("Available domains:", extra={"data": factory.get_available_domains()})
    logger.info("Created templates:", extra={"data": list(factory.template_registry.keys())})
    
    # Test template applicability
    test_query = "Analyze the experimental results and draw conclusions"
    features = {'domain_specificity': 0.8}
    
    scientific_score = scientific_template.is_applicable(test_query, features)
    business_score = business_template.is_applicable(test_query, features)
    
    logger.info("Scientific template score: {}", extra={"scientific_score": scientific_score})
    logger.info("Business template score: {}", extra={"business_score": business_score}) 