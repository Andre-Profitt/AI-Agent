"""
Enhanced FSM Implementation
===========================

A comprehensive Finite State Machine implementation with:
- Hierarchical states (composite and atomic)
- Probabilistic transitions with context-aware learning
- Dynamic state discovery engine
- Comprehensive metrics and monitoring
- Visualization capabilities
"""

import time
import json
import logging
import random
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

class StateType(Enum):
    """Types of states in the hierarchical FSM"""
    ATOMIC = "atomic"
    COMPOSITE = "composite"

@dataclass
class StateMetrics:
    """Metrics for tracking state performance"""
    entry_count: int = 0
    exit_count: int = 0
    total_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_entry: Optional[datetime] = None
    last_exit: Optional[datetime] = None
    
    @property
    def avg_time(self) -> float:
        """Average time spent in state"""
        return self.total_time / max(1, self.exit_count)
    
    @property
    def success_rate(self) -> float:
        """Success rate of state transitions"""
        return self.success_count / max(1, self.exit_count)

class State:
    """Base class for all states"""
    
    def __init__(self, name: str, state_type: StateType = StateType.ATOMIC):
        self.name = name
        self.state_type = state_type
        self.action: Optional[Callable] = None
        self.entry_action: Optional[Callable] = None
        self.exit_action: Optional[Callable] = None
        self.metrics = StateMetrics()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the state's action"""
        start_time = time.time()
        self.metrics.entry_count += 1
        self.metrics.last_entry = datetime.now()
        
        try:
            # Execute entry action
            if self.entry_action:
                context = self.entry_action(context)
            
            # Execute main action
            if self.action:
                context = self.action(context)
            
            # Execute exit action
            if self.exit_action:
                context = self.exit_action(context)
            
            self.metrics.success_count += 1
            
        except Exception as e:
            self.metrics.failure_count += 1
            logger.error("Error executing state {}: {}", extra={"self_name": self.name, "e": e})
            raise
        
        finally:
            end_time = time.time()
            self.metrics.total_time += (end_time - start_time)
            self.metrics.exit_count += 1
            self.metrics.last_exit = datetime.now()
        
        return context

class AtomicState(State):
    """Atomic state that cannot contain other states"""
    
    def __init__(self, name: str):
        super().__init__(name, StateType.ATOMIC)

class CompositeState(State):
    """Composite state that can contain other states"""
    
    def __init__(self, name: str):
        super().__init__(name, StateType.COMPOSITE)
        self.substates: Dict[str, State] = {}
        self.current_substate: Optional[State] = None
    
    def add_substate(self, state: State):
        """Add a substate to this composite state"""
        self.substates[state.name] = state
    
    def get_substate(self, name: str) -> Optional[State]:
        """Get a substate by name"""
        return self.substates.get(name)

class ProbabilisticTransition:
    """Probabilistic transition between states"""
    
    def __init__(self, from_state: str, to_state: str, base_probability: float = 1.0):
        self.from_state = from_state
        self.to_state = to_state
        self.base_probability = base_probability
        self.context_modifiers: List[Dict[str, Any]] = []
        self.usage_count = 0
        self.success_count = 0
    
    def add_context_modifier(self, condition: str, probability_modifier: float):
        """Add a context-based probability modifier"""
        self.context_modifiers.append({
            'condition': condition,
            'probability_modifier': probability_modifier
        })
    
    def get_probability(self, context: Dict[str, Any]) -> float:
        """Calculate the probability of this transition given the context"""
        probability = self.base_probability
        
        # Apply context modifiers
        for modifier in self.context_modifiers:
            if self._evaluate_condition(modifier['condition'], context):
                probability = modifier['probability_modifier']
                break
        
        # Apply learning from usage
        if self.usage_count > 0:
            success_rate = self.success_count / self.usage_count
            probability *= (0.5 + 0.5 * success_rate)
        
        return max(0.0, min(1.0, probability))
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against the context"""
        try:
            # Simple condition evaluation
            if '<' in condition:
                var, val = condition.split('<')
                return context.get(var.strip(), 0) < float(val.strip())
            elif '>' in condition:
                var, val = condition.split('>')
                return context.get(var.strip(), 0) > float(val.strip())
            elif '==' in condition:
                var, val = condition.split('==')
                return context.get(var.strip()) == val.strip().strip('"\'')
            else:
                return bool(context.get(condition.strip(), False))
        except:
            return False
    
    def record_usage(self, success: bool):
        """Record usage of this transition"""
        self.usage_count += 1
        if success:
            self.success_count += 1

@dataclass
class DiscoveredPattern:
    """A discovered state pattern"""
    name: str
    features: Dict[str, Any]
    confidence: float
    frequency: int
    discovered_at: datetime = field(default_factory=datetime.now)

class StateDiscoveryEngine:
    """Engine for discovering new states based on context patterns"""
    
    def __init__(self, similarity_threshold: float = 0.8, min_pattern_frequency: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_pattern_frequency = min_pattern_frequency
        self.patterns: List[DiscoveredPattern] = []
        self.context_history: List[Dict[str, Any]] = []
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(eps=0.3, min_samples=2)
    
    def analyze_context(self, context: Dict[str, Any]) -> Optional[DiscoveredPattern]:
        """Analyze context and potentially discover new patterns"""
        self.context_history.append(context)
        
        # Extract features from context
        features = self._extract_features(context)
        
        # Check if this matches existing patterns
        for pattern in self.patterns:
            similarity = self._calculate_similarity(features, pattern.features)
            if similarity >= self.similarity_threshold:
                pattern.frequency += 1
                return pattern
        
        # Check if we have enough similar contexts to form a new pattern
        similar_contexts = self._find_similar_contexts(features)
        if len(similar_contexts) >= self.min_pattern_frequency:
            # Create new pattern
            pattern = DiscoveredPattern(
                name=f"DiscoveredPattern_{len(self.patterns) + 1}",
                features=features,
                confidence=self._calculate_confidence(similar_contexts),
                frequency=len(similar_contexts)
            )
            self.patterns.append(pattern)
            return pattern
        
        return None
    
    def _extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numerical features from context"""
        features = {}
        
        # Extract tool usage patterns
        if 'recent_tools' in context:
            features['tool_count'] = len(context['recent_tools'])
            features['unique_tools'] = len(set(context['recent_tools']))
        
        # Extract error patterns
        if 'error_types' in context:
            features['error_count'] = len(context['error_types'])
        
        # Extract data statistics
        if 'data_stats' in context:
            stats = context['data_stats']
            features['result_count'] = stats.get('result_count', 0)
            features['error_rate'] = stats.get('error_rate', 0.0)
        
        # Extract metrics
        if 'metrics' in context:
            metrics = context['metrics']
            features['execution_time'] = metrics.get('execution_time', 0.0)
            features['confidence'] = metrics.get('confidence', 0.0)
        
        return features
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
        
        # Convert to vectors
        keys = set(features1.keys()) | set(features2.keys())
        vec1 = [features1.get(k, 0) for k in keys]
        vec2 = [features2.get(k, 0) for k in keys]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _find_similar_contexts(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find contexts similar to the given features"""
        similar = []
        for context in self.context_history:
            context_features = self._extract_features(context)
            similarity = self._calculate_similarity(features, context_features)
            if similarity >= self.similarity_threshold:
                similar.append(context)
        return similar
    
    def _calculate_confidence(self, similar_contexts: List[Dict[str, Any]]) -> float:
        """Calculate confidence for a discovered pattern"""
        return min(1.0, len(similar_contexts) / self.min_pattern_frequency)
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered patterns"""
        if not self.patterns:
            return {
                'total_patterns': 0,
                'most_used_pattern': None,
                'recent_discoveries': 0,
                'average_confidence': 0.0
            }
        
        most_used = max(self.patterns, key=lambda p: p.frequency)
        recent_discoveries = len([p for p in self.patterns 
                                if (datetime.now() - p.discovered_at).days < 7])
        avg_confidence = sum(p.confidence for p in self.patterns) / len(self.patterns)
        
        return {
            'total_patterns': len(self.patterns),
            'most_used_pattern': most_used.name,
            'recent_discoveries': recent_discoveries,
            'average_confidence': avg_confidence
        }

class HierarchicalFSM:
    """Hierarchical Finite State Machine with enhanced features"""
    
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: List[ProbabilisticTransition] = []
        self.current_state: Optional[State] = None
        self.context: Dict[str, Any] = {}
        self.transition_log: List[Dict[str, Any]] = []
        self.discovery_engine = StateDiscoveryEngine()
        self.started = False
    
    def add_state(self, state: State):
        """Add a state to the FSM"""
        self.states[state.name] = state
    
    def add_transition(self, transition: ProbabilisticTransition):
        """Add a transition to the FSM"""
        self.transitions.append(transition)
    
    def start(self, initial_state: str, context: Dict[str, Any]):
        """Start the FSM in the specified initial state"""
        if initial_state not in self.states:
            raise ValueError(f"Initial state '{initial_state}' not found")
        
        self.current_state = self.states[initial_state]
        self.context = context.copy()
        self.started = True
        
        logger.info("FSM '{}' started in state '{}'", extra={"self_name": self.name, "initial_state": initial_state})
    
    def transition_to(self, target_state: str, context: Dict[str, Any]) -> bool:
        """Transition to a target state"""
        if not self.started or not self.current_state:
            raise RuntimeError("FSM not started")
        
        if target_state not in self.states:
            logger.error("Target state '{}' not found", extra={"target_state": target_state})
            return False
        
        # Find transition
        transition = self._find_transition(self.current_state.name, target_state)
        if not transition:
            logger.error("No transition from '{}' to '{}'", extra={"self_current_state_name": self.current_state.name, "target_state": target_state})
            return False
        
        # Calculate probability
        probability = transition.get_probability(context)
        
        # Decide whether to take the transition
        if random.random() <= probability:
            # Record transition
            old_state = self.current_state.name
            self.current_state = self.states[target_state]
            
            # Update context
            self.context.update(context)
            
            # Log transition
            self.transition_log.append({
                'timestamp': datetime.now(),
                'from_state': old_state,
                'to_state': target_state,
                'probability': probability,
                'context': context.copy()
            })
            
            # Record transition usage
            transition.record_usage(True)
            
            logger.info("Transitioned from '{}' to '{}' (p={})", extra={"old_state": old_state, "target_state": target_state, "probability": probability})
            return True
        else:
            transition.record_usage(False)
            logger.info("Transition to '{}' rejected (p={})", extra={"target_state": target_state, "probability": probability})
            return False
    
    def _find_transition(self, from_state: str, to_state: str) -> Optional[ProbabilisticTransition]:
        """Find a transition between two states"""
        for transition in self.transitions:
            if transition.from_state == from_state and transition.to_state == to_state:
                return transition
        return None
    
    def get_available_transitions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available transitions from current state with probabilities"""
        if not self.current_state:
            return []
        
        available = []
        for transition in self.transitions:
            if transition.from_state == self.current_state.name:
                probability = transition.get_probability(context)
                available.append({
                    'to_state': transition.to_state,
                    'probability': probability,
                    'transition': transition
                })
        
        return available
    
    def execute_current_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the current state's action"""
        if not self.current_state:
            raise RuntimeError("No current state")
        
        # Update context
        self.context.update(context)
        
        # Execute state
        result_context = self.current_state.execute(self.context)
        
        # Update FSM context
        self.context.update(result_context)
        
        return result_context
    
    def get_state_metrics(self) -> Dict[str, StateMetrics]:
        """Get metrics for all states"""
        return {name: state.metrics for name, state in self.states.items()}
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export comprehensive metrics"""
        return {
            'fsm_name': self.name,
            'total_states': len(self.states),
            'total_transitions': len(self.transitions),
            'current_state': self.current_state.name if self.current_state else None,
            'started': self.started,
            'state_metrics': {
                name: {
                    'entry_count': metrics.entry_count,
                    'exit_count': metrics.exit_count,
                    'total_time': metrics.total_time,
                    'success_count': metrics.success_count,
                    'failure_count': metrics.failure_count,
                    'avg_time': metrics.avg_time,
                    'success_rate': metrics.success_rate
                }
                for name, metrics in self.get_state_metrics().items()
            },
            'transition_log': [
                {
                    'timestamp': log['timestamp'].isoformat(),
                    'from_state': log['from_state'],
                    'to_state': log['to_state'],
                    'probability': log['probability']
                }
                for log in self.transition_log
            ],
            'discovery_stats': self.discovery_engine.get_pattern_statistics()
        }
    
    def visualize(self) -> str:
        """Generate a text-based visualization of the FSM"""
        if not self.states:
            return "Empty FSM"
        
        lines = [f"FSM: {self.name}", "=" * (len(self.name) + 5)]
        
        # States
        lines.append("\nStates:")
        for name, state in self.states.items():
            current_marker = " (CURRENT)" if self.current_state and self.current_state.name == name else ""
            lines.append(f"  {name} [{state.state_type.value}]{current_marker}")
        
        # Transitions
        lines.append("\nTransitions:")
        for transition in self.transitions:
            lines.append(f"  {transition.from_state} -> {transition.to_state} (p={transition.base_probability:.2f})")
        
        # Current metrics
        if self.current_state:
            metrics = self.current_state.metrics
            lines.append(f"\nCurrent State Metrics:")
            lines.append(f"  Entries: {metrics.entry_count}")
            lines.append(f"  Exits: {metrics.exit_count}")
            lines.append(f"  Success Rate: {metrics.success_rate:.1%}")
            lines.append(f"  Avg Time: {metrics.avg_time:.3f}s")
        
        return "\n".join(lines)
    
    def save_visualization(self, filename: str):
        """Save a graphical visualization of the FSM"""
        try:
            G = nx.DiGraph()
            
            # Add nodes
            for name, state in self.states.items():
                G.add_node(name, state_type=state.state_type.value)
            
            # Add edges
            for transition in self.transitions:
                G.add_edge(transition.from_state, transition.to_state, 
                          weight=transition.base_probability)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=2000, alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # Draw edge labels
            edge_labels = {(u, v): f"{d['weight']:.2f}" 
                          for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title(f"FSM: {self.name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("FSM visualization saved to {}", extra={"filename": filename})
            
        except Exception as e:
            logger.error("Failed to save visualization: {}", extra={"e": e})
            raise
