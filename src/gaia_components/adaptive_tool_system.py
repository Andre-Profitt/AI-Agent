"""
Adaptive Tool System for GAIA-enhanced FSMReActAgent
Implements ML-based tool recommendation and intelligent failure recovery
"""

import logging
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import pickle
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class ToolType(str, Enum):
    """Types of tools"""
    SEARCH = "search"
    CALCULATION = "calculation"
    ANALYSIS = "analysis"
    CREATION = "creation"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    STORAGE = "storage"

class ToolStatus(str, Enum):
    """Tool status indicators"""
    AVAILABLE = "available"
    BUSY = "busy"
    FAILING = "failing"
    OFFLINE = "offline"
    DEPRECATED = "deprecated"

class ToolPerformance:
    """Tracks tool performance metrics"""
    
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_response_time = 0.0
        self.last_success = None
        self.last_failure = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.error_history = deque(maxlen=100)
        self.response_time_history = deque(maxlen=100)
    
    def record_call(self, success: bool, response_time: float, error: str = None):
        """Record a tool call"""
        self.total_calls += 1
        self.total_response_time += response_time
        self.response_time_history.append(response_time)
        
        if success:
            self.successful_calls += 1
            self.last_success = datetime.now()
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.failed_calls += 1
            self.last_failure = datetime.now()
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            if error:
                self.error_history.append({
                    'timestamp': datetime.now(),
                    'error': error
                })
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    def get_average_response_time(self) -> float:
        """Calculate average response time"""
        if self.total_calls == 0:
            return 0.0
        return self.total_response_time / self.total_calls
    
    def get_recent_success_rate(self, window: int = 10) -> float:
        """Calculate success rate for recent calls"""
        recent_calls = list(self.response_time_history)[-window:]
        if not recent_calls:
            return 0.0
        
        # Estimate recent success rate based on response times
        # (This is a simplified approach - in practice, you'd track success/failure explicitly)
        avg_recent_time = sum(recent_calls) / len(recent_calls)
        if avg_recent_time < 1.0:  # Fast responses likely indicate success
            return 0.9
        elif avg_recent_time < 5.0:
            return 0.7
        else:
            return 0.3
    
    def is_healthy(self) -> bool:
        """Check if tool is healthy"""
        if self.total_calls < 5:
            return True  # New tool, assume healthy
        
        success_rate = self.get_success_rate()
        recent_success_rate = self.get_recent_success_rate()
        
        return success_rate > 0.7 and recent_success_rate > 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.get_success_rate(),
            "average_response_time": self.get_average_response_time(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "is_healthy": self.is_healthy()
        }

@dataclass
class ToolCapability:
    """Represents a tool's capability"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class Tool:
    """Represents a tool in the adaptive system"""
    id: str
    name: str
    tool_type: ToolType
    capabilities: List[ToolCapability]
    status: ToolStatus = ToolStatus.AVAILABLE
    performance: ToolPerformance = field(default_factory=ToolPerformance)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = ToolPerformance()
    
    def use(self):
        """Mark tool as used"""
        self.last_used = datetime.now()
    
    def get_reliability_score(self) -> float:
        """Calculate reliability score"""
        if not self.performance.is_healthy():
            return 0.0
        
        success_rate = self.performance.get_success_rate()
        avg_response_time = self.performance.get_average_response_time()
        
        # Normalize response time (faster is better)
        time_score = max(0.0, 1.0 - (avg_response_time / 10.0))
        
        # Combine scores
        return (success_rate * 0.7) + (time_score * 0.3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tool_type": self.tool_type.value,
            "status": self.status.value,
            "performance": self.performance.to_dict(),
            "reliability_score": self.get_reliability_score(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata
        }

class ToolRecommendationEngine:
    """ML-based tool recommendation engine"""
    
    def __init__(self):
        self.tool_embeddings: Dict[str, List[float]] = {}
        self.task_embeddings: Dict[str, List[float]] = {}
        self.usage_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.success_patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def update_tool_embedding(self, tool_id: str, description: str):
        """Update tool embedding"""
        # Simple hash-based embedding (in practice, use proper NLP embeddings)
        embedding = self._create_embedding(description)
        self.tool_embeddings[tool_id] = embedding
    
    def update_task_embedding(self, task_id: str, description: str):
        """Update task embedding"""
        embedding = self._create_embedding(description)
        self.task_embeddings[task_id] = embedding
    
    def record_usage(self, task_description: str, tool_id: str, success: bool):
        """Record tool usage for a task"""
        task_key = self._normalize_task(task_description)
        self.usage_patterns[task_key][tool_id] += 1
        
        # Update success rate
        current_success_rate = self.success_patterns[task_key][tool_id]
        total_usage = self.usage_patterns[task_key][tool_id]
        
        if success:
            new_success_rate = ((current_success_rate * (total_usage - 1)) + 1) / total_usage
        else:
            new_success_rate = (current_success_rate * (total_usage - 1)) / total_usage
        
        self.success_patterns[task_key][tool_id] = new_success_rate
    
    def recommend_tools(self, task_description: str, available_tools: List[Tool], 
                       max_recommendations: int = 5) -> List[Tuple[Tool, float]]:
        """Recommend tools for a task"""
        task_key = self._normalize_task(task_description)
        task_embedding = self._create_embedding(task_description)
        
        recommendations = []
        
        for tool in available_tools:
            if tool.status != ToolStatus.AVAILABLE:
                continue
            
            # Calculate recommendation score
            score = self._calculate_recommendation_score(
                task_key, task_embedding, tool
            )
            
            if score > 0.1:  # Minimum threshold
                recommendations.append((tool, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_recommendations]
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create a simple embedding for text"""
        # Simple hash-based embedding (in practice, use proper NLP embeddings)
        hash_val = hash(text.lower()) % 1000
        embedding = [float(hash_val + i) / 1000.0 for i in range(50)]
        return embedding
    
    def _normalize_task(self, task_description: str) -> str:
        """Normalize task description for pattern matching"""
        return task_description.lower().strip()
    
    def _calculate_recommendation_score(self, task_key: str, task_embedding: List[float], 
                                      tool: Tool) -> float:
        """Calculate recommendation score for a tool"""
        score = 0.0
        
        # 1. Usage pattern score (40%)
        usage_count = self.usage_patterns[task_key].get(tool.id, 0)
        usage_score = min(usage_count / 10.0, 1.0)  # Normalize to 0-1
        score += usage_score * 0.4
        
        # 2. Success pattern score (30%)
        success_rate = self.success_patterns[task_key].get(tool.id, 0.5)
        score += success_rate * 0.3
        
        # 3. Tool reliability score (20%)
        reliability_score = tool.get_reliability_score()
        score += reliability_score * 0.2
        
        # 4. Semantic similarity score (10%)
        if tool.id in self.tool_embeddings:
            similarity = self._cosine_similarity(task_embedding, self.tool_embeddings[tool.id])
            score += similarity * 0.1
        
        return score
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class FailureRecoveryEngine:
    """Intelligent failure recovery engine"""
    
    def __init__(self):
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.recovery_strategies: Dict[str, List[Callable]] = defaultdict(list)
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies"""
        # Retry strategy
        self.recovery_strategies["retry"] = [
            self._exponential_backoff_retry,
            self._linear_backoff_retry
        ]
        
        # Fallback strategy
        self.recovery_strategies["fallback"] = [
            self._find_alternative_tool,
            self._use_cached_result,
            self._degrade_functionality
        ]
        
        # Circuit breaker strategy
        self.recovery_strategies["circuit_breaker"] = [
            self._open_circuit_breaker,
            self._half_open_circuit_breaker
        ]
    
    def record_failure(self, tool_id: str, error: str, context: Dict[str, Any] = None):
        """Record a tool failure"""
        failure_record = {
            'timestamp': datetime.now(),
            'error': error,
            'context': context or {},
            'recovery_attempted': False
        }
        
        self.failure_patterns[tool_id].append(failure_record)
        
        # Update circuit breaker
        self._update_circuit_breaker(tool_id, False)
    
    def record_success(self, tool_id: str):
        """Record a tool success"""
        # Update circuit breaker
        self._update_circuit_breaker(tool_id, True)
    
    def get_recovery_plan(self, tool_id: str, error: str) -> List[Dict[str, Any]]:
        """Get recovery plan for a tool failure"""
        recovery_plan = []
        
        # Check circuit breaker status
        circuit_status = self._get_circuit_breaker_status(tool_id)
        if circuit_status == "open":
            recovery_plan.append({
                'strategy': 'circuit_breaker',
                'action': 'wait_for_recovery',
                'priority': 1,
                'description': 'Circuit breaker is open, waiting for recovery'
            })
            return recovery_plan
        
        # Analyze failure pattern
        failure_count = len(self.failure_patterns[tool_id])
        
        if failure_count == 1:
            # First failure - try retry
            recovery_plan.append({
                'strategy': 'retry',
                'action': 'exponential_backoff',
                'priority': 1,
                'description': 'First failure, retrying with exponential backoff'
            })
        
        elif failure_count <= 3:
            # Multiple failures - try different retry strategy
            recovery_plan.append({
                'strategy': 'retry',
                'action': 'linear_backoff',
                'priority': 1,
                'description': 'Multiple failures, trying linear backoff'
            })
            recovery_plan.append({
                'strategy': 'fallback',
                'action': 'find_alternative',
                'priority': 2,
                'description': 'Looking for alternative tool'
            })
        
        else:
            # Many failures - use fallback strategies
            recovery_plan.append({
                'strategy': 'fallback',
                'action': 'use_cached_result',
                'priority': 1,
                'description': 'Using cached result due to repeated failures'
            })
            recovery_plan.append({
                'strategy': 'fallback',
                'action': 'degrade_functionality',
                'priority': 2,
                'description': 'Degrading functionality to continue operation'
            })
            recovery_plan.append({
                'strategy': 'circuit_breaker',
                'action': 'open_circuit',
                'priority': 3,
                'description': 'Opening circuit breaker to prevent further failures'
            })
        
        return recovery_plan
    
    def execute_recovery_strategy(self, strategy: str, action: str, 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a recovery strategy"""
        context = context or {}
        
        if strategy == "retry":
            if action == "exponential_backoff":
                return self._exponential_backoff_retry(context)
            elif action == "linear_backoff":
                return self._linear_backoff_retry(context)
        
        elif strategy == "fallback":
            if action == "find_alternative":
                return self._find_alternative_tool(context)
            elif action == "use_cached_result":
                return self._use_cached_result(context)
            elif action == "degrade_functionality":
                return self._degrade_functionality(context)
        
        elif strategy == "circuit_breaker":
            if action == "open_circuit":
                return self._open_circuit_breaker(context)
            elif action == "half_open_circuit_breaker":
                return self._half_open_circuit_breaker(context)
        
        return {
            'success': False,
            'error': f'Unknown recovery strategy: {strategy}.{action}'
        }
    
    def _exponential_backoff_retry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Exponential backoff retry strategy"""
        attempt = context.get('attempt', 1)
        max_attempts = context.get('max_attempts', 3)
        
        if attempt > max_attempts:
            return {
                'success': False,
                'error': 'Max retry attempts exceeded'
            }
        
        wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60 seconds
        
        return {
            'success': True,
            'action': 'wait_and_retry',
            'wait_time': wait_time,
            'attempt': attempt + 1
        }
    
    def _linear_backoff_retry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Linear backoff retry strategy"""
        attempt = context.get('attempt', 1)
        max_attempts = context.get('max_attempts', 5)
        
        if attempt > max_attempts:
            return {
                'success': False,
                'error': 'Max retry attempts exceeded'
            }
        
        wait_time = attempt * 5  # Linear backoff, 5 seconds per attempt
        
        return {
            'success': True,
            'action': 'wait_and_retry',
            'wait_time': wait_time,
            'attempt': attempt + 1
        }
    
    def _find_alternative_tool(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find alternative tool strategy"""
        failed_tool_id = context.get('failed_tool_id')
        available_tools = context.get('available_tools', [])
        
        # Find tools of the same type
        failed_tool = None
        for tool in available_tools:
            if tool.id == failed_tool_id:
                failed_tool = tool
                break
        
        if not failed_tool:
            return {
                'success': False,
                'error': 'Failed tool not found'
            }
        
        # Find alternative tools of the same type
        alternatives = [
            tool for tool in available_tools
            if tool.id != failed_tool_id and tool.tool_type == failed_tool.tool_type
        ]
        
        if alternatives:
            # Choose the most reliable alternative
            best_alternative = max(alternatives, key=lambda t: t.get_reliability_score())
            return {
                'success': True,
                'action': 'use_alternative',
                'alternative_tool_id': best_alternative.id,
                'alternative_tool_name': best_alternative.name
            }
        else:
            return {
                'success': False,
                'error': 'No alternative tools available'
            }
    
    def _use_cached_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use cached result strategy"""
        cache_key = context.get('cache_key')
        
        if not cache_key:
            return {
                'success': False,
                'error': 'No cache key provided'
            }
        
        # In a real implementation, this would check an actual cache
        # For now, return a mock response
        return {
            'success': True,
            'action': 'use_cached_result',
            'cache_key': cache_key,
            'result': 'cached_result_placeholder'
        }
    
    def _degrade_functionality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Degrade functionality strategy"""
        return {
            'success': True,
            'action': 'degrade_functionality',
            'message': 'Functionality degraded to continue operation',
            'degraded_features': ['advanced_analysis', 'real_time_processing']
        }
    
    def _open_circuit_breaker(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Open circuit breaker strategy"""
        tool_id = context.get('tool_id')
        if tool_id:
            self.circuit_breakers[tool_id] = {
                'status': 'open',
                'opened_at': datetime.now(),
                'failure_count': len(self.failure_patterns[tool_id])
            }
        
        return {
            'success': True,
            'action': 'circuit_opened',
            'message': 'Circuit breaker opened to prevent further failures'
        }
    
    def _half_open_circuit_breaker(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Half-open circuit breaker strategy"""
        tool_id = context.get('tool_id')
        if tool_id and tool_id in self.circuit_breakers:
            self.circuit_breakers[tool_id]['status'] = 'half-open'
        
        return {
            'success': True,
            'action': 'circuit_half_open',
            'message': 'Circuit breaker half-open for testing'
        }
    
    def _update_circuit_breaker(self, tool_id: str, success: bool):
        """Update circuit breaker status"""
        if tool_id not in self.circuit_breakers:
            self.circuit_breakers[tool_id] = {
                'status': 'closed',
                'failure_count': 0,
                'success_count': 0
            }
        
        breaker = self.circuit_breakers[tool_id]
        
        if success:
            breaker['success_count'] += 1
            if breaker['status'] == 'half-open' and breaker['success_count'] >= 3:
                breaker['status'] = 'closed'
                breaker['failure_count'] = 0
        else:
            breaker['failure_count'] += 1
            if breaker['failure_count'] >= 5:
                breaker['status'] = 'open'
                breaker['opened_at'] = datetime.now()
    
    def _get_circuit_breaker_status(self, tool_id: str) -> str:
        """Get circuit breaker status for a tool"""
        if tool_id not in self.circuit_breakers:
            return 'closed'
        
        breaker = self.circuit_breakers[tool_id]
        
        if breaker['status'] == 'open':
            # Check if enough time has passed to try half-open
            opened_at = breaker.get('opened_at')
            if opened_at and (datetime.now() - opened_at).total_seconds() > 300:  # 5 minutes
                breaker['status'] = 'half-open'
                return 'half-open'
            return 'open'
        
        return breaker['status']

class AdaptiveToolSystem:
    """Main adaptive tool system"""
    
    def __init__(self, persist_directory: str = "./tool_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.tools: Dict[str, Tool] = {}
        self.recommendation_engine = ToolRecommendationEngine()
        self.recovery_engine = FailureRecoveryEngine()
        
        # Load existing tools
        self._load_tools()
        
        logger.info("Adaptive Tool System initialized")
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.id] = tool
        
        # Update recommendation engine
        for capability in tool.capabilities:
            self.recommendation_engine.update_tool_embedding(
                tool.id, capability.description
            )
        
        logger.info(f"Registered tool: {tool.name} ({tool.id})")
    
    def unregister_tool(self, tool_id: str):
        """Unregister a tool"""
        if tool_id in self.tools:
            del self.tools[tool_id]
            logger.info(f"Unregistered tool: {tool_id}")
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID"""
        return self.tools.get(tool_id)
    
    def get_available_tools(self) -> List[Tool]:
        """Get all available tools"""
        return [tool for tool in self.tools.values() if tool.status == ToolStatus.AVAILABLE]
    
    def recommend_tools_for_task(self, task_description: str, 
                                max_recommendations: int = 5) -> List[Tuple[Tool, float]]:
        """Recommend tools for a task"""
        available_tools = self.get_available_tools()
        return self.recommendation_engine.recommend_tools(
            task_description, available_tools, max_recommendations
        )
    
    def execute_tool(self, tool_id: str, parameters: Dict[str, Any], 
                    task_description: str = None) -> Dict[str, Any]:
        """Execute a tool with failure recovery"""
        if tool_id not in self.tools:
            return {
                'success': False,
                'error': f'Tool not found: {tool_id}'
            }
        
        tool = self.tools[tool_id]
        start_time = time.time()
        
        try:
            # Check circuit breaker
            circuit_status = self.recovery_engine._get_circuit_breaker_status(tool_id)
            if circuit_status == "open":
                return {
                    'success': False,
                    'error': 'Circuit breaker is open',
                    'recovery_plan': self.recovery_engine.get_recovery_plan(tool_id, "circuit_open")
                }
            
            # Execute tool (mock execution)
            result = self._mock_tool_execution(tool, parameters)
            
            # Record success
            response_time = time.time() - start_time
            tool.performance.record_call(True, response_time)
            tool.use()
            self.recovery_engine.record_success(tool_id)
            
            # Update recommendation engine
            if task_description:
                self.recommendation_engine.record_usage(task_description, tool_id, True)
            
            return {
                'success': True,
                'result': result,
                'response_time': response_time,
                'tool_id': tool_id
            }
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            tool.performance.record_call(False, response_time, str(e))
            self.recovery_engine.record_failure(tool_id, str(e))
            
            # Get recovery plan
            recovery_plan = self.recovery_engine.get_recovery_plan(tool_id, str(e))
            
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'tool_id': tool_id,
                'recovery_plan': recovery_plan
            }
    
    def execute_with_recovery(self, tool_id: str, parameters: Dict[str, Any], 
                            task_description: str = None) -> Dict[str, Any]:
        """Execute tool with automatic recovery"""
        # Initial execution
        result = self.execute_tool(tool_id, parameters, task_description)
        
        if result['success']:
            return result
        
        # Try recovery strategies
        recovery_plan = result.get('recovery_plan', [])
        
        for recovery_step in recovery_plan:
            strategy = recovery_step['strategy']
            action = recovery_step['action']
            
            recovery_result = self.recovery_engine.execute_recovery_strategy(
                strategy, action, {
                    'failed_tool_id': tool_id,
                    'available_tools': list(self.tools.values()),
                    'parameters': parameters,
                    'task_description': task_description
                }
            )
            
            if recovery_result['success']:
                # Try the recovery action
                if recovery_result['action'] == 'use_alternative':
                    alternative_tool_id = recovery_result['alternative_tool_id']
                    return self.execute_tool(alternative_tool_id, parameters, task_description)
                elif recovery_result['action'] == 'wait_and_retry':
                    # In a real implementation, you'd wait and retry
                    time.sleep(recovery_result['wait_time'])
                    return self.execute_tool(tool_id, parameters, task_description)
                elif recovery_result['action'] == 'use_cached_result':
                    return {
                        'success': True,
                        'result': recovery_result['result'],
                        'cached': True,
                        'tool_id': tool_id
                    }
        
        return result
    
    def _mock_tool_execution(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Mock tool execution (replace with actual tool execution)"""
        # Simulate different tool types
        if tool.tool_type == ToolType.SEARCH:
            return f"Search results for: {parameters.get('query', 'unknown')}"
        elif tool.tool_type == ToolType.CALCULATION:
            return f"Calculation result: {parameters.get('expression', 'unknown')}"
        elif tool.tool_type == ToolType.ANALYSIS:
            return f"Analysis result for: {parameters.get('data', 'unknown')}"
        else:
            return f"Tool execution result: {parameters}"
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_tools = len(self.tools)
        available_tools = len(self.get_available_tools())
        
        tool_stats = {}
        for tool in self.tools.values():
            tool_stats[tool.name] = {
                'status': tool.status.value,
                'reliability_score': tool.get_reliability_score(),
                'performance': tool.performance.to_dict()
            }
        
        return {
            'total_tools': total_tools,
            'available_tools': available_tools,
            'tool_statistics': tool_stats,
            'recommendation_engine_stats': {
                'tool_embeddings': len(self.recommendation_engine.tool_embeddings),
                'task_embeddings': len(self.recommendation_engine.task_embeddings),
                'usage_patterns': len(self.recommendation_engine.usage_patterns)
            },
            'recovery_engine_stats': {
                'failure_patterns': len(self.recovery_engine.failure_patterns),
                'circuit_breakers': len(self.recovery_engine.circuit_breakers)
            }
        }
    
    def _load_tools(self):
        """Load tools from persistent storage"""
        try:
            tools_file = self.persist_directory / "tools.pkl"
            if tools_file.exists():
                with open(tools_file, 'rb') as f:
                    tools_data = pickle.load(f)
                    for tool_data in tools_data:
                        tool = Tool(
                            id=tool_data['id'],
                            name=tool_data['name'],
                            tool_type=ToolType(tool_data['tool_type']),
                            capabilities=[],  # Simplified for now
                            status=ToolStatus(tool_data['status'])
                        )
                        self.tools[tool.id] = tool
                
                logger.info(f"Loaded {len(self.tools)} tools from persistent storage")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
    
    def save_tools(self):
        """Save tools to persistent storage"""
        try:
            tools_data = [tool.to_dict() for tool in self.tools.values()]
            with open(self.persist_directory / "tools.pkl", 'wb') as f:
                pickle.dump(tools_data, f)
            
            logger.info("Tools saved to persistent storage")
        except Exception as e:
            logger.error(f"Failed to save tools: {e}")
    
    def create_mock_tools(self):
        """Create mock tools for testing"""
        mock_tools = [
            Tool(
                id="search_tool_1",
                name="Web Search",
                tool_type=ToolType.SEARCH,
                capabilities=[
                    ToolCapability(
                        name="web_search",
                        description="Search the web for information",
                        input_schema={"query": "string"},
                        output_schema={"results": "array"},
                        tags=["search", "web", "information"]
                    )
                ]
            ),
            Tool(
                id="calc_tool_1",
                name="Calculator",
                tool_type=ToolType.CALCULATION,
                capabilities=[
                    ToolCapability(
                        name="calculate",
                        description="Perform mathematical calculations",
                        input_schema={"expression": "string"},
                        output_schema={"result": "number"},
                        tags=["calculation", "math", "computation"]
                    )
                ]
            ),
            Tool(
                id="analysis_tool_1",
                name="Data Analyzer",
                tool_type=ToolType.ANALYSIS,
                capabilities=[
                    ToolCapability(
                        name="analyze",
                        description="Analyze data and extract insights",
                        input_schema={"data": "object"},
                        output_schema={"insights": "array"},
                        tags=["analysis", "data", "insights"]
                    )
                ]
            )
        ]
        
        for tool in mock_tools:
            self.register_tool(tool)
        
        logger.info(f"Created {len(mock_tools)} mock tools")