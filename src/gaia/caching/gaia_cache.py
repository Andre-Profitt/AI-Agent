import hashlib
import time
import re
from functools import lru_cache
from typing import Dict, Any, Optional
from collections import defaultdict

class GAIAResponseCache:
    """GAIA-specific caching for common patterns and expensive operations"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
        self.access_count = defaultdict(int)
        
    def get_cache_key(self, question: str, question_type: str) -> str:
        """Generate cache key for GAIA questions"""
        # Normalize question for better cache hits
        normalized = question.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        return hashlib.md5(f"{normalized}:{question_type}".encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if not expired"""
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.access_count[key] += 1
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache response with timestamp"""
        self.cache[key] = (time.time(), value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "total_entries": len(self.cache),
            "most_accessed": dict(sorted(self.access_count.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]),
            "hit_rate": sum(self.access_count.values()) / max(len(self.cache), 1)
        }

class GAIAQuestionCache:
    """Cache for question type analysis and planning"""
    
    def __init__(self):
        self.question_types = {}
        self.plans = {}
    
    @lru_cache(maxsize=1000)
    def get_question_type(self, question: str) -> Dict[str, str]:
        """Cached question type analysis"""
        from src.advanced_agent_fsm import analyze_question_type
        return analyze_question_type(question)
    
    @lru_cache(maxsize=1000)
    def get_plan(self, question: str) -> Dict[str, Any]:
        """Cached execution plan"""
        from src.advanced_agent_fsm import create_gaia_optimized_plan
        return create_gaia_optimized_plan(question)

class GAIAErrorCache:
    """Cache for error recovery strategies"""
    
    def __init__(self):
        self.error_strategies = {}
        self.recovery_success = defaultdict(lambda: {"success": 0, "total": 0})
    
    def get_recovery_strategy(self, error_type: str) -> Optional[str]:
        """Get cached recovery strategy for error type"""
        return self.error_strategies.get(error_type)
    
    def record_recovery_attempt(self, error_type: str, success: bool):
        """Record recovery attempt success/failure"""
        self.recovery_success[error_type]["total"] += 1
        if success:
            self.recovery_success[error_type]["success"] += 1
    
    def get_success_rate(self, error_type: str) -> float:
        """Get success rate for error recovery"""
        stats = self.recovery_success[error_type]
        return stats["success"] / max(stats["total"], 1)

# Global cache instances
response_cache = GAIAResponseCache()
question_cache = GAIAQuestionCache()
error_cache = GAIAErrorCache() 