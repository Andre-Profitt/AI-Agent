from agent import query
from benchmarks.cot_performance import avg_confidence
from benchmarks.cot_performance import filename
from benchmarks.cot_performance import insights
from benchmarks.cot_performance import recommendations
from benchmarks.cot_performance import template
from benchmarks.cot_performance import timestamp
from fix_security_issues import patterns
from fix_security_issues import report
from performance_dashboard import stats
from tests.load_test import data
from tests.performance.cot_benchmark_suite import p

from src.analytics.usage_analyzer import analyzer
from src.analytics.usage_analyzer import confidence_score
from src.analytics.usage_analyzer import confidence_trend
from src.analytics.usage_analyzer import day_counts
from src.analytics.usage_analyzer import dt
from src.analytics.usage_analyzer import effectiveness
from src.analytics.usage_analyzer import effectiveness_score
from src.analytics.usage_analyzer import first_avg_confidence
from src.analytics.usage_analyzer import first_avg_time
from src.analytics.usage_analyzer import first_hour
from src.analytics.usage_analyzer import hour_counts
from src.analytics.usage_analyzer import hour_key
from src.analytics.usage_analyzer import hourly_stats
from src.analytics.usage_analyzer import last_avg_confidence
from src.analytics.usage_analyzer import last_avg_time
from src.analytics.usage_analyzer import last_hour
from src.analytics.usage_analyzer import mock_result
from src.analytics.usage_analyzer import one_hour_ago
from src.analytics.usage_analyzer import peak_day
from src.analytics.usage_analyzer import peak_hours
from src.analytics.usage_analyzer import performance_trends
from src.analytics.usage_analyzer import problem_patterns
from src.analytics.usage_analyzer import recent_confidence
from src.analytics.usage_analyzer import recent_queries
from src.analytics.usage_analyzer import recent_time
from src.analytics.usage_analyzer import report_file
from src.analytics.usage_analyzer import sample_queries
from src.analytics.usage_analyzer import satisfaction_score
from src.analytics.usage_analyzer import session_analysis
from src.analytics.usage_analyzer import session_duration
from src.analytics.usage_analyzer import sorted_hours
from src.analytics.usage_analyzer import sorted_patterns
from src.analytics.usage_analyzer import speed_score
from src.analytics.usage_analyzer import template_effectiveness
from src.analytics.usage_analyzer import template_stats
from src.analytics.usage_analyzer import time_insights
from src.analytics.usage_analyzer import time_trend
from src.analytics.usage_analyzer import usage_entry
from src.analytics.usage_analyzer import worst_template
from src.collaboration.realtime_collaboration import session_id
from src.config.settings import issues
from src.core.optimized_chain_of_thought import n
from src.database.models import user_id
from src.gaia_components.performance_optimization import entry
from src.meta_cognition import confidence
from src.meta_cognition import query_lower
from src.templates.template_factory import avg_time
from src.templates.template_factory import pattern
from src.unified_architecture.shared_memory import entries

from collections import deque
from datetime import datetime
from src.reasoning.reasoning_path import ReasoningPath
from typing import Dict
from typing import Optional
# TODO: Fix undefined variables: Any, Dict, Optional, analyzer, avg_confidence, avg_time, confidence, confidence_score, confidence_trend, data, datetime, day_counts, defaultdict, deque, dt, e, effectiveness, effectiveness_score, entries, entry, exec_time, f, filename, first_avg_confidence, first_avg_time, first_hour, hour, hour_counts, hour_key, hourly_stats, i, ind, indicators, insights, issues, json, last_avg_confidence, last_avg_time, last_hour, logging, max_history, mock_result, n, one_hour_ago, os, p, pattern_type, patterns, peak_day, peak_hours, performance_trends, problem_patterns, query, query_lower, recent_confidence, recent_queries, recent_time, recommendations, report, report_file, result, sample_queries, satisfaction_score, session_analysis, session_duration, session_id, sorted_hours, sorted_patterns, speed_score, stats, template, template_effectiveness, template_stats, time, time_insights, time_trend, timestamp, usage_entry, user_feedback, user_id, worst_template, x
import pattern

# TODO: Fix undefined variables: analyzer, avg_confidence, avg_time, confidence, confidence_score, confidence_trend, data, day_counts, dt, e, effectiveness, effectiveness_score, entries, entry, exec_time, f, filename, first_avg_confidence, first_avg_time, first_hour, hour, hour_counts, hour_key, hourly_stats, i, ind, indicators, insights, issues, last_avg_confidence, last_avg_time, last_hour, max_history, mock_result, n, one_hour_ago, p, pattern, pattern_type, patterns, peak_day, peak_hours, performance_trends, problem_patterns, query, query_lower, recent_confidence, recent_queries, recent_time, recommendations, report, report_file, result, sample_queries, satisfaction_score, self, session_analysis, session_duration, session_id, sorted_hours, sorted_patterns, speed_score, stats, template, template_effectiveness, template_stats, time_insights, time_trend, timestamp, usage_entry, user_feedback, user_id, worst_template, x
"""
Usage Pattern Analyzer for Chain of Thought System
Analyze usage patterns to improve system performance
"""

from typing import Any

import time
import json
import os

from collections import defaultdict, deque

import logging

# Import the CoT system




logger = logging.getLogger(__name__)


class UsagePatternAnalyzer:
    """Analyze usage patterns to improve system performance"""
    
    def __init__(self, max_history: int = 10000) -> None:
        self.query_patterns = defaultdict(lambda: {
            'count': 0,
            'avg_complexity': 0,
            'avg_confidence': 0,
            'avg_time': 0,
            'common_templates': defaultdict(int),
            'time_distribution': defaultdict(int),  # hour of day
            'day_distribution': defaultdict(int),   # day of week
            'user_satisfaction': [],
            'error_count': 0,
            'last_used': None
        })
        self.pattern_clusters = []
        self.usage_history = deque(maxlen=max_history)
        self.session_data = defaultdict(list)
        
        # Create analytics directory if it doesn't exist
        os.makedirs('analytics', exist_ok=True)
    
    def analyze_query(self, query: str, result: ReasoningPath, 
                     timestamp: Optional[float] = None,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     user_feedback: Optional[float] = None) -> Any:
        """Analyze a query and its result"""
        if timestamp is None:
            timestamp = time.time()
        
        # Extract query pattern
        pattern = self._extract_pattern(query)
        
        # Update pattern statistics
        data = self.query_patterns[pattern]
        data['count'] += 1
        
        # Update running averages
        n = data['count']
        data['avg_complexity'] = (
            (data['avg_complexity'] * (n - 1) + result.complexity_score) / n
        )
        data['avg_confidence'] = (
            (data['avg_confidence'] * (n - 1) + result.total_confidence) / n
        )
        data['avg_time'] = (
            (data['avg_time'] * (n - 1) + result.execution_time) / n
        )
        
        # Track template usage
        if result.template_used:
            data['common_templates'][result.template_used] += 1
        
        # Track time distribution
        dt = datetime.fromtimestamp(timestamp)
        data['time_distribution'][dt.hour] += 1
        data['day_distribution'][dt.weekday()] += 1
        data['last_used'] = timestamp
        
        # Track user satisfaction
        if user_feedback is not None:
            data['user_satisfaction'].append(user_feedback)
        
        # Track errors
        if result.total_confidence < 0.3:
            data['error_count'] += 1
        
        # Store in usage history
        usage_entry = {
            'timestamp': timestamp,
            'query': query,
            'pattern': pattern,
            'result': {
                'confidence': result.total_confidence,
                'execution_time': result.execution_time,
                'steps_count': len(result.steps),
                'template_used': result.template_used,
                'complexity_score': result.complexity_score
            },
            'user_id': user_id,
            'session_id': session_id,
            'user_feedback': user_feedback
        }
        self.usage_history.append(usage_entry)
        
        # Track session data
        if session_id:
            self.session_data[session_id].append(usage_entry)
    
    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query"""
        # Simple pattern extraction - can be enhanced with NLP
        patterns = {
            'definition': ['what is', 'define', 'meaning of', 'explain'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'contrast'],
            'explanation': ['how does', 'why', 'explain', 'describe'],
            'analysis': ['analyze', 'evaluate', 'assess', 'examine'],
            'calculation': ['calculate', 'compute', 'solve', 'find'],
            'prediction': ['predict', 'forecast', 'estimate', 'project'],
            'recommendation': ['recommend', 'suggest', 'advise', 'propose'],
            'troubleshooting': ['fix', 'debug', 'error', 'problem', 'issue']
        }
        
        query_lower = query.lower()
        
        for pattern_type, indicators in patterns.items():
            if any(ind in query_lower for ind in indicators):
                return pattern_type
        
        return 'general'
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Generate insights for system optimization"""
        insights = {
            'popular_patterns': self._get_popular_patterns(),
            'problem_patterns': self._get_problem_patterns(),
            'time_based_insights': self._get_time_insights(),
            'template_effectiveness': self._analyze_template_effectiveness(),
            'session_analysis': self._analyze_sessions(),
            'performance_trends': self._analyze_performance_trends(),
            'recommendations': []
        }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations(insights)
        
        return insights
    
    def _get_popular_patterns(self) -> Any:
        """Identify most common query patterns"""
        sorted_patterns = sorted(
            self.query_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [
            {
                'pattern': pattern,
                'count': data['count'],
                'avg_complexity': data['avg_complexity'],
                'avg_confidence': data['avg_confidence'],
                'avg_time': data['avg_time']
            }
            for pattern, data in sorted_patterns[:10]
        ]
    
    def _get_problem_patterns(self) -> Any:
        """Identify patterns with low confidence or high errors"""
        problem_patterns = []
        
        for pattern, data in self.query_patterns.items():
            if data['count'] < 5:
                continue  # Not enough data
            
            issues = []
            
            if data['avg_confidence'] < 0.6:
                issues.append(f"Low confidence ({data['avg_confidence']:.2f})")
            
            if data['error_count'] > data['count'] * 0.1:  # More than 10% errors
                issues.append(f"High error rate ({data['error_count']}/{data['count']})")
            
            if data['avg_time'] > 2.0:  # Very slow
                issues.append(f"Slow execution ({data['avg_time']:.2f}s)")
            
            if issues:
                problem_patterns.append({
                    'pattern': pattern,
                    'count': data['count'],
                    'issues': issues,
                    'avg_confidence': data['avg_confidence'],
                    'avg_time': data['avg_time']
                })
        
        return sorted(problem_patterns, key=lambda x: len(x['issues']), reverse=True)
    
    def _get_time_insights(self) -> Any:
        """Analyze time-based usage patterns"""
        if not self.usage_history:
            return {'message': 'No usage data available'}
        
        # Analyze peak hours
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for entry in self.usage_history:
            dt = datetime.fromtimestamp(entry['timestamp'])
            hour_counts[dt.hour] += 1
            day_counts[dt.weekday()] += 1
        
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
        peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None
        
        return {
            'peak_hours': [hour for hour, count in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]],
            'peak_day': peak_day,
            'hour_distribution': dict(hour_counts),
            'day_distribution': dict(day_counts),
            'total_queries': len(self.usage_history)
        }
    
    def _analyze_template_effectiveness(self) -> Any:
        """Analyze template effectiveness"""
        template_stats = defaultdict(lambda: {
            'usage_count': 0,
            'avg_confidence': 0,
            'avg_time': 0,
            'user_satisfaction': []
        })
        
        for entry in self.usage_history:
            template = entry['result']['template_used']
            if template:
                stats = template_stats[template]
                stats['usage_count'] += 1
                
                # Update running averages
                n = stats['usage_count']
                stats['avg_confidence'] = (
                    (stats['avg_confidence'] * (n - 1) + entry['result']['confidence']) / n
                )
                stats['avg_time'] = (
                    (stats['avg_time'] * (n - 1) + entry['result']['execution_time']) / n
                )
                
                if entry['user_feedback'] is not None:
                    stats['user_satisfaction'].append(entry['user_feedback'])
        
        # Calculate effectiveness scores
        effectiveness = []
        for template, stats in template_stats.items():
            if stats['usage_count'] < 3:
                continue
            
            # Simple effectiveness score
            confidence_score = stats['avg_confidence']
            speed_score = 1.0 / max(stats['avg_time'], 0.1)
            satisfaction_score = (
                sum(stats['user_satisfaction']) / len(stats['user_satisfaction'])
                if stats['user_satisfaction'] else 0.5
            )
            
            effectiveness_score = (confidence_score * 0.4 + speed_score * 0.3 + satisfaction_score * 0.3)
            
            effectiveness.append({
                'template': template,
                'usage_count': stats['usage_count'],
                'avg_confidence': stats['avg_confidence'],
                'avg_time': stats['avg_time'],
                'avg_satisfaction': satisfaction_score,
                'effectiveness_score': effectiveness_score
            })
        
        return sorted(effectiveness, key=lambda x: x['effectiveness_score'], reverse=True)
    
    def _analyze_sessions(self) -> Any:
        """Analyze user session patterns"""
        session_analysis = []
        
        for session_id, entries in self.session_data.items():
            if len(entries) < 2:
                continue
            
            # Sort entries by timestamp
            entries.sort(key=lambda x: x['timestamp'])
            
            session_duration = entries[-1]['timestamp'] - entries[0]['timestamp']
            avg_confidence = sum(e['result']['confidence'] for e in entries) / len(entries)
            avg_time = sum(e['result']['execution_time'] for e in entries) / len(entries)
            
            session_analysis.append({
                'session_id': session_id,
                'query_count': len(entries),
                'duration': session_duration,
                'avg_confidence': avg_confidence,
                'avg_time': avg_time,
                'patterns_used': list(set(e['pattern'] for e in entries))
            })
        
        return session_analysis
    
    def _analyze_performance_trends(self) -> Any:
        """Analyze performance trends over time"""
        if len(self.usage_history) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Group by time periods (e.g., hourly)
        hourly_stats = defaultdict(lambda: {
            'count': 0,
            'total_confidence': 0,
            'total_time': 0
        })
        
        for entry in self.usage_history:
            dt = datetime.fromtimestamp(entry['timestamp'])
            hour_key = dt.replace(minute=0, second=0, microsecond=0)
            
            stats = hourly_stats[hour_key]
            stats['count'] += 1
            stats['total_confidence'] += entry['result']['confidence']
            stats['total_time'] += entry['result']['execution_time']
        
        # Calculate trends
        sorted_hours = sorted(hourly_stats.items())
        
        if len(sorted_hours) < 2:
            return {'message': 'Insufficient time data for trend analysis'}
        
        # Simple trend calculation
        first_hour = sorted_hours[0]
        last_hour = sorted_hours[-1]
        
        first_avg_confidence = first_hour[1]['total_confidence'] / first_hour[1]['count']
        last_avg_confidence = last_hour[1]['total_confidence'] / last_hour[1]['count']
        
        first_avg_time = first_hour[1]['total_time'] / first_hour[1]['count']
        last_avg_time = last_hour[1]['total_time'] / last_hour[1]['count']
        
        confidence_trend = (last_avg_confidence - first_avg_confidence) / first_avg_confidence * 100
        time_trend = (last_avg_time - first_avg_time) / first_avg_time * 100
        
        return {
            'confidence_trend': confidence_trend,
            'time_trend': time_trend,
            'total_periods': len(sorted_hours),
            'trend_direction': 'improving' if confidence_trend > 0 else 'declining'
        }
    
    def _generate_recommendations(self, insights) -> Any:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check for popular patterns that could benefit from specialized templates
        for pattern in insights['popular_patterns']:
            if pattern['count'] > 100:
                recommendations.append(
                    f"Create specialized template for '{pattern['pattern']}' "
                    f"queries (used {pattern['count']} times)"
                )
        
        # Check for problem patterns
        for pattern in insights['problem_patterns']:
            recommendations.append(
                f"Review and improve handling of '{pattern['pattern']}' "
                f"queries: {', '.join(pattern['issues'])}"
            )
        
        # Time-based recommendations
        time_insights = insights['time_based_insights']
        if 'peak_hours' in time_insights and time_insights['peak_hours']:
            peak_hours = time_insights['peak_hours']
            recommendations.append(
                f"Consider pre-warming cache before peak hours: {peak_hours}"
            )
        
        # Template effectiveness recommendations
        template_effectiveness = insights['template_effectiveness']
        if template_effectiveness:
            worst_template = template_effectiveness[-1]
            if worst_template['effectiveness_score'] < 0.5:
                recommendations.append(
                    f"Review and optimize template '{worst_template['template']}' "
                    f"(effectiveness: {worst_template['effectiveness_score']:.2f})"
                )
        
        # Performance trend recommendations
        performance_trends = insights['performance_trends']
        if 'trend_direction' in performance_trends:
            if performance_trends['trend_direction'] == 'declining':
                recommendations.append(
                    "Performance is declining over time. Consider system optimization."
                )
        
        return recommendations
    
    def save_analytics_report(self, filename: Optional[str] = None) -> bool:
        """Save analytics report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics/usage_analysis_{timestamp}.json"
        
        insights = self.get_optimization_insights()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'insights': insights,
            'summary': {
                'total_queries': len(self.usage_history),
                'unique_patterns': len(self.query_patterns),
                'total_sessions': len(self.session_data),
                'data_period': {
                    'start': min(e['timestamp'] for e in self.usage_history) if self.usage_history else None,
                    'end': max(e['timestamp'] for e in self.usage_history) if self.usage_history else None
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Analytics report saved to {}", extra={"filename": filename})
        return filename
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time usage metrics"""
        if not self.usage_history:
            return {'message': 'No usage data available'}
        
        # Last hour metrics
        one_hour_ago = time.time() - 3600
        recent_queries = [
            entry for entry in self.usage_history
            if entry['timestamp'] > one_hour_ago
        ]
        
        if not recent_queries:
            return {'message': 'No recent usage data'}
        
        recent_confidence = sum(e['result']['confidence'] for e in recent_queries) / len(recent_queries)
        recent_time = sum(e['result']['execution_time'] for e in recent_queries) / len(recent_queries)
        
        return {
            'queries_last_hour': len(recent_queries),
            'avg_confidence_last_hour': recent_confidence,
            'avg_time_last_hour': recent_time,
            'most_common_pattern': max(
                (e['pattern'] for e in recent_queries),
                key=lambda p: sum(1 for e in recent_queries if e['pattern'] == p)
            ) if recent_queries else None
        }


# Example usage
def run_usage_analysis() -> Any:
    """Example of running usage analysis"""
    logger.info("Running usage pattern analysis", extra={
        "operation": "usage_analysis",
        "phase": "start"
    })
    
    # Create analyzer
    analyzer = UsagePatternAnalyzer()
    
    # Simulate some usage data
    sample_queries = [
        ("What is machine learning?", 0.8, 1.2),
        ("Compare Python and Java", 0.7, 1.5),
        ("How does neural networks work?", 0.9, 2.1),
        ("What is 2+2?", 0.95, 0.3),
        ("Analyze the impact of AI", 0.6, 3.2),
        ("Define blockchain", 0.8, 1.1),
        ("Explain recursion", 0.7, 1.8),
        ("What causes climate change?", 0.8, 2.5)
    ]
    
    # Add sample data
    for i, (query, confidence, exec_time) in enumerate(sample_queries):
        # Create mock result
        mock_result = type('MockResult', (), {
            'total_confidence': confidence,
            'execution_time': exec_time,
            'steps': [type('MockStep', (), {'thought': 'Sample step'})() for _ in range(3)],
            'template_used': 'default',
            'complexity_score': 0.5 + (i * 0.1)
        })()
        
        analyzer.analyze_query(
            query=query,
            result=mock_result,
            timestamp=time.time() - (i * 3600),  # Spread over hours
            user_id=f"user_{i % 3}",
            session_id=f"session_{i // 2}",
            user_feedback=confidence + 0.1  # Slightly higher than confidence
        )
    
    # Get insights
    insights = analyzer.get_optimization_insights()
    
    logger.info("Usage analysis completed", extra={
        "operation": "usage_analysis",
        "total_queries": len(analyzer.usage_history),
        "unique_patterns": len(analyzer.query_patterns),
        "popular_patterns_count": len(insights['popular_patterns']),
        "problem_patterns_count": len(insights['problem_patterns']),
        "recommendations_count": len(insights['recommendations'])
    })
    
    # Save report
    report_file = analyzer.save_analytics_report()
    logger.info("Analytics report saved", extra={
        "operation": "save_report",
        "filename": report_file
    })
    
    return insights


if __name__ == "__main__":
    # Run usage analysis
    run_usage_analysis() 