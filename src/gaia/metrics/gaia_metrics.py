from collections import defaultdict
from typing import Dict, List, Any
import time
import json

class GAIAMetrics:
    """GAIA-specific performance metrics and analytics"""
    
    def __init__(self):
        self.question_type_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        self.tool_effectiveness = defaultdict(lambda: {"successful": 0, "total": 0})
        self.avg_time_by_type = defaultdict(list)
        self.error_analysis = defaultdict(int)
        self.confidence_calibration = defaultdict(list)
        
    def log_result(self, question_type: str, correct: bool, time_taken: float, 
                  tools_used: List[str], confidence: float, error_type: str = None):
        """Track GAIA-specific performance metrics"""
        # Question type accuracy
        self.question_type_accuracy[question_type]["total"] += 1
        if correct:
            self.question_type_accuracy[question_type]["correct"] += 1
        
        # Time tracking
        self.avg_time_by_type[question_type].append(time_taken)
        
        # Tool effectiveness
        for tool in tools_used:
            self.tool_effectiveness[tool]["total"] += 1
            if correct:
                self.tool_effectiveness[tool]["successful"] += 1
        
        # Confidence calibration
        self.confidence_calibration[question_type].append({
            "confidence": confidence,
            "correct": correct
        })
        
        # Error tracking
        if error_type:
            self.error_analysis[error_type] += 1
    
    def get_accuracy_by_type(self) -> Dict[str, float]:
        """Get accuracy breakdown by question type"""
        return {
            qtype: stats["correct"] / max(stats["total"], 1)
            for qtype, stats in self.question_type_accuracy.items()
        }
    
    def get_tool_effectiveness(self) -> Dict[str, float]:
        """Get tool success rates"""
        return {
            tool: stats["successful"] / max(stats["total"], 1)
            for tool, stats in self.tool_effectiveness.items()
        }
    
    def get_avg_time_by_type(self) -> Dict[str, float]:
        """Get average response time by question type"""
        return {
            qtype: sum(times) / len(times) if times else 0
            for qtype, times in self.avg_time_by_type.items()
        }
    
    def get_confidence_calibration(self) -> Dict[str, Dict[str, float]]:
        """Get confidence calibration metrics"""
        calibration = {}
        for qtype, data in self.confidence_calibration.items():
            if data:
                avg_confidence = sum(d["confidence"] for d in data) / len(data)
                accuracy = sum(1 for d in data if d["correct"]) / len(data)
                calibration[qtype] = {
                    "avg_confidence": avg_confidence,
                    "accuracy": accuracy,
                    "calibration_error": abs(avg_confidence - accuracy)
                }
        return calibration
    
    def get_error_analysis(self) -> Dict[str, int]:
        """Get error frequency analysis"""
        return dict(self.error_analysis)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_questions = sum(stats["total"] for stats in self.question_type_accuracy.values())
        total_correct = sum(stats["correct"] for stats in self.question_type_accuracy.values())
        
        return {
            "overall_accuracy": total_correct / max(total_questions, 1),
            "total_questions": total_questions,
            "question_types_tested": len(self.question_type_accuracy),
            "tools_used": len(self.tool_effectiveness),
            "avg_response_time": sum(sum(times) for times in self.avg_time_by_type.values()) / 
                               max(sum(len(times) for times in self.avg_time_by_type.values()), 1),
            "accuracy_by_type": self.get_accuracy_by_type(),
            "tool_effectiveness": self.get_tool_effectiveness(),
            "confidence_calibration": self.get_confidence_calibration(),
            "error_analysis": self.get_error_analysis()
        }
    
    def export_metrics(self, filename: str = "gaia_metrics.json"):
        """Export metrics to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.get_overall_stats(), f, indent=2)
    
    def print_summary(self):
        """Print human-readable metrics summary"""
        stats = self.get_overall_stats()
        
        print("=" * 60)
        print("GAIA PERFORMANCE METRICS SUMMARY")
        print("=" * 60)
        print(f"Overall Accuracy: {stats['overall_accuracy']:.1%}")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Question Types Tested: {stats['question_types_tested']}")
        print(f"Average Response Time: {stats['avg_response_time']:.2f}s")
        
        print("\n📊 Accuracy by Question Type:")
        for qtype, accuracy in stats['accuracy_by_type'].items():
            print(f"   {qtype}: {accuracy:.1%}")
        
        print("\n🛠️  Tool Effectiveness:")
        for tool, effectiveness in stats['tool_effectiveness'].items():
            print(f"   {tool}: {effectiveness:.1%}")
        
        print("\n📈 Confidence Calibration:")
        for qtype, cal in stats['confidence_calibration'].items():
            print(f"   {qtype}: {cal['accuracy']:.1%} accuracy, {cal['avg_confidence']:.1%} confidence")
        
        if stats['error_analysis']:
            print("\n🚨 Error Analysis:")
            for error, count in stats['error_analysis'].items():
                print(f"   {error}: {count} occurrences")

# Global metrics instance
gaia_metrics = GAIAMetrics() 