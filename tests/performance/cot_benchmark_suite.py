"""
Chain of Thought Performance Benchmarks and Testing Suite
Comprehensive benchmarking for the Optimized CoT System
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
import psutil
import tracemalloc
import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import the CoT system
try:
    from core.optimized_chain_of_thought import (
        OptimizedChainOfThought, 
        ReasoningType, 
        ReasoningPath,
        ReasoningStep
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.core.optimized_chain_of_thought import (
        OptimizedChainOfThought, 
        ReasoningType, 
        ReasoningPath,
        ReasoningStep
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# Benchmark Data Structures
# =============================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    query: str
    complexity: float
    execution_time: float
    confidence: float
    steps_count: int
    cache_hit: bool
    memory_usage: float
    cpu_usage: float
    template_used: str
    paths_explored: int
    reasoning_types: List[str]
    final_answer: str

@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    name: str
    results: List[BenchmarkResult]
    timestamp: float
    config: Dict[str, Any]

# =============================
# Test Query Sets
# =============================

class QueryDataset:
    """Dataset of queries for benchmarking"""
    
    @staticmethod
    def get_simple_queries() -> List[str]:
        """Simple queries (complexity < 0.3)"""
        return [
            "What is 2 + 2?",
            "What color is the sky?",
            "Define democracy.",
            "What is the capital of France?",
            "How many days in a week?",
            "What is water made of?",
            "When did World War II end?",
            "What is the speed of light?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet?"
        ]
    
    @staticmethod
    def get_medium_queries() -> List[str]:
        """Medium complexity queries (0.3 < complexity < 0.7)"""
        return [
            "Explain the process of photosynthesis.",
            "Compare renewable and non-renewable energy sources.",
            "What are the main causes of climate change?",
            "Describe the water cycle and its importance.",
            "How does the stock market work?",
            "Explain the difference between virus and bacteria.",
            "What factors contributed to the Industrial Revolution?",
            "How do vaccines work to prevent diseases?",
            "Describe the structure of DNA.",
            "What are the pros and cons of social media?"
        ]
    
    @staticmethod
    def get_complex_queries() -> List[str]:
        """Complex queries (complexity > 0.7)"""
        return [
            "Analyze the potential long-term socioeconomic impacts of artificial intelligence on global employment patterns, considering both displacement effects and new job creation.",
            "Compare and contrast the philosophical foundations of utilitarianism and deontological ethics, providing examples of how each would approach modern ethical dilemmas.",
            "Evaluate the effectiveness of different monetary policy tools in combating inflation while maintaining economic growth, considering recent global economic trends.",
            "Discuss the role of epigenetics in evolution and heredity, explaining how environmental factors can influence gene expression across generations.",
            "Analyze the geopolitical implications of renewable energy transition on international relations and global power dynamics.",
            "Examine the intersection of quantum mechanics and consciousness, discussing various interpretations and their philosophical implications.",
            "Evaluate the challenges and opportunities of establishing a sustainable human colony on Mars, considering technological, biological, and social factors.",
            "Analyze the impact of social media algorithms on democratic processes and public discourse, proposing potential regulatory frameworks.",
            "Discuss the ethical implications of gene editing technologies like CRISPR, considering medical benefits, risks, and societal concerns.",
            "Examine the role of cognitive biases in financial decision-making and their impact on market efficiency."
        ]
    
    @staticmethod
    def get_mathematical_queries() -> List[str]:
        """Mathematical reasoning queries"""
        return [
            "Solve for x: 2x + 5 = 13",
            "Calculate the derivative of f(x) = 3x^2 + 2x - 1",
            "Find the area of a circle with radius 7",
            "Solve the quadratic equation: x^2 - 5x + 6 = 0",
            "Calculate the compound interest on $1000 at 5% for 3 years",
            "Find the integral of sin(x) from 0 to π",
            "Determine if the series Σ(1/n^2) converges",
            "Calculate the probability of getting exactly 3 heads in 5 coin flips",
            "Find the eigenvalues of the matrix [[2, 1], [1, 2]]",
            "Solve the differential equation: dy/dx = 2y"
        ]
    
    @staticmethod
    def get_ai_agent_queries() -> List[str]:
        """AI Agent specific queries"""
        return [
            "How does the FSM agent handle recursive reasoning?",
            "Compare the performance of Chain of Thought vs FSM reasoning approaches",
            "What are the advantages of hybrid architecture in AI agents?",
            "How does the metacognitive layer improve reasoning quality?",
            "Analyze the trade-offs between cache size and memory usage in CoT systems",
            "What makes the OptimizedChainOfThought system different from basic CoT?",
            "How does the complexity analyzer determine reasoning depth?",
            "Explain the multi-path exploration strategy in reasoning systems",
            "What role does template selection play in reasoning quality?",
            "How can we optimize reasoning performance for real-time applications?"
        ]

# =============================
# Benchmarking Engine
# =============================

class CoTBenchmark:
    """Benchmarking engine for Chain of Thought system"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.cot_configs = [
            {'max_paths': 1, 'cache_size': 100},
            {'max_paths': 3, 'cache_size': 500},
            {'max_paths': 5, 'cache_size': 1000},
            {'max_paths': 7, 'cache_size': 2000}
        ]
        
    async def run_benchmark(self, cot_system, queries: List[str], 
                          name: str = "default") -> BenchmarkSuite:
        """Run benchmark on a set of queries"""
        results = []
        
        logger.info("Running benchmark: {}", extra={"name": name})
        logger.info("Number of queries: {}", extra={"len_queries_": len(queries)})
        print("-" * 50)
        
        for i, query in enumerate(queries):
            logger.info("Processing query {}/{}: {}...", extra={"i_1": i+1, "len_queries_": len(queries), "query_": query[})
            
            # Measure performance
            result = await self._benchmark_single_query(cot_system, query)
            results.append(result)
            
        suite = BenchmarkSuite(
            name=name,
            results=results,
            timestamp=time.time(),
            config=cot_system.config
        )
        
        return suite
    
    async def _benchmark_single_query(self, cot_system, query: str) -> BenchmarkResult:
        """Benchmark a single query"""
        # Start monitoring
        tracemalloc.start()
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        
        # Check if query is in cache
        cache_result = cot_system.reasoning_cache.get(query)
        cache_hit = cache_result is not None
        
        # Execute reasoning
        start_time = time.time()
        result = await cot_system.reason(query)
        execution_time = time.time() - start_time
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024
        
        # Get CPU usage
        cpu_after = process.cpu_percent()
        cpu_usage = cpu_after - cpu_before
        
        # Get complexity
        complexity, _ = cot_system.complexity_analyzer.analyze(query)
        
        # Extract reasoning types
        reasoning_types = [step.reasoning_type.name for step in result.steps]
        
        return BenchmarkResult(
            query=query,
            complexity=complexity,
            execution_time=execution_time,
            confidence=result.total_confidence,
            steps_count=len(result.steps),
            cache_hit=cache_hit,
            memory_usage=memory_mb,
            cpu_usage=cpu_usage,
            template_used=result.template_used,
            paths_explored=len(result.steps),  # Simplified
            reasoning_types=reasoning_types,
            final_answer=result.final_answer or "No answer generated"
        )
    
    async def run_comparative_benchmark(self) -> Dict[str, BenchmarkSuite]:
        """Run benchmarks with different configurations"""
        all_queries = (
            QueryDataset.get_simple_queries()[:5] +
            QueryDataset.get_medium_queries()[:5] +
            QueryDataset.get_complex_queries()[:5]
        )
        
        results = {}
        
        for config in self.cot_configs:
            config_name = f"paths_{config['max_paths']}_cache_{config['cache_size']}"
            
            # Create CoT system with config
            cot = OptimizedChainOfThought("benchmark_cot", config)
            
            # Run benchmark
            suite = await self.run_benchmark(cot, all_queries, config_name)
            results[config_name] = suite
            
        return results
    
    def analyze_results(self, suite: BenchmarkSuite) -> Dict[str, Any]:
        """Analyze benchmark results"""
        results = suite.results
        
        # Basic statistics
        execution_times = [r.execution_time for r in results]
        confidences = [r.confidence for r in results]
        steps_counts = [r.steps_count for r in results]
        
        # Group by complexity
        simple_results = [r for r in results if r.complexity < 0.3]
        medium_results = [r for r in results if 0.3 <= r.complexity < 0.7]
        complex_results = [r for r in results if r.complexity >= 0.7]
        
        analysis = {
            'overall': {
                'total_queries': len(results),
                'avg_execution_time': statistics.mean(execution_times),
                'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'avg_confidence': statistics.mean(confidences),
                'avg_steps': statistics.mean(steps_counts),
                'cache_hit_rate': sum(1 for r in results if r.cache_hit) / len(results),
                'avg_memory_mb': statistics.mean([r.memory_usage for r in results]),
                'avg_cpu_percent': statistics.mean([r.cpu_usage for r in results])
            },
            'by_complexity': {
                'simple': self._analyze_group(simple_results),
                'medium': self._analyze_group(medium_results),
                'complex': self._analyze_group(complex_results)
            },
            'by_template': self._analyze_by_template(results),
            'by_reasoning_type': self._analyze_by_reasoning_type(results)
        }
        
        return analysis
    
    def _analyze_group(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Analyze a group of results"""
        if not results:
            return {}
            
        return {
            'count': len(results),
            'avg_execution_time': statistics.mean([r.execution_time for r in results]),
            'avg_confidence': statistics.mean([r.confidence for r in results]),
            'avg_steps': statistics.mean([r.steps_count for r in results])
        }
    
    def _analyze_by_template(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """Analyze results by template used"""
        template_results = {}
        
        for result in results:
            template = result.template_used
            if template not in template_results:
                template_results[template] = []
            template_results[template].append(result)
        
        analysis = {}
        for template, template_group in template_results.items():
            analysis[template] = self._analyze_group(template_group)
            
        return analysis
    
    def _analyze_by_reasoning_type(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """Analyze results by reasoning type"""
        type_results = {}
        
        for result in results:
            for reasoning_type in result.reasoning_types:
                if reasoning_type not in type_results:
                    type_results[reasoning_type] = []
                type_results[reasoning_type].append(result)
        
        analysis = {}
        for reasoning_type, type_group in type_results.items():
            analysis[reasoning_type] = self._analyze_group(type_group)
            
        return analysis

# =============================
# Performance Visualization
# =============================

class BenchmarkVisualizer:
    """Visualize benchmark results"""
    
    @staticmethod
    def plot_execution_time_by_complexity(suite: BenchmarkSuite):
        """Plot execution time vs complexity"""
        complexities = [r.complexity for r in suite.results]
        execution_times = [r.execution_time for r in suite.results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(complexities, execution_times, alpha=0.6)
        plt.xlabel('Query Complexity')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time vs Query Complexity')
        
        # Add trend line
        z = np.polyfit(complexities, execution_times, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 1, 100)
        plt.plot(x_trend, p(x_trend), 'r--', alpha=0.8, label='Trend')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confidence_distribution(suite: BenchmarkSuite):
        """Plot confidence score distribution"""
        confidences = [r.confidence for r in suite.results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Confidence Scores')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_template_performance(analysis: Dict[str, Any]):
        """Plot performance by template"""
        template_data = analysis['by_template']
        
        if not template_data:
            logger.info("No template data available for visualization")
            return
            
        templates = list(template_data.keys())
        avg_times = [template_data[t]['avg_execution_time'] for t in templates]
        avg_confidence = [template_data[t]['avg_confidence'] for t in templates]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Execution time
        ax1.bar(templates, avg_times, alpha=0.7)
        ax1.set_xlabel('Template')
        ax1.set_ylabel('Average Execution Time (s)')
        ax1.set_title('Average Execution Time by Template')
        ax1.tick_params(axis='x', rotation=45)
        
        # Confidence
        ax2.bar(templates, avg_confidence, alpha=0.7, color='green')
        ax2.set_xlabel('Template')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Confidence by Template')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparative_results(comparative_results: Dict[str, BenchmarkSuite]):
        """Plot comparative benchmark results"""
        configs = []
        avg_times = []
        avg_confidences = []
        
        for config_name, suite in comparative_results.items():
            analysis = CoTBenchmark().analyze_results(suite)
            configs.append(config_name)
            avg_times.append(analysis['overall']['avg_execution_time'])
            avg_confidences.append(analysis['overall']['avg_confidence'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Execution time comparison
        ax1.plot(configs, avg_times, 'o-', markersize=8)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Average Execution Time (s)')
        ax1.set_title('Execution Time by Configuration')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Confidence comparison
        ax2.plot(configs, avg_confidences, 'o-', markersize=8, color='green')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Confidence by Configuration')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_memory_usage_analysis(suite: BenchmarkSuite):
        """Plot memory usage analysis"""
        memory_usage = [r.memory_usage for r in suite.results]
        complexities = [r.complexity for r in suite.results]
        
        plt.figure(figsize=(12, 8))
        
        # Memory vs complexity
        plt.subplot(2, 2, 1)
        plt.scatter(complexities, memory_usage, alpha=0.6)
        plt.xlabel('Query Complexity')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Complexity')
        plt.grid(True, alpha=0.3)
        
        # Memory distribution
        plt.subplot(2, 2, 2)
        plt.hist(memory_usage, bins=15, alpha=0.7, edgecolor='black')
        plt.xlabel('Memory Usage (MB)')
        plt.ylabel('Frequency')
        plt.title('Memory Usage Distribution')
        plt.axvline(np.mean(memory_usage), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(memory_usage):.2f} MB')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Memory vs execution time
        plt.subplot(2, 2, 3)
        execution_times = [r.execution_time for r in suite.results]
        plt.scatter(execution_times, memory_usage, alpha=0.6)
        plt.xlabel('Execution Time (s)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Execution Time')
        plt.grid(True, alpha=0.3)
        
        # Memory vs steps
        plt.subplot(2, 2, 4)
        steps_counts = [r.steps_count for r in suite.results]
        plt.scatter(steps_counts, memory_usage, alpha=0.6)
        plt.xlabel('Number of Steps')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Steps Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =============================
# Performance Testing Suite
# =============================

async def run_performance_tests():
    """Run comprehensive performance tests"""
    logger.info("=== Chain of Thought Performance Testing Suite ===\n")
    
    # Initialize benchmark engine
    benchmark = CoTBenchmark()
    
    # Test 1: Basic Performance
    logger.info("Test 1: Basic Performance")
    print("-" * 50)
    
    cot = OptimizedChainOfThought("test_cot", {'max_paths': 3})
    
    # Get test queries
    test_queries = (
        QueryDataset.get_simple_queries()[:3] +
        QueryDataset.get_medium_queries()[:3] +
        QueryDataset.get_complex_queries()[:2]
    )
    
    suite = await benchmark.run_benchmark(cot, test_queries, "basic_performance")
    analysis = benchmark.analyze_results(suite)
    
    logger.info("\nBasic Performance Results:")
    logger.info("Average execution time: {}s", extra={"analysis__overall___avg_execution_time_": analysis['overall']['avg_execution_time']})
    logger.info("Average confidence: {}", extra={"analysis__overall___avg_confidence_": analysis['overall']['avg_confidence']})
    logger.info("Cache hit rate: {}", extra={"analysis__overall___cache_hit_rate_": analysis['overall']['cache_hit_rate']})
    logger.info("Average memory usage: {} MB", extra={"analysis__overall___avg_memory_mb_": analysis['overall']['avg_memory_mb']})
    
    # Test 2: Stress Test
    logger.info("\n\nTest 2: Stress Test (50 queries)")
    print("-" * 50)
    
    # Generate many queries
    stress_queries = []
    for i in range(50):
        complexity = np.random.random()
        if complexity < 0.3:
            stress_queries.append(f"Simple query {i}: What is {i} + {i+1}?")
        elif complexity < 0.7:
            stress_queries.append(f"Medium query {i}: Explain concept {i} in detail.")
        else:
            stress_queries.append(
                f"Complex query {i}: Analyze the multifaceted implications of topic {i} "
                f"considering various perspectives and long-term consequences."
            )
    
    stress_suite = await benchmark.run_benchmark(cot, stress_queries, "stress_test")
    stress_analysis = benchmark.analyze_results(stress_suite)
    
    logger.info("\nStress Test Results:")
    logger.info("Total queries processed: {}", extra={"stress_analysis__overall___total_queries_": stress_analysis['overall']['total_queries']})
    logger.info("Average execution time: {}s", extra={"stress_analysis__overall___avg_execution_time_": stress_analysis['overall']['avg_execution_time']})
    logger.info("Peak memory usage: {} MB", extra={"max_r_memory_usage_for_r_in_stress_suite_results_": max(r.memory_usage for r in stress_suite.results)})
    
    # Test 3: Cache Performance
    logger.info("\n\nTest 3: Cache Performance")
    print("-" * 50)
    
    # Test cache effectiveness
    cache_test_queries = QueryDataset.get_medium_queries()[:5]
    
    # First run - no cache
    logger.info("First run (cold cache)...")
    first_run = await benchmark.run_benchmark(cot, cache_test_queries, "cache_cold")
    
    # Second run - with cache
    logger.info("Second run (warm cache)...")
    second_run = await benchmark.run_benchmark(cot, cache_test_queries, "cache_warm")
    
    first_times = [r.execution_time for r in first_run.results]
    second_times = [r.execution_time for r in second_run.results]
    
    speedup = statistics.mean(first_times) / statistics.mean(second_times)
    logger.info("\nCache speedup: {}x", extra={"speedup": speedup})
    logger.info("First run avg: {}s", extra={"statistics_mean_first_times_": statistics.mean(first_times)})
    logger.info("Second run avg: {}s", extra={"statistics_mean_second_times_": statistics.mean(second_times)})
    
    # Test 4: Comparative Configuration Test
    logger.info("\n\nTest 4: Configuration Comparison")
    print("-" * 50)
    
    comparative_results = await benchmark.run_comparative_benchmark()
    
    logger.info("\nConfiguration Comparison Results:")
    for config_name, suite in comparative_results.items():
        analysis = benchmark.analyze_results(suite)
        logger.info("\n{}:", extra={"config_name": config_name})
        logger.info("  Avg execution time: {}s", extra={"analysis__overall___avg_execution_time_": analysis['overall']['avg_execution_time']})
        logger.info("  Avg confidence: {}", extra={"analysis__overall___avg_confidence_": analysis['overall']['avg_confidence']})
        logger.info("  Avg memory: {} MB", extra={"analysis__overall___avg_memory_mb_": analysis['overall']['avg_memory_mb']})
    
    # Test 5: Domain-Specific Performance
    logger.info("\n\nTest 5: Domain-Specific Performance")
    print("-" * 50)
    
    # Test different query domains
    domain_queries = {
        'mathematical': QueryDataset.get_mathematical_queries()[:5],
        'ai_agent': QueryDataset.get_ai_agent_queries()[:5]
    }
    
    domain_results = {}
    for domain, queries in domain_queries.items():
        logger.info("Testing {} queries...", extra={"domain": domain})
        domain_suite = await benchmark.run_benchmark(cot, queries, f"{domain}_domain")
        domain_analysis = benchmark.analyze_results(domain_suite)
        domain_results[domain] = domain_analysis
        
        logger.info("  {} domain:", extra={"domain_capitalize__": domain.capitalize()})
        logger.info("    Avg execution time: {}s", extra={"domain_analysis__overall___avg_execution_time_": domain_analysis['overall']['avg_execution_time']})
        logger.info("    Avg confidence: {}", extra={"domain_analysis__overall___avg_confidence_": domain_analysis['overall']['avg_confidence']})
    
    # Visualize results
    logger.info("\n\nGenerating visualizations...")
    visualizer = BenchmarkVisualizer()
    
    # Plot execution time vs complexity
    visualizer.plot_execution_time_by_complexity(suite)
    
    # Plot confidence distribution
    visualizer.plot_confidence_distribution(suite)
    
    # Plot template performance
    visualizer.plot_template_performance(analysis)
    
    # Plot comparative results
    visualizer.plot_comparative_results(comparative_results)
    
    # Plot memory usage analysis
    visualizer.plot_memory_usage_analysis(suite)
    
    return {
        'basic_performance': analysis,
        'stress_test': stress_analysis,
        'cache_performance': {
            'speedup': speedup,
            'first_run_avg': statistics.mean(first_times),
            'second_run_avg': statistics.mean(second_times)
        },
        'comparative': comparative_results,
        'domain_specific': domain_results
    }

# =============================
# Main Execution
# =============================

if __name__ == "__main__":
    # Run performance tests
    results = asyncio.run(run_performance_tests())
    
    # Save results
    with open('cot_benchmark_results.json', 'w') as f:
        # Convert results to serializable format
        serializable_results = {
            'timestamp': time.time(),
            'basic_performance': results['basic_performance'],
            'stress_test': results['stress_test'],
            'cache_performance': results['cache_performance'],
            'domain_specific': results['domain_specific']
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info("\n\nBenchmark results saved to 'cot_benchmark_results.json'")
    logger.info("Performance testing completed successfully!") 