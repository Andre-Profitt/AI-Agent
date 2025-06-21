from agent import query
from benchmarks.cot_performance import analysis
from benchmarks.cot_performance import avg_confidence
from benchmarks.cot_performance import avg_duration
from benchmarks.cot_performance import avg_steps
from benchmarks.cot_performance import ax1
from benchmarks.cot_performance import ax2
from benchmarks.cot_performance import ax3
from benchmarks.cot_performance import ax4
from benchmarks.cot_performance import ax5
from benchmarks.cot_performance import ax6
from benchmarks.cot_performance import baseline
from benchmarks.cot_performance import benchmark_suite
from benchmarks.cot_performance import cache_rate
from benchmarks.cot_performance import complex_time
from benchmarks.cot_performance import complexity_results
from benchmarks.cot_performance import confidences
from benchmarks.cot_performance import cot_system
from benchmarks.cot_performance import current
from benchmarks.cot_performance import df
from benchmarks.cot_performance import duration
from benchmarks.cot_performance import end
from benchmarks.cot_performance import failed
from benchmarks.cot_performance import failed_count
from benchmarks.cot_performance import filename
from benchmarks.cot_performance import first
from benchmarks.cot_performance import growth_rate
from benchmarks.cot_performance import high_variance
from benchmarks.cot_performance import insights
from benchmarks.cot_performance import last
from benchmarks.cot_performance import load_results
from benchmarks.cot_performance import load_tester
from benchmarks.cot_performance import low_conf
from benchmarks.cot_performance import memory_profiler
from benchmarks.cot_performance import memory_results
from benchmarks.cot_performance import memory_usage
from benchmarks.cot_performance import performance_results
from benchmarks.cot_performance import qps
from benchmarks.cot_performance import recommendations
from benchmarks.cot_performance import results_data
from benchmarks.cot_performance import runs
from benchmarks.cot_performance import semaphore
from benchmarks.cot_performance import snapshot
from benchmarks.cot_performance import start
from benchmarks.cot_performance import subset
from benchmarks.cot_performance import successful
from benchmarks.cot_performance import template
from benchmarks.cot_performance import templates
from benchmarks.cot_performance import times
from benchmarks.cot_performance import timestamp
from benchmarks.cot_performance import top_stats
from benchmarks.cot_performance import topic1
from benchmarks.cot_performance import topic2
from benchmarks.cot_performance import topics
from benchmarks.cot_performance import total
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import end_time
from examples.parallel_execution_example import results
from tests.load_test import queries

from src.database.supabase_manager import total_duration
from src.gaia_components.production_vector_store import count
from src.meta_cognition import complexity

from src.core.optimized_chain_of_thought import OptimizedChainOfThought
# TODO: Fix undefined variables: analysis, avg_confidence, avg_duration, avg_steps, ax1, ax2, ax3, ax4, ax5, ax6, axes, baseline, benchmark_suite, cache_rate, complex_time, complexity, complexity_results, confidences, cot_system, count, current, datetime, df, duration, e, end, end_time, f, failed, failed_count, filename, first, growth_rate, high_variance, i, index, insights, json, last, load_results, load_tester, logging, low_conf, memory_results, memory_usage, metrics, num_concurrent, num_queries, os, performance_results, qps, queries, query, r, rec, recommendations, result, results, results_data, runs, semaphore, snapshot, start, start_time, stat, subset, successful, tasks, template, templates, time, times, timestamp, top_stats, topic1, topic2, topics, total, total_duration, total_queries
import memory_profiler

# TODO: Fix undefined variables: analysis, avg_confidence, avg_duration, avg_steps, ax1, ax2, ax3, ax4, ax5, ax6, axes, baseline, benchmark_suite, cache_rate, complex_time, complexity, complexity_results, confidences, cot_system, count, current, df, duration, e, end, end_time, f, failed, failed_count, filename, first, growth_rate, high_variance, i, index, insights, last, load_results, load_tester, low_conf, memory_profiler, memory_results, memory_usage, metrics, num_concurrent, num_queries, performance_results, plt, psutil, qps, queries, query, r, rec, recommendations, result, results, results_data, runs, self, semaphore, snapshot, start, start_time, stat, statistics, subset, successful, tasks, template, templates, times, timestamp, top_stats, topic1, topic2, topics, total, total_duration, total_queries, tracemalloc
"""
Comprehensive benchmarking for CoT system
Measures execution time, confidence, cache performance, and provides recommendations
"""

import time
import asyncio
import statistics

import matplotlib.pyplot as plt
import pandas as pd

import json
import os
from datetime import datetime

# Import the CoT system
from src.core.optimized_chain_of_thought import (
    OptimizedChainOfThought,
    ReasoningPath,
    ComplexityAnalyzer
)

import logging

logger = logging.getLogger(__name__)

class CoTBenchmarkSuite:
    """Comprehensive benchmarking for CoT system"""

    def __init__(self):
        self.results = []
        self.query_sets = {
            'simple': [
                "What is 2+2?",
                "Define machine learning",
                "What color is the sky?",
                "How do you make coffee?",
                "What is the capital of France?"
            ],
            'medium': [
                "Explain how neural networks work",
                "Compare Python and Java programming languages",
                "What causes climate change?",
                "How does blockchain technology work?",
                "Explain the concept of recursion"
            ],
            'complex': [
                "Analyze the socioeconomic impacts of AI on employment and propose policy recommendations",
                "Compare different approaches to solving the traveling salesman problem and their trade-offs",
                "Discuss the philosophical implications of consciousness in AI systems and their ethical considerations",
                "Evaluate the effectiveness of different machine learning algorithms for natural language processing",
                "Analyze the security implications of quantum computing on current cryptographic systems"
            ]
        }

        # Create benchmarks directory if it doesn't exist
        os.makedirs('benchmarks', exist_ok=True)

    async def run_benchmarks(self, cot_system):
        """Run comprehensive benchmarks"""
        logger.info("Running CoT Performance Benchmarks...")
        print("=" * 60)

        for complexity, queries in self.query_sets.items():
            logger.info("\n{} Queries:", extra={"complexity_upper__": complexity.upper()})
            print("-" * 40)

            complexity_results = await self._benchmark_query_set(
                cot_system, queries, complexity
            )
            self.results.extend(complexity_results)

        return self._analyze_results()

    async def _benchmark_query_set(self, cot_system, queries, complexity):
        """Benchmark a set of queries"""
        results = []

        for query in queries:
            logger.info("  Benchmarking: {}...", extra={"query_": query[:50]})

            # Warm up
            await cot_system.reason(query)

            # Actual benchmark (multiple runs)
            runs = 5
            times = []
            confidences = []

            for run in range(runs):
                start = time.perf_counter()
                result = await cot_system.reason(query)
                end = time.perf_counter()

                times.append(end - start)
                confidences.append(result.total_confidence)

                # Small delay between runs
                await asyncio.sleep(0.1)

            results.append({
                'query': query[:50] + '...' if len(query) > 50 else query,
                'complexity': complexity,
                'avg_time': statistics.mean(times),
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'min_time': min(times),
                'max_time': max(times),
                'avg_confidence': statistics.mean(confidences),
                'std_confidence': statistics.stdev(confidences) if len(confidences) > 1 else 0,
                'cache_hit_rate': self._calculate_cache_hit_rate(cot_system),
                'steps_count': len(result.steps),
                'template_used': result.template_used
            })

        return results

    def _calculate_cache_hit_rate(self, cot_system):
        """Calculate current cache hit rate"""
        metrics = cot_system.performance_metrics
        total = metrics['cache_hits'] + metrics['cache_misses']
        return metrics['cache_hits'] / total if total > 0 else 0

    def _analyze_results(self):
        """Analyze and visualize results"""
        df = pd.DataFrame(self.results)

        analysis = {
            'summary': {
                'total_queries': len(self.results),
                'avg_execution_time': df['avg_time'].mean(),
                'avg_confidence': df['avg_confidence'].mean(),
                'cache_effectiveness': df['cache_hit_rate'].mean(),
                'total_steps': df['steps_count'].sum()
            },
            'by_complexity': df.groupby('complexity').agg({
                'avg_time': ['mean', 'std', 'min', 'max'],
                'avg_confidence': ['mean', 'std'],
                'cache_hit_rate': 'mean',
                'steps_count': ['mean', 'sum']
            }).to_dict(),
            'recommendations': self._generate_recommendations(df),
            'performance_insights': self._generate_performance_insights(df)
        }

        # Generate visualizations
        self._create_visualizations(df)

        # Save detailed results
        self._save_detailed_results(df, analysis)

        return analysis

    def _generate_recommendations(self, df):
        """Generate performance recommendations"""
        recommendations = []

        # Check if complex queries are too slow
        complex_time = df[df['complexity'] == 'complex']['avg_time'].mean()
        if complex_time > 1.0:  # More than 1 second
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f"Complex queries averaging {complex_time:.2f}s. Consider increasing max_paths or optimizing templates.",
                'impact': 'High'
            })

        # Check cache effectiveness
        cache_rate = df['cache_hit_rate'].mean()
        if cache_rate < 0.3:
            recommendations.append({
                'type': 'cache',
                'priority': 'medium',
                'message': f"Low cache hit rate ({cache_rate:.1%}). Consider increasing cache size or adjusting similarity threshold.",
                'impact': 'Medium'
            })

        # Check confidence levels
        low_conf = df[df['avg_confidence'] < 0.6]
        if len(low_conf) > 0:
            recommendations.append({
                'type': 'quality',
                'priority': 'high',
                'message': f"{len(low_conf)} queries with low confidence. Review templates and reasoning depth settings.",
                'impact': 'High'
            })

        # Check execution time consistency
        high_variance = df[df['std_time'] > df['avg_time'] * 0.5]
        if len(high_variance) > 0:
            recommendations.append({
                'type': 'stability',
                'priority': 'medium',
                'message': f"{len(high_variance)} queries show high execution time variance. Consider optimizing caching or reducing complexity.",
                'impact': 'Medium'
            })

        return recommendations

    def _generate_performance_insights(self, df):
        """Generate detailed performance insights"""
        insights = {
            'fastest_queries': df.nsmallest(3, 'avg_time')[['query', 'avg_time', 'complexity']].to_dict('records'),
            'slowest_queries': df.nlargest(3, 'avg_time')[['query', 'avg_time', 'complexity']].to_dict('records'),
            'highest_confidence': df.nlargest(3, 'avg_confidence')[['query', 'avg_confidence', 'complexity']].to_dict('records'),
            'lowest_confidence': df.nsmallest(3, 'avg_confidence')[['query', 'avg_confidence', 'complexity']].to_dict('records'),
            'complexity_analysis': {
                'simple_avg_time': df[df['complexity'] == 'simple']['avg_time'].mean(),
                'medium_avg_time': df[df['complexity'] == 'medium']['avg_time'].mean(),
                'complex_avg_time': df[df['complexity'] == 'complex']['avg_time'].mean(),
                'complexity_scaling': 'linear' if df.groupby('complexity')['avg_time'].mean().is_monotonic_increasing else 'non-linear'
            }
        }

        return insights

    def _create_visualizations(self, df):
        """Create performance visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Execution time by complexity
        ax1 = axes[0, 0]
        df.groupby('complexity')['avg_time'].mean().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average Execution Time by Complexity')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)

        # Confidence by complexity
        ax2 = axes[0, 1]
        df.groupby('complexity')['avg_confidence'].mean().plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Average Confidence by Complexity')
        ax2.set_ylabel('Confidence Score')
        ax2.tick_params(axis='x', rotation=45)

        # Time vs Confidence scatter
        ax3 = axes[0, 2]
        for complexity in df['complexity'].unique():
            subset = df[df['complexity'] == complexity]
            ax3.scatter(subset['avg_time'], subset['avg_confidence'],
                       label=complexity, alpha=0.7, s=100)
        ax3.set_xlabel('Execution Time (s)')
        ax3.set_ylabel('Confidence Score')
        ax3.set_title('Execution Time vs Confidence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Cache hit rate over time
        ax4 = axes[1, 0]
        df['cache_hit_rate'].plot(ax=ax4, marker='o', linestyle='-', color='orange')
        ax4.set_title('Cache Hit Rate Progression')
        ax4.set_ylabel('Hit Rate')
        ax4.set_xlabel('Query Index')
        ax4.grid(True, alpha=0.3)

        # Steps count by complexity
        ax5 = axes[1, 1]
        df.groupby('complexity')['steps_count'].mean().plot(kind='bar', ax=ax5, color='purple')
        ax5.set_title('Average Steps by Complexity')
        ax5.set_ylabel('Number of Steps')
        ax5.tick_params(axis='x', rotation=45)

        # Execution time distribution
        ax6 = axes[1, 2]
        df['avg_time'].hist(bins=10, ax=ax6, color='lightcoral', alpha=0.7)
        ax6.set_title('Execution Time Distribution')
        ax6.set_xlabel('Time (seconds)')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('benchmarks/cot_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Visualizations saved to benchmarks/cot_performance_analysis.png")

    def _save_detailed_results(self, df, analysis):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_data = {
            'timestamp': timestamp,
            'raw_data': df.to_dict('records'),
            'analysis': analysis,
            'metadata': {
                'total_queries': len(df),
                'complexities': df['complexity'].value_counts().to_dict(),
                'templates_used': df['template_used'].value_counts().to_dict()
            }
        }

        filename = f'benchmarks/cot_benchmark_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info("Detailed results saved to {}", extra={"filename": filename})

class MemoryProfiler:
    """Profile memory usage of CoT system"""

    def __init__(self):
        self.process = None
        self.snapshots = []

        try:
            import psutil
            self.process = psutil.Process(os.getpid())
        except ImportError:
            logger.info("Warning: psutil not available. Memory profiling disabled.")

    async def profile_memory_usage(self, cot_system, num_queries=100):
        """Profile memory usage over multiple queries"""
        if not self.process:
            logger.info("Memory profiling not available")
            return []

        import tracemalloc
        tracemalloc.start()

        # Baseline memory
        baseline = self.process.memory_info().rss / 1024 / 1024  # MB

        queries = [
            f"Query {i}: " + "x" * (i % 100)  # Variable length queries
            for i in range(num_queries)
        ]

        memory_usage = []

        for i, query in enumerate(queries):
            await cot_system.reason(query)

            if i % 10 == 0:
                current = self.process.memory_info().rss / 1024 / 1024
                memory_usage.append({
                    'query_num': i,
                    'memory_mb': current,
                    'delta_mb': current - baseline,
                    'cache_size': len(cot_system.reasoning_cache.cache)
                })

                # Take snapshot
                snapshot = tracemalloc.take_snapshot()
                self.snapshots.append(snapshot)

        # Analyze memory growth
        self._analyze_memory_growth(memory_usage)

        tracemalloc.stop()
        return memory_usage

    def _analyze_memory_growth(self, memory_usage):
        """Analyze memory growth patterns"""
        if len(memory_usage) < 2:
            return

        # Calculate growth rate
        first = memory_usage[0]['memory_mb']
        last = memory_usage[-1]['memory_mb']
        growth_rate = (last - first) / first * 100

        logger.info("\nMemory Analysis:")
        logger.info("Initial memory: {} MB", extra={"first": first})
        logger.info("Final memory: {} MB", extra={"last": last})
        logger.info("Growth rate: {}%", extra={"growth_rate": growth_rate})

        # Check for memory leaks
        if growth_rate > 50:
            logger.info("WARNING: High memory growth detected. Possible memory leak.")

        # Analyze top memory consumers
        if self.snapshots:
            self._show_top_memory_consumers()

    def _show_top_memory_consumers(self):
        """Show top memory consuming lines"""
        if len(self.snapshots) < 2:
            return

        first = self.snapshots[0]
        last = self.snapshots[-1]

        top_stats = last.compare_to(first, 'lineno')

        logger.info("\nTop memory consumers:")
        for stat in top_stats[:10]:
            logger.info("{}", extra={"stat": stat})

class LoadTester:
    """Load testing for CoT system"""

    def __init__(self, cot_system):
        self.cot_system = cot_system
        self.results = []

    async def run_load_test(self, num_concurrent=10, total_queries=100):
        """Run load test with concurrent queries"""
        logger.info("Starting load test: {} concurrent, {} total", extra={"num_concurrent": num_concurrent, "total_queries": total_queries})

        queries = self._generate_test_queries(total_queries)
        start_time = time.time()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(num_concurrent)

        async def process_with_limit(query, index):
            async with semaphore:
                return await self._process_query(query, index)

        # Run all queries
        tasks = [
            process_with_limit(query, i)
            for i, query in enumerate(queries)
        ]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        # Analyze results
        analysis = self._analyze_load_test_results(results, duration)

        return analysis

    async def _process_query(self, query, index):
        """Process a single query and record metrics"""
        start = time.perf_counter()

        try:
            result = await self.cot_system.reason(query)
            end = time.perf_counter()

            return {
                'index': index,
                'success': True,
                'duration': end - start,
                'confidence': result.total_confidence,
                'steps': len(result.steps),
                'error': None
            }
        except Exception as e:
            end = time.perf_counter()

            return {
                'index': index,
                'success': False,
                'duration': end - start,
                'confidence': 0,
                'steps': 0,
                'error': str(e)
            }

    def _generate_test_queries(self, count):
        """Generate diverse test queries"""
        templates = [
            "Explain the concept of {}",
            "What are the benefits of {}?",
            "Compare {} and {}",
            "How does {} work?",
            "Analyze the impact of {} on {}"
        ]

        topics = [
            "machine learning", "quantum computing", "blockchain",
            "renewable energy", "artificial intelligence", "cybersecurity",
            "biotechnology", "space exploration", "climate change"
        ]

        queries = []
        for i in range(count):
            template = templates[i % len(templates)]
            topic1 = topics[i % len(topics)]
            topic2 = topics[(i + 1) % len(topics)]

            if '{}' in template and template.count('{}') == 2:
                query = template.format(topic1, topic2)
            else:
                query = template.format(topic1)

            queries.append(query)

        return queries

    def _analyze_load_test_results(self, results, total_duration):
        """Analyze load test results"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        if successful:
            avg_duration = sum(r['duration'] for r in successful) / len(successful)
            avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
            avg_steps = sum(r['steps'] for r in successful) / len(successful)
        else:
            avg_duration = avg_confidence = avg_steps = 0

        analysis = {
            'summary': {
                'total_queries': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) * 100,
                'total_duration': total_duration,
                'queries_per_second': len(results) / total_duration
            },
            'performance': {
                'avg_query_duration': avg_duration,
                'min_duration': min(r['duration'] for r in successful) if successful else 0,
                'max_duration': max(r['duration'] for r in successful) if successful else 0,
                'avg_confidence': avg_confidence,
                'avg_steps': avg_steps
            },
            'errors': [
                {'index': r['index'], 'error': r['error']}
                for r in failed
            ],
            'recommendations': self._generate_load_recommendations(results, total_duration)
        }

        return analysis

    def _generate_load_recommendations(self, results, duration):
        """Generate recommendations based on load test"""
        recommendations = []

        failed_count = sum(1 for r in results if not r['success'])
        if failed_count > len(results) * 0.05:  # More than 5% failure
            recommendations.append(
                f"High failure rate ({failed_count}/{len(results)}). "
                "Consider implementing better error handling or reducing concurrency."
            )

        qps = len(results) / duration
        if qps < 10:
            recommendations.append(
                f"Low throughput ({qps:.1f} queries/second). "
                "Consider optimizing reasoning paths or increasing cache size."
            )

        return recommendations

async def run_comprehensive_benchmarks():
    """Run all comprehensive benchmarks"""
    logger.info("🚀 Starting Comprehensive CoT Benchmark Suite")
    print("=" * 60)

    # Create CoT system for benchmarking
    cot_system = OptimizedChainOfThought(
        "benchmark_cot",
        config={
            'max_paths': 3,
            'cache_size': 500,
            'cache_ttl': 24,
            'parallel_threshold': 0.5,
            'confidence_threshold': 0.7
        }
    )

    # Run performance benchmarks
    logger.info("\n📊 Running Performance Benchmarks...")
    benchmark_suite = CoTBenchmarkSuite()
    performance_results = await benchmark_suite.run_benchmarks(cot_system)

    # Run memory profiling
    logger.info("\n🧠 Running Memory Profiling...")
    memory_profiler = MemoryProfiler()
    memory_results = await memory_profiler.profile_memory_usage(cot_system, num_queries=50)

    # Run load testing
    logger.info("\n⚡ Running Load Testing...")
    load_tester = LoadTester(cot_system)
    load_results = await load_tester.run_load_test(num_concurrent=5, total_queries=50)

    # Print summary
    print("\n" + "=" * 60)
    logger.info("📈 BENCHMARK SUMMARY")
    print("=" * 60)

    logger.info("\nPerformance Metrics:")
    logger.info("  Average Execution Time: {}s", extra={"performance_results__summary___avg_execution_time_": performance_results['summary']['avg_execution_time']})
    logger.info("  Average Confidence: {}", extra={"performance_results__summary___avg_confidence_": performance_results['summary']['avg_confidence']})
    logger.info("  Cache Effectiveness: {}", extra={"performance_results__summary___cache_effectiveness_": performance_results['summary']['cache_effectiveness']})
    logger.info("  Total Steps Generated: {}", extra={"performance_results__summary___total_steps_": performance_results['summary']['total_steps']})

    logger.info("\nLoad Test Results:")
    logger.info("  Success Rate: {}%", extra={"load_results__summary___success_rate_": load_results['summary']['success_rate']})
    logger.info("  Queries per Second: {}", extra={"load_results__summary___queries_per_second_": load_results['summary']['queries_per_second']})
    logger.info("  Average Query Duration: {}s", extra={"load_results__performance___avg_query_duration_": load_results['performance']['avg_query_duration']})

    logger.info("\nRecommendations:")
    for rec in performance_results['recommendations']:
        logger.info("  [{}] {}", extra={"rec__priority__upper__": rec['priority'].upper(), "rec__message_": rec['message']})

    for rec in load_results['recommendations']:
        logger.info("  [LOAD] {}", extra={"rec": rec})

    return {
        'performance': performance_results,
        'memory': memory_results,
        'load': load_results
    }

if __name__ == "__main__":
    # Run comprehensive benchmarks
    asyncio.run(run_comprehensive_benchmarks())
