#!/usr/bin/env python3
"""
Chain of Thought Benchmark Runner
Simple script to run CoT performance benchmarks with different options
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.cot_benchmark_suite import (
    run_performance_tests,
    CoTBenchmark,
    QueryDataset,
    BenchmarkVisualizer
)

async def run_basic_benchmark():
    """Run basic performance benchmark"""
    print("Running basic performance benchmark...")
    benchmark = CoTBenchmark()
    
    # Create CoT system
    from src.core.optimized_chain_of_thought import OptimizedChainOfThought
    cot = OptimizedChainOfThought("basic_test", {'max_paths': 3, 'cache_size': 500})
    
    # Test with simple queries
    queries = QueryDataset.get_simple_queries()[:5]
    suite = await benchmark.run_benchmark(cot, queries, "basic_test")
    analysis = benchmark.analyze_results(suite)
    
    print("\nBasic Benchmark Results:")
    print(f"Total queries: {analysis['overall']['total_queries']}")
    print(f"Average execution time: {analysis['overall']['avg_execution_time']:.3f}s")
    print(f"Average confidence: {analysis['overall']['avg_confidence']:.3f}")
    print(f"Cache hit rate: {analysis['overall']['cache_hit_rate']:.2%}")
    
    return analysis

async def run_complexity_benchmark():
    """Run benchmark across different complexity levels"""
    print("Running complexity-based benchmark...")
    benchmark = CoTBenchmark()
    
    from src.core.optimized_chain_of_thought import OptimizedChainOfThought
    cot = OptimizedChainOfThought("complexity_test", {'max_paths': 5, 'cache_size': 1000})
    
    # Test with different complexity levels
    all_queries = (
        QueryDataset.get_simple_queries()[:3] +
        QueryDataset.get_medium_queries()[:3] +
        QueryDataset.get_complex_queries()[:3]
    )
    
    suite = await benchmark.run_benchmark(cot, all_queries, "complexity_test")
    analysis = benchmark.analyze_results(suite)
    
    print("\nComplexity Benchmark Results:")
    print("By complexity level:")
    for level, data in analysis['by_complexity'].items():
        if data:
            print(f"  {level.capitalize()}:")
            print(f"    Count: {data['count']}")
            print(f"    Avg time: {data['avg_execution_time']:.3f}s")
            print(f"    Avg confidence: {data['avg_confidence']:.3f}")
    
    return analysis

async def run_configuration_benchmark():
    """Run benchmark with different configurations"""
    print("Running configuration comparison benchmark...")
    benchmark = CoTBenchmark()
    
    comparative_results = await benchmark.run_comparative_benchmark()
    
    print("\nConfiguration Comparison Results:")
    for config_name, suite in comparative_results.items():
        analysis = benchmark.analyze_results(suite)
        print(f"\n{config_name}:")
        print(f"  Avg execution time: {analysis['overall']['avg_execution_time']:.3f}s")
        print(f"  Avg confidence: {analysis['overall']['avg_confidence']:.3f}")
        print(f"  Avg memory: {analysis['overall']['avg_memory_mb']:.2f} MB")
    
    return comparative_results

async def run_domain_benchmark():
    """Run benchmark for specific domains"""
    print("Running domain-specific benchmark...")
    benchmark = CoTBenchmark()
    
    from src.core.optimized_chain_of_thought import OptimizedChainOfThought
    cot = OptimizedChainOfThought("domain_test", {'max_paths': 3, 'cache_size': 500})
    
    # Test different domains
    domains = {
        'mathematical': QueryDataset.get_mathematical_queries()[:5],
        'ai_agent': QueryDataset.get_ai_agent_queries()[:5]
    }
    
    results = {}
    for domain, queries in domains.items():
        print(f"\nTesting {domain} domain...")
        suite = await benchmark.run_benchmark(cot, queries, f"{domain}_domain")
        analysis = benchmark.analyze_results(suite)
        results[domain] = analysis
        
        print(f"  {domain.capitalize()} domain results:")
        print(f"    Avg execution time: {analysis['overall']['avg_execution_time']:.3f}s")
        print(f"    Avg confidence: {analysis['overall']['avg_confidence']:.3f}")
        print(f"    Avg steps: {analysis['overall']['avg_steps']:.1f}")
    
    return results

async def run_full_benchmark():
    """Run the complete benchmark suite"""
    print("Running full benchmark suite...")
    return await run_performance_tests()

def main():
    parser = argparse.ArgumentParser(description="Run Chain of Thought Performance Benchmarks")
    parser.add_argument(
        '--type', 
        choices=['basic', 'complexity', 'config', 'domain', 'full'],
        default='basic',
        help='Type of benchmark to run'
    )
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--output', 
        type=str,
        default='cot_benchmark_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Run the selected benchmark
    if args.type == 'basic':
        results = asyncio.run(run_basic_benchmark())
    elif args.type == 'complexity':
        results = asyncio.run(run_complexity_benchmark())
    elif args.type == 'config':
        results = asyncio.run(run_configuration_benchmark())
    elif args.type == 'domain':
        results = asyncio.run(run_domain_benchmark())
    elif args.type == 'full':
        results = asyncio.run(run_full_benchmark())
    
    # Save results
    import json
    import time
    
    with open(args.output, 'w') as f:
        serializable_results = {
            'timestamp': time.time(),
            'benchmark_type': args.type,
            'results': results
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nBenchmark results saved to '{args.output}'")
    print("Benchmark completed successfully!")

if __name__ == "__main__":
    main() 