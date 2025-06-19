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
import logging

logger = logging.getLogger(__name__)


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
    logger.info("Running basic performance benchmark...")
    benchmark = CoTBenchmark()
    
    # Create CoT system
    from src.core.optimized_chain_of_thought import OptimizedChainOfThought
    cot = OptimizedChainOfThought("basic_test", {'max_paths': 3, 'cache_size': 500})
    
    # Test with simple queries
    queries = QueryDataset.get_simple_queries()[:5]
    suite = await benchmark.run_benchmark(cot, queries, "basic_test")
    analysis = benchmark.analyze_results(suite)
    
    logger.info("\nBasic Benchmark Results:")
    logger.info("Total queries: {}", extra={"analysis__overall___total_queries_": analysis['overall']['total_queries']})
    logger.info("Average execution time: {}s", extra={"analysis__overall___avg_execution_time_": analysis['overall']['avg_execution_time']})
    logger.info("Average confidence: {}", extra={"analysis__overall___avg_confidence_": analysis['overall']['avg_confidence']})
    logger.info("Cache hit rate: {}", extra={"analysis__overall___cache_hit_rate_": analysis['overall']['cache_hit_rate']})
    
    return analysis

async def run_complexity_benchmark():
    """Run benchmark across different complexity levels"""
    logger.info("Running complexity-based benchmark...")
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
    
    logger.info("\nComplexity Benchmark Results:")
    logger.info("By complexity level:")
    for level, data in analysis['by_complexity'].items():
        if data:
            logger.info("  {}:", extra={"level_capitalize__": level.capitalize()})
            logger.info("    Count: {}", extra={"data__count_": data['count']})
            logger.info("    Avg time: {}s", extra={"data__avg_execution_time_": data['avg_execution_time']})
            logger.info("    Avg confidence: {}", extra={"data__avg_confidence_": data['avg_confidence']})
    
    return analysis

async def run_configuration_benchmark():
    """Run benchmark with different configurations"""
    logger.info("Running configuration comparison benchmark...")
    benchmark = CoTBenchmark()
    
    comparative_results = await benchmark.run_comparative_benchmark()
    
    logger.info("\nConfiguration Comparison Results:")
    for config_name, suite in comparative_results.items():
        analysis = benchmark.analyze_results(suite)
        logger.info("\n{}:", extra={"config_name": config_name})
        logger.info("  Avg execution time: {}s", extra={"analysis__overall___avg_execution_time_": analysis['overall']['avg_execution_time']})
        logger.info("  Avg confidence: {}", extra={"analysis__overall___avg_confidence_": analysis['overall']['avg_confidence']})
        logger.info("  Avg memory: {} MB", extra={"analysis__overall___avg_memory_mb_": analysis['overall']['avg_memory_mb']})
    
    return comparative_results

async def run_domain_benchmark():
    """Run benchmark for specific domains"""
    logger.info("Running domain-specific benchmark...")
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
        logger.info("\nTesting {} domain...", extra={"domain": domain})
        suite = await benchmark.run_benchmark(cot, queries, f"{domain}_domain")
        analysis = benchmark.analyze_results(suite)
        results[domain] = analysis
        
        logger.info("  {} domain results:", extra={"domain_capitalize__": domain.capitalize()})
        logger.info("    Avg execution time: {}s", extra={"analysis__overall___avg_execution_time_": analysis['overall']['avg_execution_time']})
        logger.info("    Avg confidence: {}", extra={"analysis__overall___avg_confidence_": analysis['overall']['avg_confidence']})
        logger.info("    Avg steps: {}", extra={"analysis__overall___avg_steps_": analysis['overall']['avg_steps']})
    
    return results

async def run_full_benchmark():
    """Run the complete benchmark suite"""
    logger.info("Running full benchmark suite...")
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
    
    logger.info("\nBenchmark results saved to '{}'", extra={"args_output": args.output})
    logger.info("Benchmark completed successfully!")

if __name__ == "__main__":
    main() 