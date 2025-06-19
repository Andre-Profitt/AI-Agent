"""Integration Hub Improvements - Usage Examples"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_tool_compatibility_checker():
    """Demonstrate ToolCompatibilityChecker functionality"""
    print("\n=== Tool Compatibility Checker Demo ===")
    
    from src.integration_hub import get_tool_compatibility_checker
    
    checker = get_tool_compatibility_checker()
    if not checker:
        print("Tool compatibility checker not available")
        return
    
    # Register tool requirements
    checker.register_tool_requirements("search_tool", {
        "api_version": "v1.0",
        "dependencies": [
            {"name": "requests", "version": "2.28.0"},
            {"name": "beautifulsoup4", "version": "4.11.0"}
        ]
    })
    
    checker.register_tool_requirements("file_processor", {
        "api_version": "v1.0",
        "dependencies": [
            {"name": "pandas", "version": "1.5.0"},
            {"name": "numpy", "version": "1.24.0"}
        ]
    })
    
    checker.register_tool_requirements("incompatible_tool", {
        "api_version": "v2.0",  # Different API version
        "dependencies": [
            {"name": "requests", "version": "3.0.0"}  # Conflicting version
        ]
    })
    
    # Check compatibility
    print(f"search_tool + file_processor compatible: {checker.check_compatibility('search_tool', 'file_processor')}")
    print(f"search_tool + incompatible_tool compatible: {checker.check_compatibility('search_tool', 'incompatible_tool')}")
    
    # Get compatible tools
    compatible = checker.get_compatible_tools("search_tool")
    print(f"Tools compatible with search_tool: {compatible}")
    
    incompatible = checker.get_incompatible_tools("search_tool")
    print(f"Tools incompatible with search_tool: {incompatible}")

async def demonstrate_semantic_tool_discovery():
    """Demonstrate SemanticToolDiscovery functionality"""
    print("\n=== Semantic Tool Discovery Demo ===")
    
    from src.integration_hub import get_semantic_discovery
    
    discovery = get_semantic_discovery()
    if not discovery:
        print("Semantic tool discovery not available")
        return
    
    # Index tools with descriptions and examples
    discovery.index_tool(
        "web_search",
        "Search the web for information",
        ["Find information about AI", "Search for latest news", "Look up technical documentation"]
    )
    
    discovery.index_tool(
        "file_reader",
        "Read and process files",
        ["Read PDF documents", "Parse CSV files", "Extract text from images"]
    )
    
    discovery.index_tool(
        "calculator",
        "Perform mathematical calculations",
        ["Calculate percentages", "Solve equations", "Convert units"]
    )
    
    # Find tools for specific tasks
    tasks = [
        "I need to find information about machine learning",
        "I want to read a PDF file",
        "Calculate the area of a circle"
    ]
    
    for task in tasks:
        tools = discovery.find_tools_for_task(task, top_k=3)
        print(f"\nTask: {task}")
        for tool_name, similarity in tools:
            print(f"  - {tool_name}: {similarity:.3f}")

async def demonstrate_resource_pool_manager():
    """Demonstrate ResourcePoolManager functionality"""
    print("\n=== Resource Pool Manager Demo ===")
    
    from src.integration_hub import get_resource_manager
    
    resource_manager = get_resource_manager()
    if not resource_manager:
        print("Resource pool manager not available")
        return
    
    # Create a mock database connection factory
    async def create_db_connection():
        await asyncio.sleep(0.1)  # Simulate connection time
        return {"connection_id": f"conn_{int(time.time() * 1000)}", "status": "connected"}
    
    # Create a pool for database connections
    await resource_manager.create_pool("database", create_db_connection, min_size=2, max_size=5)
    
    # Demonstrate acquiring and releasing resources
    print("Acquiring database connections...")
    connections = []
    
    for i in range(3):
        conn = await resource_manager.acquire("database")
        connections.append(conn)
        print(f"  Acquired connection {i+1}: {conn['connection_id']}")
    
    # Check pool stats
    stats = resource_manager.get_pool_stats("database")
    print(f"Pool stats: {stats}")
    
    # Release connections
    for i, conn in enumerate(connections):
        await resource_manager.release("database", conn)
        print(f"  Released connection {i+1}: {conn['connection_id']}")
    
    # Check final stats
    final_stats = resource_manager.get_pool_stats("database")
    print(f"Final pool stats: {final_stats}")

async def demonstrate_tool_version_manager():
    """Demonstrate ToolVersionManager functionality"""
    print("\n=== Tool Version Manager Demo ===")
    
    from src.integration_hub import get_tool_version_manager
    
    version_manager = get_tool_version_manager()
    if not version_manager:
        print("Tool version manager not available")
        return
    
    # Register different versions of a tool
    version_manager.register_version("search_tool", "1.0", {
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "default": 10}
        }
    })
    
    version_manager.register_version("search_tool", "2.0", {
        "parameters": {
            "search_term": {"type": "string", "required": True},  # Renamed from query
            "max_results": {"type": "integer", "default": 10},
            "filters": {"type": "object", "default": {}}  # New parameter
        }
    })
    
    version_manager.register_version("search_tool", "3.0", {
        "parameters": {
            "search_term": {"type": "string", "required": True},
            "max_results": {"type": "integer", "default": 10},
            "filter_config": {"type": "object", "default": {}},  # Renamed from filters
            "include_metadata": {"type": "boolean", "default": False}  # New parameter
        }
    })
    
    # Get latest version
    latest = version_manager.get_latest_version("search_tool")
    print(f"Latest version of search_tool: {latest}")
    
    # Test parameter migration
    old_params = {"query": "AI research", "max_results": 5}
    migrated_params = version_manager.migrate_params("search_tool", old_params, "1.0", "2.0")
    print(f"Migrated params 1.0->2.0: {migrated_params}")
    
    migrated_params_2 = version_manager.migrate_params("search_tool", migrated_params, "2.0", "3.0")
    print(f"Migrated params 2.0->3.0: {migrated_params_2}")
    
    # Deprecate old version
    version_manager.deprecate_version("search_tool", "1.0")
    print("Deprecated search_tool version 1.0")

async def demonstrate_rate_limit_manager():
    """Demonstrate RateLimitManager functionality"""
    print("\n=== Rate Limit Manager Demo ===")
    
    from src.integration_hub import get_rate_limit_manager
    
    rate_manager = get_rate_limit_manager()
    if not rate_manager:
        print("Rate limit manager not available")
        return
    
    # Set rate limits for different tools
    rate_manager.set_limit("api_tool", calls_per_minute=10, burst_size=15)
    rate_manager.set_limit("search_tool", calls_per_minute=5, burst_size=8)
    
    # Simulate tool calls
    print("Simulating tool calls with rate limiting...")
    
    async def simulate_tool_call(tool_name: str, call_number: int):
        await rate_manager.check_and_wait(tool_name)
        print(f"  {tool_name} call {call_number} executed at {datetime.now().strftime('%H:%M:%S')}")
    
    # Make multiple calls to test rate limiting
    tasks = []
    for i in range(15):
        tasks.append(simulate_tool_call("api_tool", i + 1))
        tasks.append(simulate_tool_call("search_tool", i + 1))
    
    await asyncio.gather(*tasks)
    
    # Check rate limit statistics
    api_stats = rate_manager.get_tool_stats("api_tool")
    search_stats = rate_manager.get_tool_stats("search_tool")
    
    print(f"\nAPI tool stats: {api_stats}")
    print(f"Search tool stats: {search_stats}")

async def demonstrate_monitoring_dashboard():
    """Demonstrate MonitoringDashboard functionality"""
    print("\n=== Monitoring Dashboard Demo ===")
    
    from src.integration_hub import get_monitoring_dashboard
    
    dashboard = get_monitoring_dashboard()
    if not dashboard:
        print("Monitoring dashboard not available")
        return
    
    # Collect metrics
    print("Collecting metrics...")
    metrics = await dashboard.collect_metrics()
    
    print(f"Tool metrics: {metrics.get('tool_metrics', {})}")
    print(f"Session metrics: {metrics.get('session_metrics', {})}")
    print(f"Resource metrics: {metrics.get('resource_metrics', {})}")
    
    # Check for alerts
    alerts = dashboard.get_alerts()
    if alerts:
        print(f"Active alerts: {alerts}")
    else:
        print("No active alerts")
    
    # Get alerts by severity
    critical_alerts = dashboard.get_alerts(severity="critical")
    warning_alerts = dashboard.get_alerts(severity="warning")
    
    print(f"Critical alerts: {len(critical_alerts)}")
    print(f"Warning alerts: {len(warning_alerts)}")

async def demonstrate_integration_test_framework():
    """Demonstrate IntegrationTestFramework functionality"""
    print("\n=== Integration Test Framework Demo ===")
    
    from src.integration_hub import get_test_framework, get_integration_hub
    
    test_framework = get_test_framework()
    if not test_framework:
        print("Integration test framework not available")
        return
    
    # Run integration tests
    print("Running integration tests...")
    results = await test_framework.run_integration_tests()
    
    print("Test Results:")
    for test_name, result in results.items():
        status = "PASSED" if result['passed'] else "FAILED"
        print(f"  {test_name}: {status}")
        if not result['passed']:
            print(f"    Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"    Details: {result.get('details', 'No details')}")

async def demonstrate_migration_helper():
    """Demonstrate MigrationHelper functionality"""
    print("\n=== Migration Helper Demo ===")
    
    from src.integration_hub import MigrationHelper, get_unified_registry
    
    # Create a mock old registry
    class MockOldRegistry:
        def __init__(self):
            self.tools = {
                "old_search": type('MockTool', (), {'name': 'old_search', 'description': 'Old search tool'})(),
                "old_file_reader": type('MockTool', (), {'name': 'old_file_reader', 'description': 'Old file reader'})()
            }
            self.mcp_announcements = {
                "old_search": {"name": "old_search", "description": "Old search tool"}
            }
    
    old_registry = MockOldRegistry()
    unified_registry = get_unified_registry()
    
    # Create migration helper
    migration_helper = MigrationHelper(old_registry, unified_registry)
    
    # Perform migration
    print("Migrating tools from old registry...")
    migration_report = migration_helper.migrate_tools()
    
    print(f"Migration results:")
    print(f"  Migrated: {migration_report['migrated']}")
    print(f"  Failed: {migration_report['failed']}")
    print(f"  Warnings: {migration_report['warnings']}")

async def demonstrate_advanced_orchestrator_features():
    """Demonstrate advanced ToolOrchestrator features"""
    print("\n=== Advanced Orchestrator Features Demo ===")
    
    from src.integration_hub import get_tool_orchestrator
    
    orchestrator = get_tool_orchestrator()
    if not orchestrator:
        print("Tool orchestrator not available")
        return
    
    # Test compatibility checking
    print("Testing compatibility checking...")
    result = await orchestrator.execute_with_compatibility_check("test_tool", {"param": "value"})
    print(f"Compatibility check result: {result}")
    
    # Test resource pool execution
    print("Testing resource pool execution...")
    result = await orchestrator.execute_with_resource_pool("test_tool", {"param": "value"}, "database")
    print(f"Resource pool result: {result}")

async def main():
    """Run all demonstrations"""
    print("Integration Hub Improvements - Comprehensive Demo")
    print("=" * 50)
    
    # Initialize integration hub
    from src.integration_hub import initialize_integrations
    await initialize_integrations()
    
    try:
        # Run all demonstrations
        await demonstrate_tool_compatibility_checker()
        await demonstrate_semantic_tool_discovery()
        await demonstrate_resource_pool_manager()
        await demonstrate_tool_version_manager()
        await demonstrate_rate_limit_manager()
        await demonstrate_monitoring_dashboard()
        await demonstrate_integration_test_framework()
        await demonstrate_migration_helper()
        await demonstrate_advanced_orchestrator_features()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        logger.exception("Demonstration failed")
    
    finally:
        # Cleanup
        from src.integration_hub import cleanup_integrations
        await cleanup_integrations()

if __name__ == "__main__":
    asyncio.run(main())
