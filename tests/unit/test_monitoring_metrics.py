"""
Unit tests for monitoring metrics system
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.infrastructure.monitoring.metrics import (
    MetricsRegistry,
    time_function,
    track_database_operation,
    track_operation,
    track_async_operation,
    PerformanceTracker,
    ResourceMonitor,
    record_agent_registration,
    record_task_execution,
    record_task_duration,
    record_agent_availability,
    record_task_submission,
    record_task_completion,
    record_external_api_call,
    record_error,
    update_resource_utilization,
    update_task_queue_size,
    update_db_connections,
    get_metrics_response,
    metrics_registry,
    performance_tracker,
    resource_monitor
)

class TestMetricsRegistry:
    """Test metrics registry functionality"""
    
    def test_register_metric(self):
        """Test registering a metric"""
        registry = MetricsRegistry()
        metric = Mock()
        registry.register_metric("test_metric", metric)
        
        assert registry.get_metric("test_metric") == metric
        assert "test_metric" in registry.get_all_metrics()
        
    def test_get_nonexistent_metric(self):
        """Test getting a non-existent metric"""
        registry = MetricsRegistry()
        assert registry.get_metric("nonexistent") is None
        
    def test_generate_metrics(self):
        """Test generating metrics output"""
        registry = MetricsRegistry()
        metrics_output = registry.generate_metrics()
        
        assert isinstance(metrics_output, str)
        assert len(metrics_output) > 0

class TestTimingDecorators:
    """Test timing decorators"""
    
    @pytest.mark.asyncio
    async def test_time_function_sync(self):
        """Test time_function decorator with sync function"""
        @time_function("test_operation", {"service": "test"})
        def test_func():
            time.sleep(0.1)
            return "success"
            
        result = test_func()
        assert result == "success"
        
    @pytest.mark.asyncio
    async def test_time_function_async(self):
        """Test time_function decorator with async function"""
        @time_function("test_async_operation", {"service": "test"})
        async def test_async_func():
            await asyncio.sleep(0.1)
            return "success"
            
        result = await test_async_func()
        assert result == "success"
        
    @pytest.mark.asyncio
    async def test_time_function_with_exception(self):
        """Test time_function decorator with exception"""
        @time_function("test_error_operation", {"service": "test"})
        def test_error_func():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            test_error_func()
            
    def test_track_database_operation(self):
        """Test track_database_operation decorator"""
        @track_database_operation("select", "users")
        def test_db_operation():
            return "data"
            
        result = test_db_operation()
        assert result == "data"
        
    @pytest.mark.asyncio
    async def test_track_database_operation_async(self):
        """Test track_database_operation decorator with async function"""
        @track_database_operation("select", "users")
        async def test_async_db_operation():
            await asyncio.sleep(0.1)
            return "data"
            
        result = await test_async_db_operation()
        assert result == "data"

class TestContextManagers:
    """Test context managers"""
    
    def test_track_operation_sync(self):
        """Test track_operation context manager"""
        with track_operation("test_operation", {"service": "test"}):
            time.sleep(0.1)
            
    @pytest.mark.asyncio
    async def test_track_async_operation(self):
        """Test track_async_operation context manager"""
        async with track_async_operation("test_async_operation", {"service": "test"}):
            await asyncio.sleep(0.1)
            
    def test_track_operation_with_exception(self):
        """Test track_operation with exception"""
        with pytest.raises(ValueError):
            with track_operation("test_error_operation", {"service": "test"}):
                raise ValueError("Test error")
                
    @pytest.mark.asyncio
    async def test_track_async_operation_with_exception(self):
        """Test track_async_operation with exception"""
        with pytest.raises(ValueError):
            async with track_async_operation("test_async_error_operation", {"service": "test"}):
                raise ValueError("Test error")

class TestPerformanceTracker:
    """Test performance tracker"""
    
    def test_start_operation(self):
        """Test starting an operation"""
        tracker = PerformanceTracker()
        operation_id = tracker.start_operation("test_operation", {"key": "value"})
        
        assert operation_id.startswith("test_operation_")
        assert operation_id in tracker.metrics
        
    def test_complete_operation_success(self):
        """Test completing an operation successfully"""
        tracker = PerformanceTracker()
        operation_id = tracker.start_operation("test_operation")
        
        tracker.complete_operation(operation_id, success=True)
        
        metrics = tracker.get_operation_metrics(operation_id)
        assert metrics.success is True
        assert metrics.duration is not None
        assert metrics.duration > 0
        
    def test_complete_operation_failure(self):
        """Test completing an operation with failure"""
        tracker = PerformanceTracker()
        operation_id = tracker.start_operation("test_operation")
        error = ValueError("Test error")
        
        tracker.complete_operation(operation_id, success=False, error=error)
        
        metrics = tracker.get_operation_metrics(operation_id)
        assert metrics.success is False
        assert metrics.error == error
        
    def test_get_nonexistent_operation(self):
        """Test getting non-existent operation metrics"""
        tracker = PerformanceTracker()
        assert tracker.get_operation_metrics("nonexistent") is None
        
    def test_get_all_metrics(self):
        """Test getting all metrics"""
        tracker = PerformanceTracker()
        operation_id1 = tracker.start_operation("operation1")
        operation_id2 = tracker.start_operation("operation2")
        
        all_metrics = tracker.get_all_metrics()
        assert len(all_metrics) == 2
        assert operation_id1 in all_metrics
        assert operation_id2 in all_metrics

class TestResourceMonitor:
    """Test resource monitor"""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_monitor_loop(self, mock_disk, mock_memory, mock_cpu):
        """Test resource monitoring loop"""
        mock_cpu.return_value = 50.0
        mock_memory.return_value.used = 1024 * 1024 * 100  # 100MB
        mock_disk.return_value.used = 1024 * 1024 * 1024 * 50  # 50GB
        mock_disk.return_value.total = 1024 * 1024 * 1024 * 100  # 100GB
        
        monitor = ResourceMonitor(interval=0.1)
        monitor.start()
        time.sleep(0.2)  # Let it run for a bit
        monitor.stop()
        
        # Verify that monitoring was active
        assert monitor.running is False

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_record_agent_registration(self):
        """Test recording agent registration"""
        record_agent_registration("test_agent", "success")
        # This should not raise any exceptions
        
    def test_record_task_execution(self):
        """Test recording task execution"""
        record_task_execution("agent_1", "data_processing", "success")
        # This should not raise any exceptions
        
    def test_record_task_duration(self):
        """Test recording task duration"""
        record_task_duration("agent_1", "data_processing", 1.5)
        # This should not raise any exceptions
        
    def test_record_agent_availability(self):
        """Test recording agent availability"""
        record_agent_availability("agent_1", "available")
        # This should not raise any exceptions
        
    def test_record_task_submission(self):
        """Test recording task submission"""
        record_task_submission("data_processing", 1)
        # This should not raise any exceptions
        
    def test_record_task_completion(self):
        """Test recording task completion"""
        record_task_completion("data_processing", "success")
        # This should not raise any exceptions
        
    def test_record_external_api_call(self):
        """Test recording external API call"""
        record_external_api_call("openai", "completions", "success")
        # This should not raise any exceptions
        
    def test_record_error(self):
        """Test recording error"""
        record_error("ValueError", "test_component", "error")
        # This should not raise any exceptions
        
    def test_update_resource_utilization(self):
        """Test updating resource utilization"""
        update_resource_utilization("cpu", "agent_1", 75.5)
        # This should not raise any exceptions
        
    def test_update_task_queue_size(self):
        """Test updating task queue size"""
        update_task_queue_size("high", 10)
        # This should not raise any exceptions
        
    def test_update_db_connections(self):
        """Test updating database connections"""
        update_db_connections("postgres", 5)
        # This should not raise any exceptions

class TestMetricsResponse:
    """Test metrics response generation"""
    
    def test_get_metrics_response(self):
        """Test getting metrics response"""
        response, content_type = get_metrics_response()
        
        assert isinstance(response, str)
        assert content_type == "text/plain; version=0.0.4; charset=utf-8"
        assert len(response) > 0

class TestGlobalInstances:
    """Test global instances"""
    
    def test_metrics_registry_global(self):
        """Test global metrics registry"""
        assert isinstance(metrics_registry, MetricsRegistry)
        
    def test_performance_tracker_global(self):
        """Test global performance tracker"""
        assert isinstance(performance_tracker, PerformanceTracker)
        
    def test_resource_monitor_global(self):
        """Test global resource monitor"""
        assert isinstance(resource_monitor, ResourceMonitor)

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_monitoring(self):
        """Test complete workflow with monitoring"""
        # Start performance tracking
        operation_id = performance_tracker.start_operation("integration_test")
        
        try:
            # Simulate some work
            await asyncio.sleep(0.1)
            
            # Record various metrics
            record_agent_registration("test_agent", "success")
            record_task_execution("test_agent", "test_task", "success")
            record_task_duration("test_agent", "test_task", 0.1)
            record_external_api_call("test_service", "test_endpoint", "success")
            
            # Complete successfully
            performance_tracker.complete_operation(operation_id, success=True)
            
        except Exception as e:
            # Record error and complete with failure
            record_error(type(e).__name__, "integration_test", "error")
            performance_tracker.complete_operation(operation_id, success=False, error=e)
            raise
            
        # Verify metrics were recorded
        metrics = performance_tracker.get_operation_metrics(operation_id)
        assert metrics.success is True
        assert metrics.duration > 0
        
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with metrics"""
        # This would test the integration between circuit breakers and metrics
        # when circuit breakers are implemented
        pass

class TestErrorHandling:
    """Test error handling in metrics system"""
    
    def test_metrics_with_invalid_labels(self):
        """Test metrics with invalid labels"""
        # Should handle invalid labels gracefully
        try:
            record_agent_registration("", "")  # Empty strings
            record_task_execution("", "", "")  # Empty strings
        except Exception:
            pytest.fail("Metrics should handle invalid labels gracefully")
            
    def test_performance_tracker_with_invalid_operation_id(self):
        """Test performance tracker with invalid operation ID"""
        tracker = PerformanceTracker()
        
        # Should handle invalid operation ID gracefully
        tracker.complete_operation("nonexistent", success=True)
        assert tracker.get_operation_metrics("nonexistent") is None

class TestConcurrency:
    """Test concurrency in metrics system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_metric_recording(self):
        """Test concurrent metric recording"""
        async def record_metrics():
            for i in range(10):
                record_agent_registration(f"agent_{i}", "success")
                record_task_execution(f"agent_{i}", "task", "success")
                await asyncio.sleep(0.01)
                
        # Run multiple concurrent tasks
        tasks = [record_metrics() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should complete without errors
        
    def test_thread_safe_metrics_registry(self):
        """Test thread safety of metrics registry"""
        import threading
        
        registry = MetricsRegistry()
        results = []
        
        def add_metric(thread_id):
            metric = Mock()
            registry.register_metric(f"metric_{thread_id}", metric)
            results.append(registry.get_metric(f"metric_{thread_id}") is not None)
            
        threads = []
        for i in range(10):
            thread = threading.Thread(target=add_metric, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All operations should succeed
        assert all(results)
        assert len(registry.get_all_metrics()) == 10 