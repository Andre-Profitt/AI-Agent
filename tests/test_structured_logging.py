"""
Test structured logging implementation
"""

import pytest
import ast
import os
from pathlib import Path
from unittest.mock import patch
from src.utils.structured_logging import get_structured_logger, StructuredLogger

class TestStructuredLogging:
    """Test structured logging implementation"""
    
    def test_no_print_statements_in_src(self):
        """Scan for print statements in production code"""
        src_dir = Path(__file__).parent.parent / 'src'
        files_with_prints = []
        
        for py_file in src_dir.rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, 'id') and node.func.id == 'print':
                                files_with_prints.append(str(py_file))
                                break
            except:
                pass
        
        assert len(files_with_prints) == 0, f"Found print statements in: {files_with_prints}"
    
    def test_no_f_string_logging(self):
        """Verify no f-string logging in production code"""
        src_dir = Path(__file__).parent.parent / 'src'
        files_with_f_strings = []
        
        for py_file in src_dir.rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    # Look for f-string logging patterns
                    if 'logger.info(f"' in content or 'logger.error(f"' in content:
                        files_with_f_strings.append(str(py_file))
            except:
                pass
        
        assert len(files_with_f_strings) == 0, f"Found f-string logging in: {files_with_f_strings}"
    
    def test_structured_logger_creation(self):
        """Test structured logger creation"""
        logger = get_structured_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
    
    def test_structured_logger_with_context(self):
        """Test structured logger with context"""
        logger = StructuredLogger("test_context")
        
        # Test context logging
        with patch('logging.Logger.info') as mock_info:
            logger.info("Context message", context_id="456", operation="test")
            mock_info.assert_called_once()
    
    def test_error_logging_with_exception(self):
        """Test error logging with exception details"""
        logger = get_structured_logger("test_error")
        
        test_exception = ValueError("Test error")
        
        with patch('logging.Logger.error') as mock_error:
            logger.error("Error occurred", error=test_exception, component="test")
            mock_error.assert_called_once()
    
    def test_logging_performance(self):
        """Test logging performance"""
        logger = get_structured_logger("test_performance")
        
        import time
        start_time = time.time()
        
        # Log many messages
        for i in range(1000):
            logger.info("Performance test", iteration=i)
        
        duration = time.time() - start_time
        
        # Should complete quickly
        assert duration < 1.0, f"Logging too slow: {duration}s"
    
    def test_logging_thread_safety(self):
        """Test logging thread safety"""
        import threading
        import queue
        
        logger = get_structured_logger("test_threading")
        results = queue.Queue()
        
        def log_worker(worker_id):
            try:
                for i in range(100):
                    logger.info("Thread test", worker_id=worker_id, iteration=i)
                results.put(f"worker_{worker_id}_complete")
            except Exception as e:
                results.put(f"worker_{worker_id}_error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        completed = 0
        while not results.empty():
            result = results.get()
            if "complete" in result:
                completed += 1
        
        assert completed == 5, "Not all threads completed successfully"
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        logger = get_structured_logger("test_config")
        
        # Verify logger has proper handlers
        assert len(logger.handlers) > 0
        
        # Verify logger level is appropriate
        assert logger.level <= 20  # INFO level or lower
    
    def test_logging_integration(self):
        """Test logging integration with other components"""
        from src.services.integration_hub import IntegrationHub
        
        # Test that integration hub uses structured logging
        with patch('src.utils.structured_logging.get_structured_logger') as mock_logger:
            hub = IntegrationHub()
            mock_logger.assert_called()
    
    def test_logging_error_handling(self):
        """Test logging error handling"""
        logger = get_structured_logger("test_error_handling")
        
        # Test logging with invalid data
        try:
            logger.info("Test with invalid data", invalid_key=object())
            # Should not raise exception
        except Exception as e:
            pytest.fail(f"Logging should handle invalid data gracefully: {e}")
    
    def test_logging_format_consistency(self):
        """Test logging format consistency"""
        logger = get_structured_logger("test_format")
        
        # Test different log levels
        with patch('logging.Logger.info') as mock_info:
            with patch('logging.Logger.error') as mock_error:
                with patch('logging.Logger.warning') as mock_warning:
                    
                    logger.info("Info message", level="info")
                    logger.error("Error message", level="error")
                    logger.warning("Warning message", level="warning")
                    
                    # Verify all were called
                    assert mock_info.called
                    assert mock_error.called
                    assert mock_warning.called 