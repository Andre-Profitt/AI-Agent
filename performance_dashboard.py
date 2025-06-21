#!/usr/bin/env python3
from performance_dashboard import alerts
from performance_dashboard import cpu_1min
from performance_dashboard import cpu_5min
from performance_dashboard import cpu_avg
from performance_dashboard import cpu_percent
from performance_dashboard import cpu_trend
from performance_dashboard import cutoff_time
from performance_dashboard import dashboard
from performance_dashboard import disk_percent
from performance_dashboard import error_rate
from performance_dashboard import memory_1min
from performance_dashboard import memory_5min
from performance_dashboard import memory_avg
from performance_dashboard import memory_mb
from performance_dashboard import memory_percent
from performance_dashboard import memory_trend
from performance_dashboard import metric
from performance_dashboard import network_recv
from performance_dashboard import network_sent
from performance_dashboard import recent
from performance_dashboard import response_time
from performance_dashboard import stats
from performance_dashboard import status_icon
from performance_dashboard import uptime
from setup_environment import value

from src.agents.enhanced_fsm import state
from src.database.models import metadata
from src.database.models import metric_type
from src.tools_introspection import name

from src.agents.advanced_agent_fsm import Agent
from datetime import timedelta
# TODO: Fix undefined variables: alert, alerts, breaker_stats, cpu_1min, cpu_5min, cpu_avg, cpu_percent, cpu_trend, cutoff_time, dashboard, disk_percent, e, error_rate, m, max_history, memory_1min, memory_5min, memory_avg, memory_mb, memory_percent, memory_trend, metadata, metric, metric_type, minutes, name, network_recv, network_sent, recent, response_time, self, state, stats, status_icon, update_interval, uptime, value

"""
Real-time Performance Monitoring Dashboard for AI Agent System
Provides live metrics, circuit breaker states, and performance analytics
"""

from typing import Optional
from typing import Any
from typing import List

import time

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os
import random

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock circuit breaker registry for testing
class MockCircuitBreakerRegistry:
    """Mock circuit breaker registry for testing"""
    def __init__(self):
        self.breakers = {}
        self._create_mock_breakers()

    def _create_mock_breakers(self):
        """Create mock circuit breakers"""
        self.breakers = {
            'database': {
                'state': 'closed',
                'total_requests': 1250,
                'successful_requests': 1245,
                'failed_requests': 5,
                'success_rate': 99.6
            },
            'api': {
                'state': 'closed',
                'total_requests': 890,
                'successful_requests': 885,
                'failed_requests': 5,
                'success_rate': 99.4
            },
            'cache': {
                'state': 'closed',
                'total_requests': 2100,
                'successful_requests': 2098,
                'failed_requests': 2,
                'success_rate': 99.9
            },
            'external_service': {
                'state': 'open',
                'total_requests': 45,
                'successful_requests': 20,
                'failed_requests': 25,
                'success_rate': 44.4
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers"""
        return self.breakers

# Mock circuit breaker registry
circuit_breaker_registry = MockCircuitBreakerRegistry()

class PerformanceMetrics:
    """Collect and store performance metrics"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: List[Dict[str, Any]] = []
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()

    def add_metric(self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a new metric"""
        metric = {
            'timestamp': time.time(),
            'type': metric_type,
            'value': value,
            'metadata': metadata or {}
        }

        self.metrics_history.append(metric)

        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

    def get_recent_metrics(self, metric_type: str, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent metrics of a specific type"""
        cutoff_time = time.time() - (minutes * 60)
        return [
            m for m in self.metrics_history
            if m['type'] == metric_type and m['timestamp'] > cutoff_time
        ]

    def get_average(self, metric_type: str, minutes: int = 5) -> float:
        """Get average of recent metrics"""
        recent = self.get_recent_metrics(metric_type, minutes)
        if not recent:
            return 0.0
        return sum(m['value'] for m in recent) / len(recent)

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        uptime = time.time() - self.start_time

        return {
            'uptime_seconds': uptime,
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': (self.error_count / max(self.request_count, 1)) * 100,
            'requests_per_second': self.request_count / max(uptime, 1),
            'metrics_count': len(self.metrics_history)
        }

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics = PerformanceMetrics()
        self.running = False
        self.monitoring_thread = None

        # Circuit breaker states
        self.circuit_breaker_states = {}

        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_ms': 2000.0,
            'error_rate_percent': 5.0
        }

    def start_monitoring(self):
        """Start the monitoring dashboard"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("🚀 Performance monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("🛑 Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_circuit_breaker_metrics()
                self._display_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(self.update_interval)

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # CPU usage (mock)
        cpu_percent = random.uniform(20, 75)
        self.metrics.add_metric('cpu_percent', cpu_percent)

        # Memory usage (mock)
        memory_percent = random.uniform(40, 85)
        memory_mb = random.uniform(200, 800)
        self.metrics.add_metric('memory_percent', memory_percent)
        self.metrics.add_metric('memory_mb', memory_mb)

        # Disk usage (mock)
        disk_percent = random.uniform(30, 70)
        self.metrics.add_metric('disk_percent', disk_percent)

        # Network I/O (mock)
        network_sent = random.uniform(1000000, 5000000)
        network_recv = random.uniform(2000000, 8000000)
        self.metrics.add_metric('network_bytes_sent', network_sent)
        self.metrics.add_metric('network_bytes_recv', network_recv)

        # Request metrics (mock)
        self.metrics.request_count += random.randint(1, 10)
        if random.random() < 0.02:  # 2% error rate
            self.metrics.error_count += 1

        # Response time (mock)
        response_time = random.uniform(50, 500)  # ms
        self.metrics.add_metric('response_time_ms', response_time)

    def _collect_circuit_breaker_metrics(self):
        """Collect circuit breaker metrics"""
        try:
            if hasattr(circuit_breaker_registry, 'get_stats'):
                stats = circuit_breaker_registry.get_stats()
                for name, breaker_stats in stats.items():
                    self.circuit_breaker_states[name] = {
                        'state': breaker_stats['state'],
                        'total_requests': breaker_stats['total_requests'],
                        'successful_requests': breaker_stats['successful_requests'],
                        'failed_requests': breaker_stats['failed_requests'],
                        'success_rate': breaker_stats['success_rate']
                    }
        except Exception as e:
            # Circuit breaker registry not available
            pass

    def _display_dashboard(self):
        """Display the performance dashboard"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")

        # Header
        print("=" * 80)
        print("🚀 AI AGENT SYSTEM - REAL-TIME PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # System Overview
        self._display_system_overview()
        print()

        # Performance Metrics
        self._display_performance_metrics()
        print()

        # Circuit Breaker Status
        self._display_circuit_breaker_status()
        print()

        # Alerts
        self._display_alerts()
        print()

        # Footer
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)

    def _display_system_overview(self):
        """Display system overview"""
        print("📊 SYSTEM OVERVIEW")
        print("-" * 40)

        stats = self.metrics.get_stats()

        # Uptime
        print(f"⏱️  Uptime: {stats['uptime_formatted']}")

        # Request stats
        print(f"📈 Total Requests: {stats['total_requests']:,}")
        print(f"📊 Requests/sec: {stats['requests_per_second']:.2f}")
        print(f"❌ Total Errors: {stats['total_errors']:,}")
        print(f"⚠️  Error Rate: {stats['error_rate']:.2f}%")

        # Current metrics
        cpu_avg = self.metrics.get_average('cpu_percent', 1)
        memory_avg = self.metrics.get_average('memory_percent', 1)

        print(f"🖥️  CPU Usage: {cpu_avg:.1f}%")
        print(f"💾 Memory Usage: {memory_avg:.1f}%")

    def _display_performance_metrics(self):
        """Display performance metrics"""
        print("📈 PERFORMANCE METRICS")
        print("-" * 40)

        # CPU trend
        cpu_1min = self.metrics.get_average('cpu_percent', 1)
        cpu_5min = self.metrics.get_average('cpu_percent', 5)
        cpu_trend = "↗️" if cpu_1min > cpu_5min else "↘️" if cpu_1min < cpu_5min else "➡️"
        print(f"🖥️  CPU: {cpu_1min:.1f}% (5min avg: {cpu_5min:.1f}%) {cpu_trend}")

        # Memory trend
        memory_1min = self.metrics.get_average('memory_percent', 1)
        memory_5min = self.metrics.get_average('memory_percent', 5)
        memory_trend = "↗️" if memory_1min > memory_5min else "↘️" if memory_1min < memory_5min else "➡️"
        print(f"💾 Memory: {memory_1min:.1f}% (5min avg: {memory_5min:.1f}%) {memory_trend}")

        # Memory in MB
        memory_mb = self.metrics.get_average('memory_mb', 1)
        print(f"💾 Memory Usage: {memory_mb:.1f} MB")

        # Disk usage
        disk_percent = self.metrics.get_average('disk_percent', 1)
        print(f"💿 Disk Usage: {disk_percent:.1f}%")

        # Network I/O
        network_sent = self.metrics.get_average('network_bytes_sent', 1)
        network_recv = self.metrics.get_average('network_bytes_recv', 1)
        print(f"🌐 Network Sent: {network_sent/1024/1024:.2f} MB")
        print(f"🌐 Network Recv: {network_recv/1024/1024:.2f} MB")

        # Response time
        response_time = self.metrics.get_average('response_time_ms', 1)
        print(f"⚡ Response Time: {response_time:.1f}ms")

    def _display_circuit_breaker_status(self):
        """Display circuit breaker status"""
        print("🔌 CIRCUIT BREAKER STATUS")
        print("-" * 40)

        if not self.circuit_breaker_states:
            print("⚠️  No circuit breakers available")
            return

        for name, state in self.circuit_breaker_states.items():
            status_icon = {
                'closed': '🟢',
                'open': '🔴',
                'half_open': '🟡'
            }.get(state['state'], '❓')

            print(f"{status_icon} {name}: {state['state'].upper()}")
            print(f"   Requests: {state['total_requests']:,} | "
                  f"Success: {state['successful_requests']:,} | "
                  f"Failed: {state['failed_requests']:,} | "
                  f"Rate: {state['success_rate']:.1f}%")

    def _display_alerts(self):
        """Display performance alerts"""
        print("🚨 ALERTS")
        print("-" * 40)

        alerts = []

        # CPU alert
        cpu_avg = self.metrics.get_average('cpu_percent', 1)
        if cpu_avg > self.thresholds['cpu_percent']:
            alerts.append(f"🔴 High CPU usage: {cpu_avg:.1f}%")

        # Memory alert
        memory_avg = self.metrics.get_average('memory_percent', 1)
        if memory_avg > self.thresholds['memory_percent']:
            alerts.append(f"🔴 High memory usage: {memory_avg:.1f}%")

        # Response time alert
        response_time = self.metrics.get_average('response_time_ms', 1)
        if response_time > self.thresholds['response_time_ms']:
            alerts.append(f"🔴 Slow response time: {response_time:.1f}ms")

        # Error rate alert
        error_rate = self.metrics.get_stats()['error_rate']
        if error_rate > self.thresholds['error_rate_percent']:
            alerts.append(f"🔴 High error rate: {error_rate:.2f}%")

        # Circuit breaker alerts
        for name, state in self.circuit_breaker_states.items():
            if state['state'] == 'open':
                alerts.append(f"🔴 Circuit breaker '{name}' is OPEN")
            elif state['success_rate'] < 95:
                alerts.append(f"🟡 Circuit breaker '{name}' has low success rate: {state['success_rate']:.1f}%")

        if alerts:
            for alert in alerts:
                print(alert)
        else:
            print("✅ All systems operational")

def main():
    """Start the performance dashboard"""
    print("🚀 Starting AI Agent System Performance Dashboard")
    print("This will provide real-time monitoring of system performance")
    print("Press Ctrl+C to stop\n")

    dashboard = PerformanceDashboard(update_interval=2.0)

    try:
        dashboard.start_monitoring()

        # Keep the main thread alive
        while dashboard.running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping performance dashboard...")
        dashboard.stop_monitoring()
        print("✅ Performance dashboard stopped")

if __name__ == "__main__":
    main()