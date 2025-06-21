from agent import query
from agent import response
from examples.enhanced_unified_example import metrics
from examples.parallel_execution_example import tool_name
from performance_dashboard import metric
from tests.load_test import success

from src.collaboration.realtime_collaboration import session_id
from src.database.models import metadata
from src.database.models import metric_name
from src.database.models import metric_value
from src.database_extended import avg_latency
from src.database_extended import common_errors
from src.database_extended import cursor
from src.database_extended import cutoff_date
from src.database_extended import deleted
from src.database_extended import error_counts
from src.database_extended import failure_count
from src.database_extended import interaction
from src.database_extended import interactions_deleted
from src.database_extended import last_used
from src.database_extended import original_counts
from src.database_extended import performance
from src.database_extended import performances
from src.database_extended import row
from src.database_extended import success_count
from src.database_extended import success_rate
from src.database_extended import system_metrics_deleted
from src.database_extended import tool_metrics_deleted
from src.database_extended import tool_names
from src.database_extended import total_calls
from src.database_extended import total_deleted
from src.services.next_gen_integration import latency_ms
from src.tools_introspection import error
from src.tools_introspection import error_message
from src.utils.tools_introspection import field

from src.tools.base_tool import Tool
# TODO: Fix undefined variables: Any, Dict, List, Optional, avg_latency, common_errors, cursor, cutoff_date, dataclass, datetime, days, db_path, deleted, e, error, error_counts, error_message, failure_count, field, hours, i, interaction, interaction_type, interactions_deleted, json, last_used, latency_ms, logging, m, metadata, metric, metric_name, metric_value, metrics, original_counts, performance, performances, query, response, row, session_id, success, success_count, success_rate, system_metrics_deleted, timedelta, tool_metrics_deleted, tool_name, tool_names, total_calls, total_deleted, v, x

"""
from typing import Dict
from datetime import timedelta
# TODO: Fix undefined variables: avg_latency, common_errors, cursor, cutoff_date, days, db_path, deleted, e, error, error_counts, error_message, failure_count, hours, i, interaction, interaction_type, interactions_deleted, last_used, latency_ms, m, metadata, metric, metric_name, metric_value, metrics, original_counts, performance, performances, query, response, row, self, session_id, sqlite3, success, success_count, success_rate, system_metrics_deleted, tool_metrics_deleted, tool_name, tool_names, total_calls, total_deleted, v, x

Extended Database Module
Provides extended database functionality for tool performance tracking and metrics
"""

from typing import Optional
from dataclasses import field
from typing import Any
from typing import List

import logging

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class ToolMetric:
    """Represents a tool performance metric"""
    tool_name: str
    success: bool
    latency_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolPerformance:
    """Aggregated tool performance data"""
    tool_name: str
    total_calls: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    success_rate: float
    common_errors: List[str]
    last_used: Optional[datetime] = None

class ExtendedDatabase:
    """
    Extended database functionality for tracking tool performance,
    user interactions, and system metrics
    """

    def __init__(self, db_path: str = "extended_metrics.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the database with required tables"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()

            # Tool metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    latency_ms REAL NOT NULL,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            # User interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    query TEXT,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_metrics_tool_name ON tool_metrics(tool_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_metrics_timestamp ON tool_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_interactions_session ON user_interactions(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name)")

            self.connection.commit()
            logger.info("Extended database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize extended database: {e}")
            # Fallback to in-memory storage
            self.connection = None
            self._fallback_storage = {
                "tool_metrics": [],
                "user_interactions": [],
                "system_metrics": []
            }

    def update_tool_metric(self, tool_name: str, success: bool, latency_ms: float,
                          error_message: Optional[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update tool performance metrics

        Args:
            tool_name: Name of the tool
            success: Whether the tool call was successful
            latency_ms: Execution time in milliseconds
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO tool_metrics (tool_name, success, latency_ms, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    tool_name,
                    success,
                    latency_ms,
                    error_message,
                    json.dumps(metadata) if metadata else None
                ))
                self.connection.commit()
            else:
                # Fallback to in-memory storage
                metric = ToolMetric(
                    tool_name=tool_name,
                    success=success,
                    latency_ms=latency_ms,
                    error_message=error_message,
                    metadata=metadata or {}
                )
                self._fallback_storage["tool_metrics"].append(metric)

            logger.debug(f"Updated tool metric for {tool_name}: success={success}, latency={latency_ms}ms")
            return True

        except Exception as e:
            logger.error(f"Failed to update tool metric: {e}")
            return False

    def get_tool_performance(self, tool_name: str, days: int = 30) -> Optional[ToolPerformance]:
        """
        Get performance data for a specific tool

        Args:
            tool_name: Name of the tool
            days: Number of days to look back

        Returns:
            ToolPerformance object or None if not found
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cutoff_date = datetime.now() - timedelta(days=days)

                cursor.execute("""
                    SELECT
                        COUNT(*) as total_calls,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                        AVG(latency_ms) as avg_latency,
                        MAX(timestamp) as last_used
                    FROM tool_metrics
                    WHERE tool_name = ? AND timestamp >= ?
                """, (tool_name, cutoff_date.isoformat()))

                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return None

                total_calls, success_count, avg_latency, last_used = row
                failure_count = total_calls - success_count
                success_rate = success_count / total_calls if total_calls > 0 else 0.0

                # Get common errors
                cursor.execute("""
                    SELECT error_message, COUNT(*) as count
                    FROM tool_metrics
                    WHERE tool_name = ? AND success = 0 AND timestamp >= ?
                    GROUP BY error_message
                    ORDER BY count DESC
                    LIMIT 5
                """, (tool_name, cutoff_date.isoformat()))

                common_errors = [row[0] for row in cursor.fetchall() if row[0]]

                return ToolPerformance(
                    tool_name=tool_name,
                    total_calls=total_calls,
                    success_count=success_count,
                    failure_count=failure_count,
                    avg_latency_ms=avg_latency or 0.0,
                    success_rate=success_rate,
                    common_errors=common_errors,
                    last_used=datetime.fromisoformat(last_used) if last_used else None
                )
            else:
                # Fallback to in-memory storage
                cutoff_date = datetime.now() - timedelta(days=days)
                metrics = [
                    m for m in self._fallback_storage["tool_metrics"]
                    if m.tool_name == tool_name and m.timestamp >= cutoff_date
                ]

                if not metrics:
                    return None

                total_calls = len(metrics)
                success_count = sum(1 for m in metrics if m.success)
                failure_count = total_calls - success_count
                avg_latency = sum(m.latency_ms for m in metrics) / total_calls
                success_rate = success_count / total_calls

                # Get common errors
                error_counts = {}
                for metric in metrics:
                    if not metric.success and metric.error_message:
                        error_counts[metric.error_message] = error_counts.get(metric.error_message, 0) + 1

                common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                common_errors = [error for error, count in common_errors]

                last_used = max(m.timestamp for m in metrics) if metrics else None

                return ToolPerformance(
                    tool_name=tool_name,
                    total_calls=total_calls,
                    success_count=success_count,
                    failure_count=failure_count,
                    avg_latency_ms=avg_latency,
                    success_rate=success_rate,
                    common_errors=common_errors,
                    last_used=last_used
                )

        except Exception as e:
            logger.error(f"Failed to get tool performance for {tool_name}: {e}")
            return None

    def get_all_tool_performance(self, days: int = 30) -> List[ToolPerformance]:
        """
        Get performance data for all tools

        Args:
            days: Number of days to look back

        Returns:
            List of ToolPerformance objects
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cutoff_date = datetime.now() - timedelta(days=days)

                cursor.execute("""
                    SELECT DISTINCT tool_name FROM tool_metrics
                    WHERE timestamp >= ?
                """, (cutoff_date.isoformat(),))

                tool_names = [row[0] for row in cursor.fetchall()]

                performances = []
                for tool_name in tool_names:
                    performance = self.get_tool_performance(tool_name, days)
                    if performance:
                        performances.append(performance)

                return performances
            else:
                # Fallback to in-memory storage
                cutoff_date = datetime.now() - timedelta(days=days)
                tool_names = set(
                    m.tool_name for m in self._fallback_storage["tool_metrics"]
                    if m.timestamp >= cutoff_date
                )

                performances = []
                for tool_name in tool_names:
                    performance = self.get_tool_performance(tool_name, days)
                    if performance:
                        performances.append(performance)

                return performances

        except Exception as e:
            logger.error(f"Failed to get all tool performance: {e}")
            return []

    def add_user_interaction(self, session_id: str, interaction_type: str,
                           query: Optional[str] = None, response: Optional[str] = None,
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Add a user interaction record

        Args:
            session_id: User session ID
            interaction_type: Type of interaction
            query: User query
            response: System response
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO user_interactions (session_id, interaction_type, query, response, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    interaction_type,
                    query,
                    response,
                    json.dumps(metadata) if metadata else None
                ))
                self.connection.commit()
            else:
                # Fallback to in-memory storage
                interaction = {
                    "session_id": session_id,
                    "interaction_type": interaction_type,
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now(),
                    "metadata": metadata or {}
                }
                self._fallback_storage["user_interactions"].append(interaction)

            logger.debug(f"Added user interaction: {interaction_type} for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add user interaction: {e}")
            return False

    def add_system_metric(self, metric_name: str, metric_value: float,
                         metadata: Dict[str, Any] = None) -> bool:
        """
        Add a system metric

        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO system_metrics (metric_name, metric_value, metadata)
                    VALUES (?, ?, ?)
                """, (
                    metric_name,
                    metric_value,
                    json.dumps(metadata) if metadata else None
                ))
                self.connection.commit()
            else:
                # Fallback to in-memory storage
                metric = {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "timestamp": datetime.now(),
                    "metadata": metadata or {}
                }
                self._fallback_storage["system_metrics"].append(metric)

            logger.debug(f"Added system metric: {metric_name} = {metric_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to add system metric: {e}")
            return False

    def get_system_metrics(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get system metrics for a specific metric name

        Args:
            metric_name: Name of the metric
            hours: Number of hours to look back

        Returns:
            List of metric records
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cutoff_date = datetime.now() - timedelta(hours=hours)

                cursor.execute("""
                    SELECT metric_value, timestamp, metadata
                    FROM system_metrics
                    WHERE metric_name = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (metric_name, cutoff_date.isoformat()))

                return [
                    {
                        "value": row[0],
                        "timestamp": datetime.fromisoformat(row[1]),
                        "metadata": json.loads(row[2]) if row[2] else {}
                    }
                    for row in cursor.fetchall()
                ]
            else:
                # Fallback to in-memory storage
                cutoff_date = datetime.now() - timedelta(hours=hours)
                metrics = [
                    m for m in self._fallback_storage["system_metrics"]
                    if m["metric_name"] == metric_name and m["timestamp"] >= cutoff_date
                ]

                return sorted(metrics, key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get system metrics for {metric_name}: {e}")
            return []

    def cleanup_old_data(self, days: int = 90) -> int:
        """
        Clean up old data to prevent database bloat

        Args:
            days: Keep data newer than this many days

        Returns:
            Number of records deleted
        """
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cutoff_date = datetime.now() - timedelta(days=days)

                # Delete old tool metrics
                cursor.execute("DELETE FROM tool_metrics WHERE timestamp < ?", (cutoff_date.isoformat(),))
                tool_metrics_deleted = cursor.rowcount

                # Delete old user interactions
                cursor.execute("DELETE FROM user_interactions WHERE timestamp < ?", (cutoff_date.isoformat(),))
                interactions_deleted = cursor.rowcount

                # Delete old system metrics
                cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_date.isoformat(),))
                system_metrics_deleted = cursor.rowcount

                self.connection.commit()

                total_deleted = tool_metrics_deleted + interactions_deleted + system_metrics_deleted
                logger.info(f"Cleaned up {total_deleted} old records")
                return total_deleted
            else:
                # Fallback to in-memory storage
                cutoff_date = datetime.now() - timedelta(days=days)

                original_counts = {
                    "tool_metrics": len(self._fallback_storage["tool_metrics"]),
                    "user_interactions": len(self._fallback_storage["user_interactions"]),
                    "system_metrics": len(self._fallback_storage["system_metrics"])
                }

                self._fallback_storage["tool_metrics"] = [
                    m for m in self._fallback_storage["tool_metrics"]
                    if m.timestamp >= cutoff_date
                ]

                self._fallback_storage["user_interactions"] = [
                    i for i in self._fallback_storage["user_interactions"]
                    if i["timestamp"] >= cutoff_date
                ]

                self._fallback_storage["system_metrics"] = [
                    m for m in self._fallback_storage["system_metrics"]
                    if m["timestamp"] >= cutoff_date
                ]

                deleted = sum(original_counts.values()) - sum(len(v) for v in self._fallback_storage.values())
                logger.info(f"Cleaned up {deleted} old records from memory")
                return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

    def close(self) -> None:
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Extended database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()