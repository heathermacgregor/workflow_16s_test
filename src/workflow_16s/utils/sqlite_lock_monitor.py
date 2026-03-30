"""
SQLite Database Lock Monitoring and Error Handling Module

This module provides comprehensive monitoring for SQLite database lock errors,
including metrics collection, structured logging, and retry strategies with
configurable thresholds for production alerting.

Features:
- Lock error detection and recovery
- Metrics collection (lock errors, retry attempts, wait times)
- Structured JSON logging for log aggregation
- Configurable retry strategies with exponential backoff
- Alert threshold management
- Statistical tracking for operational insights
"""

import json
import logging
import sqlite3
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple


class LockErrorType(Enum):
    """Types of database lock errors."""
    DATABASE_LOCKED = "database_is_locked"
    DISK_FULL = "disk_io_error"
    BUSY = "database_busy"
    CONSTRAINT_VIOLATION = "constraint_violation"
    OTHER = "other"


@dataclass
class LockErrorEvent:
    """Structured representation of a database lock error event."""
    timestamp: str
    operation_type: str
    error_type: LockErrorType
    error_message: str
    retry_count: int
    wait_time_seconds: float
    resolution_method: str
    success: bool
    additional_context: Dict[str, Any]

    def to_json(self) -> str:
        """Convert to JSON for structured logging."""
        data = asdict(self)
        data['error_type'] = data['error_type'].value
        data['timestamp'] = data['timestamp']
        return json.dumps(data, default=str)


@dataclass
class SQLiteLockConfig:
    """Configuration for SQLite lock handling."""
    timeout_seconds: int = 10
    max_retries: int = 3
    initial_retry_delay_ms: int = 100
    max_retry_delay_ms: int = 5000
    enable_wal_mode: bool = True
    busy_timeout_ms: int = 10000
    warn_threshold_pct: float = 5.0
    error_threshold_pct: float = 10.0
    max_operation_time_seconds: float = 5.0


class SQLiteLockMonitor:
    """
    Comprehensive monitoring system for SQLite database lock errors.

    Tracks lock errors, retry attempts, and provides metrics for alerting.
    Designed to work with production environments and log aggregation systems.
    """

    def __init__(self, name: str = "sqlite_lock_monitor", config: Optional[SQLiteLockConfig] = None):
        """Initialize the lock monitor.

        Args:
            name: Logger name for this monitor
            config: Configuration object (uses defaults if None)
        """
        self.name = name
        self.config = config or SQLiteLockConfig()
        self.logger = logging.getLogger(name)
        self._lock = Lock()

        # Initialize metrics
        self.metrics = {
            'total_operations': 0,
            'lock_errors_caught': 0,
            'successful_retries': 0,
            'max_retries_exceeded': 0,
            'lock_errors_by_type': {},
            'total_wait_time_seconds': 0.0,
            'max_wait_time_seconds': 0.0,
            'min_wait_time_seconds': float('inf'),
            'error_events': [],  # Keep recent events
        }
        self.max_error_events_kept = 1000

    def get_lock_error_type(self, error_message: str) -> LockErrorType:
        """Determine the type of lock error from the error message."""
        msg_lower = error_message.lower()

        if 'database is locked' in msg_lower or 'locked' in msg_lower:
            return LockErrorType.DATABASE_LOCKED
        elif 'disk' in msg_lower or 'i/o' in msg_lower:
            return LockErrorType.DISK_FULL
        elif 'busy' in msg_lower:
            return LockErrorType.BUSY
        elif 'constraint' in msg_lower or 'unique' in msg_lower:
            return LockErrorType.CONSTRAINT_VIOLATION
        else:
            return LockErrorType.OTHER

    def record_lock_error(self, event: LockErrorEvent):
        """Record a lock error event with metrics."""
        with self._lock:
            self.metrics['lock_errors_caught'] += 1
            error_type = event.error_type.value

            if error_type not in self.metrics['lock_errors_by_type']:
                self.metrics['lock_errors_by_type'][error_type] = 0
            self.metrics['lock_errors_by_type'][error_type] += 1

            if event.retry_count < self.config.max_retries:
                self.metrics['successful_retries'] += 1
            else:
                self.metrics['max_retries_exceeded'] += 1

            # Update wait time statistics
            self.metrics['total_wait_time_seconds'] += event.wait_time_seconds
            self.metrics['max_wait_time_seconds'] = max(
                self.metrics['max_wait_time_seconds'],
                event.wait_time_seconds
            )
            if event.wait_time_seconds > 0:
                self.metrics['min_wait_time_seconds'] = min(
                    self.metrics['min_wait_time_seconds'],
                    event.wait_time_seconds
                )

            # Store event (keep most recent)
            self.metrics['error_events'].append(event)
            if len(self.metrics['error_events']) > self.max_error_events_kept:
                self.metrics['error_events'] = self.metrics['error_events'][-self.max_error_events_kept:]

            # Log structured event
            self.logger.warning(f"SQLite lock error: {event.to_json()}")

    def check_alert_thresholds(self) -> Tuple[bool, bool, Optional[str]]:
        """
        Check if alert thresholds have been exceeded.

        Returns:
            Tuple of (warn_threshold_exceeded, error_threshold_exceeded, message)
        """
        with self._lock:
            if self.metrics['total_operations'] == 0:
                return False, False, None

            lock_error_rate = (
                self.metrics['lock_errors_caught'] / self.metrics['total_operations'] * 100
            )

            messages = []
            warn_exceeded = False
            error_exceeded = False

            if lock_error_rate > self.config.error_threshold_pct:
                error_exceeded = True
                messages.append(
                    f"ERROR: Lock error rate {lock_error_rate:.2f}% exceeds threshold "
                    f"{self.config.error_threshold_pct}%"
                )
            elif lock_error_rate > self.config.warn_threshold_pct:
                warn_exceeded = True
                messages.append(
                    f"WARN: Lock error rate {lock_error_rate:.2f}% exceeds threshold "
                    f"{self.config.warn_threshold_pct}%"
                )

            # Check max wait time
            if self.metrics['max_wait_time_seconds'] > self.config.max_operation_time_seconds:
                error_exceeded = True
                messages.append(
                    f"ERROR: Max lock wait time {self.metrics['max_wait_time_seconds']:.2f}s "
                    f"exceeds limit {self.config.max_operation_time_seconds}s"
                )

            message = " | ".join(messages) if messages else None
            return warn_exceeded, error_exceeded, message

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current metrics."""
        with self._lock:
            if self.metrics['total_operations'] == 0:
                lock_error_rate = 0.0
            else:
                lock_error_rate = (
                    self.metrics['lock_errors_caught'] / self.metrics['total_operations'] * 100
                )

            avg_wait_time = (
                self.metrics['total_wait_time_seconds'] / self.metrics['lock_errors_caught']
                if self.metrics['lock_errors_caught'] > 0
                else 0.0
            )

            min_wait = (
                self.metrics['min_wait_time_seconds']
                if self.metrics['min_wait_time_seconds'] != float('inf')
                else 0.0
            )

            return {
                'total_operations': self.metrics['total_operations'],
                'lock_errors_caught': self.metrics['lock_errors_caught'],
                'lock_error_rate_pct': round(lock_error_rate, 2),
                'successful_retries': self.metrics['successful_retries'],
                'max_retries_exceeded': self.metrics['max_retries_exceeded'],
                'lock_errors_by_type': self.metrics['lock_errors_by_type'].copy(),
                'average_wait_time_seconds': round(avg_wait_time, 3),
                'max_wait_time_seconds': round(self.metrics['max_wait_time_seconds'], 3),
                'min_wait_time_seconds': round(min_wait, 3),
                'total_wait_time_seconds': round(self.metrics['total_wait_time_seconds'], 2),
                'config': asdict(self.config),
            }

    def reset_metrics(self):
        """Reset metrics (useful for testing or periodic resets)."""
        with self._lock:
            self.metrics = {
                'total_operations': 0,
                'lock_errors_caught': 0,
                'successful_retries': 0,
                'max_retries_exceeded': 0,
                'lock_errors_by_type': {},
                'total_wait_time_seconds': 0.0,
                'max_wait_time_seconds': 0.0,
                'min_wait_time_seconds': float('inf'),
                'error_events': [],
            }


# Global monitor instance
_global_monitor = SQLiteLockMonitor("workflow_16s.sqlite")


def get_global_monitor() -> SQLiteLockMonitor:
    """Get the global SQLite lock monitor instance."""
    return _global_monitor


def execute_with_lock_handling(
    db_path: Path,
    operation: Callable[[sqlite3.Connection], Any],
    operation_type: str = "unknown",
    config: Optional[SQLiteLockConfig] = None,
    monitor: Optional[SQLiteLockMonitor] = None,
) -> Any:
    """
    Execute a database operation with lock error handling and monitoring.

    Args:
        db_path: Path to the SQLite database
        operation: Callable that takes a connection and performs the operation
        operation_type: Description of the operation (for logging)
        config: SQLite lock configuration
        monitor: Lock monitor instance (uses global if None)

    Returns:
        Result from the operation

    Raises:
        sqlite3.OperationalError: If lock persists after all retries
    """
    config = config or SQLiteLockConfig()
    monitor = monitor or _global_monitor

    monitor.metrics['total_operations'] += 1

    start_time = time.time()
    last_error = None

    for attempt in range(config.max_retries + 1):
        try:
            conn = sqlite3.connect(
                str(db_path),
                timeout=config.timeout_seconds,
                check_same_thread=False
            )

            if config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={config.busy_timeout_ms}")

            result = operation(conn)
            conn.close()

            return result

        except sqlite3.OperationalError as e:
            last_error = e
            error_type = monitor.get_lock_error_type(str(e))
            wait_time = time.time() - start_time

            if attempt < config.max_retries:
                # Calculate exponential backoff with jitter
                delay_ms = min(
                    config.initial_retry_delay_ms * (2 ** attempt),
                    config.max_retry_delay_ms
                )
                delay_seconds = delay_ms / 1000.0

                # Log and retry
                monitor.logger.debug(
                    f"Database lock detected for {operation_type} "
                    f"(attempt {attempt + 1}/{config.max_retries + 1}). "
                    f"Retrying in {delay_seconds:.3f}s. Error: {str(e)[:100]}"
                )

                time.sleep(delay_seconds)
                continue
            else:
                # All retries exhausted
                event = LockErrorEvent(
                    timestamp=datetime.now().isoformat(),
                    operation_type=operation_type,
                    error_type=error_type,
                    error_message=str(e),
                    retry_count=attempt,
                    wait_time_seconds=wait_time,
                    resolution_method="none",
                    success=False,
                    additional_context={'db_path': str(db_path)}
                )

                monitor.record_lock_error(event)

                # Check alert thresholds
                warn, error, msg = monitor.check_alert_thresholds()
                if msg:
                    if error:
                        monitor.logger.error(msg)
                    else:
                        monitor.logger.warning(msg)

                raise last_error

        except Exception as e:
            monitor.logger.error(
                f"Unexpected error during {operation_type}: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    # Should not reach here, but just in case
    raise last_error


def sqlite_safe_operation(
    db_path: Path,
    sql_query: str,
    params: tuple = (),
    operation_type: str = "query",
    config: Optional[SQLiteLockConfig] = None,
) -> List[tuple]:
    """
    Execute a single SQL query with lock handling.

    Args:
        db_path: Path to the SQLite database
        sql_query: SQL query to execute
        params: Query parameters
        operation_type: Description of the operation
        config: SQLite lock configuration

    Returns:
        List of result tuples
    """
    def _operation(conn: sqlite3.Connection) -> List[tuple]:
        cursor = conn.cursor()
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        conn.commit()
        return results

    return execute_with_lock_handling(
        db_path, _operation, operation_type, config
    )


def sqlite_safe_write(
    db_path: Path,
    sql_query: str,
    params: tuple = (),
    operation_type: str = "write",
    config: Optional[SQLiteLockConfig] = None,
) -> int:
    """
    Execute a single SQL write operation with lock handling.

    Args:
        db_path: Path to the SQLite database
        sql_query: SQL query to execute
        params: Query parameters
        operation_type: Description of the operation
        config: SQLite lock configuration

    Returns:
        Number of rows affected
    """
    def _operation(conn: sqlite3.Connection) -> int:
        cursor = conn.cursor()
        cursor.execute(sql_query, params)
        conn.commit()
        return cursor.rowcount

    return execute_with_lock_handling(
        db_path, _operation, operation_type, config
    )
