# workflow_16s/utils/monitoring.py
"""
Performance monitoring utilities for workflow tracking.

Provides memory tracking, timing decorators, and performance reporting.
"""

import time
import psutil
import functools
from pathlib import Path
from typing import Dict, Optional, Callable
from contextlib import contextmanager

from workflow_16s.utils.logger import get_logger


class PerformanceMonitor:
    """Track memory usage and timing for workflow phases."""
    
    def __init__(self):
        self.phase_timings: Dict[str, float] = {}
        self.phase_memory: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.process = psutil.Process()
        self.logger = get_logger("workflow_16s")
        
    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        self.start_times[phase_name] = time.time()
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.phase_memory[f"{phase_name}_start"] = memory_mb
        self.logger.info(f"📊 [{phase_name}] Starting (Memory: {memory_mb:.1f} MB)")
    
    def end_phase(self, phase_name: str):
        """End timing a phase and log results."""
        if phase_name not in self.start_times:
            self.logger.warning(f"Phase '{phase_name}' was not started")
            return
            
        elapsed = time.time() - self.start_times[phase_name]
        self.phase_timings[phase_name] = elapsed
        
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.phase_memory[f"{phase_name}_end"] = memory_mb
        
        start_mem = self.phase_memory.get(f"{phase_name}_start", 0)
        memory_delta = memory_mb - start_mem
        
        self.logger.info(
            f"✅ [{phase_name}] Completed in {elapsed:.1f}s "
            f"(Memory: {memory_mb:.1f} MB, Δ{memory_delta:+.1f} MB)"
        )
        
        del self.start_times[phase_name]
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def log_memory(self, label: str = "Current"):
        """Log current memory usage."""
        memory_mb = self.get_current_memory()
        self.logger.debug(f"💾 {label} memory: {memory_mb:.1f} MB")
    
    def generate_summary(self, output_path: Optional[Path] = None) -> str:
        """Generate a summary report of all timings and memory usage."""
        if not self.phase_timings:
            return "No timing data available"
        
        # Sort by timing (slowest first)
        sorted_phases = sorted(
            self.phase_timings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total_time = sum(self.phase_timings.values())
        
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nTotal Runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
        lines.append(f"\nPhase Breakdown:")
        lines.append("-" * 80)
        lines.append(f"{'Phase':<40} {'Time (s)':<12} {'% Total':<10} {'Memory Δ':<10}")
        lines.append("-" * 80)
        
        for phase, elapsed in sorted_phases:
            pct = (elapsed / total_time) * 100
            start_mem = self.phase_memory.get(f"{phase}_start", 0)
            end_mem = self.phase_memory.get(f"{phase}_end", 0)
            mem_delta = end_mem - start_mem
            
            lines.append(
                f"{phase:<40} {elapsed:>10.1f}s  {pct:>7.1f}%  "
                f"{mem_delta:>+8.1f} MB"
            )
        
        lines.append("=" * 80)
        
        # Peak memory
        all_memory_values = [v for k, v in self.phase_memory.items() if k.endswith('_end')]
        if all_memory_values:
            peak_memory = max(all_memory_values)
            lines.append(f"\nPeak Memory Usage: {peak_memory:.1f} MB")
        
        summary = "\n".join(lines)
        
        if output_path:
            output_path.write_text(summary)
            self.logger.info(f"Performance summary saved to: {output_path}")
        
        return summary


# Global monitor instance
_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _monitor


@contextmanager
def track_phase(phase_name: str):
    """Context manager for tracking a phase."""
    monitor = get_monitor()
    monitor.start_phase(phase_name)
    try:
        yield monitor
    finally:
        monitor.end_phase(phase_name)


def timed_phase(phase_name: Optional[str] = None):
    """Decorator for timing functions as phases."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = phase_name or func.__name__
            with track_phase(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_memory_usage(label: str = ""):
    """Log current memory usage."""
    monitor = get_monitor()
    monitor.log_memory(label or "Current")
