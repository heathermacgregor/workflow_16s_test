"""
TelemetryCollector: Thread-safe workflow telemetry event collection.

Optimized for minimal memory overhead (<1MB) with efficient deque-based storage.
Events are immutable namedtuples with standardized fields for dashboard consumption.
"""

import time
import threading
from collections import deque, namedtuple
from typing import Any, Dict, List, Optional

# Define immutable telemetry event structure (minimal memory footprint)
TelemetryEvent = namedtuple(
    'TelemetryEvent',
    ['timestamp', 'event_type', 'phase', 'message', 'metrics']
)

class TelemetryCollector:
    """
    Thread-safe telemetry event collection for workflow orchestration.
    
    Event Types:
    - 'step_start': Phase beginning (e.g., ingestion, preprocessing)
    - 'step_end': Phase completion with metrics (duration, counts, memory)
    - 'file_written': Output file created (size, path)
    - 'analysis_module': Individual analysis module status
    - 'error': Error event
    - 'info': General information
    
    Stores last 200 events to minimize memory (~1-2 MB with typical payloads).
    """
    
    # Memory-efficient size limit
    MAX_EVENTS = 200
    
    def __init__(self):
        self._events: deque = deque(maxlen=self.MAX_EVENTS)
        self._lock = threading.Lock()
        self._phase_times: Dict[str, float] = {}
        self._phase_glock = threading.Lock()
        self._disabled_modules: set = set()
    
    def start_phase(self, phase_name: str) -> None:
        """Mark the start of a workflow phase."""
        with self._phase_glock:
            self._phase_times[phase_name] = time.time()
        
        self.emit(
            event_type='step_start',
            phase=phase_name,
            message=f"Starting {phase_name}",
            metrics={'timestamp': time.time()}
        )
    
    def end_phase(self, phase_name: str, metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Mark the end of a workflow phase. Returns duration in seconds.
        
        Args:
            phase_name: Name of the phase
            metrics: Optional dict with {n_samples, n_features, memory_gb, etc.}
        
        Returns:
            Duration in seconds
        """
        with self._phase_glock:
            start_time = self._phase_times.pop(phase_name, None)
        
        if start_time is None:
            duration = 0.0
        else:
            duration = time.time() - start_time
        
        event_metrics = metrics or {}
        event_metrics['duration_seconds'] = duration
        
        self.emit(
            event_type='step_end',
            phase=phase_name,
            message=f"Completed {phase_name} in {duration:.2f}s",
            metrics=event_metrics
        )
        
        return duration
    
    def emit(
        self, 
        event_type: str, 
        phase: str = 'general', 
        message: str = '', 
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit a telemetry event (thread-safe).
        
        Args:
            event_type: Type of event (step_start, step_end, error, etc.)
            phase: Phase/module name
            message: Human-readable message
            metrics: Optional dict with structured data
        """
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type=event_type,
            phase=phase,
            message=message,
            metrics=metrics or {}
        )
        
        with self._lock:
            self._events.append(event)
    
    def get_events(self, limit: int = 50) -> List[TelemetryEvent]:
        """
        Retrieve the last N events (thread-safe).
        
        Args:
            limit: Maximum number of events to return
        
        Returns:
            List of TelemetryEvent namedtuples (newest last)
        """
        with self._lock:
            return list(self._events)[-limit:]
    
    def get_latest_event(self) -> Optional[TelemetryEvent]:
        """Get the most recent event, or None if empty."""
        with self._lock:
            return self._events[-1] if self._events else None
    
    def mark_module_disabled(self, module_name: str) -> None:
        """Mark a module as disabled in configuration."""
        with self._lock:
            self._disabled_modules.add(module_name)
    
    def get_disabled_modules(self) -> set:
        """Get the set of disabled modules."""
        with self._lock:
            return set(self._disabled_modules)
    
    def get_phase_status(self) -> Dict[str, str]:
        """
        Get current phase activity status for dashboard display.
        
        Returns:
            Dict mapping phase names to status ('running', 'completed', 'pending', 'disabled', 'failed')
        """
        with self._lock:
            status = {}
            seen_phases = set()
            
            # First, mark all disabled modules
            for module in self._disabled_modules:
                status[module] = 'disabled'
                seen_phases.add(module)
            
            # Iterate events in reverse to find most recent status per phase
            for event in reversed(self._events):
                if event.phase not in seen_phases:
                    if event.event_type == 'step_start':
                        status[event.phase] = 'running'
                    elif event.event_type == 'step_end':
                        status[event.phase] = 'completed'
                    elif event.event_type == 'analysis_module':
                        # Handle module-level completion events (functional_traits, ecotype_stratification, etc.)
                        module_status = event.metrics.get('status', 'unknown')
                        status[event.phase] = 'completed' if module_status == 'success' else 'failed'
                    elif event.event_type == 'error':
                        status[event.phase] = 'failed'
                    
                    seen_phases.add(event.phase)
            
            return status
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a workflow execution summary from telemetry.
        
        Returns:
            Dict with overall stats (total_duration, n_phases, errors, etc.)
        """
        with self._lock:
            if not self._events:
                return {'events_collected': 0}
            
            events_list = list(self._events)
            start_time = events_list[0].timestamp
            end_time = events_list[-1].timestamp
            
            # Count event types
            event_counts = {}
            total_duration = 0.0
            errors = []
            
            for event in events_list:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
                
                if event.event_type == 'step_end':
                    total_duration += event.metrics.get('duration_seconds', 0)
                elif event.event_type == 'error':
                    errors.append(event.message)
            
            return {
                'events_collected': len(events_list),
                'total_runtime_seconds': end_time - start_time,
                'cumulative_phase_time_seconds': total_duration,
                'event_types': event_counts,
                'errors': errors,
                'phases_completed': sum(1 for e in events_list if e.event_type == 'step_end')
            }
    
    def clear(self) -> None:
        """Clear all stored events."""
        with self._lock:
            self._events.clear()


# Module-level singleton for easy access
_global_telemetry: Optional[TelemetryCollector] = None

def get_global_telemetry() -> TelemetryCollector:
    """Get or create the global telemetry collector."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = TelemetryCollector()
    return _global_telemetry
