"""
Real-Time Optuna Trial Monitor
================================

Captures Optuna optimization progress in real-time and streams it to:
1. EnhancedDashboardMonitor (Rich TUI panel with live updates)
2. Telemetry system (for metrics collection)
3. Logger (for CLI feedback)
4. JSON stream (for post-analysis)

Enables users to watch hyperparameter optimization progress in real-time
through the dashboard while the search is running.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict

import optuna
from optuna.trial import Trial, TrialState
import pandas as pd

from workflow_16s.utils.logger import get_logger
from workflow_16s.utils.telemetry import TelemetryCollector


class OptunaRealtimeCallback:
    """
    Optuna callback that captures trial progress in real-time.
    
    Integration points:
    - Logs to logger (CLI + log file)
    - Emits events to telemetry (for dashboard ingestion)
    - Writes to JSON stream (for post-analysis + dashboards)
    - Calls external callbacks (if provided)
    """
    
    def __init__(
        self, 
        output_dir: Path,
        telemetry: Optional[TelemetryCollector] = None,
        trial_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        enable_json_stream: bool = True
    ):
        """
        Args:
            output_dir: Directory to write trial data files
            telemetry: Optional TelemetryCollector for dashboard integration
            trial_callback: Optional function to call after each trial
            enable_json_stream: If True, write streaming JSON file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.telemetry = telemetry
        self.trial_data_file = self.output_dir / "optuna_trials_realtime.json"
        self.summary_file = self.output_dir / "optuna_summary.json"
        self.stream_file = self.output_dir / ".optuna_stream"
        
        self.logger = get_logger("workflow_16s.optuna")
        self.trial_callback = trial_callback
        self.enable_json_stream = enable_json_stream
        
        # In-memory cache
        self.trials_cache: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # Metadata
        self.best_value = None
        self.best_trial_number = None
        self.best_params = {}
        self.start_time = datetime.now()
        self.target_name = "unknown"
        self.task_type = "ClassificationUnknown"
        
        self.logger.info(f"🎯 Optuna Real-Time Monitor initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Telemetry integration: {telemetry is not None}")
    
    def __call__(self, study: optuna.Study, trial: Trial) -> None:
        """
        Called after each trial completes.
        Optuna expects: callback(study, trial) signature.
        
        This method:
        1. Extracts trial data
        2. Updates internal state
        3. Logs to CLI
        4. Writes to JSON
        5. Emits telemetry event (dashboard)
        6. Calls user callback
        """
        with self.lock:
            # Extract and process trial data
            trial_dict = self._extract_trial_data(study, trial)
            
            # Update cache
            self.trials_cache[trial.number] = trial_dict
            
            # Log to CLI
            self._log_trial_progress(trial_dict, study)
            
            # Write to files (JSON stream)
            if self.enable_json_stream:
                self._write_trial_data()
            
            # Emit to telemetry → Dashboard
            self._emit_telemetry_event(trial_dict, study)
            
            # Call user callback if provided
            if self.trial_callback:
                try:
                    self.trial_callback(trial_dict)
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Trial callback failed: {e}")
    
    def _extract_trial_data(self, study: optuna.Study, trial: Trial) -> Dict[str, Any]:
        """Extract all relevant data from a trial."""
        
        trial_dict = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'value': trial.value if trial.value is not None else None,
            'timestamp': datetime.now().isoformat(),
            'params': dict(trial.params),
            'user_attrs': dict(trial.user_attrs),
            'direction': str(study.direction),
        }
        
        # Update best tracking
        if study.best_trial is not None:
            self.best_value = study.best_trial.value
            self.best_trial_number = study.best_trial.number
            self.best_params = dict(study.best_trial.params)
            
            trial_dict['best_value'] = self.best_value
            trial_dict['best_trial_number'] = self.best_trial_number
            trial_dict['best_params'] = self.best_params
        
        # Study metadata
        trial_dict['total_trials'] = len(study.trials)
        trial_dict['completed_trials'] = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        trial_dict['failed_trials'] = len([t for t in study.trials if t.state == TrialState.FAIL])
        
        # Progress percentage
        if trial_dict['total_trials'] > 0:
            trial_dict['progress_pct'] = (trial_dict['completed_trials'] / trial_dict['total_trials']) * 100
        
        return trial_dict
    
    def _log_trial_progress(self, trial_dict: Dict[str, Any], study: optuna.Study) -> None:
        """Log trial progress to CLI."""
        
        trial_num = trial_dict['trial_number']
        total = trial_dict['total_trials']
        completed = trial_dict['completed_trials']
        failed = trial_dict['failed_trials']
        value = trial_dict['value']
        progress = trial_dict.get('progress_pct', 0)
        
        # Format value - handle None gracefully
        value_str = f"{value:.4f}" if value is not None else "FAILED"
        
        # Build log message
        direction = "↑" if study.direction == optuna.study.StudyDirection.MAXIMIZE else "↓"
        status = f"[{completed:3d}/{total}]"
        
        # Best so far
        best_str = ""
        if self.best_value is not None:
            best_value_fmt = f"{self.best_value:.4f}"
            best_str = f" | Best: {direction} {best_value_fmt} (Trial #{self.best_trial_number})"
        
        # Log
        self.logger.info(
            f"   Optuna {status} Trial #{trial_num}: {direction} {value_str} "
            f"({progress:.1f}%){best_str}"
        )
        
        # Log params (abbreviated) at debug level
        if trial_dict['params']:
            params = trial_dict['params']
            param_items = list(params.items())[:3]
            param_str = ", ".join([f"{k}={v}" for k, v in param_items])
            if len(params) > 3:
                param_str += f", +{len(params) - 3} more"
            self.logger.debug(f"      Parameters: {param_str}")
    
    def _emit_telemetry_event(self, trial_dict: Dict[str, Any], study: optuna.Study) -> None:
        """Emit event to telemetry system for dashboard ingestion."""
        
        if self.telemetry is None:
            return
        
        try:
            # Handle None value gracefully
            trial_value = trial_dict['value'] if trial_dict['value'] is not None else float('-inf')
            value_str = f"{trial_value:.4f}" if trial_value != float('-inf') else "FAILED"
            
            # Format event for dashboard
            event_message = (
                f"Optuna Trial #{trial_dict['trial_number']}: "
                f"Score={value_str}, "
                f"Progress: {trial_dict['completed_trials']}/{trial_dict['total_trials']} "
                f"({trial_dict.get('progress_pct', 0):.1f}%)"
            )
            
            if self.best_value is not None:
                event_message += f", Best: {self.best_value:.4f}"
            
            # Emit as a telemetry event
            self.telemetry.emit(
                event_type='optuna_trial',
                phase='hyperparameter_search',
                message=event_message,
                metrics={
                    'trial_number': trial_dict['trial_number'],
                    'trial_value': trial_value,
                    'best_value': self.best_value,
                    'completed_trials': trial_dict['completed_trials'],
                    'total_trials': trial_dict['total_trials'],
                    'progress_pct': trial_dict.get('progress_pct', 0),
                    'failed_trials': trial_dict['failed_trials']
                }
            )
        except Exception as e:
            self.logger.debug(f"Failed to emit telemetry: {e}")
    
    def _write_trial_data(self) -> None:
        """Write all trial data to JSON file."""
        
        try:
            # Prepare data
            trials_list = list(self.trials_cache.values())
            trials_list = sorted(trials_list, key=lambda x: x['trial_number'])
            
            # Summary stats
            completed = len([t for t in trials_list if t['state'] == 'COMPLETE'])
            total = len(trials_list)
            failed = len([t for t in trials_list if t['state'] == 'FAIL'])
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': (datetime.now() - self.start_time).total_seconds(),
                'total_trials': total,
                'completed_trials': completed,
                'failed_trials': failed,
                'best_value': self.best_value,
                'best_trial_number': self.best_trial_number,
                'best_params': self.best_params,
                'trials': trials_list
            }
            
            # Write trials file
            with open(self.trial_data_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            # Write summary file
            summary = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': output_data['elapsed_seconds'],
                'total_trials': total,
                'completed_trials': completed,
                'failed_trials': failed,
                'success_rate': (completed / total * 100) if total > 0 else 0,
                'best_value': self.best_value,
                'best_trial_number': self.best_trial_number,
                'best_params': self.best_params
            }
            
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.warning(f"   ⚠️ Failed to write trial data: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current optimization summary."""
        
        with self.lock:
            if self.summary_file.exists():
                with open(self.summary_file, 'r') as f:
                    return json.load(f)
            return {}


class OptunaProgressReader:
    """
    Reads real-time Optuna progress from stream files.
    Used by dashboard to display live-updating plots.
    """
    
    def __init__(self, stream_file: Path):
        """
        Args:
            stream_file: Path to the Optuna stream file
        """
        self.stream_file = Path(stream_file)
        self.logger = get_logger("workflow_16s.optuna")
    
    def read_trials(self) -> List[Dict[str, Any]]:
        """Read all trials from stream file."""
        
        if not self.stream_file.exists():
            return []
        
        trials = []
        try:
            with open(self.stream_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            trials.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            self.logger.debug(f"Failed to read stream: {e}")
        
        return trials
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get trials as a DataFrame for plotting."""
        
        trials = self.read_trials()
        if not trials:
            return pd.DataFrame()
        
        return pd.DataFrame(trials)


def create_optuna_callback(
    output_dir: Path,
    telemetry: Optional[TelemetryCollector] = None,
    trial_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    enable_json_stream: bool = True
) -> OptunaRealtimeCallback:
    """
    Factory function to create an Optuna callback with real-time monitoring.
    
    This integrates Optuna trial progress directly into:
    - EnhancedDashboardMonitor (via telemetry events)
    - CLI logging
    - JSON stream files (for post-analysis)
    
    Usage:
        from workflow_16s.downstream.machine_learning.optuna.realtime_monitor import create_optuna_callback
        
        callback = create_optuna_callback(
            output_dir=Path("output/optuna"),
            telemetry=workflow.telemetry,  # TelemetryCollector from workflow
            enable_json_stream=True
        )
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, callbacks=[callback])
    
    Args:
        output_dir: Directory to write trial data files
        telemetry: Optional TelemetryCollector for dashboard integration
        trial_callback: Optional custom callback for each trial
        enable_json_stream: If True, write streaming JSON for offline analysis
    
    Returns:
        OptunaRealtimeCallback instance ready to use with study.optimize()
    """
    return OptunaRealtimeCallback(
        output_dir=output_dir,
        telemetry=telemetry,
        trial_callback=trial_callback,
        enable_json_stream=enable_json_stream
    )
