"""
EnhancedDashboardMonitor: Live Rich TUI dashboard with 4-panel layout.

Panels:
1. System Status: CPU/RAM/Load/Disk
2. Pipeline Progress: Current step + % complete
3. Module Status Grid: Visual status of analysis modules
4. Console Feed: Last 15 log lines (non-scrollable)

Optimized refresh rate: 1 Hz (vs 2 Hz for CPU savings).
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil

from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TaskProgressColumn
from rich.console import Group

from workflow_16s.utils.telemetry import TelemetryCollector
from workflow_16s.utils.ui.dashboard import BackfillProgressPanel


class EnhancedDashboardMonitor:
    """Live Rich dashboard with 4 core panels for workflow monitoring."""
    
    def __init__(self, telemetry: TelemetryCollector, logger: logging.Logger):
        """
        Args:
            telemetry: TelemetryCollector instance for event tracking
            logger: Logger instance for console feed
        """
        self.telemetry = telemetry
        self.logger = logger
        self.backfill_progress = BackfillProgressPanel(telemetry)
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:.0f}%"),
            transient=True
        )
        self._console_lines: List[str] = []
        self._console_lock = __import__('threading').Lock()
        self._start_time = time.time()
        
    def _get_system_panel(self) -> Panel:
        """Render system status panel (CPU/RAM/Load/Disk)."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            
            try:
                load = os.getloadavg()
                load_str = f"{load[0]:.1f}, {load[1]:.1f}, {load[2]:.1f}"
            except AttributeError:
                load_str = "N/A"
            
            cpu_color = "red" if cpu > 90 else "yellow" if cpu > 70 else "green"
            ram_color = "red" if ram.percent > 90 else "yellow" if ram.percent > 80 else "green"
            
            text = Text()
            text.append("╭─ Load Avg (1/5/15m)\n", style="bold cyan")
            text.append(f"│ {load_str}\n\n", style="white")
            
            text.append("╭─ CPU Usage\n", style="bold cyan")
            text.append(f"│ [{'█' * int(cpu/5)}{'░' * (20 - int(cpu/5))}] {cpu}%\n\n", style=cpu_color)
            
            text.append("╭─ Memory Usage\n", style="bold cyan")
            text.append(f"│ [{'█' * int(ram.percent/5)}{'░' * (20 - int(ram.percent/5))}] {ram.percent}%\n", style=ram_color)
            text.append(f"│ {ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB\n\n", style="white")
            
            text.append("╰─ Process Tree\n", style="bold cyan")
            parent = psutil.Process(os.getpid())
            text.append(f"  Parent + {len(parent.children(recursive=True))} children\n", style="white")
            
            return Panel(text, title="[bold white]🖥️  System Status", border_style="cyan", expand=False)
        except Exception as e:
            return Panel(f"[red]Error: {e}", title="System Status", border_style="red")
    
    def _get_pipeline_progress_panel(self) -> Panel:
        """Render pipeline progress panel."""
        try:
            phase_status = self.telemetry.get_phase_status()
            latest = self.telemetry.get_latest_event()
            summary = self.telemetry.get_summary()
            
            text = Text()
            text.append("╭─ Current Status\n", style="bold cyan")
            
            if latest:
                text.append(f"│ Phase: {latest.phase}\n", style="white")
                text.append(f"│ Message: {latest.message[:60]}\n\n", style="white")
            else:
                text.append("│ Initializing...\n\n", style="yellow")
            
            text.append("╭─ Phases Completed\n", style="bold cyan")
            completed = summary.get('phases_completed', 0)
            total_runtime = summary.get('total_runtime_seconds', 0)
            text.append(f"│ {completed} phases | {total_runtime:.1f}s elapsed\n\n", style="white")
            
            elapsed = time.time() - self._start_time
            text.append("╰─ Session Time\n", style="bold cyan")
            text.append(f"  {int(elapsed//60):02d}m {int(elapsed%60):02d}s\n", style="white")
            
            return Panel(text, title="[bold white]⚙️  Pipeline Progress", border_style="cyan", expand=False)
        except Exception as e:
            return Panel(f"[red]Error: {e}", title="Pipeline Progress", border_style="red")
    
    def _get_currently_running_module(self) -> Optional[str]:
        """Get the currently running module from telemetry events."""
        try:
            events = self.telemetry.get_events(limit=100)
            # Find most recent step_start event without corresponding step_end
            phases_started = {}
            for event in events:
                if event.event_type == 'step_start':
                    phases_started[event.phase] = event.timestamp
                elif event.event_type == 'step_end':
                    phases_started.pop(event.phase, None)
            
            # Return the phase with the most recent start time
            if phases_started:
                return max(phases_started.items(), key=lambda x: x[1])[0]
        except Exception:
            pass
        return None
    
    def _get_modules_grid_panel(self) -> Panel:
        """Render module status grid (visual indicators)."""
        try:
            phase_status = self.telemetry.get_phase_status()
            current_module = self._get_currently_running_module()
            
            # Define known analysis phases
            known_phases = [
                'Geo_Enrichment', 'QC_Profiling', 'CST_Typing',
                'Phylo_Diversity', 'Alpha_Diversity', 'Beta_Diversity',
                'Diff_Abundance', 'ML_Discovery_Matrix',
                'Ordination', 'Network_Analysis'
            ]
            
            text = Text()
            text.append("╭─ Analysis Modules\n", style="bold cyan")
            
            for phase in known_phases:
                status = phase_status.get(phase, 'pending')
                is_current = phase == current_module
                
                if status == 'disabled':
                    # Module is disabled in configuration
                    icon = "⊘ "
                    color = "dim"
                elif is_current:
                    # Currently running - highlight with bright cyan
                    icon = "▶️ "
                    color = "bold cyan"
                elif status == 'completed':
                    icon = "✅"
                    color = "green"
                elif status == 'running':
                    # Other running modules (shouldn't happen with sequential execution)
                    icon = "⏳"
                    color = "yellow"
                elif status == 'failed':
                    icon = "❌"
                    color = "red"
                else:
                    icon = "⭕"
                    color = "white"
                
                display_text = f"│ {icon} {phase:23s} {status:10s}\n"
                text.append(display_text, style=color)
            
            text.append("╰─ Grid end\n", style="bold cyan")
            
            return Panel(text, title="[bold white]📊 Module Status", border_style="cyan", expand=False)
        except Exception as e:
            return Panel(f"[red]Error: {e}", title="Module Status", border_style="red")
    
    def _capture_log_line(self, record: logging.LogRecord) -> None:
        """Capture log record for console feed (called by LogHandler)."""
        msg = record.getMessage()
        with self._console_lock:
            self._console_lines.append(msg)
            # Keep only last 30 lines (display 15)
            if len(self._console_lines) > 30:
                self._console_lines.pop(0)
    
    def _get_console_feed_panel(self) -> Panel:
        """Render last N log lines."""
        try:
            with self._console_lock:
                # Show last 15 lines
                display_lines = self._console_lines[-15:] if self._console_lines else ["[yellow]No logs yet..."]

            text = Text()
            text.append("╭─ Console Feed (last 15 lines)\n", style="bold cyan")

            for line in display_lines:
                # Truncate long lines to 80 chars
                line_short = (line[:80] + '...') if len(line) > 80 else line
                text.append(f"│ {line_short}\n", style="white")

            text.append("╰─ End of feed\n", style="bold cyan")

            return Panel(text, title="[bold white]📋 Live Logs", border_style="cyan", expand=False)
        except Exception as e:
            return Panel(f"[red]Error: {e}", title="Live Logs", border_style="red")

    def _get_backfill_progress_panel(self) -> Panel:
        """Render backfill progress panel."""
        try:
            backfill_text = self.backfill_progress.render()
            return Panel(backfill_text, title="[bold white]📊 Backfill Enrichment", border_style="cyan", expand=False)
        except Exception as e:
            return Panel(f"[red]Error rendering backfill progress: {e}", title="Backfill Progress", border_style="red")
    
    def __rich__(self) -> Panel:
        """Render the complete dashboard (called by Rich Live)."""
        system_panel = self._get_system_panel()
        progress_panel = self._get_pipeline_progress_panel()
        backfill_panel = self._get_backfill_progress_panel()
        modules_panel = self._get_modules_grid_panel()
        console_panel = self._get_console_feed_panel()

        # Arrange in 3 rows:
        # Row 1: System + Pipeline (2 columns)
        # Row 2: Backfill (full width)
        # Row 3: Modules + Console (2 columns)
        layout = Layout()
        layout.split_column(
            Layout(name="row1"),
            Layout(name="row2"),
            Layout(name="row3")
        )

        # Row 1: System Status | Pipeline Progress
        layout["row1"].split_row(
            Layout(Panel(system_panel.renderable, title="", border_style=""), name="row1_left"),
            Layout(Panel(progress_panel.renderable, title="", border_style=""), name="row1_right")
        )

        # Row 2: Backfill Progress (full width)
        layout["row2"].update(
            Layout(Panel(backfill_panel.renderable, title="", border_style=""), name="backfill")
        )

        # Row 3: Module Status | Console Feed
        layout["row3"].split_row(
            Layout(Panel(modules_panel.renderable, title="", border_style=""), name="row3_left"),
            Layout(Panel(console_panel.renderable, title="", border_style=""), name="row3_right")
        )

        return Panel(
            layout,
            title="[bold yellow]🚀 16S Workflow Dashboard",
            border_style="yellow",
            padding=(0, 2)
        )


def create_dashboard_with_telemetry(telemetry: TelemetryCollector, logger: logging.Logger) -> EnhancedDashboardMonitor:
    """
    Factory function to create dashboard and attach logger handler.
    
    Args:
        telemetry: TelemetryCollector instance
        logger: Logger to wire into dashboard
    
    Returns:
        Configured EnhancedDashboardMonitor
    """
    dashboard = EnhancedDashboardMonitor(telemetry, logger)
    
    # Attach handler to logger to capture log lines
    handler = logging.Handler()
    handler.emit = lambda record: dashboard._capture_log_line(record)
    logger.addHandler(handler)
    
    return dashboard
