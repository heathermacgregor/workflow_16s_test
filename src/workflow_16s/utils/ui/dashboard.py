"""
BackfillProgressPanel: Multi-stage backfill progress display for Rich dashboard.

Tracks the 6 stages of enrichment:
1. ENA/SRA Metadata Enrichment
2. Arkin Agents LLM Backfill
3. NFC GIS Facility Matching
4. Google Earth Engine (GEE) Enrichment
5. Environmental Data Collection
6. CSU Soil Heavy Metal Enrichment

Shows current stage name and completion percentage with integrated telemetry.
"""

from typing import Dict, Optional
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
from workflow_16s.utils.telemetry import TelemetryCollector


class BackfillProgressPanel:
    """Multi-stage backfill progress tracker with telemetry integration."""

    # Define the 6 backfill enrichment stages in order
    BACKFILL_STAGES = [
        ("ENA/SRA Metadata", "📅"),
        ("Arkin Agents LLM", "🤖"),
        ("NFC Facility Match", "☢️"),
        ("GEE Enrichment", "🌍"),
        ("Environmental Data", "🌎"),
        ("CSU Soil Metals", "⚗️"),
    ]

    def __init__(self, telemetry: TelemetryCollector):
        """
        Initialize backfill progress panel.

        Args:
            telemetry: TelemetryCollector instance for event tracking
        """
        self.telemetry = telemetry
        self._current_stage_index = -1  # Not started
        self._stage_completion: Dict[int, float] = {}  # stage_index -> completion %

    def _get_current_stage_from_telemetry(self) -> Optional[int]:
        """
        Determine current stage index from telemetry events.

        Returns:
            Index (0-5) of current stage, or None if not started
        """
        try:
            events = self.telemetry.get_events(limit=100)

            # Map phase names from backfill.py to stage indices
            phase_to_stage = {
                'backfill_ena': 0,
                'backfill_arkin': 1,
                'backfill_nfc': 2,
                'backfill_gee': 3,
                'backfill_environmental': 4,
                'backfill_csu_soil': 5,
                'backfill_geochemical': 6,  # Bonus stage
            }

            # Find most recent active stage by looking for step_start without step_end
            stages_started = {}
            for event in events:
                if event.phase in phase_to_stage:
                    stage_idx = phase_to_stage[event.phase]
                    if event.event_type == 'step_start':
                        stages_started[stage_idx] = event.timestamp
                    elif event.event_type == 'step_end':
                        stages_started.pop(stage_idx, None)

            # Return the stage with the most recent start time
            if stages_started:
                return max(stages_started.items(), key=lambda x: x[1])[0]
        except Exception:
            pass

        return None

    def _get_stages_completed(self) -> int:
        """Count how many stages have been completed (step_end events)."""
        try:
            events = self.telemetry.get_events(limit=100)
            phase_to_stage = {
                'backfill_ena': 0,
                'backfill_arkin': 1,
                'backfill_nfc': 2,
                'backfill_gee': 3,
                'backfill_environmental': 4,
                'backfill_csu_soil': 5,
                'backfill_geochemical': 6,
            }

            completed = set()
            for event in events:
                if event.phase in phase_to_stage and event.event_type == 'step_end':
                    completed.add(phase_to_stage[event.phase])

            return len(completed)
        except Exception:
            return 0

    def render(self) -> Text:
        """
        Render the backfill progress panel as Rich Text.

        Returns:
            Formatted Rich Text with stage progress indicators
        """
        try:
            current_stage = self._get_current_stage_from_telemetry()
            completed_count = self._get_stages_completed()

            text = Text()
            text.append("╭─ Backfill Progress (6 Stages)\n", style="bold cyan")

            total_stages = len(self.BACKFILL_STAGES)
            overall_pct = int(100 * completed_count / total_stages) if total_stages > 0 else 0

            # Progress bar line
            filled = int(overall_pct / 10)  # 10 chars for 100%
            bar = "█" * filled + "░" * (10 - filled)
            text.append(f"│ [{bar}] {overall_pct}% ({completed_count}/{total_stages})\n\n", style="white")

            # Individual stage status
            for stage_idx, (stage_name, emoji) in enumerate(self.BACKFILL_STAGES):
                if stage_idx < completed_count:
                    # Completed stage
                    icon = "✅"
                    style = "green"
                    status = "Done"
                elif stage_idx == current_stage:
                    # Currently running
                    icon = "▶️ "
                    style = "bold cyan"
                    status = "Running"
                else:
                    # Pending
                    icon = "⭕"
                    style = "white"
                    status = "Pending"

                display = f"│ {icon} [{stage_idx+1}] {stage_name:25s} {status:10s}\n"
                text.append(display, style=style)

            text.append("╰─ Backfill stages\n", style="bold cyan")

            return text

        except Exception as e:
            error_text = Text()
            error_text.append(f"╭─ Backfill Progress [Error: {str(e)[:40]}]\n", style="yellow")
            error_text.append("╰─ Unable to render progress\n", style="yellow")
            return error_text
