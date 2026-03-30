# src/workflow_16s/downstream/machine_learning/Project_Discovery_Dashboard.py
import json
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime
from pathlib import Path

from workflow_16s.utils.logger import get_logger
from .validation.quality_audit import BiomarkerAuditor
from .visualization.validation_plots import plot_strategy_resilience


class DiscoveryDashboardGenerator:
    """
    Synthesizes all ML artifacts into a single 'Forensic Evidence' Dashboard.
    Provides stakeholders with a high-level view of which biomarkers are 
    scientifically certified for deployment.
    """
    
    def __init__(self, ml_output_dir: Path, project_name: str = "16S Discovery Project"):
        self.ml_dir = Path(ml_output_dir)
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.audit_summaries = []
        # 1. FIX: Initialize the missing attribute
        self.eligible_targets = []

    def _collect_target_data(self):
        """Scans the ML directory for all targets and their audit statuses."""
        # Find all Genus-level target folders in the agnostic run
        target_folders = list((self.ml_dir / "agnostic").glob("Genus_*"))
        
        # Reset to ensure cleanliness if called multiple times
        self.eligible_targets = []
        self.audit_summaries = []

        for folder in target_folders:
            target_name = folder.name.replace("Genus_", "")
            
            # 2. FIX: Track the target name in the class attribute
            self.eligible_targets.append(target_name)
            
            auditor = BiomarkerAuditor(self.ml_dir, target_name)
            auditor.run_audit()
            
            # Store audit metrics for the summary table
            summary = auditor.get_summary_df()
            summary['Target'] = target_name
            self.audit_summaries.append(summary)

    def _generate_resilience_metrics(self, target_name: str):
        """
        Extracts performance scores across strategies to build the Resilience Audit.
        """
        strategies = ['baseline', 'agnostic', 'lopocv', 'shuffle']
        plot_data = []

        for strat in strategies:
            # Path logic based on our directory structure
            result_path = self.ml_dir / strat / f"Genus_{target_name}" / "performance_metrics.json"
            if result_path.exists():
                with open(result_path, 'r') as f:
                    try:
                        data = json.load(f)
                        plot_data.append({
                            'strategy': strat,
                            'metric_score': data.get('best_score', 0),
                            'std_dev': data.get('std_dev', 0)
                        })
                    except json.JSONDecodeError:
                        pass

        if plot_data:
            res_df = pd.DataFrame(plot_data)
            output_plot = self.ml_dir / "summary_reports" / f"{target_name}_resilience_audit.html"
            output_plot.parent.mkdir(exist_ok=True, parents=True)
            plot_strategy_resilience(res_df, output_plot, target_name)
            return output_plot
        return None
    
    def create_dashboard(self, output_path: Path):
        """Generates the interactive HTML Synthesis."""
        self._collect_target_data()
        
        if not self.audit_summaries:
            logger = get_logger("workflow_16s")
            logger.warning("⚠️  No ML artifacts found to generate dashboard. This is expected if ML modules were skipped or no targets passed filtering.")
            return

        master_audit = pd.concat(self.audit_summaries)
        
        # 1. Create the 'Certification Matrix'
        # Pivoting to show Target vs Check Status
        matrix = master_audit.pivot(index='Target', columns='Check', values='Status')
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in ["Target Variable"] + list(matrix.columns)],
                fill_color='#2c3e50',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[matrix.index] + [matrix[c] for c in matrix.columns],
                fill_color='#f9f9f9',
                align='left',
                font_size=12
            ))
        ])

        fig.update_layout(
            title=f"📊 Discovery Certification Matrix: {self.project_name}<br><sub>Generated: {self.timestamp}</sub>",
            height=400,
            template='plotly_white'
        )

        # 2. Build the HTML Document
        html_content = f"""
        <html>
        <head>
            <title>{self.project_name} - Discovery Dashboard</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
                .header {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 30px; }}
                .status-pass {{ color: green; font-weight: bold; }}
                .status-fail {{ color: red; font-weight: bold; }}
                .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 2px 2px 5px #eee; }}
                h2 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.project_name}</h1>
                <p><b>Executive Discovery Summary</b> | Pipeline Version: 2.0 (Forensic Edition)</p>
            </div>

            <div class="card">
                <h2>1. Scientific Certification Matrix</h2>
                <p>Each target variable is subjected to a 4-tier audit (Eligibility, Overfitting, Significance, Stability).</p>
                {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>

            <div class="card">
                <h2>2. Deep-Dive Visual Evidence</h2>
                <p>Click the links below to view the detailed diagnostic reports for each discovery:</p>
                <ul>
        """

        for target in matrix.index:
            html_content += f"""
                    <li>
                        <b>{target.upper()}</b>: 
                        <a href="agnostic/Genus_{target}/discovery_audit_{target}.html" target="_blank">Full Diagnostic Audit</a> | 
                        <a href="meta_analysis/Genus_{target}/biomarker_stability_heatmap.html" target="_blank">Consensus Heatmap</a>
                    </li>
            """

        html_content += """
                </ul>
            </div>
            
            <div class="card" style="background-color: #f8f9fa;">
                <h2>3. Glossary of Terms</h2>
                <ul>
                    <li><b>Biological Signal (MCC)</b>: Measures predictive power (0 to 1). Pass requires > 0.4.</li>
                    <li><b>Overfitting Gap</b>: Difference between Train and Test scores. Pass requires < 0.15.</li>
                    <li><b>Significance (p-val)</b>: Probability the result is random chance. Pass requires < 0.05.</li>
                    <li><b>Stable Biomarkers</b>: Number of taxa found in >50% of independent studies.</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # 3. Append Resilience Plots
        if self.eligible_targets:
            resilience_html = "<div class='card'><h2>4. Resilience & Batch-Effect Audit</h2>"
            for target in self.eligible_targets:
                plot_path = self._generate_resilience_metrics(target)
                if plot_path:
                    # Use relative path for portability
                    try:
                        rel_path = plot_path.relative_to(output_path.parent)
                    except ValueError:
                        rel_path = plot_path.name
                        
                    resilience_html += f"""
                        <div style="margin-top: 20px;">
                            <h3>Target: {target}</h3>
                            <iframe src="{rel_path}" width="100%" height="500px" frameborder="0"></iframe>
                            <p><i>Interpretation: A large gap between Baseline and Agnostic indicates the discovery 
                            may be driven by technical artifacts (Batch Effects).</i></p>
                        </div>
                        <hr>
                    """
            resilience_html += "</div>"
            html_content = html_content.replace("</body>", f"{resilience_html}</body>")
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        get_logger("workflow_16s").info(f"🏆 Executive Dashboard synthesized: {output_path}")