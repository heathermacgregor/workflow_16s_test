"""
Pagel's Lambda Phylogenetic Signal Analysis + Visualization

Separate module for phylogenetic signal (λ) quantification and publication-ready plots.
Not integrated into ML pipeline - stands alone for trait conservation analysis.

Answers: 
- Which functional traits follow evolution (conserved, λ ≈ 1)?
- Which traits appear randomly (adaptive/HGT, λ ≈ 0)?
- Which traits show intermediate signal (mixed inheritance modes)?

References:
- Pagel, M. (1999). Inferring the historical patterns of biological evolution.
  Nature 401, 877–884.
- Revell, L. J., Harmon, L. J., & Collar, D. C. (2008). Phylogenetic signal,
  evolutionary process, and rate. Systematic Biology 57(4), 591–601.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from workflow_16s.utils.logger import get_logger
from workflow_16s.downstream.phylogenetic_signal import (
    compute_phylogenetic_signal_fast,
    estimate_distance_matrix_from_abundance
)

logger = get_logger("workflow_16s")


@dataclass
class PagelsLambdaResult:
    """Result from Pagel's lambda analysis for a single trait."""
    trait_name: str
    lambda_estimate: float  # 0-1, phylogenetic signal measure
    p_value: float
    interpretation: str  # CONSERVED, ADAPTIVE, UNCLEAR
    n_otus_with_trait: int
    n_otus_total: int
    significance_level: str  # "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "ns" (not significant)


class PagelsLambdaAnalyzer:
    """
    Comprehensive Pagel's lambda analysis with visualization.
    
    Independent module for phylogenetic signal testing.
    """
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        output_dir : Path
            Base output directory for results and plots
        dpi : int
            Resolution for static plots (PNG)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.results: Dict[str, PagelsLambdaResult] = {}
    
    def analyze(
        self,
        function_matrix: pd.DataFrame,
        otu_table: pd.DataFrame,
        taxonomy_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, PagelsLambdaResult]:
        """
        Compute Pagel's lambda for each trait.
        
        Parameters
        ----------
        function_matrix : pd.DataFrame
            OTU × Trait/Function matrix (binary or continuous)
        otu_table : pd.DataFrame
            OTU abundance table (samples × OTU)
        taxonomy_df : pd.DataFrame, optional
            OTU taxonomy for annotation
        
        Returns
        -------
        Dict[str, PagelsLambdaResult]
            Results per trait
        """
        logger.info("\n" + "="*80)
        logger.info("PAGEL'S LAMBDA PHYLOGENETIC SIGNAL ANALYSIS")
        logger.info("="*80)
        logger.info(
            f"Input: {len(function_matrix)} OTUs × {len(function_matrix.columns)} traits\n"
        )
        
        # Estimate distance matrix from OTU abundances
        logger.info("Estimating phylogenetic distance matrix from OTU co-occurrence...")
        dist_matrix = estimate_distance_matrix_from_abundance(otu_table)
        
        # Analyze each trait
        for trait_name in function_matrix.columns:
            trait_vector = function_matrix[trait_name].values
            
            # Compute phylogenetic signal
            signal_result = compute_phylogenetic_signal_fast(
                trait_vector,
                dist_matrix,
                function_matrix.index.tolist()
            )
            
            # Format significance level
            p_val = signal_result["p_value"]
            if np.isnan(p_val):
                sig_level = "N/A"
            elif p_val < 0.001:
                sig_level = "***"
            elif p_val < 0.01:
                sig_level = "**"
            elif p_val < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"
            
            # Create result object
            result = PagelsLambdaResult(
                trait_name=trait_name,
                lambda_estimate=signal_result.get("lambda", np.nan),
                p_value=signal_result.get("p_value", np.nan),
                interpretation=signal_result.get("interpretation", "Unknown"),
                n_otus_with_trait=int(signal_result.get("n_otus_with_trait", 0)),
                n_otus_total=int(signal_result.get("n_otus_total", len(trait_vector))),
                significance_level=sig_level
            )
            
            self.results[trait_name] = result
            
            # Log result
            if not np.isnan(result.lambda_estimate):
                logger.info(
                    f"  {trait_name}: λ={result.lambda_estimate:.3f} {sig_level} "
                    f"(p={result.p_value:.4f}) - {result.interpretation}"
                )
            else:
                logger.info(f"  {trait_name}: {result.interpretation}")
        
        return self.results
    
    def plot_lambda_distribution(self) -> go.Figure:
        """
        Create forest plot of Pagel's lambda estimates.
        
        Shows each trait's λ value with significance markers.
        """
        if not self.results:
            logger.warning("No results to plot. Run analyze() first.")
            return None
        
        # Prepare data
        traits = []
        lambdas = []
        p_values = []
        colors = []
        
        for trait_name, result in self.results.items():
            traits.append(trait_name)
            lambdas.append(result.lambda_estimate)
            p_values.append(result.p_value)
            
            # Color by interpretation
            if np.isnan(result.lambda_estimate):
                colors.append("lightgray")
            elif result.lambda_estimate > 0.6:
                colors.append("darkgreen")  # Conserved
            elif result.lambda_estimate < 0.3:
                colors.append("darkred")    # Adaptive
            else:
                colors.append("orange")     # Mixed
        
        # Sort by lambda value
        sorted_idx = np.argsort(lambdas)
        traits = [traits[i] for i in sorted_idx]
        lambdas = [lambdas[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]
        p_values = [p_values[i] for i in sorted_idx]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=traits,
            x=lambdas,
            orientation='h',
            marker=dict(color=colors, line=dict(color='black', width=1)),
            text=[
                f"λ={l:.2f}" if not np.isnan(l) else "N/A"
                for l in lambdas
            ],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>λ = %{x:.3f}<br>p = %{customdata:.4f}<extra></extra>',
            customdata=p_values
        ))
        
        # Add reference lines
        fig.add_vline(x=0.0, line_dash="dash", line_color="red", annotation_text="λ=0 (Adaptive)")
        fig.add_vline(x=1.0, line_dash="dash", line_color="green", annotation_text="λ=1 (Conserved)")
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title="Pagel's Lambda: Phylogenetic Signal by Trait",
            xaxis_title="Pagel's λ (Phylogenetic Signal)",
            yaxis_title="Trait",
            height=max(400, 40 * len(traits)),
            template="plotly_white",
            hovermode="closest",
            showlegend=False,
            xaxis=dict(range=[0, 1.1])
        )
        
        return fig
    
    def plot_lambda_scatter(self) -> go.Figure:
        """
        Scatter plot of λ vs significance level.
        
        Shows relationship between signal strength and statistical support.
        """
        if not self.results:
            return None
        
        traits = list(self.results.keys())
        lambdas = [self.results[t].lambda_estimate for t in traits]
        neg_log_p = [-np.log10(max(self.results[t].p_value, 1e-10)) for t in traits]
        significance = [self.results[t].significance_level for t in traits]
        interpretations = [self.results[t].interpretation for t in traits]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=lambdas,
            y=neg_log_p,
            mode='markers+text',
            marker=dict(
                size=10,
                color=lambdas,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="λ (Signal)"),
                line=dict(color='black', width=1)
            ),
            text=traits,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>λ = %{x:.3f}<br>-log10(p) = %{y:.2f}<extra></extra>'
        ))
        
        # Add significance threshold line
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray",
                     annotation_text="p = 0.05")
        
        fig.update_layout(
            title="Phylogenetic Signal vs Statistical Support",
            xaxis_title="Pagel's λ (0 = Adaptive, 1 = Conserved)",
            yaxis_title="-log10(p-value)",
            template="plotly_white",
            height=500,
            width=700
        )
        
        return fig
    
    def plot_interpretation_summary(self) -> go.Figure:
        """
        Pie/sunburst chart of trait categories by signal interpretation.
        """
        if not self.results:
            return None
        
        categories = {
            "Conserved (λ > 0.6)": 0,
            "Mixed (0.3 ≤ λ ≤ 0.6)": 0,
            "Adaptive (λ < 0.3)": 0,
            "Unclear": 0
        }
        
        for result in self.results.values():
            if np.isnan(result.lambda_estimate):
                categories["Unclear"] += 1
            elif result.lambda_estimate > 0.6:
                categories["Conserved (λ > 0.6)"] += 1
            elif result.lambda_estimate < 0.3:
                categories["Adaptive (λ < 0.3)"] += 1
            else:
                categories["Mixed (0.3 ≤ λ ≤ 0.6)"] += 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(categories.keys()),
            values=list(categories.values()),
            hole=0.3,
            marker=dict(colors=['darkgreen', 'orange', 'darkred', 'lightgray']),
            textinfo="label+percent+value"
        )])
        
        fig.update_layout(
            title="Trait Distribution by Phylogenetic Signal Category",
            height=500,
            width=600
        )
        
        return fig
    
    def save_results_table(self, filename: str = "pagels_lambda_results.csv"):
        """Save results to CSV table."""
        if not self.results:
            logger.warning("No results to save.")
            return
        
        data = []
        for trait_name, result in self.results.items():
            data.append({
                'Trait': trait_name,
                "Pagel's λ": result.lambda_estimate,
                'p-value': result.p_value,
                'Significance': result.significance_level,
                'OTUs with Trait': result.n_otus_with_trait,
                'Total OTUs': result.n_otus_total,
                'Interpretation': result.interpretation
            })
        
        df = pd.DataFrame(data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Results table saved: {output_path}")
        
        return df
    
    def generate_report(self) -> str:
        """Generate markdown report of phylogenetic signal findings."""
        if not self.results:
            return "No results to report."
        
        report = [
            "# Pagel's Lambda Phylogenetic Signal Analysis Report\n",
            f"## Summary\n",
            f"- Traits analyzed: {len(self.results)}\n",
            f"- Conserved traits (λ > 0.6): {sum(1 for r in self.results.values() if not np.isnan(r.lambda_estimate) and r.lambda_estimate > 0.6)}\n",
            f"- Adaptive traits (λ < 0.3): {sum(1 for r in self.results.values() if not np.isnan(r.lambda_estimate) and r.lambda_estimate < 0.3)}\n",
            f"- Mixed signal traits: {sum(1 for r in self.results.values() if not np.isnan(r.lambda_estimate) and 0.3 <= r.lambda_estimate <= 0.6)}\n",
            f"\n## Detailed Results\n\n"
        ]
        
        # Sort by lambda
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].lambda_estimate if not np.isnan(x[1].lambda_estimate) else -1,
            reverse=True
        )
        
        for trait_name, result in sorted_results:
            report.append(f"### {trait_name}\n")
            report.append(f"- **Pagel's λ**: {result.lambda_estimate:.3f} {result.significance_level}\n")
            report.append(f"- **p-value**: {result.p_value:.6f}\n")
            report.append(f"- **Interpretation**: {result.interpretation}\n")
            report.append(f"- **Prevalence**: {result.n_otus_with_trait} / {result.n_otus_total} OTUs\n\n")
        
        report.append("\n## Interpretation Guide\n\n")
        report.append("- **λ ≈ 1 (Conserved)**: Trait follows evolutionary relationships closely. ")
        report.append("Likely ancestral origin with vertical inheritance.\n")
        report.append("- **λ ≈ 0 (Adaptive)**: Trait appears independent of evolutionary history. ")
        report.append("Suggests horizontal gene transfer (HGT), convergent evolution, or recent acquisition.\n")
        report.append("- **λ ≈ 0.5 (Mixed)**: Intermediate signal suggesting mixed inheritance modes ")
        report.append("or trait loss in some lineages.\n")
        report.append("- **Not significant**: No phylogenetic signal detected (p ≥ 0.05).\n")
        
        return "".join(report)


def run_pagels_lambda_pipeline(
    config: Dict,
    function_matrix: Optional[pd.DataFrame] = None,
    otu_table: Optional[pd.DataFrame] = None,
    trait_data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    End-to-end Pagel's lambda analysis pipeline.
    
    Standalone entry point (not part of ML pipeline).
    
    Parameters
    ----------
    config : Dict
        Pipeline configuration with pagels_lambda settings
    function_matrix : pd.DataFrame, optional
        OTU × Trait matrix. If None, loads from trait_data
    otu_table : pd.DataFrame, optional
        OTU abundance table for distance estimation
    trait_data : pd.DataFrame, optional
        Alternative format: traits × OTU
    output_dir : Path, optional
        Output directory for results/plots
    
    Returns
    -------
    Dict
        Results dictionary with plots and tables
    """
    if output_dir is None:
        output_dir = Path(config.get('output_dir', './pagels_lambda_results'))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("PAGEL'S LAMBDA PHYLOGENETIC SIGNAL PIPELINE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}\n")
    
    # Initialize analyzer
    analyzer = PagelsLambdaAnalyzer(output_dir)
    
    # Prepare data
    if function_matrix is None and trait_data is not None:
        function_matrix = trait_data.T
    
    if function_matrix is None or otu_table is None:
        logger.error("require either (function_matrix + otu_table) or trait_data")
        return None
    
    # Run analysis
    results = analyzer.analyze(function_matrix, otu_table)
    
    # Generate plots
    logger.info("\nGenerating visualizations...")
    
    try:
        # Forest plot
        fig_forest = analyzer.plot_lambda_distribution()
        if fig_forest:
            fig_forest.write_html(output_dir / "pagels_lambda_forest.html")
            fig_forest.write_image(output_dir / "pagels_lambda_forest.png", width=1000, height=max(400, 40*len(results)), scale=2)
            logger.info("✓ Forest plot saved")
    except Exception as e:
        logger.warning(f"Forest plot generation failed: {e}")
    
    try:
        # Scatter plot
        fig_scatter = analyzer.plot_lambda_scatter()
        if fig_scatter:
            fig_scatter.write_html(output_dir / "pagels_lambda_scatter.html")
            fig_scatter.write_image(output_dir / "pagels_lambda_scatter.png", scale=2)
            logger.info("✓ Scatter plot saved")
    except Exception as e:
        logger.warning(f"Scatter plot generation failed: {e}")
    
    try:
        # Interpretation summary
        fig_summary = analyzer.plot_interpretation_summary()
        if fig_summary:
            fig_summary.write_html(output_dir / "pagels_lambda_summary.html")
            fig_summary.write_image(output_dir / "pagels_lambda_summary.png", scale=2)
            logger.info("✓ Summary plot saved")
    except Exception as e:
        logger.warning(f"Summary plot generation failed: {e}")
    
    # Save results table
    analyzer.save_results_table()
    
    # Generate markdown report
    report = analyzer.generate_report()
    report_path = output_dir / "phylogenetic_signal_report.md"
    report_path.write_text(report)
    logger.info(f"✓ Report saved: {report_path}")
    
    logger.info("\n✓ Pagel's lambda analysis complete!")
    
    return {
        'analyzer': analyzer,
        'results': results,
        'output_dir': output_dir
    }
