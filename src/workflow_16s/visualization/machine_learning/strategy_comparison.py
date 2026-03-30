# workflow_16s/visualization/machine_learning/strategy_comparison.py

from pathlib import Path
from typing import List, Optional
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from workflow_16s.utils.logger import get_logger

# STRATEGY COMPARISON & AUDIT DASHBOARDS
def create_strategy_comparison_dashboard(
    catboost_dir: Path,
    target_variable: str,
    output_dir: Path,
    strategies: List[str] = ['baseline', 'agnostic', 'lopocv']
) -> Optional[go.Figure]:
    """
    Creates a 4-panel 'Scientific Audit' dashboard for a specific ML target.
    
    PANELS:
    1. Performance: Standard metrics (MCC, F1, Accuracy).
    2. Overfitting Audit: The Gap between Training and Validation (Nested CV).
    3. Robustness Matrix: Feature stability across independent studies.
    4. Technical Bias: Dependency on 'Batch' vs 'Biological' features.
    
    Parameters
    ----------
    catboost_dir : Path
        Directory containing CatBoost ML results per strategy.
    target_variable : str
        The metadata variable being predicted (e.g., 'disease_status').
    output_dir : Path
        Directory to save the dashboard HTML.
    strategies : List[str]
        List of ML strategies to compare.
        
    Returns
    -------
    go.Figure
        Plotly Figure object of the dashboard, or None if no data found.
    """
    logger = get_logger("workflow_16s")
    logger.info(f" 📊 Generating Comprehensive Audit Dashboard for {target_variable}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data Collection
    strategy_results = {}
    for strategy in strategies:
        strat_path = catboost_dir / strategy / f"Genus_{target_variable}"
        summary_file = strat_path / "results_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
                # NEW: Attempt to load Overfitting Audit data
                audit_file = strat_path / "overfitting_audit" / "audit_results.json"
                if audit_file.exists():
                    with open(audit_file, 'r') as af:
                        data['audit'] = json.load(af)
                
                # NEW: Attempt to load Robustness Weighted features
                robust_file = strat_path / "robustness_weighted_biomarkers.csv"
                if robust_file.exists():
                    data['robust_df'] = pd.read_csv(robust_file)
                
                strategy_results[strategy] = data

    if not strategy_results:
        logger.warning(f"No results found for {target_variable} comparison.")
        return None

    # Define Layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>I. Performance Metrics</b> (Higher is Better)',
            '<b>II. Generalization Audit</b> (Lower Gap is Better)',
            '<b>III. Global Robustness</b> (Top 10 Universal Taxa)',
            '<b>IV. Batch Dependency</b> (Red = Technical Leakage)'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ]
    )

    # --- PANEL 1: Performance ---
    metrics = ['accuracy', 'mcc', 'f1']
    for strategy in strategies:
        if strategy not in strategy_results: continue
        res = strategy_results[strategy].get('test_scores', {})
        fig.add_trace(go.Bar(
            name=strategy.upper(),
            x=[m.upper() for m in metrics],
            y=[res.get(m, 0) for m in metrics],
            text=[f"{res.get(m, 0):.2f}" for m in metrics],
            textposition='auto'
        ), row=1, col=1)

    # --- PANEL 2: Overfitting Gap ---
    # Visualizes the difference between training score and validation score
    for strategy in strategies:
        if strategy not in strategy_results: continue
        audit = strategy_results[strategy].get('audit', {})
        # Extract gap from Nested CV or Learning Curve
        gap = audit.get('nested_cv', {}).get('overfitting_gap', 0)
        
        fig.add_trace(go.Bar(
            name=strategy.upper(),
            x=['Overfitting Gap'],
            y=[gap],
            marker_color='rgba(255, 0, 0, 0.6)' if gap > 0.15 else 'rgba(0, 128, 0, 0.6)',
            showlegend=False
        ), row=1, col=2)

    # --- PANEL 3: Robustness (SHAP vs Meta-Frequency) ---
    # Top features from the 'agnostic' or most robust strategy
    best_strat = 'agnostic' if 'agnostic' in strategy_results else strategies[0]
    if 'robust_df' in strategy_results[best_strat]:
        rdf = strategy_results[best_strat]['robust_df'].head(10)
        fig.add_trace(go.Scatter(
            x=rdf['importance'],
            y=rdf['Meta_Frequency'],
            mode='markers+text',
            text=[f.split('__')[-1] for f in rdf['feature']],
            textposition="top center",
            marker=dict(size=12, color=rdf['Meta_Frequency'], colorscale='Viridis', showscale=False),
            name="Universal Biomarkers"
        ), row=2, col=1)

    # --- PANEL 4: Batch Effect Impact ---
    # Checks if 'Lab ID' or 'Study ID' is a top predictor (High Importance = Bad)
    batch_keywords = ['batch', 'study', 'accession', 'dataset', 'center']
    for strategy in strategies:
        if strategy not in strategy_results: continue
        feat_imp = strategy_results[strategy].get('feature_importance', {})
        # Sum importance of any feature containing batch keywords
        batch_imp = sum([v for k, v in feat_imp.items() if any(kw in k.lower() for kw in batch_keywords)])
        
        fig.add_trace(go.Bar(
            name=strategy.upper(),
            x=['Batch Leakage'],
            y=[batch_imp],
            marker_color='rgba(255, 69, 0, 0.8)' if batch_imp > 10 else 'rgba(100, 149, 237, 0.8)',
            showlegend=False
        ), row=2, col=2)

    # Final Styling
    fig.update_layout(
        height=1000, width=1400,
        title_text=f"Microbial Discovery Audit: {target_variable.replace('_', ' ').title()}",
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Score Difference", row=1, col=2)
    fig.update_yaxes(title_text="Study Frequency (%)", row=2, col=1)
    fig.update_xaxes(title_text="SHAP Power", row=2, col=1)
    fig.update_yaxes(title_text="Relative Importance", row=2, col=2)

    dashboard_path = output_dir / f"discovery_audit_{target_variable}.html"
    fig.write_html(str(dashboard_path))
    logger.info(f"✅ Dashboard saved: {dashboard_path.name}")
    
    return fig

# FEATURE ROBUSTNESS VISUALIZATIONS
def plot_robustness_vs_importance(robust_df: pd.DataFrame, output_path: Path):
    """
    Plots SHAP Importance against Meta-Analysis Frequency.
    The 'Golden' Biomarkers are in the top-right quadrant.
    
    Parameters
    ----------
    robust_df : pd.DataFrame
        DataFrame with 'importance' and 'Meta_Frequency' columns.
    output_path : Path
        Path to save the output HTML plot.
    """
    fig = px.scatter(
        robust_df,
        x='importance',
        y='Meta_Frequency',
        color='Is_Universal',
        text=[f.split('__')[-1] for f in robust_df['feature']],
        size='Adjusted_Importance',
        labels={'importance': 'Global Predictive Power (SHAP)', 'Meta_Frequency': 'Cross-Study Consistency (0-1)'},
        title="Biomarker Discovery: Power vs. Consistency",
        template='plotly_white'
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Robustness Threshold")
    fig.update_traces(textposition='top center')
    fig.write_html(str(output_path))