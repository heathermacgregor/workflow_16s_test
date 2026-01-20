import logging
from pathlib import Path
from typing import Dict
import anndata as ad
import pandas as pd
import plotly.express as px

from workflow_16s.logger import get_logger

logger = get_logger()

class ComprehensiveAnalysis:
    """
    Synthesizes results and generates a final report and AnnData object.
    """
    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata
        self.comp_config = self.config.get('comprehensive_analysis', {})

    def run(self, output_dir: Path):
        """Generates a summary and saves the final AnnData object."""
        if not self.comp_config.get('enabled', True):
            logger.info("Comprehensive analysis disabled, skipping final report.")
            # Still save the object even if report is disabled
            self._save_final_adata(output_dir)
            return

        logger.info("STEP 8: Generating comprehensive analysis and report...")
        key_features_summary = self._identify_key_features()
        self.adata.uns['key_features'] = key_features_summary
        
        self._generate_html_report(output_dir)
        self._save_final_adata(output_dir)

    def _save_final_adata(self, output_dir: Path):
        """Saves the final, fully analyzed AnnData object."""
        final_adata_path = output_dir / "final_analyzed_data.h5ad"
        try:
            # AnnData objects with dataframes in .uns need a specific file format
            self.adata.write(final_adata_path, compression="gzip")
            logger.info(f"✅ Final AnnData object saved to: {final_adata_path}")
        except Exception as e:
            logger.error(f"Failed to save final AnnData object: {e}")

    def _identify_key_features(self) -> pd.DataFrame:
        """Cross-references results to find features that are consistently important."""
        p_val_threshold = self.comp_config.get('p_value_threshold', 0.05)
        top_n_ml = self.comp_config.get('top_n_ml_features', 50)
        significant_features = {}

        stats_results = self.adata.uns.get('statistical_tests', {})
        for group, layers in stats_results.items():
            for layer, tests in layers.items():
                for test, result_df in tests.items():
                    p_col = next((c for c in result_df.columns if 'p_adj' in c or 'p_value_bonferroni' in c), None)
                    if p_col:
                        sig = result_df[result_df[p_col] < p_val_threshold]
                        for feature in sig.index:
                            key = feature
                            if key not in significant_features: significant_features[key] = []
                            significant_features[key].append(f"Stat-Sig ({test} vs {group})")

        ml_results = self.adata.uns.get('ml', {})
        for target, layers in ml_results.items():
            for layer, methods in layers.items():
                for method, result_dict in methods.items():
                    if 'feature_importances' in result_dict:
                        top_feats = result_dict['feature_importances'].head(top_n_ml)
                        for feature in top_feats['feature']:
                            key = feature
                            if key not in significant_features: significant_features[key] = []
                            significant_features[key].append(f"Top-ML ({method} for {target})")

        if not significant_features:
            return pd.DataFrame(columns=['Feature', 'Evidence Count', 'Evidence'])
        
        summary_data = [[feature, len(reasons), "; ".join(sorted(list(set(reasons))))] for feature, reasons in significant_features.items()]
        summary_df = pd.DataFrame(summary_data, columns=['Feature', 'Evidence Count', 'Evidence'])
        return summary_df.sort_values(by='Evidence Count', ascending=False)

    def _generate_html_report(self, report_dir: Path):
        """Creates a summary HTML report of the entire analysis."""
        report_dir = report_dir / "comprehensive_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        volcano_plots = self._create_volcano_plots(report_dir)
        key_features_df = self.adata.uns.get('key_features', pd.DataFrame())

        def df_to_html(df: pd.DataFrame, title: str) -> str:
            if df is None or df.empty: return f"<h3>{title}</h3><p>No data available.</p>"
            return f"<h3>{title}</h3><div style='max-height: 400px; overflow-y: auto;'>{df.to_html(classes='table table-striped table-hover', border=0, index=False)}</div>"
        
        key_features_html = df_to_html(key_features_df.head(50), "Top 50 Key Features Across All Analyses")
        volcano_html = "<h2>Volcano Plots (Effect Size vs. Significance)</h2>"
        for title, path in volcano_plots.items():
            volcano_html += f"<h3>{title}</h3><iframe src='{Path(path).name}' width='100%' height='600px' style='border:none;'></iframe>"

        html_template = f"""
        <html><head><title>Comprehensive Analysis Report</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style> body {{ padding: 2rem; }} </style></head>
        <body><div class="container-fluid">
            <h1 class="text-center mb-4">🔬 Comprehensive Analysis Report</h1><hr>
            <h2>Key Feature Identification</h2>
            <p>Features identified as significant across multiple statistical tests and machine learning models.</p>
            {key_features_html}<hr>
            {volcano_html}<hr>
            <p class="text-muted text-center">Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div></body></html>"""
        
        report_path = report_dir / "comprehensive_report.html"
        with open(report_path, "w") as f: f.write(html_template)
        logger.info(f"✅ Comprehensive HTML report saved to: {report_path}")

    def _create_volcano_plots(self, report_dir: Path) -> Dict[str, Path]:
        """Generates and saves volcano plots for differential abundance results."""
        volcano_paths = {}
        stats_results = self.adata.uns.get('statistical_tests', {})
        p_val_threshold = self.comp_config.get('p_value_threshold', 0.05)
        
        for group, layers in stats_results.items():
            for layer, tests in layers.items():
                if 'clr' not in layer: continue
                for test, result_df in tests.items():
                    lfc_col = next((c for c in result_df.columns if 'log' in c and 'fold' in c), None)
                    p_col = next((c for c in result_df.columns if 'p_adj' in c or 'bonferroni' in c), None)
                    if lfc_col and p_col:
                        title = f"Volcano Plot: {group} ({layer}, {test})"
                        result_df['-log10(p-value)'] = -np.log10(result_df[p_col])
                        result_df['Significance'] = result_df[p_col] < p_val_threshold
                        fig = px.scatter(result_df, x=lfc_col, y='-log10(p-value)', color='Significance',
                                         hover_name=result_df.index, title=title,
                                         color_discrete_map={{True: 'red', False: 'grey'}})
                        fig.add_hline(y=-np.log10(p_val_threshold), line_dash="dash")
                        plot_path = report_dir / f"volcano_{group}_{layer}_{test}.html"
                        fig.write_html(plot_path)
                        volcano_paths[title] = plot_path
        return volcano_paths