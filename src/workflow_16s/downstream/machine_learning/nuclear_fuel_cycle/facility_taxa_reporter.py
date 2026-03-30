# src/workflow_16s/downstream/machine_learning/nuclear_fuel_cycle/facilty_taxa_reporter.py

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union
import plotly.express as px
from scipy.stats import spearmanr

from workflow_16s.utils.logger import get_logger


class FacilityMicrobeReporter:
    """
    Orchestrates the generation of facility-microbe association reports.
    
    This module synthesizes CatBoost feature selection results to identify 
    microbial biomarkers predictive of nuclear facility types, operational 
    statuses, and environmental impacts.
    """
    
    def __init__(
        self, 
        catboost_results_dir: Union[str, Path], 
        output_dir: Union[str, Path], 
        adata: Optional[Any] = None
    ):
        """
        Initialize the Forensic Reporter.

        Args:
            catboost_results_dir: Path to the root of CatBoost selection outputs.
            output_dir: Destination for the generated reports and plots.
            adata: Optional AnnData object for raw abundance correlation checks (Required for Gradients).
        """
        self.catboost_dir = Path(catboost_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.adata = adata
        
        self.facility_targets = [
            'nuclear_contamination_status',
            'facility',
            'facility_match',
            'facility_type',
            'facility_status',
            'facility_capacity',
            'facility_distance_km'
        ]

    def load_target_results(
        self, 
        target: str, 
        level: str = 'Genus'
    ) -> Optional[Dict]:
        """Loads the results_summary.json for a given target and taxonomic level."""
        # Check standard flat structure: {level}_{target}/results_summary.json
        results_file = self.catboost_dir / f"{level}_{target}" / "results_summary.json"
        
        # Fallback: Check hierarchical structure: {level}/{transform}/{target}/results_summary.json
        if not results_file.exists():
            potential_files = list(self.catboost_dir.glob(f"**/{level}/**/{target}/results_summary.json"))
            if potential_files:
                results_file = potential_files[0]

        if results_file.exists():
            try:
                with open(results_file) as f:
                    return json.load(f)
            except Exception as e:
                logger = get_logger("workflow_16s")
                logger.error(f"［］Error parsing results for {target}: {e}")
        return None

    def extract_all_associations(self, n: int = 10) -> pd.DataFrame:
        """Extracts and aggregates forensic associations across all targets and ranks."""
        records = []
        ranks = ['Genus', 'Family', 'Order', 'Class', 'Phylum']
        
        for target in self.facility_targets:
            for level in ranks:
                data = self.load_target_results(target, level)
                if data:
                    test_scores = data.get('test_scores', {})
                    records.append({
                        'target': target,
                        'taxonomic_level': level,
                        'task_type': data.get('task_type'),
                        'cv_mcc': data.get('best_cv_score'),
                        'test_mcc': test_scores.get('mcc'),
                        'test_accuracy': test_scores.get('accuracy'),
                        'n_features': len(data.get('top_features', [])),
                        'top_features': ', '.join([f.strip() for f in data.get('top_features', [])[:n]])
                    })
        
        df = pd.DataFrame(records)
        if not df.empty:
            summary_file = self.output_dir / "facility_microbe_summary.csv"
            df.to_csv(summary_file, index=False)
            logger = get_logger("workflow_16s")
            logger.info(f"［🔍］Extracted {len(df)} facility-target associations to {summary_file}.")
        return df

    def create_capacity_gradient_analysis(
        self, 
        level: str = 'Genus'
    ) -> None:
        """
        Correlates top taxa with facility power capacity (MW) to find scaling indicators.
        """
        logger = get_logger("workflow_16s")
        if self.adata is None:
            logger.warning("［］AnnData missing; skipping Capacity Gradient Analysis.")
            return

        data = self.load_target_results('facility_capacity', level)
        if not data or 'top_features' not in data: return

        top_taxa = [f.strip() for f in data['top_features'][:20]]
        
        # Ensure capacity is numeric
        obs_df = self.adata.obs.copy()
        obs_df['cap_numeric'] = pd.to_numeric(obs_df['facility_capacity'], errors='coerce')
        valid_idx = obs_df['cap_numeric'].dropna().index
        
        correlations = []
        for taxon in top_taxa:
            if taxon not in self.adata.var_names: continue
            
            # Extract abundance across valid samples
            abundance = self.adata[valid_idx, taxon].X
            if hasattr(abundance, "toarray"): abundance = abundance.toarray().flatten()
            
            rho, pval = spearmanr(abundance, obs_df.loc[valid_idx, 'cap_numeric'])
            correlations.append({
                'Taxon': taxon.split('__')[-1],
                'Spearman_Rho': rho,
                'p_value': pval
            })

        corr_df = pd.DataFrame(correlations).sort_values('Spearman_Rho', ascending=False)
        
        fig = px.bar(
            corr_df, x='Spearman_Rho', y='Taxon', color='Spearman_Rho',
            color_continuous_scale='RdBu_r', range_color=[-1, 1],
            title=f"Forensic Scaling: Microbial Abundance vs. Facility Capacity (MW)",
            labels={'Spearman_Rho': 'Spearman Correlation (ρ)'},
            template='plotly_white'
        )
        fig.write_html(self.output_dir / "facility_capacity_gradients.html")
        logger.info(f"［💾］Capacity Gradient Analysis complete. "
                    f"Plot saved to {self.output_dir / 'facility_capacity_gradients.html'}.")

    def create_microbe_network_plot(
        self, 
        microbe_lists: Dict[str, pd.DataFrame]
    ) -> None:
        """Visualizes taxa that serve as indicators for multiple different facility targets."""
        all_microbes = {}
        for target, df in microbe_lists.items():
            for _, row in df.iterrows():
                genus = row['genus'].strip()
                if genus not in all_microbes: all_microbes[genus] = []
                all_microbes[genus].append((target, row['rank']))
        
        multi_target = {g: tgs for g, tgs in all_microbes.items() if len(tgs) > 1}
        if not multi_target: return

        plot_data = []
        for genus, targets in sorted(multi_target.items(), key=lambda x: len(x[1]), reverse=True)[:30]:
            for target, rank in targets:
                plot_data.append({
                    'Taxon': genus.replace('g__', ''),
                    'Forensic Target': target.replace('_', ' ').title(),
                    'Rank': rank,
                    'Importance': 21 - rank
                })
        
        fig = px.scatter(
            pd.DataFrame(plot_data), x='Forensic Target', y='Taxon',
            size='Importance', color='Rank', color_continuous_scale='RdYlGn_r',
            title="Core Forensic Indicators: Multi-Target Diagnostic Microbes"
        )
        fig.update_layout(xaxis_tickangle=-45, height=800, template='plotly_white')
        fig.write_html(self.output_dir / "multi_target_diagnostic_taxa.html")
        logger = get_logger("workflow_16s")
        logger.info(f"［💾］Multi-Target Diagnostic Taxa plot saved to {self.output_dir / 'multi_target_diagnostic_taxa.html'}.")
    
    def generate_summary_text(
        self, 
        summary_df: pd.DataFrame
    ) -> None:
        """Generates a human-readable interpretation report."""
        lines = ["="*80, "FACILITY-MICROBE FORENSIC INTERPRETATION", "="*80, ""]
        
        # Hardcoded expert interpretations for nuclear fuel cycle taxa
        interpretations = {
            'nuclear_contamination_status': ["Sphingomonas: Known for radiation tolerance and bioremediation potentials."],
            'facility_type': ["Aquicella: Water-associated, often found in industrial cooling systems."]
        }

        for target in self.facility_targets[:5]:
            subset = summary_df[(summary_df['target'] == target) & (summary_df['taxonomic_level'] == 'Genus')]
            if subset.empty: continue
            
            row = subset.iloc[0]
            lines.append(f"\nTARGET: {target.replace('_', ' ').title()}")
            lines.append(f"  Accuracy: {row['test_accuracy']:.1%} | MCC: {row['test_mcc']:.3f}")
            lines.append(f"  Top Diagnostic Genera: {row['top_features']}")
            
            if target in interpretations:
                lines.append("  Expert Context:")
                for note in interpretations[target]: lines.append(f"    • {note}")

        with open(self.output_dir / "forensic_interpretation_report.txt", 'w') as f:
            f.write("\n".join(lines))

    def generate_report(self) -> None:
        """Executes the complete reporting suite."""
        logger = get_logger("workflow_16s")
        logger.info("Generating Facility-Microbe Forensics Report...")
        
        summary_df = self.extract_all_associations()
        if summary_df.empty:
            logger.warning("No associations found. Report aborted.")
            return

        microbe_lists = {}
        for target in self.facility_targets:
            data = self.load_target_results(target, 'Genus')
            if data and 'top_features' in data:
                microbe_lists[target] = pd.DataFrame({
                    'genus': [f.strip() for f in data['top_features']],
                    'rank': range(1, len(data['top_features']) + 1)
                })

        self.generate_summary_text(summary_df)
        self.create_microbe_network_plot(microbe_lists)
        self.create_capacity_gradient_analysis(level='Genus')
        
        logger.info(f"［💾］Facility-Microbe Forensics Suite Complete. "
                    f"Results in: {self.output_dir}")

def run_facility_microbe_report(
    catboost_dir: Path, 
    adata: Any, 
    output_dir: Optional[Path] = None
) -> None:
    """Main entry point for the forensics reporting suite."""
    out = output_dir or catboost_dir.parent / "facility_microbe_report"
    reporter = FacilityMicrobeReporter(catboost_dir, out, adata=adata)
    reporter.generate_report()