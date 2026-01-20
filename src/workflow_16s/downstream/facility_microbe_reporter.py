"""
Generate facility-microbe association reports from CatBoost feature selection results.

This module extracts and summarizes which microbes are diagnostic for:
- Nuclear contamination
- Specific facilities
- Facility types (PWR, BWR, etc.)
- Facility operational status
- Facility capacity
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from workflow_16s.utils.logger import get_logger

logger = get_logger("workflow_16s")


class FacilityMicrobeReporter:
    """Extract and visualize facility-microbe associations from CatBoost results."""
    
    def __init__(self, catboost_results_dir: Path, output_dir: Path):
        """
        Initialize reporter.
        
        Args:
            catboost_results_dir: Path to catboost_feature_selection directory
            output_dir: Where to save reports
        """
        self.catboost_dir = Path(catboost_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.facility_targets = [
            'nuclear_contamination_status',
            'facility',
            'facility_match',
            'facility_type',
            'facility_status',
            'facility_capacity',
            'facility_distance_km'
        ]
        
    def load_target_results(self, target: str, level: str = 'Genus') -> Optional[Dict]:
        """Load results for a specific target."""
        results_file = self.catboost_dir / f"{level}_{target}" / "results_summary.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return None
    
    def extract_all_associations(self) -> pd.DataFrame:
        """Extract microbe associations for all facility targets."""
        records = []
        
        for target in self.facility_targets:
            for level in ['Genus', 'Class', 'Family', 'Order', 'Phylum']:
                data = self.load_target_results(target, level)
                if data:
                    test_scores = data.get('test_scores', {})
                    records.append({
                        'target': target,
                        'taxonomic_level': level,
                        'task_type': data.get('task_type'),
                        'cv_mcc': data.get('best_cv_score'),
                        'test_accuracy': test_scores.get('accuracy'),
                        'test_mcc': test_scores.get('mcc'),
                        'test_f1': test_scores.get('f1'),
                        'n_features': len(data.get('top_features', [])),
                        'top_features': ', '.join([f.strip() for f in data.get('top_features', [])[:10]])
                    })
        
        df = pd.DataFrame(records)
        summary_file = self.output_dir / "facility_microbe_summary.csv"
        df.to_csv(summary_file, index=False)
        logger.info(f"Saved facility-microbe summary: {summary_file}")
        return df
    
    def create_microbe_lists(self) -> Dict[str, pd.DataFrame]:
        """Create individual CSV files for each target's top microbes."""
        microbe_lists = {}
        
        for target in self.facility_targets:
            data = self.load_target_results(target, 'Genus')
            if data and data.get('top_features'):
                microbes_df = pd.DataFrame({
                    'rank': range(1, len(data['top_features']) + 1),
                    'genus': [f.strip() for f in data['top_features']],
                    'target': target,
                    'mcc': data.get('best_cv_score')
                })
                
                output_file = self.output_dir / f"{target}_microbes.csv"
                microbes_df.to_csv(output_file, index=False)
                microbe_lists[target] = microbes_df
                logger.info(f"  → {target}: {len(microbes_df)} microbes")
        
        return microbe_lists
    
    def generate_summary_text(self) -> str:
        """Generate a text summary of facility-microbe associations."""
        lines = []
        lines.append("=" * 80)
        lines.append("FACILITY-MICROBE ASSOCIATIONS SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Key findings for each target
        target_descriptions = {
            'nuclear_contamination_status': 'Nuclear Contamination Indicators',
            'facility': 'Facility-Specific Signatures',
            'facility_match': 'Facility Proximity Markers',
            'facility_type': 'Reactor Type Discriminators',
            'facility_status': 'Operational Status Indicators',
            'facility_capacity': 'Capacity-Associated Microbes'
        }
        
        for target in self.facility_targets[:6]:  # Skip distance (regression)
            data = self.load_target_results(target, 'Genus')
            if data:
                desc = target_descriptions.get(target, target.replace('_', ' ').title())
                lines.append(f"\n{desc}")
                lines.append("-" * 80)
                
                cv_mcc = data.get('best_cv_score', 0)
                test_acc = data.get('test_scores', {}).get('accuracy', 0)
                test_mcc = data.get('test_scores', {}).get('mcc', 0)
                
                lines.append(f"  Model Performance:")
                lines.append(f"    CV MCC: {cv_mcc:.3f}")
                if test_mcc and str(test_mcc) != 'nan':
                    lines.append(f"    Test MCC: {test_mcc:.3f}")
                lines.append(f"    Test Accuracy: {test_acc:.1%}")
                
                lines.append(f"\n  Top 10 Diagnostic Genera:")
                for i, genus in enumerate(data.get('top_features', [])[:10], 1):
                    genus_clean = genus.strip().replace('g__', '')
                    lines.append(f"    {i:2d}. {genus_clean}")
                lines.append("")
        
        lines.append("\n" + "=" * 80)
        lines.append("BIOLOGICAL INTERPRETATION")
        lines.append("=" * 80)
        lines.append("")
        
        # Add interpretations
        interpretations = {
            'nuclear_contamination_status': [
                "Aquicella: Waterborne bacteria, may thrive in cooling water systems",
                "Cellulomonas: Cellulose-degrading, known radiation resistance",
                "Sphingomonas: Documented radiation tolerance, bioremediation",
                "Bryobacter: Acidobacteria, soil health indicator"
            ],
            'facility': [
                "NS5_marine_group / SAR86_clade: Coastal facility signatures",
                "Nocardia: Soil actinobacteria, inland facility marker",
                "WD2101_soil_group: Terrestrial environment indicator"
            ],
            'facility_type': [
                "Pandoraea: Opportunistic, water-associated",
                "Bifidobacterium: Typically gut-associated, unexpected finding",
                "Amphiplicatus: Environmental Proteobacteria"
            ]
        }
        
        for target, interp_list in interpretations.items():
            desc = target_descriptions.get(target, target)
            lines.append(f"\n{desc}:")
            for interp in interp_list:
                lines.append(f"  • {interp}")
        
        summary_text = "\n".join(lines)
        
        # Save to file
        text_file = self.output_dir / "facility_microbe_interpretation.txt"
        with open(text_file, 'w') as f:
            f.write(summary_text)
        
        logger.info(f"Saved interpretation: {text_file}")
        return summary_text
    
    def create_heatmap_visualization(self, summary_df: pd.DataFrame) -> None:
        """Create heatmap showing MCC scores across targets and taxonomic levels."""
        genus_df = summary_df[summary_df['taxonomic_level'] == 'Genus'].copy()
        
        if genus_df.empty:
            return
        
        # Prepare data for heatmap
        fig = go.Figure()
        
        # Sort by MCC
        genus_df = genus_df.sort_values('test_mcc', ascending=False)
        
        # Create horizontal bar chart of MCC scores
        fig = go.Figure(go.Bar(
            x=genus_df['test_mcc'],
            y=genus_df['target'].str.replace('_', ' ').str.title(),
            orientation='h',
            marker=dict(
                color=genus_df['test_mcc'],
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                colorbar=dict(title="MCC Score")
            ),
            text=[f"{val:.3f}" if pd.notna(val) else "N/A" for val in genus_df['test_mcc']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Facility Prediction Performance (Matthews Correlation Coefficient)",
            xaxis_title="MCC Score (higher = better)",
            yaxis_title="",
            height=400,
            template='plotly_white'
        )
        
        html_file = self.output_dir / "facility_prediction_performance.html"
        fig.write_html(html_file)
        logger.info(f"Saved performance plot: {html_file}")
    
    def create_microbe_network_plot(self, microbe_lists: Dict[str, pd.DataFrame]) -> None:
        """Create visualization showing which microbes are important for multiple targets."""
        # Find microbes that appear in multiple targets
        all_microbes = {}
        for target, df in microbe_lists.items():
            for _, row in df.iterrows():
                genus = row['genus'].strip()
                if genus not in all_microbes:
                    all_microbes[genus] = []
                all_microbes[genus].append((target, row['rank']))
        
        # Filter to microbes in multiple targets
        multi_target_microbes = {g: targets for g, targets in all_microbes.items() 
                                 if len(targets) > 1}
        
        if not multi_target_microbes:
            return
        
        # Create data for plotting
        plot_data = []
        for genus, targets in sorted(multi_target_microbes.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:20]:
            for target, rank in targets:
                plot_data.append({
                    'genus': genus.replace('g__', ''),
                    'target': target.replace('_', ' ').title(),
                    'rank': rank,
                    'importance': 21 - rank  # Reverse rank for plotting
                })
        
        if not plot_data:
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create bubble chart
        fig = px.scatter(
            plot_df,
            x='target',
            y='genus',
            size='importance',
            color='rank',
            color_continuous_scale='RdYlGn_r',
            title="Multi-Target Diagnostic Microbes",
            labels={'rank': 'Rank', 'importance': 'Importance'},
            height=max(600, len(plot_df['genus'].unique()) * 30)
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            template='plotly_white',
            xaxis={'tickangle': -45}
        )
        
        html_file = self.output_dir / "multi_target_microbes.html"
        fig.write_html(html_file)
        logger.info(f"Saved multi-target plot: {html_file}")
    
    def generate_report(self) -> None:
        """Generate complete facility-microbe association report."""
        logger.info("=" * 80)
        logger.info("Generating Facility-Microbe Association Report")
        logger.info("=" * 80)
        
        # Extract all associations
        logger.info("\n1. Extracting microbe associations...")
        summary_df = self.extract_all_associations()
        
        # Create individual microbe lists
        logger.info("\n2. Creating microbe lists per target...")
        microbe_lists = self.create_microbe_lists()
        
        # Generate summary text
        logger.info("\n3. Generating interpretation text...")
        summary_text = self.generate_summary_text()
        
        # Create visualizations
        logger.info("\n4. Creating visualizations...")
        self.create_heatmap_visualization(summary_df)
        self.create_microbe_network_plot(microbe_lists)
        
        # Print summary to log
        logger.info("\n" + "=" * 80)
        logger.info("REPORT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info(f"  - facility_microbe_summary.csv")
        logger.info(f"  - facility_microbe_interpretation.txt")
        logger.info(f"  - facility_prediction_performance.html")
        logger.info(f"  - multi_target_microbes.html")
        logger.info(f"  - {len(microbe_lists)} individual microbe lists")
        
        # Print key findings
        logger.info("\n" + "=" * 80)
        logger.info("KEY FINDINGS")
        logger.info("=" * 80)
        
        genus_df = summary_df[summary_df['taxonomic_level'] == 'Genus']
        for _, row in genus_df.iterrows():
            if row['test_mcc'] and pd.notna(row['test_mcc']) and row['test_mcc'] > 0.6:
                target_name = row['target'].replace('_', ' ').title()
                logger.info(f"\n{target_name}:")
                logger.info(f"  MCC: {row['test_mcc']:.3f} | Accuracy: {row['test_accuracy']:.1%}")
                microbes = [m.strip().replace('g__', '') for m in row['top_features'].split(',')][:5]
                logger.info(f"  Top 5: {', '.join(microbes)}")


def run_facility_microbe_report(catboost_dir: Path, output_dir: Path = None) -> None:
    """
    Main entry point for generating facility-microbe association report.
    
    Args:
        catboost_dir: Path to catboost_feature_selection directory
        output_dir: Where to save reports (default: catboost_dir/../facility_microbe_report)
    """
    if output_dir is None:
        output_dir = catboost_dir.parent / "facility_microbe_report"
    
    reporter = FacilityMicrobeReporter(catboost_dir, output_dir)
    reporter.generate_report()
