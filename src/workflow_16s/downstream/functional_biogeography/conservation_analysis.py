"""
Conservation Analysis

Integrates functional traits and phylogenetic signal to answer the core question:
"What defines adaptive functionality across different environments?"

Quantifies the relationship between:
1. Taxonomic conservation (functions that phylogenetically cluster)
2. Locale conservation (functions that repeat in similar environments)
3. Adaptation patterns (which environments drive which functions)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from .functional_trait_mapping import create_trait_matrix
from .phylogenetic_signal import (
    calculate_phylogenetic_signal,
    assess_trait_phylogenetic_structure
)

logger = logging.getLogger(__name__)


class ConservationAnalyzer:
    """
    Comprehensive analysis of functional/taxonomic conservation patterns.
    """
    
    def __init__(
        self,
        adata,
        otu_metadata_path: Optional[str] = None,
        otu_level: int = 99,
        user_email: str = "macgregor@berkeley.edu",
        use_jgi: bool = True,
        taxonomy_columns: Optional[List[str]] = None
    ):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object with OTU/sample data
        otu_metadata_path : str, optional
            Path to OTU metadata file (e.g., otus.97.allinfo or otus.99.allinfo)
        otu_level : int
            OTU clustering level (97, 99, etc.) for reference/logging
        user_email : str
            Berkeley email for JGI database access
        use_jgi : bool
            Whether to use JGI/KEGG database for trait definitions
        taxonomy_columns : List[str], optional
            Names of taxonomy columns in adata.var
        """
        self.adata = adata
        self.otu_metadata_path = otu_metadata_path
        self.otu_level = otu_level
        self.user_email = user_email
        self.use_jgi = use_jgi
        self.taxonomy_columns = taxonomy_columns or [
            'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'
        ]
        
        # Extract taxonomy data
        self.taxonomy_data = self._extract_taxonomy()
        
        # Will be populated by analysis
        self.trait_matrix = None
        self.phylogenetic_results = None
        self.environmental_associations = None
    
    def _extract_taxonomy(self) -> pd.DataFrame:
        """Extract taxonomy from adata.var into dataframe."""
        if not hasattr(self.adata, 'var') or self.adata.var.empty:
            logger.warning("No var data in adata")
            return pd.DataFrame()
        
        tax_cols = [c for c in self.taxonomy_columns if c in self.adata.var.columns]
        
        if not tax_cols:
            logger.warning("No taxonomy columns found in adata.var")
            return pd.DataFrame()
        
        taxonomy_df = self.adata.var[tax_cols].copy()
        return taxonomy_df
    
    def run_analysis(
        self,
        confidence_threshold: float = 0.5,
        environmental_variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete conservation analysis.
        
        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence for trait assignment
        environmental_variable : str, optional
            Metadata variable to correlate with traits (e.g., 'pH', 'metal_concentration')
        
        Returns
        -------
        Dict[str, Any]
            Complete analysis results
        """
        logger.info("Starting conservation analysis...")
        
        # Step 1: Create trait matrix
        logger.info("Step 1: Creating trait matrix...")
        self.trait_matrix, trait_db = create_trait_matrix(
            self.adata,
            otu_metadata_path=self.otu_metadata_path,
            user_email=self.user_email,
            use_jgi=self.use_jgi
        )
        
        # Step 2: Calculate phylogenetic signal
        logger.info("Step 2: Calculating phylogenetic signal...")
        self.phylogenetic_results = calculate_phylogenetic_signal(
            self.trait_matrix,
            self.taxonomy_data,
            threshold=confidence_threshold
        )
        
        # Step 3: Assess full phylogenetic structure
        logger.info("Step 3: Assessing phylogenetic structure...")
        phylo_assessment = assess_trait_phylogenetic_structure(
            self.trait_matrix,
            self.taxonomy_data
        )
        
        # Step 4: Analyze environmental associations (if available)
        logger.info("Step 4: Analyzing environmental associations...")
        environmental_results = None
        if environmental_variable and environmental_variable in self.adata.obs.columns:
            environmental_results = self._analyze_environmental_associations(
                environmental_variable
            )
        
        # Compile results
        results = {
            'trait_matrix': self.trait_matrix,
            'phylogenetic_signal': self.phylogenetic_results,
            'phylogenetic_assessment': phylo_assessment,
            'environmental_associations': environmental_results,
            'trait_database': trait_db
        }
        
        return results
    
    def _analyze_environmental_associations(
        self,
        environmental_variable: str
    ) -> Dict[str, Any]:
        """
        Analyze which traits are associated with specific environments.
        
        Parameters
        ----------
        environmental_variable : str
            Column name in adata.obs
        
        Returns
        -------
        Dict[str, Any]
            Trait-environment associations
        """
        associations = {}
        
        if not hasattr(self.adata, 'obs') or environmental_variable not in self.adata.obs.columns:
            logger.warning(f"Environmental variable '{environmental_variable}' not found")
            return associations
        
        # For each trait, calculate correlation with environment
        for trait in self.trait_matrix.columns:
            trait_presence = self.trait_matrix[trait]
            env_var = self.adata.obs[environmental_variable]
            
            # Filter out NaN values
            valid_idx = ~(trait_presence.isna() | env_var.isna())
            
            if valid_idx.sum() < 2:
                continue
            
            trait_clean = trait_presence[valid_idx]
            env_clean = env_var[valid_idx]
            
            # Calculate correlation
            try:
                corr = np.corrcoef(trait_clean, env_clean)[0, 1]
                if not np.isnan(corr):
                    associations[trait] = {
                        'correlation': corr,
                        'n_samples': valid_idx.sum()
                    }
            except Exception as e:
                logger.debug(f"Could not correlate {trait} with {environmental_variable}: {e}")
        
        return associations
    
    def summarize_results(self) -> str:
        """
        Generate human-readable summary of analysis.
        
        Returns
        -------
        str
            Formatted summary text
        """
        if self.phylogenetic_results is None:
            return "Analysis not yet run"
        
        summary = [
            "=" * 80,
            "CONSERVATION ANALYSIS SUMMARY",
            "=" * 80,
            ""
        ]
        
        # Phylogenetic signal summary
        summary.append("PHYLOGENETIC SIGNAL (Pagel's Lambda)")
        summary.append("-" * 80)
        summary.append(f"Total traits analyzed: {len(self.phylogenetic_results)}")
        
        conserved = self.phylogenetic_results[self.phylogenetic_results['pagels_lambda'] > 0.7]
        random = self.phylogenetic_results[self.phylogenetic_results['pagels_lambda'] < 0.3]
        
        summary.append(f"\nPhylogenetically Conserved (λ > 0.7):  {len(conserved)} traits")
        for _, row in conserved.iterrows():
            summary.append(f"  • {row['trait']:30s} λ={row['pagels_lambda']:.3f} ({row['n_otus_with_trait']}/{row['n_otus_total']} OTUs)")
        
        summary.append(f"\nRandomly Distributed (λ < 0.3):       {len(random)} traits")
        for _, row in random.iterrows():
            summary.append(f"  • {row['trait']:30s} λ={row['pagels_lambda']:.3f} ({row['n_otus_with_trait']}/{row['n_otus_total']} OTUs)")
        
        summary.append("")
        summary.append("INTERPRETATION")
        summary.append("-" * 80)
        summary.append("Conserved traits (high λ):")
        summary.append("  → Follow the evolutionary tree, likely vertically inherited")
        summary.append("  → Core functions defining major taxonomic groups")
        summary.append("")
        summary.append("Random traits (low λ):")
        summary.append("  → Distributed across distantly related taxa")
        summary.append("  → Strong evidence of horizontal gene transfer")
        summary.append("  → Likely adaptive responses to local environmental pressures")
        
        return "\n".join(summary)


def analyze_functional_vs_taxonomic_conservation(
    adata,
    otu_metadata_path: Optional[str] = None,
    otu_level: int = 99,
    user_email: str = "macgregor@berkeley.edu",
    use_jgi: bool = True,
    environmental_variable: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level function to run complete conservation analysis.
    
    This directly answers Adam's question:
    "What is the difference between taxonomic conservation of function 
     vs locale conservation of function?"
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    otu_metadata_path : str, optional
        Path to OTU metadata file (e.g., otus.97.allinfo or otus.99.allinfo)
    otu_level : int
        OTU clustering level (97, 99, etc.) for reference/logging
    user_email : str
        Berkeley email for JGI database access
    use_jgi : bool
        Whether to use JGI/KEGG database integration
    environmental_variable : str, optional
        Metadata variable to analyze (e.g., 'pH', 'metal_concentration')
    output_dir : str, optional
        Directory to save results
    
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    analyzer = ConservationAnalyzer(
        adata,
        otu_metadata_path,
        otu_level,
        user_email,
        use_jgi
    )
    
    results = analyzer.run_analysis(
        environmental_variable=environmental_variable
    )
    
    # Print summary
    logger.info("\n" + analyzer.summarize_results())
    
    # Save results if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save phylogenetic results
        results['phylogenetic_signal'].to_csv(
            output_path / 'phylogenetic_signal.csv',
            index=False
        )
        
        # Save trait matrix
        results['trait_matrix'].to_csv(
            output_path / 'otu_trait_matrix.csv'
        )
        
        # Save summary text
        with open(output_path / 'analysis_summary.txt', 'w') as f:
            f.write(analyzer.summarize_results())
        
        logger.info(f"Results saved to {output_dir}")
    
    return results


def generate_conservation_report(
    analysis_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate detailed HTML/text report from analysis results.
    
    Parameters
    ----------
    analysis_results : Dict[str, Any]
        Results from conservation analysis
    output_path : str
        Path to save report
    """
    phylo_results = analysis_results['phylogenetic_signal']
    phylo_assessment = analysis_results['phylogenetic_assessment']
    
    report_lines = [
        "# Functional-Taxonomic Conservation Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"**Total OTUs analyzed:** {phylo_assessment['total_traits_analyzed']}",
        f"**Phylogenetically conserved traits:** {phylo_assessment['phylogenetically_conserved_traits']}", 
        f"**Randomly distributed traits:** {phylo_assessment['randomly_distributed_traits']}",
        f"**Mixed pattern traits:** {phylo_assessment['mixed_pattern_traits']}",
        "",
        "## Key Findings",
        "",
        "### Conserved Functions (Vertical Inheritance)",
        f"Mean Pagel's λ: {phylo_assessment['mean_lambda']:.3f}",
        "",
    ]
    
    for trait in phylo_assessment['conserved_trait_names'][:10]:
        report_lines.append(f"- {trait}")
    
    report_lines.extend([
        "",
        "### Randomly Distributed Functions (Horizontal Transfer)",
        "",
    ])
    
    for trait in phylo_assessment['random_trait_names'][:10]:
        report_lines.append(f"- {trait}")
    
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "Functions that follow the evolutionary tree (high λ) are core to their lineages.",
        "Functions randomly distributed across taxa suggest adaptation to local conditions",
        "via horizontal gene transfer or convergent evolution.",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {output_path}")
