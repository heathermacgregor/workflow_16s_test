"""
Comprehensive QC Pipeline Integration

Orchestrates all quality control modules:
1. Metadata validation and cleaning
2. Primer quality control
3. Sample identity validation
4. Contamination detection
5. Report generation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import anndata as ad

from .contamination import (
    detect_contaminants_reference_based,
    detect_cross_sample_contamination,
    remove_contaminants as _remove_contaminants
)
from .primer_qc import PrimerQC
from .validation import (
    validate_config,
    validate_metadata,
    validate_adata,
    ENVOOntology,
    MetadataValidator,
    SampleIdentityValidator,
    QCValidationError
)
from workflow_16s.utils.logger import get_logger
logger = get_logger("workflow_16s")


class ComprehensiveQC:
    """
    Unified QC pipeline for workflow_16s.
    
    Runs all validation steps and generates comprehensive reports.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize QC pipeline.
        
        Args:
            config: Optional configuration dict with QC parameters
        """
        self.config = config or {}
        self.results = {}
        
        # Validate configuration
        qc_config = self.config.get('quality_control', {})
        if qc_config:
            is_valid, errors = validate_config(qc_config)
            if not is_valid:
                logger.warning(f"QC configuration issues: {errors}")
                logger.warning("Continuing with default values where possible")
        
    def run_metadata_qc(
        self,
        metadata: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run comprehensive metadata validation.
        
        Args:
            metadata: Metadata DataFrame
            output_dir: Optional output directory for reports
        
        Returns:
            Dict with 'cleaned_metadata', 'report', 'n_removed_columns'
        """
        logger.info("="*80)
        logger.info("METADATA QC")
        logger.info("="*80)
        
        # Validate input
        is_valid, errors = validate_metadata(metadata)
        if not is_valid:
            logger.error(f"［］Metadata validation failed: {errors}")
            return {
                'cleaned_metadata': metadata, 
                'report': pd.DataFrame(), 
                'n_removed_columns': 0
            }
        
        try:
            validator = MetadataValidator(metadata, self.config)
            cleaned_metadata, report = validator.validate_all()
            
            n_removed = len(metadata.columns) - len(cleaned_metadata.columns)
            
            # Save report if output_dir provided
            if output_dir:
                try:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    report_path = output_dir / 'metadata_validation_report.csv'
                    report.to_csv(report_path, index=False)
                    logger.info(f"［］Saved metadata validation report: {report_path}")
                except Exception as e:
                    logger.warning(f"［］Could not save metadata report: {e}")
            
            self.results['metadata_validation'] = report
            
            return {
                'cleaned_metadata': cleaned_metadata,
                'report': report,
                'n_removed_columns': n_removed
            }
        
        except Exception as e:
            logger.error(f"［］Metadata QC failed: {e}", exc_info=True)
            return {'cleaned_metadata': metadata, 'report': pd.DataFrame(), 'n_removed_columns': 0}
    
    def run_primer_qc(
        self, 
        fastq_files: Sequence[Union[str, Path]], 
        primers: Dict[str, str],
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Run primer quality control on FASTQ files.
        
        Args:
            fastq_files: List of FASTQ file paths
            primers: Dict of primer_name -> sequence
            output_dir: Optional directory for reports
        
        Returns:
            DataFrame with primer QC results
        """
        logger.info("="*80)
        logger.info("PRIMER QC")
        logger.info("="*80)
        
        primer_qc = PrimerQC(
            primers=primers,
            max_error_rate=self.config.get('primer_max_error_rate', 0.15),
            max_reads=self.config.get('primer_max_reads', 10000),
            n_cores=self.config.get('n_cores', 4)
        )
        
        # Run batch check
        report_path = output_dir / 'primer_qc_report.html' if output_dir else None
        results = primer_qc.batch_check(list(fastq_files), output_report=report_path)
        
        self.results['primer_qc'] = results
        
        return results
    
    def run_sample_validation(
        self, 
        adata: ad.AnnData,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run sample identity validation.
        
        Args:
            adata: AnnData object
            output_dir: Optional output directory
        
        Returns:
            Dict with 'validation_df' and summary statistics
        """
        logger.info("="*80)
        logger.info("SAMPLE VALIDATION")
        logger.info("="*80)

        # Ensure ENVO categorization is done
        if 'env_category_type' not in adata.obs.columns:
            logger.info("Running ENVO categorization first...")
            envo = ENVOOntology()
            categories = adata.obs.apply(
                lambda row: envo.categorize_sample(
                    row.get('env_biome'),
                    row.get('env_feature'),
                    row.get('env_material')
                ),
                axis=1
            )
            adata.obs['env_category_type'] = categories.apply(lambda x: x['category'])
            adata.obs['env_category_confidence'] = categories.apply(lambda x: x['confidence'])
        
                # Validate input
        is_valid, errors = validate_adata(adata)
        if not is_valid:
            logger.error(f"［］AnnData validation failed: {errors}")
            return {'validation_df': pd.DataFrame()}
        
        try:
            validator = SampleIdentityValidator(adata)
            validation_df = validator.validate_all()
            
            # Save report if output_dir provided
            if output_dir:
                try:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    report_path = output_dir / 'sample_validation_report.csv'
                    validation_df.to_csv(report_path)
                    logger.info(f"［］Saved sample validation report: {report_path}")
                except Exception as e:
                    logger.warning(f"［］Could not save sample validation report: {e}")
            
            self.results['sample_validation'] = validation_df
            
            return {'validation_df': validation_df}
        
        except Exception as e:
            logger.error(f"［］Sample validation failed: {e}", exc_info=True)
            return {'validation_df': pd.DataFrame()}
    
    def run_contamination_detection(
        self, 
        adata: ad.AnnData,
        method: str = 'combined',
        remove_contaminants: bool = False,
        threshold: float = 0.5
    ) -> Tuple[ad.AnnData, pd.DataFrame]:
        """
        Run contamination detection.
        
        Args:
            adata: AnnData object
            method: Detection method ('database', 'frequency', 'ubiquity', 'combined')
            remove_contaminants: Whether to remove detected contaminants
            threshold: Score threshold for removal
        
        Returns:
            Tuple of (cleaned AnnData, contamination scores)
        """
        logger.info("="*80,
                    "\nCONTAMINATION DETECTION",
                    "="*80)
        
        # Determine environment types to exclude from human contamination checks
        exclude_env = []
        if 'env_category_type' in adata.obs.columns:
            env_types = adata.obs['env_category_type'].unique()
            # Don't flag human taxa if we have human-associated samples
            if any(env in env_types for env in ['gut', 'skin', 'oral']):
                exclude_env = ['gut', 'skin', 'oral']
        
        # Run reference-based detection
        contam_scores = detect_contaminants_reference_based(
            adata,
            method=method,
            exclude_env_types=exclude_env
        )
        
        # Check for cross-sample contamination
        if 'batch' in adata.obs.columns or 'dataset_id' in adata.obs.columns:
            batch_col = 'batch' if 'batch' in adata.obs.columns else 'dataset_id'
            cross_contam = detect_cross_sample_contamination(adata, batch_column=batch_col)
            
            if cross_contam:
                logger.warning(f"［］Detected potential cross-contamination in {len(cross_contam)} batches")
                self.results['cross_contamination'] = cross_contam
        
        self.results['contamination_scores'] = contam_scores
        
        # Remove contaminants if requested
        if remove_contaminants:
            adata_clean = _remove_contaminants(
                adata,
                contam_scores,
                threshold=threshold,
                inplace=False
            )
            return adata_clean, contam_scores
        
        return adata, contam_scores
    
    def run_all(
        self, 
        adata: ad.AnnData,
        fastq_files: Optional[List[Path]] = None,
        primers: Optional[Dict[str, str]] = None,
        output_dir: Optional[Path] = None,
        remove_contaminants: bool = True
    ) -> Tuple[ad.AnnData, Dict[str, Any]]:
        """
        Run complete QC pipeline.
        
        Args:
            adata: AnnData object with metadata and features
            fastq_files: Optional list of FASTQ files for primer QC
            primers: Optional dict of primers (required if fastq_files provided)
            output_dir: Optional output directory for reports
            remove_contaminants: Whether to remove detected contaminants
        
        Returns:
            Tuple of (cleaned AnnData, results dict)
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE QC PIPELINE")
        logger.info("="*80)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metadata QC
        logger.info("［1/4］Metadata Validation")
        cleaned_metadata, metadata_report = self.run_metadata_qc(adata.obs)
        adata.obs = cleaned_metadata
        
        if output_dir:
            metadata_report.to_csv(output_dir / 'metadata_validation_report.csv', index=False)
        
        # 2. Primer QC (if FASTQ files provided)
        if fastq_files and primers:
            logger.info("［2/4］Primer Quality Control")
            primer_results = self.run_primer_qc(fastq_files, primers, output_dir)
            
            # Flag samples with poor primer detection
            if output_dir:
                primer_results.to_csv(output_dir / 'primer_qc_results.csv', index=False)
        else:
            logger.info("［2/4］Primer QC skipped (no FASTQ files provided)")
        
        # 3. Sample Identity Validation
        logger.info("［3/4］Sample Identity Validation")
        sample_validation_result = self.run_sample_validation(adata)
        sample_validation = sample_validation_result['validation_df']
        
        # Add validation results to obs
        for col in sample_validation.columns:
            adata.obs[f'qc_{col}'] = sample_validation[col]
        
        if output_dir:
            sample_validation.to_csv(output_dir / 'sample_validation_report.csv')
        
        # 4. Contamination Detection
        logger.info("［4/4］Contamination Detection")
        adata, contam_scores = self.run_contamination_detection(
            adata,
            method='combined',
            remove_contaminants=remove_contaminants,
            threshold=0.5
        )
        
        if output_dir:
            contam_scores.to_csv(output_dir / 'contamination_scores.csv')
        
        # Generate summary report
        summary = self._generate_summary()
        
        if output_dir:
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(output_dir / 'qc_summary.csv', index=False)
            
            # Generate comprehensive HTML report
            self._generate_html_report(output_dir / 'qc_report.html')
        
        logger.info("="*80)
        logger.info("QC PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Samples: {len(adata.obs)}\n"
                    f"Features: {len(adata.var)}\n"
                    f"Flagged samples: {(adata.obs['qc_overall_flag'] != 'PASS').sum()}\n"
                    f"Removed contaminants: {contam_scores['is_contaminant'].sum()}")
        
        return adata, self.results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from all QC steps."""
        summary = {}
        
        # Metadata validation
        if 'metadata_validation' in self.results:
            report = self.results['metadata_validation']
            summary['metadata_errors'] = (report['level'] == 'ERROR').sum()
            summary['metadata_warnings'] = (report['level'] == 'WARNING').sum()
        
        # Sample validation
        if 'sample_validation' in self.results:
            val = self.results['sample_validation']
            summary['samples_flagged_warning'] = (val['overall_flag'] == 'WARNING').sum()
            summary['samples_flagged_fail'] = (val['overall_flag'] == 'FAIL').sum()
        
        # Contamination
        if 'contamination_scores' in self.results:
            contam = self.results['contamination_scores']
            summary['features_flagged_contaminant'] = contam['is_contaminant'].sum()
            summary['mean_contamination_score'] = contam['combined_score'].mean()
        
        return summary
    
    def _generate_html_report(self, output_path: Path):
        """Generate comprehensive HTML QC report."""
        # TODO: Implement full HTML report with visualizations
        # For now, create a simple summary
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>workflow_16s QC Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .pass {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>workflow_16s Comprehensive QC Report</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        summary = self._generate_summary()
        for key, value in summary.items():
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>\n"
        
        html += """
            </table>
            
            <h2>Details</h2>
            <p>See individual CSV reports for detailed results:</p>
            <ul>
                <li>metadata_validation_report.csv</li>
                <li>sample_validation_report.csv</li>
                <li>contamination_scores.csv</li>
                <li>qc_summary.csv</li>
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report: {output_path}")


def quick_qc(
    adata: ad.AnnData, 
    output_dir: Optional[Path] = None,
    remove_contaminants: bool = True
) -> ad.AnnData:
    """
    Quick QC function for easy integration.
    
    Args:
        adata: AnnData object
        output_dir: Optional output directory
        remove_contaminants: Whether to remove contaminants
    
    Returns:
        Cleaned AnnData object
    
    Example:
        >>> from workflow_16s.qc import quick_qc
        >>> adata_clean = quick_qc(adata, output_dir='qc_results')
    """
    qc = ComprehensiveQC()
    adata_clean, results = qc.run_all(
        adata,
        output_dir=output_dir,
        remove_contaminants=remove_contaminants
    )
    return adata_clean
