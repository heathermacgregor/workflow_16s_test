"""
Auto-tuning configuration utilities.

Automatically adjust analysis parameters based on dataset characteristics
for optimal performance and results quality.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("workflow_16s")


class AutoTuner:
    """Automatically tune analysis parameters based on dataset size."""
    
    def __init__(self):
        self.adjustments = {}
        self.original_config = {}
        
    def tune_parameters(
        self,
        n_samples: int,
        n_features: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tune analysis parameters based on dataset characteristics.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in dataset
        n_features : int
            Number of features (ASVs/taxa) in dataset
        config : dict
            Original configuration dictionary
            
        Returns
        -------
        dict
            Tuned configuration
        """
        tuned_config = config.copy()
        self.original_config = config.copy()
        
        logger.info(f"🎛️  Auto-tuning for {n_samples} samples × {n_features} features...")
        
        # Variance filtering threshold
        variance_threshold = self._tune_variance_threshold(n_features)
        if variance_threshold != config.get('variance_threshold', 1e-6):
            tuned_config['variance_threshold'] = variance_threshold
            self.adjustments['variance_threshold'] = {
                'original': config.get('variance_threshold', 1e-6),
                'tuned': variance_threshold,
                'reason': f'Optimized for {n_features} features'
            }
        
        # Top N plots/results
        top_n = self._tune_top_n(n_features)
        if top_n != config.get('top_n', 50):
            tuned_config['top_n'] = top_n
            self.adjustments['top_n'] = {
                'original': config.get('top_n', 50),
                'tuned': top_n,
                'reason': f'Scaled for {n_features} features'
            }
        
        # Minimum samples per group
        min_samples = self._tune_min_samples(n_samples)
        if min_samples != config.get('min_samples_per_group', 3):
            tuned_config['min_samples_per_group'] = min_samples
            self.adjustments['min_samples_per_group'] = {
                'original': config.get('min_samples_per_group', 3),
                'tuned': min_samples,
                'reason': f'Optimized for {n_samples} samples'
            }
        
        # Machine learning parameters
        ml_config = config.get('machine_learning', {})
        tuned_ml = self._tune_ml_parameters(n_samples, n_features, ml_config)
        if tuned_ml != ml_config:
            tuned_config['machine_learning'] = tuned_ml
        
        # Max plots threshold
        max_plots = self._tune_max_plots(n_features)
        if max_plots != config.get('max_plots', 1000):
            tuned_config['max_plots'] = max_plots
            self.adjustments['max_plots'] = {
                'original': config.get('max_plots', 1000),
                'tuned': max_plots,
                'reason': 'Prevent plotting bottleneck'
            }
        
        # Statistical test thresholds
        alpha = self._tune_alpha_threshold(n_features)
        if alpha != config.get('alpha', 0.05):
            tuned_config['alpha'] = alpha
            self.adjustments['alpha'] = {
                'original': config.get('alpha', 0.05),
                'tuned': alpha,
                'reason': 'Adjusted for multiple testing burden'
            }
        
        if self.adjustments:
            logger.info(f"✅ Applied {len(self.adjustments)} auto-tuning adjustments")
            for param, details in self.adjustments.items():
                logger.info(
                    f"  • {param}: {details['original']} → {details['tuned']} "
                    f"({details['reason']})"
                )
        else:
            logger.info("No auto-tuning adjustments needed")
        
        return tuned_config
    
    def _tune_variance_threshold(self, n_features: int) -> float:
        """
        Tune variance filtering threshold.
        
        More features → stricter threshold to reduce noise
        """
        if n_features > 100000:
            return 1e-5
        elif n_features > 50000:
            return 5e-6
        elif n_features > 10000:
            return 1e-6
        else:
            return 5e-7
    
    def _tune_top_n(self, n_features: int) -> int:
        """
        Tune number of top results to report/plot.
        
        Scale with feature count but cap at reasonable limit
        """
        # Use log scale to avoid explosion
        base_n = int(50 * np.log10(max(n_features, 10)))
        return min(max(base_n, 50), 500)
    
    def _tune_min_samples(self, n_samples: int) -> int:
        """
        Tune minimum samples per group for statistical tests.
        
        More samples → stricter requirement for robust statistics
        """
        if n_samples > 10000:
            return 10
        elif n_samples > 5000:
            return 7
        elif n_samples > 1000:
            return 5
        else:
            return 3
    
    def _tune_ml_parameters(
        self,
        n_samples: int,
        n_features: int,
        ml_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tune machine learning parameters."""
        tuned = ml_config.copy()
        
        # n_estimators for Random Forest
        if n_samples > 10000:
            n_estimators = 200
        elif n_samples > 5000:
            n_estimators = 150
        else:
            n_estimators = 100
            
        if n_estimators != ml_config.get('n_estimators', 100):
            tuned['n_estimators'] = n_estimators
            self.adjustments['ml.n_estimators'] = {
                'original': ml_config.get('n_estimators', 100),
                'tuned': n_estimators,
                'reason': f'Optimized for {n_samples} samples'
            }
        
        # max_features for Random Forest
        max_features = min(
            int(np.sqrt(n_features)),
            500
        )
        
        if max_features != ml_config.get('max_features', 'sqrt'):
            if max_features < 100:
                tuned['max_features'] = 'sqrt'
            else:
                tuned['max_features'] = max_features
                self.adjustments['ml.max_features'] = {
                    'original': ml_config.get('max_features', 'sqrt'),
                    'tuned': max_features,
                    'reason': 'Capped at 500 for performance'
                }
        
        return tuned
    
    def _tune_max_plots(self, n_features: int) -> int:
        """
        Tune maximum plots to generate.
        
        Prevent plotting bottlenecks for large feature sets
        """
        if n_features > 100000:
            return 500
        elif n_features > 50000:
            return 750
        else:
            return 1000
    
    def _tune_alpha_threshold(self, n_features: int) -> float:
        """
        Tune significance threshold for multiple testing.
        
        More features → more conservative to control FDR
        """
        if n_features > 100000:
            return 0.01
        elif n_features > 50000:
            return 0.02
        else:
            return 0.05
    
    def save_report(self, output_path: Path):
        """Save auto-tuning report to file."""
        if not self.adjustments:
            return
            
        lines = []
        lines.append("=" * 80)
        lines.append("AUTO-TUNING REPORT")
        lines.append("=" * 80)
        lines.append(f"\nApplied {len(self.adjustments)} adjustments:\n")
        
        for param, details in self.adjustments.items():
            lines.append(f"{param}:")
            lines.append(f"  Original: {details['original']}")
            lines.append(f"  Tuned:    {details['tuned']}")
            lines.append(f"  Reason:   {details['reason']}")
            lines.append("")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        output_path.write_text(report)
        logger.info(f"Auto-tuning report saved to: {output_path}")


# Global auto-tuner instance
_tuner = AutoTuner()


def get_auto_tuner() -> AutoTuner:
    """Get the global auto-tuner instance."""
    return _tuner


def auto_tune_config(
    n_samples: int,
    n_features: int,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to auto-tune configuration.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    config : dict
        Original configuration
    output_dir : Path, optional
        Directory to save tuning report
        
    Returns
    -------
    dict
        Tuned configuration
    """
    tuner = get_auto_tuner()
    tuned_config = tuner.tune_parameters(n_samples, n_features, config)
    
    if output_dir and tuner.adjustments:
        report_path = output_dir / "auto_tuning_report.txt"
        tuner.save_report(report_path)
    
    return tuned_config
