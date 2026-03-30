"""Statistical Analysis Module

Implements differential abundance testing and variance partitioning.
Wraps R packages (ANCOMBC, vegan) via rpy2 for compositional analysis.

Main components:
- ANCOM-BC: Differential abundance with compositional bias correction
- ElasticNet CV: Leave-one-study-out feature selection
- Variance Partitioning: Decompose multivariate variance by factor groups
"""

from .differential_abundance import (
    ANCAMBCWrapper,
    ANCAMBCResult,
    ElasticNetCV,
    CandidateFeaturesSelector,
)
from .variance_partitioning import (
    VariancePartitioningAnalyzer,
    VarpartResult,
    perform_variance_partitioning,
)

__all__ = [
    'ANCAMBCWrapper',
    'ANCAMBCResult',
    'ElasticNetCV',
    'CandidateFeaturesSelector',
    'VariancePartitioningAnalyzer',
    'VarpartResult',
    'perform_variance_partitioning',
]
