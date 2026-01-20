# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from skbio.stats.ordination import OrdinationResults
from statsmodels.stats.multitest import multipletests
from biom.table import Table

# Visualization imports (moved to separate section)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import umap

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.statistical_analyses import (
    run_statistical_tests_for_group, TopFeaturesAnalyzer
)
from workflow_16s.amplicon_data.top_features import top_features_plots
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.amplicon_data.downstream.downstream import Downstream

# ================================= CONFIGURATION & CONSTANTS ========================= #

logger = logging.getLogger("workflow_16s")
umap_lock = threading.Lock()  # Global lock for UMAP operations

# Default analysis parameters
DEFAULT_TOP_N_FEATURES = 20
DEFAULT_NETWORK_CORRELATION_THRESHOLD = 0.3
DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.5

# ================================= RESULTS ANALYZER CLASS ========================= #

class DownstreamResultsAnalyzer:
    """Comprehensive analysis framework for integrating downstream pipeline results."""
    
    def __init__(self, downstream_results: Downstream, config: Dict, verbose: bool = True):
        """Initialize analyzer with downstream pipeline results."""
        self.downstream = downstream_results
        self.config = config
        self.verbose = verbose
        
        # Extract data components
        self.metadata = downstream_results.results.metadata
        self.tables = downstream_results.results.tables
        self.stats = downstream_results.results.stats or {}
        self.alpha_diversity = downstream_results.results.alpha_diversity or {}
        self.ordination = downstream_results.results.ordination or {}
        self.models = downstream_results.results.models or {}
        self.top_features = downstream_results.results.top_features or {}
        
        # Initialize analysis containers
        self.integrated_results: Dict[str, Any] = {}
        self.consensus_features: Optional[pd.DataFrame] = None
        self.environmental_thresholds: Dict[str, Any] = {}
        self.networks: Dict[str, Any] = {}
        self.functional_analysis: Dict[str, Any] = {}
        
        if self.verbose:
            logger.info("DownstreamResultsAnalyzer initialized successfully")

    # ========================== FEATURE IMPORTANCE SYNTHESIS ======================== #
    
    def synthesize_feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """Combine feature importance from multiple analysis approaches."""
        if self.verbose:
            logger.info("Synthesizing feature importance across modules...")
        
        importance_scores = {}
        
        # Extract scores from different analysis types
        extractors = [
            ('statistical', self._extract_statistical_importance),
            ('ml_importance', self._extract_ml_importance),
            ('alpha_association', self._extract_alpha_associations),
            ('beta_loading', self._extract_beta_loadings)
        ]
        
        for score_type, extractor in extractors:
            try:
                scores = extractor()
                for feature, score in scores.items():
                    importance_scores.setdefault(feature, {})[score_type] = score
            except Exception as e:
                logger.warning(f"Error extracting {score_type} scores: {e}")
        
        if not importance_scores:
            logger.warning("No feature importance scores found")
            return pd.DataFrame()
        logger.info(importance_scores)
        # Create consensus DataFrame
        consensus_df = pd.DataFrame.from_dict(importance_scores, orient='index').fillna(0)
        
        # Calculate weighted consensus score
        weights = {
            'statistical': 0.3,
            'ml_importance': 0.3,
            'alpha_association': 0.2,
            'beta_loading': 0.2
        }
        
        consensus_df['consensus_score'] = 0
        for col, weight in weights.items():
            if col in consensus_df.columns:
                consensus_df['consensus_score'] += consensus_df[col] * weight
        
        # Sort and return top features
        consensus_df = consensus_df.sort_values('consensus_score', ascending=False)
        self.consensus_features = consensus_df.head(top_n)
        
        if self.verbose:
            logger.info(f"Generated consensus ranking for {len(consensus_df)} features")
        
        return self.consensus_features
    
    def _extract_statistical_importance(self) -> Dict[str, float]:
        """Extract feature importance scores from statistical tests."""
        importance_dict = {}
        
        if not self.top_features or 'stats' not in self.top_features:
            return importance_dict
            
        for group_data, df in self.top_features['stats'].items():
            if not isinstance(group_data, dict):
                continue
            for condition, df in group_data.items():
                if isinstance(df, pd.DataFrame) and 'Feature' in df.columns and 'P-value' in df.columns:
                    for _, row in df.iterrows():
                        # Use negative log p-value as importance score
                        importance_dict[row['Feature']] = -np.log10(max(row['P-value'], 1e-10))
        logger.info("Extracted feature importance scores from statistical tests.")
        return importance_dict
    
    def _extract_ml_importance(self) -> Dict[str, float]:
        """Extract feature importance from ML models."""
        importance_dict = {}
        
        if not self.top_features or 'models' not in self.top_features:
            return importance_dict
            
        for group_col, df in self.top_features['models'].items():
            if isinstance(df, pd.DataFrame) and 'Feature' in df.columns and 'Importance' in df.columns:
                for _, row in df.iterrows():
                    importance_str = row['Importance']
                    try:
                        # Convert importance to float
                        importance = float(importance_str) if importance_str != "N/A" else 0
                        if row['Feature'] in importance_dict:
                            importance_dict[row['Feature']] += importance
                        else:
                            importance_dict[row['Feature']] = importance
                    except (ValueError, TypeError):
                        continue
        
        # Normalize by number of models if needed
        if importance_dict:
            max_importance = max(importance_dict.values())
            if max_importance > 0:
                importance_dict = {k: v/max_importance for k, v in importance_dict.items()}
        logger.info("Extracted feature importance from ML models.")
        return importance_dict
    
    def _extract_alpha_associations(self) -> Dict[str, float]:
        """Extract features associated with alpha diversity metrics."""
        associations = {}
        
        if not self.alpha_diversity:
            return associations
            
        # Look for correlation results with features
        for metric, results in self.alpha_diversity.items():
            if isinstance(results, dict) and 'correlations' in results:
                for feature, corr_data in results['correlations'].items():
                    if isinstance(corr_data, dict) and 'correlation' in corr_data:
                        associations[feature] = abs(corr_data['correlation'])
        
        return associations
    
    def _extract_beta_loadings(self) -> Dict[str, float]:
        """Extract feature loadings from ordination analysis."""
        loadings = {}
        
        if not self.ordination:
            return loadings
            
        for group_col, group_results in self.ordination.items():
            if not isinstance(group_results, dict):
                continue
                
            for level, level_results in group_results.items():
                if not isinstance(level_results, dict):
                    continue
                    
                for method, results in level_results.items():
                    if method in ['pca', 'umap', 'tsne'] and isinstance(results, dict):
                        if 'loadings' in results:
                            method_loadings = results['loadings']
                            if isinstance(method_loadings, dict):
                                for feature, loading_value in method_loadings.items():
                                    if isinstance(loading_value, (list, np.ndarray)):
                                        loadings[feature] = np.linalg.norm(loading_value)
                                    elif isinstance(loading_value, (int, float)):
                                        loadings[feature] = abs(loading_value)
                    elif method == 'pcoa' and isinstance(results, OrdinationResults):
                        if hasattr(results, 'features') and results.features is not None:
                            feature_loadings = results.features
                            for i, feature in enumerate(feature_loadings.index):
                                # Use L2 norm across first two axes
                                loading_norm = np.linalg.norm(feature_loadings.iloc[i, :2])
                                loadings[feature] = loading_norm
        logger.info("Extracted feature loadings from ordination analysis.")
        return loadings

    # ========================== NETWORK ANALYSIS ========================== #
    
    def build_integrated_networks(self, method: str = 'spearman', 
                                 threshold: float = DEFAULT_NETWORK_CORRELATION_THRESHOLD) -> Dict[str, Any]:
        """Create networks connecting statistically significant features."""
        if self.verbose:
            logger.info(f"Building integrated networks using {method}...")
        
        network_results = {}
        
        if self.consensus_features is None:
            logger.warning("No consensus features available for network analysis")
            return network_results
            
        top_features = self.consensus_features.index.tolist()[:30]
        
        # Extract abundance data for top features
        abundance_data = self._get_feature_abundance_matrix(top_features)
        
        if abundance_data is None:
            logger.warning("Could not extract abundance data for network analysis")
            return network_results
        
        # Calculate correlation matrix
        try:
            if method == 'spearman':
                corr_matrix = abundance_data.corr(method='spearman')
            elif method == 'pearson':
                corr_matrix = abundance_data.corr(method='pearson')
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Create network from correlation matrix
            network = self._create_network_from_correlations(corr_matrix, threshold)
            
            # Calculate network properties
            network_properties = self._calculate_network_properties(network)
            
            network_results[method] = {
                'network': network,
                'correlation_matrix': corr_matrix,
                'properties': network_properties,
                'adjacency_matrix': nx.adjacency_matrix(network).todense()
            }
            
            if self.verbose:
                logger.info(f"Network created: {network_properties.get('n_nodes', 0)} nodes, "
                     f"{network_properties.get('n_edges', 0)} edges")
                     
        except Exception as e:
            logger.error(f"Error creating {method} network: {e}")
        
        self.networks = network_results
        self.integrated_results['networks'] = network_results
        return network_results
    
    def _get_feature_abundance_matrix(self, features: List[str]) -> Optional[pd.DataFrame]:
        """Get abundance matrix for specified features."""
        community_data = self._extract_community_matrix()
        
        if community_data is not None:
            available_features = [f for f in features if f in community_data.columns]
            if available_features:
                return community_data[available_features].copy()
        
        return None
    
    def _extract_community_matrix(self) -> Optional[pd.DataFrame]:
        """Extract community abundance matrix from tables."""
        for mode in ['genus', 'asv']:
            if mode not in self.tables:
                continue
                
            tables = self.tables[mode]
            if not isinstance(tables, dict):
                continue
                
            for subset_name, table_data in tables.items():
                try:
                    if hasattr(table_data, 'to_dataframe'):
                        # BIOM table - transpose so samples are rows
                        community_data = table_data.to_dataframe().T
                        return community_data
                    elif isinstance(table_data, pd.DataFrame):
                        # Regular DataFrame - ensure samples as rows
                        if table_data.shape[0] < table_data.shape[1]:
                            return table_data.T
                        else:
                            return table_data
                except Exception as e:
                    logger.warning(f"Could not extract data from {mode}/{subset_name}: {e}")
                    continue
        
        return None
    
    def _create_network_from_correlations(self, corr_matrix: pd.DataFrame, 
                                        threshold: float) -> nx.Graph:
        """Create network graph from correlation matrix."""
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(corr_matrix.index)
        
        # Add edges for correlations above threshold
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= threshold and not np.isnan(corr_val):
                    G.add_edge(
                        corr_matrix.index[i], 
                        corr_matrix.index[j], 
                        weight=corr_val
                    )
        
        return G
    
    def _calculate_network_properties(self, network: nx.Graph) -> Dict[str, Any]:
        """Calculate network topology properties."""
        properties = {
            'n_nodes': network.number_of_nodes(),
            'n_edges': network.number_of_edges(),
            'density': 0.0,
            'clustering_coefficient': 0.0,
            'n_components': 0
        }
        
        if properties['n_nodes'] == 0:
            return properties
            
        properties['density'] = nx.density(network)
        properties['n_components'] = nx.number_connected_components(network)
        properties['clustering_coefficient'] = nx.average_clustering(network)
        
        # Calculate centrality measures only if we have edges
        if properties['n_edges'] > 0:
            try:
                properties['degree_centrality'] = nx.degree_centrality(network)
                properties['betweenness_centrality'] = nx.betweenness_centrality(network)
                properties['closeness_centrality'] = nx.closeness_centrality(network)
                # Eigenvector centrality can fail for disconnected graphs
                if nx.is_connected(network):
                    properties['eigenvector_centrality'] = nx.eigenvector_centrality(network)
            except Exception as e:
                logger.warning(f"Error calculating network centralities: {e}")
        
        return properties

    # ========================== ENVIRONMENTAL ANALYSIS ========================== #
    
    def analyze_environmental_gradients(self, continuous_vars: List[str] = None) -> Dict[str, Any]:
        """Analyze environmental gradients using canonical correspondence analysis."""
        if self.verbose:
            logger.info("Analyzing environmental gradients...")
        
        if continuous_vars is None:
            continuous_vars = ['ph', 'facility_distance_km']
        
        gradient_results = {}
        
        # Extract data
        env_data = self._extract_environmental_data(continuous_vars)
        community_data = self._extract_community_matrix()
        
        if env_data is None or community_data is None:
            logger.warning("Could not extract environmental or community data")
            return gradient_results
        
        # Align samples
        common_samples = list(set(env_data.index) & set(community_data.index))
        if len(common_samples) < 10:
            logger.warning(f"Insufficient overlapping samples: {len(common_samples)}")
            return gradient_results
        
        env_aligned = env_data.loc[common_samples]
        comm_aligned = community_data.loc[common_samples]
        
        # Remove any remaining NaN values
        env_aligned = env_aligned.dropna()
        comm_aligned = comm_aligned.loc[env_aligned.index]
        
        if len(env_aligned) < 10:
            logger.warning("Insufficient samples after removing NaN values")
            return gradient_results
        
        try:
            # Perform canonical correspondence analysis approximation
            from sklearn.cross_decomposition import CCA
            
            n_components = min(len(continuous_vars), 3, len(env_aligned.columns))
            cca = CCA(n_components=n_components)
            env_scores, comm_scores = cca.fit_transform(env_aligned, comm_aligned)
            
            gradient_results['cca_results'] = {
                'environmental_scores': pd.DataFrame(
                    env_scores, 
                    index=env_aligned.index, 
                    columns=[f'CCA{i+1}_env' for i in range(env_scores.shape[1])]
                ),
                'community_scores': pd.DataFrame(
                    comm_scores,
                    index=env_aligned.index,
                    columns=[f'CCA{i+1}_comm' for i in range(comm_scores.shape[1])]
                ),
                'explained_variance': cca.score(env_aligned, comm_aligned)
            }
            
            # Calculate feature loadings
            feature_loadings = self._calculate_cca_loadings(
                comm_aligned, comm_scores, continuous_vars
            )
            gradient_results['feature_loadings'] = feature_loadings
            
            if self.verbose:
                logger.info(f"CCA analysis completed with {n_components} components")
                
        except Exception as e:
            logger.error(f"Error in CCA analysis: {e}")
        
        self.integrated_results['environmental_gradients'] = gradient_results
        return gradient_results
    
    def _extract_environmental_data(self, variables: List[str] = None) -> Optional[pd.DataFrame]:
        """Extract environmental variables from metadata."""
        if variables is None:
            variables = ['ph', 'facility_distance_km']
        
        # Try to extract from different metadata levels
        for mode in ['genus', 'asv']:
            if mode not in self.metadata:
                continue
                
            metadata = self.metadata[mode]
            if isinstance(metadata, dict):
                for subset_name, subset_data in metadata.items():
                    if isinstance(subset_data, pd.DataFrame):
                        available_vars = [v for v in variables if v in subset_data.columns]
                        if available_vars:
                            return subset_data[available_vars].copy()
            elif isinstance(metadata, pd.DataFrame):
                available_vars = [v for v in variables if v in metadata.columns]
                if available_vars:
                    return metadata[available_vars].copy()
        
        return None
    
    def _calculate_cca_loadings(self, community_data: pd.DataFrame, 
                              comm_scores: np.ndarray, env_vars: List[str]) -> Dict[str, Dict]:
        """Calculate feature loadings on CCA axes."""
        loadings = {}
        
        for i in range(comm_scores.shape[1]):
            axis_name = f'CCA{i+1}'
            axis_loadings = {}
            
            for feature in community_data.columns:
                try:
                    corr, p_val = spearmanr(community_data[feature], comm_scores[:, i])
                    axis_loadings[feature] = {
                        'loading': corr if not np.isnan(corr) else 0.0,
                        'p_value': p_val if not np.isnan(p_val) else 1.0
                    }
                except Exception:
                    axis_loadings[feature] = {'loading': 0.0, 'p_value': 1.0}
            
            loadings[axis_name] = axis_loadings
        
        return loadings

    # ========================== COMPREHENSIVE ANALYSIS RUNNER ======================== #
    
    def run_comprehensive_analysis(self, output_dir: str = 'integrated_analysis_output') -> Dict[str, Any]:
        """Run comprehensive integrated analysis pipeline."""
        if self.verbose:
            logger.info("=" * 60)
            logger.info("RUNNING COMPREHENSIVE DOWNSTREAM ANALYSIS")
            logger.info("=" * 60)
        
        results = {}
        
        try:
            # 1. Feature importance synthesis
            consensus_features = self.synthesize_feature_importance()
            results['consensus_features'] = consensus_features
            
            # 2. Network analysis
            if consensus_features is not None and not consensus_features.empty:
                network_results = self.build_integrated_networks()
                results['networks'] = network_results
            
            # 3. Environmental gradient analysis
            gradient_results = self.analyze_environmental_gradients()
            results['environmental_gradients'] = gradient_results
            
            # 4. Create output directory and save results
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save consensus features
            if consensus_features is not None and not consensus_features.empty:
                consensus_features.to_csv(f"{output_dir}/consensus_features.csv")
                if self.verbose:
                    logger.info(f"Saved consensus features to {output_dir}/consensus_features.csv")
            
            # Generate summary report
            summary_report = self._generate_analysis_summary(results)
            with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
                f.write(summary_report)
            
            if self.verbose:
                logger.info(f"Analysis completed! Results saved to: {output_dir}")
                logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            if self.verbose:
                raise
            return {}
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis summary report."""
        report_lines = [
            "INTEGRATED DOWNSTREAM ANALYSIS SUMMARY",
            "=" * 50,
            ""
        ]
        
        # Consensus features summary
        if 'consensus_features' in results and results['consensus_features'] is not None:
            cf = results['consensus_features']
            report_lines.extend([
                "CONSENSUS FEATURES ANALYSIS",
                "-" * 30,
                f"Total features analyzed: {len(cf)}",
                f"Top 5 features by consensus score:"
            ])
            
            for i, (feature, row) in enumerate(cf.head().iterrows(), 1):
                score = row.get('consensus_score', 0)
                report_lines.append(f"  {i}. {feature} (score: {score:.3f})")
            
            report_lines.append("")
        
        # Network analysis summary
        if 'networks' in results and results['networks']:
            report_lines.extend([
                "NETWORK ANALYSIS",
                "-" * 20
            ])
            
            for method, network_data in results['networks'].items():
                props = network_data.get('properties', {})
                n_nodes = props.get('n_nodes', 0)
                n_edges = props.get('n_edges', 0)
                density = props.get('density', 0)
                
                report_lines.append(
                    f"{method.capitalize()} network: {n_nodes} nodes, "
                    f"{n_edges} edges (density: {density:.3f})"
                )
            
            report_lines.append("")
        
        # Environmental gradients summary
        if 'environmental_gradients' in results and results['environmental_gradients']:
            eg = results['environmental_gradients']
            report_lines.extend([
                "ENVIRONMENTAL GRADIENTS",
                "-" * 25
            ])
            
            if 'cca_results' in eg:
                explained_var = eg['cca_results'].get('explained_variance', 0)
                report_lines.append(f"CCA explained variance: {explained_var:.3f}")
                
                env_scores = eg['cca_results'].get('environmental_scores')
                if env_scores is not None:
                    n_components = env_scores.shape[1]
                    n_samples = len(env_scores)
                    report_lines.append(f"Components: {n_components}, Samples: {n_samples}")
            
            report_lines.append("")
        
        # Analysis modules status
        modules_status = []
        if self.stats:
            modules_status.append("Statistical Analysis")
        if self.alpha_diversity:
            modules_status.append("Alpha Diversity")
        if self.ordination:
            modules_status.append("Beta Diversity")
        if self.models:
            modules_status.append("Machine Learning")
        
        if modules_status:
            report_lines.extend([
                "COMPLETED ANALYSIS MODULES",
                "-" * 30,
                ", ".join(modules_status),
                ""
            ])
        
        # Data summary
        if self.metadata and self.tables:
            report_lines.extend([
                "DATA SUMMARY",
                "-" * 15,
                "Available data modes:"
            ])
            
            for mode in ['genus', 'asv']:
                if mode in self.tables:
                    report_lines.append(f"  - {mode}: available")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 50,
            "Analysis completed successfully"
        ])
        
        return "\n".join(report_lines)


# ================================= MAIN EXECUTION EXAMPLE ========================== #

def run_integrated_downstream_analysis(config: Dict, project_dir: Any, **kwargs) -> Tuple[Downstream, DownstreamResultsAnalyzer]:
    """
    Run the complete integrated downstream analysis pipeline.
    
    Args:
        config: Analysis configuration dictionary
        project_dir: Project directory object
        **kwargs: Additional parameters for Downstream class
    
    Returns:
        Tuple of (Downstream results object, Results analyzer object)
    """
    
    # Run main downstream analysis
    logger.info("Starting integrated downstream analysis...")
    
    downstream = Downstream(
        config=config,
        project_dir=project_dir,
        **kwargs
    )
    
    # Create results analyzer
    analyzer = DownstreamResultsAnalyzer(
        downstream_results=downstream,
        config=config,
        verbose=kwargs.get('verbose', False)
    )
    
    # Run comprehensive analysis
    integrated_results = analyzer.run_comprehensive_analysis()
    
    logger.info("Integrated downstream analysis completed successfully")
    
    return downstream, analyzer
