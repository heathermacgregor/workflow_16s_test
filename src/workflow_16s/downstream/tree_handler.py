"""
Phylogenetic Tree Handler Module

This module provides strategies for handling missing or incomplete phylogenetic trees
when working with concatenated datasets from multiple sources.

Strategies:
1. Graceful Degradation: Skip phylogenetic metrics, use only non-phylogenetic diversity
2. Tree Merging: Combine per-dataset trees into a supertree
3. De Novo Tree Building: Build a new tree from concatenated features
4. Partial Analysis: Analyze only datasets with trees
5. Subset Tree Extraction: Extract tree for features present in analysis
"""

import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import pandas as pd
import anndata as ad
from skbio import TreeNode

logger = logging.getLogger(__name__)


class TreeHandlingStrategy:
    """Base class for tree handling strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def handle(self, adata: ad.AnnData, config, output_dir: Path) -> Optional[Path]:
        """Execute the strategy. Returns path to tree file if successful."""
        raise NotImplementedError


class GracefulDegradationStrategy(TreeHandlingStrategy):
    """
    Strategy 1: Graceful Degradation
    
    Skip phylogenetic diversity metrics and continue with non-phylogenetic analysis.
    This is the safest option when trees are missing or unreliable.
    
    Pros:
    - No additional computation needed
    - Analysis can proceed immediately
    - No risk of using incorrect tree topology
    
    Cons:
    - Loses information from phylogenetic relationships
    - Cannot calculate Faith's PD, UniFrac distances
    """
    
    def __init__(self):
        super().__init__(
            name="graceful_degradation",
            description="Skip phylogenetic metrics, use only taxonomy-based diversity"
        )
    
    def handle(self, adata: ad.AnnData, config, output_dir: Path) -> Optional[Path]:
        logger.warning("Using Graceful Degradation strategy: Phylogenetic diversity metrics will be skipped")
        logger.info("Available metrics: Shannon, Simpson, Observed Features, Evenness")
        logger.info("Beta diversity will use Bray-Curtis, Jaccard (no UniFrac)")
        return None


class TreeMergingStrategy(TreeHandlingStrategy):
    """
    Strategy 2: Tree Merging (Supertree Construction)
    
    Combine multiple per-dataset trees into a single supertree. Uses a simple
    star topology to connect disjoint tree components.
    
    Pros:
    - Preserves phylogenetic relationships from each dataset
    - Allows partial phylogenetic analysis
    
    Cons:
    - May create artificial topology for features not in original trees
    - Requires careful validation
    - Star topology connections have no meaningful branch lengths
    
    Implementation:
    - Extract trees from adata.uns['phylogenetic_tree'] for each dataset
    - Join trees using star topology for disjoint components
    - Prune to only features present in concatenated adata
    """
    
    def __init__(self):
        super().__init__(
            name="tree_merging",
            description="Merge per-dataset trees into a supertree with star topology"
        )
    
    def handle(self, adata: ad.AnnData, config, output_dir: Path) -> Optional[Path]:
        logger.info(f"Using Tree Merging strategy: Combining trees from datasets...")
        
        # Check if we have tree information in adata.uns
        if 'phylogenetic_tree' not in adata.uns or not adata.uns['phylogenetic_tree']:
            logger.error("No phylogenetic tree found in adata.uns")
            return None
        
        # If tree is a simple string (single tree), just export it
        if isinstance(adata.uns['phylogenetic_tree'], str):
            tree_path = output_dir / "all_features.tree"
            with open(tree_path, 'w') as f:
                f.write(adata.uns['phylogenetic_tree'])
            logger.info(f"Exported single tree to {tree_path}")
            return tree_path
        
        # If tree is a dict with per-dataset trees, merge them
        if isinstance(adata.uns['phylogenetic_tree'], dict):
            trees = []
            for dataset_id, tree_str in adata.uns['phylogenetic_tree'].items():
                if tree_str:
                    try:
                        tree = TreeNode.read(io.StringIO(tree_str), format='newick')
                        trees.append((dataset_id, tree))
                        logger.info(f"Loaded tree from dataset: {dataset_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load tree from {dataset_id}: {e}")
            
            if not trees:
                logger.error("No valid trees found in adata.uns['phylogenetic_tree'] dict")
                return None
            
            # Merge trees using star topology
            merged_tree = self._merge_trees_star_topology(trees, adata.var_names)
            
            # Export merged tree
            tree_path = output_dir / "all_features.tree"
            with open(tree_path, 'w') as f:
                merged_tree.write(f, format='newick')
            logger.info(f"Exported merged tree to {tree_path}")
            return tree_path
        
        logger.error(f"Unknown phylogenetic_tree format: {type(adata.uns['phylogenetic_tree'])}")
        return None
    
    def _merge_trees_star_topology(self, trees: List[Tuple[str, TreeNode]], 
                                   all_features: pd.Index) -> TreeNode:
        """
        Merge multiple trees using star topology for disjoint components.
        
        Creates a root node with all input trees as children. Adds missing
        features as single-node children of the root.
        """
        # Create root node
        root = TreeNode(name="root")
        
        # Track which features are covered
        covered_features = set()
        
        # Add each tree as a child of the root
        for dataset_id, tree in trees:
            # Get tips from this tree
            tips = {tip.name for tip in tree.tips()}
            covered_features.update(tips)
            
            # Add tree as child
            root.append(tree)
            logger.info(f"Added tree from {dataset_id} with {len(tips)} features")
        
        # Add missing features as single nodes
        missing_features = set(all_features) - covered_features
        if missing_features:
            logger.warning(f"Adding {len(missing_features)} missing features as single nodes")
            for feature in missing_features:
                root.append(TreeNode(name=feature, length=1.0))
        
        return root


class DeNovoTreeBuildingStrategy(TreeHandlingStrategy):
    """
    Strategy 3: De Novo Tree Building
    
    Build a completely new phylogenetic tree from the concatenated feature set.
    This is the most robust approach but computationally expensive.
    
    Pros:
    - Tree topology is consistent with concatenated dataset
    - No assumptions about tree compatibility
    - Can handle any feature set
    
    Cons:
    - Computationally expensive (alignment + tree building)
    - May take significant time for large feature sets
    - Requires sequence data in adata.var
    
    Implementation:
    - Export feature sequences to FASTA
    - Run MAFFT alignment
    - Build tree with FastTree
    - Root tree with midpoint rooting
    """
    
    def __init__(self):
        super().__init__(
            name="denovo_tree_building",
            description="Build new phylogenetic tree from concatenated features"
        )
    
    def handle(self, adata: ad.AnnData, config, output_dir: Path) -> Optional[Path]:
        logger.info(f"Using De Novo Tree Building strategy: Building tree from scratch...")
        
        # Import here to avoid circular dependency
        from workflow_16s.downstream.preprocessing import rebuild_tree, export_fasta
        
        # Export FASTA
        fasta_path = output_dir / "all_features.fasta"
        if not fasta_path.exists():
            logger.info("Exporting feature sequences to FASTA...")
            export_fasta(adata, config, output_dir)
        
        if not fasta_path.exists():
            logger.error("Failed to export FASTA file - cannot build tree")
            return None
        
        # Build tree
        logger.info("Building phylogenetic tree (this may take several minutes)...")
        tree_path = rebuild_tree(adata, config, output_dir)
        
        if tree_path and tree_path.exists():
            logger.info(f"Successfully built de novo tree: {tree_path}")
            return tree_path
        else:
            logger.error("De novo tree building failed")
            return None


class PartialAnalysisStrategy(TreeHandlingStrategy):
    """
    Strategy 4: Partial Analysis
    
    Analyze only the subset of samples/features that have phylogenetic tree coverage.
    This preserves tree accuracy but may exclude data.
    
    Pros:
    - Uses only reliable phylogenetic information
    - No artificial tree topology
    
    Cons:
    - Reduces sample/feature count
    - May introduce bias if missing trees are not random
    - Splits analysis into phylo vs non-phylo
    
    Implementation:
    - Identify features with tree coverage
    - Create subset AnnData for phylogenetic analysis
    - Run full analysis on complete data (non-phylo metrics)
    - Run phylo analysis on subset
    """
    
    def __init__(self):
        super().__init__(
            name="partial_analysis",
            description="Analyze only samples/features with tree coverage"
        )
    
    def handle(self, adata: ad.AnnData, config, output_dir: Path) -> Optional[Path]:
        logger.info(f"Using Partial Analysis strategy: Analyzing tree-covered subset...")
        
        # Extract tree if available
        if 'phylogenetic_tree' in adata.uns and adata.uns['phylogenetic_tree']:
            tree_str = adata.uns['phylogenetic_tree']
            
            # Parse tree to get covered features
            try:
                if isinstance(tree_str, str):
                    tree = TreeNode.read(io.StringIO(tree_str), format='newick')
                    covered_features = {tip.name for tip in tree.tips()}
                    
                    # Subset to covered features
                    feature_mask = adata.var_names.isin(covered_features)
                    n_covered = feature_mask.sum()
                    n_total = len(adata.var_names)
                    
                    logger.info(f"Tree covers {n_covered}/{n_total} features ({100*n_covered/n_total:.1f}%)")
                    
                    if n_covered == 0:
                        logger.error("No features overlap with tree")
                        return None
                    
                    # Prune tree to only covered features that are in adata
                    tree = self._prune_tree_to_features(tree, adata.var_names[feature_mask])
                    
                    # Export pruned tree
                    tree_path = output_dir / "all_features.tree"
                    with open(tree_path, 'w') as f:
                        tree.write(f, format='newick')
                    
                    logger.info(f"Exported tree for {n_covered} features to {tree_path}")
                    logger.warning(f"Phylogenetic diversity will only use {n_covered} features")
                    
                    return tree_path
                    
            except Exception as e:
                logger.error(f"Failed to process tree: {e}")
                return None
        
        logger.error("No phylogenetic tree available for partial analysis")
        return None
    
    def _prune_tree_to_features(self, tree: TreeNode, features: pd.Index) -> TreeNode:
        """Prune tree to only include specified features."""
        features_set = set(features)
        tree = tree.shear(features_set)
        return tree


class SubsetTreeExtractionStrategy(TreeHandlingStrategy):
    """
    Strategy 5: Subset Tree Extraction
    
    Extract a subtree containing only the features present in the analysis.
    Similar to partial analysis but focuses on tree pruning.
    
    Pros:
    - Maintains accurate phylogenetic relationships
    - Can handle features from multiple source trees
    - Removes unnecessary branches
    
    Cons:
    - May not work if features span multiple disjoint trees
    - Requires features to be present in source tree
    
    Implementation:
    - Load tree from adata.uns or file
    - Prune to features in adata.var_names
    - Export pruned tree
    """
    
    def __init__(self):
        super().__init__(
            name="subset_tree_extraction",
            description="Extract subtree for features in analysis"
        )
    
    def handle(self, adata: ad.AnnData, config, output_dir: Path) -> Optional[Path]:
        logger.info(f"Using Subset Tree Extraction strategy: Pruning tree to analysis features...")
        
        if 'phylogenetic_tree' not in adata.uns or not adata.uns['phylogenetic_tree']:
            logger.error("No phylogenetic tree found in adata.uns")
            return None
        
        try:
            tree_str = adata.uns['phylogenetic_tree']
            tree = TreeNode.read(io.StringIO(tree_str), format='newick')
            
            # Get tree tips
            tree_tips = {tip.name for tip in tree.tips()}
            
            # Find overlap with current features
            current_features = set(adata.var_names)
            overlap = tree_tips & current_features
            
            if not overlap:
                logger.error("No features overlap between tree and adata")
                return None
            
            coverage = len(overlap) / len(current_features)
            logger.info(f"Tree covers {len(overlap)}/{len(current_features)} features ({100*coverage:.1f}%)")
            
            # Prune tree to current features
            tree = tree.shear(overlap)
            
            # Export pruned tree
            tree_path = output_dir / "all_features.tree"
            with open(tree_path, 'w') as f:
                tree.write(f, format='newick')
            
            logger.info(f"Exported pruned tree to {tree_path}")
            
            if coverage < 0.5:
                logger.warning(f"Only {100*coverage:.1f}% of features have tree coverage")
                logger.warning("Consider using de novo tree building strategy instead")
            
            return tree_path
            
        except Exception as e:
            logger.error(f"Failed to extract subset tree: {e}")
            return None


def get_tree_handling_strategy(strategy_name: str) -> TreeHandlingStrategy:
    """
    Get a tree handling strategy by name.
    
    Parameters
    ----------
    strategy_name : str
        Name of the strategy:
        - 'graceful_degradation': Skip phylogenetic metrics
        - 'tree_merging': Merge per-dataset trees
        - 'denovo_tree_building': Build new tree from scratch
        - 'partial_analysis': Analyze only tree-covered subset
        - 'subset_tree_extraction': Extract subtree for current features
    
    Returns
    -------
    TreeHandlingStrategy
        The requested strategy instance
    """
    strategies = {
        'graceful_degradation': GracefulDegradationStrategy,
        'tree_merging': TreeMergingStrategy,
        'denovo_tree_building': DeNovoTreeBuildingStrategy,
        'partial_analysis': PartialAnalysisStrategy,
        'subset_tree_extraction': SubsetTreeExtractionStrategy,
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Unknown strategy '{strategy_name}', using graceful_degradation")
        strategy_name = 'graceful_degradation'
    
    return strategies[strategy_name]()


def handle_missing_tree(
    adata: ad.AnnData,
    config,
    output_dir: Path,
    strategy: str = 'auto'
) -> Optional[Path]:
    """
    Handle missing or incomplete phylogenetic trees using specified strategy.
    
    Parameters
    ----------
    adata : ad.AnnData
        The concatenated AnnData object
    config : AppConfig
        Configuration object
    output_dir : Path
        Output directory for tree files
    strategy : str, optional
        Strategy to use:
        - 'auto': Automatically select best strategy based on available data
        - 'graceful_degradation': Skip phylogenetic metrics
        - 'tree_merging': Merge per-dataset trees
        - 'denovo_tree_building': Build new tree
        - 'partial_analysis': Analyze tree-covered subset only
        - 'subset_tree_extraction': Extract subtree for current features
    
    Returns
    -------
    Optional[Path]
        Path to tree file if successful, None otherwise
    
    Examples
    --------
    >>> tree_path = handle_missing_tree(adata, config, output_dir, strategy='auto')
    >>> if tree_path:
    ...     print(f"Tree available at {tree_path}")
    ... else:
    ...     print("No tree available, skipping phylogenetic analysis")
    """
    logger.info(f"Handling phylogenetic tree with strategy: {strategy}")
    
    # Auto-select strategy
    if strategy == 'auto':
        strategy = _auto_select_strategy(adata, config)
        logger.info(f"Auto-selected strategy: {strategy}")
    
    # Get and execute strategy
    handler = get_tree_handling_strategy(strategy)
    logger.info(f"Strategy: {handler.description}")
    
    tree_path = handler.handle(adata, config, output_dir)
    
    if tree_path:
        logger.info(f"✅ Tree handling successful: {tree_path}")
    else:
        logger.warning("⚠️  Tree handling did not produce a tree file")
    
    return tree_path


def _auto_select_strategy(adata: ad.AnnData, config) -> str:
    """
    Automatically select the best tree handling strategy based on available data.
    
    Decision logic:
    1. If tree exists in adata.uns and covers >80% features: subset_tree_extraction
    2. If tree exists and covers 50-80% features: partial_analysis
    3. If multiple trees in dict format: tree_merging
    4. If sequence data available: denovo_tree_building
    5. Otherwise: graceful_degradation
    """
    # Check if tree exists
    has_tree = 'phylogenetic_tree' in adata.uns and adata.uns['phylogenetic_tree']
    
    if not has_tree:
        # Check if sequences available for de novo building
        if 'sequence' in adata.var.columns and adata.var['sequence'].notna().sum() > 0:
            logger.info("No tree found, but sequences available - will build de novo tree")
            return 'denovo_tree_building'
        else:
            logger.info("No tree or sequences available - will skip phylogenetic metrics")
            return 'graceful_degradation'
    
    # Parse tree to check coverage
    tree_data = adata.uns['phylogenetic_tree']
    
    # Multiple trees (dict format) - merge them
    if isinstance(tree_data, dict):
        logger.info("Multiple per-dataset trees detected - will merge")
        return 'tree_merging'
    
    # Single tree string - check coverage
    if isinstance(tree_data, str):
        try:
            tree = TreeNode.read(io.StringIO(tree_data), format='newick')
            tree_tips = {tip.name for tip in tree.tips()}
            current_features = set(adata.var_names)
            overlap = len(tree_tips & current_features)
            coverage = overlap / len(current_features)
            
            if coverage >= 0.8:
                logger.info(f"Tree covers {100*coverage:.1f}% of features - will extract subset")
                return 'subset_tree_extraction'
            elif coverage >= 0.5:
                logger.info(f"Tree covers {100*coverage:.1f}% of features - will use partial analysis")
                return 'partial_analysis'
            else:
                logger.info(f"Tree covers only {100*coverage:.1f}% of features")
                # Check if we can build de novo
                if 'sequence' in adata.var.columns and adata.var['sequence'].notna().sum() > 0:
                    logger.info("Will build de novo tree for better coverage")
                    return 'denovo_tree_building'
                else:
                    logger.info("Will skip phylogenetic metrics")
                    return 'graceful_degradation'
        except Exception as e:
            logger.error(f"Failed to parse tree: {e}")
            return 'graceful_degradation'
    
    # Unknown format
    logger.warning(f"Unknown tree format: {type(tree_data)}")
    return 'graceful_degradation'
