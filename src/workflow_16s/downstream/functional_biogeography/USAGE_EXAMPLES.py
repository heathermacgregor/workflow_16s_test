"""
Module 1: Functional Biogeography - Usage Examples

This module directly addresses Adam Arkin's guidance:
"What defines adaptive functionality across different environments?
Look at the difference between taxonomic conservation of function 
vs locale conservation of function."

Three core components:

1. FUNCTIONAL TRAIT MAPPING
   - Identifies genes for metal resistance, uranium reduction, energy metabolism
   - Maps these functional traits to each OTU
   - Output: OTU × Trait matrix (confidence scores)

2. PHYLOGENETIC SIGNAL ANALYSIS (Pagel's Lambda)
   - Measures if traits follow the evolutionary tree (λ ≈ 1)
   - Or are randomly distributed (λ ≈ 0)
   - High λ = vertical inheritance (core to lineage)
   - Low λ = horizontal transfer (adaptive response)

3. CONSERVATION ANALYSIS
   - Synthesizes traits × phylogenetic signal × environments
   - Identifies functional guilds (groups of taxa with same functions)
   - Reveals which environments select for which adaptations

EXAMPLE USAGE
=============
"""

import sys
sys.path.insert(0, '/usr2/people/macgregor/amplicon/workflow_16s/src')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from workflow_16s.downstream.functional_biogeography import (
    MetalResistanceGeneDatabase,
    create_trait_matrix,
    analyze_functional_vs_taxonomic_conservation,
    generate_conservation_report
)

def example_basic_workflow(adata, otu_metadata_path=None):
    """
    Basic workflow: Extract traits and analyze conservation.
    
    Parameters
    ----------
    adata : AnnData
        Your AnnData object with OTU data
    otu_metadata_path : str, optional
        Path to otus.97.allinfo file with RAST annotations
    
    Returns
    -------
    Dict with analysis results
    """
    print("\n" + "="*80)
    print("FUNCTIONAL BIOGEOGRAPHY ANALYSIS")
    print("Adams question: 'What defines adaptive functionality?'")
    print("="*80 + "\n")
    
    # Step 1: Create trait matrix
    print("STEP 1: Mapping functional traits to OTUs...")
    print("-" * 80)
    trait_matrix, trait_db = create_trait_matrix(
        adata,
        otu_metadata_path=otu_metadata_path
    )
    print(f"  ✓ Created trait matrix: {trait_matrix.shape[0]} OTUs × {trait_matrix.shape[1]} traits")
    print(f"  Available traits: {', '.join(trait_matrix.columns.tolist()[:5])}...")
    print()
    
    # Step 2: Run conservation analysis
    print("STEP 2: Analyzing functional-taxonomic conservation...")
    print("-" * 80)
    results = analyze_functional_vs_taxonomic_conservation(
        adata,
        otu_metadata_path=otu_metadata_path,
        output_dir='/tmp/conservation_analysis'
    )
    print()
    
    # Step 3: Interpret results
    print("\nSTEP 3: Results")
    print("-" * 80)
    phylo_results = results['phylogenetic_signal']
    print(f"\nPhylogenetic Signal Summary:")
    print(f"  Conserved traits (λ > 0.7):   {(phylo_results['pagels_lambda'] > 0.7).sum()}")
    print(f"  Random traits (λ < 0.3):      {(phylo_results['pagels_lambda'] < 0.3).sum()}")
    print(f"  Mixed patterns (0.3 ≤ λ ≤ 0.7): {((phylo_results['pagels_lambda'] >= 0.3) & (phylo_results['pagels_lambda'] <= 0.7)).sum()}")
    
    # Show examples
    print("\n  Top CONSERVED traits (high λ = follow evolutionary tree):")
    conserved = phylo_results[phylo_results['pagels_lambda'] > 0.7].head()
    for _, row in conserved.iterrows():
        print(f"    • {row['trait']:25s} λ = {row['pagels_lambda']:.3f}")
    
    print("\n  Top RANDOM traits (low λ = suggest horizontal transfer):")
    random = phylo_results[phylo_results['pagels_lambda'] < 0.3].head()
    for _, row in random.iterrows():
        print(f"    • {row['trait']:25s} λ = {row['pagels_lambda']:.3f}")
    
    return results


def example_with_environment(adata, environmental_variable='pH'):
    """
    Advanced: Analyze which traits correlate with specific environments.
    
    This addresses: "Look at locale conservation of function"
    (Same function appearing in similar environments despite taxonomic distance)
    
    Parameters
    ----------
    adata : AnnData
    environmental_variable : str
        Column name in adata.obs (e.g., 'pH', 'metal_concentration')
    """
    print("\n" + "="*80)
    print("ENVIRONMENTAL ADAPTATION ANALYSIS")
    print(f"Linking traits to environment: {environmental_variable}")
    print("="*80 + "\n")
    
    results = analyze_functional_vs_taxonomic_conservation(
        adata,
        environmental_variable=environmental_variable,
        output_dir='/tmp/env_adaptation'
    )
    
    env_assoc = results['environmental_associations']
    
    if env_assoc:
        print(f"Traits correlated with {environmental_variable}:")
        print("-" * 80)
        
        # Sort by correlation strength
        sorted_assoc = sorted(env_assoc.items(), 
                            key=lambda x: abs(x[1]['correlation']), 
                            reverse=True)
        
        for trait, data in sorted_assoc[:10]:
            direction = "↑ enriched" if data['correlation'] > 0 else "↓ depleted"
            print(f"  • {trait:30s} r = {data['correlation']:+.3f} ({direction})")
    else:
        print(f"No environmental variable '{environmental_variable}' found in metadata")
    
    return results


def interpret_lambda_results(phylo_results_df):
    """
    Interpret Pagel's lambda results for scientific communication.
    
    This is what you tell Adam Arkin about your results.
    """
    print("\n" + "="*80)
    print("SCIENTIFIC INTERPRETATION")
    print("="*80 + "\n")
    
    conserved = phylo_results_df[phylo_results_df['pagels_lambda'] > 0.7]
    random = phylo_results_df[phylo_results_df['pagels_lambda'] < 0.3]
    
    print("TAXONOMIC CONSERVATION (High λ):")
    print("-" * 80)
    print(f"These {len(conserved)} functions follow the evolutionary tree.")
    print("Interpretation: Vertically inherited, define major taxonomic groups")
    print("Example: Basic energy metabolism pathways")
    print()
    
    print("LOCALE CONSERVATION (Low λ):")
    print("-" * 80)
    print(f"These {len(random)} functions appear randomly across the tree.")
    print("Interpretation: Horizontally transferred, driven by environment")
    print("Example: Metal-resistance genes acquired via plasmids")
    print()
    
    print("This difference quantifies how much of community structure is driven by:")
    print("  • Phylogeny (historical inheritance)")
    print("  • Environment (contemporary adaptation)")


if __name__ == '__main__':
    logger.info(__doc__)
    
    print("""
Module 1: Functional Biogeography is now ready to use!

To use in your workflow:

1. Load your data:
   >>> import scanpy as sc
   >>> adata = sc.read_h5ad('your_data.h5ad')

2. Run the analysis:
   >>> from workflow_16s.downstream.functional_biogeography import (
   ...     analyze_functional_vs_taxonomic_conservation
   ... )
   >>> results = analyze_functional_vs_taxonomic_conservation(
   ...     adata,
   ...     otu_metadata_path='/path/to/otus.97.allinfo',
   ...     environmental_variable='pH',  # optional
   ...     output_dir='./conservation_results'
   ... )

3. Explore results:
   >>> results['phylogenetic_signal']  # Pagel's lambda values
   >>> results['trait_matrix']  # OTU × Trait matrix
   >>> results['environmental_associations']  # Trait-environment correlations

Key outputs answer:
✓ Which functions follow evolution (vertical inheritance)?
✓ Which functions cross taxonomic boundaries (horizontal transfer)?  
✓ Which environments select for which functions?
✓ Is community structure driven by phylogeny or adaptation?
    """)
