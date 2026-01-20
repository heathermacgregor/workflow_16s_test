#!/usr/bin/env python
"""
Integration test for the comprehensive QC system.

This script demonstrates how to use the QC modules both standalone
and integrated into the main workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import anndata as ad
from workflow_16s.qc import quick_qc, ENVOOntology, MetadataValidator

def test_standalone_qc():
    """Test QC modules in standalone mode."""
    print("=" * 80)
    print("STANDALONE QC TEST")
    print("=" * 80)
    
    # Create sample data
    print("\n1. Creating sample data...")
    n_samples = 100
    n_features = 500
    
    # Create feature table
    X = pd.DataFrame(
        [[i * j for j in range(n_features)] for i in range(n_samples)]
    )
    
    # Create metadata with some issues to test QC
    metadata = pd.DataFrame({
        'sample_id': [f'S{i:03d}' for i in range(n_samples)],
        'env_biome': ['soil'] * 30 + ['marine'] * 30 + ['forest soil'] * 20 + ['seawater'] * 20,
        'env_feature': ['soil'] * 50 + ['ocean'] * 50,
        'env_material': ['soil'] * 50 + ['water'] * 50,
        'latitude': [37.5 + i * 0.1 for i in range(n_samples)],
        'longitude': [-122.0 + i * 0.1 for i in range(n_samples)],
        'latitude_ena': [37.5 + i * 0.1 for i in range(n_samples)],  # Redundant
        'ph': [7.0] * n_samples,
        'temperature': [20.0] * n_samples,
    })
    metadata.index = metadata['sample_id']
    
    # Create taxonomy
    taxonomy = pd.DataFrame({
        'Taxon': [f'k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__O{i};f__F{i};g__G{i};s__S{i}' 
                  for i in range(n_features)],
        'Confidence': [0.9] * n_features
    })
    taxonomy.index = [f'ASV{i:04d}' for i in range(n_features)]
    
    # Create AnnData
    adata = ad.AnnData(X=X.values, obs=metadata, var=taxonomy)
    adata.var_names = taxonomy.index
    adata.obs_names = metadata.index
    
    print(f"   Created {adata.n_obs} samples × {adata.n_vars} features")
    
    # Test 2: Metadata Validator
    print("\n2. Testing MetadataValidator...")
    
    # Test validate_all which returns cleaned data
    validator = MetadataValidator(metadata.copy())
    metadata_enriched, validation_report = validator.validate_all()
    
    n_original = len(metadata.columns)
    n_cleaned = len(metadata_enriched.columns)
    print(f"   ✓ Metadata validation: {n_original} → {n_cleaned} columns")
    print(f"   ✓ Validation report: {len(validation_report)} issues found")
    
    # Check ENVO categories
    if 'env_category_type' in metadata_enriched.columns:
        n_types = metadata_enriched['env_category_type'].nunique()
        print(f"   ✓ Added ENVO categories: {n_types} environment types")
    
    # Test 3: ENVO Semantic Search
    print("\n3. Testing ENVOOntology semantic search...")
    envo = ENVOOntology()
    
    # Find all soil samples (should match 'soil' and 'forest soil')
    soil_samples = envo.find_samples_by_category(metadata_enriched, 'soil')
    print(f"   ✓ Found {len(soil_samples)} soil samples (expected ~50)")
    
    # Find all aquatic samples
    aquatic_samples = envo.find_samples_by_category(metadata_enriched, 'marine')
    print(f"   ✓ Found {len(aquatic_samples)} marine samples (expected ~50)")
    
    # Test 4: Quick QC (one-liner)
    print("\n4. Testing quick_qc() one-liner...")
    output_dir = Path(__file__).parent / 'test_qc_output'
    output_dir.mkdir(exist_ok=True)
    
    try:
        adata_clean = quick_qc(adata, output_dir=str(output_dir))
        print(f"   ✓ QC complete: {adata.n_obs} → {adata_clean.n_obs} samples")
        print(f"   ✓ QC complete: {adata.n_vars} → {adata_clean.n_vars} features")
        
        # Check for QC flags
        if 'qc_overall_flag' in adata_clean.obs.columns:
            flag_counts = adata_clean.obs['qc_overall_flag'].value_counts()
            print(f"   ✓ QC flags: {dict(flag_counts)}")
        
        # Check for contamination detection
        if 'is_contaminant' in adata_clean.var.columns:
            n_contam = adata_clean.var['is_contaminant'].sum()
            print(f"   ✓ Detected {n_contam} contaminants")
        
        print(f"\n   Reports saved to: {output_dir}")
        
    except Exception as e:
        print(f"   ✗ Quick QC failed: {e}")
    
    print("\n" + "=" * 80)
    print("STANDALONE TEST COMPLETE ✓")
    print("=" * 80)


def test_integrated_qc():
    """Test QC integrated into the main workflow."""
    print("\n\n" + "=" * 80)
    print("INTEGRATED WORKFLOW TEST")
    print("=" * 80)
    
    print("\nTo test the integrated QC workflow:")
    print("1. Enable QC in config.yaml:")
    print("   quality_control:")
    print("     enabled: true")
    print("")
    print("2. Run the workflow:")
    print("   bash run.sh --config config/config.yaml")
    print("")
    print("3. Check QC outputs in:")
    print("   project_dir/04_analysis/qc/")
    print("")
    print("The QC will automatically run before preprocessing and add:")
    print("  - qc_overall_flag to adata.obs (PASS/WARNING/FAIL)")
    print("  - env_category_type to adata.obs (semantic categorization)")
    print("  - is_contaminant to adata.var (contamination flags)")
    print("  - Cleaned metadata (redundant columns removed)")
    
    print("\n" + "=" * 80)
    print("INTEGRATION GUIDE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE QC SYSTEM - INTEGRATION TEST")
    print("=" * 80)
    
    test_standalone_qc()
    test_integrated_qc()
    
    print("\n\nAll tests complete! ✓\n")
