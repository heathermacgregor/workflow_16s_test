#!/usr/bin/env python3
"""
FAST correlation analysis: Nitrososphaeraceae vs Anaeromyxobacter by facility_match
Uses backed mode + selective loading for speed
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import time

print("="*70)
print("LOADING DATA (backed mode for speed)...")
print("="*70)

# Use backed mode - only loads metadata, not full matrix
start = time.time()
adata = sc.read_h5ad(
    "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_4/final_processed_adata.h5ad",
    backed='r'
)
print(f"✓ Loaded in {time.time()-start:.1f}s: {adata.n_obs} samples × {adata.n_vars} features")

# Check facility_match
if 'facility_match' not in adata.obs.columns:
    print(f"\nERROR: 'facility_match' not found!")
    print(f"Available columns: {sorted(adata.obs.columns)[:20]}...")
    exit(1)

print(f"\nfacility_match distribution:")
fm_counts = adata.obs['facility_match'].value_counts()
print(fm_counts)

# Find target genera in .var (taxonomy columns)
print("\n" + "="*70)
print("FINDING TARGET GENERA...")
print("="*70)

if 'Genus' not in adata.var.columns:
    print("ERROR: 'Genus' column not in .var!")
    print(f"Available .var columns: {list(adata.var.columns)}")
    exit(1)

# Search for genera
target_genera = ['Nitrososphaeraceae', 'Anaeromyxobacter']
genus_col = adata.var['Genus'].astype(str)  # Convert categorical to string
found_features = {}

for target in target_genera:
    # Case-insensitive search
    matches = genus_col[genus_col.str.contains(target, case=False, na=False)]
    if len(matches) > 0:
        # Get all feature indices for this genus
        feature_indices = matches.index.tolist()
        found_features[target] = feature_indices
        print(f"✓ Found {len(feature_indices)} features for {target}")
        print(f"  First few: {feature_indices[:3]}")
    else:
        print(f"✗ No matches for {target}")
        # Show similar genera
        similar = genus_col[genus_col.str.contains(target[:5], case=False, na=False)].unique()[:5]
        if len(similar) > 0:
            print(f"  Similar genera: {list(similar)}")

if len(found_features) < 2:
    print("\nERROR: Need both genera. Exiting.")
    exit(1)

# Extract and aggregate counts for each genus
print("\n" + "="*70)
print("EXTRACTING COUNTS...")
print("="*70)

genus_abundances = {}
for genus_name, feature_list in found_features.items():
    print(f"Extracting {len(feature_list)} features for {genus_name}...")
    # Load just these features (not entire matrix)
    counts_list = []
    for feat_idx in feature_list[:100]:  # Limit to first 100 if many features
        # Access single column - adata.X is chunked/backed, so this is efficient
        col_data = adata[:, feat_idx].X
        if hasattr(col_data, 'toarray'):
            col_data = col_data.toarray().flatten()
        elif hasattr(col_data, 'flatten'):
            col_data = col_data.flatten()
        counts_list.append(col_data)
    
    # Sum across all features of this genus
    genus_total = np.sum(counts_list, axis=0)
    genus_abundances[genus_name] = genus_total
    print(f"  Total abundance range: {genus_total.min():.0f} - {genus_total.max():.0f}")

# Apply CLR transformation
print("\n" + "="*70)
print("APPLYING CLR TRANSFORMATION...")
print("="*70)

genus_clr = {}
for genus_name, counts in genus_abundances.items():
    pseudocount = 1
    counts_pc = counts + pseudocount
    geom_mean = np.exp(np.log(counts_pc).mean())
    clr = np.log(counts_pc / geom_mean)
    genus_clr[genus_name] = clr
    print(f"  {genus_name}: CLR range = [{clr.min():.2f}, {clr.max():.2f}]")

# Get the two genera
genera_names = list(found_features.keys())
genus1, genus2 = genera_names[0], genera_names[1]
g1_clr = genus_clr[genus1]
g2_clr = genus_clr[genus2]

# Get facility_match for correlation analysis
facility_match = adata.obs['facility_match']

print("\n" + "="*70)
print("CALCULATING CORRELATIONS")
print("="*70)

results = []

for match_status in [True, False]:
    # Create mask for this group
    if match_status == True:
        mask = (facility_match == True) | (facility_match.astype(str).str.lower() == 'true')
    else:
        mask = (facility_match == False) | (facility_match.astype(str).str.lower() == 'false')
    
    n_samples = mask.sum()
    if n_samples < 3:
        print(f"\nSkipping facility_match={match_status}: only {n_samples} samples")
        continue
    
    # Get abundances for this group
    g1_subset = g1_clr[mask]
    g2_subset = g2_clr[mask]
    
    # Calculate Spearman correlation
    rho, pval = spearmanr(g1_subset, g2_subset)
    
    results.append({
        'facility_match': match_status,
        'n_samples': n_samples,
        'rho': rho,
        'p_value': pval
    })
    
    print(f"\n{'─'*70}")
    print(f"FACILITY_MATCH = {match_status}")
    print(f"{'─'*70}")
    print(f"  N samples: {n_samples:,}")
    print(f"\n  {genus1} (CLR-transformed):")
    print(f"    Mean: {g1_subset.mean():8.3f} ± {g1_subset.std():.3f}")
    print(f"    Range: [{g1_subset.min():7.2f}, {g1_subset.max():7.2f}]")
    print(f"\n  {genus2} (CLR-transformed):")
    print(f"    Mean: {g2_subset.mean():8.3f} ± {g2_subset.std():.3f}")
    print(f"    Range: [{g2_subset.min():7.2f}, {g2_subset.max():7.2f}]")
    print(f"\n  📊 Spearman correlation (ρ): {rho:+.4f}")
    print(f"     P-value: {pval:.2e}")
    
    # Significance
    if pval < 0.001:
        sig = "***"
        interp = "HIGHLY SIGNIFICANT"
    elif pval < 0.01:
        sig = "**"
        interp = "VERY SIGNIFICANT"
    elif pval < 0.05:
        sig = "*"
        interp = "SIGNIFICANT"
    else:
        sig = "ns"
        interp = "NOT SIGNIFICANT"
    print(f"     Significance: {sig} ({interp})")
    
    # Correlation strength
    if abs(rho) > 0.7:
        strength = "STRONG"
    elif abs(rho) > 0.4:
        strength = "MODERATE"
    elif abs(rho) > 0.2:
        strength = "WEAK"
    else:
        strength = "VERY WEAK/NONE"
    
    if rho > 0:
        direction = "POSITIVE (co-occurrence)"
    else:
        direction = "NEGATIVE (anti-correlation)"
    
    print(f"     Strength: {strength} {direction}")

# Final interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if len(results) >= 2:
    fac_res = results[0] if results[0]['facility_match'] == True else results[1]
    non_fac_res = results[1] if results[0]['facility_match'] == True else results[0]
    
    delta_rho = fac_res['rho'] - non_fac_res['rho']
    
    print(f"\n📈 Correlation Comparison:")
    print(f"   Facility samples:     ρ = {fac_res['rho']:+.4f} (p={fac_res['p_value']:.2e})")
    print(f"   Non-facility samples: ρ = {non_fac_res['rho']:+.4f} (p={non_fac_res['p_value']:.2e})")
    print(f"   Δρ (difference):      {delta_rho:+.4f}")
    
    if abs(delta_rho) > 0.3:
        print(f"\n🔍 STRONG DIFFERENTIAL CORRELATION!")
        if delta_rho > 0:
            print(f"   → Genera are MORE positively correlated in facility samples")
            print(f"   → Suggests facility-specific co-occurrence pattern")
        else:
            print(f"   → Genera are LESS correlated in facility samples")
            print(f"   → Facility conditions may disrupt natural association")
    elif abs(delta_rho) > 0.1:
        print(f"\n📊 Moderate difference in correlation patterns")
    else:
        print(f"\n✓ Similar correlation in both groups")
    
    print(f"\n🧬 Biological Context:")
    print(f"   {genus1}:")
    print(f"     - Family Nitrososphaeraceae (ammonia-oxidizing archaea)")
    print(f"     - Key players in nitrogen cycling")
    print(f"   {genus2}:")
    print(f"     - Metal-reducing bacteria")
    print(f"     - Often found in contaminated/reduced environments")
    
    if fac_res['rho'] > 0.3 and fac_res['p_value'] < 0.05:
        print(f"\n   ✓ In facility samples: POSITIVE correlation")
        print(f"     → Both thrive in facility-impacted conditions")
        print(f"     → May indicate shared response to contamination/disturbance")
        print(f"     → Could suggest coupled nitrogen/metal biogeochemistry")
    elif fac_res['rho'] < -0.3 and fac_res['p_value'] < 0.05:
        print(f"\n   ✓ In facility samples: NEGATIVE correlation")
        print(f"     → Competitive or mutually exclusive niches")
        print(f"     → Different microenvironmental preferences")
    
print("\n" + "="*70)
print(f"Analysis completed in {time.time()-start:.1f} seconds")
print("="*70)
