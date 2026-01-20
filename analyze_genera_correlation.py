#!/usr/bin/env python3
"""
Quick analysis: Correlation between Nitrososphaeraceae and Anaeromyxobacter
Split by facility_match status (True vs False)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# Load the processed AnnData
DATA_FILE = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_4/final_processed_adata.h5ad"
print(f"Loading data from: {DATA_FILE}")
adata = sc.read_h5ad(DATA_FILE)

print(f"Loaded AnnData: {adata.n_obs} samples × {adata.n_vars} features")
print(f"Available .obs columns: {list(adata.obs.columns)[:10]}...")

# Check if facility_match exists
if 'facility_match' not in adata.obs.columns:
    print("ERROR: 'facility_match' column not found in metadata")
    print(f"Available metadata columns: {sorted(adata.obs.columns)}")
    exit(1)

# Aggregate to Genus level
print("\nAggregating to Genus level...")
genus_counts = {}
for var_idx, row in adata.var.iterrows():
    genus = row.get('Genus', 'Unassigned')
    if genus == 'Unassigned' or pd.isna(genus):
        continue
    if genus not in genus_counts:
        genus_counts[genus] = adata[:, var_idx].X.toarray().flatten() if hasattr(adata.X, 'toarray') else adata[:, var_idx].X.flatten()
    else:
        genus_counts[genus] += adata[:, var_idx].X.toarray().flatten() if hasattr(adata.X, 'toarray') else adata[:, var_idx].X.flatten()

genus_df = pd.DataFrame(genus_counts, index=adata.obs_names)
print(f"Aggregated to {genus_df.shape[1]} genera")

# Find the target genera (allowing partial matches)
target_genera = ['Nitrososphaeraceae', 'Anaeromyxobacter']
found_genera = []

for target in target_genera:
    matches = [g for g in genus_df.columns if target.lower() in g.lower()]
    if matches:
        found_genera.append(matches[0])  # Take first match
        print(f"  Found: {matches[0]} (target: {target})")
    else:
        print(f"  WARNING: No match for {target}")
        print(f"  Available genera with similar names: {[g for g in genus_df.columns if any(t in g.lower() for t in target.lower().split('_'))][:5]}")

if len(found_genera) < 2:
    print("\nERROR: Could not find both genera. Exiting.")
    print(f"Available genera (first 20): {sorted(genus_df.columns)[:20]}")
    exit(1)

genus1, genus2 = found_genera[0], found_genera[1]

# Apply CLR transformation (as used in the pipeline)
print(f"\nApplying CLR transformation...")
pseudocount = 1
genus_df_clr = genus_df.copy()
for col in genus_df_clr.columns:
    counts = genus_df[col] + pseudocount
    geometric_mean = np.exp(np.log(counts).mean())
    genus_df_clr[col] = np.log(counts / geometric_mean)

# Get facility_match status
facility_match = adata.obs['facility_match'].copy()
print(f"\nfacility_match distribution:")
print(facility_match.value_counts())

# Split by facility_match and calculate correlations
results = []

for match_status in [True, False, 'Unknown']:
    # Handle different possible values
    if match_status == 'Unknown':
        mask = facility_match.isna() | (facility_match.astype(str).str.lower() == 'unknown')
    elif match_status == True:
        mask = (facility_match == True) | (facility_match.astype(str).str.lower() == 'true')
    else:
        mask = (facility_match == False) | (facility_match.astype(str).str.lower() == 'false')
    
    n_samples = mask.sum()
    if n_samples < 3:
        print(f"\nSkipping facility_match={match_status}: only {n_samples} samples")
        continue
    
    # Get abundances for this group
    g1_abund = genus_df_clr.loc[mask, genus1].values
    g2_abund = genus_df_clr.loc[mask, genus2].values
    
    # Calculate Spearman correlation
    rho, pval = spearmanr(g1_abund, g2_abund)
    
    results.append({
        'facility_match': match_status,
        'n_samples': n_samples,
        'rho': rho,
        'p_value': pval,
        'genus1_mean': g1_abund.mean(),
        'genus1_std': g1_abund.std(),
        'genus2_mean': g2_abund.mean(),
        'genus2_std': g2_abund.std()
    })
    
    print(f"\n{'='*70}")
    print(f"facility_match = {match_status}")
    print(f"{'='*70}")
    print(f"  N samples: {n_samples}")
    print(f"  {genus1}:")
    print(f"    Mean CLR abundance: {g1_abund.mean():.3f} ± {g1_abund.std():.3f}")
    print(f"    Min/Max: {g1_abund.min():.3f} / {g1_abund.max():.3f}")
    print(f"  {genus2}:")
    print(f"    Mean CLR abundance: {g2_abund.mean():.3f} ± {g2_abund.std():.3f}")
    print(f"    Min/Max: {g2_abund.min():.3f} / {g2_abund.max():.3f}")
    print(f"\n  Spearman correlation (ρ): {rho:.4f}")
    print(f"  P-value: {pval:.2e}")
    print(f"  Significance: {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

# Summary interpretation
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df[['facility_match', 'n_samples', 'rho', 'p_value']].to_string(index=False))

# Compare correlations
if len(results) >= 2:
    facility_result = [r for r in results if r['facility_match'] == True]
    non_facility_result = [r for r in results if r['facility_match'] == False]
    
    if facility_result and non_facility_result:
        rho_facility = facility_result[0]['rho']
        rho_non_facility = non_facility_result[0]['rho']
        delta_rho = rho_facility - rho_non_facility
        
        print(f"\nΔρ (facility - non-facility) = {delta_rho:.4f}")
        
        if abs(delta_rho) > 0.3:
            print("\n🔍 STRONG DIFFERENTIAL CORRELATION:")
            if rho_facility > rho_non_facility:
                print(f"   {genus1} and {genus2} are MORE positively correlated")
                print(f"   in facility-associated samples (Δρ = {delta_rho:.3f})")
            else:
                print(f"   {genus1} and {genus2} are LESS correlated")
                print(f"   in facility-associated samples (Δρ = {delta_rho:.3f})")
        elif abs(delta_rho) > 0.1:
            print("\n📊 MODERATE DIFFERENTIAL CORRELATION:")
            print(f"   Correlation differs by Δρ = {delta_rho:.3f} between groups")
        else:
            print("\n✓ SIMILAR CORRELATIONS:")
            print(f"   Both groups show similar correlation patterns (Δρ = {delta_rho:.3f})")
        
        # Biological interpretation
        print("\nBIOLOGICAL CONTEXT:")
        if rho_facility > 0.3 and facility_result[0]['p_value'] < 0.05:
            print(f"   • In facility samples: POSITIVE correlation (ρ={rho_facility:.3f})")
            print(f"     → These genera may co-occur in facility-impacted environments")
            print(f"     → Could indicate shared ecological niche or metabolic coupling")
        elif rho_facility < -0.3 and facility_result[0]['p_value'] < 0.05:
            print(f"   • In facility samples: NEGATIVE correlation (ρ={rho_facility:.3f})")
            print(f"     → These genera may compete or occupy different niches")
            print(f"     → Competitive exclusion or opposite environmental preferences")
        else:
            print(f"   • In facility samples: WEAK/NO correlation (ρ={rho_facility:.3f})")
            print(f"     → Genera appear to vary independently")
        
        if rho_non_facility > 0.3 and non_facility_result[0]['p_value'] < 0.05:
            print(f"\n   • In non-facility samples: POSITIVE correlation (ρ={rho_non_facility:.3f})")
            print(f"     → Natural co-occurrence pattern")
        elif rho_non_facility < -0.3 and non_facility_result[0]['p_value'] < 0.05:
            print(f"\n   • In non-facility samples: NEGATIVE correlation (ρ={rho_non_facility:.3f})")
            print(f"     → Natural competitive dynamics")
        else:
            print(f"\n   • In non-facility samples: WEAK/NO correlation (ρ={rho_non_facility:.3f})")
            print(f"     → Independent variation in natural environments")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
