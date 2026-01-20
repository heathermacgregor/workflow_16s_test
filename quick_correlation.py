#!/usr/bin/env python3
"""
Quick correlation analysis: Nitrososphaeraceae vs Anaeromyxobacter
Split by facility_match status
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys

print("Loading data (backed mode for speed)...")
DATA_FILE = "/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_4/final_processed_adata.h5ad"

# Load in backed mode (fast, read-only)
adata = sc.read_h5ad(DATA_FILE, backed='r')
print(f"✓ Loaded: {adata.n_obs} samples × {adata.n_vars} features")

# Check facility_match
if 'facility_match' not in adata.obs.columns:
    print("ERROR: facility_match not found!")
    sys.exit(1)

# Get facility_match as regular series (not categorical)
facility_match = adata.obs['facility_match'].copy()
if hasattr(facility_match, 'cat'):
    facility_match = facility_match.astype(str)

print(f"\nfacility_match distribution:")
print(facility_match.value_counts())

# Find target genera in .var
print("\nSearching for target genera in taxonomy...")
target1, target2 = "Nitrososphaeraceae", "Anaeromyxobacter"

# Check if Genus column exists
if 'Genus' not in adata.var.columns:
    print("ERROR: No 'Genus' column in .var!")
    print(f"Available .var columns: {list(adata.var.columns)}")
    sys.exit(1)

# Find matches
genera = adata.var['Genus'].astype(str)
mask1 = genera.str.contains(target1, case=False, na=False)
mask2 = genera.str.contains(target2, case=False, na=False)

idx1 = np.where(mask1)[0]
idx2 = np.where(mask2)[0]

if len(idx1) == 0:
    print(f"ERROR: No features found matching '{target1}'")
    print(f"Sample genera: {genera[~genera.isna()].unique()[:20]}")
    sys.exit(1)

if len(idx2) == 0:
    print(f"ERROR: No features found matching '{target2}'")
    sys.exit(1)

print(f"✓ Found {len(idx1)} features for {target1}")
print(f"✓ Found {len(idx2)} features for {target2}")

# Aggregate counts for each genus (sum across all matching features)
print("\nAggregating feature counts...")
if hasattr(adata.X, 'toarray'):
    counts1 = adata.X[:, idx1].toarray().sum(axis=1).A1
    counts2 = adata.X[:, idx2].toarray().sum(axis=1).A1
else:
    counts1 = np.array(adata.X[:, idx1].sum(axis=1)).flatten()
    counts2 = np.array(adata.X[:, idx2].sum(axis=1)).flatten()

# CLR transform
print("Applying CLR transformation...")
pseudocount = 1
def clr_transform(counts):
    counts_pseudo = counts + pseudocount
    geom_mean = np.exp(np.mean(np.log(counts_pseudo)))
    return np.log(counts_pseudo / geom_mean)

clr1 = clr_transform(counts1)
clr2 = clr_transform(counts2)

# Analyze by facility_match status
print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

results = []
for status in ['True', 'False']:
    mask = facility_match == status
    n = mask.sum()
    
    if n < 3:
        print(f"\nSkipping facility_match={status}: only {n} samples")
        continue
    
    x = clr1[mask]
    y = clr2[mask]
    
    rho, pval = spearmanr(x, y)
    
    results.append({
        'status': status,
        'n': n,
        'rho': rho,
        'pval': pval
    })
    
    print(f"\nfacility_match = {status}")
    print(f"  Samples: {n}")
    print(f"  {target1}: mean={x.mean():.3f} ± {x.std():.3f}")
    print(f"  {target2}: mean={y.mean():.3f} ± {y.std():.3f}")
    print(f"  Spearman ρ = {rho:.4f}")
    print(f"  P-value = {pval:.2e} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

# Interpretation
if len(results) == 2:
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    facility = results[0]
    non_facility = results[1]
    
    delta = facility['rho'] - non_facility['rho']
    
    print(f"\nΔρ (facility - non-facility) = {delta:+.4f}")
    
    if abs(delta) > 0.3:
        print("\n🔍 STRONG DIFFERENTIAL CORRELATION!")
    elif abs(delta) > 0.1:
        print("\n📊 Moderate differential correlation")
    else:
        print("\n✓ Similar correlations in both groups")
    
    print(f"\nFacility samples (n={facility['n']}):")
    if facility['rho'] > 0.3 and facility['pval'] < 0.05:
        print(f"  → POSITIVE correlation (ρ={facility['rho']:.3f})")
        print(f"  → Genera co-occur in facility environments")
    elif facility['rho'] < -0.3 and facility['pval'] < 0.05:
        print(f"  → NEGATIVE correlation (ρ={facility['rho']:.3f})")
        print(f"  → Genera show competitive exclusion")
    else:
        print(f"  → WEAK/NO correlation (ρ={facility['rho']:.3f})")
        print(f"  → Genera vary independently")
    
    print(f"\nNon-facility samples (n={non_facility['n']}):")
    if non_facility['rho'] > 0.3 and non_facility['pval'] < 0.05:
        print(f"  → POSITIVE correlation (ρ={non_facility['rho']:.3f})")
        print(f"  → Natural co-occurrence")
    elif non_facility['rho'] < -0.3 and non_facility['pval'] < 0.05:
        print(f"  → NEGATIVE correlation (ρ={non_facility['rho']:.3f})")
        print(f"  → Natural competition")
    else:
        print(f"  → WEAK/NO correlation (ρ={non_facility['rho']:.3f})")
        print(f"  → Independent variation")

print("\n" + "="*70)
print("✓ Analysis complete!")
print("="*70)
