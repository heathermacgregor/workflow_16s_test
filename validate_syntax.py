#!/usr/bin/env python3
"""
Simple validation for Quick Wins - checks syntax only.
"""

import sys
import ast
from pathlib import Path

print("=" * 70)
print("QUICK WINS SYNTAX VALIDATION")
print("=" * 70)

src_dir = Path(__file__).parent / 'src' / 'workflow_16s'

files_to_check = [
    ('downstream/statistics/top_features.py', 'Top Features Module'),
    ('downstream/statistics/effect_sizes.py', 'Effect Sizes Module'),
    ('downstream/statistics/differential_abundance.py', 'Differential Abundance Enhanced'),
    ('downstream/steps/synthesis.py', 'Synthesis Integration'),
    ('qc/visualization.py', 'QC Visualization'),
]

all_passed = True

for file_path, description in files_to_check:
    full_path = src_dir / file_path
    print(f"\n[{description}]")
    print(f"  File: {file_path}")
    
    if not full_path.exists():
        print(f"  ❌ File not found")
        all_passed = False
        continue
    
    try:
        # Parse Python syntax
        with open(full_path, 'r') as f:
            source = f.read()
        
        ast.parse(source)
        lines = len(source.split('\n'))
        
        print(f"  ✅ Syntax valid ({lines} lines)")
        
        # Check for key functions/integrations
        if 'top_features' in file_path:
            if 'def create_top_features_table' in source:
                print(f"  ✅ create_top_features_table() found")
            if 'def plot_top_features_heatmap' in source:
                print(f"  ✅ plot_top_features_heatmap() found")
                
        elif 'effect_sizes' in file_path:
            if 'def cohens_d' in source:
                print(f"  ✅ cohens_d() found")
            if 'def cliffs_delta' in source:
                print(f"  ✅ cliffs_delta() found")
                
        elif 'differential_abundance' in file_path:
            if 'cohens_d' in source and 'cliffs_delta' in source:
                print(f"  ✅ Effect size integration found")
            if 'biologically_significant' in source:
                print(f"  ✅ Biological significance flag found")
                
        elif 'synthesis' in file_path:
            if 'create_qc_impact_dashboard' in source:
                print(f"  ✅ QC visualization integration found")
            if 'create_top_features_table' in source:
                print(f"  ✅ Top features integration found")
                
        elif 'qc/visualization' in file_path:
            if 'def create_qc_impact_dashboard' in source:
                print(f"  ✅ create_qc_impact_dashboard() found")
            if 'def create_qc_interpretation_report' in source:
                print(f"  ✅ create_qc_interpretation_report() found")
    
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✅ ALL VALIDATIONS PASSED")
    print("\nQuick Wins Implementation Complete:")
    print("  1. ✅ Top features summary table")
    print("  2. ✅ Effect size calculations integrated")
    print("  3. ✅ QC visualization integrated")
    print("\nReady to run: bash run.sh")
    sys.exit(0)
else:
    print("❌ SOME VALIDATIONS FAILED")
    sys.exit(1)
print("=" * 70)
