#!/usr/bin/env python3
"""
Extract facility-specific microbial signatures from CatBoost results.
"""
import json
import pandas as pd
from pathlib import Path
import sys

def extract_facility_microbes(results_dir: Path, output_dir: Path = None):
    """
    Extract top predictive microbes for each facility-related target.
    
    Args:
        results_dir: Path to catboost_feature_selection directory
        output_dir: Where to save output CSV files (default: same as results_dir)
    """
    if output_dir is None:
        output_dir = results_dir.parent / "facility_microbe_summaries"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all facility-related result directories
    facility_targets = [
        'nuclear_contamination_status',
        'facility',
        'facility_match',
        'facility_type',
        'facility_status',
        'facility_capacity',
        'facility_distance_km'
    ]
    
    all_results = []
    
    for target in facility_targets:
        # Check both with and without Genus_ prefix
        for prefix in ['Genus_', 'Class_', 'Family_', 'Order_', 'Phylum_', '']:
            target_dir = results_dir / f"{prefix}{target}"
            summary_file = target_dir / "results_summary.json"
            
            if summary_file.exists():
                print(f"✓ Found: {target} ({prefix or 'no prefix'})")
                
                with open(summary_file) as f:
                    data = json.load(f)
                
                # Extract key info
                target_name = data.get('target', target)
                level = data.get('level', prefix.rstrip('_') or 'Unknown')
                task_type = data.get('task_type', 'Unknown')
                mcc = data.get('best_cv_score', 'N/A')
                test_scores = data.get('test_scores', {})
                top_features = data.get('top_features', [])
                
                # Create detailed record
                result_record = {
                    'target': target_name,
                    'taxonomic_level': level,
                    'task_type': task_type,
                    'cv_mcc': mcc,
                    'test_accuracy': test_scores.get('accuracy', 'N/A'),
                    'test_mcc': test_scores.get('mcc', 'N/A'),
                    'test_f1': test_scores.get('f1', 'N/A'),
                    'test_roc_auc': test_scores.get('roc_auc', 'N/A'),
                    'n_top_features': len(top_features),
                    'top_features': ', '.join([f.strip() for f in top_features[:10]])
                }
                all_results.append(result_record)
                
                # Save individual target details
                if top_features:
                    feature_df = pd.DataFrame({
                        'rank': range(1, len(top_features) + 1),
                        'genus': [f.strip() for f in top_features],
                        'target': target_name,
                        'taxonomic_level': level
                    })
                    feature_file = output_dir / f"{target_name}_top_microbes.csv"
                    feature_df.to_csv(feature_file, index=False)
                    print(f"  → Saved {len(top_features)} features to {feature_file.name}")
    
    # Save summary of all targets
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = output_dir / "facility_predictions_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")
        print(f"✓ Total targets analyzed: {len(all_results)}")
        
        # Print summary table
        print("\n" + "="*80)
        print("FACILITY MICROBE PREDICTION SUMMARY")
        print("="*80)
        for _, row in summary_df.iterrows():
            cv_mcc = f"{row['cv_mcc']:.3f}" if isinstance(row['cv_mcc'], (int, float)) else str(row['cv_mcc'])
            test_mcc = f"{row['test_mcc']:.3f}" if isinstance(row['test_mcc'], (int, float)) else str(row['test_mcc'])
            print(f"\n{row['target'].upper()}")
            print(f"  Level: {row['taxonomic_level']} | Task: {row['task_type']}")
            print(f"  CV MCC: {cv_mcc} | Test MCC: {test_mcc}")
            print(f"  Top microbes: {row['top_features'][:100]}...")
    else:
        print("⚠ No facility-related results found!")
    
    return output_dir

if __name__ == "__main__":
    # Default path
    default_path = Path("/usr2/people/macgregor/amplicon/project_01/04_analysis/testing_4/catboost_feature_selection")
    
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = default_path
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        print(f"Usage: {sys.argv[0]} [path_to_catboost_feature_selection]")
        sys.exit(1)
    
    output_dir = extract_facility_microbes(results_dir)
    print(f"\n✓ All outputs saved to: {output_dir}")
