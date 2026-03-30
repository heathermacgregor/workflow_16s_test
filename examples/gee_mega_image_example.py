#!/usr/bin/env python
"""
GEE Mega-Image Example Usage Script

This script demonstrates how to use the mega-image system for efficient
GEE enrichment of large amplicon datasets (400K+ samples).

Usage:
    cd /usr2/people/macgregor/amplicon/workflow_16s
    python examples/gee_mega_image_example.py

Expected Output:
    🚀 GEE enrichment for 400000 samples
      ✓ Found coordinates: 400000/400000 samples (100.0%)
    → MEGA-IMAGE MODE: Async parallel exports to Cloud Storage
    Step 1: Creating mega-image from enabled datasets...
    ✓ Mega-image created: 8 bands from 4 datasets
    Step 2: Converting coordinates to FeatureCollection...
    ✓ FeatureCollection created: 400000 coordinates
    Step 3: Starting async exports to Cloud Storage...
    ✓ Submitted 40 export tasks to Cloud Storage
    Step 4: Monitoring export tasks and downloading results...
    ✓ All export tasks completed
    ✓ Downloaded and merged 400000 samples
    ✅ Mega-image enrichment complete: 8 new columns added
    
    Enrichment time: ~50-60 minutes for 400K samples (vs 16.6 hours)
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from workflow_16s.api.environmental_data.other.tools._google_earth_engine import enrich_with_gee_data


def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample observation metadata for testing.
    
    Args:
        n_samples: Number of test samples (default 100)
    
    Returns:
        DataFrame with latitude, longitude, and metadata columns
    """
    print(f"Creating sample data with {n_samples} observations...")
    
    # Generate random coordinates across Africa & Europe for testing
    np.random.seed(42)
    
    lats = np.random.uniform(-35, 72, n_samples)  # Africa to Europe
    lons = np.random.uniform(-37, 55, n_samples)  # Africa to Eastern Europe
    
    df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'sample_id': [f'SAMPLE_{i:06d}' for i in range(n_samples)],
        'study_id': [f'STUDY_{i%5}' for i in range(n_samples)],
        'depth_cm': np.random.uniform(0, 50, n_samples),
    })
    
    print(f"  ✓ Created {len(df)} samples")
    print(f"  Lat range: {df['latitude'].min():.1f}° to {df['latitude'].max():.1f}°")
    print(f"  Lon range: {df['longitude'].min():.1f}° to {df['longitude'].max():.1f}°")
    
    return df


def load_config() -> dict:
    """
    Load GEE configuration from config.yaml.
    
    Returns:
        Config dict, or empty dict if file not found
    """
    config_path = PROJECT_ROOT / 'config.yaml'
    
    if config_path.exists():
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("  Creating minimal config...")
        return {
            'gee_assets': {
                'enabled': True,
                'tiers': {'HIGH': True, 'STANDARD': True, 'REGIONAL': True},
                'datasets': {
                    'copernicus_dem': {'enabled': True, 'tier': 'STANDARD'},
                    'era5_climate': {'enabled': True, 'tier': 'STANDARD'},
                    'jrc_global_water': {'enabled': True, 'tier': 'HIGH'},
                    'viirs_nighttime_lights': {'enabled': True, 'tier': 'HIGH'},
                }
            }
        }


def example_standard_mode(obs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Example 1: Standard mode (batch queries) - slower, simpler.
    
    Best for: Small datasets (< 50K samples)
    Time: ~50 min for 50K samples
    
    Args:
        obs: Observation DataFrame with coordinates
        config: GEE configuration dict
    
    Returns:
        Enriched DataFrame with new columns
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: STANDARD MODE (Batch Queries)")
    print("="*70)
    print(f"Enriching {len(obs)} samples with batch sampling...")
    print("Expected time: 10 min per 10K samples")
    
    try:
        enriched = enrich_with_gee_data(
            obs,
            auth_flag=False,  # Set to True if GEE authenticated
            batch_size=30,
            use_cache=True,
            use_mega_image=False,  # Standard mode
            gee_config=config.get('gee_assets')
        )
        
        new_cols = [c for c in enriched.columns if c not in obs.columns]
        print(f"\n✅ Standard mode enrichment complete!")
        print(f"   New columns: {len(new_cols)}")
        if new_cols:
            print(f"   Example: {', '.join(new_cols[:5])}")
        
        return enriched
        
    except Exception as e:
        print(f"⚠️  Standard mode not available: {e}")
        return obs


def example_mega_image_mode(obs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Example 2: Mega-image mode (async exports) - faster, cloud-based.
    
    Best for: Large datasets (400K+ samples)
    Time: ~50-60 min for 400K samples (16x speedup!)
    
    Requires:
    - GEE authentication: earthengine authenticate
    - GCS bucket: gsutil mb gs://my-research-bucket
    - GCS auth: gcloud auth application-default login
    
    Args:
        obs: Observation DataFrame with coordinates
        config: GEE configuration dict
    
    Returns:
        Enriched DataFrame with new columns
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: MEGA-IMAGE MODE (Async Exports to Cloud Storage)")
    print("="*70)
    print(f"Enriching {len(obs)} samples with mega-image mode...")
    print("Expected time: 50-60 min for 400K samples (16x speedup!)")
    print("\nRequired Setup:")
    print("  1. GEE: earthengine authenticate")
    print("  2. GCS: gsutil mb gs://my-research-bucket")
    print("  3. GCS: gcloud auth application-default login")
    
    # Configuration
    GCS_BUCKET = 'my-research-bucket'  # ⚠️  Change this to your bucket!
    GCS_PROJECT = os.environ.get('GCP_PROJECT', 'my-gcp-project')  # Optional
    
    print("\nSettings:")
    print(f"  GCS Bucket: {GCS_BUCKET}")
    print(f"  GCP Project: {GCS_PROJECT}")
    
    try:
        enriched = enrich_with_gee_data(
            obs,
            auth_flag=False,  # Set to True if GEE authenticated
            use_mega_image=True,  # Mega-image mode!
            gee_config=config.get('gee_assets'),
            gcs_bucket=GCS_BUCKET,
            gcs_project=GCS_PROJECT,
            gcs_output_dir='/tmp/gee_exports'
        )
        
        new_cols = [c for c in enriched.columns if c not in obs.columns]
        print(f"\n✅ Mega-image enrichment complete!")
        print(f"   New columns: {len(new_cols)}")
        if new_cols:
            print(f"   Example: {', '.join(new_cols[:5])}")
        
        return enriched
        
    except Exception as e:
        print(f"⚠️  Mega-image mode not available (expected if not authenticated):")
        print(f"    {e}")
        return obs


def example_direct_functions(obs: pd.DataFrame, config: dict) -> None:
    """
    Example 3: Direct function usage (advanced).
    
    Demonstrates calling the 4 functions directly:
    1. create_mega_image_from_config()
    2. convert_coordinates_to_feature_collection()
    3. export_mega_image_samples()
    4. monitor_and_download_exports()
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: DIRECT FUNCTION USAGE (Advanced)")
    print("="*70)
    print("This example shows calling the 4 mega-image functions directly.")
    print("(Requires GEE authentication and GCS access)\n")
    
    try:
        from workflow_16s.api.environmental_data.other.tools._google_earth_engine import (
            create_mega_image_from_config,
            convert_coordinates_to_feature_collection,
            export_mega_image_samples,
            monitor_and_download_exports
        )
        
        print("Step 1: Create mega-image from config...")
        mega_img = create_mega_image_from_config(config.get('gee_assets', {}))
        
        if mega_img:
            print(f"  ✓ Mega-image created")
            print(f"\nStep 2: Convert coordinates to FeatureCollection...")
            
            # Extract coordinates from sample data
            lats = obs['latitude'].values
            lons = obs['longitude'].values
            sample_ids = np.arange(len(obs))
            
            fc = convert_coordinates_to_feature_collection(lats, lons, sample_ids)
            
            if fc:
                print(f"  ✓ FeatureCollection created with {len(obs)} features")
                print(f"\nStep 3: Export mega-image samples...")
                
                result = export_mega_image_samples(
                    fc=fc,
                    mega_image=mega_img,
                    bucket='my-research-bucket',  # ⚠️  Change to your bucket!
                    batch_size=10000
                )
                
                if result:
                    print(f"  ✓ Submitted {result['batch_count']} export tasks")
                    print(f"\nStep 4: Monitor & download exports...")
                    print(f"  (This will take some time...)")
                    
                    merged_df = monitor_and_download_exports(
                        task_dict=result,
                        output_dir='/tmp/gee_mega_exports',
                        poll_interval=60,
                        max_wait_hours=24
                    )
                    
                    if merged_df is not None:
                        print(f"  ✓ Downloaded {len(merged_df)} samples")
                        print(f"  Columns: {list(merged_df.columns[:5])}...")
        else:
            print("  ⚠️  Could not create mega-image (GEE not initialized)")
            
    except Exception as e:
        print(f"⚠️  Direct function usage not available: {e}")


def main():
    """Main example script."""
    print("\n" + "="*70)
    print("GEE MEGA-IMAGE EXAMPLES")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT.name}")
    
    # Load or create sample data
    obs = create_sample_data(n_samples=100)  # Use 100 for testing
    # For production: obs = create_sample_data(n_samples=400000)
    
    # Load configuration
    config = load_config()
    
    # Run examples
    print("\n" + "="*70)
    print("CHOOSE YOUR ENRICHMENT MODE")
    print("="*70)
    
    # Example 1: Standard mode (batch queries)
    obs_standard = example_standard_mode(obs, config)
    
    # Example 2: Mega-image mode (async exports)
    obs_mega = example_mega_image_mode(obs, config)
    
    # Example 3: Direct function usage
    example_direct_functions(obs, config)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original samples: {len(obs)}")
    print(f"Standard mode returned: {len(obs_standard)} samples")
    print(f"Mega-image mode returned: {len(obs_mega)} samples")
    
    print("\nMode Comparison:")
    print("  Standard Mode: Good for < 50K samples, simpler setup")
    print("  Mega-Image Mode: Good for 400K+ samples, 16x faster!")
    
    print("\nFor Production Use:")
    print("1. Set GCS_BUCKET to your actual Cloud Storage bucket")
    print("2. Authenticate: earthengine authenticate && gcloud auth application-default login")
    print("3. Use mega-image mode for large datasets:")
    print(f"   enriched = enrich_with_gee_data(obs, auth_flag=True, use_mega_image=True, gcs_bucket='...')")
    
    print("\nFor More Information:")
    print("  - GEE_MEGA_IMAGE_IMPLEMENTATION.md (comprehensive guide)")
    print("  - GEE_MEGA_IMAGE_QUICK_REFERENCE.md (quick reference)")
    print("  - workflow_16s/tests/test_gee_mega_image.py (test examples)")
    
    print("\n" + "="*70)
    print("✅ Examples complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
