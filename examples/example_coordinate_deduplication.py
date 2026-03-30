"""
Example: Using Coordinate Deduplication with GEE Enrichment

This example demonstrates how to use the coordinate deduplication system
to reduce API calls when enriching environmental data from Google Earth Engine.
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_deduplication():
    """Example 1: Basic deduplication with sample coordinates."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Basic Coordinate Deduplication")
    logger.info("="*70)

    from workflow_16s.api.environmental_data.google.coordinate_deduplicator import CoordinateDeduplicator

    # Create sample coordinates (multiple samples from same locations)
    lats = np.array([
        40.7128, 40.7129,  # NYC area (2 samples)
        51.5074, 51.5075,  # London area (2 samples)
        34.0522, 34.0523,  # LA area (2 samples)
    ])

    lons = np.array([
        -74.0060, -74.0061,  # NYC
        -0.1278, -0.1279,    # London
        -118.2437, -118.2438,  # LA
    ])

    logger.info(f"Input: {len(lats)} samples")

    # Initialize deduplicator
    dedup = CoordinateDeduplicator(tolerance_meters=100)

    # Deduplicate
    u_lats, u_lons, mapping = dedup.deduplicate_coordinates(lats, lons)

    logger.info(f"Unique coordinates: {len(u_lats)}")
    logger.info(f"Mapping: {mapping}")
    logger.info("  (0 and 1 map to cluster 0, 2 and 3 map to cluster 1, etc.)")

    # Simulate GEE results
    gee_results = {
        0: {'elevation': 10, 'land_cover': 'urban'},
        1: {'elevation': 20, 'land_cover': 'water'},
        2: {'elevation': 5, 'land_cover': 'urban'},
    }

    # Expand back
    full_results = dedup.expand_results(gee_results, mapping)

    logger.info(f"\nExpanded results: {len(full_results)}")
    logger.info(f"Sample 0 result: {full_results.get(0)}")
    logger.info(f"Sample 1 result: {full_results.get(1)}")
    logger.info("Note: Samples 0 and 1 both get the same result (from cluster 0)")


def example_2_large_dataset():
    """Example 2: Large dataset with realistic clustering."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Large Dataset (463K samples)")
    logger.info("="*70)

    from workflow_16s.api.environmental_data.google.coordinate_deduplicator import CoordinateDeduplicator

    # Generate realistic dataset
    np.random.seed(42)
    n_samples = 463000

    # Create clustered coordinates (simulating field surveys)
    n_clusters = 200
    cluster_centers = np.random.uniform(-60, 60, (n_clusters, 2))
    assignment = np.random.randint(0, n_clusters, n_samples)

    # Add variation within clusters (±200m)
    lats = cluster_centers[assignment, 0] + np.random.normal(0, 0.0005, n_samples)
    lons = cluster_centers[assignment, 1] + np.random.normal(0, 0.0005, n_samples)

    logger.info(f"Dataset: {n_samples:,} samples from {n_clusters} survey locations")

    # Deduplicate
    dedup = CoordinateDeduplicator(tolerance_meters=100)
    u_lats, u_lons, mapping = dedup.deduplicate_coordinates(lats, lons)

    reduction = 100 * (1 - len(u_lats) / n_samples)
    api_calls_saved = n_samples - len(u_lats)

    logger.info(f"\nDeduplication Results:")
    logger.info(f"  Unique coordinates: {len(u_lats):,}")
    logger.info(f"  Reduction: {reduction:.1f}%")
    logger.info(f"  API calls saved: {api_calls_saved:,}")
    logger.info(f"\nPerformance Impact:")
    logger.info(f"  Without dedup: {n_samples:,} API calls")
    logger.info(f"  With dedup: {len(u_lats):,} API calls")
    logger.info(f"  Speedup: {n_samples/len(u_lats):.2f}x")


def example_3_with_geodataframe():
    """Example 3: Integration with geodata (realistic workflow)."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Integration with Sample Metadata")
    logger.info("="*70)

    from workflow_16s.api.environmental_data.google.coordinate_deduplicator import CoordinateDeduplicator

    # Create sample metadata (like in real enrichment)
    metadata = pd.DataFrame({
        'sample_id': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006'],
        'latitude': [40.7128, 40.7129, 51.5074, 51.5075, 34.0522, 34.0523],
        'longitude': [-74.0060, -74.0061, -0.1278, -0.1279, -118.2437, -118.2438],
        'collection_date': pd.date_range('2024-01-01', periods=6),
    })

    logger.info(f"\nOriginal metadata:")
    logger.info(metadata.to_string())

    # Extract coordinates
    lats = metadata['latitude'].values
    lons = metadata['longitude'].values

    # Deduplicate
    dedup = CoordinateDeduplicator(tolerance_meters=100)
    u_lats, u_lons, mapping = dedup.deduplicate_coordinates(lats, lons)

    # Add deduplication info to metadata
    metadata['dedup_cluster_id'] = mapping
    metadata['is_unique'] = metadata['dedup_cluster_id'].isin(range(len(u_lats)))

    logger.info(f"\nMetadata with deduplication info:")
    logger.info(metadata.to_string())

    logger.info(f"\nUnique locations: {len(u_lats)}")
    logger.info(f"Samples to query: {metadata['is_unique'].sum()}")
    logger.info(f"Samples that can reuse results: {(~metadata['is_unique']).sum()}")


def example_4_tolerance_comparison():
    """Example 4: Compare different tolerance levels."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Tolerance Level Comparison")
    logger.info("="*70)

    from workflow_16s.api.environmental_data.google.coordinate_deduplicator import CoordinateDeduplicator

    # Create test data
    np.random.seed(42)
    n_samples = 100000
    lats = np.random.uniform(-90, 90, n_samples)
    lons = np.random.uniform(-180, 180, n_samples)

    # Test different tolerances
    tolerances = [10, 50, 100, 200, 500]

    logger.info(f"\nTesting {n_samples:,} samples with different tolerances:")
    logger.info(f"{'Tolerance (m)':<20} {'Unique Coords':<20} {'Reduction':<15}")
    logger.info("-" * 55)

    for tol in tolerances:
        dedup = CoordinateDeduplicator(tolerance_meters=tol)
        u_lats, u_lons, mapping = dedup.deduplicate_coordinates(lats, lons, tol)
        reduction = 100 * (1 - len(u_lats) / n_samples)
        logger.info(f"{tol:<20} {len(u_lats):<20,} {reduction:<15.1f}%")


def example_5_statistics():
    """Example 5: Analyzing coordinate statistics."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Coordinate Distribution Statistics")
    logger.info("="*70)

    from workflow_16s.api.environmental_data.google.coordinate_deduplicator import CoordinateDeduplicator

    # Create clustered data
    np.random.seed(42)
    n_samples = 50000

    # Dense clusters
    n_clusters = 100
    cluster_centers = np.random.uniform(-60, 60, (n_clusters, 2))
    assignment = np.random.randint(0, n_clusters, n_samples)
    lats = cluster_centers[assignment, 0] + np.random.normal(0, 0.001, n_samples)
    lons = cluster_centers[assignment, 1] + np.random.normal(0, 0.001, n_samples)

    dedup = CoordinateDeduplicator(tolerance_meters=100)

    # Get statistics
    stats = dedup.get_deduplication_stats(lats, lons, tolerance_meters=100)

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total samples: {stats['n_input']:,}")
    logger.info(f"  Valid samples: {stats['n_valid']:,}")
    logger.info(f"  Unique latitudes: {stats['unique_lats']:,}")
    logger.info(f"  Unique longitudes: {stats['unique_lons']:,}")

    if 'min_distance_m' in stats:
        logger.info(f"\nNearest Neighbor Distances:")
        logger.info(f"  Minimum: {stats['min_distance_m']:,.0f}m")
        logger.info(f"  Maximum: {stats['max_distance_m']:,.0f}m")
        logger.info(f"  Mean: {stats['mean_distance_m']:,.0f}m")


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("COORDINATE DEDUPLICATION EXAMPLES")
    logger.info("="*70)

    try:
        example_1_basic_deduplication()
        example_2_large_dataset()
        example_3_with_geodataframe()
        example_4_tolerance_comparison()
        example_5_statistics()

        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
