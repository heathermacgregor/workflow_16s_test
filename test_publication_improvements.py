#!/usr/bin/env python3
"""
Test script for publication fetcher improvements.

This script tests the enhanced publication fetcher and metadata extraction
with rate limiting, caching, and Unpaywall integration.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test imports
try:
    from workflow_16s.config import get_config
    from workflow_16s.utils.publication_fetcher import PublicationFetcher
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_rate_limiting():
    """Test that rate limiting is working correctly."""
    print("\n" + "="*60)
    print("TEST 1: Rate Limiting")
    print("="*60)
    
    config = get_config("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    fetcher = PublicationFetcher(config=config, cache_path=None)
    
    # Test NCBI rate limiting (should take >= 0.68s for 3 calls)
    print("Testing NCBI rate limiting (3 consecutive calls)...")
    start = time.time()
    
    for i in range(3):
        fetcher._rate_limit('ncbi')
        print(f"  Call {i+1} completed")
    
    elapsed = time.time() - start
    expected_min = 0.34 * 2  # 2 intervals between 3 calls
    
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"Expected minimum: {expected_min:.3f}s")
    
    if elapsed >= expected_min * 0.9:  # Allow 10% tolerance
        print("✓ Rate limiting working correctly")
        return True
    else:
        print("✗ Rate limiting may not be enforcing delays")
        return False

def test_cache_creation():
    """Test enhanced cache table creation."""
    print("\n" + "="*60)
    print("TEST 2: Enhanced Cache Creation")
    print("="*60)
    
    cache_path = "/tmp/test_publication_cache.db"
    if Path(cache_path).exists():
        Path(cache_path).unlink()
    
    config = get_config("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    fetcher = PublicationFetcher(config=config, cache_path=cache_path)
    
    # Check cache file was created
    if not Path(cache_path).exists():
        print("✗ Cache file not created")
        return False
    
    print("✓ Cache file created")
    
    # Check tables exist
    import sqlite3
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    
    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    
    expected_tables = ['publication_cache', 'doi_metadata_cache', 'failed_lookups']
    
    print(f"Found tables: {table_names}")
    
    for table in expected_tables:
        if table in table_names:
            print(f"✓ Table '{table}' exists")
        else:
            print(f"✗ Table '{table}' missing")
            conn.close()
            return False
    
    # Check indices
    indices = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    index_names = [i[0] for i in indices]
    
    print(f"Found indices: {index_names}")
    
    conn.close()
    
    # Cleanup
    Path(cache_path).unlink()
    
    return True

def test_api_statistics():
    """Test API statistics tracking."""
    print("\n" + "="*60)
    print("TEST 3: API Statistics Tracking")
    print("="*60)
    
    config = get_config("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    fetcher = PublicationFetcher(config=config, cache_path=None)
    
    # Check attributes exist
    attributes = ['api_calls', 'source_success', 'rate_limits', 'last_request_times']
    
    for attr in attributes:
        if hasattr(fetcher, attr):
            value = getattr(fetcher, attr)
            print(f"✓ Attribute '{attr}' exists: {type(value).__name__}")
        else:
            print(f"✗ Attribute '{attr}' missing")
            return False
    
    # Check api_calls structure
    expected_keys = ['total', 'cached', 'failed']
    for key in expected_keys:
        if key in fetcher.api_calls:
            print(f"✓ api_calls['{key}'] exists")
        else:
            print(f"✗ api_calls['{key}'] missing")
            return False
    
    # Check rate_limits structure
    expected_apis = ['ncbi', 'crossref', 'semantic_scholar', 'europe_pmc', 'unpaywall', 'default']
    for api in expected_apis:
        if api in fetcher.rate_limits:
            print(f"✓ rate_limits['{api}'] = {fetcher.rate_limits[api]}s")
        else:
            print(f"✗ rate_limits['{api}'] missing")
            return False
    
    return True

def test_unpaywall_method():
    """Test that Unpaywall method exists and has correct signature."""
    print("\n" + "="*60)
    print("TEST 4: Unpaywall Integration")
    print("="*60)
    
    config = get_config("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    fetcher = PublicationFetcher(config=config, cache_path=None)
    
    # Check method exists
    if not hasattr(fetcher, '_get_open_access_pdf_url'):
        print("✗ Method '_get_open_access_pdf_url' not found")
        return False
    
    print("✓ Method '_get_open_access_pdf_url' exists")
    
    # Check unpaywall_email configured
    if not hasattr(fetcher, 'unpaywall_email'):
        print("✗ Attribute 'unpaywall_email' not found")
        return False
    
    print(f"✓ unpaywall_email configured: {fetcher.unpaywall_email}")
    
    # Test with fake DOI (should return None for non-existent DOI)
    print("Testing with non-existent DOI (should handle gracefully)...")
    result = fetcher._get_open_access_pdf_url("10.1234/fake.doi.test.99999")
    
    if result is None:
        print("✓ Handled non-existent DOI correctly (returned None)")
    else:
        print(f"✗ Unexpected result for fake DOI: {result}")
        return False
    
    return True

def test_enhanced_ncbi_fetcher():
    """Test enhanced NCBI fetcher with error handling."""
    print("\n" + "="*60)
    print("TEST 5: Enhanced NCBI Fetcher")
    print("="*60)
    
    config = get_config("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml")
    fetcher = PublicationFetcher(config=config, cache_path=None)
    
    print("Testing NCBI fetcher with invalid accession (should handle gracefully)...")
    result = fetcher._get_publications_from_ncbi("INVALID_ACCESSION_12345")
    
    if isinstance(result, list):
        print(f"✓ Returned list (length: {len(result)})")
        print("✓ Error handling working correctly")
        return True
    else:
        print(f"✗ Unexpected return type: {type(result)}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("PUBLICATION FETCHER IMPROVEMENTS - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Rate Limiting", test_rate_limiting),
        ("Cache Creation", test_cache_creation),
        ("API Statistics", test_api_statistics),
        ("Unpaywall Integration", test_unpaywall_method),
        ("Enhanced NCBI Fetcher", test_enhanced_ncbi_fetcher),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
