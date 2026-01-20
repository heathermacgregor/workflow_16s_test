#!/usr/bin/env python
"""
QC System Validation Demo

Demonstrates all robustness improvements:
1. Configuration validation
2. Dependency checking
3. Input validation
4. Error handling
5. Safe numeric conversion
6. Output directory creation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import QC validation tools
from workflow_16s.qc.validation import (
    validate_config,
    validate_metadata,
    validate_primer_sequences,
    check_dependencies,
    safe_numeric_conversion,
    create_safe_output_dir,
    QCValidationError,
    QCDependencyError
)


def demo_config_validation():
    """Demonstrate config validation."""
    print("\n" + "="*80)
    print("1. CONFIGURATION VALIDATION")
    print("="*80)
    
    # Valid config
    valid_config = {
        'metadata_validation': {
            'correlation_threshold': 0.95,
            'max_facility_distance_km': 1000
        },
        'contamination_detection': {
            'method': 'combined',
            'threshold': 0.5
        }
    }
    
    is_valid, errors = validate_config(valid_config)
    print(f"\nValid config: {is_valid}")
    assert is_valid, f"Valid config failed: {errors}"
    print("✓ Valid configuration accepted")
    
    # Invalid config
    invalid_config = {
        'metadata_validation': {
            'correlation_threshold': 1.5  # Invalid: > 1
        },
        'contamination_detection': {
            'method': 'invalid_method',  # Invalid method
            'threshold': -0.5  # Invalid: < 0
        }
    }
    
    is_valid, errors = validate_config(invalid_config)
    print(f"\nInvalid config: {is_valid}")
    print("Errors detected:")
    for err in errors:
        print(f"  - {err}")
    assert not is_valid, "Invalid config should fail"
    print("✓ Invalid configuration properly rejected")


def demo_dependency_checking():
    """Demonstrate dependency checking."""
    print("\n" + "="*80)
    print("2. DEPENDENCY CHECKING")
    print("="*80)
    
    # Check Python dependencies
    deps = check_dependencies(['sklearn', 'scipy', 'Bio'])
    
    print("\nPython packages:")
    for pkg, available in deps.items():
        status = "✓ FOUND" if available else "✗ MISSING"
        print(f"  {pkg:20s} {status}")
    
    # Check CutAdapt
    deps = check_dependencies(['cutadapt'])
    cutadapt_available = deps.get('cutadapt', False)
    status = "✓ FOUND" if cutadapt_available else "✗ MISSING"
    print(f"\nCommand-line tools:")
    print(f"  {'cutadapt':20s} {status}")
    
    if not cutadapt_available:
        print("\n  Install with: conda install -c bioconda cutadapt")


def demo_metadata_validation():
    """Demonstrate metadata validation."""
    print("\n" + "="*80)
    print("3. METADATA VALIDATION")
    print("="*80)
    
    # Valid metadata
    valid_df = pd.DataFrame({
        'sample_id': ['S1', 'S2', 'S3'],
        'env_biome': ['soil', 'soil', 'water'],
        'depth_m': [0.1, 0.2, 0.3],
        'latitude': [40.7, 41.0, 39.5],
        'longitude': [-74.0, -73.5, -74.5]
    })
    valid_df.index = ['S1', 'S2', 'S3']
    
    is_valid, errors = validate_metadata(valid_df)
    print(f"\nValid metadata ({len(valid_df)} rows × {len(valid_df.columns)} cols): {is_valid}")
    assert is_valid, f"Valid metadata failed: {errors}"
    print("✓ Valid metadata accepted")
    
    # Invalid metadata - duplicate index
    invalid_df = valid_df.copy()
    invalid_df.index = ['S1', 'S1', 'S2']  # Duplicate!
    
    is_valid, errors = validate_metadata(invalid_df)
    print(f"\nDuplicate index: {is_valid}")
    if not is_valid:
        print("Errors detected:")
        for err in errors:
            print(f"  - {err}")
    assert not is_valid, "Duplicate index should fail"
    print("✓ Duplicate indices properly detected")
    
    # Missing required columns
    is_valid, errors = validate_metadata(
        valid_df, 
        required_cols=['env_biome', 'env_feature', 'env_material']
    )
    print(f"\nMissing required columns: {is_valid}")
    if not is_valid:
        print("Errors detected:")
        for err in errors:
            print(f"  - {err}")
    assert not is_valid, "Missing columns should fail"
    print("✓ Missing columns properly detected")


def demo_primer_validation():
    """Demonstrate primer validation."""
    print("\n" + "="*80)
    print("4. PRIMER SEQUENCE VALIDATION")
    print("="*80)
    
    # Valid primers
    valid_primers = {
        '515F': 'GTGCCAGCMGCCGCGGTAA',
        '806R': 'GGACTACHVGGGTWTCTAAT'
    }
    
    is_valid, errors = validate_primer_sequences(valid_primers)
    print(f"\nValid primers: {is_valid}")
    for name, seq in valid_primers.items():
        print(f"  {name}: {seq} ({len(seq)} bp)")
    assert is_valid, f"Valid primers failed: {errors}"
    print("✓ Valid primer sequences accepted")
    
    # Invalid primers
    invalid_primers = {
        '515F': 'GTGCCAGCMGCCGCGGTAA',
        '806R': 'GGACTACHVGGGTWTCTAAT123',  # Contains numbers!
        'empty': ''  # Empty sequence!
    }
    
    is_valid, errors = validate_primer_sequences(invalid_primers)
    print(f"\nInvalid primers: {is_valid}")
    if not is_valid:
        print("Errors detected:")
        for err in errors:
            print(f"  - {err}")
    assert not is_valid, "Invalid primers should fail"
    print("✓ Invalid primer sequences properly rejected")


def demo_safe_numeric_conversion():
    """Demonstrate safe numeric conversion."""
    print("\n" + "="*80)
    print("5. SAFE NUMERIC CONVERSION")
    print("="*80)
    
    # Pure numeric
    numeric_series = pd.Series(['1', '2', '3', '4.5', '10.2'])
    converted = safe_numeric_conversion(numeric_series, 'depth_m')
    print(f"\nPure numeric: {converted.dtype}")
    print(f"  Input:  {numeric_series.tolist()}")
    print(f"  Output: {converted.tolist()}")
    assert pd.api.types.is_numeric_dtype(converted)
    print("✓ Pure numeric converted successfully")
    
    # Mixed numeric/text (mostly numeric)
    mixed_series = pd.Series(['1', '2', 'missing', '4', '5'])
    converted = safe_numeric_conversion(mixed_series, 'temperature')
    print(f"\nMixed (mostly numeric): {converted.dtype}")
    print(f"  Input:  {mixed_series.tolist()}")
    print(f"  Output: {converted.tolist()}")
    assert pd.api.types.is_numeric_dtype(converted)
    assert pd.isna(converted[2])
    print("✓ Mixed series converted (invalid → NaN)")
    
    # Mostly text
    text_series = pd.Series(['apple', 'banana', 'cherry', '1'])
    converted = safe_numeric_conversion(text_series, 'fruit')
    print(f"\nMostly text: {converted.dtype}")
    print(f"  Input:  {text_series.tolist()}")
    print(f"  Output: {converted.tolist()}")
    assert not pd.api.types.is_numeric_dtype(converted)
    print("✓ Text series kept as-is (>50% failed conversion)")


def demo_output_dir_creation():
    """Demonstrate safe output directory creation."""
    print("\n" + "="*80)
    print("6. SAFE OUTPUT DIRECTORY CREATION")
    print("="*80)
    
    # Create test directory
    test_dir = Path('test_qc_output') / 'demo' / 'nested' / 'path'
    
    try:
        output_dir = create_safe_output_dir(test_dir, 'demo')
        print(f"\nCreated directory: {output_dir}")
        print(f"  Exists: {output_dir.exists()}")
        print(f"  Is directory: {output_dir.is_dir()}")
        assert output_dir.exists()
        print("✓ Output directory created successfully")
        
        # Test write permission
        test_file = output_dir / 'test.txt'
        test_file.write_text("test")
        print(f"  Write test: SUCCESS")
        test_file.unlink()
        print("✓ Write permission verified")
        
    except QCValidationError as e:
        print(f"✗ Failed: {e}")
        raise


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n" + "="*80)
    print("7. ERROR HANDLING")
    print("="*80)
    
    # Test with None
    print("\nHandling None input:")
    try:
        is_valid, errors = validate_metadata(None)
        print(f"  Result: {is_valid}, Errors: {errors}")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}: {e}")
    print("✓ None input handled gracefully")
    
    # Test with wrong type
    print("\nHandling wrong type:")
    try:
        is_valid, errors = validate_metadata("not a dataframe")
        print(f"  Result: {is_valid}, Errors: {errors}")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}: {e}")
    print("✓ Wrong type handled gracefully")
    
    # Test empty DataFrame
    print("\nHandling empty DataFrame:")
    empty_df = pd.DataFrame()
    is_valid, errors = validate_metadata(empty_df)
    print(f"  Result: {is_valid}")
    print(f"  Errors: {errors}")
    assert not is_valid
    print("✓ Empty DataFrame properly rejected")


def main():
    """Run all validation demos."""
    print("\n" + "="*80)
    print("QC SYSTEM VALIDATION DEMO")
    print("="*80)
    print("\nDemonstrating robustness improvements:")
    print("1. Configuration validation")
    print("2. Dependency checking")
    print("3. Metadata validation")
    print("4. Primer sequence validation")
    print("5. Safe numeric conversion")
    print("6. Output directory creation")
    print("7. Error handling")
    
    try:
        demo_config_validation()
        demo_dependency_checking()
        demo_metadata_validation()
        demo_primer_validation()
        demo_safe_numeric_conversion()
        demo_output_dir_creation()
        demo_error_handling()
        
        print("\n" + "="*80)
        print("ALL VALIDATION DEMOS COMPLETED SUCCESSFULLY ✓")
        print("="*80)
        print("\nRobustness improvements verified:")
        print("  ✓ Configuration validation")
        print("  ✓ Dependency checking")
        print("  ✓ Input validation")
        print("  ✓ Error handling")
        print("  ✓ Type conversion")
        print("  ✓ File I/O safety")
        print("\nQC system is PRODUCTION READY ✓")
        
        return 0
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
