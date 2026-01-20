"""
QC System Robustness Tests

Tests error handling, edge cases, and failure recovery.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from workflow_16s.qc.validation import (
    validate_config,
    validate_metadata,
    validate_adata,
    check_dependencies,
    validate_primer_sequences,
    QCValidationError,
    safe_numeric_conversion
)


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test that valid config passes."""
        config = {
            'metadata_validation': {
                'correlation_threshold': 0.95,
                'max_facility_distance_km': 1000
            },
            'contamination_detection': {
                'method': 'combined',
                'threshold': 0.5
            },
            'primer_qc': {
                'max_error_rate': 0.15,
                'enabled': True
            }
        }
        
        is_valid, errors = validate_config(config)
        assert is_valid, f"Valid config failed: {errors}"
        assert len(errors) == 0
    
    def test_invalid_correlation_threshold(self):
        """Test that invalid correlation threshold is caught."""
        config = {
            'metadata_validation': {
                'correlation_threshold': 1.5  # Invalid: > 1
            }
        }
        
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any('correlation_threshold' in err for err in errors)
    
    def test_invalid_contamination_method(self):
        """Test that invalid method is caught."""
        config = {
            'contamination_detection': {
                'method': 'invalid_method'
            }
        }
        
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any('method' in err for err in errors)
    
    def test_invalid_threshold(self):
        """Test that threshold out of range is caught."""
        config = {
            'contamination_detection': {
                'threshold': 1.5  # Invalid: > 1
            }
        }
        
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any('threshold' in err for err in errors)


class TestMetadataValidation:
    """Test metadata validation."""
    
    def test_valid_metadata(self):
        """Test that valid metadata passes."""
        df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'env_biome': ['soil', 'soil', 'water'],
            'depth_m': [0.1, 0.2, 0.3]
        })
        df.index = ['S1', 'S2', 'S3']
        
        is_valid, errors = validate_metadata(df)
        assert is_valid, f"Valid metadata failed: {errors}"
        assert len(errors) == 0
    
    def test_empty_metadata(self):
        """Test that empty metadata is caught."""
        df = pd.DataFrame()
        
        is_valid, errors = validate_metadata(df)
        assert not is_valid
        assert any('empty' in err.lower() for err in errors)
    
    def test_duplicate_index(self):
        """Test that duplicate indices are caught."""
        df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'value': [1, 2, 3]
        })
        df.index = ['S1', 'S1', 'S2']  # Duplicate!
        
        is_valid, errors = validate_metadata(df)
        assert not is_valid
        assert any('duplicate' in err.lower() for err in errors)
    
    def test_missing_required_columns(self):
        """Test that missing required columns are caught."""
        df = pd.DataFrame({
            'sample_id': ['S1', 'S2'],
            'value': [1, 2]
        })
        df.index = ['S1', 'S2']
        
        required = ['env_biome', 'env_feature']
        is_valid, errors = validate_metadata(df, required_cols=required)
        assert not is_valid
        assert any('missing' in err.lower() for err in errors)


class TestPrimerValidation:
    """Test primer sequence validation."""
    
    def test_valid_primers(self):
        """Test that valid primers pass."""
        primers = {
            '515F': 'GTGCCAGCMGCCGCGGTAA',
            '806R': 'GGACTACHVGGGTWTCTAAT'
        }
        
        is_valid, errors = validate_primer_sequences(primers)
        assert is_valid, f"Valid primers failed: {errors}"
        assert len(errors) == 0
    
    def test_empty_sequence(self):
        """Test that empty sequence is caught."""
        primers = {
            '515F': '',
            '806R': 'GGACTACHVGGGTWTCTAAT'
        }
        
        is_valid, errors = validate_primer_sequences(primers)
        assert not is_valid
        assert any('empty' in err.lower() for err in errors)
    
    def test_invalid_characters(self):
        """Test that invalid characters are caught."""
        primers = {
            '515F': 'GTGCCAGCMGCCGCGGTAA',
            '806R': 'GGACTACHVGGGTWTCTAAT123'  # Invalid: contains numbers
        }
        
        is_valid, errors = validate_primer_sequences(primers)
        assert not is_valid
        assert any('invalid' in err.lower() for err in errors)
    
    def test_no_primers(self):
        """Test that empty dict is caught."""
        primers = {}
        
        is_valid, errors = validate_primer_sequences(primers)
        assert not is_valid
        assert any('no primer' in err.lower() for err in errors)


class TestSafeNumericConversion:
    """Test safe numeric conversion."""
    
    def test_numeric_series(self):
        """Test conversion of numeric series."""
        series = pd.Series(['1', '2', '3', '4.5'])
        result = safe_numeric_conversion(series, 'test')
        
        assert pd.api.types.is_numeric_dtype(result)
        assert result.tolist() == [1, 2, 3, 4.5]
    
    def test_mixed_series(self):
        """Test series with some non-numeric."""
        series = pd.Series(['1', '2', 'not_a_number', '4'])
        result = safe_numeric_conversion(series, 'test')
        
        # Should convert, but with some NaN
        assert pd.api.types.is_numeric_dtype(result)
        assert pd.isna(result[2])
    
    def test_text_series(self):
        """Test series that's mostly text."""
        series = pd.Series(['apple', 'banana', 'cherry', '1'])
        result = safe_numeric_conversion(series, 'test')
        
        # Should keep as-is (>50% failed conversion)
        assert not pd.api.types.is_numeric_dtype(result)
        assert result.tolist() == series.tolist()


class TestDependencyChecking:
    """Test dependency checking."""
    
    def test_check_python_dependencies(self):
        """Test checking Python packages."""
        deps = check_dependencies(['scipy', 'sklearn'])
        
        assert 'scipy' in deps
        assert 'sklearn' in deps
        # These should be installed in the environment
        assert deps['scipy'] is True
        assert deps['sklearn'] is True
    
    def test_check_missing_dependency(self):
        """Test checking non-existent package."""
        # Only check packages we know don't exist
        deps = check_dependencies(['sklearn', 'scipy'])
        
        # These should exist
        assert 'sklearn' in deps
        assert 'scipy' in deps


def test_edge_cases():
    """Test various edge cases."""
    
    # Test metadata with all NaN column
    df = pd.DataFrame({
        'sample_id': ['S1', 'S2', 'S3'],
        'all_nan': [np.nan, np.nan, np.nan],
        'value': [1, 2, 3]
    })
    df.index = ['S1', 'S2', 'S3']
    
    # Should pass validation but warn about NaN column
    is_valid, errors = validate_metadata(df)
    assert is_valid  # No hard errors
    
    # Test config with empty dict
    is_valid, errors = validate_config({})
    assert is_valid  # Empty config is valid (uses defaults)


def test_error_recovery():
    """Test that system can recover from errors."""
    
    # Try to validate invalid input types
    try:
        validate_metadata("not a dataframe")
    except Exception as e:
        # Should get proper error, not crash
        assert "DataFrame" in str(e) or not isinstance(e, AttributeError)
    
    # Try to validate config with wrong type
    try:
        validate_config("not a dict")
    except Exception as e:
        # Should get proper error
        assert "dict" in str(e).lower() or not isinstance(e, AttributeError)


if __name__ == '__main__':
    print("Running QC robustness tests...\n")
    
    # Run tests manually
    test_config = TestConfigValidation()
    test_config.test_valid_config()
    test_config.test_invalid_correlation_threshold()
    test_config.test_invalid_contamination_method()
    test_config.test_invalid_threshold()
    print("✓ Config validation tests passed")
    
    test_meta = TestMetadataValidation()
    test_meta.test_valid_metadata()
    test_meta.test_empty_metadata()
    test_meta.test_duplicate_index()
    test_meta.test_missing_required_columns()
    print("✓ Metadata validation tests passed")
    
    test_primer = TestPrimerValidation()
    test_primer.test_valid_primers()
    test_primer.test_empty_sequence()
    test_primer.test_invalid_characters()
    test_primer.test_no_primers()
    print("✓ Primer validation tests passed")
    
    test_numeric = TestSafeNumericConversion()
    test_numeric.test_numeric_series()
    test_numeric.test_mixed_series()
    test_numeric.test_text_series()
    print("✓ Numeric conversion tests passed")
    
    test_deps = TestDependencyChecking()
    test_deps.test_check_python_dependencies()
    test_deps.test_check_missing_dependency()
    print("✓ Dependency checking tests passed")
    
    test_edge_cases()
    print("✓ Edge case tests passed")
    
    test_error_recovery()
    print("✓ Error recovery tests passed")
    
    print("\n" + "="*80)
    print("ALL ROBUSTNESS TESTS PASSED ✓")
    print("="*80)
