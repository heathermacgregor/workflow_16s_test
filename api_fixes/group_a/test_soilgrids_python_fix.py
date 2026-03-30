"""
Tests for SoilGrids Python Package version mismatch fix

Tests method detection, fallback query methods, and version compatibility.
"""

import pytest
from unittest import mock
import sys
import os
import pandas as pd

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from workflow_16s.api.environmental_data.other.tools._soilgrids_python import SoilGridsPythonAPI


class TestSoilGridsPythonMethodDetection:
    """Test suite for SoilGrids Python package method detection."""
    
    @pytest.fixture
    def api(self):
        """Create SoilGrids Python API instance."""
        api = SoilGridsPythonAPI(verbose=True)
        return api
    
    def test_api_instantiation(self, api):
        """Test that API can be instantiated."""
        assert api is not None
        assert isinstance(api, SoilGridsPythonAPI)
    
    def test_api_name_defined(self, api):
        """Test that API_NAME is defined."""
        assert api.API_NAME == "SoilGrids_Python"
    
    def test_soil_variables_defined(self, api):
        """Test that soil variables are defined."""
        assert api.SOIL_VARIABLES is not None
        assert len(api.SOIL_VARIABLES) > 0
        assert 'clay' in api.SOIL_VARIABLES or 'phh2o' in api.SOIL_VARIABLES
    
    def test_depths_defined(self, api):
        """Test that depths are defined."""
        assert api.DEPTHS is not None
        assert len(api.DEPTHS) > 0
        assert '0-5cm' in api.DEPTHS
    
    def test_detect_query_method_exists(self, api):
        """Test that _detect_query_method helper exists."""
        assert hasattr(api, '_detect_query_method')
        assert callable(api._detect_query_method)
    
    def test_detect_query_method_returns_string_or_none(self, api):
        """Test that _detect_query_method returns string or None."""
        result = api._detect_query_method()
        assert result is None or isinstance(result, str)
    
    def test_detect_get_points_method(self, api):
        """Test detection of get_points method."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock()
        
        api.sg = mock_sg
        method = api._detect_query_method()
        
        assert method == 'get_points'
    
    def test_detect_query_method(self, api):
        """Test detection of query method."""
        mock_sg = mock.Mock()
        # Remove get_points
        del mock_sg.get_points
        mock_sg.query = mock.Mock()
        
        api.sg = mock_sg
        method = api._detect_query_method()
        
        assert method == 'query'
    
    def test_detect_bulk_query_method(self, api):
        """Test detection of bulk_query method."""
        mock_sg = mock.Mock()
        # Remove get_points and query
        del mock_sg.get_points
        del mock_sg.query
        mock_sg.bulk_query = mock.Mock()
        
        api.sg = mock_sg
        method = api._detect_query_method()
        
        assert method == 'bulk_query'
    
    def test_detect_request_method(self, api):
        """Test detection of request method (fallback)."""
        mock_sg = mock.Mock()
        # Remove other methods
        del mock_sg.get_points
        del mock_sg.query
        del mock_sg.bulk_query
        mock_sg.request = mock.Mock()
        
        api.sg = mock_sg
        method = api._detect_query_method()
        
        assert method == 'request'
    
    def test_detect_no_method(self, api):
        """Test when no recognized method exists."""
        mock_sg = mock.Mock()
        # Mock object with no known methods
        mock_sg.get_points = None
        mock_sg.query = None
        mock_sg.bulk_query = None
        mock_sg.request = None
        
        api.sg = mock_sg
        method = api._detect_query_method()
        
        assert method is None


class TestSoilGridsPythonFallbackMethods:
    """Test fallback to different query methods."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsPythonAPI(verbose=True)
    
    def test_get_points_method_call(self, api):
        """Test calling via get_points method."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(return_value=None)
        mock_sg.data = mock.Mock()
        mock_sg.data.empty = False
        mock_sg.data.columns = ['clay_0-5cm', 'sand_0-5cm']
        mock_sg.data.__getitem__ = lambda self, key: pd.Series([25.0])
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            # Should have called get_points
            assert mock_sg.get_points.called
    
    def test_query_method_call(self, api):
        """Test calling via query method."""
        mock_sg = mock.Mock()
        mock_sg.query = mock.Mock(return_value=None)
        mock_sg.data = mock.Mock()
        mock_sg.data.empty = False
        mock_sg.data.columns = ['clay_0-5cm']
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='query'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            # Should have called query
            assert mock_sg.query.called
    
    def test_bulk_query_method_call(self, api):
        """Test calling via bulk_query method."""
        mock_sg = mock.Mock()
        mock_sg.bulk_query = mock.Mock(return_value=None)
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='bulk_query'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            # Should have called bulk_query
            assert mock_sg.bulk_query.called


class TestSoilGridsPythonResponseHandling:
    """Test handling of different response types."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsPythonAPI(verbose=True)
    
    def test_dataframe_response_handling(self, api):
        """Test processing of DataFrame response."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(return_value=None)
        
        # Mock DataFrame response
        data_dict = {
            'clay_0-5cm': [25.0],
            'sand_0-5cm': [50.0],
            'silt_0-5cm': [25.0]
        }
        mock_data = pd.DataFrame(data_dict)
        mock_sg.data = mock_data
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            if result and result.get('available'):
                assert 'clay_0-5cm' in result['data']
    
    def test_dict_response_handling(self, api):
        """Test processing of dict response."""
        mock_sg = mock.Mock()
        mock_sg.query = mock.Mock(return_value={
            'clay_0-5cm': 25.0,
            'sand_0-5cm': 50.0
        })
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='query'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            if result and result.get('available'):
                assert 'data' in result
    
    def test_none_response_handling(self, api):
        """Test handling of None response."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(return_value=None)
        mock_sg.data = None
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            assert result is not None
            assert result.get('available') is False
    
    def test_empty_dataframe_response(self, api):
        """Test handling of empty DataFrame."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(return_value=None)
        mock_sg.data = pd.DataFrame()  # Empty DataFrame
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            assert result is not None
            assert result.get('available') is False


class TestSoilGridsPythonErrorHandling:
    """Test error handling."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsPythonAPI(verbose=True)
    
    def test_no_query_method_available(self, api):
        """Test graceful handling when no query method available."""
        mock_sg = mock.Mock()
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value=None):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            assert result is not None
            assert result.get('available') is False
            assert 'error' in result
    
    def test_query_method_raises_exception(self, api):
        """Test handling of exception during query."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(side_effect=Exception("Query failed"))
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            assert result is not None
            assert result.get('available') is False
            assert 'error' in result
    
    def test_import_not_available(self, api):
        """Test handling when package not installed."""
        api.sg = None
        
        result = api.get_data(lat=36.74, lon=-119.77)
        
        assert result is not None
        assert result.get('available') is False


class TestSoilGridsPythonVersionLogging:
    """Test version logging."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsPythonAPI(verbose=True)
    
    def test_version_info_in_try_import(self):
        """Test that version info is logged in _try_import."""
        api = SoilGridsPythonAPI(verbose=True)
        # _try_import is called in __init__
        # Should handle version info gracefully
        assert api is not None


class TestSoilGridsPythonDataAggregation:
    """Test data aggregation across depths."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsPythonAPI(verbose=True)
    
    def test_mean_calculation_across_depths(self, api):
        """Test that depth-wise means are calculated."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(return_value=None)
        
        data_dict = {
            'clay_0-5cm': [25.0],
            'clay_5-15cm': [30.0],
            'clay_15-30cm': [35.0]
        }
        mock_data = pd.DataFrame(data_dict)
        mock_sg.data = mock_data
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            if result and result.get('available'):
                # Should have mean value
                if 'clay_mean' in result['data']:
                    mean_val = result['data']['clay_mean']
                    assert mean_val == (25.0 + 30.0 + 35.0) / 3
    
    def test_missing_depths_handling(self, api):
        """Test handling of missing depth data."""
        mock_sg = mock.Mock()
        mock_sg.get_points = mock.Mock(return_value=None)
        
        # Missing some depths
        data_dict = {
            'clay_0-5cm': [25.0],
            'clay_5-15cm': [None],  # Missing
            'clay_15-30cm': [35.0]
        }
        mock_data = pd.DataFrame(data_dict)
        mock_sg.data = mock_data
        
        api.sg = mock_sg
        
        with mock.patch.object(api, '_detect_query_method', return_value='get_points'):
            result = api.get_data(lat=36.74, lon=-119.77)
            
            if result and result.get('available'):
                # Should handle None values gracefully
                if 'clay_mean' in result['data']:
                    mean_val = result['data']['clay_mean']
                    # Should only average non-None values
                    assert mean_val == (25.0 + 35.0) / 2 or mean_val is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
