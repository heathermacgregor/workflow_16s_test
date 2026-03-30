"""
Tests for USGS Geology JSON Parse error fix

Tests JSON parsing error handling and empty response handling.
"""

import pytest
from unittest import mock
import json
import requests
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from workflow_16s.api.environmental_data.other.tools._usgs_geology import USGSGeologicUnitsAPI


class TestUSGSGeologyJSONParsing:
    """Test suite for USGS Geology JSON parsing fix."""
    
    @pytest.fixture
    def api(self):
        """Create USGS Geology API instance."""
        return USGSGeologicUnitsAPI(verbose=True)
    
    def test_api_instantiation(self, api):
        """Test that API can be instantiated."""
        assert api is not None
        assert isinstance(api, USGSGeologicUnitsAPI)
    
    def test_api_name_defined(self, api):
        """Test that API_NAME is defined."""
        assert api.API_NAME == "USGS_Geologic_Units"
    
    def test_base_url_defined(self, api):
        """Test that base URL is defined."""
        assert api.BASE_URL == "https://mrdata.usgs.gov/geology/state/point-unit.php"
    
    def test_timeout_defined(self, api):
        """Test that timeout is defined."""
        assert hasattr(api, 'timeout')
        assert api.timeout > 0
    
    def test_valid_geologic_data_parsing(self, api):
        """Test parsing of valid geologic data."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection',
            'features': [{
                'properties': {
                    'unit_name': 'Granite',
                    'age': 'Mesozoic',
                    'type': 'Igneous',
                    'description': 'Granitic intrusion'
                }
            }]
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert result['unit_found'] is True
            assert result['unit_name'] == 'Granite'
    
    def test_empty_response_string(self, api):
        """Test handling of empty response string."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = ""  # Empty response
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=60.19, lon=-125.51)
            
            # Should return empty dict, not crash
            assert result['unit_found'] is False
            assert result['unit_name'] is None
    
    def test_whitespace_only_response(self, api):
        """Test handling of whitespace-only response."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "   \n\t  "  # Whitespace only
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=60.19, lon=-125.51)
            
            # Should return empty dict
            assert result['unit_found'] is False
    
    def test_invalid_json_response(self, api):
        """Test handling of invalid JSON."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "{invalid json"
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=60.19, lon=-125.51)
            
            # Should return empty dict, not crash
            assert result['unit_found'] is False
    
    def test_missing_features_array(self, api):
        """Test handling of response without features array."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection'
            # No 'features' array
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            # Should handle gracefully
            assert result['unit_found'] is False
    
    def test_missing_type_field(self, api):
        """Test handling of response without 'type' field."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({'features': []})
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            # Should handle gracefully
            assert result['unit_found'] is False
    
    def test_none_response_body(self, api):
        """Test handling of None response body."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = None
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert result['unit_found'] is False


class TestUSGSGeologyHTTPErrors:
    """Test HTTP error handling."""
    
    @pytest.fixture
    def api(self):
        return USGSGeologicUnitsAPI(verbose=True)
    
    def test_404_error_outside_us(self, api):
        """Test 404 error for non-US location."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=51.5, lon=0.0)  # London, UK
            
            # Should return empty dict, not crash
            assert result['unit_found'] is False
    
    def test_other_http_errors(self, api):
        """Test other HTTP errors."""
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            # Should return empty dict
            assert result['unit_found'] is False


class TestUSGSGeologyNetworkErrors:
    """Test network error handling."""
    
    @pytest.fixture
    def api(self):
        return USGSGeologicUnitsAPI(verbose=True)
    
    def test_connection_error(self, api):
        """Test handling of connection error."""
        with mock.patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
            
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert result['unit_found'] is False
    
    def test_timeout_error(self, api):
        """Test handling of timeout."""
        with mock.patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
            
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert result['unit_found'] is False


class TestUSGSGeologyRockClassification:
    """Test rock type classification."""
    
    @pytest.fixture
    def api(self):
        return USGSGeologicUnitsAPI(verbose=True)
    
    def test_igneous_classification(self, api):
        """Test classification of igneous rock."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection',
            'features': [{
                'properties': {
                    'unit_name': 'Granite',
                    'age': 'Mesozoic',
                    'type': 'Igneous',
                    'description': 'Granitic intrusion'
                }
            }]
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert 'igneous' in result['rock_types']
    
    def test_sedimentary_classification(self, api):
        """Test classification of sedimentary rock."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection',
            'features': [{
                'properties': {
                    'unit_name': 'Sandstone',
                    'description': 'Clastic sedimentary'
                }
            }]
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert 'sedimentary' in result['rock_types']
    
    def test_metamorphic_classification(self, api):
        """Test classification of metamorphic rock."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection',
            'features': [{
                'properties': {
                    'unit_name': 'Schist',
                    'description': 'Metamorphic'
                }
            }]
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert 'metamorphic' in result['rock_types']


class TestUSGSGeologyDataExtraction:
    """Test data extraction from response."""
    
    @pytest.fixture
    def api(self):
        return USGSGeologicUnitsAPI(verbose=True)
    
    def test_extract_unit_properties(self, api):
        """Test extraction of unit properties."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection',
            'features': [{
                'properties': {
                    'unit_name': 'Granite',
                    'age': 'Cretaceous',
                    'type': 'Igneous',
                    'description': 'Granitic pluton'
                }
            }]
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert result['unit_name'] == 'Granite'
            assert result['unit_age'] == 'Cretaceous'
            assert result['description'] == 'Granitic pluton'
    
    def test_missing_properties(self, api):
        """Test handling of missing properties."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            'type': 'FeatureCollection',
            'features': [{
                'properties': {
                    'unit_name': 'Granite'
                    # Missing other properties
                }
            }]
        })
        mock_response.json.return_value = json.loads(mock_response.text)
        
        with mock.patch('requests.get', return_value=mock_response):
            result = api.get_data(lat=40.0, lon=-100.0)
            
            assert result['unit_found'] is True
            assert result['unit_name'] == 'Granite'
            assert result['unit_age'] == ''  # Empty string for missing property


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
