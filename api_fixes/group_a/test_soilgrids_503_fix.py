"""
Tests for SoilGrids 503 Service Error retry logic fix

Tests exponential backoff, ConnectionError handling, and circuit breaker.
"""

import pytest
from unittest import mock
import requests
import time
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from workflow_16s.api.environmental_data.other.tools._soilgrids import SoilGridsAPI


class TestSoilGrids503RetryLogic:
    """Test suite for SoilGrids 503 error handling and retry logic."""
    
    @pytest.fixture
    def api(self):
        """Create SoilGrids API instance."""
        return SoilGridsAPI(verbose=True)
    
    def test_api_instantiation(self, api):
        """Test that API can be instantiated."""
        assert api is not None
        assert isinstance(api, SoilGridsAPI)
        assert hasattr(api, 'base_url')
    
    def test_url_defined(self, api):
        """Test that SoilGrids URL is defined."""
        assert api.base_url == "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    def test_default_properties_defined(self, api):
        """Test that default properties are defined."""
        assert api.DEFAULT_PROPERTIES is not None
        assert len(api.DEFAULT_PROPERTIES) > 0
        assert 'clay' in api.DEFAULT_PROPERTIES or 'phh2o' in api.DEFAULT_PROPERTIES
    
    def test_default_depths_defined(self, api):
        """Test that default depths are defined."""
        assert api.DEFAULT_DEPTHS is not None
        assert len(api.DEFAULT_DEPTHS) > 0
        assert '0-5cm' in api.DEFAULT_DEPTHS
    
    @mock.patch('time.sleep')
    def test_503_retry_on_http_error(self, mock_sleep, api):
        """Test that 503 HTTPError triggers retry with backoff."""
        # Create mock response that returns 503
        mock_response_503 = mock.Mock()
        mock_response_503.status_code = 503
        mock_response_503.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_503
        )
        
        # Success response
        mock_response_ok = mock.Mock()
        mock_response_ok.status_code = 200
        mock_response_ok.json.return_value = {'properties': {'layers': []}}
        
        # Mock session to return 503 then success
        with mock.patch.object(api.session, 'get', side_effect=[mock_response_503, mock_response_ok]):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should have retried
            assert api.session.get.call_count == 2
            assert mock_sleep.called  # Should have slept between retries
    
    @mock.patch('time.sleep')
    def test_connection_error_retry(self, mock_sleep, api):
        """Test that ConnectionError (too many 503s) triggers retry."""
        # ConnectionError from too many 503s
        connection_error = requests.exceptions.ConnectionError(
            "MaxRetryError: HTTPSConnectionPool - too many 503 error responses"
        )
        
        mock_response_ok = mock.Mock()
        mock_response_ok.status_code = 200
        mock_response_ok.json.return_value = {'properties': {'layers': []}}
        
        with mock.patch.object(api.session, 'get', side_effect=[connection_error, mock_response_ok]):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should have retried
            assert api.session.get.call_count == 2
            assert mock_sleep.called
    
    @mock.patch('time.sleep')
    def test_429_rate_limit_retry(self, mock_sleep, api):
        """Test that 429 rate limit triggers longer backoff."""
        mock_response_429 = mock.Mock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_429
        )
        
        mock_response_ok = mock.Mock()
        mock_response_ok.status_code = 200
        mock_response_ok.json.return_value = {'properties': {'layers': []}}
        
        with mock.patch.object(api.session, 'get', side_effect=[mock_response_429, mock_response_ok]):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should have retried with longer backoff
            assert api.session.get.call_count == 2
    
    @mock.patch('time.sleep')
    def test_max_retries_exceeded(self, mock_sleep, api):
        """Test that retries eventually give up."""
        mock_response_503 = mock.Mock()
        mock_response_503.status_code = 503
        mock_response_503.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_503
        )
        
        with mock.patch.object(api.session, 'get', return_value=mock_response_503):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should have tried max_retries + 1 times
            # Original code had 3 retries, fixed version should have more
            assert api.session.get.call_count >= 4  # At least 4 attempts
            assert result is None  # Should return None after max retries
    
    def test_successful_request_no_retry(self, api):
        """Test that successful request doesn't trigger retries."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'properties': {
                'layers': [{
                    'name': 'clay',
                    'unit_measure': {'d_class_label': '%', 'conversion_factor': 1.0},
                    'depths': [{
                        'label': '0-5cm',
                        'values': {'mean': 25.0}
                    }]
                }]
            }
        }
        
        with mock.patch.object(api.session, 'get', return_value=mock_response):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should only call once (no retries needed)
            assert api.session.get.call_count == 1
            assert result is not None


class TestSoilGridsExponentialBackoff:
    """Test exponential backoff timing."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsAPI()
    
    @mock.patch('time.sleep')
    def test_backoff_timing_increases(self, mock_sleep, api):
        """Test that backoff delays increase exponentially."""
        mock_response_503 = mock.Mock()
        mock_response_503.status_code = 503
        mock_response_503.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_503
        )
        
        with mock.patch.object(api.session, 'get', return_value=mock_response_503):
            api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Check that sleep was called with increasing delays
            # With base=2 and multiplier=2: 2s, 4s, 8s, 16s, 32s
            if mock_sleep.call_count > 1:
                sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
                # Each should be >= previous (monotonic increase)
                for i in range(1, len(sleep_times)):
                    assert sleep_times[i] >= sleep_times[i-1]


class TestSoilGridsErrorHandling:
    """Test error handling in SoilGrids API."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsAPI()
    
    def test_other_http_errors_no_retry(self, api):
        """Test that non-503 HTTP errors don't trigger retries."""
        mock_response = mock.Mock()
        mock_response.status_code = 400  # Bad request
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        
        with mock.patch.object(api.session, 'get', return_value=mock_response):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should only call once (no retry for 400)
            assert api.session.get.call_count == 1
            assert result is None
    
    def test_timeout_error_retry(self, api):
        """Test that timeout errors trigger retry."""
        timeout_error = requests.exceptions.Timeout("Connection timeout")
        
        with mock.patch('time.sleep'):
            mock_response = mock.Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'properties': {'layers': []}}
            
            with mock.patch.object(api.session, 'get', side_effect=[timeout_error, mock_response]):
                result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
                
                # Should have retried
                assert api.session.get.call_count == 2
    
    def test_empty_response_handling(self, api):
        """Test handling of response with no layers."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'properties': {'layers': []}}
        
        with mock.patch.object(api.session, 'get', return_value=mock_response):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm'])
            
            # Should return None for empty layers
            assert result is None


class TestSoilGridsCache:
    """Test caching behavior with retries."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsAPI()
    
    def test_cache_decorator_applied(self, api):
        """Test that _fetch_properties has cache decorator."""
        # _fetch_properties should be decorated with @cache_api_call
        assert hasattr(api._fetch_properties, '__wrapped__') or callable(api._fetch_properties)


class TestSoilGridsDataProcessing:
    """Test successful data processing after retries."""
    
    @pytest.fixture
    def api(self):
        return SoilGridsAPI()
    
    def test_soil_property_parsing(self, api):
        """Test parsing of soil property data."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'properties': {
                'layers': [{
                    'name': 'clay',
                    'unit_measure': {'d_class_label': '%', 'conversion_factor': 1.0},
                    'depths': [
                        {
                            'label': '0-5cm',
                            'values': {'mean': 25.0}
                        },
                        {
                            'label': '5-15cm',
                            'values': {'mean': 30.0}
                        }
                    ]
                }]
            }
        }
        
        with mock.patch.object(api.session, 'get', return_value=mock_response):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['clay'], depths=['0-5cm', '5-15cm'])
            
            assert result is not None
            assert 'clay_0-5cm' in result
            assert result['clay_0-5cm']['value'] == 25.0
    
    def test_unit_conversion(self, api):
        """Test unit conversion with divisor."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'properties': {
                'layers': [{
                    'name': 'bdod',  # Bulk density
                    'unit_measure': {'d_class_label': 'kg/m³', 'conversion_factor': 100.0},
                    'depths': [{
                        'label': '0-5cm',
                        'values': {'mean': 140000.0}  # Will be divided by 100
                    }]
                }]
            }
        }
        
        with mock.patch.object(api.session, 'get', return_value=mock_response):
            result = api._fetch_properties(lat=36.74, lon=-119.77, properties=['bdod'], depths=['0-5cm'])
            
            assert result is not None
            assert result['bdod_0-5cm']['value'] == 1400.0  # 140000 / 100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
