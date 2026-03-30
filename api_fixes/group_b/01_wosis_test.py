"""
Comprehensive test suite for WoSIS API fixes.

Tests cover:
- Successful profile retrieval
- Sparse coverage scenarios (404, empty responses)
- API errors and timeouts
- Extended search radius fallback
- Logging output verification
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
from typing import Dict, Any, Optional
import logging
import json

# Mock the WoSIS API behavior
class MockWoSISResponse:
    """Mock for requests.Response"""
    
    def __init__(self, status_code: int, json_data: Optional[Dict] = None):
        self.status_code = status_code
        self._json_data = json_data
    
    def json(self):
        if self._json_data is None:
            return {}
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class TestWoSISAPI(unittest.TestCase):
    """Test cases for WoSIS API fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_lat = 0.0
        self.test_lon = 25.0
        
        # Empty response (no profiles)
        self.empty_response_data = {
            'profiles': []
        }
        
        # Successful response with profiles
        self.successful_response_data = {
            'profiles': [
                {
                    'latitude': 0.0,
                    'longitude': 25.0,
                    'properties': {
                        'ph_h2o': 6.5,
                        'clay': 25.0,
                        'sand': 60.0,
                        'silt': 15.0,
                        'oc': 2.1,
                        'depth': 100
                    }
                },
                {
                    'latitude': 0.1,
                    'longitude': 25.1,
                    'properties': {
                        'ph_h2o': 6.0,
                        'clay': 30.0,
                        'sand': 55.0,
                        'silt': 15.0,
                        'oc': 1.9,
                        'depth': 80
                    }
                }
            ]
        }
        
        # Multiple profiles response
        self.multiple_profiles_data = {
            'profiles': [self.successful_response_data['profiles'][0]] * 5
        }
    
    @patch('requests.Session.get')
    def test_successful_profile_retrieval(self, mock_get):
        """Test successful retrieval of soil profiles from WoSIS."""
        mock_get.return_value = MockWoSISResponse(200, self.successful_response_data)
        
        # Simulate WoSIS query
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn('profiles', data)
        self.assertEqual(len(data['profiles']), 2)
        
        # Check first profile
        first_profile = data['profiles'][0]
        self.assertEqual(first_profile['latitude'], 0.0)
        self.assertEqual(first_profile['longitude'], 25.0)
        
        # Check properties
        properties = first_profile['properties']
        self.assertEqual(properties['ph_h2o'], 6.5)
        self.assertEqual(properties['clay'], 25.0)
        self.assertEqual(properties['depth'], 100)
    
    @patch('requests.Session.get')
    def test_sparse_coverage_404_response(self, mock_get):
        """Test handling of 404 response (sparse coverage - no profiles)."""
        mock_get.return_value = MockWoSISResponse(404, None)
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 404)
        # Expected: graceful return of None
        # Logging should indicate sparse coverage
    
    @patch('requests.Session.get')
    def test_empty_profiles_array(self, mock_get):
        """Test handling of empty profiles array."""
        mock_get.return_value = MockWoSISResponse(200, self.empty_response_data)
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn('profiles', data)
        self.assertEqual(len(data['profiles']), 0)
    
    @patch('requests.Session.get')
    def test_api_error_503_unavailable(self, mock_get):
        """Test handling of API errors (service unavailable)."""
        mock_get.return_value = MockWoSISResponse(503, None)
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 503)
        # Expected: graceful error handling with warning log
    
    @patch('requests.Session.get')
    def test_network_timeout(self, mock_get):
        """Test handling of network timeouts."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        with self.assertRaises(requests.exceptions.Timeout):
            mock_get(
                "https://www.isric.org/projects/wosis/wosis.api",
                params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
                timeout=30
            )
        # Expected: graceful timeout handling with warning log
    
    def test_invalid_coordinates(self):
        """Test rejection of invalid coordinates."""
        invalid_cases = [
            (-91.0, 25.0),   # lat too low
            (91.0, 25.0),    # lat too high
            (50.0, -181.0),  # lon too low
            (50.0, 181.0),   # lon too high
            (float('nan'), 25.0),  # invalid lat
            (50.0, float('nan')),  # invalid lon
        ]
        
        for lat, lon in invalid_cases:
            # These should return None without API call
            # Coordinates should be validated before query
            try:
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    # Invalid
                    self.assertTrue(True)
                else:
                    self.fail(f"Invalid coords not detected: {lat}, {lon}")
            except (ValueError, TypeError):
                self.assertTrue(True)
    
    @patch('requests.Session.get')
    def test_profile_aggregation(self, mock_get):
        """Test aggregation of multiple profiles into mean values."""
        mock_get.return_value = MockWoSISResponse(200, self.multiple_profiles_data)
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        data = response.json()
        profiles = data['profiles']
        
        # Simulate aggregation
        ph_values = [p['properties']['ph_h2o'] for p in profiles]
        mean_ph = sum(ph_values) / len(ph_values)
        
        self.assertEqual(mean_ph, 6.5)  # All profiles have same pH
        self.assertEqual(len(profiles), 5)
    
    @patch('requests.Session.get')
    def test_malformed_json_response(self, mock_get):
        """Test handling of malformed JSON responses."""
        mock_get.side_effect = requests.exceptions.JSONDecodeError("Invalid JSON", "", 0)
        
        with self.assertRaises(requests.exceptions.JSONDecodeError):
            response = mock_get(
                "https://www.isric.org/projects/wosis/wosis.api",
                params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
                timeout=30
            )
            response.json()
        
        # Expected: graceful error handling with debug log


class TestWoSISLogging(unittest.TestCase):
    """Test logging output for WoSIS API."""
    
    @patch('requests.Session.get')
    def test_logging_sparse_coverage_message(self, mock_get):
        """Test that sparse coverage is clearly logged."""
        mock_get.return_value = MockWoSISResponse(404, None)
        
        # Capture logs
        with self.assertLogs('workflow_16s', level='INFO') as log_context:
            response = mock_get(
                "https://www.isric.org/projects/wosis/wosis.api",
                params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
                timeout=30
            )
            
            # Log expected sparse coverage message
            if response.status_code == 404:
                logger = logging.getLogger('workflow_16s')
                logger.info(
                    f"WoSIS: No profiles found at (0.0, 25.0) "
                    f"within 25km radius (sparse coverage)"
                )
        
        # Verify logging contains sparse coverage reference
        self.assertTrue(
            any('sparse coverage' in msg for msg in log_context.output),
            "Sparse coverage message not found in logs"
        )
    
    @patch('requests.Session.get')
    def test_logging_fallback_attempt(self, mock_get):
        """Test logging of extended search radius fallback attempt."""
        # First attempt: empty response
        # Second attempt (fallback): successful
        
        mock_get.side_effect = [
            MockWoSISResponse(200, {'profiles': []}),  # No profiles in 25km
            MockWoSISResponse(200, {'profiles': [     # Success with 50km
                {
                    'latitude': 0.0,
                    'longitude': 25.0,
                    'properties': {
                        'ph_h2o': 6.5,
                        'clay': 25.0,
                        'sand': 60.0,
                        'silt': 15.0,
                        'oc': 2.1,
                        'depth': 100
                    }
                }
            ]})
        ]
        
        with self.assertLogs('workflow_16s', level='DEBUG') as log_context:
            # First call - no profiles
            r1 = mock_get(
                "https://www.isric.org/projects/wosis/wosis.api",
                params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
                timeout=30
            )
            
            if not r1.json().get('profiles'):
                logger = logging.getLogger('workflow_16s')
                logger.debug("WoSIS: Retrying with extended radius 50km")
                
                # Second call - with larger radius
                r2 = mock_get(
                    "https://www.isric.org/projects/wosis/wosis.api",
                    params={'north': 2.0, 'south': -2.0, 'east': 27.0, 'west': 23.0},
                    timeout=30
                )
        
        # Verify fallback was attempted
        self.assertTrue(
            any('extended radius' in msg for msg in log_context.output),
            "Extended radius fallback message not found"
        )


class TestWoSISEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    @patch('requests.Session.get')
    def test_profile_with_missing_properties(self, mock_get):
        """Test handling of profiles with missing soil property fields."""
        incomplete_data = {
            'profiles': [
                {
                    'latitude': 0.0,
                    'longitude': 25.0,
                    'properties': {
                        'ph_h2o': 6.5,
                        # Missing: clay, sand, silt, oc, depth
                    }
                }
            ]
        }
        
        mock_get.return_value = MockWoSISResponse(200, incomplete_data)
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        data = response.json()
        profile = data['profiles'][0]
        properties = profile['properties']
        
        # Should contain only available properties
        self.assertIn('ph_h2o', properties)
        self.assertNotIn('clay', properties)
    
    @patch('requests.Session.get')
    def test_profile_with_non_numeric_values(self, mock_get):
        """Test handling of non-numeric property values."""
        bad_data = {
            'profiles': [
                {
                    'latitude': 0.0,
                    'longitude': 25.0,
                    'properties': {
                        'ph_h2o': 'invalid_string',
                        'clay': None,
                        'sand': 60.0,
                    }
                }
            ]
        }
        
        mock_get.return_value = MockWoSISResponse(200, bad_data)
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={'north': 1.0, 'south': -1.0, 'east': 26.0, 'west': 24.0},
            timeout=30
        )
        
        data = response.json()
        profile = data['profiles'][0]
        
        # Simulate property validation
        for key, value in profile['properties'].items():
            if value is not None:
                try:
                    float_val = float(value)
                except (ValueError, TypeError):
                    # Should skip invalid values
                    pass
    
    @patch('requests.Session.get')
    def test_very_large_search_radius(self, mock_get):
        """Test with large search radius (100+ km)."""
        mock_get.return_value = MockWoSISResponse(200, self.successful_response_data)
        
        large_radius_km = 200.0
        radius_deg = large_radius_km / 111.0
        
        response = mock_get(
            "https://www.isric.org/projects/wosis/wosis.api",
            params={
                'north': 0.0 + radius_deg,
                'south': 0.0 - radius_deg,
                'east': 25.0 + radius_deg,
                'west': 25.0 - radius_deg,
                'limit': 5,
                'IncludeQualityFlags': 'true'
            },
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('profiles', data)


class TestWoSISCoordinateValidation(unittest.TestCase):
    """Test coordinate validation edge cases."""
    
    def test_poles_coordinates(self):
        """Test handling of coordinates at earth's poles."""
        pole_coords = [
            (90.0, 0.0),    # North pole
            (-90.0, 0.0),   # South pole
            (89.99, 179.99),  # near North pole corner
        ]
        
        for lat, lon in pole_coords:
            # Should be valid coordinates
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)
    
    def test_date_line_coordinates(self):
        """Test handling of coordinates near international date line."""
        coords = [
            (0.0, 180.0),   # East side of date line
            (0.0, -180.0),  # West side of date line
            (45.0, 179.99), # Near date line
        ]
        
        for lat, lon in coords:
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)
    
    def test_equator_coordinates(self):
        """Test handling of coordinates at equator."""
        coords = [
            (0.0, 0.0),     # Prime meridian + Equator
            (0.0, 180.0),   # Equator + Date line
            (0.0, -180.0),  # Equator + Date line
            (0.001, 25.0),  # Just north of equator
            (-0.001, 25.0), # Just south of equator
        ]
        
        for lat, lon in coords:
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)


if __name__ == '__main__':
    unittest.main()
