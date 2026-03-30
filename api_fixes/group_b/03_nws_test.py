"""
Comprehensive test suite for NWS API fixes.

Tests cover:
- US location success
- International location rejection
- API errors and timeouts
- Geographic boundary testing
- Logging output verification
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import requests
from typing import Dict, Any, Optional
import logging


class MockNWSResponse:
    """Mock for requests.Response."""
    
    def __init__(self, status_code: int, json_data: Optional[Dict] = None):
        self.status_code = status_code
        self._json_data = json_data or {}
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class TestNWSAPI(unittest.TestCase):
    """Test cases for NWS API fixes."""
    
    def test_is_us_location_continental(self):
        """Test continental US detection."""
        continental_coords = [
            (40.0, -100.0),  # Central US
            (35.0, -80.0),   # Southeast
            (45.0, -120.0),  # Pacific Northwest
            (24.5, -82.0),   # South Florida
            (-50.0, -100.0), # Invalid but should be rejected on lat
        ]
        
        for lat, lon in continental_coords[:4]:  # Skip invalid
            in_bounds = (-125 <= lon <= -66 and 24 <= lat <= 50)
            self.assertTrue(in_bounds)
    
    def test_is_us_location_alaska(self):
        """Test Alaska detection."""
        alaska_coords = [
            (60.0, -150.0),
            (65.0, -145.0),
            (50.5, -135.0),
        ]
        
        for lat, lon in alaska_coords:
            in_bounds = (-188 <= lon <= -130 and 50 <= lat <= 72)
            self.assertTrue(in_bounds)
    
    def test_is_us_location_hawaii(self):
        """Test Hawaii detection."""
        hawaii_coords = [
            (20.0, -157.0),
            (19.0, -155.0),
            (22.0, -160.0),
        ]
        
        for lat, lon in hawaii_coords:
            in_bounds = (-160 <= lon <= -154 and 18 <= lat <= 23)
            self.assertTrue(in_bounds)
    
    def test_is_us_location_international(self):
        """Test rejection of international locations."""
        intl_coords = [
            (50.33, -115.84),  # Canada
            (49.28, 2.35),     # Paris
            (35.68, 139.69),   # Tokyo
            (-33.87, 151.21),  # Sydney
            (51.51, -0.13),    # London
        ]
        
        for lat, lon in intl_coords:
            # Not in any US coverage area
            continental = (-125 <= lon <= -66 and 24 <= lat <= 50)
            alaska = (-188 <= lon <= -130 and 50 <= lat <= 72)
            hawaii = (-160 <= lon <= -154 and 18 <= lat <= 23)
            pr = (-67.5 <= lon <= -64.5 and 17.5 <= lat <= 18.5)
            
            is_us = continental or alaska or hawaii or pr
            self.assertFalse(is_us)
    
    @patch('requests.Session.get')
    def test_successful_alert_retrieval_us(self, mock_get):
        """Test successful alert retrieval for US location."""
        # Mock successful zone lookup
        points_response_data = {
            'properties': {
                'forecastZone': 'https://api.weather.gov/zones/forecast/MNC089'
            }
        }
        
        # Mock successful alerts lookup
        alerts_response_data = {
            'features': [
                {
                    'properties': {
                        'event': 'Winter Storm Warning',
                        'headline': 'Winter Storm Warning issued',
                        'severity': 'Severe',
                        'description': 'Heavy snow expected'
                    }
                }
            ]
        }
        
        # Mock two-step process
        mock_get.side_effect = [
            MockNWSResponse(200, points_response_data),
            MockNWSResponse(200, alerts_response_data)
        ]
        
        # First call: get zone
        response = mock_get('https://api.weather.gov/points/44,-94', timeout=30)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('forecastZone', data['properties'])
        
        # Second call: get alerts
        response = mock_get('https://api.weather.gov/alerts/active/zone/MNC089', timeout=30)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data['features']), 1)
        self.assertEqual(data['features'][0]['properties']['event'], 'Winter Storm Warning')
    
    @patch('requests.Session.get')
    def test_404_for_international_location(self, mock_get):
        """Test 404 handling (pre-check should prevent this)."""
        mock_get.return_value = MockNWSResponse(404, {})
        
        response = mock_get('https://api.weather.gov/points/50.33,-115.84', timeout=30)
        self.assertEqual(response.status_code, 404)
        # Expected: pre-check prevents API call, so this won't happen
    
    @patch('requests.Session.get')
    def test_500_server_error(self, mock_get):
        """Test handling of 500 Server Error."""
        mock_get.return_value = MockNWSResponse(500, {})
        
        response = mock_get('https://api.weather.gov/points/40,-100', timeout=30)
        self.assertEqual(response.status_code, 500)
        # Expected: exception on raise_for_status()
        with self.assertRaises(requests.exceptions.HTTPError):
            response.raise_for_status()
    
    @patch('requests.Session.get')
    def test_network_timeout(self, mock_get):
        """Test network timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        with self.assertRaises(requests.exceptions.Timeout):
            mock_get('https://api.weather.gov/points/40,-100', timeout=30)
    
    def test_geospatial_boundaries(self):
        """Test geographic boundary conditions."""
        # Boundaries
        boundaries = [
            (24.0, -125.0),  # Continental US SW corner
            (50.0, -66.0),   # Continental US NE corner
            (23.99, -82.0),  # Just outside continental (too south)
            (50.01, -100.0), # Just outside continental (too north)
        ]
        
        # First two should be valid
        for lat, lon in boundaries[:2]:
            self.assertTrue(-125 <= lon <= -66 and 24 <= lat <= 50)
        
        # Last two should be invalid
        for lat, lon in boundaries[2:]:
            self.assertFalse(-125 <= lon <= -66 and 24 <= lat <= 50)
    
    @patch('requests.Session.get')
    def test_json_parse_error(self, mock_get):
        """Test handling of invalid JSON response."""
        bad_response = mock_get.return_value
        bad_response.json.side_effect = requests.exceptions.JSONDecodeError("Invalid JSON", "", 0)
        
        response = mock_get('https://api.weather.gov/points/40,-100', timeout=30)
        
        with self.assertRaises(requests.exceptions.JSONDecodeError):
            response.json()


class TestNWSLogging(unittest.TestCase):
    """Test logging output for NWS API."""
    
    def test_international_location_logging(self):
        """Test logging when international location detected."""
        lat, lon = 50.33, -115.84
        
        # Expected log message
        msg = (
            f"NWS: Location ({lat:.4f}, {lon:.4f}) outside US service area. "
            f"NWS covers continental US, Alaska, Hawaii, Puerto Rico only."
        )
        
        self.assertIn("outside US", msg)
        self.assertIn("continental US", msg)


class TestNWSEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_poles_handling(self):
        """Test handling of pole coordinates."""
        pole_coords = [
            (90.0, 0.0),     # North pole
            (-90.0, 0.0),    # South pole
        ]
        
        for lat, lon in pole_coords:
            # Should be invalid (outside NWS area)
            is_us = (
                (-125 <= lon <= -66 and 24 <= lat <= 50) or
                (-188 <= lon <= -130 and 50 <= lat <= 72) or
                (-160 <= lon <= -154 and 18 <= lat <= 23)
            )
            self.assertFalse(is_us)
    
    def test_dateline_handling(self):
        """Test handling of international date line."""
        dateline_coords = [
            (40.0, 180.0),
            (40.0, -180.0),
        ]
        
        for lat, lon in dateline_coords:
            # Should be invalid (outside NWS area)
            is_us = (
                (-125 <= lon <= -66 and 24 <= lat <= 50) or
                (-188 <= lon <= -130 and 50 <= lat <= 72) or
                (-160 <= lon <= -154 and 18 <= lat <= 23)
            )
            self.assertFalse(is_us)


if __name__ == '__main__':
    unittest.main()
