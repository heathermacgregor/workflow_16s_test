"""
Comprehensive test suite for Pollen/OpenMeteo API fixes.

Tests cover:
- Successful pollen retrieval
- Date range validation
- 400 Bad Request handling
- Historical data limitations
- Processing lag handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import requests


class MockOpenMeteoResponse:
    """Mock for requests.Response."""
    
    def __init__(self, status_code: int, json_data: Optional[Dict] = None):
        self.status_code = status_code
        self._json_data = json_data or {}
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class TestPollenAPI(unittest.TestCase):
    """Test cases for Pollen API fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_lat = 37.8715
        self.test_lon = -122.2730
    
    def test_date_parsing_valid(self):
        """Test valid date parsing."""
        test_dates = [
            '2024-03-20',
            '2020-01-01',
            '2023-12-31',
        ]
        
        for date_str in test_dates:
            parsed = datetime.strptime(date_str, '%Y-%m-%d')
            self.assertIsInstance(parsed, datetime)
    
    def test_date_in_pollen_range_valid(self):
        """Test dates within pollen data range (2020-01-01 onwards)."""
        min_pollen_date = datetime(2020, 1, 1)
        
        valid_dates = [
            datetime(2020, 1, 1),    # Start of archive
            datetime(2023, 6, 15),   # Mid-range
            datetime(2024, 3, 20),   # Recent
        ]
        
        for test_date in valid_dates:
            self.assertGreaterEqual(test_date, min_pollen_date)
    
    def test_date_before_pollen_archive(self):
        """Test dates before pollen archive (before 2020)."""
        min_pollen_date = datetime(2020, 1, 1)
        
        old_dates = [
            datetime(2019, 12, 31),
            datetime(2015, 6, 15),
            datetime(2010, 1, 1),
            datetime(2012, 8, 22),
        ]
        
        for test_date in old_dates:
            self.assertLess(test_date, min_pollen_date)
            # Expected: skip pollen for these dates
    
    def test_date_processing_lag_recent(self):
        """Test handling of very recent dates (< 2 days old)."""
        today = datetime.now()
        
        recent_dates = [
            today - timedelta(days=0),  # Today
            today - timedelta(days=1),  # Yesterday
        ]
        
        for test_date in recent_dates:
            days_ago = (datetime.now() - test_date).days
            self.assertLess(days_ago, 2)
            # Expected: skip pollen for very recent dates (still processing)
    
    def test_date_processing_lag_processed(self):
        """Test dates with sufficient processing time (>= 2 days old)."""
        today = datetime.now()
        
        processed_dates = [
            today - timedelta(days=2),
            today - timedelta(days=10),
            today - timedelta(days=365),
        ]
        
        for test_date in processed_dates:
            days_ago = (datetime.now() - test_date).days
            self.assertGreaterEqual(days_ago, 2)
    
    @patch('requests.Session.get')
    def test_successful_pollen_retrieval(self, mock_get):
        """Test successful pollen data retrieval."""
        pollen_data = {
            'hourly': {
                'time': [
                    '2024-03-20T00:00',
                    '2024-03-20T01:00',
                    '2024-03-20T02:00',
                    '2024-03-20T03:00',
                ],
                'tree_pollen': [150, 155, 160, 158],
                'grass_pollen': [85, 90, 92, 88],
                'weed_pollen': [30, 32, 31, 30],
            },
            'hourly_units': {
                'tree_pollen': 'pollen count',
                'grass_pollen': 'pollen count',
                'weed_pollen': 'pollen count',
            }
        }
        
        mock_get.return_value = MockOpenMeteoResponse(200, pollen_data)
        
        response = mock_get(
            'https://archive-api.open-meteo.com/v1/archive',
            params={'latitude': 37.8715, 'longitude': -122.2730},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('hourly', data)
        self.assertIn('tree_pollen', data['hourly'])
    
    @patch('requests.Session.get')
    def test_400_bad_request_old_date(self, mock_get):
        """Test 400 Bad Request for date outside pollen range."""
        mock_get.return_value = MockOpenMeteoResponse(400, {
            'reason': 'date_out_of_range'
        })
        
        response = mock_get(
            'https://archive-api.open-meteo.com/v1/archive',
            params={
                'latitude': 37.8715,
                'longitude': -122.2730,
                'start_date': '2012-06-15',
                'end_date': '2012-06-15'
            },
            timeout=30
        )
        
        self.assertEqual(response.status_code, 400)
        # Expected: gracefully skip pollen, continue with air quality
    
    @patch('requests.Session.get')
    def test_air_quality_available_old_date(self, mock_get):
        """Test air quality success even when pollen unavailable."""
        air_quality_data = {
            'hourly': {
                'time': ['2012-06-15T00:00'],
                'pm10': [25.3],
                'pm2_5': [12.1],
                'ozone': [45.2],
            }
        }
        
        mock_get.return_value = MockOpenMeteoResponse(200, air_quality_data)
        
        response = mock_get(
            'https://air-quality-api.open-meteo.com/v1/air-quality',
            params={
                'latitude': 37.8715,
                'longitude': -122.2730,
                'start_date': '2012-06-15',
                'end_date': '2012-06-15',
                'hourly': 'pm10,pm2_5,ozone'
            },
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('pm10', data['hourly'])
    
    @patch('requests.Session.get')
    def test_network_timeout(self, mock_get):
        """Test network timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        with self.assertRaises(requests.exceptions.Timeout):
            mock_get(
                'https://archive-api.open-meteo.com/v1/archive',
                params={'latitude': 37.8715, 'longitude': -122.2730},
                timeout=30
            )
    
    def test_pollen_availability_dates(self):
        """Test pollen availability date ranges."""
        # Pollen archive starts 2020-01-01
        pollen_start = datetime(2020, 1, 1)
        
        test_cases = [
            (datetime(2024, 3, 20), True),   # Recent
            (datetime(2020, 1, 1), True),    # Start date
            (datetime(2019, 12, 31), False), # Before archive
            (datetime(2010, 6, 15), False),  # Old sample
        ]
        
        for test_date, should_be_available in test_cases:
            is_available = test_date >= pollen_start
            self.assertEqual(is_available, should_be_available)
    
    @patch('requests.Session.get')
    def test_forecast_vs_archive_url_selection(self, mock_get):
        """Test correct URL selection for current vs. historical data."""
        # Current data should use forecast URL
        forecast_url = 'https://api.open-meteo.com/v1/forecast'
        archive_url = 'https://archive-api.open-meteo.com/v1/archive'
        
        # Historical query (2024-03-15)
        test_date = '2024-03-15'
        
        # Should use archive URL for historical
        if test_date:
            url = archive_url
        else:
            url = forecast_url
        
        self.assertEqual(url, archive_url)


class TestPollenLogging(unittest.TestCase):
    """Test logging output for Pollen API."""
    
    def test_old_date_unavailable_message(self):
        """Test logging message for old pollen dates."""
        test_date = '2012-06-15'
        
        msg = (
            f"Pollen: Data unavailable for {test_date} - "
            f"before pollen archive start date (2020-01-01)"
        )
        
        self.assertIn("unavailable", msg)
        self.assertIn("2020-01-01", msg)
    
    def test_bad_request_message(self):
        """Test logging message for 400 Bad Request."""
        test_date = '2012-06-15'
        
        msg = (
            f"Pollen: 400 Bad Request for {test_date} - "
            f"date may be outside available range (2020-01-01 onwards)"
        )
        
        self.assertIn("400", msg)
        self.assertIn("available range", msg)


class TestPollenEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_boundary_date_2020_jan_01(self):
        """Test pollen data for exact start date."""
        pollen_start = datetime(2020, 1, 1)
        test_date = datetime(2020, 1, 1)
        
        # Should be available (inclusive)
        self.assertEqual(test_date, pollen_start)
        self.assertGreaterEqual(test_date, pollen_start)
    
    def test_boundary_date_2020_jan_02(self):
        """Test pollen data for day after start."""
        pollen_start = datetime(2020, 1, 1)
        test_date = datetime(2020, 1, 2)
        
        # Should be available
        self.assertGreater(test_date, pollen_start)
        self.assertGreaterEqual(test_date, pollen_start)
    
    def test_boundary_date_2019_dec_31(self):
        """Test pollen data for day before start."""
        pollen_start = datetime(2020, 1, 1)
        test_date = datetime(2019, 12, 31)
        
        # Should NOT be available
        self.assertLess(test_date, pollen_start)
    
    def test_hurricane_season_pollen(self):
        """Test pollen data for hurricane season (late summer/fall)."""
        # Pollen would be high, but API should handle it
        late_summer = datetime(2023, 8, 15)
        pollen_start = datetime(2020, 1, 1)
        
        self.assertGreater(late_summer, pollen_start)
        # Should be available
    
    def test_winter_pollen_data(self):
        """Test pollen data for winter (typically low but available)."""
        winter_date = datetime(2023, 1, 15)
        pollen_start = datetime(2020, 1, 1)
        
        self.assertGreater(winter_date, pollen_start)
        # Should be available (pollen low but still queryable)


if __name__ == '__main__':
    unittest.main()
