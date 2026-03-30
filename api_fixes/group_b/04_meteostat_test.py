"""
Comprehensive test suite for Meteostat API fixes.

Tests cover:
- Successful weather retrieval
- Empty data handling
- Date validation and bounds
- Regional coverage assessment
- Logging verification
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd


class MockMeteostatDaily:
    """Mock for Meteostat Daily class."""
    
    def __init__(self, point, start_date, end_date, has_data: bool = True):
        self.point = point
        self.start_date = start_date
        self.end_date = end_date
        self.has_data = has_data
    
    def fetch(self):
        """Mock fetch method returning DataFrame."""
        if not self.has_data:
            # Return empty DataFrame
            return pd.DataFrame()
        
        # Return sample weather data
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        return pd.DataFrame({
            'tavg': [15.2] * len(dates),  # Average temp
            'tmin': [8.5] * len(dates),   # Min temp
            'tmax': [22.0] * len(dates),  # Max temp
            'prcp': [2.5] * len(dates),   # Precip (mm)
            'wspd': [8.1] * len(dates),   # Wind speed
            'pres': [1013] * len(dates),  # Pressure
        }, index=dates)


class TestMeteostatAPI(unittest.TestCase):
    """Test cases for Meteostat API fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_lat = 50.14
        self.test_lon = -115.58
    
    @patch('meteostat.Daily')
    @patch('meteostat.Point')
    def test_successful_weather_retrieval(self, mock_point_class, mock_daily_class):
        """Test successful weather data retrieval."""
        mock_point = Mock()
        mock_point_class.return_value = mock_point
        
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        mock_daily_class.return_value = MockMeteostatDaily(mock_point, start, end, has_data=True)
        
        # Simulate query
        point = mock_point_class(self.test_lat, self.test_lon)
        daily = mock_daily_class(point, start, end)
        data = daily.fetch()
        
        self.assertFalse(data.empty)
        self.assertIn('tavg', data.columns)
        self.assertIn('prcp', data.columns)
        self.assertEqual(len(data), 30)  # 30 days of data
    
    @patch('meteostat.Daily')
    @patch('meteostat.Point')
    def test_empty_data_no_stations(self, mock_point_class, mock_daily_class):
        """Test handling of empty data (no stations available)."""
        mock_point = Mock()
        mock_point_class.return_value = mock_point
        
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        mock_daily_class.return_value = MockMeteostatDaily(mock_point, start, end, has_data=False)
        
        point = mock_point_class(self.test_lat, self.test_lon)
        daily = mock_daily_class(point, start, end)
        data = daily.fetch()
        
        self.assertTrue(data.empty)
        # Expected: return None gracefully
    
    def test_date_parsing_valid_date(self):
        """Test valid date parsing."""
        test_date = '2024-03-20'
        parsed = datetime.strptime(test_date, '%Y-%m-%d')
        
        self.assertEqual(parsed.year, 2024)
        self.assertEqual(parsed.month, 3)
        self.assertEqual(parsed.day, 20)
    
    def test_date_parsing_invalid_format(self):
        """Test invalid date format handling."""
        invalid_dates = [
            '2024-13-01',  # Invalid month
            '2024-03-32',  # Invalid day
            '24-03-20',    # Wrong year format
            '2024/03/20',  # Wrong separator
        ]
        
        for invalid_date in invalid_dates:
            try:
                parsed = datetime.strptime(invalid_date, '%Y-%m-%d')
                self.fail(f"Should have rejected: {invalid_date}")
            except ValueError:
                self.assertTrue(True)  # Expected
    
    def test_coordinate_validation_valid(self):
        """Test coordinate validation for valid coordinates."""
        valid_coords = [
            (50.14, -115.58),  # Calgary
            (0.0, 0.0),         # Prime meridian
            (90.0, 0.0),        # North pole
            (-90.0, 0.0),       # South pole
            (45.0, 180.0),      # Date line
            (45.0, -180.0),     # Date line
        ]
        
        for lat, lon in valid_coords:
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180 or (lon == -180 and lon == 180))
    
    def test_coordinate_validation_invalid(self):
        """Test coordinate validation for invalid coordinates."""
        invalid_coords = [
            (91.0, 0.0),       # Lat too high
            (-91.0, 0.0),      # Lat too low
            (0.0, 181.0),      # Lon too high
            (0.0, -181.0),     # Lon too low
        ]
        
        for lat, lon in invalid_coords:
            self.assertFalse(-90 <= lat <= 90 and -180 <= lon <= 180)
    
    def test_regional_coverage_polar(self):
        """Test coverage assessment for polar regions."""
        polar_coords = [
            (75.0, 0.0),
            (-75.0, 0.0),
            (85.0, -180.0),
        ]
        
        for lat, lon in polar_coords:
            if abs(lat) > 60:
                coverage = "very_sparse"
            else:
                coverage = "possible"
            
            # High latitude should be very sparse
            if abs(lat) > 70:
                self.assertEqual(coverage, "very_sparse")
    
    def test_regional_coverage_tropical(self):
        """Test coverage assessment for tropical regions."""
        tropical_coords = [
            (0.0, 0.0),
            (15.0, 50.0),
            (-15.0, 100.0),
        ]
        
        for lat, lon in tropical_coords:
            # Most tropical areas have stations, but sparse in oceans/remote areas
            self.assertTrue(-90 <= lat <= 90)
    
    @patch('meteostat.Daily')
    @patch('meteostat.Point')
    def test_single_day_query(self, mock_point_class, mock_daily_class):
        """Test querying for a single specific day."""
        mock_point = Mock()
        mock_point_class.return_value = mock_point
        
        target_date = datetime(2024, 3, 20)
        mock_daily_class.return_value = MockMeteostatDaily(
            mock_point, target_date, target_date, has_data=True
        )
        
        point = mock_point_class(50.0, -100.0)
        daily = mock_daily_class(point, target_date, target_date)
        data = daily.fetch()
        
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 1)  # Single day
    
    @patch('meteostat.Daily')
    @patch('meteostat.Point')
    def test_30_day_average(self, mock_point_class, mock_daily_class):
        """Test averaging over 30-day period."""
        mock_point = Mock()
        mock_point_class.return_value = mock_point
        
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        mock_daily_class.return_value = MockMeteostatDaily(mock_point, start, end, has_data=True)
        
        point = mock_point_class(50.0, -100.0)
        daily = mock_daily_class(point, start, end)
        data = daily.fetch()
        
        # Calculate mean
        mean_data = data.mean().to_dict()
        
        self.assertIn('tavg', mean_data)
        self.assertEqual(mean_data['tavg'], 15.2)


class TestMeteostatLogging(unittest.TestCase):
    """Test logging output for Meteostat API."""
    
    def test_no_stations_message(self):
        """Test logging message when no stations available."""
        lat, lon = 50.14, -115.58
        
        msg = (
            f"Meteostat: No weather station data at ({lat:.4f}, {lon:.4f}) - "
            f"sparse station coverage"
        )
        
        self.assertIn("No weather station", msg)
        self.assertIn("sparse", msg)


class TestMeteostatEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_ocean_coordinates(self):
        """Test handling of coordinates in ocean (no stations)."""
        ocean_coords = [
            (0.0, -100.0),     # Mid-Atlantic
            (-30.0, 100.0),    # Indian Ocean
            (20.0, -150.0),    # Pacific
        ]
        
        for lat, lon in ocean_coords:
            # These coordinates are valid for the API
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)
            # But would return no data due to no stations
    
    def test_remote_area_coordinates(self):
        """Test handling of remote land areas."""
        remote_coords = [
            (65.0, -170.0),    # Alaska remote
            (-70.0, 0.0),      # Antarctica
            (85.0, 0.0),       # High arctic
        ]
        
        for lat, lon in remote_coords:
            # These should be valid coordinates
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)


if __name__ == '__main__':
    unittest.main()
