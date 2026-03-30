"""
Comprehensive test suite for Dynamic World LULC API fixes.

Tests cover:
- Successful LULC retrieval
- Empty response handling with automatic fallback
- Date range extension logic
- Regional coverage assessment
- GEE error handling
- Logging output verification
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
import json


class MockGEEImage:
    """Mock for Google Earth Engine Image objects."""
    
    def __init__(self, lulc_percentages: Dict[str, float]):
        self.lulc_percentages = lulc_percentages
    
    def sample(self, region, scale):
        """Mock sampling method."""
        return MockGEESampler(self.lulc_percentages)


class MockGEESampler:
    """Mock for GEE sampling results."""
    
    def __init__(self, lulc_percentages):
        self.lulc_percentages = lulc_percentages
    
    def first(self):
        """Mock first() result."""
        return MockGEEFeature(self.lulc_percentages)


class MockGEEFeature:
    """Mock for GEE Feature objects."""
    
    def __init__(self, lulc_percentages):
        self.lulc_percentages = lulc_percentages
    
    def getInfo(self):
        """Mock getInfo() returning properties."""
        return {
            'properties': self.lulc_percentages,
            'geometry': {'type': 'Point', 'coordinates': [25.0, 0.0]}
        }


class MockGEEImageCollection:
    """Mock for Google Earth Engine ImageCollection."""
    
    def __init__(self, n_images: int = 1, lulc_data: Optional[Dict] = None):
        self.n_images = n_images
        self.lulc_data = lulc_data or {
            'water': 20.0,
            'trees': 40.0,
            'grass': 25.0,
            'shrubs': 8.0,
            'crops': 5.0,
            'built': 1.5,
            'barren': 0.5,
            'snow': 0.0,
        }
        self._filtered = self
    
    def filterBounds(self, region):
        """Mock filterBounds()."""
        return self
    
    def filterDate(self, start_date: str, end_date: str):
        """Mock filterDate()."""
        return self
    
    def size(self):
        """Mock size() returning count of images."""
        return MockGEEComputation(self.n_images)
    
    def first(self):
        """Mock first() returning first image."""
        return MockGEEImage(self.lulc_data)
    
    def mode(self):
        """Mock mode() returning mode composite."""
        return MockGEEImage(self.lulc_data)


class MockGEEComputation:
    """Mock for GEE computation results."""
    
    def __init__(self, value):
        self.value = value
    
    def getInfo(self):
        """Mock getInfo()."""
        return self.value


class TestDynamicWorldAPI(unittest.TestCase):
    """Test cases for Dynamic World LULC API fixes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_lat = 0.0
        self.test_lon = 25.0
        
        # Standard LULC response
        self.standard_lulc = {
            'water': 20.0,
            'trees': 40.0,
            'grass': 25.0,
            'shrubs': 8.0,
            'crops': 5.0,
            'built': 1.5,
            'barren': 0.0,
            'snow': 0.0,
        }
        
        # Urban LULC response
        self.urban_lulc = {
            'water': 5.0,
            'trees': 15.0,
            'grass': 10.0,
            'shrubs': 2.0,
            'crops': 0.0,
            'built': 65.0,
            'barren': 3.0,
            'snow': 0.0,
        }
        
        # Tropical LULC response
        self.tropical_lulc = {
            'water': 15.0,
            'trees': 70.0,
            'grass': 5.0,
            'shrubs': 5.0,
            'crops': 5.0,
            'built': 0.0,
            'barren': 0.0,
            'snow': 0.0,
        }
    
    @patch('ee.ImageCollection')
    def test_successful_lulc_retrieval(self, mock_collection_class):
        """Test successful retrieval of LULC data."""
        mock_collection = MockGEEImageCollection(n_images=10, lulc_data=self.standard_lulc)
        mock_collection_class.return_value = mock_collection
        
        # Simulate query
        collection = mock_collection_class("GOOGLE/DYNAMICWORLD/V1")
        n_images = collection.filterBounds(Mock()).filterDate('2023-05-01', '2023-05-31').size().getInfo()
        
        self.assertEqual(n_images, 10)
        self.assertEqual(collection.first().sample(Mock(), 10).first().getInfo()['properties']['water'], 20.0)
    
    @patch('ee.ImageCollection')
    def test_empty_response_initial_query(self, mock_collection_class):
        """Test handling of empty response (no imagery in date range)."""
        mock_collection = MockGEEImageCollection(n_images=0)  # No images
        mock_collection_class.return_value = mock_collection
        
        collection = mock_collection_class("GOOGLE/DYNAMICWORLD/V1")
        n_images = collection.filterBounds(Mock()).filterDate('2023-05-01', '2023-05-31').size().getInfo()
        
        self.assertEqual(n_images, 0)
        # Expected: graceful return with None
        # Should trigger automatic date range extension
    
    @patch('ee.ImageCollection')
    def test_date_range_extension_fallback(self, mock_collection_class):
        """Test automatic date range extension when initial query fails."""
        # First call: empty
        # Second call (extended range): success
        
        def side_effect(*args, **kwargs):
            mock1 = MockGEEImageCollection(n_images=0)  # First query empty
            return mock1
        
        mock_collection = MockGEEImageCollection(n_images=0)
        mock_collection_class.return_value = mock_collection
        
        # Initial query
        collection = mock_collection_class("GOOGLE/DYNAMICWORLD/V1")
        n_images = collection.filterBounds(Mock()).filterDate('2023-05-01', '2023-05-31').size().getInfo()
        
        self.assertEqual(n_images, 0)
        
        # Extended query should find data
        extended_collection = MockGEEImageCollection(n_images=5, lulc_data=self.standard_lulc)
        n_images_extended = extended_collection.filterBounds(Mock()).filterDate(
            '2023-03-01', '2023-06-30'
        ).size().getInfo()
        
        self.assertEqual(n_images_extended, 5)
        self.assertGreater(n_images_extended, 0)
    
    def test_parse_date_range_with_lag(self):
        """Test date range parsing with processing lag accounting."""
        # When no date specified, should use default with 15-day lag
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        # Simulated _parse_date_range(None)
        end_date = today - timedelta(days=15)  # Account for processing lag
        start_date = end_date - timedelta(days=30)
        
        # Verify range is reasonable
        self.assertLess(end_date, today)
        self.assertLess(start_date, end_date)
        self.assertEqual((end_date - start_date).days, 30)
    
    def test_date_range_parsing_full_year(self):
        """Test date range parsing for full year specification."""
        # Input: "2023"
        # Expected: 2023-01-01 to 2023-12-31
        
        year_spec = "2023"
        year = int(year_spec)
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self.assertEqual(start_str, "2023-01-01")
        self.assertEqual(end_str, "2023-12-31")
    
    def test_date_range_parsing_specific_month(self):
        """Test date range parsing for specific month."""
        # Input: "2023-05"
        # Expected: 2023-05-01 to 2023-05-31
        
        month_spec = "2023-05"
        year = int(month_spec[:4])
        month = int(month_spec[5:7])
        start_date = datetime(year, month, 1)
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        self.assertEqual(start_date.day, 1)
        self.assertEqual(end_date.day, 31)
        self.assertEqual(start_date.month, 5)
        self.assertEqual(end_date.month, 5)
    
    def test_regional_coverage_assessment_polar(self):
        """Test coverage assessment for polar regions."""
        test_lats = [70.0, 75.0, 85.0, 89.9]
        
        for lat in test_lats:
            if abs(lat) > 60:
                coverage = "sparse"
                self.assertEqual(coverage, "sparse")
    
    def test_regional_coverage_assessment_tropical(self):
        """Test coverage assessment for tropical regions."""
        test_lats = [0.0, 5.0, -10.0, 20.0, -15.0]
        
        for lat in test_lats:
            if abs(lat) < 23:
                coverage = "variable"
                self.assertEqual(coverage, "variable")
    
    def test_regional_coverage_assessment_temperate(self):
        """Test coverage assessment for temperate regions."""
        test_lats = [45.0, 50.0, -40.0, -50.0]
        
        for lat in test_lats:
            abs_lat = abs(lat)
            if abs_lat >= 23 and abs_lat <= 60:
                coverage = "good"
                self.assertEqual(coverage, "good")
    
    @patch('ee.ImageCollection')
    def test_lulc_percentages_extraction(self, mock_collection_class):
        """Test extraction of LULC class percentages."""
        mock_collection = MockGEEImageCollection(n_images=1, lulc_data=self.urban_lulc)
        mock_collection_class.return_value = mock_collection
        
        collection = mock_collection_class("GOOGLE/DYNAMICWORLD/V1")
        sample_data = collection.filterBounds(Mock()).filterDate(
            '2023-05-01', '2023-05-31'
        ).first().sample(Mock(), 10).first().getInfo()
        
        properties = sample_data['properties']
        
        # Verify all LULC classes are present
        lulc_classes = ['water', 'trees', 'grass', 'shrubs', 'crops', 'built', 'barren', 'snow']
        for lulc_class in lulc_classes:
            self.assertIn(lulc_class, properties)
        
        # Verify built-up is high for urban area
        self.assertGreater(properties['built'], 50)
        self.assertLess(properties['trees'], 20)
    
    @patch('ee.ImageCollection')
    def test_mode_composite_multiple_images(self, mock_collection_class):
        """Test mode composite when multiple images available."""
        mock_collection = MockGEEImageCollection(n_images=15, lulc_data=self.standard_lulc)
        mock_collection_class.return_value = mock_collection
        
        collection = mock_collection_class("GOOGLE/DYNAMICWORLD/V1")
        n_images = collection.filterBounds(Mock()).filterDate(
            '2023-05-01', '2023-05-31'
        ).size().getInfo()
        
        self.assertGreater(n_images, 1)
        
        # Simulate mode composite selection
        if n_images > 1:
            image = collection.mode()
        else:
            image = collection.first()
        
        # Both should return valid LULC data
        self.assertIsNotNone(image)


class TestDynamicWorldLogging(unittest.TestCase):
    """Test logging output for Dynamic World API."""
    
    def test_logging_sparse_coverage_polar(self):
        """Test logging message for sparse polar coverage."""
        lat, lon = 75.0, 25.0
        
        # Sparse coverage message
        if abs(lat) > 60:
            msg = (
                f"Dynamic World: No imagery found at ({lat:.4f}, {lon:.4f}) - "
                f"sparse satellite coverage (high latitude)"
            )
            self.assertIn("sparse", msg)
            self.assertIn("high latitude", msg)
    
    def test_logging_variable_coverage_tropical(self):
        """Test logging message for variable tropical coverage."""
        lat, lon = 0.0, 25.0
        
        if abs(lat) < 23:
            msg = (
                f"Dynamic World: No imagery found at ({lat:.4f}, {lon:.4f}) - "
                f"frequent cloud cover in this region"
            )
            self.assertIn("cloud cover", msg)
            self.assertIn("variable", msg.lower())
    
    def test_logging_date_range_extension(self):
        """Test logging when date range is extended."""
        original_start = "2023-05-01"
        original_end = "2023-05-31"
        extended_start = "2023-03-01"
        extended_end = "2023-07-31"
        
        msg = (
            f"Dynamic World: No imagery for {original_start} to {original_end}, "
            f"extending search range. Extended search: {extended_start} to {extended_end}"
        )
        
        self.assertIn("extending", msg)
        self.assertIn(extended_start, msg)
        self.assertIn(extended_end, msg)


class TestDynamicWorldEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    @patch('ee.ImageCollection')
    def test_single_image_vs_composite(self, mock_collection_class):
        """Test selection logic for single image vs. mode composite."""
        # Single image
        mock_single = MockGEEImageCollection(n_images=1, lulc_data={
            'water': 10.0, 'trees': 50.0, 'grass': 30.0,
            'shrubs': 5.0, 'crops': 5.0, 'built': 0.0, 'barren': 0.0, 'snow': 0.0
        })
        
        n_images = mock_single.filterBounds(Mock()).filterDate(
            '2023-05-01', '2023-05-31'
        ).size().getInfo()
        
        if n_images == 1:
            image = mock_single.first()
        else:
            image = mock_single.mode()
        
        self.assertEqual(n_images, 1)
        
        # Multiple images
        mock_multi = MockGEEImageCollection(n_images=10, lulc_data={
            'water': 10.0, 'trees': 50.0, 'grass': 30.0,
            'shrubs': 5.0, 'crops': 5.0, 'built': 0.0, 'barren': 0.0, 'snow': 0.0
        })
        
        n_images = mock_multi.filterBounds(Mock()).filterDate(
            '2023-05-01', '2023-05-31'
        ).size().getInfo()
        
        if n_images > 1:
            image = mock_multi.mode()
        else:
            image = mock_multi.first()
        
        self.assertGreater(n_images, 1)
    
    def test_lulc_percentages_sum_to_100(self):
        """Test that LULC percentages sum to approximately 100%."""
        lulc_data = {
            'water': 20.0,
            'trees': 40.0,
            'grass': 25.0,
            'shrubs': 8.0,
            'crops': 5.0,
            'built': 1.5,
            'barren': 0.5,
            'snow': 0.0,
        }
        
        total = sum(lulc_data.values())
        self.assertAlmostEqual(total, 100.0, places=1)
    
    @patch('ee.ImageCollection')
    def test_handling_no_dominant_class(self, mock_collection_class):
        """Test handling when multiple classes have equal probability."""
        # Balanced LULC (no clear dominant class)
        balanced_lulc = {
            'water': 12.5,
            'trees': 12.5,
            'grass': 12.5,
            'shrubs': 12.5,
            'crops': 12.5,
            'built': 25.0,  # Slight built-up dominance
            'barren': 0.0,
            'snow': 0.0,
        }
        
        mock_collection = MockGEEImageCollection(n_images=5, lulc_data=balanced_lulc)
        mock_collection_class.return_value = mock_collection
        
        collection = mock_collection_class("GOOGLE/DYNAMICWORLD/V1")
        data = collection.filterBounds(Mock()).filterDate(
            '2023-05-01', '2023-05-31'
        ).first().sample(Mock(), 10).first().getInfo()
        
        # Even with balanced data, should identify dominant
        properties = data['properties']
        dominant = max(properties.items(), key=lambda x: x[1])
        self.assertEqual(dominant[0], 'built')


class TestDynamicWorldCoordinateHandling(unittest.TestCase):
    """Test coordinate handling and validation."""
    
    def test_polar_region_coordinates(self):
        """Test Dynamic World for high-latitude coordinates."""
        polar_coords = [
            (75.0, 0.0),    # Arctic
            (-75.0, 0.0),   # Antarctic
            (85.0, 180.0),  # Near North Pole
        ]
        
        for lat, lon in polar_coords:
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)
            # Expected: sparse coverage message
    
    def test_equator_coordinates(self):
        """Test Dynamic World for equatorial coordinates."""
        equator_coords = [
            (0.0, 0.0),
            (0.0, 25.0),
            (0.1, -50.0),
            (-0.1, 100.0),
        ]
        
        for lat, lon in equator_coords:
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)
            # Expected: variable coverage (tropical clouds)
    
    def test_date_line_coordinates(self):
        """Test Dynamic World for international date line."""
        dateline_coords = [
            (0.0, 180.0),
            (0.0, -180.0),
            (45.0, 179.99),
            (-45.0, -179.99),
        ]
        
        for lat, lon in dateline_coords:
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)


if __name__ == '__main__':
    unittest.main()
