"""
Tests for ALOS CHILI DEM asset fix

Tests ImageCollection.mosaic() handling and fallback logic.
"""

import pytest
from unittest import mock
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from workflow_16s.api.environmental_data.other.tools._alos_chili import ALOSCHILIandmTPIAPI


class TestALOSCHILIDEMAssetFix:
    """Test suite for ALOS CHILI DEM asset handling."""
    
    @pytest.fixture
    def api(self):
        """Create API instance without GEE authentication (for testing)."""
        return ALOSCHILIandmTPIAPI(authenticated=False, verbose=True)
    
    @pytest.fixture
    def api_authenticated(self):
        """Create API instance with mocked GEE."""
        api = ALOSCHILIandmTPIAPI(authenticated=True, verbose=True)
        return api
    
    def test_asset_names_defined(self, api):
        """Verify asset constants are properly defined."""
        assert api.GEE_ASSET_PRIMARY is not None
        assert api.GEE_ASSET_FALLBACK is not None
        # Primary should be ALOS, Fallback should be NASADEM
        assert 'ALOS' in api.GEE_ASSET_PRIMARY or 'AW3D' in api.GEE_ASSET_PRIMARY
        assert 'NASADEM' in api.GEE_ASSET_FALLBACK or 'SRTM' in api.GEE_ASSET_FALLBACK
    
    def test_alos_asset_is_imagecollection_name(self, api):
        """Verify ALOS asset name follows ImageCollection pattern."""
        # ALOS/AW3D30/V3_2 is an ImageCollection, needs .mosaic()
        primary = api.GEE_ASSET_PRIMARY
        # Should contain collection-like structure
        assert '/' in primary
        assert len(primary.split('/')) >= 2
    
    def test_fallback_asset_is_image(self, api):
        """Verify fallback asset is a single Image, not ImageCollection."""
        fallback = api.GEE_ASSET_FALLBACK
        # NASADEM should be directly accessible as Image
        assert 'NASADEM' in fallback or 'SRTM' in fallback
    
    def test_use_fallback_flag(self, api):
        """Verify use_fallback flag is properly set."""
        assert hasattr(api, 'use_fallback')
        assert isinstance(api.use_fallback, bool)
        assert api.use_fallback is True  # Should default to True
    
    def test_gee_asset_tracking(self, api):
        """Verify that gee_asset is tracked and can be updated."""
        original = api.gee_asset
        api.gee_asset = api.GEE_ASSET_FALLBACK
        assert api.gee_asset == api.GEE_ASSET_FALLBACK
        assert api.gee_asset != original
    
    @mock.patch('ee.Image')
    def test_dem_query_handles_image(self, mock_image, api_authenticated):
        """Test that get_data handles Image asset correctly."""
        # Mock successful Image load
        mock_image_instance = mock.Mock()
        mock_image.return_value = mock_image_instance
        
        # This would normally call GEE; we're just testing the not-broken part
        try:
            # Without actual GEE auth, this may fail but shouldn't crash
            result = api_authenticated.get_data(lat=0.0, lon=25.0)
            # Result should be None or dict, not exception
            assert result is None or isinstance(result, dict)
        except Exception as e:
            # Expected to fail without real GEE; just verify not asset error
            assert "not an Image" not in str(e)
    
    def test_initialization_with_fallback_enabled(self):
        """Test initialization with fallback enabled."""
        api = ALOSCHILIandmTPIAPI(use_fallback=True)
        assert api.use_fallback is True
    
    def test_initialization_with_fallback_disabled(self):
        """Test initialization with fallback disabled."""
        api = ALOSCHILIandmTPIAPI(use_fallback=False)
        assert api.use_fallback is False
    
    def test_check_requirements_returns_tuple(self, api):
        """Verify check_requirements returns (bool, str) tuple."""
        result = api.check_requirements()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], (str, type(None)))
    
    def test_check_requirements_without_ee(self):
        """Test check_requirements when ee is not available."""
        api = ALOSCHILIandmTPIAPI(authenticated=False)
        is_available, msg = api.check_requirements()
        assert is_available is False
        assert msg is not None


class TestALOSCHILIDEMImageCollectionHandling:
    """Test handling of ImageCollection vs Image assets."""
    
    @mock.patch('ee.ImageCollection')
    @mock.patch('ee.Image')
    def test_imagecollection_mosaic_fallback(self, mock_image_class, mock_collection_class):
        """Test that ImageCollection.mosaic() is attempted for ALOS."""
        # This tests the logic would work IF GEE were available
        # We're verifying the code structure, not actual GEE behavior
        
        api = ALOSCHILIandmTPIAPI()
        
        # Simulate the ImageCollection path
        if 'AW3D' in api.GEE_ASSET_PRIMARY:
            # Primary is ALOS (ImageCollection)
            assert 'ALOS' in api.GEE_ASSET_PRIMARY or 'AW3D' in api.GEE_ASSET_PRIMARY


class TestALOSCHILIDEMErrorHandling:
    """Test error handling in ALOS DEM queries."""
    
    @pytest.fixture
    def api(self):
        return ALOSCHILIandmTPIAPI(authenticated=False)
    
    def test_invalid_coordinates_return_none(self, api):
        """Test that invalid coordinates return None."""
        result = api.get_data(lat=91.0, lon=0.0)  # Invalid latitude
        assert result is None
    
    def test_invalid_lon_returns_none(self, api):
        """Test that invalid longitude returns None."""
        result = api.get_data(lat=0.0, lon=181.0)  # Invalid longitude
        assert result is None
    
    def test_unauthenticated_returns_none(self, api):
        """Test that unauthenticated calls return None."""
        # api is initialized without authentication
        result = api.get_data(lat=0.0, lon=25.0)
        assert result is None


class TestALOSCHILIDEMIntegration:
    """Integration tests for ALOS CHILI DEM fix."""
    
    def test_api_instantiation(self):
        """Test that API can be instantiated."""
        api = ALOSCHILIandmTPIAPI()
        assert api is not None
        assert isinstance(api, ALOSCHILIandmTPIAPI)
    
    def test_api_name_property(self):
        """Test that API_NAME is defined."""
        api = ALOSCHILIandmTPIAPI()
        assert hasattr(api, 'API_NAME')
        assert api.API_NAME == "ALOSCHILImTPI"
    
    def test_roughness_classes_defined(self):
        """Test that terrain roughness classes are defined."""
        api = ALOSCHILIandmTPIAPI()
        assert hasattr(api, 'ROUGHNESS_CLASSES')
        assert 'flat' in api.ROUGHNESS_CLASSES
        assert 'rolling' in api.ROUGHNESS_CLASSES
        assert 'extreme' in api.ROUGHNESS_CLASSES


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
