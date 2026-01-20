import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Assume your original function is in a file named embedding_utils.py
from workflow_16s.api.environmental_data.google import get_alpha_earth_embeddings

class TestAlphaEarthEmbeddings(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for all tests."""
        self.sample_df = pd.DataFrame({
            'FacilityName': ['Site A', 'Site B'],
            'Latitude': [10.0, 20.0],
            'Longitude': [-10.0, -20.0]
        })

    @patch('workflow_16s.api.environmental_data.google.genai.embed_content')
    def test_successful_embedding_retrieval(self, mock_embed_content):
        """Tests the function when the API call is successful."""
        # Configure the mock to return a fake embedding vector
        fake_embedding = [0.1, 0.2, 0.3]
        mock_embed_content.return_value = {'embedding': fake_embedding}

        # Run the function
        result_df = get_alpha_earth_embeddings(self.sample_df.copy(), 'Latitude', 'Longitude')

        # --- Assertions ---
        # 1. Check that the API was called for each row
        self.assertEqual(mock_embed_content.call_count, 2)
        
        # 2. Check that the 'embedding' column was added
        self.assertIn('embedding', result_df.columns)
        
        # 3. Check that the embedding values are correct
        self.assertTrue(np.array_equal(result_df['embedding'].iloc[0], fake_embedding))
        self.assertTrue(np.array_equal(result_df['embedding'].iloc[1], fake_embedding))

    @patch('workflow_16s.api.environmental_data.google.genai.embed_content')
    def test_api_failure(self, mock_embed_content):
        """Tests the function's error handling when the API call fails."""
        # Configure the mock to raise an exception
        mock_embed_content.side_effect = Exception("API Error")

        # Run the function
        result_df = get_alpha_earth_embeddings(self.sample_df.copy(), 'Latitude', 'Longitude')

        # --- Assertions ---
        # 1. Check that the 'embedding' column was still added
        self.assertIn('embedding', result_df.columns)
        
        # 2. Check that the value for the failed calls is None
        self.assertIsNone(result_df['embedding'].iloc[0])
        self.assertIsNone(result_df['embedding'].iloc[1])

if __name__ == '__main__':
    unittest.main()