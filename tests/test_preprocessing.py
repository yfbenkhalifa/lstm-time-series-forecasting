"""
Unit tests for the time series forecasting project.

Run with: pytest tests/
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import (
    normalize_data,
    create_sequences,
    split_data
)


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions."""

    def setUp(self):
        """Set up test data."""
        self.test_data = np.random.randn(100, 1)

    def test_normalize_data_minmax(self):
        """Test MinMax normalization."""
        normalized, scaler = normalize_data(self.test_data, scaler_type="minmax")
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertGreaterEqual(normalized.min(), 0.0)

    def test_normalize_data_standard(self):
        """Test standard normalization."""
        normalized, scaler = normalize_data(self.test_data, scaler_type="standard")
        self.assertAlmostEqual(normalized.mean(), 0, places=1)

    def test_create_sequences(self):
        """Test sequence creation."""
        sequence_length = 10
        X, y = create_sequences(self.test_data, sequence_length)

        self.assertEqual(X.shape[0], len(self.test_data) - sequence_length)
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(y.shape[0], len(self.test_data) - sequence_length)

    def test_split_data(self):
        """Test data splitting."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, validation_size=0.1
        )

        total_samples = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_samples, len(X))


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity."""

    def test_no_nan_values(self):
        """Test that normalized data has no NaN values."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        normalized, _ = normalize_data(data, scaler_type="minmax")
        self.assertFalse(np.isnan(normalized).any())

    def test_sequence_continuity(self):
        """Test that sequences are continuous."""
        data = np.arange(100).reshape(-1, 1)
        X, y = create_sequences(data, sequence_length=10)

        # Check that sequences are continuous
        for i in range(len(X)):
            expected_next = data[i + 10]
            self.assertEqual(y[i, 0], expected_next[0])


if __name__ == "__main__":
    unittest.main()

