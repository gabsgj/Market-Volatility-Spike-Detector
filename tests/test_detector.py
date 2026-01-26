"""
Unit Tests for Market Volatility Spike Detector

Tests cover:
- Anomaly detection algorithm
- Feature engineering
- Event correlation
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import RollingGaussianDetector
from data.market_data import MarketDataLoader


class TestRollingGaussianDetector(unittest.TestCase):
    """Test suite for anomaly detection algorithm."""
    
    def setUp(self):
        """Initialize detector with test parameters."""
        self.detector = RollingGaussianDetector(
            window_size=10,
            epsilon_reg=1e-6,
            threshold_percentile=95.0
        )
    
    def test_fit_rolling_normal_data(self):
        """Test that normal data produces low anomaly rate."""
        np.random.seed(42)
        # Generate normal data
        X = np.random.randn(100, 4)
        
        distances, anomalies = self.detector.fit_rolling(X)
        
        # Should have ~5% anomalies with 95th percentile threshold
        anomaly_rate = np.sum(anomalies) / len(anomalies)
        self.assertLess(anomaly_rate, 0.10)  # Less than 10%
    
    def test_fit_rolling_with_outliers(self):
        """Test that outliers are detected."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        
        # Insert extreme outliers
        X[50] = [10, 10, 10, 10]  # 10 standard deviations
        X[75] = [-8, -8, -8, -8]
        
        distances, anomalies = self.detector.fit_rolling(X)
        
        # Outliers should be detected
        self.assertTrue(anomalies[50])
        self.assertTrue(anomalies[75])
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance calculation."""
        # Simple case: identity covariance
        x = np.array([3, 4])
        mu = np.array([0, 0])
        cov_inv = np.eye(2)
        
        dist = self.detector._mahalanobis_distance(x, mu, cov_inv)
        
        # Should equal Euclidean distance for identity covariance
        expected = np.sqrt(3**2 + 4**2)
        self.assertAlmostEqual(np.sqrt(dist), expected, places=5)
    
    def test_threshold_calculation(self):
        """Test that threshold is calculated correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        
        self.detector.fit_rolling(X)
        
        # Threshold should be positive
        self.assertGreater(self.detector.threshold, 0)


class TestMarketDataLoader(unittest.TestCase):
    """Test suite for market data loading and feature engineering."""
    
    def setUp(self):
        """Initialize data loader."""
        self.loader = MarketDataLoader(volatility_window=14)
    
    def test_compute_features(self):
        """Test feature computation from OHLCV data."""
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Open': np.random.randn(50).cumsum() + 100,
            'High': np.random.randn(50).cumsum() + 101,
            'Low': np.random.randn(50).cumsum() + 99,
            'Close': np.random.randn(50).cumsum() + 100,
            'Volume': np.abs(np.random.randn(50)) * 1000000
        }, index=dates)
        
        features_df, feature_matrix = self.loader.compute_features(df)
        
        # Check feature columns exist
        self.assertIn('log_return', features_df.columns)
        self.assertIn('rolling_volatility', features_df.columns)
        self.assertIn('abs_return', features_df.columns)
        self.assertIn('volume_change', features_df.columns)
        
        # Check feature matrix shape
        self.assertEqual(feature_matrix.shape[1], 4)
    
    def test_event_correlation(self):
        """Test event correlation matching."""
        anomaly_dates = ['2020-03-23', '2020-02-21', '2024-08-05']
        
        correlations = self.loader.correlate_with_events(anomaly_dates)
        
        # Should find COVID crash and VIX spike
        event_names = [c['event'] for c in correlations]
        self.assertIn('COVID Crash Bottom', event_names)
    
    def test_known_events_database(self):
        """Test that known events database is populated."""
        events = self.loader.get_known_events()
        
        self.assertGreater(len(events), 5)
        self.assertIn('COVID Crash Start', events)
        self.assertIn('SVB Collapse', events)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering edge cases."""
    
    def test_log_return_calculation(self):
        """Test log return is computed correctly."""
        loader = MarketDataLoader()
        
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = [100, 105, 103, 108, 110, 107, 112, 115, 113, 118]
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 2 for p in prices],
            'Low': [p - 2 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 10
        }, index=dates)
        
        features_df, _ = loader.compute_features(df)
        
        # First valid log return should be log(105/100)
        expected = np.log(105/100)
        # Check that log return is approximately correct
        # (may not be exact first value due to NaN handling)
        self.assertTrue(any(
            abs(features_df['log_return'].values[i] - expected) < 0.01 
            for i in range(min(3, len(features_df)))
        ))


if __name__ == '__main__':
    unittest.main()
