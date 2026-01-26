"""
Configuration settings for Market Volatility Spike Detector
"""

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'market-volatility-detector-secret-key'
    
    # Rolling window settings
    ROLLING_WINDOW = 60  # days
    
    # Feature engineering settings
    VOLATILITY_WINDOW = 14  # days for rolling volatility
    
    # Anomaly detection settings
    EPSILON_REG = 1e-6  # Regularization for covariance matrix
    THRESHOLD_PERCENTILE = 99.5  # Percentile for anomaly threshold
    
    # Market indices
    INDICES = {
        'NIFTY50': '^NSEI',
        'SPY': 'SPY',
        'NASDAQ': '^IXIC'
    }
    
    # Default data period
    DEFAULT_PERIOD = '2y'  # 2 years of data
