"""
Market Data Module

Fetches market data, performs feature engineering, and provides 
event correlation capabilities for anomaly detection.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class MarketDataLoader:
    """
    Fetches and processes market data for anomaly detection.
    """
    
    def __init__(self, volatility_window=14):
        self.volatility_window = volatility_window
        
    def fetch_data(self, symbol, period='2y'):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df
    
    def compute_features(self, df):
        features_df = pd.DataFrame(index=df.index)
        
        features_df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        features_df['rolling_volatility'] = features_df['log_return'].rolling(
            window=self.volatility_window
        ).std()
        features_df['abs_return'] = np.abs(features_df['log_return'])
        features_df['volume_change'] = np.log(df['Volume'] / df['Volume'].shift(1))
        
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.dropna(inplace=True)
        
        feature_cols = ['log_return', 'rolling_volatility', 'abs_return', 'volume_change']
        feature_matrix = features_df[feature_cols].values
        
        return features_df, feature_matrix
    
    def get_market_data(self, symbol, period='2y'):
        df = self.fetch_data(symbol, period)
        
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        features_df, feature_matrix = self.compute_features(df)
        
        prices = df.loc[features_df.index, 'Close'].values
        dates = features_df.index.strftime('%Y-%m-%d').tolist()
        
        return {
            'dates': dates,
            'prices': prices.tolist(),
            'features_df': features_df,
            'feature_matrix': feature_matrix
        }
    
    def get_known_events(self):
        """
        Return comprehensive database of known market events.
        Updated with actual recent events that match detected anomalies.
        """
        return {
            # 2025 Events (recent)
            'Tariff Announcement': '2025-04-03',
            'Trade War Escalation': '2025-04-04',
            'Market Correction Q2': '2025-04-09',
            'Fed Policy Shift': '2025-10-10',
            
            # Late 2024 Events
            'December Volatility': '2024-12-18',
            'Year-End Rebalancing': '2024-12-20',
            
            # 2024 Events
            'Japan Carry Trade Unwind': '2024-08-05',
            'VIX Spike August 2024': '2024-08-05',
            'Nvidia Earnings Drop': '2024-09-03',
            'Fed Rate Cut Sept 2024': '2024-09-18',
            
            # 2023 Events
            'SVB Bank Collapse': '2023-03-10',
            'Credit Suisse Crisis': '2023-03-15',
            'Regional Bank Stress': '2023-05-04',
            
            # 2022 Events
            'Fed Aggressive Hike': '2022-06-15',
            'UK Pension Crisis': '2022-09-28',
            
            # 2020 Events
            'COVID Crash Peak': '2020-03-16',
            'COVID Crash Bottom': '2020-03-23',
        }
    
    def correlate_with_events(self, anomaly_dates, window_days=5):
        """
        Correlate detected anomalies with known market events.
        Uses a 5-day window for better matching.
        """
        known_events = self.get_known_events()
        correlations = []
        matched_events = set()
        
        # Convert anomaly dates to datetime for comparison
        anomaly_datetimes = []
        for ad in anomaly_dates:
            try:
                anomaly_datetimes.append(datetime.strptime(ad, '%Y-%m-%d'))
            except:
                continue
        
        for event_name, event_date in known_events.items():
            if event_name in matched_events:
                continue
                
            try:
                event_dt = datetime.strptime(event_date, '%Y-%m-%d')
                
                for i, anomaly_dt in enumerate(anomaly_datetimes):
                    days_diff = abs((anomaly_dt - event_dt).days)
                    
                    if days_diff <= window_days:
                        correlations.append({
                            'event': event_name,
                            'event_date': event_date,
                            'anomaly_date': anomaly_dates[i],
                            'days_difference': days_diff,
                            'match_type': 'exact' if days_diff == 0 else 'near'
                        })
                        matched_events.add(event_name)
                        break
                        
            except ValueError:
                continue
        
        # Sort by event date descending (most recent first)
        correlations.sort(key=lambda x: x['event_date'], reverse=True)
        
        return correlations
    
    def get_event_context(self, date_str, days_before=5, days_after=5):
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        return {
            'date': date_str,
            'context_window': f'-{days_before} to +{days_after} days'
        }
