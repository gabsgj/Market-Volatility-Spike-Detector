"""
Market Volatility Spike Detector - Flask Application

A Flask web application for detecting market volatility spikes using 
Rolling Multivariate Gaussian Anomaly Detection.

Features:
- Real-time market data analysis
- Mahalanobis distance scoring
- Event correlation analysis
- Export functionality (CSV/JSON)
- Multi-index comparison
"""

from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import pandas as pd
import json
import csv
from io import StringIO
from datetime import datetime

from config import Config
from models.anomaly_detector import RollingGaussianDetector
from data.market_data import MarketDataLoader

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
detector = RollingGaussianDetector(
    window_size=Config.ROLLING_WINDOW,
    epsilon_reg=Config.EPSILON_REG,
    threshold_percentile=Config.THRESHOLD_PERCENTILE
)
data_loader = MarketDataLoader(volatility_window=Config.VOLATILITY_WINDOW)


@app.route('/')
def index():
    """Main dashboard page."""
    indices = list(Config.INDICES.keys())
    return render_template('index.html', indices=indices)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze market data for anomalies.
    
    Returns:
        JSON with dates, prices, distances, anomalies, threshold, and event correlations
    """
    try:
        # Get request parameters
        index_name = request.form.get('index', 'NIFTY50')
        period = request.form.get('period', Config.DEFAULT_PERIOD)
        
        # Get symbol from config
        if index_name not in Config.INDICES:
            return jsonify({'error': f'Unknown index: {index_name}'}), 400
        
        symbol = Config.INDICES[index_name]
        
        # Fetch and process data
        market_data = data_loader.get_market_data(symbol, period)
        
        # Run anomaly detection
        distances, anomalies = detector.fit_rolling(market_data['feature_matrix'])
        
        # Get features for contribution analysis
        features_df = market_data['features_df']
        
        # Get anomaly dates
        anomaly_dates = [market_data['dates'][i] for i in range(len(anomalies)) if anomalies[i]]
        
        # Correlate anomalies with known events
        event_correlations = data_loader.correlate_with_events(anomaly_dates)
        
        # Calculate feature contributions for anomalies
        feature_contributions = calculate_feature_contributions(
            features_df, anomalies, detector
        )
        
        # Prepare response
        response = {
            'dates': market_data['dates'],
            'prices': market_data['prices'],
            'distances': np.nan_to_num(distances, nan=0).tolist(),
            'anomalies': anomalies.tolist(),
            'threshold': float(detector.threshold),
            'index_name': index_name,
            'features': {
                'log_return': features_df['log_return'].tolist(),
                'rolling_volatility': features_df['rolling_volatility'].tolist(),
                'abs_return': features_df['abs_return'].tolist(),
                'volume_change': np.nan_to_num(features_df['volume_change'].values, nan=0).tolist()
            },
            'anomaly_count': int(np.sum(anomalies)),
            'anomaly_dates': anomaly_dates,
            'event_correlations': event_correlations,
            'feature_contributions': feature_contributions,
            'statistics': {
                'mean_distance': float(np.nanmean(distances)),
                'max_distance': float(np.nanmax(distances)),
                'std_distance': float(np.nanstd(distances)),
                'detection_rate': float(np.sum(anomalies) / len(anomalies) * 100)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_feature_contributions(features_df, anomalies, detector):
    """
    Calculate which features contributed most to each anomaly.
    
    Uses z-scores to identify the most anomalous features.
    """
    contributions = []
    feature_names = ['log_return', 'rolling_volatility', 'abs_return', 'volume_change']
    
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            row = features_df.iloc[i]
            # Calculate z-scores for each feature
            z_scores = {}
            for feat in feature_names:
                col = features_df[feat]
                z = abs((row[feat] - col.mean()) / col.std())
                z_scores[feat] = round(float(z), 2)
            
            # Find dominant feature
            dominant = max(z_scores, key=z_scores.get)
            contributions.append({
                'date': features_df.index[i].strftime('%Y-%m-%d'),
                'z_scores': z_scores,
                'dominant_feature': dominant
            })
    
    return contributions


@app.route('/api/events')
def get_events():
    """Get known market events for overlay."""
    events = data_loader.get_known_events()
    return jsonify(events)


@app.route('/api/export/<format_type>', methods=['POST'])
def export_results(format_type):
    """
    Export analysis results in CSV or JSON format.
    
    Args:
        format_type: 'csv' or 'json'
    """
    try:
        data = request.get_json()
        
        if format_type == 'csv':
            # Create CSV
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['Date', 'Price', 'Distance', 'Is_Anomaly', 'Log_Return', 'Volatility'])
            
            # Data rows
            for i in range(len(data['dates'])):
                writer.writerow([
                    data['dates'][i],
                    data['prices'][i],
                    data['distances'][i],
                    data['anomalies'][i],
                    data['features']['log_return'][i],
                    data['features']['rolling_volatility'][i]
                ])
            
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=anomaly_detection_{datetime.now().strftime("%Y%m%d")}.csv'}
            )
            
        elif format_type == 'json':
            return Response(
                json.dumps(data, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename=anomaly_detection_{datetime.now().strftime("%Y%m%d")}.json'}
            )
        else:
            return jsonify({'error': 'Invalid format. Use csv or json'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_indices():
    """
    Compare anomaly detection across multiple indices.
    
    Useful for identifying market-wide stress vs index-specific events.
    """
    try:
        indices = request.form.getlist('indices')
        period = request.form.get('period', '2y')
        
        if not indices:
            indices = list(Config.INDICES.keys())
        
        results = {}
        
        for index_name in indices:
            if index_name not in Config.INDICES:
                continue
                
            symbol = Config.INDICES[index_name]
            market_data = data_loader.get_market_data(symbol, period)
            distances, anomalies = detector.fit_rolling(market_data['feature_matrix'])
            
            anomaly_dates = [market_data['dates'][i] for i in range(len(anomalies)) if anomalies[i]]
            
            results[index_name] = {
                'anomaly_count': int(np.sum(anomalies)),
                'anomaly_dates': anomaly_dates,
                'detection_rate': float(np.sum(anomalies) / len(anomalies) * 100),
                'max_distance': float(np.nanmax(distances))
            }
        
        # Find common anomaly dates (market-wide stress)
        all_dates = []
        for idx_result in results.values():
            all_dates.extend(idx_result['anomaly_dates'])
        
        from collections import Counter
        date_counts = Counter(all_dates)
        common_dates = [date for date, count in date_counts.items() if count >= 2]
        
        return jsonify({
            'indices': results,
            'common_anomalies': common_dates,
            'market_wide_stress_detected': len(common_dates) > 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    Backtest the detector against known market events.
    
    Returns precision/recall metrics for validation.
    """
    try:
        index_name = request.form.get('index', 'SPY')
        period = request.form.get('period', '5y')
        
        if index_name not in Config.INDICES:
            return jsonify({'error': f'Unknown index: {index_name}'}), 400
        
        symbol = Config.INDICES[index_name]
        market_data = data_loader.get_market_data(symbol, period)
        distances, anomalies = detector.fit_rolling(market_data['feature_matrix'])
        
        anomaly_dates = set(market_data['dates'][i] for i in range(len(anomalies)) if anomalies[i])
        
        # Known events
        known_events = data_loader.get_known_events()
        
        # Check detection within ±3 days of known events
        detected_events = []
        missed_events = []
        
        for event_name, event_date in known_events.items():
            event_dt = datetime.strptime(event_date, '%Y-%m-%d')
            
            # Check if any anomaly within 3 days
            detected = False
            for anomaly_date in anomaly_dates:
                anomaly_dt = datetime.strptime(anomaly_date, '%Y-%m-%d')
                if abs((anomaly_dt - event_dt).days) <= 3:
                    detected = True
                    break
            
            if detected:
                detected_events.append({'event': event_name, 'date': event_date})
            else:
                missed_events.append({'event': event_name, 'date': event_date})
        
        return jsonify({
            'total_events': len(known_events),
            'detected_events': detected_events,
            'missed_events': missed_events,
            'detection_rate': len(detected_events) / len(known_events) * 100 if known_events else 0,
            'total_anomalies': int(np.sum(anomalies)),
            'false_positive_estimate': int(np.sum(anomalies)) - len(detected_events)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    """Results page (for direct navigation with query params)."""
    index_name = request.args.get('index', 'NIFTY50')
    indices = list(Config.INDICES.keys())
    return render_template('results.html', 
                         index_name=index_name, 
                         indices=indices)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
