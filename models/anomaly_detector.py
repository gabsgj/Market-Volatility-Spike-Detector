"""
Rolling Multivariate Gaussian Anomaly Detector for Market Volatility

This module implements adaptive anomaly detection using rolling multivariate 
Gaussian models and Mahalanobis distance to identify volatility spikes.
"""

import numpy as np
from scipy.stats import chi2

class RollingGaussianDetector:
    """
    Adaptive market stress detector using rolling multivariate Gaussian models.
    
    At each time step t, market state x_t is modeled as:
        x_t ~ N(μ_t, Σ_t)
    
    Where μ_t and Σ_t are estimated from the last W days.
    """
    
    def __init__(self, window_size=60, epsilon_reg=1e-6, threshold_percentile=99.5):
        """
        Initialize the detector.
        
        Args:
            window_size: Rolling window size (W) in days
            epsilon_reg: Regularization constant for covariance matrix
            threshold_percentile: Percentile for anomaly threshold (e.g., 99.5)
        """
        self.window_size = window_size
        self.epsilon_reg = epsilon_reg
        self.threshold_percentile = threshold_percentile
        self.mu = None
        self.sigma = None
        self.distances = []
        self.threshold = None
        
    def estimate_gaussian(self, X):
        """
        Calculates mean and covariance of all features in the dataset.
        
        Args:
            X (ndarray): (m, n) Data matrix
        
        Returns:
            mu (ndarray): (n,) Mean of all features
            sigma (ndarray): (n, n) Covariance matrix
        """
        m, n = X.shape
        
        # Compute mean
        mu = np.mean(X, axis=0)
        
        # Compute covariance matrix
        sigma = np.cov(X, rowvar=False)
        
        # Handle 1D case
        if sigma.ndim == 0:
            sigma = np.array([[sigma]])
        elif sigma.ndim == 1:
            sigma = np.diag(sigma)
        
        # Regularize to avoid numerical instability
        sigma = sigma + self.epsilon_reg * np.eye(n)
        
        return mu, sigma
    
    def mahalanobis_distance(self, x, mu, sigma):
        """
        Compute Mahalanobis distance for a point.
        
        D(x) = (x - μ)^T Σ^(-1) (x - μ)
        
        Args:
            x: Feature vector for current day
            mu: Mean vector
            sigma: Covariance matrix
            
        Returns:
            distance: Mahalanobis distance (scalar)
        """
        diff = x - mu
        sigma_inv = np.linalg.pinv(sigma)  # Use pseudo-inverse for stability
        distance = np.dot(np.dot(diff, sigma_inv), diff.T)
        return distance
    
    def multivariate_gaussian(self, X, mu, var):
        """
        Computes the probability density function of the examples X 
        under the multivariate Gaussian distribution.
        
        Args:
            X: Data matrix (m, n)
            mu: Mean vector
            var: Variance (1D) or covariance matrix (2D)
            
        Returns:
            p: Probability densities
        """
        k = len(mu)
        
        if var.ndim == 1:
            var = np.diag(var)
            
        X_centered = X - mu
        det_var = np.linalg.det(var)
        
        if det_var == 0:
            det_var = self.epsilon_reg
            
        p = (2 * np.pi)**(-k/2) * det_var**(-0.5) * \
            np.exp(-0.5 * np.sum(np.matmul(X_centered, np.linalg.pinv(var)) * X_centered, axis=1))
        
        return p
    
    def fit_rolling(self, X):
        """
        Fit rolling Gaussian model and compute distances for each point.
        
        Args:
            X: Feature matrix (m, n) where m is number of days
            
        Returns:
            distances: Array of Mahalanobis distances
            anomalies: Boolean array indicating anomalies
        """
        m, n = X.shape
        distances = np.zeros(m)
        
        # Need at least window_size points to start
        for t in range(self.window_size, m):
            # Get historical window
            X_window = X[t-self.window_size:t]
            
            # Estimate parameters from window
            mu_t, sigma_t = self.estimate_gaussian(X_window)
            
            # Compute distance for current point
            distances[t] = self.mahalanobis_distance(X[t], mu_t, sigma_t)
        
        # Set distances for initial window to NaN (not enough data)
        distances[:self.window_size] = np.nan
        
        self.distances = distances
        
        # Compute threshold from valid distances
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) > 0:
            self.threshold = np.percentile(valid_distances, self.threshold_percentile)
        else:
            self.threshold = 0
        
        # Identify anomalies
        anomalies = distances > self.threshold
        anomalies[:self.window_size] = False  # Initial window cannot be anomalies
        
        return distances, anomalies
    
    def get_chi2_threshold(self, n_features, alpha=0.995):
        """
        Get threshold using chi-squared distribution.
        
        D_t > χ²_{d,α}
        
        Args:
            n_features: Number of features (degrees of freedom)
            alpha: Significance level
            
        Returns:
            threshold: Chi-squared threshold
        """
        return chi2.ppf(alpha, df=n_features)
    
    def select_threshold(self, y_val, p_val):
        """
        Finds the best threshold using F1 score on validation set.
        
        Args:
            y_val: Ground truth labels (1 for anomaly, 0 for normal)
            p_val: Probability values from Gaussian model
            
        Returns:
            best_epsilon: Optimal threshold
            best_F1: Best F1 score
        """
        best_epsilon = 0
        best_F1 = 0
        
        step_size = (max(p_val) - min(p_val)) / 1000
        
        for epsilon in np.arange(min(p_val), max(p_val), step_size):
            predictions = (p_val < epsilon)
            
            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))
            
            if (tp + fp) == 0 or (tp + fn) == 0:
                continue
                
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            
            if (prec + rec) == 0:
                continue
                
            F1 = 2 * prec * rec / (prec + rec)
            
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon
        
        return best_epsilon, best_F1
    
    def get_feature_contributions(self, x, mu, sigma):
        """
        Analyze which features contribute most to the anomaly score.
        
        Args:
            x: Current feature vector
            mu: Mean vector
            sigma: Covariance matrix
            
        Returns:
            contributions: Dict mapping feature index to contribution
        """
        diff = x - mu
        sigma_inv = np.linalg.pinv(sigma)
        
        # Individual feature contributions (simplified)
        contributions = {}
        for i in range(len(diff)):
            contributions[i] = abs(diff[i]) * sigma_inv[i, i]
            
        return contributions
