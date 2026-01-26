# Market Volatility Spike Detector

**🔴 Live Demo:** [marketvolatilityanomaly.gabrieljames.me](https://marketvolatilityanomaly.gabrieljames.me)

**📁 GitHub:** [github.com/gabsgj/Market-Volatility-Spike-Detector](https://github.com/gabsgj/Market-Volatility-Spike-Detector)

A statistical approach to detecting unusual market behavior using multivariate Gaussian density estimation and Mahalanobis distance scoring. This project was developed as part of my exploration into probabilistic machine learning methods for financial time series, building on concepts from the **Machine Learning Specialization by Andrew Ng and DeepLearning.AI**.

---

## The Problem I Wanted to Solve

When I started looking into market volatility, I noticed something interesting: traditional volatility indicators like VIX or simple rolling standard deviation treat each data point independently. But market stress rarely shows up in just one dimension. During the COVID crash in March 2020, for example, we didn't just see large price drops—we saw *simultaneous* spikes in volatility, unusual volume patterns, and correlation breakdowns across asset classes.

This got me thinking: what if we modeled market behavior as a joint distribution across multiple features, and then measured how "unlikely" each day's observation was under that distribution?

That's exactly what this project does.

---
<img width="1897" height="2245" alt="Image" src="https://github.com/user-attachments/assets/927679b1-82a0-47e6-978a-a91db940b362" />
<img width="1897" height="1057" alt="Image" src="https://github.com/user-attachments/assets/af73e1b7-5fa0-43b9-a975-825583c685fe" />
<img width="1897" height="1057" alt="Image" src="https://github.com/user-attachments/assets/b08134fd-003d-4387-a3b2-93a146bd9ce3" />
<img width="1897" height="4749" alt="Image" src="https://github.com/user-attachments/assets/ed8e668c-8503-460b-accd-ea5f480dcf4a" />

---

## How the Model Works

### The Core Idea

Instead of looking at individual indicators, I model the market's "normal" state as a multivariate Gaussian distribution over four features:

1. **Log Return**: `log(Pₜ / Pₜ₋₁)` — captures daily price changes
2. **Rolling Volatility**: 14-day standard deviation of returns — measures recent variability
3. **Absolute Return**: `|log return|` — captures movement magnitude regardless of direction
4. **Volume Change**: `log(Vₜ / Vₜ₋₁)` — detects unusual trading activity

Each trading day becomes a 4-dimensional point in feature space.

### Why These Features?

I spent some time thinking about what features would actually capture market stress. Log returns are standard—they're approximately normally distributed and have nice mathematical properties (additivity, etc.). Rolling volatility captures the "volatility clustering" effect that Bollerslev documented in his 1986 GARCH paper. Volume changes matter because institutional activity often precedes major moves. And absolute returns help distinguish high-volatility regimes from calm ones.

### The Rolling Estimation Approach

Markets are non-stationary. What's "normal" in 2019 looks very different from "normal" in mid-2020 during COVID. So rather than fitting a single global Gaussian, I use a **rolling 60-day window** to estimate the mean vector μ and covariance matrix Σ.

For each day t, I:
1. Take the previous 60 days of feature vectors
2. Compute the sample mean μ̂ and sample covariance Σ̂
3. Add a small regularization term (ε·I) to ensure Σ̂ is invertible
4. Calculate the Mahalanobis distance for day t's observation

This means the model adapts to changing market regimes automatically.

### Mahalanobis Distance: The Key Metric

The Mahalanobis distance is defined as:

```
D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Why not just use Euclidean distance? Because Mahalanobis accounts for two crucial things:

1. **Different variances across features**: A 2% daily move might be unusual for bonds but normal for crypto. Mahalanobis normalizes by each feature's variance.

2. **Correlations between features**: If returns and volume are correlated, a point that's "far" in both dimensions might not actually be that unusual. The covariance matrix captures these relationships.

Under the Gaussian assumption, D² follows a chi-squared distribution with p degrees of freedom (where p = number of features = 4). This gives us a principled way to set thresholds.

### Threshold Selection

I use the 99.5th percentile of observed D² values as the anomaly threshold. This means roughly 0.5% of days get flagged as anomalies—consistent with the idea that we're looking for truly unusual events, not just slightly elevated volatility.

The percentile-based approach is robust because it doesn't assume the chi-squared distribution holds exactly (which it won't, since returns have fat tails).

---

## How Event Correlation Works

This is one of the features I'm most proud of. After detecting anomalies, the system automatically matches them against a database of known market events.

### The Matching Algorithm

1. I maintain a curated list of significant market events with their dates (COVID crash, SVB collapse, Fed announcements, etc.)

2. For each detected anomaly date, I check if any known event falls within ±5 days

3. If there's a match, I record it along with metadata:
   - Event name
   - Event date
   - Anomaly date
   - Days difference
   - Match type ("exact" for same day, "near" for ±1-5 days)

### Why This Matters

Event correlation serves as validation. If my detector flags March 16, 2020 as an anomaly and that matches the COVID crash, it suggests the model is capturing real market stress events. If all my anomalies are on random Tuesdays with no news, something's probably wrong.

It also provides interpretability. Instead of just saying "this day was anomalous," I can say "this anomaly corresponds to the SVB bank collapse."

---

## Data Pipeline

### Data Source

I use the `yfinance` library to fetch historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance. The library handles all the API complexity—rate limiting, data formatting, corporate actions adjustments, etc.

### Fetch Process

When you select an index (SPY, NIFTY50, NASDAQ) and time period (1y, 2y, 5y):

1. `yfinance.Ticker(symbol).history(period=period)` fetches the raw data
2. I get back a pandas DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
3. The index is datetime, making time-series operations straightforward

### Feature Engineering Pipeline

The raw OHLCV data goes through these transformations:

```python
# Log return
features['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

# Rolling volatility (14-day)
features['rolling_volatility'] = features['log_return'].rolling(14).std()

# Absolute return
features['abs_return'] = np.abs(features['log_return'])

# Volume change
features['volume_change'] = np.log(df['Volume'] / df['Volume'].shift(1))
```

I handle edge cases like:
- NaN values from the rolling window initialization (first 13 days)
- Infinite values from zero-volume days
- Missing data from market holidays

The final feature matrix has shape `(n_days - window_size, 4)`.

---

## The Anomaly Detection Algorithm

Here's the actual implementation logic:

```python
def fit_rolling(self, X):
    n_samples = X.shape[0]
    distances = np.zeros(n_samples)
    
    for t in range(self.window_size, n_samples):
        # Get the rolling window
        window = X[t - self.window_size:t]
        
        # Estimate parameters
        mu = np.mean(window, axis=0)
        sigma = np.cov(window.T) + self.epsilon * np.eye(X.shape[1])
        
        # Compute Mahalanobis distance
        sigma_inv = np.linalg.inv(sigma)
        diff = X[t] - mu
        distances[t] = diff @ sigma_inv @ diff
    
    # Set threshold
    valid_distances = distances[self.window_size:]
    self.threshold = np.percentile(valid_distances, self.threshold_percentile)
    
    # Flag anomalies
    anomalies = distances > self.threshold
    
    return distances, anomalies
```

The regularization term `epsilon * np.eye(...)` is crucial. Without it, if any feature has zero variance in the window (rare but possible), the covariance matrix becomes singular and we can't invert it.

---

## Technical Architecture

### Backend

- **Flask**: Lightweight Python web framework. Perfect for this scale of project.
- **NumPy**: All the linear algebra (matrix operations, inversions)
- **Pandas**: Data manipulation and time-series indexing
- **yfinance**: Yahoo Finance API wrapper

### Frontend

- **Chart.js**: Interactive charts for price series and distance plots
- **Vanilla JavaScript**: Form handling, AJAX calls, DOM manipulation
- **CSS Variables**: Color scheme and spacing constants

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Dashboard page |
| `/analyze` | POST | Run anomaly detection |
| `/api/export/<format>` | POST | Export results as CSV or JSON |
| `/api/compare` | POST | Compare multiple indices |
| `/api/backtest` | POST | Validate against known events |

---

## Configuration Parameters

These are the knobs you can tune:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Rolling Window | 60 days | ~3 months of trading days. Long enough for stable estimates, short enough to adapt to regimes. |
| Threshold Percentile | 99.5 | Flags only the most extreme 0.5% of observations. |
| Volatility Window | 14 days | Standard 2-week period for rolling volatility. |
| Regularization (ε) | 1e-6 | Small enough to not affect results, large enough to ensure numerical stability. |

---

## What I Learned Building This

### On Gaussian Assumptions

Real financial returns have fat tails—extreme events happen more often than a Gaussian would predict. But for anomaly detection, this actually works in our favor. If we fit a Gaussian and an observation is 5+ standard deviations away, it really is unusual, even if the true distribution has fatter tails.

### On Feature Engineering

Getting the features right mattered more than sophisticated modeling. I initially tried just using returns, but adding volume and volatility features significantly improved detection of events like SVB collapse where volume spiked before prices crashed.

### On Rolling vs. Static Models

I initially fit a single global Gaussian to all historical data. The problem: the COVID period had such high volatility that it dominated the covariance estimate, making post-COVID anomalies hard to detect. Rolling estimation fixed this.

---

## Running the Project

```bash
# Clone and setup
pip install -r requirements.txt

# Run the server
python app.py
```

Then open http://localhost:5000 in your browser.

For best results, select SPY (S&P 500 ETF) with a 5-year period. This captures multiple market regimes including COVID, the 2022 rate hikes, and recent volatility.

---

## Limitations and Future Work

**Current Limitations:**
- Assumes feature independence within each window (the covariance helps, but still)
- Threshold is static across regimes
- No real-time streaming (batch processing only)

**Ideas for Extension:**
- Use Student-t distribution instead of Gaussian for heavier tails
- Dynamic threshold based on recent volatility regime
- WebSocket for real-time updates
- Feature importance decomposition (which feature caused the anomaly?)
- Cross-asset correlation analysis

---

## References and Acknowledgments

This project builds on concepts from:

- **Machine Learning Specialization** by Andrew Ng & DeepLearning.AI — particularly the Gaussian anomaly detection module
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity" — volatility clustering
- Mahalanobis, P.C. (1936). "On the generalized distance in statistics" — the distance metric
- Cont, R. (2001). "Empirical properties of asset returns" — stylized facts about returns

---

## Author

**Gabriel James**

Built as a capstone project exploring probabilistic ML methods for financial applications.
