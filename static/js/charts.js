/**
 * Chart.js configuration for Market Volatility Spike Detector
 * Using color palette: #EFECE3, #8FABD4, #4A70A9, #000000
 */

// Global chart instances
let priceChart = null;
let distanceChart = null;
let volatilityChart = null;
let volumeChart = null;

// Chart.js defaults for light theme
Chart.defaults.color = '#4a4a4a';
Chart.defaults.borderColor = '#d4d0c8';
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/**
 * Render the price chart with anomaly markers
 */
function renderPriceChart(data) {
    const ctx = document.getElementById('price-chart');
    if (!ctx) return;

    if (priceChart) {
        priceChart.destroy();
    }

    const anomalyPrices = data.prices.map((price, idx) =>
        data.anomalies[idx] ? price : null
    );

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: data.index_name + ' Price',
                    data: data.prices,
                    borderColor: '#4A70A9',
                    backgroundColor: 'rgba(74, 112, 169, 0.08)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Anomalies',
                    data: anomalyPrices,
                    borderColor: 'transparent',
                    backgroundColor: '#c0392b',
                    pointRadius: 7,
                    pointStyle: 'circle',
                    showLine: false,
                    pointBorderWidth: 2,
                    pointBorderColor: '#ffffff'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#ffffff',
                    titleColor: '#000000',
                    bodyColor: '#4a4a4a',
                    borderColor: '#d4d0c8',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 6,
                    boxPadding: 4
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    grid: {
                        color: '#EFECE3'
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

/**
 * Render the Mahalanobis distance chart with threshold line
 */
function renderDistanceChart(data) {
    const ctx = document.getElementById('distance-chart');
    if (!ctx) return;

    if (distanceChart) {
        distanceChart.destroy();
    }

    const thresholdLine = new Array(data.dates.length).fill(data.threshold);

    const distanceColors = data.distances.map((d, idx) =>
        data.anomalies[idx] ? '#c0392b' : '#8FABD4'
    );

    distanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Mahalanobis Distance',
                    data: data.distances,
                    backgroundColor: distanceColors,
                    borderColor: distanceColors,
                    borderWidth: 0,
                    barPercentage: 1,
                    categoryPercentage: 1
                },
                {
                    label: 'Threshold (' + data.threshold.toFixed(2) + ')',
                    data: thresholdLine,
                    type: 'line',
                    borderColor: '#4A70A9',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#ffffff',
                    titleColor: '#000000',
                    bodyColor: '#4a4a4a',
                    borderColor: '#d4d0c8',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 6,
                    callbacks: {
                        label: function (context) {
                            if (context.datasetIndex === 0) {
                                const isAnomaly = data.anomalies[context.dataIndex];
                                return 'Distance: ' + context.raw.toFixed(4) +
                                    (isAnomaly ? ' [ANOMALY]' : '');
                            }
                            return context.dataset.label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    grid: {
                        color: '#EFECE3'
                    },
                    title: {
                        display: true,
                        text: 'Distance',
                        font: {
                            size: 12,
                            weight: 500
                        }
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

/**
 * Render feature contribution charts
 */
function renderFeatureCharts(data) {
    const volCtx = document.getElementById('volatility-chart');
    if (volCtx) {
        if (volatilityChart) volatilityChart.destroy();

        volatilityChart = new Chart(volCtx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Rolling Volatility',
                    data: data.features.rolling_volatility,
                    borderColor: '#4A70A9',
                    backgroundColor: 'rgba(74, 112, 169, 0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: { size: 11 }
                        }
                    }
                },
                scales: {
                    x: { display: false },
                    y: {
                        grid: { color: '#EFECE3' },
                        ticks: { font: { size: 10 } }
                    }
                }
            }
        });
    }

    const volChangeCtx = document.getElementById('volume-chart');
    if (volChangeCtx) {
        if (volumeChart) volumeChart.destroy();

        volumeChart = new Chart(volChangeCtx, {
            type: 'bar',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Volume Change (log)',
                    data: data.features.volume_change,
                    backgroundColor: data.features.volume_change.map(v =>
                        v > 0 ? 'rgba(39, 174, 96, 0.6)' : 'rgba(192, 57, 43, 0.6)'
                    ),
                    borderWidth: 0,
                    barPercentage: 1,
                    categoryPercentage: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: { size: 11 }
                        }
                    }
                },
                scales: {
                    x: { display: false },
                    y: {
                        grid: { color: '#EFECE3' },
                        ticks: { font: { size: 10 } }
                    }
                }
            }
        });
    }
}

/**
 * Render detection statistics
 */
function renderStats(data) {
    const statsGrid = document.getElementById('stats-grid');
    if (!statsGrid) return;

    statsGrid.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Total Days Analyzed</span>
            <span class="stat-value">${data.dates.length}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Anomalies Detected</span>
            <span class="stat-value anomaly-count">${data.anomaly_count}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Anomaly Rate</span>
            <span class="stat-value">${((data.anomaly_count / data.dates.length) * 100).toFixed(2)}%</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Threshold</span>
            <span class="stat-value">${data.threshold.toFixed(4)}</span>
        </div>
    `;
}

/**
 * Render list of anomaly dates
 */
function renderAnomalyList(data) {
    const anomalyList = document.getElementById('anomaly-list');
    if (!anomalyList) return;

    if (data.anomaly_dates.length === 0) {
        anomalyList.innerHTML = '<p style="color: #27ae60; font-weight: 500;">No anomalies detected in this period.</p>';
        return;
    }

    anomalyList.innerHTML = data.anomaly_dates.map(date =>
        `<span class="anomaly-date">${date}</span>`
    ).join('');
}
