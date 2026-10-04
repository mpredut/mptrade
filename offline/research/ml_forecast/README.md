# Machine Learning & Deep Learning Forecast Research (`offline/research/ml_forecast/`)

This directory isolates experimental machine learning and sequence forecasting models.
**None of these scripts run in live bot execution or trade real funds.**

## Modules

1. **`priceprediction.py` (PyTorch LSTM Price Prediction)**
   * CPU-efficient PyTorch implementation of next-step sequence prediction using a sliding window.
   * Replaces legacy/dormant Keras/TensorFlow prototype without requiring TensorFlow.
   * Run standalone:
     ```bash
     myenv/bin/python3 offline/research/ml_forecast/priceprediction.py
     ```

2. **`forecast.py` (Multi-Horizon Walk-Forward ML Benchmark)**
   * Compares 24h classification models (HistGradientBoosting, LogisticRegression) against the Lindy persistence baseline over unseen test windows.
   * Extracts multi-horizon returns, Mann-Kendall Z-score, and Hurst exponent from `intelligence.internal.state`.
   * Run evaluation:
     ```bash
     myenv/bin/python3 offline/research/ml_forecast/forecast.py --symbol TAOUSDC --days 400 --eval
     ```

3. **`vol_chronos.py` (Amazon Chronos Foundation Model for Volatility)**
   * Tests zero-shot foundation model volatility prediction against trailing realized volatility persistence.
   * Run evaluation:
     ```bash
     myenv/bin/python3 offline/research/ml_forecast/vol_chronos.py --symbol TAOUSDC --days 400 --win 24 --horizon 24 --eval
     ```
