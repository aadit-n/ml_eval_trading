# ML Eval Trading

This project is a Streamlit app for market ML model evaluation. It builds a unified dataset from market data, lets you select technical indicators, and compares many sklearn models on the same feature set. It supports volatility forecasting, rolling stability analysis, and market regime detection.

It is not a trading system and does not do backtesting. It is strictly for model evaluation on time-ordered splits.

**Try it out here - [https://ml-market-eval.streamlit.app](https://ml-market-eval.streamlit.app)**

---

## What This Project Does (At a Glance)

- Downloads OHLCV data for a user-chosen ticker/period/interval.
- Builds a feature table from OHLCV + returns + Tulipy indicators.
- Lets you choose a target:
  - Return signal (binary)
  - Volatility regression (next-period realized volatility)
  - Volatility regime classification (low/medium/high)
- Trains many sklearn models on the same dataset.
- Reports model metrics.
- Computes permutation-importance heatmaps.
- Runs rolling-window stability analysis and predictive decay.
- Detects market regimes using clustering and shows regime-aware metrics.
- Plots price + indicators and predicted signal points per model.

---

## Project Structure

- `streamlit_app.py`
  - UI + orchestration: inputs, dataset build, model training, metrics, plots.
  - Handles session state to persist outputs across reruns.

- `data_pipeline.py`
  - Data fetching and feature construction.
  - Builds targets for return signal / volatility regression / volatility regimes.

- `model_registry.py`
  - Model catalogs for classification and regression.

- `evaluation.py`
  - Computes model metrics (classification + regression).

- `feature_importance.py`
  - Permutation importance engine.

- `stability_analysis.py`
  - Rolling window evaluation and time-sliced permutation importance.

- `regime_detection.py`
  - Clustering-based regime detection and regime naming.

- `indicator_registry.py`
  - Discovers Tulipy indicators and provides metadata for inputs/params.

---

## Requirements

Install dependencies (example):

```bash
pip install streamlit yfinance tulipy scikit-learn pandas numpy matplotlib seaborn
```

---

## How to Run

```bash
streamlit run streamlit_app.py
```

---

## Full Feature Guide (Step-by-Step)

### 1) Data Inputs

- **Ticker**: the symbol to download from Yahoo Finance (e.g., `JPM`, `AAPL`, `SPY`).
- **Period**: how much history to fetch (yfinance period string, e.g., `1y`, `5y`, `10y`).
- **Interval**: bar size for the data (e.g., `1d`, `1h`, `15m`).

Click **Build dataset** to download data and generate features.

---

### 2) Target Selection

#### A) Return Signal (binary)
- Target = `1` if next-period return > threshold, else `0`.
- **Signal threshold**: minimum next-bar return to count as a positive signal.
- Task type: classification.

#### B) Volatility (regression)
- Target = next-period realized volatility.
- Volatility is computed from rolling std of returns.
- Task type: regression.

#### C) Volatility Regime (3-class)
- Target = low / medium / high volatility regime.
- Regimes are defined by quantiles of future volatility based on the training slice.
- Task type: classification.

---

### 3) Volatility Settings

- **Volatility window (bars)**: rolling window length used to compute realized volatility.
- For volatility regimes:
  - **Low regime quantile**: threshold for the low-vol regime (e.g., 0.33).
  - **High regime quantile**: threshold for the high-vol regime (e.g., 0.66).

---

### 4) Train/Test Split

- **Train size** sets the time-ordered split (no shuffling).
- Typical values: `0.7`-`0.85`.

---

### 5) Indicators (Tulipy)

- The app detects installed Tulipy indicators and lists them.
- **Inputs**: which price/volume series the indicator uses (e.g., `close`, `high`, `low`).
- **Params**: indicator-specific parameters (e.g., lookback windows).
- Inputs options are defined in `indicator_registry.py`.
- You can also add custom indicators via JSON:

```json
[{"name": "rsi", "inputs": ["close"], "params": [14], "prefix": "rsi_custom"}]
```

Custom indicator fields:
- **name**: Tulipy function name.
- **inputs**: list of input series names.
- **params**: list of numeric parameters.
- **prefix**: output column prefix (optional).

---

### 6) Feature Selection

- After building the dataset, select which columns feed models.
- You can include or exclude indicators, returns, or OHLCV columns.

---

### 7) Model Comparison

Click **Run comparison**.

Outputs:
- A model comparison table.
- Errors table (if any model fails).
- Target summary (class balance or regression stats).

Model types:
- Classification: Logistic, SVM, trees, ensembles, Naive Bayes, etc.
- Regression: Linear, Ridge/Lasso/ElasticNet, SVR, RF/GB/ExtraTrees, etc.

---

### 8) Permutation Importance Heatmap

- **Importance metric**: what performance metric to measure drop against (varies by task type).
- **Permutation repeats**: number of shuffles per feature (higher = more stable, slower).
- Click **Compute heatmap** to generate feature importance matrix.

---

### 9) Rolling Stability Analysis (Predictive Decay)

This evaluates how model performance changes over time.

Controls:
- **Rolling train window**: rows per training slice.
- **Rolling test window**: rows per test slice.
- **Step size**: how far the window advances each iteration.
- **Rolling metric**: metric used for the rolling score.
- **Models to include**: limit to reduce runtime.
- **Time-sliced permutation importance**: optional, very expensive.

Outputs:
- Window-by-window scores.
- Summary table with mean score, std, slope (decay), and first-to-last delta.
- Optional feature stability via rank correlation across windows.

---

### 10) Market Regime Detection (Clustering)

- **Enable clustering-based regimes**: runs KMeans on chosen features.
- **Regime features**: columns used for clustering (volatility, volume, returns, etc.).
- **Regime clusters**: number of regimes to discover.
- Click **Compute regimes**.

Outputs:
- Regime distribution.
- Regime-aware metrics per model.
- Regime plots with highlighted spans.

Regime naming is based on average realized volatility:
- Low Vol / Mean-Reverting
- Normal Vol / Mixed
- High Vol / Risk-Off
- Extreme Vol / Crisis (if >3 clusters)

---

### 11) Model Signal Plots

For classification targets only:
- Each model gets its own price chart.
- Predicted signal points are plotted on the price.
- **Indicators to plot**: overlay these series under the price chart.

---

### 12) Model Regime Plots

- If regimes are computed, each model gets its own chart.
- Regimes are highlighted as shaded background regions.

---
