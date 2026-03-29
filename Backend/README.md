# ML Investment Portfolio Rebalancer

## Project Goal & Scope — Volatility Prediction

The goal of this project is to build a machine learning system that predicts short-term volatility for selected stocks in the S&P 500, enabling data-driven portfolio rebalancing and risk-aware investment decisions. Using historical stock prices and engineered features such as rolling returns and technical indicators, the system applies classical ML models (Random Forest, XGBoost) and deep learning (LSTM) to forecast future volatility. Predicted volatilities feed into a portfolio optimization engine, which computes weight allocations using mean-variance optimization, risk parity, and ML-driven heuristics. The project includes backtesting of strategies, Monte Carlo simulations, and explainability via SHAP, all visualized in an interactive Streamlit dashboard.

## Project Structure

```
investment-ml-project/
├── README.md                      # Project description & goal
├── requirements.txt               # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                       # Raw downloaded OHLCV CSVs
│   ├── processed/                 # Cleaned & engineered features
│   └── features/                  # Feature matrix for ML
│
├── notebooks/                     # Jupyter notebooks (run in sequence)
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_train_baseline_models.ipynb
│   ├── 05_train_random_forest_xgboost.ipynb
│   ├── 06_train_lstm_tensorflow.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_portfolio_optimization.ipynb
│   ├── 09_backtesting_engine.ipynb
│   ├── 10_shap_analysis.ipynb
│   └── 11_dashboard_preparation.ipynb
│
├── models/
│   ├── saved/                     # Trained models
│   └── metrics/                   # Evaluation & optimization results
│
├── optimization/
│   ├── mpt_utils.py               # Mean-variance optimization helpers
│   ├── risk_parity.py
│   └── heuristic_optimizer.py
│
├── backtesting/
│   └── backtester.py              # Rolling backtest engine
│
├── utils/
│   ├── data_utils.py              # Data download/load/save functions
│   ├── feature_utils.py           # Feature engineering helpers
│   └── model_utils.py             # Model save/load utilities
│
├── dashboard/
│   └── streamlit_app.py           # Full interactive dashboard
│
└── reports/
    └── final_report.pdf           # Optional PDF report
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Notebooks

Execute the notebooks in sequential order (01-11):

1. **01_data_collection.ipynb** - Download and collect stock data
2. **02_feature_engineering.ipynb** - Create features for ML models
3. **03_eda_visualization.ipynb** - Exploratory data analysis
4. **04_train_baseline_models.ipynb** - Train baseline models
5. **05_train_random_forest_xgboost.ipynb** - Train ML models
6. **06_train_lstm_tensorflow.ipynb** - Train deep learning model
7. **07_model_evaluation.ipynb** - Evaluate and compare models
8. **08_portfolio_optimization.ipynb** - Portfolio optimization strategies
9. **09_backtesting_engine.ipynb** - Backtest strategies with transaction costs
10. **10_shap_analysis.ipynb** - SHAP explainability analysis
11. **11_dashboard_preparation.ipynb** - Prepare data for dashboard

### Running the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Features

### Strategies Implemented

1. **Buy & Hold** - Equal weight, monthly rebalancing
2. **Momentum** - Rank-based momentum with monthly rebalancing
3. **Mean Reversion** - Contrarian strategy with monthly rebalancing
4. **Risk Parity** - Inverse volatility weighting with monthly rebalancing
5. **Volatility Targeting** - Rolling volatility windows to scale exposure

### Key Features

- **Monthly Rebalancing** - Realistic rebalancing frequency
- **Transaction Costs** - Bid-ask spread + commission modeling
- **Turnover Calculation** - Measure trading activity
- **Comprehensive Metrics** - Sharpe ratio, max drawdown, annualized returns/volatility
- **Visualizations** - Cumulative returns, turnover, transaction costs, risk-return profiles

## Project Phases

- **Phase 1 (Day 1-3):** Data gathering and cleaning
- **Phase 2 (Day 4-6):** Exploratory analysis
- **Phase 3 (Day 7-15):** Modeling (baseline, ML, DL)
- **Phase 4 (Day 16-22):** Portfolio optimization
- **Phase 5 (Day 23-28):** Backtesting
- **Phase 6 (Day 29-31):** Explainability (SHAP)

## License

This project is for educational and demonstration purposes.

