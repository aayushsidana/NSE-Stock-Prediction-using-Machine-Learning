#NSE Stock Prediction (India-focused)
-This project shows a basic way to use machine learning on Indian stock data. It fetches prices from NSE, creates simple technical indicators, and trains models to predict the next day’s return.

#What it does
-Downloads stock data from NSE using yfinance (examples: RELIANCE.NS, TCS.NS, INFY.NS).
-Builds features like SMA, EMA, RSI, and MACD.
-Trains two models: Linear Regression and Random Forest.
-Saves results as metrics and a prediction plot.

#How to run
pip install -r requirements.txt
python prepare_nse_data.py --ticker RELIANCE.NS --start 2015-01-01 --end 2025-09-01
python train_nse_model.py --data data/nse_RELIANCE_NS_features.csv --target Target_Return_1d

#What you get
-Cleaned dataset: data/nse_RELIANCE_NS_features.csv
-Model results: outputs/metrics.json
-Prediction vs actual plot: outputs/pred_vs_actual.png