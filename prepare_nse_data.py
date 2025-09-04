#!/usr/bin/env python3
"""
prepare_nse_data.py
Fetches Indian stock data (NSE) using yfinance and builds a clean feature set.
Outputs: data/nse_{ticker}_features.csv
Run: python prepare_nse_data.py --ticker RELIANCE.NS --start 2015-01-01 --end 2025-09-01
"""
import argparse
import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

# Suppress yfinance FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return_1d"] = out["Close"].pct_change()
    out["SMA_5"] = out["Close"].rolling(5).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["RSI_14"] = rsi(out["Close"], 14)
    macd_line, signal_line, hist = macd(out["Close"])
    out["MACD"] = macd_line
    out["MACD_Signal"] = signal_line
    out["MACD_Hist"] = hist
    out["Vol_Change"] = out["Volume"].pct_change()
    # Next-day targets
    out["Target_Close"] = out["Close"].shift(-1)
    out["Target_Return_1d"] = out["Return_1d"].shift(-1)
    out = out.dropna().copy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="RELIANCE.NS", help="NSE ticker (e.g., RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS)")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = yf.download(args.ticker, start=args.start, end=args.end, progress=False, auto_adjust=False)
    if df.empty:
        raise SystemExit(f"No data returned for {args.ticker}. Check ticker or dates.")
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        expected_cols.insert(4, "Adj Close")
    df = df[expected_cols].copy()
    feats = build_features(df)
    outpath = os.path.join(args.outdir, f"nse_{args.ticker.replace('.','_')}_features.csv")
    feats.to_csv(outpath, index=True)
    print(f"Saved: {outpath}  | rows={len(feats)} cols={feats.shape[1]}")

if __name__ == "__main__":
    main()