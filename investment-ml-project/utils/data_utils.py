"""
Data utilities for downloading, loading, and saving stock data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import List, Optional, Union


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : List[str]
        List of stock tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str, optional
        End date (YYYY-MM-DD). If None, uses today.
    interval : str
        Data interval ('1d', '1wk', '1mo')
    
    Returns:
    --------
    pd.DataFrame : OHLCV data with MultiIndex (Date, Ticker)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    data_dict = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            if not df.empty:
                df['Ticker'] = ticker
                data_dict[ticker] = df
                print(f"✓ Downloaded {ticker}: {len(df)} rows")
            else:
                print(f"✗ No data for {ticker}")
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {e}")
    
    if not data_dict:
        raise ValueError("No data downloaded. Check tickers and dates.")
    
    # Combine all data
    all_data = pd.concat(data_dict.values())
    all_data = all_data.reset_index()
    all_data = all_data.set_index(['Date', 'Ticker'])
    
    return all_data


def save_raw_data(data: pd.DataFrame, filepath: str):
    """Save raw data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath)
    print(f"Saved raw data to {filepath}")


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    df = pd.read_csv(filepath, index_col=[0, 1], parse_dates=True)
    return df


def clean_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data: handle missing values, outliers, etc.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with MultiIndex (Date, Ticker)
    
    Returns:
    --------
    pd.DataFrame : Cleaned data
    """
    # Forward fill missing values (within each ticker)
    data = data.groupby(level='Ticker').ffill()
    
    # Drop rows with all NaN
    data = data.dropna(how='all')
    
    # Remove outliers (prices that are > 3 std dev from mean)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in data.columns:
            mean = data[col].mean()
            std = data[col].std()
            data = data[(data[col] >= mean - 3*std) & (data[col] <= mean + 3*std)]
    
    return data


def compute_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Compute returns from prices.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data (Close prices)
    method : str
        'simple' or 'log'
    
    Returns:
    --------
    pd.DataFrame : Returns
    """
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices).diff()
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    return returns.dropna()


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 30,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute rolling volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    window : int
        Rolling window size (days)
    annualize : bool
        If True, annualize volatility (multiply by sqrt(252))
    
    Returns:
    --------
    pd.DataFrame : Rolling volatility
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def pivot_to_wide_format(data: pd.DataFrame, value_col: str = 'Close') -> pd.DataFrame:
    """
    Pivot MultiIndex DataFrame to wide format (Date index, Ticker columns).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with MultiIndex (Date, Ticker)
    value_col : str
        Column to pivot
    
    Returns:
    --------
    pd.DataFrame : Wide format DataFrame
    """
    if isinstance(data.index, pd.MultiIndex):
        data_reset = data.reset_index()
        wide = data_reset.pivot(index='Date', columns='Ticker', values=value_col)
    else:
        wide = data.pivot_table(index=data.index, columns='Ticker', values=value_col)
    
    return wide


def save_processed_data(data: pd.DataFrame, filepath: str):
    """Save processed data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath)
    print(f"Saved processed data to {filepath}")


def load_processed_data(filepath: str) -> pd.DataFrame:
    """Load processed data from CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

