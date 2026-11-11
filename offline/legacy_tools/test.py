"""A historical offline experiment; it is not an automated test."""

import datetime
import os
import sys
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import safe_nonneg_float, safe_pos_float

KLINE_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore',
]


def _klines_to_df(data):
    df = pd.DataFrame(data, columns=KLINE_COLUMNS)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df


def fetch_data():
    # The API for the historical data (for example: the Binance API).
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - 30 * 24 * 60 * 60 * 1000  # Last 30 days

    url = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={start_time}&endTime={end_time}'
    response = requests.get(url)
    data = response.json()

    # Create a DataFrame
    df = _klines_to_df(data)
    print(df.head())
    print(f"Total intervals (should be close to 43200 for 30 days of minute data): {len(df)}")
    return df

def _compute_interval_drops(df, lag_bars: int) -> tuple[int, list[float]]:
    total_intervals = max(0, len(df) - lag_bars)
    if total_intervals == 0:
        return 0, []
    drops = []
    for i in range(total_intervals):
        start_price = safe_pos_float(df['close'].iloc[i], default=None)
        end_price = safe_nonneg_float(df['close'].iloc[i + lag_bars], default=None)
        if start_price is None or end_price is None:
            continue
        drops.append((start_price - end_price) / start_price * 100)
    return total_intervals, drops


def analyze_drops(df):
    total_intervals, drops = _compute_interval_drops(df, 20)
    if total_intervals == 0:
        return 0, 0, 0.0
    count_drops = sum(1 for drop in drops if drop >= 5)
    probability = count_drops / total_intervals * 100
    return count_drops, total_intervals, probability


if __name__ == "__main__":
    # Fetch data and analyze
    df = fetch_data()
    count_drops, total_intervals, probability = analyze_drops(df)

    print(f"Number of 5% drops in 20 minutes: {count_drops}")
    print(f"Total 20-minute intervals in 30 days: {total_intervals}")
    print(f"Probability of a 5% drop in 20 minutes: {probability:.2f}%")
