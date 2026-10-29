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

def fetch_data():
    # The API for the historical data (for example: the Binance API).
    end_time = int(datetime.datetime.now().timestamp() * 1000)
    start_time = end_time - 30 * 24 * 60 * 60 * 1000  # Ultimele 30 de zile

    url = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={start_time}&endTime={end_time}'
    response = requests.get(url)
    data = response.json()

    # Crearea unui DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    print(df.head())
    print(f"Total intervals (should be close to 43200 for 30 days of minute data): {len(df)}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df

def analyze_drops(df):
    count_drops = 0
    total_intervals = max(0, len(df) - 20)  # Intervale de 20 de minute
    if total_intervals == 0:
        return 0, 0, 0.0

    for i in range(total_intervals):
        start_price = safe_pos_float(df['close'].iloc[i], default=None)
        end_price = safe_nonneg_float(df['close'].iloc[i + 20], default=None)
        if start_price is None or end_price is None:
            continue
        drop_percent = (start_price - end_price) / start_price * 100

        if drop_percent >= 5:
            count_drops += 1

    probability = count_drops / total_intervals * 100
    return count_drops, total_intervals, probability


if __name__ == "__main__":
    # Fetch data and analyze
    df = fetch_data()
    count_drops, total_intervals, probability = analyze_drops(df)

    print(f"Numar de scaderi de 5% in 20 de minute: {count_drops}")
    print(f"Total intervale de 20 de minute in 30 de zile: {total_intervals}")
    print(f"Probabilitatea unei scaderi de 5% in 20 de minute: {probability:.2f}%")
