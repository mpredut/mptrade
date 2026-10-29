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

    # We adjust so the data is fetched in several requests, because Binance limits how much it returns.
    df_list = []
    while start_time < end_time:
        url = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={start_time}&endTime={end_time}&limit=1000'
        response = requests.get(url)
        data = response.json()
        
        if not data:
            break
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].astype(float)
        
        df_list.append(df)
        
        start_time = int(df.index[-1].timestamp() * 1000) + 60000  # Move to the next minute after the last timestamp.
        
    full_df = pd.concat(df_list)
    return full_df

import matplotlib.pyplot as plt

def analyze_drops_with_distribution(df, show_plot: bool = False):
    drops = []
    total_intervals = max(0, len(df) - 2 * 60)  # Intervale de 2*60 de minute
    if total_intervals == 0:
        return 0, 0.0, 0, 0.0, 0, 0.0

    for i in range(total_intervals):
        start_price = safe_pos_float(df['close'].iloc[i], default=None)
        end_price = safe_nonneg_float(df['close'].iloc[i + 2 * 60], default=None)
        if start_price is None or end_price is None:
            continue
        drop_percent = (start_price - end_price) / start_price * 100
        drops.append(drop_percent)

    if show_plot and drops:
        # Verificam distributia scaderilor
        plt.hist(drops, bins=50, edgecolor='black')
        plt.title('Distributia scaderilor procentuale in 2*60 de minute')
        plt.xlabel('Procent de scadere')
        plt.ylabel('Frecventa')
        plt.show()

    count_drops_5 = sum(1 for drop in drops if drop >= 5)
    count_drops_3 = sum(1 for drop in drops if drop >= 3)
    count_drops_2 = sum(1 for drop in drops if drop >= 2)

    probability_5 = count_drops_5 / total_intervals * 100
    probability_3 = count_drops_3 / total_intervals * 100
    probability_2 = count_drops_2 / total_intervals * 100

    return count_drops_5, probability_5, count_drops_3, probability_3, count_drops_2, probability_2


if __name__ == "__main__":
    # Fetch data and analyze
    df = fetch_data()
    # Analyze with distribution
    count_drops_5, probability_5, count_drops_3, probability_3, count_drops_2, probability_2 = analyze_drops_with_distribution(df, show_plot=True)

    print(f"Numar de scaderi de 5% in 20 de minute: {count_drops_5}")
    print(f"Probabilitatea unei scaderi de 5% in 20 de minute: {probability_5:.2f}%")
    print(f"Numar de scaderi de 3% in 20 de minute: {count_drops_3}")
    print(f"Probabilitatea unei scaderi de 3% in 20 de minute: {probability_3:.2f}%")
    print(f"Numar de scaderi de 2% in 20 de minute: {count_drops_2}")
    print(f"Probabilitatea unei scaderi de 2% in 20 de minute: {probability_2:.2f}%")

