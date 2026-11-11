"""A historical offline experiment; it is not an automated test."""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from offline.legacy_tools.test import _compute_interval_drops, _klines_to_df, _thirty_day_window_ms

def fetch_data():
    # The API for the historical data (for example: the Binance API).
    start_time, end_time = _thirty_day_window_ms()

    # We adjust so the data is fetched in several requests, because Binance limits how much it returns.
    df_list = []
    while start_time < end_time:
        url = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={start_time}&endTime={end_time}&limit=1000'
        response = requests.get(url)
        data = response.json()
        
        if not data:
            break
        
        df = _klines_to_df(data)
        
        df_list.append(df)
        
        start_time = int(df.index[-1].timestamp() * 1000) + 60000  # Move to the next minute after the last timestamp.

    if not df_list:
        return _klines_to_df([])
    full_df = pd.concat(df_list)
    return full_df


def analyze_drops_with_distribution(df, show_plot: bool = False):
    _total_raw, drops = _compute_interval_drops(df, 2 * 60)
    valid_intervals = len(drops)
    if valid_intervals == 0:
        return 0, 0.0, 0, 0.0, 0, 0.0

    if show_plot and drops:
        # Plot the drop distribution
        plt.hist(drops, bins=50, edgecolor='black')
        plt.title('Distribution of percentage drops in 120 minutes')
        plt.xlabel('Drop percentage')
        plt.ylabel('Frequency')
        plt.show()

    count_drops_5 = sum(1 for drop in drops if drop >= 5)
    count_drops_3 = sum(1 for drop in drops if drop >= 3)
    count_drops_2 = sum(1 for drop in drops if drop >= 2)

    probability_5 = count_drops_5 / valid_intervals * 100
    probability_3 = count_drops_3 / valid_intervals * 100
    probability_2 = count_drops_2 / valid_intervals * 100

    return count_drops_5, probability_5, count_drops_3, probability_3, count_drops_2, probability_2


if __name__ == "__main__":
    # Fetch data and analyze
    df = fetch_data()
    # Analyze with distribution
    count_drops_5, probability_5, count_drops_3, probability_3, count_drops_2, probability_2 = analyze_drops_with_distribution(df, show_plot=True)

    print(f"Number of 5% drops in 120 minutes: {count_drops_5}")
    print(f"Probability of a 5% drop in 120 minutes: {probability_5:.2f}%")
    print(f"Number of 3% drops in 120 minutes: {count_drops_3}")
    print(f"Probability of a 3% drop in 120 minutes: {probability_3:.2f}%")
    print(f"Number of 2% drops in 120 minutes: {count_drops_2}")
    print(f"Probability of a 2% drop in 120 minutes: {probability_2:.2f}%")

