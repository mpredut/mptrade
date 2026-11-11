"""A manual simulation; it is not part of the live runtime."""

from collections import deque
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from offline.simulations.sim import PriceWindow
import matplotlib.pyplot as plt  # noqa: E402


def plot_graphs(price_window, full_prices):
    """Plot the charts with every price and the current window."""
    # Clear previous charts.
    plt.clf()

    # Plot full chart.
    plt.subplot(2, 1, 1)
    plt.plot(full_prices, marker='o', linestyle='-', label='All Prices')
    plt.title('All Prices (Last 500)')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Plot sliding window chart.
    plt.subplot(2, 1, 2)
    price_window.plot_window(label='Window Prices', clear_and_pause=False, plt_mod=plt)

    # Update charts.
    plt.tight_layout()
    plt.pause(0.1)


def main():
    window_size = 10
    max_full_prices = 500
    price_window = PriceWindow(window_size)
    full_prices = deque(maxlen=max_full_prices)  # We keep up to 500 values.
    random.seed(42)

    plt.ion()  # Enable interactive mode so the charts can be watched in real time.

    try:
        while True:
            price = random.uniform(1, 100)
            full_prices.append(price)  # Add the price to the full chart.
            price_window.process_price(price)
            plot_graphs(price_window, full_prices)
            time.sleep(1.2)  # A small delay to simulate real time.
    except KeyboardInterrupt:
        print("Simulation stopped manually.")


if __name__ == "__main__":
    main()
