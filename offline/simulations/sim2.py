"""A manual simulation; it is not part of the live runtime."""

from collections import deque
import os
import random
import sys
import time
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from offline.simulations.sim import PriceWindow


def plot_graphs(price_window, full_prices):
    """Plot the charts with every price and the current window."""
    # Curatam graficele anterioare
    plt.clf()

    # Plotam graficul complet
    plt.subplot(2, 1, 1)
    plt.plot(full_prices, marker='o', linestyle='-', label='All Prices')
    plt.title('All Prices (Last 500)')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.grid(True)

    # Plotam graficul ferestrei glisante
    plt.subplot(2, 1, 2)
    prices = price_window.get_prices()
    min_price, min_pos = price_window.get_min()
    max_price, max_pos = price_window.get_max()

    plt.plot(prices, marker='o', linestyle='-', label='Window Prices')
    
    if min_price is not None:
        plt.plot(min_pos, min_price, 'ro', label=f'Min: {min_price}')
    
    if max_price is not None:
        plt.plot(max_pos, max_price, 'go', label=f'Max: {max_price}')
    
    plt.title('Price Window with Min and Max')
    plt.xlabel('Relative Position in Window')
    plt.ylabel('Price')
    plt.grid(True)
    
    # Actualizam graficele
    plt.tight_layout()
    plt.pause(0.1)


def main():
    window_size = 10
    max_full_prices = 500
    price_window = PriceWindow(window_size)
    full_prices = deque(maxlen=max_full_prices)  # We keep up to 500 values.

    plt.ion()  # Enable interactive mode so the charts can be watched in real time.

    try:
        while True:
            price = random.uniform(1, 100)
            full_prices.append(price)  # Add the price to the full chart.
            price_window.process_price(price)
            plot_graphs(price_window, full_prices)
            time.sleep(1.2)  # A small delay to simulate real time.
    except KeyboardInterrupt:
        print("Simulare oprita manual.")


if __name__ == "__main__":
    main()
