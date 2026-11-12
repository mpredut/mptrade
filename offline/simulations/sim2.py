"""A manual simulation; it is not part of the live runtime."""

from collections import deque
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from offline.simulations.sim import PriceWindow, run_interactive_simulation
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["PriceWindow", "main", "plot_graphs", "run_interactive_simulation"]


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
    full_prices = deque(maxlen=500)  # We keep up to 500 values.
    run_interactive_simulation(
        10,
        sleep_sec=1.2,
        on_tick_fn=lambda price_window, price: (
            full_prices.append(price),
            plot_graphs(price_window, full_prices),
        ),
    )


if __name__ == "__main__":
    main()
