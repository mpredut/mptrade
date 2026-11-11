"""A manual simulation; it is not part of the live runtime."""

import os
import random
import sys
import time
from collections import deque

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import exact_pos_int, safe_float, safe_nonneg_float  # noqa: E402


class PriceWindow:
    def __init__(self, window_size, max_index=1000000, epsilon=1e-5):
        ws = exact_pos_int(window_size, default=None)
        if ws is None:
            raise ValueError("window_size must be a positive integer")
        mi = exact_pos_int(max_index, default=None)
        if mi is None:
            raise ValueError("max_index must be a positive integer")
        eps = safe_nonneg_float(epsilon, default=None) if isinstance(epsilon, (int, float)) else None
        if eps is None:
            raise ValueError("epsilon must be a non-negative finite number")
        self.window_size = ws
        self.prices = deque()  # Keeps every price in the window.
        self.min_deque = deque()  # Minimum tracking.
        self.max_deque = deque()  # Maximum tracking.
        self.current_index = 0  # An internal counter tracking the index.
        self.max_index = mi  # The threshold at which normalisation happens.
        self.epsilon = eps  # The tolerance for approximately equal lows.

    def _evict_before(self, dq: deque, min_valid_index: int) -> None:
        while dq and dq[0][0] < min_valid_index:
            dq.popleft()

    def _evict_expired(self, dq: deque) -> None:
        self._evict_before(dq, self.current_index - self.window_size + 1)

    def normalize_indices(self):
        """Normalise the indices when max_index is reached."""
        window_start = self.current_index - len(self.prices)
        self._evict_before(self.min_deque, window_start)
        self._evict_before(self.max_deque, window_start)
        min_index = min(
            (d[0][0] for d in (self.min_deque, self.max_deque) if d),
            default=self.current_index,
        )
        self.min_deque = deque([(index - min_index, price) for index, price in self.min_deque])
        self.max_deque = deque([(index - min_index, price) for index, price in self.max_deque])
        self.current_index -= min_index  # Adjust the current index.

    def process_price(self, price):
        px = safe_float(price, default=None)
        if px is None:
            return

        if self.current_index >= self.max_index:
            self.normalize_indices()

        # Add the new price to the price list.
        self.prices.append(px)

        # Remove the prices that fall out of the window.
        if len(self.prices) > self.window_size:
            self.prices.popleft()

        # Handling of the current low and high.
        self._manage_minimum(px)
        self._manage_maximum(px)

        # Increment the internal index.
        self.current_index += 1

    def _manage_minimum(self, price):
        """Handling of the current low in the window."""
        self._evict_expired(self.min_deque)

        # Remove trailing elements that are strictly larger beyond epsilon, while still
        # appending near-equal lows behind earlier lows so expiry of the old low promotes
        # the newer in-window low instead of losing it.
        while self.min_deque and self.min_deque[-1][1] > price + self.epsilon:
            self.min_deque.pop()

        self.min_deque.append((self.current_index, price))

    def _manage_maximum(self, price):
        """Handling of the current high in the window."""
        self._evict_expired(self.max_deque)

        # Remove the trailing elements that are smaller than the current price (to keep the latest high).
        while self.max_deque and self.max_deque[-1][1] <= price:
            self.max_deque.pop()

        # Add the current price.
        self.max_deque.append((self.current_index, price))

    def _deque_front(self, dq: deque):
        if not dq:
            return None, None
        idx, val = dq[0]
        return val, idx - (self.current_index - len(self.prices))

    def get_min(self):
        """Return the current low in the window and its relative position."""
        return self._deque_front(self.min_deque)

    def get_max(self):
        """Return the current high in the window and its relative position."""
        return self._deque_front(self.max_deque)

    def plot_window(self):
        """Plot the values in the window and mark the low and the high."""
        import matplotlib.pyplot as plt

        prices = list(self.prices)
        min_price, min_pos = self.get_min()
        max_price, max_pos = self.get_max()

        plt.clf()  # Clear the current figure to prepare the new chart.
        plt.plot(prices, marker='o', linestyle='-', label='Prices')
        if min_price is not None:
            plt.plot(min_pos, min_price, 'ro', label=f'Min: {min_price:.2f}')
        if max_price is not None:
            plt.plot(max_pos, max_price, 'go', label=f'Max: {max_price:.2f}')

        plt.title('Price Window with Min and Max')
        plt.xlabel('Relative Position in Window')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)  # A short pause so the chart updates.


def main():
    import matplotlib.pyplot as plt

    window_size = 10
    price_window = PriceWindow(window_size)
    random.seed(42)

    plt.ion()  # Enable interactive mode so the charts can be watched in real time.
    try:
        while True:
            price = random.uniform(1, 100)
            price_window.process_price(price)
            price_window.plot_window()
            time.sleep(7.5)  # A small delay to simulate real time.
    except KeyboardInterrupt:
        print("Simulation stopped manually.")


if __name__ == "__main__":
    main()
