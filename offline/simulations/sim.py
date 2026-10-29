"""A manual simulation; it is not part of the live runtime."""

import os
import sys
import matplotlib.pyplot as plt
import random
import time
from collections import deque

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import safe_float, safe_nonneg_float, safe_pos_int  # noqa: E402


class PriceWindow:
    def __init__(self, window_size, max_index=1000000, epsilon=1e-5):
        ws = safe_pos_int(window_size, default=None) if isinstance(window_size, (int, float)) else None
        if ws is None:
            raise ValueError("window_size must be a positive integer")
        mi = safe_pos_int(max_index, default=None) if isinstance(max_index, (int, float)) else None
        if mi is None:
            raise ValueError("max_index must be a positive integer")
        eps = safe_nonneg_float(epsilon, default=None) if isinstance(epsilon, (int, float)) else None
        if eps is None:
            raise ValueError("epsilon must be a non-negative finite number")
        self.window_size = ws
        self.prices = deque()  # Keeps every price in the window.
        self.min_deque = deque()  # Gestionarea minimului
        self.max_deque = deque()  # Gestionarea maximului
        self.current_index = 0  # An internal counter tracking the index.
        self.max_index = mi  # The threshold at which normalisation happens.
        self.epsilon = eps  # The tolerance for approximately equal lows.

    def _evict_expired(self, dq: deque) -> None:
        cutoff = self.current_index - self.window_size
        while dq and dq[0][0] <= cutoff:
            dq.popleft()

    def normalize_indices(self):
        """Normalise the indices when max_index is reached."""
        min_index = min(
            (d[0][0] for d in (self.min_deque, self.max_deque) if d),
            default=self.current_index,
        )
        self.min_deque = deque([(index - min_index, price) for index, price in self.min_deque])
        self.max_deque = deque([(index - min_index, price) for index, price in self.max_deque])
        self.current_index -= min_index  # Ajustam indexul curent

    def process_price(self, price):
        px = safe_float(price, default=None)
        if px is None:
            return

        # Add the new price to the price list.
        self.prices.append(px)

        # Remove the prices that fall out of the window.
        if len(self.prices) > self.window_size:
            self.prices.popleft()

        # Remove expired elements from both deques and normalize once before inserting.
        self._evict_expired(self.min_deque)
        self._evict_expired(self.max_deque)
        if self.current_index >= self.max_index:
            self.normalize_indices()

        # Handling of the current low and high.
        self._manage_minimum(px)
        self._manage_maximum(px)

        # Incrementam indexul intern
        self.current_index += 1

    def _manage_minimum(self, price):
        """Handling of the current low in the window."""
        self._evict_expired(self.min_deque)

        # Check whether the current price is approximately equal to any price already in `min_deque`.
        for index, existing_price in self.min_deque:
            if abs(existing_price - price) <= self.epsilon:
                return  # Do not add the current price if an equivalent already exists.
        
        # Remove the trailing elements that are larger than the current price.
        while self.min_deque and self.min_deque[-1][1] > price:
            self.min_deque.pop()

        # Add the current price.
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

    def get_prices(self):
        """Return every current price in the window."""
        return list(self.prices)

    def plot_window(self):
        """Plot the values in the window and mark the low and the high."""
        prices = self.get_prices()
        min_price, min_pos = self.get_min()
        max_price, max_pos = self.get_max()
        
        plt.clf()  # Clear the current figure to prepare the new chart.
        plt.plot(prices, marker='o', linestyle='-', label='Prices')
        
        
        plt.title('Price Window with Min and Max')
        plt.xlabel('Relative Position in Window')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)  # A short pause so the chart updates.


def main():
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
        print("Simulare oprita manual.")


if __name__ == "__main__":
    main()
