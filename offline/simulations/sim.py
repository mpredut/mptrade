"""A manual simulation; it is not part of the live runtime."""

import os
import random
import sys
import time
from collections import deque

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import _parse_bool_flag, exact_pos_int, safe_float, safe_nonneg_float  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


class PriceWindow:
    def __init__(self, window_size, max_index=1000000, epsilon=1e-5):
        ws = exact_pos_int(window_size, default=None)
        if ws is None:
            raise ValueError("window_size must be a positive integer")
        mi = exact_pos_int(max_index, default=None)
        if mi is None:
            raise ValueError("max_index must be a positive integer")
        eps = (
            safe_nonneg_float(epsilon, default=None)
            if isinstance(epsilon, (int, float)) and not isinstance(epsilon, bool)
            else None
        )
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

    def normalize_indices(self):
        """Normalise the indices when max_index is reached."""
        window_start = max(0, self.current_index - len(self.prices))
        self._evict_before(self.min_deque, window_start)
        self._evict_before(self.max_deque, window_start)
        if window_start <= 0:
            return
        self.min_deque = deque([(index - window_start, price) for index, price in self.min_deque])
        self.max_deque = deque([(index - window_start, price) for index, price in self.max_deque])
        self.current_index -= window_start  # Adjust the current index.

    def process_price(self, price):
        px = safe_float(price, default=None)
        if px is None:
            return

        if self.current_index >= max(self.max_index, self.window_size + 1):
            self.normalize_indices()

        # Add the new price to the price list.
        self.prices.append(px)

        # Remove the prices that fall out of the window.
        if len(self.prices) > self.window_size:
            self.prices.popleft()

        # Handling of the current low and high.
        min_valid_idx = self.current_index - self.window_size + 1
        self._evict_before(self.min_deque, min_valid_idx)
        while self.min_deque and self.min_deque[-1][1] > px:
            self.min_deque.pop()
        self.min_deque.append((self.current_index, px))
        self._evict_before(self.max_deque, min_valid_idx)
        while self.max_deque and self.max_deque[-1][1] <= px:
            self.max_deque.pop()
        self.max_deque.append((self.current_index, px))

        # Increment the internal index.
        self.current_index += 1

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

    def plot_window(self, *, label: str = 'Prices', clear_and_pause: bool = True, plt_mod=None):
        """Plot the values in the window and mark the low and the high."""
        p = plt_mod or plt
        do_clear_pause = _parse_bool_flag(clear_and_pause, True)
        prices = list(self.prices)
        min_price, min_pos = self.get_min()
        max_price, max_pos = self.get_max()

        if do_clear_pause:
            p.clf()  # Clear the current figure to prepare the new chart.
        p.plot(prices, marker='o', linestyle='-', label=label)
        if min_price is not None:
            p.plot(min_pos, min_price, 'ro', label=f'Min: {min_price:.2f}')
        if max_price is not None:
            p.plot(max_pos, max_price, 'go', label=f'Max: {max_price:.2f}')

        p.title('Price Window with Min and Max')
        p.xlabel('Relative Position in Window')
        p.ylabel('Price')
        p.legend()
        p.grid(True)
        if do_clear_pause:
            p.pause(0.1)  # A short pause so the chart updates.


def run_interactive_simulation(
    window_size: int = 10,
    *,
    sleep_sec: float = 7.5,
    on_tick_fn=None,
) -> None:
    price_window = PriceWindow(window_size)
    sleep_s = safe_nonneg_float(sleep_sec, default=7.5)
    if sleep_s is None:
        sleep_s = 7.5
    random.seed(42)

    plt.ion()  # Enable interactive mode so the charts can be watched in real time.
    try:
        while True:
            price = random.uniform(1, 100)
            price_window.process_price(price)
            if on_tick_fn is not None:
                on_tick_fn(price_window, price)
            else:
                price_window.plot_window()
            time.sleep(sleep_s)  # A small delay to simulate real time.
    except KeyboardInterrupt:
        print("Simulation stopped manually.")


def main():
    run_interactive_simulation(10, sleep_sec=7.5)


if __name__ == "__main__":
    main()
