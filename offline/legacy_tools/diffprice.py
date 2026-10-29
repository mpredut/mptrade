"""A historical manual experiment; it may reach the real API."""

from binance.client import Client
from keys.apikeys import api_key, api_secret


# Initialize the client for Binance.com
client_com = Client(api_key, api_secret)

# Initialize the client for Binance.us
client_us = Client(api_key, api_secret, tld='us')

# Get price from Binance.com
price_com = client_com.get_symbol_ticker(symbol='BTCUSDT')

# Get price from Binance.us
price_us = client_us.get_symbol_ticker(symbol='BTCUSDT')

print(f"Price on Binance.com: {price_com['price']}")
print(f"Price on Binance.us: {price_us['price']}")
