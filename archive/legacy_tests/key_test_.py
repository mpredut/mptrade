from binance.client import Client
from keys.apikeys import api_key, api_secret

client = Client(api_key, api_secret)

print(client.stream_get_listen_key())
