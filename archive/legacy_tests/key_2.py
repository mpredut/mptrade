import requests
from keys.apikeys import api_key, api_secret

response = requests.post(
    'https://api.binance.com/api/v3/userDataStream',
    headers={'X-MBX-APIKEY': api_key}
)
print(response.status_code)
print(response.text)
print(response.headers)
