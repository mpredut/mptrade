"""A manual WebSocket diagnostic; it may reach the real API."""

import asyncio
import requests
from binance import AsyncClient, BinanceSocketManager
from keys.apikeys import api_key, api_secret

def get_listen_key(api_key):
    resp = requests.post(
        "https://api3.binance.com/api/v3/userDataStream",
        headers={"X-MBX-APIKEY": api_key}
    )
    resp.raise_for_status()
    return resp.json()["listenKey"]


async def test_ws():
    # A separate AsyncClient — do not reuse the sync client.
    #async_client = await AsyncClient.create(
    #    api_key=api.client.API_KEY,
    #    api_secret=api.client.API_SECRET,
    #)

    get_listen_key(api_key)

    async_client = await AsyncClient.create(
        api_key=api_key,
        api_secret=api_secret,
        requests_params={"timeout": 20},
        tld="com",
    )
    
    bm = BinanceSocketManager(async_client)
    
    print("Conectare la User Data Stream...")
    async with bm.user_socket() as stream:
        print("✅ Connected! Place an order from the app now...")
        while True:
            event = await stream.recv()
            print(f"EVENT: {event}")

asyncio.run(test_ws())
