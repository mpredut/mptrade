"""A manual WebSocket diagnostic; it may reach the real API."""

import asyncio
import requests
import websockets
import json
from keys.apikeys import api_key

def get_listen_key(api_key):
    resp = requests.post(
        "https://api.binance.com/api/v3/userDataStream",
        headers={"X-MBX-APIKEY": api_key}
    )
    resp.raise_for_status()
    return resp.json()["listenKey"]

def keepalive_listen_key(api_key, listen_key):
    requests.put(
        "https://api.binance.com/api/v3/userDataStream",
        headers={"X-MBX-APIKEY": api_key},
        params={"listenKey": listen_key}
    )

async def test_ws():
    
    listen_key = get_listen_key(api_key)
    print(f"Listen key: {listen_key[:10]}...")
    
    url = f"wss://stream.binance.com:9443/ws/{listen_key}"
    async with websockets.connect(url, ping_interval=20) as ws:
        print("✅ Connected! Place an order from the app now...")
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                print(f"EVENT: {json.loads(msg)}")
            except asyncio.TimeoutError:
                print("... waiting ...")
                keepalive_listen_key(api_key, listen_key)

asyncio.run(test_ws())
