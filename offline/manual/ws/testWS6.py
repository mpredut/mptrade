"""A manual WebSocket diagnostic; it may reach the real API."""

import base64, time, json, asyncio, websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import load_pem_private_key

api_key="-----BEGIN PUBLIC KEY-----MCowBQYDK2VwAyEAskWLaewXW9FEgnEBysc1kaJHZ/aBiXjZ4WDidcvL/b8=-----END PUBLIC KEY-----"

with open("ed25519_private.pem", "rb") as f:
    private_key = load_pem_private_key(f.read(), password=None, backend=default_backend())

#with open("ed25519_private.pem", "rb") as f:
    #private_key = load_pem_private_key(f.read(), password=None)


def sign_ed25519(payload: str) -> str:
    sig = private_key.sign(payload.encode())
    return base64.b64encode(sig).decode()

async def test_ws():
    url = "wss://ws-api.binance.com:443/ws-api/v3"
    async with websockets.connect(url, ping_interval=20) as ws:
        print("✅ Conectat!")

        timestamp = int(time.time() * 1000)
        params_str = f"apiKey={api_key}&timestamp={timestamp}"
        signature = sign_ed25519(params_str)

        login = {
            "id": "login",
            "method": "session.logon",
            "params": {
                "apiKey": api_key,
                "timestamp": timestamp,
                "signature": signature
            }
        }
        await ws.send(json.dumps(login))
        resp = json.loads(await ws.recv())
        print(f"Login: {resp.get('status')}")

        if resp.get("status") == 200:
            await ws.send(json.dumps({"id": "sub", "method": "userDataStream.subscribe"}))
            print("Subscribed! Place an order...")
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    print(f"EVENT: {json.loads(msg)}")
                except asyncio.TimeoutError:
                    print("... waiting ...")

asyncio.run(test_ws())
