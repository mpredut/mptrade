# server.py
import os
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="/static"), name="static")
#app.mount("/files", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="files")

# The service currently allows every origin, method, and header. This is a runtime
# configuration fact, not an assertion that the policy is safe for public exposure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept requests from every origin.
    allow_credentials=True,
    allow_methods=["*"],  # Accept every HTTP method.
    allow_headers=["*"],  # Accept every request header.
)


from binance.client import Client
from binance.exceptions import BinanceAPIException

from keys.apikeys import api_key, api_secret

# Local trading integrations.
from binance_api import bapi as api
from providers.market_api import api as mkt   # proxy unic guardat (Instrument.place)
import log





@app.get("/")
def read_root():
    headers = {"ngrok-skip-browser-warning": "true"}
    index_path = "index.html"
    
    if os.path.exists(index_path):
        return FileResponse(index_path, headers=headers)
    else:
        return Response(content="index.html not found", headers=headers, status_code=404)

    
# Request payloads.
class TradeRequest(BaseModel):
    symbol: str
    amount: float

class AlertRequest(BaseModel):
    symbol: str
    threshold: float
    direction: str  # Expected values are "up" or "down"; they are not validated here.

# Trading endpoints attempt a guarded submission. Their current response text does
# not prove that an order was accepted or filled because the return value is ignored.
@app.post("/trade/sell")
async def sell(request: TradeRequest):
    # Submit a guarded limit sell one percent above the current Binance price.
    print(f"Sold {request.amount} of {request.symbol}")
    current_price = api.get_current_price(str(request.symbol))
    sell_price = current_price * (1 + 0.01 )
    print(f"BTC price {current_price} {sell_price}")
    mkt.place(str(request.symbol), "SELL", sell_price, request.amount)   # proxy unic guardat
    return {"message": f"Sold {request.amount} of {request.symbol}"}

@app.post("/trade/buy")
async def buy(request: TradeRequest):
    # Submit a guarded limit buy one percent below the current Binance price.
    print(f"Bought {request.amount} of {request.symbol}")
    current_price = api.get_current_price(str(request.symbol))
    sell_price = current_price * (1 - 0.01 )
    print(f"BTC price {current_price} {sell_price}")
    mkt.place(str(request.symbol), "BUY", sell_price, request.amount)   # proxy unic guardat
    return {"message": f"Bought {request.amount} of {request.symbol}"}

@app.get("/status/get")
async def get_status(symbol: str):
    # Placeholder response; this endpoint does not inspect market or process state.
    return {"symbol": symbol, "status": "Stable"}

@app.post("/alert/set")
async def set_alert(request: AlertRequest):
    # Placeholder response; this endpoint does not persist or schedule an alert.
    return {
        "message": f"Alert set for {request.symbol}: {request.direction} at {request.threshold}"
    }
