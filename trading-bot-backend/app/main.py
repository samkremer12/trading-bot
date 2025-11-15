from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import ccxt
from cryptography.fernet import Fernet
from datetime import datetime
import json
import os
from enum import Enum

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ExchangeType(str, Enum):
    BINANCE = "binance"
    KUCOIN = "kucoin"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    COINBASE = "coinbase"

class TradingState:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self.exchange_type: Optional[str] = None
        self.exchange: Optional[Any] = None
        self.auto_trading_enabled: bool = False
        self.last_webhook: Optional[Dict] = None
        self.last_order: Optional[Dict] = None
        self.orders_history: List[Dict] = []
        self.total_pnl: float = 0.0
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def encrypt(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, data: bytes) -> str:
        return self.cipher.decrypt(data).decode()

state = TradingState()

class ApiKeyRequest(BaseModel):
    api_key: str
    api_secret: str
    exchange: ExchangeType

class WebhookAlert(BaseModel):
    action: str
    symbol: str
    price: Optional[str] = None
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    quantity: Optional[float] = None

class OrderRequest(BaseModel):
    symbol: str
    side: str
    amount: float
    price: Optional[float] = None
    order_type: str = "market"

class CloseOrderRequest(BaseModel):
    symbol: str

class ToggleTradingRequest(BaseModel):
    enabled: bool

def get_exchange_instance(exchange_type: str, api_key: str, api_secret: str):
    exchange_classes = {
        "binance": ccxt.binance,
        "kucoin": ccxt.kucoin,
        "kraken": ccxt.kraken,
        "bybit": ccxt.bybit,
        "coinbase": ccxt.coinbase,
    }
    
    if exchange_type not in exchange_classes:
        raise ValueError(f"Unsupported exchange: {exchange_type}")
    
    exchange_class = exchange_classes[exchange_type]
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })
    
    return exchange

def normalize_symbol(symbol: str, exchange_type: str) -> str:
    symbol = symbol.upper().replace("USDT", "").replace("USD", "")
    
    if exchange_type == "kraken":
        symbol_map = {
            "BTC": "XBT/USDT",
            "ETH": "ETH/USDT",
            "SOL": "SOL/USDT",
            "XRP": "XRP/USDT",
        }
        return symbol_map.get(symbol, f"{symbol}/USDT")
    elif exchange_type == "coinbase":
        return f"{symbol}-USDT"
    else:
        return f"{symbol}/USDT"

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/set-api-key")
async def set_api_key(request: ApiKeyRequest):
    try:
        encrypted_key = state.encrypt(request.api_key)
        encrypted_secret = state.encrypt(request.api_secret)
        
        exchange = get_exchange_instance(
            request.exchange.value,
            request.api_key,
            request.api_secret
        )
        
        await exchange.load_markets()
        
        state.api_key = encrypted_key
        state.api_secret = encrypted_secret
        state.exchange_type = request.exchange.value
        state.exchange = exchange
        
        return {
            "success": True,
            "message": "API credentials saved and verified successfully",
            "exchange": request.exchange.value
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to verify API credentials: {str(e)}")

@app.get("/webhook-url")
async def get_webhook_url(request: Request):
    base_url = str(request.base_url).rstrip('/')
    webhook_url = f"{base_url}/webhook"
    return {"webhook_url": webhook_url}

@app.post("/toggle-trading")
async def toggle_trading(request: ToggleTradingRequest):
    state.auto_trading_enabled = request.enabled
    return {
        "success": True,
        "auto_trading_enabled": state.auto_trading_enabled
    }

@app.post("/webhook")
async def webhook(alert: WebhookAlert):
    try:
        state.last_webhook = {
            "timestamp": datetime.now().isoformat(),
            "payload": alert.dict()
        }
        
        if not state.auto_trading_enabled:
            return {
                "success": True,
                "message": "Webhook received but auto-trading is disabled",
                "executed": False
            }
        
        if not state.exchange:
            raise HTTPException(status_code=400, detail="Exchange not configured. Please set API credentials first.")
        
        symbol = normalize_symbol(alert.symbol, state.exchange_type)
        action = alert.action.lower()
        
        if action in ["buy", "long"]:
            side = "buy"
        elif action in ["sell", "short", "close"]:
            side = "sell"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
        
        amount = alert.quantity if alert.quantity else 0.001
        
        order = None
        if alert.price:
            price = float(alert.price)
            order = state.exchange.create_limit_order(symbol, side, amount, price)
        else:
            order = state.exchange.create_market_order(symbol, side, amount)
        
        state.last_order = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": alert.price,
            "order_id": order.get('id'),
            "status": order.get('status')
        }
        
        state.orders_history.append(state.last_order)
        
        if alert.stop_loss or alert.take_profit:
            pass
        
        return {
            "success": True,
            "message": "Order executed successfully",
            "order": state.last_order
        }
        
    except Exception as e:
        error_msg = f"Failed to execute order: {str(e)}"
        print(f"ERROR: {error_msg}")
        state.last_order = {
            "timestamp": datetime.now().isoformat(),
            "error": error_msg
        }
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/place-order")
async def place_order(request: OrderRequest):
    try:
        if not state.exchange:
            raise HTTPException(status_code=400, detail="Exchange not configured")
        
        symbol = normalize_symbol(request.symbol, state.exchange_type)
        
        if request.order_type == "market":
            order = state.exchange.create_market_order(symbol, request.side, request.amount)
        elif request.order_type == "limit" and request.price:
            order = state.exchange.create_limit_order(symbol, request.side, request.amount, request.price)
        else:
            raise HTTPException(status_code=400, detail="Invalid order type or missing price for limit order")
        
        state.last_order = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": request.side,
            "amount": request.amount,
            "price": request.price,
            "order_id": order.get('id'),
            "status": order.get('status')
        }
        
        state.orders_history.append(state.last_order)
        
        return {
            "success": True,
            "order": state.last_order
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")

@app.post("/close-order")
async def close_order(request: CloseOrderRequest):
    try:
        if not state.exchange:
            raise HTTPException(status_code=400, detail="Exchange not configured")
        
        symbol = normalize_symbol(request.symbol, state.exchange_type)
        
        balance = state.exchange.fetch_balance()
        
        base_currency = symbol.split('/')[0]
        amount = balance.get(base_currency, {}).get('free', 0)
        
        if amount > 0:
            order = state.exchange.create_market_order(symbol, 'sell', amount)
            
            state.last_order = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "side": "sell",
                "amount": amount,
                "order_id": order.get('id'),
                "status": order.get('status'),
                "type": "close_position"
            }
            
            state.orders_history.append(state.last_order)
            
            return {
                "success": True,
                "message": "Position closed successfully",
                "order": state.last_order
            }
        else:
            return {
                "success": False,
                "message": "No open position to close"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close position: {str(e)}")

@app.get("/status")
async def get_status():
    api_connected = state.exchange is not None
    
    pnl_summary = {
        "total_orders": len(state.orders_history),
        "total_pnl": state.total_pnl,
        "recent_orders": state.orders_history[-5:] if state.orders_history else []
    }
    
    return {
        "api_connected": api_connected,
        "exchange": state.exchange_type if api_connected else None,
        "auto_trading_enabled": state.auto_trading_enabled,
        "last_webhook": state.last_webhook,
        "last_order": state.last_order,
        "pnl_summary": pnl_summary
    }

@app.post("/test-webhook")
async def test_webhook():
    test_alert = WebhookAlert(
        action="buy",
        symbol="BTCUSDT",
        price="50000"
    )
    
    state.last_webhook = {
        "timestamp": datetime.now().isoformat(),
        "payload": test_alert.dict(),
        "test": True
    }
    
    return {
        "success": True,
        "message": "Test webhook received successfully",
        "webhook_data": state.last_webhook
    }
