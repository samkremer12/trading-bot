from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import ccxt
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import json
import os
from enum import Enum
import secrets
import hashlib
import logging
import time
import hmac
import base64
import urllib.parse
import httpx
import asyncio
import jwt
import bcrypt
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

APP_PASSWORD = "Samkremer12!!"
WEBHOOK_SECRET = "Samkremer12"
JWT_SECRET = os.environ.get("JWT_SECRET", "trading-bot-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 7
webhook_request_times = []

def hash_password_bcrypt(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password_bcrypt(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash"""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def hash_password(password: str) -> str:
    """Legacy SHA256 hash for backward compatibility"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str) -> bool:
    """Legacy password verification for backward compatibility"""
    return hash_password(plain_password) == hash_password(APP_PASSWORD)

def create_jwt_token(username: str) -> str:
    """Create a JWT token with 7-day expiration and username claim"""
    expiration = datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS)
    payload = {
        "exp": expiration,
        "iat": datetime.utcnow(),
        "sub": username
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_session(authorization: Optional[str] = Header(None)) -> str:
    """Verify JWT token and return username"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def check_rate_limit() -> bool:
    """Rate limit: max 10 requests per minute"""
    current_time = time.time()
    webhook_request_times[:] = [t for t in webhook_request_times if current_time - t < 60]
    
    if len(webhook_request_times) >= 10:
        return False
    
    webhook_request_times.append(current_time)
    return True

def calculate_position_size(balance: float, risk_percentage: float = 0.02) -> float:
    """Calculate position size based on account balance (default 2%)"""
    return balance * risk_percentage

def verify_webhook_secret(secret: str) -> bool:
    """Verify webhook secret token"""
    return secret == WEBHOOK_SECRET

def round_kraken_volume(symbol: str, volume: float) -> float:
    """Round volume to Kraken's minimum step for the symbol"""
    symbol = symbol.upper().replace("USDT", "").replace("USD", "")
    
    volume_steps = {
        "BTC": 0.00001,
        "ETH": 0.0001,
        "SOL": 0.01,
        "XRP": 0.1
    }
    
    step = volume_steps.get(symbol, 0.0001)
    
    rounded = round(volume / step) * step
    
    return max(rounded, step)

class KrakenClient:
    """Direct Kraken API client with proper HMAC-SHA512 authentication"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = "https://api.kraken.com"
        self.nonce_counter = int(time.time() * 1000)
        
    def _get_nonce(self) -> str:
        """Generate strictly increasing nonce"""
        self.nonce_counter += 1
        return str(self.nonce_counter)
    
    def _sign_request(self, path: str, data: Dict[str, Any]) -> str:
        """Generate HMAC-SHA512 signature for Kraken API"""
        nonce = data['nonce']
        postdata = urllib.parse.urlencode(data)
        encoded = (str(nonce) + postdata).encode()
        message = path.encode() + hashlib.sha256(encoded).digest()
        
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512
        )
        return base64.b64encode(signature.digest()).decode()
    
    async def _private_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to Kraken private API"""
        if data is None:
            data = {}
        
        data['nonce'] = self._get_nonce()
        path = f"/0/private/{endpoint}"
        url = f"{self.base_url}{path}"
        
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._sign_request(path, data),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, headers=headers)
            result = response.json()
            
            if result.get('error') and len(result['error']) > 0:
                error_msg = ', '.join(result['error'])
                raise Exception(f"Kraken API error: {error_msg}")
            
            return result.get('result', {})
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        return await self._private_request('Balance')
    
    async def add_order(self, pair: str, side: str, order_type: str, volume: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place an order on Kraken"""
        data = {
            'pair': pair,
            'type': side,
            'ordertype': order_type,
            'volume': str(volume)
        }
        
        if price and order_type == 'limit':
            data['price'] = str(price)
        
        return await self._private_request('AddOrder', data)
    
    async def cancel_order(self, txid: str) -> Dict[str, Any]:
        """Cancel an order"""
        return await self._private_request('CancelOrder', {'txid': txid})
    
    async def get_open_orders(self) -> Dict[str, Any]:
        """Get open orders"""
        return await self._private_request('OpenOrders')
    
    async def get_closed_orders(self) -> Dict[str, Any]:
        """Get closed orders"""
        return await self._private_request('ClosedOrders')
    
    def to_kraken_pair(self, symbol: str) -> str:
        """Convert standard symbol to Kraken pair format"""
        symbol = symbol.upper().replace("USDT", "").replace("USD", "")
        
        pair_map = {
            "BTC": "XBTUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XRP": "XRPUSDT"
        }
        
        return pair_map.get(symbol, f"{symbol}USDT")
    
    async def get_ticker_price(self, pair: str) -> float:
        """Get current price from Kraken public ticker"""
        url = f"{self.base_url}/0/public/Ticker"
        params = {'pair': pair}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            result = response.json()
            
            if result.get('error') and len(result['error']) > 0:
                error_msg = ', '.join(result['error'])
                raise Exception(f"Kraken ticker error: {error_msg}")
            
            ticker_data = result.get('result', {})
            if not ticker_data:
                raise Exception(f"No ticker data for {pair}")
            
            pair_key = list(ticker_data.keys())[0]
            price_data = ticker_data[pair_key]
            last_price = float(price_data['c'][0])
            
            return last_price

class ExchangeType(str, Enum):
    BINANCE = "binance"
    KUCOIN = "kucoin"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    COINBASE = "coinbase"

class UserState:
    """Per-user trading state"""
    def __init__(self, username: str):
        self.username: str = username
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self.exchange_type: Optional[str] = None
        self.exchange: Optional[Any] = None
        self.kraken_client: Optional[KrakenClient] = None
        self.api_connected: bool = False
        self.connection_error: Optional[str] = None
        self.auto_trading_enabled: bool = False
        self.last_webhook: Optional[Dict] = None
        self.last_order: Optional[Dict] = None
        self.orders_history: List[Dict] = []
        self.open_trades: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.webhook_logs: List[Dict] = []
        self.total_pnl: float = 0.0
        self.per_coin_pnl: Dict[str, float] = {}
        self.coin_trading_enabled: Dict[str, bool] = {
            "BTC": True,
            "ETH": True,
            "SOL": True,
            "XRP": True
        }
        self.emergency_stop: bool = False
        self.stop_loss_enabled: bool = True
        self.stop_loss_pct: float = 0.02
        self._stoploss_running: bool = False

class User:
    """User account with credentials"""
    def __init__(self, username: str, password_hash: str):
        self.username: str = username
        self.password_hash: str = password_hash
        self.created_at: str = datetime.utcnow().isoformat()
        self.state: UserState = UserState(username)

stable_key = base64.urlsafe_b64encode(hashlib.sha256(JWT_SECRET.encode()).digest())
cipher = Fernet(stable_key)

users: Dict[str, User] = {}

state = None

if os.path.isdir("/data"):
    PERSISTENCE_DIR = "/data"
    logger.info("Using /data for persistent storage (Fly.io volume detected)")
elif os.environ.get("PERSISTENCE_DIR"):
    PERSISTENCE_DIR = os.environ.get("PERSISTENCE_DIR")
    logger.info(f"Using {PERSISTENCE_DIR} for persistent storage (from env var)")
else:
    PERSISTENCE_DIR = "/tmp"
    logger.warning("Using /tmp for persistent storage (ephemeral - will not survive restarts)")

USERS_FILE = os.path.join(PERSISTENCE_DIR, "users.json")
LEGACY_STATE_FILE = os.path.join(PERSISTENCE_DIR, "state.json")

def save_users():
    """Save all users to persistent storage"""
    try:
        os.makedirs(PERSISTENCE_DIR, exist_ok=True)
        
        users_data = {}
        for username, user in users.items():
            user_state = user.state
            users_data[username] = {
                "password_hash": user.password_hash,
                "created_at": user.created_at,
                "exchange_type": user_state.exchange_type,
                "api_key": cipher.encrypt(user_state.api_key.encode()).decode() if user_state.api_key else None,
                "api_secret": cipher.encrypt(user_state.api_secret.encode()).decode() if user_state.api_secret else None,
                "auto_trading_enabled": user_state.auto_trading_enabled,
                "coin_trading_enabled": user_state.coin_trading_enabled,
                "emergency_stop": user_state.emergency_stop,
                "stop_loss_enabled": user_state.stop_loss_enabled,
                "stop_loss_pct": user_state.stop_loss_pct
            }
        
        temp_file = USERS_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(users_data, f, indent=2)
        os.replace(temp_file, USERS_FILE)
        
        logger.info(f"Users saved to persistent storage at {USERS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save users to {USERS_FILE}: {str(e)}")
        return False

async def load_users_and_connect():
    """Load all users from persistent storage and reconnect"""
    logger.info(f"load_users_and_connect starting; USERS_FILE={USERS_FILE}")
    try:
        if not os.path.exists(USERS_FILE):
            logger.info(f"No users file found at {USERS_FILE}")
            await migrate_legacy_state()
            return
        
        logger.info(f"Loading users from {USERS_FILE}")
        with open(USERS_FILE, 'r') as f:
            users_data = json.load(f)
        
        for username, data in users_data.items():
            user = User(username, data["password_hash"])
            user.created_at = data.get("created_at", datetime.utcnow().isoformat())
            user_state = user.state
            
            user_state.exchange_type = data.get("exchange_type")
            user_state.auto_trading_enabled = data.get("auto_trading_enabled", False)
            user_state.coin_trading_enabled = data.get("coin_trading_enabled", {
                "BTC": True, "ETH": True, "SOL": True, "XRP": True
            })
            user_state.emergency_stop = data.get("emergency_stop", False)
            user_state.stop_loss_enabled = data.get("stop_loss_enabled", True)
            user_state.stop_loss_pct = data.get("stop_loss_pct", 0.02)
            
            encrypted_key = data.get("api_key")
            encrypted_secret = data.get("api_secret")
            
            if encrypted_key and encrypted_secret:
                try:
                    user_state.api_key = cipher.decrypt(encrypted_key.encode()).decode()
                    user_state.api_secret = cipher.decrypt(encrypted_secret.encode()).decode()
                    
                    if user_state.exchange_type == "kraken":
                        user_state.kraken_client = KrakenClient(user_state.api_key, user_state.api_secret)
                        balance_data = await user_state.kraken_client.get_balance()
                        user_state.api_connected = True
                        user_state.connection_error = None
                        logger.info(f"Reconnected user {username} to Kraken API successfully")
                    else:
                        user_state.exchange = get_exchange_instance(user_state.exchange_type, user_state.api_key, user_state.api_secret)
                        balance = user_state.exchange.fetch_balance()
                        user_state.api_connected = True
                        user_state.connection_error = None
                        logger.info(f"Reconnected user {username} to {user_state.exchange_type} successfully")
                except Exception as e:
                    user_state.api_connected = False
                    user_state.connection_error = str(e)
                    logger.error(f"Failed to reconnect user {username}: {str(e)}")
            
            users[username] = user
        
        logger.info(f"Loaded {len(users)} users from persistent storage")
                
    except Exception as e:
        logger.error(f"Failed to load users: {str(e)}")

async def migrate_legacy_state():
    """Migrate legacy single-user state to multi-user format"""
    logger.info("Checking for legacy state to migrate")
    try:
        if not os.path.exists(LEGACY_STATE_FILE):
            logger.info("No legacy state found")
            return
        
        logger.info(f"Found legacy state at {LEGACY_STATE_FILE}, migrating to multi-user format")
        with open(LEGACY_STATE_FILE, 'r') as f:
            legacy_data = json.load(f)
        
        admin_username = "admin"
        admin_password_hash = hash_password_bcrypt(APP_PASSWORD)
        admin_user = User(admin_username, admin_password_hash)
        
        admin_state = admin_user.state
        admin_state.exchange_type = legacy_data.get("exchange_type")
        admin_state.auto_trading_enabled = legacy_data.get("auto_trading_enabled", False)
        admin_state.coin_trading_enabled = legacy_data.get("coin_trading_enabled", {
            "BTC": True, "ETH": True, "SOL": True, "XRP": True
        })
        admin_state.emergency_stop = legacy_data.get("emergency_stop", False)
        
        encrypted_key = legacy_data.get("api_key")
        encrypted_secret = legacy_data.get("api_secret")
        
        if encrypted_key and encrypted_secret:
            try:
                admin_state.api_key = cipher.decrypt(encrypted_key.encode()).decode()
                admin_state.api_secret = cipher.decrypt(encrypted_secret.encode()).decode()
                
                if admin_state.exchange_type == "kraken":
                    admin_state.kraken_client = KrakenClient(admin_state.api_key, admin_state.api_secret)
                    balance_data = await admin_state.kraken_client.get_balance()
                    admin_state.api_connected = True
                    admin_state.connection_error = None
                    logger.info(f"Migrated admin user to Kraken API successfully")
                else:
                    admin_state.exchange = get_exchange_instance(admin_state.exchange_type, admin_state.api_key, admin_state.api_secret)
                    balance = admin_state.exchange.fetch_balance()
                    admin_state.api_connected = True
                    admin_state.connection_error = None
                    logger.info(f"Migrated admin user to {admin_state.exchange_type} successfully")
            except Exception as e:
                admin_state.api_connected = False
                admin_state.connection_error = str(e)
                logger.error(f"Failed to connect migrated admin user: {str(e)}")
        
        users[admin_username] = admin_user
        save_users()
        
        backup_file = LEGACY_STATE_FILE + ".migrated"
        os.rename(LEGACY_STATE_FILE, backup_file)
        logger.info(f"Legacy state migrated successfully. Admin user created with username 'admin' and existing password. Legacy file backed up to {backup_file}")
        
    except Exception as e:
        logger.error(f"Failed to migrate legacy state: {str(e)}")

class ApiKeyRequest(BaseModel):
    api_key: str
    api_secret: str
    exchange: ExchangeType

class WebhookAlert(BaseModel):
    secret: str
    action: str
    symbol: str
    price: Optional[str] = None
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    quantity: Optional[Any] = None
    quantity_usd: Optional[float] = None
    usd_amount: Optional[float] = None
    sell_all: Optional[bool] = None
    timestamp: Optional[str] = None

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

class LoginRequest(BaseModel):
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class CoinToggleRequest(BaseModel):
    coin: str
    enabled: bool

class EmergencyStopRequest(BaseModel):
    stop: bool

class ApiSettingsRequest(BaseModel):
    exchange: str
    apiKey: str
    apiSecret: str

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

async def execute_webhook_for_user(username: str, user_state: UserState, alert: WebhookAlert, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Execute webhook for a single user with error handling and timeout.
    Returns a dict with execution result for logging.
    """
    result = {
        "username": username,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "executed": False,
        "error": None,
        "order": None
    }
    
    try:
        if user_state.emergency_stop:
            result["status"] = "skipped"
            result["error"] = "Emergency stop activated"
            return result
        
        if not user_state.auto_trading_enabled:
            result["status"] = "skipped"
            result["error"] = "Auto-trading disabled"
            return result
        
        coin = alert.symbol.upper().replace("USDT", "").replace("USD", "")
        if coin in user_state.coin_trading_enabled and not user_state.coin_trading_enabled[coin]:
            result["status"] = "skipped"
            result["error"] = f"{coin} trading is disabled"
            return result
        
        if not user_state.exchange and not user_state.kraken_client:
            result["status"] = "failed"
            result["error"] = "Exchange not configured"
            return result
        
        if not user_state.api_connected:
            result["status"] = "failed"
            result["error"] = "API not connected"
            return result
        
        async def execute_trade():
            symbol = normalize_symbol(alert.symbol, user_state.exchange_type)
            action = alert.action.lower()
            
            if action in ["buy", "long"]:
                side = "buy"
            elif action in ["sell", "short", "close"]:
                side = "sell"
            else:
                raise ValueError(f"Invalid action: {action}")
            
            amount = 0.001  # default
            
            if user_state.exchange_type == "kraken" and user_state.kraken_client:
                balance_data = await user_state.kraken_client.get_balance()
                usdt_balance = float(balance_data.get('USDT', 0))
                
                if alert.sell_all:
                    coin_key = coin
                    if coin == "BTC":
                        coin_key = "XBT"
                    coin_balance = float(balance_data.get(coin_key, 0))
                    amount = round_kraken_volume(alert.symbol, coin_balance)
                elif alert.usd_amount:
                    if alert.price:
                        raw_amount = alert.usd_amount / float(alert.price)
                    else:
                        kraken_pair = user_state.kraken_client.to_kraken_pair(alert.symbol)
                        current_price = await user_state.kraken_client.get_ticker_price(kraken_pair)
                        raw_amount = alert.usd_amount / current_price
                    amount = round_kraken_volume(alert.symbol, raw_amount)
                elif alert.quantity == "all" or (isinstance(alert.quantity, str) and alert.quantity.lower() == "all"):
                    coin_key = coin
                    if coin == "BTC":
                        coin_key = "XBT"
                    coin_balance = float(balance_data.get(coin_key, 0))
                    amount = round_kraken_volume(alert.symbol, coin_balance)
                elif alert.quantity_usd:
                    if alert.price:
                        raw_amount = alert.quantity_usd / float(alert.price)
                    else:
                        kraken_pair = user_state.kraken_client.to_kraken_pair(alert.symbol)
                        current_price = await user_state.kraken_client.get_ticker_price(kraken_pair)
                        raw_amount = alert.quantity_usd / current_price
                    amount = round_kraken_volume(alert.symbol, raw_amount)
                elif alert.quantity:
                    raw_amount = float(alert.quantity)
                    amount = round_kraken_volume(alert.symbol, raw_amount)
                elif usdt_balance > 0:
                    position_size_usdt = calculate_position_size(usdt_balance, 0.02)
                    if alert.price:
                        raw_amount = position_size_usdt / float(alert.price)
                    else:
                        kraken_pair = user_state.kraken_client.to_kraken_pair(alert.symbol)
                        current_price = await user_state.kraken_client.get_ticker_price(kraken_pair)
                        raw_amount = position_size_usdt / current_price
                    amount = round_kraken_volume(alert.symbol, raw_amount)
                else:
                    amount = round_kraken_volume(alert.symbol, 0.001)
                
                order = None
                max_retries = 3
                kraken_pair = user_state.kraken_client.to_kraken_pair(alert.symbol)
                
                for attempt in range(max_retries):
                    try:
                        if alert.price:
                            price = float(alert.price)
                            order_result = await user_state.kraken_client.add_order(
                                pair=kraken_pair,
                                side=side,
                                order_type='limit',
                                volume=amount,
                                price=price
                            )
                            order = {
                                'id': order_result.get('txid', [''])[0] if order_result.get('txid') else '',
                                'status': 'open',
                                'average': price,
                                'price': price
                            }
                        else:
                            order_result = await user_state.kraken_client.add_order(
                                pair=kraken_pair,
                                side=side,
                                order_type='market',
                                volume=amount
                            )
                            order = {
                                'id': order_result.get('txid', [''])[0] if order_result.get('txid') else '',
                                'status': 'open',
                                'average': 0,
                                'price': 0
                            }
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(1)
                
                exchange_pair = kraken_pair
            else:
                balance = user_state.exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                
                if alert.sell_all:
                    coin_balance = balance.get(coin, {}).get('free', 0)
                    amount = coin_balance
                elif alert.usd_amount:
                    if alert.price:
                        amount = alert.usd_amount / float(alert.price)
                    else:
                        ticker = user_state.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        amount = alert.usd_amount / current_price
                elif alert.quantity == "all" or (isinstance(alert.quantity, str) and alert.quantity.lower() == "all"):
                    coin_balance = balance.get(coin, {}).get('free', 0)
                    amount = coin_balance
                elif alert.quantity_usd:
                    if alert.price:
                        amount = alert.quantity_usd / float(alert.price)
                    else:
                        ticker = user_state.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        amount = alert.quantity_usd / current_price
                elif alert.quantity:
                    amount = float(alert.quantity)
                elif usdt_balance > 0:
                    position_size_usdt = calculate_position_size(usdt_balance, 0.02)
                    if alert.price:
                        amount = position_size_usdt / float(alert.price)
                    else:
                        ticker = user_state.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        amount = position_size_usdt / current_price
                else:
                    amount = 0.001
                
                order = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if alert.price:
                            price = float(alert.price)
                            order = user_state.exchange.create_limit_order(symbol, side, amount, price)
                        else:
                            order = user_state.exchange.create_market_order(symbol, side, amount)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(1)
                
                exchange_pair = symbol
            
            entry_price = float(alert.price) if alert.price else order.get('average', order.get('price', 0))
            
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "coin": coin,
                "side": side,
                "amount": amount,
                "entry_price": entry_price,
                "order_id": order.get('id'),
                "status": order.get('status'),
                "stop_loss": alert.stop_loss,
                "take_profit": alert.take_profit,
                "pnl": 0.0,
                "raw_symbol": alert.symbol,
                "exchange_pair": exchange_pair
            }
            
            user_state.last_order = trade
            user_state.orders_history.append(trade)
            
            if side == "buy":
                user_state.open_trades.append(trade)
            else:
                user_state.closed_trades.append(trade)
            
            return trade
        
        trade = await asyncio.wait_for(execute_trade(), timeout=timeout_seconds)
        
        result["status"] = "executed"
        result["executed"] = True
        result["order"] = trade
        
        logger.info(f"User {username}: Order executed successfully: {trade}")
        
    except asyncio.TimeoutError:
        result["status"] = "failed"
        result["error"] = "Execution timeout"
        logger.error(f"User {username}: Webhook execution timeout")
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"User {username}: Webhook execution failed: {str(e)}")
    
    return result

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/register")
async def register(request: RegisterRequest):
    """Register a new user"""
    username = request.username.strip()
    password = request.password
    
    if not username or len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    password_hash = hash_password_bcrypt(password)
    user = User(username, password_hash)
    users[username] = user
    save_users()
    
    token = create_jwt_token(username)
    logger.info(f"New user registered: {username}")
    
    return {
        "success": True,
        "token": token,
        "username": username,
        "message": "Registration successful"
    }

@app.post("/login")
async def login(request: LoginRequest):
    """Legacy login endpoint for backward compatibility (password only)"""
    if verify_password(request.password):
        admin_username = "admin"
        if admin_username not in users:
            password_hash = hash_password_bcrypt(APP_PASSWORD)
            admin_user = User(admin_username, password_hash)
            users[admin_username] = admin_user
            save_users()
            logger.info(f"Created admin user via legacy login")
        
        token = create_jwt_token(admin_username)
        return {
            "success": True,
            "token": token,
            "username": admin_username,
            "message": "Login successful"
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid password")

@app.post("/user/login")
async def user_login(request: UserLoginRequest):
    """User login with username and password"""
    username = request.username.strip()
    password = request.password
    
    if username not in users:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user = users[username]
    if not verify_password_bcrypt(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_jwt_token(username)
    logger.info(f"User logged in: {username}")
    
    return {
        "success": True,
        "token": token,
        "username": username,
        "message": "Login successful"
    }

@app.post("/api/settings")
async def save_settings(request: ApiSettingsRequest, username: str = Depends(verify_session)):
    """Save API settings with validation and automatic connection test"""
    try:
        if username not in users:
            raise HTTPException(status_code=401, detail="User not found")
        
        user = users[username]
        user_state = user.state
        
        api_key = request.apiKey.strip()
        api_secret = request.apiSecret.strip()
        exchange = request.exchange.lower().strip()
        
        logger.info(f"Saving settings for user {username}, exchange: {exchange}")
        
        if exchange not in ["binance", "kucoin", "kraken", "bybit", "coinbase"]:
            raise HTTPException(status_code=400, detail=f"Unsupported exchange: {exchange}")
        
        user_state.api_connected = False
        user_state.connection_error = None
        
        if exchange == "kraken":
            try:
                kraken_client = KrakenClient(api_key, api_secret)
                
                logger.info(f"Testing Kraken API connection for user {username}...")
                balance = await kraken_client.get_balance()
                logger.info(f"Kraken API connection successful for user {username}. Balance keys: {list(balance.keys())}")
                
                user_state.api_key = api_key
                user_state.api_secret = api_secret
                user_state.exchange_type = exchange
                user_state.kraken_client = kraken_client
                user_state.exchange = None
                user_state.api_connected = True
                user_state.connection_error = None
                
                save_users()
                
                return {
                    "success": True,
                    "message": "Kraken API connected successfully",
                    "exchange": exchange,
                    "connected": True
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Kraken API connection failed for user {username}: {error_msg}")
                
                if "EAPI:Invalid key" in error_msg or "Invalid API key" in error_msg:
                    friendly_error = "Invalid API key"
                elif "EAPI:Invalid signature" in error_msg or "Signature error" in error_msg:
                    friendly_error = "Signature error - check your API secret"
                elif "EAPI:Permission denied" in error_msg or "Permission denied" in error_msg:
                    friendly_error = "Permission denied: Query balance required"
                elif "EAPI:Invalid nonce" in error_msg:
                    friendly_error = "Invalid nonce - please try again"
                else:
                    friendly_error = f"Kraken API error: {error_msg}"
                
                user_state.connection_error = friendly_error
                raise HTTPException(status_code=400, detail=friendly_error)
        
        else:
            try:
                exchange_instance = get_exchange_instance(exchange, api_key, api_secret)
                
                logger.info(f"Testing {exchange} API connection for user {username}...")
                await exchange_instance.load_markets()
                logger.info(f"{exchange} API connection successful for user {username}")
                
                user_state.api_key = api_key
                user_state.api_secret = api_secret
                user_state.exchange_type = exchange
                user_state.exchange = exchange_instance
                user_state.kraken_client = None
                user_state.api_connected = True
                user_state.connection_error = None
                
                save_users()
                
                return {
                    "success": True,
                    "message": f"{exchange.capitalize()} API connected successfully",
                    "exchange": exchange,
                    "connected": True
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"{exchange} API connection failed for user {username}: {error_msg}")
                
                if "Invalid API" in error_msg or "invalid api" in error_msg.lower():
                    friendly_error = "Invalid API key"
                elif "signature" in error_msg.lower():
                    friendly_error = "Signature error - check your API secret"
                elif "permission" in error_msg.lower():
                    friendly_error = "Permission denied: Check API key permissions"
                else:
                    friendly_error = f"API connection error: {error_msg}"
                
                user_state.connection_error = friendly_error
                raise HTTPException(status_code=400, detail=friendly_error)
                
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to save settings: {str(e)}"
        logger.error(error_msg)
        if username in users:
            users[username].state.connection_error = error_msg
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/api/settings")
async def get_settings(username: str = Depends(verify_session)):
    """Get current API settings and connection status"""
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    return {
        "exchange": user_state.exchange_type,
        "connected": user_state.api_connected,
        "has_api_key": user_state.api_key is not None,
        "error": user_state.connection_error
    }

@app.post("/set-api-key")
async def set_api_key(request: ApiKeyRequest, authenticated: bool = Depends(verify_session)):
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
async def get_webhook_url(request: Request, authenticated: bool = Depends(verify_session)):
    base_url = str(request.base_url).rstrip('/')
    if base_url.startswith("http://"):
        base_url = base_url.replace("http://", "https://", 1)
    webhook_url = f"{base_url}/webhook"
    return {"webhook_url": webhook_url}

@app.post("/toggle-trading")
async def toggle_trading(request: ToggleTradingRequest, username: str = Depends(verify_session)):
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    user_state.auto_trading_enabled = request.enabled
    save_users()
    logger.info(f"User {username}: Auto-trading {'enabled' if request.enabled else 'disabled'}")
    return {
        "success": True,
        "auto_trading_enabled": user_state.auto_trading_enabled
    }

@app.post("/toggle-coin")
async def toggle_coin(request: CoinToggleRequest, username: str = Depends(verify_session)):
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    if request.coin not in user_state.coin_trading_enabled:
        raise HTTPException(status_code=400, detail=f"Invalid coin: {request.coin}")
    
    user_state.coin_trading_enabled[request.coin] = request.enabled
    save_users()
    logger.info(f"User {username}: {request.coin} trading {'enabled' if request.enabled else 'disabled'}")
    
    return {
        "success": True,
        "coin": request.coin,
        "enabled": request.enabled,
        "coin_trading_enabled": user_state.coin_trading_enabled
    }

@app.post("/emergency-stop")
async def emergency_stop(request: EmergencyStopRequest, username: str = Depends(verify_session)):
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    user_state.emergency_stop = request.stop
    if request.stop:
        user_state.auto_trading_enabled = False
        logger.warning(f"User {username}: EMERGENCY STOP ACTIVATED - All trading disabled")
    else:
        logger.info(f"User {username}: Emergency stop deactivated")
    
    save_users()
    return {
        "success": True,
        "emergency_stop": user_state.emergency_stop,
        "auto_trading_enabled": user_state.auto_trading_enabled
    }

@app.post("/webhook")
async def webhook(alert: WebhookAlert):
    """
    Broadcast webhook endpoint - executes trades for all eligible users.
    Each user's execution is independent with individual error handling.
    """
    logger.info(f"Webhook received: {alert.dict()}")
    
    try:
        if not verify_webhook_secret(alert.secret):
            logger.warning(f"Invalid webhook secret received")
            raise HTTPException(status_code=401, detail="Invalid webhook secret")
        
        if not check_rate_limit():
            logger.warning(f"Rate limit exceeded for webhook endpoint")
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per minute.")
        
        # Broadcast to all users
        execution_results = []
        eligible_users = []
        
        for username, user in list(users.items()):
            user_state = user.state
            if user_state.api_connected:
                eligible_users.append((username, user_state))
        
        if not eligible_users:
            logger.info("No eligible users found for webhook execution")
            return {
                "success": True,
                "message": "Webhook received but no eligible users",
                "executed": False,
                "results": []
            }
        
        logger.info(f"Broadcasting webhook to {len(eligible_users)} eligible users")
        
        tasks = []
        for username, user_state in eligible_users:
            task = execute_webhook_for_user(username, user_state, alert)
            tasks.append(task)
        
        execution_results = await asyncio.gather(*tasks, return_exceptions=False)
        
        for result in execution_results:
            username = result.get("username")
            user_state = users[username].state
            
            webhook_log = {
                "timestamp": result.get("timestamp"),
                "payload": alert.dict(),
                "status": result.get("status"),
                "executed": result.get("executed"),
                "error": result.get("error"),
                "order": result.get("order")
            }
            user_state.webhook_logs.append(webhook_log)
            
            if result.get("executed"):
                user_state.last_webhook = {
                    "timestamp": datetime.now().isoformat(),
                    "payload": alert.dict()
                }
        
        save_users()
        
        executed_count = sum(1 for r in execution_results if r.get("executed"))
        skipped_count = sum(1 for r in execution_results if r.get("status") == "skipped")
        failed_count = sum(1 for r in execution_results if r.get("status") == "failed")
        
        logger.info(f"Webhook broadcast complete: {executed_count} executed, {skipped_count} skipped, {failed_count} failed")
        
        return {
            "success": True,
            "message": f"Webhook broadcast complete: {executed_count} executed, {skipped_count} skipped, {failed_count} failed",
            "executed": executed_count > 0,
            "results": execution_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Webhook broadcast failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/place-order")
async def place_order(request: OrderRequest, authenticated: bool = Depends(verify_session)):
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
async def close_order(request: CloseOrderRequest, authenticated: bool = Depends(verify_session)):
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
async def get_status(username: str = Depends(verify_session)):
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    
    if not user_state.api_connected and os.path.exists(USERS_FILE):
        logger.info(f"Self-heal: User {username} API not connected, attempting reconnect")
        try:
            await load_users_and_connect()
        except Exception as e:
            logger.error(f"Self-heal reconnect failed for user {username}: {str(e)}")
    
    api_connected = user_state.api_connected
    
    total_trades = len(user_state.closed_trades)
    winning_trades = len([t for t in user_state.closed_trades if t.get('pnl', 0) > 0])
    losing_trades = len([t for t in user_state.closed_trades if t.get('pnl', 0) < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_trade_size = sum([t.get('amount', 0) for t in user_state.orders_history]) / len(user_state.orders_history) if user_state.orders_history else 0
    
    total_exposure = sum([t.get('amount', 0) * t.get('entry_price', 0) for t in user_state.open_trades])
    
    pnl_summary = {
        "total_orders": len(user_state.orders_history),
        "total_pnl": user_state.total_pnl,
        "per_coin_pnl": user_state.per_coin_pnl,
        "recent_orders": user_state.orders_history[-10:] if user_state.orders_history else [],
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_trade_size": avg_trade_size,
        "total_exposure": total_exposure
    }
    
    return {
        "api_connected": api_connected,
        "exchange": user_state.exchange_type if api_connected else None,
        "auto_trading_enabled": user_state.auto_trading_enabled,
        "emergency_stop": user_state.emergency_stop,
        "stop_loss_enabled": user_state.stop_loss_enabled,
        "stop_loss_pct": user_state.stop_loss_pct,
        "coin_trading_enabled": user_state.coin_trading_enabled,
        "last_webhook": user_state.last_webhook,
        "last_order": user_state.last_order,
        "pnl_summary": pnl_summary,
        "open_trades": user_state.open_trades,
        "closed_trades": user_state.closed_trades[-20:] if user_state.closed_trades else [],
        "webhook_logs": user_state.webhook_logs[-50:] if user_state.webhook_logs else []
    }

@app.post("/test-webhook")
async def test_webhook(username: str = Depends(verify_session)):
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    
    test_alert = WebhookAlert(
        secret=WEBHOOK_SECRET,
        action="buy",
        symbol="BTCUSDT",
        price="50000",
        quantity=0.001
    )
    
    try:
        result = await webhook(test_alert)
        if user_state.webhook_logs:
            user_state.webhook_logs[-1]["test"] = True
        return {
            "success": True,
            "message": "Test webhook executed successfully",
            "result": result
        }
    except HTTPException as e:
        return {
            "success": False,
            "message": f"Test webhook failed: {e.detail}",
            "status_code": e.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Test webhook error: {str(e)}"
        }

async def monitor_stop_loss():
    """Background task to monitor stop loss for open positions across all users"""
    logger.info("Stop loss monitoring task started")
    
    while True:
        try:
            await asyncio.sleep(15)
            
            for username, user in list(users.items()):
                try:
                    user_state = user.state
                    
                    if user_state._stoploss_running:
                        continue
                    
                    user_state._stoploss_running = True
                    
                    if not user_state.stop_loss_enabled or not user_state.auto_trading_enabled or user_state.emergency_stop:
                        user_state._stoploss_running = False
                        continue
                    
                    if not user_state.open_trades:
                        user_state._stoploss_running = False
                        continue
                    
                    unique_pairs = {}
                    for trade in user_state.open_trades:
                        if trade.get('side') == 'buy':
                            pair = trade.get('exchange_pair')
                            if pair and pair not in unique_pairs:
                                unique_pairs[pair] = trade
                    
                    if not unique_pairs:
                        user_state._stoploss_running = False
                        continue
                    
                    prices = {}
                    for pair in unique_pairs.keys():
                        try:
                            if user_state.exchange_type == "kraken" and user_state.kraken_client:
                                price = await user_state.kraken_client.get_ticker_price(pair)
                            elif user_state.exchange:
                                ticker = await user_state.exchange.fetch_ticker(pair)
                                price = ticker['last']
                            else:
                                continue
                            prices[pair] = price
                        except Exception as e:
                            logger.error(f"User {username}: Failed to fetch price for {pair}: {str(e)}")
                            continue
                    
                    trades_to_close = []
                    for i, trade in enumerate(user_state.open_trades):
                        if trade.get('side') != 'buy':
                            continue
                        
                        pair = trade.get('exchange_pair')
                        if pair not in prices:
                            continue
                        
                        entry_price = trade.get('entry_price', 0)
                        if entry_price <= 0:
                            continue
                        
                        current_price = prices[pair]
                        stop_price = entry_price * (1 - user_state.stop_loss_pct)
                        
                        if current_price <= stop_price:
                            trades_to_close.append((i, trade, current_price))
                            logger.warning(f"User {username}: Stop loss triggered for {trade.get('coin')}: entry={entry_price}, current={current_price}, stop={stop_price}")
                    
                    for trade_idx, trade, exit_price in trades_to_close:
                        try:
                            amount = trade.get('amount', 0)
                            pair = trade.get('exchange_pair')
                            
                            max_retries = 3
                            order = None
                            
                            for attempt in range(max_retries):
                                try:
                                    if user_state.exchange_type == "kraken" and user_state.kraken_client:
                                        order_result = await user_state.kraken_client.add_order(
                                            pair=pair,
                                            side='sell',
                                            order_type='market',
                                            volume=amount
                                        )
                                        order = {
                                            'id': order_result.get('txid', [''])[0] if order_result.get('txid') else '',
                                            'status': 'closed'
                                        }
                                    elif user_state.exchange:
                                        order = await user_state.exchange.create_market_order(
                                            trade.get('symbol'),
                                            'sell',
                                            amount
                                        )
                                    
                                    if order:
                                        break
                                except Exception as e:
                                    logger.error(f"User {username}: Stop loss sell attempt {attempt + 1} failed: {str(e)}")
                                    if attempt == max_retries - 1:
                                        raise
                                    await asyncio.sleep(1)
                            
                            if order:
                                pnl = (exit_price - trade.get('entry_price', 0)) * amount
                                
                                closed_trade = trade.copy()
                                closed_trade['exit_price'] = exit_price
                                closed_trade['pnl'] = pnl
                                closed_trade['closed_by'] = 'stop_loss'
                                closed_trade['closed_at'] = datetime.now().isoformat()
                                
                                user_state.closed_trades.append(closed_trade)
                                user_state.open_trades.remove(trade)
                                
                                coin = trade.get('coin', '')
                                if coin:
                                    user_state.per_coin_pnl[coin] = user_state.per_coin_pnl.get(coin, 0) + pnl
                                user_state.total_pnl += pnl
                                
                                webhook_log = {
                                    "timestamp": datetime.now().isoformat(),
                                    "payload": {
                                        "action": "stop_loss",
                                        "symbol": trade.get('raw_symbol'),
                                        "amount": amount,
                                        "entry_price": trade.get('entry_price'),
                                        "exit_price": exit_price,
                                        "pnl": pnl
                                    },
                                    "status": "executed",
                                    "executed": True,
                                    "reason": "stop_loss",
                                    "error": None
                                }
                                user_state.webhook_logs.append(webhook_log)
                                
                                logger.info(f"User {username}: Stop loss executed: {trade.get('coin')} sold at {exit_price}, PnL: {pnl}")
                            
                        except Exception as e:
                            error_msg = f"Stop loss execution failed: {str(e)}"
                            logger.error(f"User {username}: {error_msg}")
                            
                            webhook_log = {
                                "timestamp": datetime.now().isoformat(),
                                "payload": {
                                    "action": "stop_loss_failed",
                                    "symbol": trade.get('raw_symbol'),
                                    "amount": trade.get('amount'),
                                    "error": error_msg
                                },
                                "status": "failed",
                                "executed": False,
                                "reason": "stop_loss",
                                "error": error_msg
                            }
                            user_state.webhook_logs.append(webhook_log)
                    
                    user_state._stoploss_running = False
                    
                except Exception as e:
                    logger.error(f"User {username}: Stop loss monitoring error: {str(e)}")
                    if username in users:
                        users[username].state._stoploss_running = False
            
        except Exception as e:
            logger.error(f"Stop loss monitoring error: {str(e)}")
            await asyncio.sleep(15)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    logger.info("startup_event begin - loading users and starting multi-user monitoring")
    await load_users_and_connect()
    asyncio.create_task(monitor_stop_loss())
    logger.info(f"Background tasks started - monitoring {len(users)} users")
