from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
from decimal import Decimal
import ccxt
from cryptography.fernet import Fernet
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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
import math
from dotenv import load_dotenv
from app.database import init_db, get_db, save_user_to_db, load_all_users_from_db, SessionLocal

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

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
JWT_EXPIRATION_DAYS = 30

def now_et_iso() -> str:
    """Return current timestamp in America/New_York timezone as ISO string"""
    return datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).isoformat()
webhook_request_times = []
processed_idempotency_keys = {}
IDEMPOTENCY_TTL_SECONDS = 3600

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
        logger.warning("verify_session: Missing authorization header")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            logger.warning("verify_session: Invalid token payload - no username")
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.debug(f"verify_session: Successfully authenticated user: {username}")
        return username
    except jwt.ExpiredSignatureError:
        logger.warning("verify_session: Token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"verify_session: Invalid token - {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

def check_rate_limit() -> bool:
    """Global rate limit: max 10 requests per minute"""
    current_time = time.time()
    webhook_request_times[:] = [t for t in webhook_request_times if current_time - t < 60]
    
    if len(webhook_request_times) >= 10:
        return False
    
    webhook_request_times.append(current_time)
    return True

def check_user_rate_limit(user_state: 'UserState', max_requests: int = 20, window_seconds: int = 60) -> bool:
    """Per-user rate limit: max 20 requests per minute by default"""
    current_time = time.time()
    user_state.rate_limit_requests[:] = [t for t in user_state.rate_limit_requests if current_time - t < window_seconds]
    
    if len(user_state.rate_limit_requests) >= max_requests:
        return False
    
    user_state.rate_limit_requests.append(current_time)
    return True

def log_event(user_state: 'UserState', action: str, status: str, details: Optional[Dict] = None, error: Optional[str] = None):
    """Log critical events for auditing"""
    event = {
        "timestamp": now_et_iso(),
        "action": action,
        "status": status,
        "details": details or {},
        "error": error
    }
    user_state.event_logs.append(event)
    
    if len(user_state.event_logs) > 1000:
        user_state.event_logs = user_state.event_logs[-1000:]
    
    log_msg = f"[{user_state.username}] {action} - {status}"
    if error:
        log_msg += f" - Error: {error}"
    if details:
        log_msg += f" - Details: {details}"
    
    if status == "failed" or error:
        logger.error(log_msg)
    else:
        logger.info(log_msg)

def calculate_position_size(balance: float, risk_percentage: float = 0.02) -> float:
    """Calculate position size based on account balance (default 2%)"""
    return balance * risk_percentage

def verify_webhook_secret(secret: str) -> bool:
    """Verify webhook secret token"""
    return secret == WEBHOOK_SECRET

async def calculate_fee_aware_buy_volume(kraken_client, pair: str, usd_amount: float, price: float, balance_data: Dict) -> float:
    """
    Calculate maximum affordable buy volume accounting for Kraken fees.
    Returns 0 if the order cannot be placed due to insufficient funds or below ordermin.
    """
    try:
        pair_info = await kraken_client.get_asset_pairs(pair)
        lot_decimals = pair_info['lot_decimals']
        ordermin = pair_info['ordermin']
        quote = pair_info['quote']
        
        fee_rate = await kraken_client.get_fee_rate(pair)
        
        quote_balance_key = kraken_client.to_kraken_balance_key(quote)
        quote_balance = float(balance_data.get(quote_balance_key, 0))
        logger.info(f"Balance check: quote={quote}, balance_key={quote_balance_key}, balance=${quote_balance:.2f}")
        
        spend_cap = min(usd_amount, quote_balance / (1 + fee_rate))
        
        raw_vol = spend_cap / price
        
        step = 10 ** (-lot_decimals)
        vol = math.floor(raw_vol / step) * step
        
        if vol < ordermin:
            ordermin_cost_with_fees = ordermin * price * (1 + fee_rate)
            if ordermin_cost_with_fees <= min(usd_amount, quote_balance):
                vol = ordermin
                logger.info(f"Volume below ordermin, bumping to {ordermin:.8f} (affordable with fees)")
            else:
                logger.warning(f"Cannot afford ordermin {ordermin:.8f} with fees. Required: ${ordermin_cost_with_fees:.2f}, Available: ${quote_balance:.2f}")
                return 0.0
        
        estimated_cost = vol * price
        est_fees = estimated_cost * fee_rate
        est_total_cost = estimated_cost + est_fees
        
        logger.info(f"Fee-aware buy calculation: quote={quote}, quote_balance=${quote_balance:.2f}, usd_requested=${usd_amount:.2f}, price=${price:.2f}, fee_rate={fee_rate*100:.3f}%, spend_cap=${spend_cap:.2f}, raw_vol={raw_vol:.8f}, step={step}, vol={vol:.8f}, est_cost=${estimated_cost:.2f}, est_fees=${est_fees:.2f}, est_total=${est_total_cost:.2f}")
        
        return vol
        
    except Exception as e:
        logger.error(f"Error in fee-aware buy calculation: {e}. Using fallback.")
        return 0.0

async def round_and_enforce_kraken_volume(kraken_client, pair: str, symbol: str, raw_volume: float, side: str) -> float:
    """
    Round volume and enforce Kraken's ordermin for the pair.
    Fetches lot_decimals and ordermin from Kraken AssetPairs API.
    
    For sells: If balance < ordermin, returns 0 to skip the order (prevents "Insufficient funds")
    For buys: Bumps volume to ordermin if needed
    """
    try:
        pair_info = await kraken_client.get_asset_pairs(pair)
        lot_decimals = pair_info['lot_decimals']
        ordermin = pair_info['ordermin']
        
        step = 10 ** (-lot_decimals)
        
        if side == "buy":
            rounded = math.ceil(raw_volume / step) * step
            final_volume = max(rounded, ordermin)
        else:
            rounded = math.floor(raw_volume / step) * step
            if rounded < ordermin:
                logger.warning(f"Sell volume {rounded:.8f} below ordermin {ordermin:.8f} for {pair}. Skipping order to avoid insufficient funds.")
                return 0.0
            final_volume = rounded
        
        logger.info(f"Kraken volume: pair={pair}, side={side}, raw={raw_volume:.8f}, lot_decimals={lot_decimals}, step={step}, rounded={rounded:.8f}, ordermin={ordermin:.8f}, FINAL={final_volume:.8f}")
        
        return final_volume
        
    except Exception as e:
        logger.error(f"Error fetching AssetPairs for {pair}: {e}. Using fallback rounding.")
        symbol_clean = symbol.upper().replace("USDT", "").replace("USD", "")
        fallback_steps = {
            "BTC": 0.00001,
            "XBT": 0.00001,
            "ETH": 0.00001,
            "SOL": 0.001,
            "XRP": 0.1
        }
        fallback_ordermin = {
            "BTC": 0.0001,
            "XBT": 0.0001,
            "ETH": 0.005,
            "SOL": 0.1,
            "XRP": 10.0
        }
        step = fallback_steps.get(symbol_clean, 0.00001)
        ordermin = fallback_ordermin.get(symbol_clean, 0.001)
        
        if side == "buy":
            rounded = math.ceil(raw_volume / step) * step
            final_volume = max(rounded, ordermin)
        else:
            rounded = math.floor(raw_volume / step) * step
            if rounded < ordermin:
                logger.warning(f"Fallback: Sell volume {rounded:.8f} below ordermin {ordermin:.8f}. Skipping order.")
                return 0.0
            final_volume = rounded
        
        logger.info(f"Fallback rounding: symbol={symbol_clean}, raw={raw_volume:.8f}, step={step}, ordermin={ordermin}, final={final_volume:.8f}")
        
        return final_volume

class KrakenClient:
    """Direct Kraken API client with proper HMAC-SHA512 authentication"""
    
    def __init__(self, api_key: str, api_secret: str, username: str = None):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = "https://api.kraken.com"
        self.username = username
        
    def _get_nonce(self) -> str:
        """Generate strictly increasing nonce using DB-backed persistence"""
        if self.username:
            from app.database import SessionLocal, get_and_increment_nonce
            db = SessionLocal()
            try:
                nonce = get_and_increment_nonce(db, self.username)
                return str(nonce)
            finally:
                db.close()
        else:
            return str(int(time.time() * 1_000_000))
    
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
        """Convert standard symbol to Kraken pair format (handles both USD and USDT)"""
        s = symbol.upper().replace('-', '/').strip()
        
        if '/' in s:
            base, quote = s.split('/', 1)
        elif s.endswith('USDT'):
            base, quote = s[:-4], 'USDT'
        elif s.endswith('USD'):
            base, quote = s[:-3], 'USD'
        else:
            base, quote = s, 'USDT'
        
        base_map = {'BTC': 'XBT'}
        base = base_map.get(base, base)
        
        return f"{base}{quote}"
    
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
    
    async def get_asset_pairs(self, pair: str) -> Dict[str, Any]:
        """Get asset pair metadata from Kraken (lot_decimals, ordermin, base, quote, etc)"""
        url = f"{self.base_url}/0/public/AssetPairs"
        params = {'pair': pair}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            result = response.json()
            
            if result.get('error') and len(result['error']) > 0:
                error_msg = ', '.join(result['error'])
                raise Exception(f"Kraken AssetPairs error: {error_msg}")
            
            pairs_data = result.get('result', {})
            if not pairs_data:
                raise Exception(f"No AssetPairs data for {pair}")
            
            pair_key = list(pairs_data.keys())[0]
            pair_info = pairs_data[pair_key]
            
            return {
                'lot_decimals': pair_info.get('lot_decimals', 8),
                'ordermin': float(pair_info.get('ordermin', '0.0001')),
                'costmin': float(pair_info.get('costmin', '0')) if 'costmin' in pair_info else None,
                'base': pair_info.get('base', ''),
                'quote': pair_info.get('quote', '')
            }
    
    async def get_fee_rate(self, pair: str) -> float:
        """Get taker fee rate for a specific pair from Kraken TradeVolume API"""
        try:
            result = await self._private_request('/0/private/TradeVolume', {'pair': pair})
            
            fees = result.get('fees', {})
            if pair in fees:
                fee_info = fees[pair]
                taker_fee = float(fee_info.get('fee', '0.0026'))
                logger.info(f"Kraken fee rate for {pair}: {taker_fee*100:.3f}% (from TradeVolume API)")
                return taker_fee
            
            maker_fee = float(result.get('fees_maker', {}).get(pair, {}).get('fee', '0.0016'))
            if maker_fee > 0:
                logger.info(f"Kraken fee rate for {pair}: {maker_fee*100:.3f}% (maker fee from TradeVolume API)")
                return maker_fee
            
            logger.warning(f"No pair-specific fee found for {pair}, using fallback 0.26%")
            return 0.0026
            
        except Exception as e:
            logger.warning(f"Failed to fetch fee rate from TradeVolume API: {e}. Using fallback 0.30%")
            return 0.003
    
    def to_kraken_balance_key(self, coin: str) -> str:
        """Convert coin symbol to Kraken balance key format"""
        balance_key_map = {
            "BTC": "XXBT",
            "ETH": "XETH",
            "SOL": "SOL",
            "XRP": "XXRP",
            "USDT": "USDT",
            "USD": "ZUSD"
        }
        return balance_key_map.get(coin.upper(), coin.upper())

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
        self.rate_limit_requests: List[float] = []
        self.event_logs: List[Dict] = []

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
    """Save all users to persistent SQLite database"""
    try:
        db = SessionLocal()
        try:
            for username, user in users.items():
                user_state = user.state
                user_data = {
                    "password_hash": user.password_hash,
                    "created_at": user.created_at,
                    "exchange_type": user_state.exchange_type,
                    "api_key": cipher.encrypt(user_state.api_key.encode()).decode() if user_state.api_key else None,
                    "api_secret": cipher.encrypt(user_state.api_secret.encode()).decode() if user_state.api_secret else None,
                    "api_connected": user_state.api_connected,
                    "connection_error": user_state.connection_error,
                    "auto_trading_enabled": user_state.auto_trading_enabled,
                    "coin_trading_enabled": user_state.coin_trading_enabled,
                    "emergency_stop": user_state.emergency_stop,
                    "stop_loss_enabled": user_state.stop_loss_enabled,
                    "stop_loss_pct": user_state.stop_loss_pct,
                    "last_webhook": user_state.last_webhook,
                    "last_order": user_state.last_order,
                    "orders_history": user_state.orders_history,
                    "open_trades": user_state.open_trades,
                    "closed_trades": user_state.closed_trades,
                    "webhook_logs": user_state.webhook_logs,
                    "event_logs": user_state.event_logs,
                    "total_pnl": user_state.total_pnl,
                    "per_coin_pnl": user_state.per_coin_pnl
                }
                save_user_to_db(db, username, user_data)
            
            logger.info(f"Users saved to SQLite database")
            return True
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to save users to database: {str(e)}")
        return False

async def load_users_and_connect():
    """Load all users from SQLite database and reconnect"""
    logger.info(f"load_users_and_connect starting; loading from SQLite database")
    try:
        db = SessionLocal()
        try:
            users_data = load_all_users_from_db(db)
            
            if not users_data:
                logger.info("No users found in database, checking for legacy data")
                if os.path.exists(USERS_FILE):
                    logger.info(f"Found legacy users file at {USERS_FILE}, migrating to database")
                    with open(USERS_FILE, 'r') as f:
                        legacy_users = json.load(f)
                    for username, data in legacy_users.items():
                        save_user_to_db(db, username, data)
                    users_data = load_all_users_from_db(db)
                    logger.info(f"Migrated {len(users_data)} users from JSON to SQLite")
                else:
                    await migrate_legacy_state()
                    return
            
            for username, data in users_data.items():
                user = User(username, data["password_hash"])
                user.created_at = data.get("created_at", datetime.utcnow().isoformat())
                user_state = user.state
                
                user_state.exchange_type = data.get("exchange_type")
                user_state.api_connected = data.get("api_connected", False)
                user_state.connection_error = data.get("connection_error")
                user_state.auto_trading_enabled = data.get("auto_trading_enabled", False)
                user_state.coin_trading_enabled = data.get("coin_trading_enabled", {
                    "BTC": True, "ETH": True, "SOL": True, "XRP": True
                })
                user_state.emergency_stop = data.get("emergency_stop", False)
                user_state.stop_loss_enabled = data.get("stop_loss_enabled", True)
                user_state.stop_loss_pct = data.get("stop_loss_pct", 0.02)
                
                user_state.last_webhook = data.get("last_webhook")
                user_state.last_order = data.get("last_order")
                user_state.orders_history = data.get("orders_history", [])
                user_state.open_trades = data.get("open_trades", [])
                user_state.closed_trades = data.get("closed_trades", [])
                user_state.webhook_logs = data.get("webhook_logs", [])
                user_state.event_logs = data.get("event_logs", [])
                user_state.total_pnl = data.get("total_pnl", 0.0)
                user_state.per_coin_pnl = data.get("per_coin_pnl", {})
                
                encrypted_key = data.get("api_key")
                encrypted_secret = data.get("api_secret")
                
                if encrypted_key and encrypted_secret:
                    try:
                        user_state.api_key = cipher.decrypt(encrypted_key.encode()).decode()
                        user_state.api_secret = cipher.decrypt(encrypted_secret.encode()).decode()
                        
                        if user_state.exchange_type == "kraken":
                            user_state.kraken_client = KrakenClient(user_state.api_key, user_state.api_secret, username)
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
            
            logger.info(f"Loaded {len(users)} users from SQLite database")
        finally:
            db.close()
                    
    except Exception as e:
        logger.error(f"Failed to load users from database: {str(e)}")

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
                    admin_state.kraken_client = KrakenClient(admin_state.api_key, admin_state.api_secret, "admin")
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
    amount: Optional[Union[float, str]] = None
    quantity_usd: Optional[float] = None
    usd_amount: Optional[float] = None
    usd: Optional[float] = None
    sell_all: Optional[bool] = None
    timestamp: Optional[str] = None
    idempotency_key: Optional[str] = None

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
    username: str
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

def normalize_symbol(symbol: str, exchange_type: str, prefer_usd: bool = False) -> str:
    """
    Normalize symbol to exchange format.
    For Kraken: Can return pairs with /USD or /USDT depending on prefer_usd flag.
    """
    symbol = symbol.upper().replace("USDT", "").replace("USD", "")
    
    if exchange_type == "kraken":
        quote = "USD" if prefer_usd else "USDT"
        symbol_map = {
            "BTC": f"XBT/{quote}",
            "ETH": f"ETH/{quote}",
            "SOL": f"SOL/{quote}",
            "XRP": f"XRP/{quote}",
        }
        return symbol_map.get(symbol, f"{symbol}/{quote}")
    elif exchange_type == "coinbase":
        return f"{symbol}-USDT"
    else:
        return f"{symbol}/USDT"

user_hydration_locks: Dict[str, asyncio.Lock] = {}

async def ensure_user_exchange_client(username: str, validate: bool = False) -> Dict[str, Any]:
    """
    Ensure user has a valid exchange client, rebuilding from DB if needed.
    Returns: {connected: bool, exchange: str|None, error: str|None}
    """
    if username not in users:
        return {"connected": False, "exchange": None, "error": "User not found"}
    
    if username not in user_hydration_locks:
        user_hydration_locks[username] = asyncio.Lock()
    
    async with user_hydration_locks[username]:
        user_state = users[username].state
        
        should_hydrate = (
            not user_state.api_key or 
            not user_state.api_secret or 
            not user_state.exchange_type or
            (user_state.exchange_type == "kraken" and user_state.kraken_client is None) or
            (user_state.exchange_type and user_state.exchange_type != "kraken" and user_state.exchange is None)
        )
        
        if should_hydrate:
            logger.info(f"Hydration needed for {username}: keys_missing={not user_state.api_key or not user_state.api_secret}, exchange_type_missing={not user_state.exchange_type}, client_missing={user_state.kraken_client is None if user_state.exchange_type == 'kraken' else user_state.exchange is None}")
            
            if not user_state.api_key or not user_state.api_secret:
                logger.info(f"Loading API keys from DB for {username}")
                try:
                    db = SessionLocal()
                    user_db = db.query(UserDB).filter(UserDB.username == username).first()
                    db.close()
                    
                    if user_db and user_db.api_key and user_db.api_secret:
                        user_state.exchange_type = user_db.exchange_type
                        
                        if not user_state.exchange_type:
                            user_state.exchange_type = "kraken"
                            logger.info(f"Defaulting exchange_type to 'kraken' for {username} (was None in DB)")
                        
                        try:
                            user_state.api_key = cipher.decrypt(user_db.api_key.encode()).decode()
                            user_state.api_secret = cipher.decrypt(user_db.api_secret.encode()).decode()
                            logger.info(f"Successfully loaded encrypted API keys from DB for {username}")
                        except Exception as decrypt_err:
                            logger.warning(f"Decrypt failed for {username}, treating as plaintext: {decrypt_err}")
                            user_state.api_key = user_db.api_key
                            user_state.api_secret = user_db.api_secret
                            logger.info(f"Successfully loaded plaintext API keys from DB for {username}")
                    else:
                        logger.info(f"No API keys found in DB for {username}")
                        user_state.api_connected = False
                        user_state.connection_error = None
                        return {"connected": False, "exchange": None, "error": None}
                except Exception as e:
                    logger.error(f"Failed to load API keys from DB for {username}: {str(e)}")
                    user_state.api_connected = False
                    user_state.connection_error = None
                    return {"connected": False, "exchange": None, "error": None}
            
            if not user_state.exchange_type and user_state.api_key and user_state.api_secret:
                user_state.exchange_type = "kraken"
                logger.info(f"Defaulting exchange_type to 'kraken' for {username} (keys exist but exchange_type was None)")
        
        if not user_state.exchange_type:
            user_state.api_connected = False
            user_state.connection_error = "No exchange type configured"
            return {"connected": False, "exchange": None, "error": "No exchange type configured"}
        
        needs_rebuild = False
        if user_state.exchange_type == "kraken":
            if user_state.kraken_client is None:
                needs_rebuild = True
        elif user_state.exchange_type:
            if user_state.exchange is None:
                needs_rebuild = True
        
        # Rebuild client if needed
        if needs_rebuild:
            try:
                logger.info(f"Lazy-hydrating {user_state.exchange_type} client for user {username}")
                if user_state.exchange_type == "kraken":
                    user_state.kraken_client = KrakenClient(user_state.api_key, user_state.api_secret, username)
                    if validate:
                        await user_state.kraken_client.get_balance()
                    user_state.api_connected = True
                    user_state.connection_error = None
                    logger.info(f"Successfully hydrated Kraken client for user {username}")
                else:
                    user_state.exchange = get_exchange_instance(
                        user_state.exchange_type,
                        user_state.api_key,
                        user_state.api_secret
                    )
                    if validate:
                        user_state.exchange.fetch_balance()
                    user_state.api_connected = True
                    user_state.connection_error = None
                    logger.info(f"Successfully hydrated {user_state.exchange_type} client for user {username}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to hydrate client for user {username}: {error_msg}")
                user_state.api_connected = False
                user_state.connection_error = error_msg
                return {"connected": False, "exchange": user_state.exchange_type, "error": error_msg}
        
        return {
            "connected": user_state.api_connected,
            "exchange": user_state.exchange_type,
            "error": user_state.connection_error
        }

async def execute_webhook_for_user(username: str, user_state: UserState, alert: WebhookAlert, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Execute webhook for a single user with error handling and timeout.
    Returns a dict with execution result for logging.
    """
    result = {
        "username": username,
        "timestamp": now_et_iso(),
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
            action = alert.action.lower()
            
            if action in ["buy", "long"]:
                side = "buy"
            elif action in ["sell", "short", "close"]:
                side = "sell"
            else:
                raise ValueError(f"Invalid action: {action}")
            
            normalized_usd_amount = None
            if alert.usd is not None:
                normalized_usd_amount = float(alert.usd)
            elif alert.usd_amount is not None:
                normalized_usd_amount = float(alert.usd_amount)
            elif alert.quantity_usd is not None:
                normalized_usd_amount = float(alert.quantity_usd)
            
            normalized_amount = None
            if alert.amount is not None:
                if isinstance(alert.amount, str) and alert.amount.lower() == "all":
                    normalized_amount = "all"
                else:
                    normalized_amount = float(alert.amount)
            
            amount = 0.001  # default
            
            if user_state.exchange_type == "kraken" and user_state.kraken_client:
                balance_data = await user_state.kraken_client.get_balance()
                usdt_balance = float(balance_data.get('USDT', 0))
                usd_balance = float(balance_data.get('ZUSD', 0))
                
                prefer_usd = (usdt_balance < 10 and usd_balance >= 10)
                if prefer_usd:
                    logger.info(f"Auto-switching to USD pairs: USDT balance ${usdt_balance:.2f} < $10, USD balance ${usd_balance:.2f} >= $10")
                else:
                    logger.info(f"Using USDT pairs: USDT balance ${usdt_balance:.2f}, USD balance ${usd_balance:.2f}")
                
                symbol = normalize_symbol(alert.symbol, user_state.exchange_type, prefer_usd=prefer_usd)
                kraken_pair = user_state.kraken_client.to_kraken_pair(symbol)
                
                logger.info(f"Symbol conversion: alert.symbol={alert.symbol} -> normalized={symbol} -> kraken_pair={kraken_pair}")
                logger.info(f"Kraken balance keys: {list(balance_data.keys())}, coin={coin}")
                
                if alert.sell_all or normalized_amount == "all":
                    coin_key = user_state.kraken_client.to_kraken_balance_key(coin)
                    coin_balance = float(balance_data.get(coin_key, 0))
                    logger.info(f"Sell all: coin={coin}, coin_key={coin_key}, balance={coin_balance:.8f}")
                    amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, coin_balance, side)
                elif normalized_usd_amount:
                    if side == "buy":
                        if alert.price:
                            price_used = float(alert.price)
                        else:
                            price_used = await user_state.kraken_client.get_ticker_price(kraken_pair)
                        amount = await calculate_fee_aware_buy_volume(user_state.kraken_client, kraken_pair, normalized_usd_amount, price_used, balance_data)
                    else:
                        if alert.price:
                            price_used = float(alert.price)
                            raw_amount = normalized_usd_amount / price_used
                            logger.info(f"USD conversion (sell): ${normalized_usd_amount} / ${price_used} = {raw_amount:.8f} {coin}")
                        else:
                            current_price = await user_state.kraken_client.get_ticker_price(kraken_pair)
                            raw_amount = normalized_usd_amount / current_price
                            logger.info(f"USD conversion (sell): ${normalized_usd_amount} / ${current_price} (live) = {raw_amount:.8f} {coin}")
                        amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, raw_amount, side)
                elif normalized_amount and normalized_amount != "all":
                    raw_amount = float(normalized_amount)
                    amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, raw_amount, side)
                elif alert.quantity == "all" or (isinstance(alert.quantity, str) and alert.quantity.lower() == "all"):
                    coin_key = user_state.kraken_client.to_kraken_balance_key(coin)
                    coin_balance = float(balance_data.get(coin_key, 0))
                    logger.info(f"Quantity all: coin={coin}, coin_key={coin_key}, balance={coin_balance:.8f}")
                    amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, coin_balance, side)
                elif normalized_usd_amount:
                    if side == "buy":
                        if alert.price:
                            price_used = float(alert.price)
                        else:
                            price_used = await user_state.kraken_client.get_ticker_price(kraken_pair)
                        amount = await calculate_fee_aware_buy_volume(user_state.kraken_client, kraken_pair, normalized_usd_amount, price_used, balance_data)
                    else:
                        if alert.price:
                            price_used = float(alert.price)
                            raw_amount = normalized_usd_amount / price_used
                            logger.info(f"USD conversion (sell): ${normalized_usd_amount} / ${price_used} = {raw_amount:.8f} {coin}")
                        else:
                            current_price = await user_state.kraken_client.get_ticker_price(kraken_pair)
                            raw_amount = normalized_usd_amount / current_price
                            logger.info(f"USD conversion (sell): ${normalized_usd_amount} / ${current_price} (live) = {raw_amount:.8f} {coin}")
                        amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, raw_amount, side)
                elif alert.quantity:
                    raw_amount = float(alert.quantity)
                    amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, raw_amount, side)
                elif usdt_balance > 0:
                    position_size_usdt = calculate_position_size(usdt_balance, 0.02)
                    if side == "buy":
                        if alert.price:
                            price_used = float(alert.price)
                        else:
                            price_used = await user_state.kraken_client.get_ticker_price(kraken_pair)
                        amount = await calculate_fee_aware_buy_volume(user_state.kraken_client, kraken_pair, position_size_usdt, price_used, balance_data)
                    else:
                        if alert.price:
                            raw_amount = position_size_usdt / float(alert.price)
                        else:
                            current_price = await user_state.kraken_client.get_ticker_price(kraken_pair)
                            raw_amount = position_size_usdt / current_price
                        amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, raw_amount, side)
                else:
                    amount = await round_and_enforce_kraken_volume(user_state.kraken_client, kraken_pair, alert.symbol, 0.001, side)
                
                if amount == 0:
                    logger.warning(f"Skipping order: volume is 0 (likely dust balance below ordermin)")
                    result["status"] = "skipped"
                    result["error"] = "Volume below minimum order size"
                    return result
                
                order = None
                max_retries = 3
                kraken_pair = user_state.kraken_client.to_kraken_pair(alert.symbol)
                
                for attempt in range(max_retries):
                    try:
                        if alert.price:
                            price = float(alert.price)
                            logger.info(f"Placing LIMIT order: pair={kraken_pair}, side={side}, volume={amount:.8f}, price={price}")
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
                            logger.info(f"Placing MARKET order: pair={kraken_pair}, side={side}, volume={amount:.8f}")
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
                        logger.error(f"Order placement attempt {attempt + 1} failed: {str(e)}")
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
                elif normalized_usd_amount:
                    if alert.price:
                        amount = normalized_usd_amount / float(alert.price)
                    else:
                        ticker = user_state.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        amount = normalized_usd_amount / current_price
                elif alert.quantity == "all" or (isinstance(alert.quantity, str) and alert.quantity.lower() == "all"):
                    coin_balance = balance.get(coin, {}).get('free', 0)
                    amount = coin_balance
                elif normalized_usd_amount:
                    if alert.price:
                        amount = normalized_usd_amount / float(alert.price)
                    else:
                        ticker = user_state.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        amount = normalized_usd_amount / current_price
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
                "timestamp": now_et_iso(),
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

@app.get("/version")
def version():
    import subprocess
    try:
        commit_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd='/opt/render/project/src').decode().strip()
    except:
        commit_sha = "unknown"
    return {"version": "v1.2.0-auto-migration", "commit": commit_sha, "timestamp": datetime.utcnow().isoformat() + "Z"}

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
    """Login with username and password"""
    username = request.username.strip()
    password = request.password
    
    if username not in users:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user = users[username]
    if not verify_password_bcrypt(password, user.password_hash):
        log_event(user.state, "login", "failed", error="Invalid password")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_jwt_token(username)
    log_event(user.state, "login", "success", details={"username": username})
    
    return {
        "success": True,
        "token": token,
        "username": username,
        "message": "Login successful"
    }

@app.post("/user/login")
async def user_login(request: UserLoginRequest):
    """User login with username and password"""
    username = request.username.strip()
    password = request.password
    
    if username not in users:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user = users[username]
    if not verify_password_bcrypt(password, user.password_hash):
        log_event(user.state, "login", "failed", error="Invalid password")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_jwt_token(username)
    log_event(user.state, "login", "success", details={"username": username})
    
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
                kraken_client = KrakenClient(api_key, api_secret, username)
                
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
                
                log_event(user_state, "settings_update", "success", details={"exchange": exchange, "api_connected": True})
                save_users()
                
                return {
                    "success": True,
                    "message": "Kraken API connected successfully",
                    "exchange": exchange,
                    "connected": True
                }
                
            except Exception as e:
                error_msg = str(e)
                log_event(user_state, "settings_update", "failed", details={"exchange": exchange}, error=error_msg)
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
    
    conn_status = await ensure_user_exchange_client(username, validate=False)
    
    user_state = users[username].state
    return {
        "exchange": conn_status["exchange"],
        "connected": conn_status["connected"],
        "has_api_key": user_state.api_key is not None,
        "error": conn_status["error"]
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
        
        if alert.idempotency_key:
            current_time = datetime.now().timestamp()
            
            if alert.idempotency_key in processed_idempotency_keys:
                cached_result = processed_idempotency_keys[alert.idempotency_key]
                if current_time - cached_result['timestamp'] < IDEMPOTENCY_TTL_SECONDS:
                    logger.info(f"Duplicate webhook detected with idempotency_key: {alert.idempotency_key}")
                    return {
                        "success": True,
                        "message": "Duplicate request - returning cached result",
                        "executed": False,
                        "duplicate": True,
                        "cached_result": cached_result['result']
                    }
                else:
                    del processed_idempotency_keys[alert.idempotency_key]
        
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
                    "timestamp": now_et_iso(),
                    "payload": alert.dict()
                }
        
        save_users()
        
        executed_count = sum(1 for r in execution_results if r.get("executed"))
        skipped_count = sum(1 for r in execution_results if r.get("status") == "skipped")
        failed_count = sum(1 for r in execution_results if r.get("status") == "failed")
        
        logger.info(f"Webhook broadcast complete: {executed_count} executed, {skipped_count} skipped, {failed_count} failed")
        
        result = {
            "success": True,
            "message": f"Webhook broadcast complete: {executed_count} executed, {skipped_count} skipped, {failed_count} failed",
            "executed": executed_count > 0,
            "results": execution_results
        }
        
        if alert.idempotency_key:
            processed_idempotency_keys[alert.idempotency_key] = {
                'timestamp': datetime.now().timestamp(),
                'result': result
            }
            logger.info(f"Cached result for idempotency_key: {alert.idempotency_key}")
        
        return result
        
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
            "timestamp": now_et_iso(),
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
                "timestamp": now_et_iso(),
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
    conn_status = {"connected": False, "exchange": None, "error": None}
    
    try:
        logger.info(f"get_status called for user: {username}")
        
        if username not in users:
            raise HTTPException(status_code=401, detail="User not found")
        
        conn_status = await ensure_user_exchange_client(username, validate=False)
        api_connected = conn_status["connected"]
        exchange = conn_status["exchange"]
        
        user_state = users[username].state
        
        try:
            # Normalize lists to ensure they're valid and contain only dicts
            closed_trades = [t for t in (user_state.closed_trades or []) if isinstance(t, dict)]
            orders_history = [o for o in (user_state.orders_history or []) if isinstance(o, dict)]
            open_trades = [t for t in (user_state.open_trades or []) if isinstance(t, dict)]
            webhook_logs = [w for w in (user_state.webhook_logs or []) if isinstance(w, dict)]
            
            total_trades = len(closed_trades)
            winning_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in closed_trades if t.get('pnl', 0) < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_trade_size = sum([o.get('amount', 0) for o in orders_history]) / len(orders_history) if orders_history else 0
            
            total_exposure = sum([t.get('amount', 0) * t.get('entry_price', 0) for t in open_trades])
            
            pnl_summary = {
                "total_orders": len(orders_history),
                "total_pnl": float(user_state.total_pnl) if user_state.total_pnl else 0.0,
                "per_coin_pnl": {k: float(v) for k, v in (user_state.per_coin_pnl or {}).items()},
                "recent_orders": orders_history[-10:],
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": float(win_rate),
                "avg_trade_size": float(avg_trade_size),
                "total_exposure": float(total_exposure)
            }
        except Exception as stats_error:
            logger.warning(f"Stats calculation failed for user {username}: {str(stats_error)}, using defaults")
            closed_trades = []
            open_trades = []
            webhook_logs = []
            pnl_summary = {
                "total_orders": 0,
                "total_pnl": 0.0,
                "per_coin_pnl": {},
                "recent_orders": [],
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_trade_size": 0,
                "total_exposure": 0
            }
        
        payload = {
            "api_connected": api_connected,
            "exchange": exchange,
            "auto_trading_enabled": user_state.auto_trading_enabled if user_state.auto_trading_enabled is not None else False,
            "emergency_stop": user_state.emergency_stop if user_state.emergency_stop is not None else False,
            "stop_loss_enabled": user_state.stop_loss_enabled if user_state.stop_loss_enabled is not None else True,
            "stop_loss_pct": float(user_state.stop_loss_pct) if user_state.stop_loss_pct is not None else 0.02,
            "coin_trading_enabled": user_state.coin_trading_enabled if user_state.coin_trading_enabled else {"BTC": True, "ETH": True, "SOL": True, "XRP": True},
            "last_webhook": user_state.last_webhook,
            "last_order": user_state.last_order,
            "pnl_summary": pnl_summary,
            "open_trades": open_trades[-20:],
            "closed_trades": closed_trades[-20:],
            "webhook_logs": webhook_logs[-50:]
        }
        
        logger.info(f"get_status returning payload for user {username}: api_connected={api_connected}, exchange={exchange}")
        return JSONResponse(content=jsonable_encoder(payload, custom_encoder={Decimal: float}))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in get_status for user {username}: {str(e)}")
        fallback_payload = {
            "api_connected": conn_status["connected"],
            "exchange": conn_status["exchange"],
            "auto_trading_enabled": False,
            "emergency_stop": False,
            "stop_loss_enabled": True,
            "stop_loss_pct": 0.02,
            "coin_trading_enabled": {"BTC": True, "ETH": True, "SOL": True, "XRP": True},
            "last_webhook": None,
            "last_order": None,
            "pnl_summary": {
                "total_orders": 0,
                "total_pnl": 0.0,
                "per_coin_pnl": {},
                "recent_orders": [],
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_trade_size": 0,
                "total_exposure": 0
            },
            "open_trades": [],
            "closed_trades": [],
            "webhook_logs": []
        }
        return JSONResponse(content=jsonable_encoder(fallback_payload, custom_encoder={Decimal: float}))

@app.get("/api/diagnostics/balance")
async def get_balance_diagnostics(username: str = Depends(verify_session)):
    """Diagnostic endpoint to check balance and pair info"""
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_state = users[username].state
    
    if not user_state.kraken_client:
        raise HTTPException(status_code=400, detail="Kraken client not configured")
    
    try:
        balance_data = await user_state.kraken_client.get_balance()
        
        pair_info_btc = await user_state.kraken_client.get_asset_pairs("XBTUSDT")
        pair_info_eth = await user_state.kraken_client.get_asset_pairs("ETHUSDT")
        
        all_balances = {key: float(value) for key, value in balance_data.items()}
        
        return {
            "balance_keys": list(balance_data.keys()),
            "all_balances": all_balances,
            "balances": {
                "USDT": float(balance_data.get('USDT', 0)),
                "ZUSD": float(balance_data.get('ZUSD', 0)),
                "XXBT": float(balance_data.get('XXBT', 0)),
                "XETH": float(balance_data.get('XETH', 0)),
                "USD.HOLD": float(balance_data.get('USD.HOLD', 0))
            },
            "pair_info": {
                "XBTUSDT": {
                    "quote": pair_info_btc.get('quote'),
                    "ordermin": pair_info_btc.get('ordermin'),
                    "lot_decimals": pair_info_btc.get('lot_decimals'),
                    "pair_decimals": pair_info_btc.get('pair_decimals')
                },
                "ETHUSDT": {
                    "quote": pair_info_eth.get('quote'),
                    "ordermin": pair_info_eth.get('ordermin'),
                    "lot_decimals": pair_info_eth.get('lot_decimals'),
                    "pair_decimals": pair_info_eth.get('pair_decimals')
                }
            },
            "balance_key_mapping": {
                "USDT": user_state.kraken_client.to_kraken_balance_key("USDT"),
                "USD": user_state.kraken_client.to_kraken_balance_key("USD"),
                "BTC": user_state.kraken_client.to_kraken_balance_key("BTC"),
                "ETH": user_state.kraken_client.to_kraken_balance_key("ETH")
            }
        }
    except Exception as e:
        logger.exception(f"Balance diagnostics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                                closed_trade['closed_at'] = now_et_iso()
                                
                                user_state.closed_trades.append(closed_trade)
                                user_state.open_trades.remove(trade)
                                
                                coin = trade.get('coin', '')
                                if coin:
                                    user_state.per_coin_pnl[coin] = user_state.per_coin_pnl.get(coin, 0) + pnl
                                user_state.total_pnl += pnl
                                
                                webhook_log = {
                                    "timestamp": now_et_iso(),
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
                                "timestamp": now_et_iso(),
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
    logger.info("startup_event begin - initializing database and loading users")
    init_db()
    logger.info("Database initialized successfully")
    await load_users_and_connect()
    asyncio.create_task(monitor_stop_loss())
    logger.info(f"Background tasks started - monitoring {len(users)} users")
