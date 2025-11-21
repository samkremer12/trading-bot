import os
import json
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from sqlalchemy import create_engine, Column, String, Boolean, Float, Text, Integer, BigInteger, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import ProgrammingError

logger = logging.getLogger(__name__)

if os.environ.get("DATABASE_URL"):
    DATABASE_URL = os.environ.get("DATABASE_URL")
    logger.info("Using managed Postgres database for persistent storage")
    engine = create_engine(DATABASE_URL)
elif os.path.isdir("/data"):
    DB_PATH = "/data/trading_bot.db"
    TEMP_DB_PATH = "/tmp/trading_bot.db"
    
    if not os.path.exists(DB_PATH) and os.path.exists(TEMP_DB_PATH):
        logger.info(f"Migrating existing database from {TEMP_DB_PATH} to {DB_PATH}")
        try:
            shutil.copy2(TEMP_DB_PATH, DB_PATH)
            logger.info(f"Successfully migrated database to {DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to migrate database: {e}")
    
    logger.info(f"Using persistent storage: {DB_PATH}")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
elif os.environ.get("PERSISTENCE_DIR"):
    DB_PATH = os.path.join(os.environ.get("PERSISTENCE_DIR"), "trading_bot.db")
    logger.info(f"Using custom persistent storage: {DB_PATH}")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    DB_PATH = "/tmp/trading_bot.db"
    logger.warning(f"Using ephemeral storage: {DB_PATH} (data will not survive full restarts)")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    
    username = Column(String, primary_key=True, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    
    api_key = Column(String, nullable=True)
    api_secret = Column(String, nullable=True)
    exchange_type = Column(String, nullable=True)
    api_connected = Column(Boolean, default=False)
    connection_error = Column(String, nullable=True)
    
    auto_trading_enabled = Column(Boolean, default=False)
    emergency_stop = Column(Boolean, default=False)
    stop_loss_enabled = Column(Boolean, default=True)
    stop_loss_pct = Column(Float, default=0.02)
    
    coin_trading_enabled = Column(Text, nullable=False)
    
    last_webhook = Column(Text, nullable=True)
    last_order = Column(Text, nullable=True)
    orders_history = Column(Text, default="[]")
    open_trades = Column(Text, default="[]")
    closed_trades = Column(Text, default="[]")
    webhook_logs = Column(Text, default="[]")
    event_logs = Column(Text, default="[]")
    
    total_pnl = Column(Float, default=0.0)
    per_coin_pnl = Column(Text, default="{}")
    
    last_nonce = Column(BigInteger, default=0)

def ensure_schema():
    """
    Ensure database schema is up-to-date by adding missing columns.
    This runs idempotently on startup to handle schema migrations.
    """
    try:
        dialect_name = engine.dialect.name
        logger.info(f"Running schema migration check for {dialect_name} database...")
        
        with engine.begin() as conn:
            if dialect_name == 'postgresql':
                conn.execute(text(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_nonce BIGINT DEFAULT 0"
                ))
                logger.info("PostgreSQL: Ensured last_nonce column exists")
            
            elif dialect_name == 'sqlite':
                result = conn.execute(text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result]
                
                if 'last_nonce' not in columns:
                    conn.execute(text(
                        "ALTER TABLE users ADD COLUMN last_nonce BIGINT DEFAULT 0"
                    ))
                    logger.info("SQLite: Added last_nonce column")
                else:
                    logger.info("SQLite: last_nonce column already exists")
            
            else:
                logger.warning(f"Unknown dialect {dialect_name}, skipping schema migration")
        
        logger.info("Schema migration completed successfully")
        
    except Exception as e:
        logger.error(f"Schema migration failed: {e}")

def init_db():
    Base.metadata.create_all(bind=engine)
    ensure_schema()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_user_to_db(db: Session, username: str, user_data: Dict[str, Any]):
    user_db = db.query(UserDB).filter(UserDB.username == username).first()
    
    if user_db is None:
        user_db = UserDB(
            username=username,
            password_hash=user_data["password_hash"],
            created_at=user_data.get("created_at", datetime.utcnow().isoformat()),
            coin_trading_enabled=json.dumps(user_data.get("coin_trading_enabled", {
                "BTC": True, "ETH": True, "SOL": True, "XRP": True
            }))
        )
        db.add(user_db)
    
    user_db.api_key = user_data.get("api_key")
    user_db.api_secret = user_data.get("api_secret")
    user_db.exchange_type = user_data.get("exchange_type")
    user_db.api_connected = user_data.get("api_connected", False)
    user_db.connection_error = user_data.get("connection_error")
    
    user_db.auto_trading_enabled = user_data.get("auto_trading_enabled", False)
    user_db.emergency_stop = user_data.get("emergency_stop", False)
    user_db.stop_loss_enabled = user_data.get("stop_loss_enabled", True)
    user_db.stop_loss_pct = user_data.get("stop_loss_pct", 0.02)
    
    user_db.coin_trading_enabled = json.dumps(user_data.get("coin_trading_enabled", {
        "BTC": True, "ETH": True, "SOL": True, "XRP": True
    }))
    
    user_db.last_webhook = json.dumps(user_data.get("last_webhook")) if user_data.get("last_webhook") else None
    user_db.last_order = json.dumps(user_data.get("last_order")) if user_data.get("last_order") else None
    user_db.orders_history = json.dumps(user_data.get("orders_history", []))
    user_db.open_trades = json.dumps(user_data.get("open_trades", []))
    user_db.closed_trades = json.dumps(user_data.get("closed_trades", []))
    user_db.webhook_logs = json.dumps(user_data.get("webhook_logs", []))
    user_db.event_logs = json.dumps(user_data.get("event_logs", []))
    
    user_db.total_pnl = user_data.get("total_pnl", 0.0)
    user_db.per_coin_pnl = json.dumps(user_data.get("per_coin_pnl", {}))
    
    db.commit()
    db.refresh(user_db)
    return user_db

def load_user_from_db(db: Session, username: str) -> Optional[Dict[str, Any]]:
    user_db = db.query(UserDB).filter(UserDB.username == username).first()
    if user_db is None:
        return None
    
    return {
        "password_hash": user_db.password_hash,
        "created_at": user_db.created_at,
        "api_key": user_db.api_key,
        "api_secret": user_db.api_secret,
        "exchange_type": user_db.exchange_type,
        "api_connected": user_db.api_connected,
        "connection_error": user_db.connection_error,
        "auto_trading_enabled": user_db.auto_trading_enabled,
        "emergency_stop": user_db.emergency_stop,
        "stop_loss_enabled": user_db.stop_loss_enabled,
        "stop_loss_pct": user_db.stop_loss_pct,
        "coin_trading_enabled": json.loads(user_db.coin_trading_enabled),
        "last_webhook": json.loads(user_db.last_webhook) if user_db.last_webhook else None,
        "last_order": json.loads(user_db.last_order) if user_db.last_order else None,
        "orders_history": json.loads(user_db.orders_history),
        "open_trades": json.loads(user_db.open_trades),
        "closed_trades": json.loads(user_db.closed_trades),
        "webhook_logs": json.loads(user_db.webhook_logs),
        "event_logs": json.loads(user_db.event_logs),
        "total_pnl": user_db.total_pnl,
        "per_coin_pnl": json.loads(user_db.per_coin_pnl)
    }

def load_all_users_from_db(db: Session) -> Dict[str, Dict[str, Any]]:
    users_db = db.query(UserDB).all()
    users_data = {}
    for user_db in users_db:
        users_data[user_db.username] = {
            "password_hash": user_db.password_hash,
            "created_at": user_db.created_at,
            "api_key": user_db.api_key,
            "api_secret": user_db.api_secret,
            "exchange_type": user_db.exchange_type,
            "api_connected": user_db.api_connected,
            "connection_error": user_db.connection_error,
            "auto_trading_enabled": user_db.auto_trading_enabled,
            "emergency_stop": user_db.emergency_stop,
            "stop_loss_enabled": user_db.stop_loss_enabled,
            "stop_loss_pct": user_db.stop_loss_pct,
            "coin_trading_enabled": json.loads(user_db.coin_trading_enabled),
            "last_webhook": json.loads(user_db.last_webhook) if user_db.last_webhook else None,
            "last_order": json.loads(user_db.last_order) if user_db.last_order else None,
            "orders_history": json.loads(user_db.orders_history),
            "open_trades": json.loads(user_db.open_trades),
            "closed_trades": json.loads(user_db.closed_trades),
            "webhook_logs": json.loads(user_db.webhook_logs),
            "event_logs": json.loads(user_db.event_logs),
            "total_pnl": user_db.total_pnl,
            "per_coin_pnl": json.loads(user_db.per_coin_pnl)
        }
    return users_data

def get_and_increment_nonce(db: Session, username: str, retry_on_schema_error: bool = True) -> int:
    """
    Get and atomically increment the nonce for a user.
    Uses microseconds and ensures strictly increasing nonces across restarts.
    
    Self-healing: If the last_nonce column is missing, runs schema migration and retries once.
    """
    import time
    
    try:
        user_db = db.query(UserDB).filter(UserDB.username == username).with_for_update().first()
        
        if user_db is None:
            raise Exception(f"User {username} not found")
        
        now_us = int(time.time() * 1_000_000)
        
        new_nonce = max(now_us, (user_db.last_nonce or 0) + 1)
        
        user_db.last_nonce = new_nonce
        db.commit()
        
        return new_nonce
        
    except ProgrammingError as e:
        error_str = str(e).lower()
        if 'last_nonce' in error_str and 'does not exist' in error_str:
            if retry_on_schema_error:
                logger.warning(f"last_nonce column missing, running schema migration and retrying...")
                db.rollback()
                ensure_schema()
                return get_and_increment_nonce(db, username, retry_on_schema_error=False)
            else:
                logger.error("Schema migration retry failed, column still missing")
                raise
        else:
            raise
