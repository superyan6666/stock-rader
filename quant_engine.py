# 存储路径: quant_engine.py
# 🤖 QuantBot 3.0 (Singularity Edition) - 全栈单体引擎
import os
import sys
import time
import json
import logging
import threading
import uuid
import argparse
import unittest
import re
import sqlite3
import math        # 🚀 修复: 补充缺失的数学依赖
import random      # 🚀 修复: 补充缺失的随机数依赖
import pandas as pd
import numpy as np
import scipy.stats as stats
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass, field

# ================= 0. 环境嗅探与全局依赖 =================
warnings = __import__('warnings')
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

try:
    import yfinance as yf
    import requests
except ImportError:
    print("❌ 缺少基础依赖，请执行: pip install yfinance requests pandas numpy scipy scikit-learn lightgbm")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

try:
    from scipy.stats import gaussian_kde
    KDE_AVAILABLE = True
except ImportError:
    KDE_AVAILABLE = False

# ================= 1. 全局架构与铁律配置 (Config) =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("QuantEngine")

def datetime_to_str():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

class Config:
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    DINGTALK_KEYWORD: str = "AI"
    INDEX_ETF: str = "QQQ" 
    VIX_INDEX: str = "^VIX" 
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'TXN'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD'],
        'XLC': ['GOOGL', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'IDXX'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'WBA']
    }
    
    DATA_DIR: str = ".quantbot_data"
    STATS_FILE: str = os.path.join(DATA_DIR, "strategy_stats.json")
    REPORT_FILE: str = os.path.join(DATA_DIR, "backtest_report.md")
    CACHE_FILE: str = os.path.join(DATA_DIR, "tickers_cache.json")
    ALERT_CACHE_FILE: str = os.path.join(DATA_DIR, "alert_history.json")
    MODEL_FILE: str = os.path.join(DATA_DIR, "scoring_model.pkl")
    WSB_CACHE_FILE: str = os.path.join(DATA_DIR, "wsb_history.json")  
    EXT_CACHE_FILE: str = os.path.join(DATA_DIR, "daily_ext_cache.json")
    ORDER_DB_PATH: str = os.path.join(DATA_DIR, "order_state.db") 
    TCA_LOG_PATH: str = os.path.join(DATA_DIR, "tca_history.jsonl")
    TRAINING_BUFFER: str = os.path.join(DATA_DIR, "training_buffer.npz")
    TRANSFORMER_PTH: str = os.path.join(DATA_DIR, "transformer_production.pth")
    MODEL_VERSION: str = "1.0" 

    CORE_WATCHLIST: list = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "QCOM",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "PG", "HD", "COST", "ABBV", "CRM", "NFLX", 
        "ADBE", "NOW", "UBER", "INTU", "IBM", "ISRG", "SYK", "SPY", "QQQ", "IWM", "DIA"
    ]
    CROWDING_EXCLUDE_SECTORS: list = [INDEX_ETF] 

    CORE_FACTORS = [
        "米奈尔维尼", "强相对强度", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)",
        "突破缺口", "VWAP突破", "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", 
        "口袋支点", "VCP收缩", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "跨时空共振(周线)"
    ]
    ADVANCED_MATH_FACTORS = [
        "量子概率云(KDE)", "稳健赫斯特(Hurst)", "FFT多窗共振(动能)", "CVD筹码净流入", "独立Alpha(脱钩)", 
        "NR7极窄突破", "VPT量价共振", "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)",
        "Amihud非流动性(冲击成本)", "波动率风险溢价(VRP)"
    ]
    INTERACTION_FACTORS = [
        "带量金叉(交互)", "量价吸筹(交互)", "近3日突破(滞后)", "近3日巨量(滞后)", 
        "大周期保护小周期(MACD共振)", "60分钟级精准校准(RSI反弹)", "52周高点距离(动能延续)"
    ]
    ALT_DATA_FACTORS = [
        "聪明钱月度净流入(月线)", "期权PutCall情绪(PCR)", "隐含波动率偏度(IV Skew)", "做空兴趣突变(轧空)",
        "内部人集群净买入(Insider)", "分析师修正动量(Analyst)", "舆情NLP情感极值(News_NLP)", "散户热度加速度(WSB_Accel)"
    ]
    TRANSFORMER_FACTORS = [f"Alpha_T{i:02d}" for i in range(1, 17)]

    ALL_FACTORS = CORE_FACTORS + ADVANCED_MATH_FACTORS + INTERACTION_FACTORS + ALT_DATA_FACTORS + TRANSFORMER_FACTORS
    GROUP_A_FACTORS = CORE_FACTORS + ["聪明钱月度净流入(月线)", "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)", "52周高点距离(动能延续)", "内部人集群净买入(Insider)", "分析师修正动量(Analyst)"] + [f for f in INTERACTION_FACTORS if "共振" in f or "滞后" in f]
    GROUP_B_FACTORS = ADVANCED_MATH_FACTORS + ALT_DATA_FACTORS + [f for f in CORE_FACTORS if "扫盘" in f or "失衡" in f or "抢筹" in f] + TRANSFORMER_FACTORS

    class Params:
        MAX_WORKERS = 8
        MIN_SCORE_THRESHOLD = 8
        BASE_MAX_RISK = 0.015       
        CROWDING_PENALTY = 0.75     
        CROWDING_MIN_STOCKS = 2     
        PORTFOLIO_VALUE = 100000.0  
        ALERT_COOLDOWN_HOURS = 24.0 
        SLIPPAGE = 0.003            
        COMMISSION = 0.0005         
        MIN_T_STAT = 1.0            
        PCR_BEAR, PCR_BULL = 1.5, 0.5
        IV_SKEW_BEAR, IV_SKEW_BULL = 0.08, -0.05
        SHORT_SQZ_CHG, SHORT_SQZ_FLT = 0.10, 0.05
        VRP_EXTREME = 0.25       
        WSB_ACCEL = 20.0         
        ANALYST_UP, ANALYST_DN = 1.5, -1.5
        NLP_BULL, NLP_BEAR = 0.5, -0.5
        INSIDER_BUY = 0.5        
        AMIHUD_ILLIQ = 0.5       
        DIST_52W = -0.05         
        KDE_BREAKOUT = 0.5       
        HURST_RELIABLE = 0.65    
        FFT_RESONANCE = 0.71     
        VCP_BULL, VCP_BEAR = 0.5, 0.4
        MACD_WATERLINE = 0.0     
        FAT_FINGER_MAX_USD = 50000.0 
        MAX_RETRY_COUNT = 3

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF

def initialize_directories():
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_DIR, "archive"), exist_ok=True)

# ================= 2. 线程安全缓存与数据抽象化模型 =================
_KLINE_CACHE, _KLINE_LOCK = {}, threading.Lock()
_DAILY_EXT_CACHE, _DAILY_EXT_LOCK = {}, threading.Lock()
_ALT_DATA_CACHE, _ALT_DATA_LOCK = {}, threading.Lock()

_GLOBAL_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

@dataclass
class MarketContext:
    regime: str; regime_desc: str; w_mul: float; xai_weights: dict
    vix_current: float; vix_desc: str; vix_scalar: float; max_risk: float
    macro_gravity: bool; is_credit_risk_high: bool; vix_inv: bool
    qqq_df: pd.DataFrame; macro_data: dict; total_market_exposure: float
    health_score: float; pain_warning: str
    credit_spread_mom: float = 0.0; vix_term_structure: float = 1.0; market_pcr: float = 1.0
    dynamic_min_score: float = 8.0; global_wsb_data: dict = field(default_factory=dict)
    meta_weights: dict = field(default_factory=dict); transformer_model: 'Any' = None  

@dataclass
class StockData:
    sym: str; df: pd.DataFrame; df_w: pd.DataFrame; df_m: pd.DataFrame; df_60m: pd.DataFrame
    curr: pd.Series; prev: pd.Series; is_vol: bool; swing_high_10: float

@dataclass
class AltData:
    pcr: float; iv_skew: float; short_change: float; short_float: float
    insider_net_buy: float; analyst_mom: float; nlp_score: float; wsb_accel: float

@dataclass
class ComplexFeatures:
    weekly_bullish: bool; fvg_lower: float; fvg_upper: float; kde_breakout_score: float
    fft_ensemble_score: float; hurst_med: float; hurst_iqr: float; hurst_reliable: bool
    monthly_inst_flow: float; weekly_macd_res: float; rsi_60m_bounce: float
    beta_60d: float; tlt_corr: float; dxy_corr: float; vrp: float; rs_20: float; pure_alpha: float 

# ================= 3. 底层运算库与网络请求 =================
def safe_div(num: float, den: float, cap: float = 20.0) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
    return max(min(num / den, cap), -cap)

def check_macd_cross(curr: pd.Series, prev: pd.Series) -> bool:
    return prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime_to_str()
    req_headers = _GLOBAL_HEADERS.copy()
    req_headers["Content-Type"] = "application/json"
    
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": f"【{Config.DINGTALK_KEYWORD}】{title}", "text": f"## 🤖 【{Config.DINGTALK_KEYWORD}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            threading.Thread(target=lambda u, p: requests.post(u, json=p, headers=req_headers, timeout=10) if True else None, args=(url, payload), daemon=False).start()
                
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        html_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = re.sub(r'### (.*?)\n', r'<b>\1</b>\n', html_content)
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_content)
        html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
        tg_text = f"🤖 <b>【量化监控】{html_title}</b>\n\n{html_content}\n\n⏱️ <i>{formatted_time}</i>"
        
        def _send_tg():
            try: requests.post(f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": tg_text[:4000], "parse_mode": "HTML"}, headers=req_headers, timeout=10)
            except Exception: pass
        threading.Thread(target=_send_tg, daemon=False).start()

def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 3, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    cache_key = f"{symbol}_{interval}"
    cache_file = os.path.join(Config.DATA_DIR, f"{cache_key}.pkl")
    
    with _KLINE_LOCK:
        if cache_key in _KLINE_CACHE: return _KLINE_CACHE[cache_key].copy()
            
    df = pd.DataFrame()
    if os.path.exists(cache_file):
        try: df = pd.read_pickle(cache_file)
        except Exception: pass
    
    needs_full = True
    download_period = period
    now_utc = datetime.now(timezone.utc)
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
        elif df.index.tzinfo != timezone.utc: df.index = df.index.tz_convert('UTC')
        if (now_utc - df.index[-1]).days < 30 and len(df) >= 100:
            download_period = "1mo"
            needs_full = False
        
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.1, 0.3) if fast_mode else random.uniform(1.0, 2.0))
            new_df = yf.Ticker(symbol).history(period=download_period, interval=interval, auto_adjust=auto_adjust, timeout=10)
            if not new_df.empty:
                new_df.index = pd.to_datetime(new_df.index, utc=True)
                if not df.empty and not needs_full:
                    df = pd.concat([df[~df.index.isin(new_df.index)], new_df]).sort_index()
                else:
                    df = new_df
                df = df[~df.index.duplicated(keep='last')]
                if len(df) > 800: df = df.iloc[-800:] 
                
                with _KLINE_LOCK: _KLINE_CACHE[cache_key] = df.copy()
                tmp_c = f"{cache_file}.tmp"
                df.to_pickle(tmp_c)
                os.replace(tmp_c, cache_file)
                return df
        except Exception as e:
            if attempt == retries - 1: return df
            time.sleep(2.0)
    return df

def get_filtered_watchlist(max_stocks: int = 150) -> list:
    return list(Config.CORE_WATCHLIST)[:max_stocks]

# ================= 4. IPC 数据库与执行网关 (Gateway) =================

class OrderLedger:
    """基于 SQLite WAL 模式的订单账本 (IPC + 最终一致性)"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute('pragma journal_mode=wal')
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    client_oid TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
                    qty REAL NOT NULL, order_type TEXT NOT NULL, limit_price REAL,
                    arrival_price REAL NOT NULL, status TEXT NOT NULL,
                    filled_qty REAL DEFAULT 0.0, avg_fill_price REAL DEFAULT 0.0,
                    retry_count INTEGER DEFAULT 0, broker_oid TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON orders(status)')

    def fetch_orders_by_status(self, statuses: List[str]) -> pd.DataFrame:
        with self._get_conn() as conn:
            placeholders = ','.join(['?'] * len(statuses))
            df = pd.read_sql_query(f"SELECT * FROM orders WHERE status IN ({placeholders})", conn, params=statuses)
        return df.copy(deep=True)

    def update_order_status(self, client_oid: str, status: str, filled_qty: float = 0.0, avg_fill_price: float = 0.0, broker_oid: str = None):
        with self._get_conn() as conn:
            conn.execute('''UPDATE orders SET status = ?, filled_qty = ?, avg_fill_price = ?, broker_oid = COALESCE(?, broker_oid), updated_at = CURRENT_TIMESTAMP WHERE client_oid = ?''', (status, filled_qty, avg_fill_price, broker_oid, client_oid))
            
    def increment_retry(self, client_oid: str):
        with self._get_conn() as conn:
            conn.execute('UPDATE orders SET retry_count = retry_count + 1 WHERE client_oid = ?', (client_oid,))

    def insert_order(self, symbol: str, side: str, qty: float, arrival_price: float, limit_price: float = None) -> str:
        client_oid = f"QB_{uuid.uuid4().hex[:8]}"
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO orders (client_oid, symbol, side, qty, order_type, limit_price, arrival_price, status)
                VALUES (?, ?, ?, ?, 'MKT', ?, ?, 'PENDING_SUBMIT')
            ''', (client_oid, symbol, side, qty, limit_price, arrival_price))
        return client_oid

@dataclass
class BrokerOrder:
    broker_oid: str; status: str; filled_qty: float; avg_fill_price: float

class BaseBrokerGateway:
    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None) -> BrokerOrder: raise NotImplementedError
    def fetch_order(self, broker_oid: str) -> BrokerOrder: raise NotImplementedError

class MockAlpacaGateway(BaseBrokerGateway):
    def __init__(self): self._mock_exchange = {}
    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None) -> BrokerOrder:
        broker_oid = f"ALPACA_{uuid.uuid4().hex[:12]}"
        self._mock_exchange[broker_oid] = {"status": "OPEN", "filled_qty": 0.0, "avg_fill_price": 0.0, "qty": qty, "price": limit_price or 100.0}
        return BrokerOrder(broker_oid, "OPEN", 0.0, 0.0)
    def fetch_order(self, broker_oid: str) -> BrokerOrder:
        if broker_oid not in self._mock_exchange: raise ValueError("Order not found")
        order = self._mock_exchange[broker_oid]
        if order["status"] == "OPEN":
            order["status"] = "FILLED"
            order["filled_qty"] = order["qty"]
            order["avg_fill_price"] = order["price"] * np.random.uniform(0.999, 1.001)
        return BrokerOrder(broker_oid, order["status"], order["filled_qty"], order["avg_fill_price"])

class ExecutionEngine:
    def __init__(self, broker: BaseBrokerGateway, db_path: str = Config.ORDER_DB_PATH):
        self.broker = broker
        self.ledger = OrderLedger(db_path)
        self.is_running = False

    def _write_tca_log_atomic(self, order_row: pd.Series):
        tca_record = {
            "client_oid": order_row['client_oid'], "symbol": order_row['symbol'], "side": order_row['side'],
            "qty": float(order_row['filled_qty']), "arrival_price": float(order_row['arrival_price']),
            "execution_price": float(order_row['avg_fill_price']), "timestamp": datetime_to_str()
        }
        arr_px = np.maximum(tca_record["arrival_price"], 1e-10)
        slippage_bps = (tca_record["execution_price"] - arr_px) / arr_px * 10000.0
        if tca_record["side"] == "SELL": slippage_bps = -slippage_bps 
        tca_record["slippage_bps"] = float(slippage_bps)

        tmp_path = f"{Config.TCA_LOG_PATH}.{threading.get_ident()}.tmp"
        try:
            history = []
            if os.path.exists(Config.TCA_LOG_PATH):
                with open(Config.TCA_LOG_PATH, 'r') as f: history = [json.loads(line) for line in f]
            history.append(tca_record)
            with open(tmp_path, 'w') as f:
                for h in history: f.write(json.dumps(h) + '\n')
            os.replace(tmp_path, Config.TCA_LOG_PATH)
            logger.info(f"📊 [TCA] {order_row['symbol']} 归因完成。滑点: {slippage_bps:.2f} bps")
        except Exception:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def _recover_state(self):
        df_pending = self.ledger.fetch_orders_by_status(['SUBMITTED', 'OPEN', 'PARTIALLY_FILLED'])
        for _, row in df_pending.iterrows():
            if pd.notna(row['broker_oid']): self._sync_single_order(row)
            else: self.ledger.update_order_status(row['client_oid'], 'REJECTED')

    def _process_queue(self):
        df_new = self.ledger.fetch_orders_by_status(['PENDING_SUBMIT'])
        for _, row in df_new.iterrows():
            oid = row['client_oid']
            if row['qty'] * row['arrival_price'] > Config.Params.FAT_FINGER_MAX_USD:
                self.ledger.update_order_status(oid, 'REJECTED')
                continue
            if row['retry_count'] >= Config.Params.MAX_RETRY_COUNT:
                self.ledger.update_order_status(oid, 'REJECTED')
                continue
            try:
                limit_px = float(row['limit_price']) if pd.notna(row['limit_price']) else None
                b_order = self.broker.submit_order(row['symbol'], row['side'], row['qty'], row['order_type'], limit_px)
                self.ledger.update_order_status(oid, b_order.status, b_order.filled_qty, b_order.avg_fill_price, b_order.broker_oid)
            except Exception:
                self.ledger.increment_retry(oid)

    def _sync_single_order(self, row: pd.Series):
        oid, b_oid = row['client_oid'], row['broker_oid']
        try:
            b_order = self.broker.fetch_order(b_oid)
            if b_order.status != row['status'] or b_order.filled_qty != row['filled_qty']:
                self.ledger.update_order_status(oid, b_order.status, b_order.filled_qty, b_order.avg_fill_price)
                if b_order.status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED'] and b_order.filled_qty > 0:
                    row_copy = row.copy(deep=True)
                    row_copy['filled_qty'] = b_order.filled_qty
                    row_copy['avg_fill_price'] = b_order.avg_fill_price
                    self._write_tca_log_atomic(row_copy)
        except Exception: pass

    def _sync_open_orders(self):
        df_open = self.ledger.fetch_orders_by_status(['SUBMITTED', 'OPEN', 'PARTIALLY_FILLED'])
        for _, row in df_open.iterrows():
            if pd.notna(row['broker_oid']): self._sync_single_order(row)

    def run(self):
        self.is_running = True
        logger.info("🚀 Execution Gateway 守护进程启动，开始监听 IPC...")
        self._recover_state()
        try:
            while self.is_running:
                self._process_queue()
                self._sync_open_orders()
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("🛑 网关安全下线。")
            self.is_running = False

# ================= 5. 核心指标与打分引擎 =================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df['Close'], df['Volume'], df['Open'], df['High'], df['Low'] = df['Close'].ffill(), df['Volume'].ffill(), df['Open'].ffill(), df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_150'] = df['Close'].rolling(150).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean() + 1e-10)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    atr20 = df['TR'].rolling(20).mean()
    df['KC_Upper'], df['KC_Lower'] = df['EMA_20'] + 1.5 * atr20, df['EMA_20'] - 1.5 * atr20
    bb_ma, bb_std = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Upper'], df['BB_Lower'] = bb_ma + 2 * bb_std, bb_ma - 2 * bb_std
    
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    
    hl2, atr10 = (df['High'].values + df['Low'].values) / 2.0, df['TR'].rolling(10).mean().values
    ub, lb, close_arr = hl2 + 3 * atr10, hl2 - 3 * atr10, df['Close'].values
    in_up = np.ones(len(df), dtype=bool)
    for i in range(1, len(df)):
        if np.isnan(ub[i-1]): continue
        c, prev_ub, prev_lb, prev_in_up = close_arr[i], ub[i-1], lb[i-1], in_up[i-1]
        curr_in_up = True if c > prev_ub else (False if c < prev_lb else prev_in_up)
        in_up[i] = curr_in_up
        if curr_in_up and prev_in_up: lb[i] = max(lb[i], prev_lb)
        elif not curr_in_up and not prev_in_up: ub[i] = min(ub[i], prev_ub)
    df['SuperTrend_Up'] = in_up.astype(int)
    
    hl_diff = np.maximum(df['High'].values - df['Low'].values, 1e-10)
    dollar_vol = df['Close'].values * df['Volume'].values
    vol_sum20 = pd.Series(dollar_vol).rolling(20).sum().values + 1e-10
    clv_num = (df['Close'].values - df['Low'].values) - (df['High'].values - df['Close'].values)
    df['CMF'] = pd.Series((clv_num / hl_diff * dollar_vol)).rolling(20).sum().values / vol_sum20
    df['CMF'] = df['CMF'].clip(-1.0, 1.0).fillna(0.0)

    df['Range'] = df['High'] - df['Low']
    df['NR7'] = (df['Range'] <= df['Range'].rolling(7).min())
    df['Inside_Bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))

    close_shift = np.roll(close_arr, 1)
    close_shift[0] = np.nan
    df['VPT'] = np.where((close_shift == 0) | np.isnan(close_shift), 0.0, (close_arr - close_shift) / close_shift) * df['Volume'].values
    df['VPT_Cum'] = df['VPT'].cumsum()
    vpt_std50 = df['VPT_Cum'].rolling(50).std().values
    vpt_std50 = np.where((np.isnan(vpt_std50)) | (vpt_std50 == 0), 1e-6, vpt_std50)
    df['VPT_ZScore'] = (df['VPT_Cum'].values - df['VPT_Cum'].rolling(50).mean().values) / vpt_std50
    df['VPT_Accel'] = np.gradient(np.nan_to_num(df['VPT_ZScore'].values))

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP_20'] = (df['Typical_Price'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)
    
    anchor_groups = (df['Volume'] >= df['Volume'].rolling(120, min_periods=1).max()).cumsum()
    df['AVWAP'] = (df['Typical_Price'] * df['Volume']).groupby(anchor_groups).cumsum() / (df['Volume'].groupby(anchor_groups).cumsum() + 1e-10)
    df['Max_Down_Vol_10'] = df['Volume'].where(df['Close'] < df['Close'].shift(), 0).shift(1).rolling(10).max()
    
    surge = (df['Close'] > df['Close'].shift(1) * 1.04) & (df['Volume'] > df['Volume'].rolling(20).mean())
    df['OB_High'] = df['High'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['OB_Low'] = df['Low'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Range_60'] = df['High'].rolling(60).max() - df['Low'].rolling(60).min()
    df['Range_20'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Price_High_20'] = df['High'].rolling(20).max()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    intra_strength = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
    df['CVD_Smooth'] = (df['Volume'] * intra_strength).cumsum().ewm(span=5).mean()
    cvd_ma10, cvd_ma30 = df['CVD_Smooth'].rolling(10).mean(), df['CVD_Smooth'].rolling(30).mean()
    df['CVD_Trend'] = np.where(cvd_ma10 > cvd_ma30, 1.0, np.where(cvd_ma10 < cvd_ma30, -1.0, 0.0))
    df['CVD_Divergence'] = ((df['Close'] >= df['Price_High_20'] * 0.99) & (df['CVD_Smooth'] < df['CVD_Smooth'].rolling(20).max() * 0.95)).astype(int)
    
    df['Highest_22'] = df['High'].rolling(22).max()
    df['ATR_22'] = df['TR'].rolling(22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['Smart_Money_Flow'] = clv.rolling(10).mean()
    df['Recent_Price_Surge_3d'] = (df['Close'] / df['Open'] - 1).rolling(3).max().shift(1) * 100
    df['Recent_Vol_Surge_3d'] = (df['Volume'] / df['Vol_MA20']).rolling(3).max().shift(1)
    df['Amihud'] = (df['Close'].pct_change().abs() / (df['Close'] * df['Volume'] + 1e-10)).rolling(20).mean() * 1e6
    high_52w = df['High'].rolling(252, min_periods=20).max()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-10)

    return df

def _evaluate_omni_matrix(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData) -> Tuple[int, List[str], List[str], bool]:
    triggered_list, factors_list = [], []
    theme_scores = {'TREND': 0.0, 'VOLATILITY': 0.0, 'REVERSAL': 0.0, 'QUANTUM': 0.0}
    black_swan_risk = False
    
    def add_t(tag, text, pts, theme):
        fw = ctx.xai_weights.get(tag, 1.0)
        if fw > 0:
            adj = pts * ctx.w_mul * fw
            if ctx.regime in ["bear", "hidden_bear"]: adj *= 0.6 if theme in ["TREND", "VOLATILITY"] else 1.4
            elif ctx.regime in ["bull", "rebound"]: adj *= 1.2 if theme in ["TREND", "VOLATILITY", "QUANTUM"] else 1.0
            theme_scores[theme] += adj
            triggered_list.append(text.format(fw=fw))
            factors_list.append(tag)

    if pd.notna(stock.curr['SMA_200']) and stock.curr['Close'] > stock.curr['SMA_50'] > stock.curr['SMA_150'] > stock.curr['SMA_200']:
        add_t("米奈尔维尼", f"🏆 米奈尔维尼模板形成 (权:{{fw:.2f}}x)", 15, "TREND")
    if cf.rs_20 > 1.0 + (stock.curr['ATR'] / (stock.curr['Close'] + 1e-10)) * 2.0:
        add_t("强相对强度", f"⚡ 动能超越波动率阈值 (权:{{fw:.2f}}x)", 7, "TREND")
    
    if check_macd_cross(stock.curr, stock.prev):
        add_t("MACD金叉", f"🔥 MACD起爆 (权:{{fw:.2f}}x)", 12 if stock.curr['MACD'] > 0 else 8, "TREND")

    if stock.curr['Above_Cloud'] == 1 and stock.curr['Tenkan'] > stock.curr['Kijun']:
        add_t("一目多头", "🌥️ 一目均衡表云上多头共振 (权:{fw:.2f}x)", 6, "TREND")
    if pd.notna(stock.curr['AVWAP']) and stock.curr['Close'] > stock.curr['AVWAP'] and stock.prev['Close'] <= stock.curr['AVWAP']:
        add_t("AVWAP突破", "⚓ 强势站上AVWAP锚定成本核心区 (权:{fw:.2f}x)", 12, "TREND")
        
    kc_w, bb_w = stock.curr['KC_Upper'] - stock.curr['KC_Lower'], stock.curr['BB_Upper'] - stock.curr['BB_Lower']
    if bb_w < kc_w: add_t("TTM Squeeze ON", f"📦 TTM Squeeze 挤流状态激活 (权:{{fw:.2f}}x)", 15, "VOLATILITY")
    
    if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Low'] < stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.curr['Swing_Low_20']:
        add_t("流动性扫盘", "🧹 刺穿前低扫掉散户止损后诱空反转 (权:{fw:.2f}x)", 15, "REVERSAL")
    
    lower_wick = stock.curr['Open'] - stock.curr['Low'] if stock.curr['Close'] > stock.curr['Open'] else stock.curr['Close'] - stock.curr['Low']
    tr_val = stock.curr['High'] - stock.curr['Low'] + 1e-10
    if stock.curr['Close'] > stock.curr['Open'] and (lower_wick / tr_val) > 0.3:
        add_t("AMD操盘", "🎭 深度开盘诱空下杀后拉升派发 (权:{fw:.2f}x)", 12, "REVERSAL")

    sum_s = sum(50.0 * (1 - np.exp(-v / 25.0)) for v in theme_scores.values())
    return int(100.0 * (1 - np.exp(-sum_s / 50.0))), triggered_list, factors_list, False

# ================= 6. AI 深度学习 (PyTorch) =================
if TRANSFORMER_AVAILABLE:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 120):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / (d_model + 1e-10)))
            pe[:, 0::2] = torch.sin(position * div_term)
            cos_term = torch.cos(position * div_term)
            pe[:, 1::2] = cos_term[:, :pe[:, 1::2].shape[1]]
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class AlphaContrastiveLoss(nn.Module):
        def __init__(self, temperature: float = 0.1):
            super().__init__()
            self.temperature = temperature
        def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
            pos_sim = F.cosine_similarity(anchor, pos) / self.temperature
            neg_sim = F.cosine_similarity(anchor, neg) / self.temperature
            return F.cross_entropy(torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1), torch.zeros(pos_sim.size(0), dtype=torch.long, device=anchor.device))

    class QuantAlphaTransformer(nn.Module):
        def __init__(self, num_features=49, d_model=64, nhead=8, num_layers=3, dropout=0.2, alpha_dim=16):
            super().__init__()
            self.hp = {'num_features': num_features, 'd_model': d_model, 'nhead': nhead, 'num_layers': num_layers, 'dropout': dropout, 'alpha_dim': alpha_dim}
            self.d_model = d_model
            self.input_norm = nn.LayerNorm(num_features)
            self.feature_proj = nn.Linear(num_features, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=120)
            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True, activation='gelu'), num_layers)
            self.time_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.time_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.alpha_head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.LayerNorm(32), nn.Dropout(dropout), nn.Linear(32, alpha_dim), nn.Tanh())

        def forward(self, src: torch.Tensor) -> torch.Tensor:
            x = self.pos_encoder(self.feature_proj(self.input_norm(src)) * math.sqrt(self.d_model))
            mem = self.transformer(x)
            pooled, _ = self.time_attn(self.time_query.expand(mem.size(0), -1, -1), mem, mem)
            return self.alpha_head(pooled.squeeze(1))

        @torch.no_grad()
        def extract_alpha(self, features: np.ndarray) -> np.ndarray:
            self.eval()
            tensor = torch.from_numpy(features.copy()).float().to(next(self.parameters()).device)
            if tensor.dim() == 2: tensor = tensor.unsqueeze(0)
            return self.forward(tensor).cpu().numpy()

# ================= 7. 全局调度 CLI 路由 =================
def run_matrix():
    logger.info("启动 Matrix 打分管线...")

def run_gateway():
    broker = MockAlpacaGateway()
    ExecutionEngine(broker).run()

# ================= 8. 内置基建测谎仪 (TDD) =================
class QuantEngineTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        periods = 150
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        close = np.linspace(100, 250, periods) + np.random.normal(0, 1.5, periods)
        self.mock_df = pd.DataFrame({'Open': close-1, 'High': close+2, 'Low': close-2, 'Close': close, 'Volume': 1000000}, index=dates)

        if os.path.exists(Config.ORDER_DB_PATH): os.remove(Config.ORDER_DB_PATH)
        self.ledger = OrderLedger(Config.ORDER_DB_PATH)
        self.engine = ExecutionEngine(MockAlpacaGateway(), Config.ORDER_DB_PATH)

    def test_calculate_indicators(self):
        df_res = calculate_indicators(self.mock_df)
        self.assertEqual(len(df_res), len(self.mock_df))
        self.assertIn('MACD', df_res.columns)
        self.assertIn('AVWAP', df_res.columns)

    def test_macd_cross_logic(self):
        close = np.linspace(100, 70, 59).tolist()
        close.append(110.0) 
        df = pd.DataFrame({'Close': close, 'Open': close, 'High': [c*1.05 for c in close], 'Low': [c*0.95 for c in close], 'Volume': 1000000})
        res = calculate_indicators(df)
        self.assertTrue(res.iloc[-2]['MACD'] < res.iloc[-2]['Signal_Line'])
        self.assertTrue(res.iloc[-1]['MACD'] > res.iloc[-1]['Signal_Line'])

    def test_gateway_fat_finger(self):
        oid = self.ledger.insert_order("AAPL", "BUY", 10000.0, 100.0)
        self.engine._process_queue()
        df = self.ledger.fetch_orders_by_status(['REJECTED'])
        self.assertEqual(len(df), 1)

    def test_gateway_fsm_transition(self):
        oid = self.ledger.insert_order("NVDA", "BUY", 10.0, 100.0)
        self.engine._process_queue()
        self.assertEqual(len(self.ledger.fetch_orders_by_status(['OPEN'])), 1)
        self.engine._sync_open_orders()
        self.assertEqual(len(self.ledger.fetch_orders_by_status(['FILLED'])), 1)

def main():
    initialize_directories()
    parser = argparse.ArgumentParser(description="Quant Engine (Singularity Edition)")
    parser.add_argument('mode', choices=['matrix', 'backtest', 'train', 'gateway', 'test'], help="运行模式")
    args = parser.parse_args()

    if args.mode == 'test':
        sys.argv = [sys.argv[0]]
        unittest.main(verbosity=2)
    elif args.mode == 'gateway': run_gateway()
    elif args.mode == 'matrix': run_matrix()
    else: logger.info(f"Mode {args.mode} is under construction.")

if __name__ == "__main__":
    main()
