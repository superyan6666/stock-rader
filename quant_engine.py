# 存储路径: quant_engine.py
# 🤖 Quant Engine 3.0 (Singularity Edition) - 全栈单体引擎 (典藏金版)
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
import math        
import random 
import gc          
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
    from quant_transformer import QuantAlphaTransformer 
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
    
    # 🔗 券商 API 配置 (Alpaca)
    ALPACA_API_KEY: str = os.environ.get('ALPACA_API_KEY', '')
    ALPACA_API_SECRET: str = os.environ.get('ALPACA_API_SECRET', '')
    ALPACA_BASE_URL: str = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'TXN'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD'],
        'XLC': ['GOOGL', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'IDXX'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'WBA']
    }
    
    DATA_DIR: str = ".quantbot_data"
    LOG_PREFIX: str = "backtest_log_"
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
    CUSTOM_CONFIG_FILE: str = "quantbot_config.json"
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

    @classmethod
    def get_current_log_file(cls) -> str:
        return os.path.join(Config.DATA_DIR, f"{cls.LOG_PREFIX}{datetime.now(timezone.utc).strftime('%Y_%m')}.jsonl")

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

# 🚀 优化：模块级全局推送线程池，告别裸线程泄漏
_ALERT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="alert_push")

def _init_ext_cache():
    global _DAILY_EXT_CACHE
    with _DAILY_EXT_LOCK:
        if not _DAILY_EXT_CACHE:
            if os.path.exists(Config.EXT_CACHE_FILE):
                try:
                    with open(Config.EXT_CACHE_FILE, 'r', encoding='utf-8') as f: _DAILY_EXT_CACHE = json.load(f)
                except Exception: pass
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if _DAILY_EXT_CACHE.get("date") != today:
                _DAILY_EXT_CACHE = {"date": today, "sentiment": {}, "alt": {}}

def _save_ext_cache():
    with _DAILY_EXT_LOCK:
        try:
            tmp = f"{Config.EXT_CACHE_FILE}.{threading.get_ident()}.tmp"
            with open(tmp, 'w', encoding='utf-8') as f: json.dump(_DAILY_EXT_CACHE, f)
            os.replace(tmp, Config.EXT_CACHE_FILE)
        except Exception: pass

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

# ================= 3. 底层网络与数据获取逻辑 =================
def safe_div(num: float, den: float, cap: float = 20.0) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
    return max(min(num / den, cap), -cap)

def check_macd_cross(curr: pd.Series, prev: pd.Series) -> bool:
    return prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']

def _push_webhook(url: str, payload: dict, headers: dict) -> None:
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200: logger.warning(f"Webhook 推送异常: {resp.status_code}")
    except Exception as e: logger.error(f"Webhook 网络错误: {e}")

def _push_telegram(token: str, chat_id: str, text: str, headers: dict) -> None:
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"}, headers=headers, timeout=10)
        if resp.status_code != 200: logger.warning(f"Telegram 推送异常: {resp.text[:100]}")
    except Exception as e: logger.error(f"Telegram 网络错误: {e}")

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime_to_str()
    req_headers = _GLOBAL_HEADERS.copy()
    req_headers["Content-Type"] = "application/json"
    
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": f"【{Config.DINGTALK_KEYWORD}】{title}", "text": f"## 🤖 【{Config.DINGTALK_KEYWORD}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            _ALERT_EXECUTOR.submit(_push_webhook, url, payload, req_headers)
                
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        html_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = re.sub(r'### (.*?)\n', r'<b>\1</b>\n', html_content)
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_content)
        html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
        tg_text = f"🤖 <b>【量化监控】{html_title}</b>\n\n{html_content}\n\n⏱️ <i>{formatted_time}</i>"
        _ALERT_EXECUTOR.submit(_push_telegram, Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID, tg_text, req_headers)

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
                else: df = new_df
                df = df[~df.index.duplicated(keep='last')]
                if len(df) > 800: df = df.iloc[-800:] 
                
                with _KLINE_LOCK: _KLINE_CACHE[cache_key] = df.copy()
                tmp_c = f"{cache_file}.tmp"
                df.to_pickle(tmp_c)
                os.replace(tmp_c, cache_file)
                return df
        except Exception:
            if attempt == retries - 1: return df
            time.sleep(2.0)
    return df

def fetch_tradingview_screener(max_tickers=150) -> list:
    try:
        url = "https://scanner.tradingview.com/america/scan"
        payload = {
            "filter": [
                {"left": "type", "operation": "in_range", "right": ["stock"]},
                {"left": "subtype", "operation": "in_range", "right": ["common"]},
                {"left": "exchange", "operation": "in_range", "right": ["AMEX", "NASDAQ", "NYSE"]},
                {"left": "close", "operation": "greater", "right": 15}, 
                {"left": "average_volume_10d_calc", "operation": "greater", "right": 2500000}, 
                {"left": "beta_1_year", "operation": "greater", "right": 1.1}
            ],
            "options": {"lang": "en"}, "markets": ["america"],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": ["name", "close", "volume", "RelativeVolume10CG"],
            "sort": {"sortBy": "RelativeVolume10CG", "sortOrder": "desc"},
            "range": [0, max_tickers]
        }
        headers = _GLOBAL_HEADERS.copy()
        headers["Content-Type"] = "application/json"
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            tickers = [item['d'][0].split(':')[1] if ':' in item['d'][0] else item['d'][0] for item in data.get('data', [])]
            return [t.replace('.', '-') for t in tickers if len(t) <= 5]
    except Exception: pass
    return []

def get_filtered_watchlist(max_stocks: int = 150) -> list:
    tickers = set(Config.CORE_WATCHLIST)
    tv_tickers = fetch_tradingview_screener(120)
    if tv_tickers: tickers.update(tv_tickers)
    
    if os.path.exists(Config.CACHE_FILE):
        try:
            mtime = os.path.getmtime(Config.CACHE_FILE)
            if time.time() - mtime < 7 * 86400:
                with open(Config.CACHE_FILE, "r", encoding="utf-8") as f: tickers.update(json.load(f))
        except Exception: pass
    tickers_list = list(tickers)

    try:
        chunk_size = 50 
        dfs = []
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            for attempt in range(3):
                try:
                    chunk_df = yf.download(chunk, period="5d", progress=False, threads=False, timeout=10)
                    if not chunk_df.empty: 
                        dfs.append(chunk_df)
                        break
                    else: time.sleep(2.0)
                except Exception: time.sleep(3.0)
            if i + chunk_size < len(tickers_list): time.sleep(2.0)
                
        if not dfs: return list(Config.CORE_WATCHLIST)[:max_stocks]
        df = pd.concat(dfs, axis=1)
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close'] if 'Close' in df.columns else df.xs('Close', level=0, axis=1)
            volume_df = df['Volume'] if 'Volume' in df.columns else df.xs('Volume', level=0, axis=1)
        else:
            close_df, volume_df = df, pd.DataFrame(1e6, index=df.index, columns=df.columns)

        available_tickers = set(close_df.columns) if isinstance(close_df, pd.DataFrame) else set()
        if len(available_tickers) > 200:
            try:
                tmp = f"{Config.CACHE_FILE}.tmp"
                with open(tmp, "w", encoding="utf-8") as f: json.dump(list(available_tickers), f)
                os.replace(tmp, Config.CACHE_FILE)
            except Exception: pass

        closes = close_df.dropna(axis=1, how='all').ffill().iloc[-1]
        volumes = volume_df.dropna(axis=1, how='all').mean()
        turnovers = (closes * volumes).dropna()
        valid_turnovers = turnovers[(closes > 10.0) & (turnovers > 30_000_000)]
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        return top_tickers if top_tickers else list(Config.CORE_WATCHLIST)[:max_stocks]
    except Exception:
        return list(Config.CORE_WATCHLIST)[:max_stocks]

def fetch_global_wsb_data() -> Dict[str, float]:
    with _ALT_DATA_LOCK:
        if "WSB_ACCEL_GLOBAL" in _ALT_DATA_CACHE: return _ALT_DATA_CACHE["WSB_ACCEL_GLOBAL"]
    history = {}
    if os.path.exists(Config.WSB_CACHE_FILE):
        try:
            with open(Config.WSB_CACHE_FILE, "r") as f: history = json.load(f)
        except Exception: pass
    current_data = {}
    try:
        response = requests.get("https://tradestie.com/api/v1/apps/reddit", headers=_GLOBAL_HEADERS, timeout=10)
        if response.status_code == 200:
            for item in response.json():
                tk = item.get('ticker')
                if tk and item.get('sentiment') == 'Bullish': current_data[tk] = item.get('no_of_comments', 0)
    except Exception: pass
    if current_data: history[datetime.now(timezone.utc).strftime('%Y-%m-%d')] = current_data
    sorted_dates = sorted(history.keys())
    if len(sorted_dates) > 5:
        for d in sorted_dates[:-5]: del history[d]
    try:
        tmp = f"{Config.WSB_CACHE_FILE}.tmp"
        with open(tmp, "w") as f: json.dump(history, f)
        os.replace(tmp, Config.WSB_CACHE_FILE)
    except Exception: pass
    
    wsb_accel_dict = {}
    dates = sorted(history.keys())
    if len(dates) >= 3:
        d0, d1, d2 = dates[-1], dates[-2], dates[-3]
        for tk in history[d0].keys():
            vel1 = history[d0].get(tk, 0) - history[d1].get(tk, 0)
            vel2 = history[d1].get(tk, 0) - history[d2].get(tk, 0)
            wsb_accel_dict[tk] = float(vel1 - vel2)
    else:
        for tk in current_data.keys(): wsb_accel_dict[tk] = 0.0
    with _ALT_DATA_LOCK: _ALT_DATA_CACHE["WSB_ACCEL_GLOBAL"] = wsb_accel_dict
    return wsb_accel_dict

def safe_get_sentiment_data(symbol: str) -> Tuple[float, float, float, float]:
    _init_ext_cache()
    with _DAILY_EXT_LOCK:
        if symbol in _DAILY_EXT_CACHE["sentiment"]: return tuple(_DAILY_EXT_CACHE["sentiment"][symbol])
    pcr, iv_skew, short_change, short_float = 0.0, 0.0, 0.0, 0.0
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        curr_short, prev_short = info.get('sharesShort', 0), info.get('sharesShortPriorMonth', 0)
        short_float = info.get('shortPercentOfFloat', 0)
        if curr_short and prev_short and prev_short > 0: short_change = (curr_short - prev_short) / prev_short
        exps = tk.options
        if exps:
            opt = tk.option_chain(exps[0]) 
            c_vol = opt.calls['volume'].sum() if 'volume' in opt.calls else 0
            p_vol = opt.puts['volume'].sum() if 'volume' in opt.puts else 0
            if c_vol > 0: pcr = p_vol / c_vol
            c_iv = opt.calls['impliedVolatility'].median() if 'impliedVolatility' in opt.calls else 0
            p_iv = opt.puts['impliedVolatility'].median() if 'impliedVolatility' in opt.puts else 0
            iv_skew = p_iv - c_iv
    except Exception: pass
    with _DAILY_EXT_LOCK: _DAILY_EXT_CACHE["sentiment"][symbol] = (pcr, iv_skew, short_change, short_float)
    _save_ext_cache()
    return pcr, iv_skew, short_change, short_float

def safe_get_alt_data(symbol: str) -> Tuple[float, float, float, str]:
    _init_ext_cache()
    with _DAILY_EXT_LOCK:
        if symbol in _DAILY_EXT_CACHE["alt"]: return tuple(_DAILY_EXT_CACHE["alt"][symbol])
    inb, am, nlp, news = 0.0, 0.0, 0.0, ""
    try:
        tk = yf.Ticker(symbol)
        try:
            insiders = tk.insider_transactions
            if insiders is not None and not insiders.empty:
                recent = insiders.head(20)
                if 'Shares' in recent.columns and 'Text' in recent.columns:
                    b = recent[recent['Text'].str.contains('Buy|Purchase', case=False, na=False)]['Shares'].sum()
                    s = recent[recent['Text'].str.contains('Sell|Sale', case=False, na=False)]['Shares'].abs().sum()
                    if b + s > 0: inb = (b - s) / (b + s)
        except Exception: pass
        try:
            upgrades = tk.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                upgrades.index = pd.to_datetime(upgrades.index, utc=True)
                recent_1y = upgrades[upgrades.index >= pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=365)].copy()
                if not recent_1y.empty and 'Action' in recent_1y.columns:
                    action_map = {'up': 1, 'down': -1, 'main': 0, 'init': 0.5, 'reit': 0}
                    recent_1y['score'] = recent_1y['Action'].map(action_map).fillna(0)
                    monthly_scores = recent_1y['score'].resample('30D').sum()
                    if len(monthly_scores) >= 3:
                        am = (monthly_scores.iloc[-1] - monthly_scores.iloc[:-1].mean()) / (monthly_scores.iloc[:-1].std() + 1e-5)
        except Exception: pass
        try:
            news_data = tk.news
            if news_data:
                compound_score = 0.0
                latest = news_data[0]
                title = latest.get('title', '')
                for n in news_data[:5]:
                    t = n.get('title', '').lower()
                    if 'lawsuit' in t or 'investigation' in t or 'plunge' in t: compound_score -= 1.0
                    if 'beat' in t or 'surge' in t or 'upgrade' in t: compound_score += 1.0
                nlp = compound_score / math.sqrt(compound_score**2 + 15.0)
                if title: news = f"{'🟢' if nlp > 0.3 else ('🔴' if nlp < -0.3 else '⚪')} {title}"
        except Exception: pass
    except Exception: pass
    with _DAILY_EXT_LOCK: _DAILY_EXT_CACHE["alt"][symbol] = (inb, am, nlp, news)
    _save_ext_cache()
    return inb, am, nlp, news

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                t3 = stats_data.get("overall", {}).get("T+3") if "overall" in stats_data else stats_data.get("T+3")
                if t3 and t3.get('total_trades', 0) > 0:
                    pf_str = f" | 盈亏比 {t3.get('profit_factor', 0.0):.2f}" if 'profit_factor' in t3 else ""
                    ai_wr_str = f" | ⚡LGBM分层元学习胜率 {t3['ai_win_rate']:.1%}" if 'ai_win_rate' in t3 else ""
                    return f"**📈 策略基底验证 (T+3):** 原始胜率 {t3['win_rate']:.1%}{ai_wr_str}{pf_str}"
    except Exception: pass
    return ""

def check_earnings_risk(symbol: str) -> bool:
    try:
        tk = yf.Ticker(symbol)
        now_date = datetime.now(timezone.utc).date()
        try:
            cal = tk.calendar
            if cal is not None and 'Earnings Date' in cal:
                ed = cal['Earnings Date'][0] if isinstance(cal['Earnings Date'], list) else cal.loc['Earnings Date'].iloc[0]
                if hasattr(ed, 'date') and 0 <= (ed.date() - now_date).days <= 5: return True
        except Exception: pass
    except Exception: pass
    return False

def get_alert_cache() -> dict:
    cache_data = {"matrix": {}, "shadow_pool": {}}
    try:
        if os.path.exists(Config.ALERT_CACHE_FILE):
            with open(Config.ALERT_CACHE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data.get("matrix"), dict): cache_data = data
    except Exception: pass
    return cache_data

def set_alerted(sym: str, is_shadow: bool = False, shadow_data: dict = None) -> None:
    cache = get_alert_cache()
    now_ts = time.time()
    if not is_shadow: cache.setdefault("matrix", {})[sym] = now_ts
    else:
        shadow_pool = cache.setdefault("shadow_pool", {})
        if shadow_data:
            shadow_data['_ts'] = now_ts
            shadow_pool[sym] = shadow_data
        else: shadow_pool[sym] = {"_ts": now_ts}
    try:
        tmp = f"{Config.ALERT_CACHE_FILE}.tmp"
        with open(tmp, 'w') as f: json.dump(cache, f)
        os.replace(tmp, Config.ALERT_CACHE_FILE)
    except Exception: pass

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

class AlpacaGateway(BaseBrokerGateway):
    """🚀 真实的 Alpaca REST API 交易网关"""
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret, "Content-Type": "application/json"}

    def _map_status(self, alpaca_status: str) -> str:
        mapping = {"new": "OPEN", "accepted": "OPEN", "partially_filled": "PARTIALLY_FILLED", "filled": "FILLED", "done_for_day": "FILLED", "canceled": "CANCELED", "expired": "EXPIRED", "replaced": "OPEN", "pending_cancel": "OPEN", "pending_replace": "OPEN", "rejected": "REJECTED", "suspended": "REJECTED"}
        return mapping.get(alpaca_status, "OPEN")

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None) -> BrokerOrder:
        url = f"{self.base_url}/v2/orders"
        payload = {"symbol": symbol, "qty": str(qty), "side": side.lower(), "type": "market" if order_type == "MKT" else "limit", "time_in_force": "day"}
        if limit_price and payload["type"] == "limit": payload["limit_price"] = str(round(limit_price, 2))
        resp = requests.post(url, json=payload, headers=self.headers, timeout=10)
        if resp.status_code in [200, 201]:
            data = resp.json()
            return BrokerOrder(broker_oid=data["id"], status=self._map_status(data["status"]), filled_qty=float(data["filled_qty"]), avg_fill_price=float(data["filled_avg_price"] or 0.0))
        raise RuntimeError(f"Alpaca API Error: {resp.text}")

    def fetch_order(self, broker_oid: str) -> BrokerOrder:
        url = f"{self.base_url}/v2/orders/{broker_oid}"
        resp = requests.get(url, headers=self.headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return BrokerOrder(broker_oid=data["id"], status=self._map_status(data["status"]), filled_qty=float(data["filled_qty"]), avg_fill_price=float(data["filled_avg_price"] or 0.0))
        raise RuntimeError(f"Alpaca Fetch Error: {resp.text}")

# 🚀 优化：TCA 日志文件互斥锁
_TCA_LOCK = threading.Lock()

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
        
        arr_px = max(float(tca_record["arrival_price"]), 1e-10)
        is_valid_arrival = float(tca_record["arrival_price"]) > 0.01
        
        slippage_bps = (tca_record["execution_price"] - arr_px) / arr_px * 10000.0
        if tca_record["side"] == "SELL": slippage_bps = -slippage_bps
        
        tca_record["slippage_bps"] = float(slippage_bps) if is_valid_arrival else None
        tca_record["valid"] = is_valid_arrival
        
        # 🚀 优化：直接加锁追加写，消除 O(n) 读写瓶颈
        try:
            with _TCA_LOCK:
                with open(Config.TCA_LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(tca_record) + '\n')
            
            log_slip = f"{slippage_bps:.2f}" if is_valid_arrival else "N/A (异常到达价)"
            logger.info(f"📊 [TCA] {order_row['symbol']} 归因完成。滑点: {log_slip} bps")
        except Exception as e:
            logger.error(f"TCA 写入失败: {e}")

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

# ================= 5. 高级数学与特征抽象计算 =================
def _robust_fft_ensemble(close_prices: np.ndarray, base_length=120, ensemble_count=7) -> float:
    if len(close_prices) < base_length + (ensemble_count // 2) * 5: return 0.0
    votes = []
    for offset in range(ensemble_count):
        win_len = base_length + (offset - ensemble_count//2) * 5
        segment = close_prices[-win_len:]
        if np.isnan(segment).any() or np.all(segment == segment[0]): continue
        detrended = segment - np.mean(segment)
        fft_res = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(win_len)
        pos_mask = (freqs > 0.01) & (freqs < 0.2)
        if not np.any(pos_mask): continue
        peak_idx = np.where(pos_mask)[0][np.argmax(np.abs(fft_res[pos_mask]))]
        phase = np.angle(fft_res[peak_idx])
        current_phase = (2 * np.pi * freqs[peak_idx] * (win_len - 1) + phase) % (2 * np.pi)
        votes.append(1.0 if 0 < current_phase < np.pi else -1.0)
    return float(sum(votes) / len(votes)) if votes else 0.0

def _robust_hurst(close_prices: np.ndarray, min_window=30, n_bootstrap=25) -> Tuple[float, float, bool]:
    # 🚀 优化：使用线程安全的独立随机生成器，并降低重采样次数
    safe_prices = np.maximum(close_prices, 1e-10)
    log_ret = np.diff(np.log(safe_prices[-121:])) if len(safe_prices) > 120 else np.diff(np.log(safe_prices))
    n = len(log_ret)
    if n <= min_window: return 0.5, 0.0, False
    
    rng = np.random.default_rng()
    sub_lens = rng.integers(min_window, n, size=n_bootstrap)
    start_offs = rng.integers(0, n - min_window, size=n_bootstrap)
    
    hurst_samples = []
    for sub_len, start in zip(sub_lens, start_offs):
        sub = log_ret[min(start, n - sub_len) : min(start, n - sub_len) + sub_len]
        S = np.std(sub)
        if S < 1e-10: continue
        cum_dev = np.cumsum(sub - sub.mean())
        R = cum_dev.max() - cum_dev.min()
        if R > 0: hurst_samples.append(np.log(R / S) / np.log(sub_len))
        
    if len(hurst_samples) < 10: return 0.5, 0.0, False
    h_median = float(np.median(hurst_samples))
    h_iqr = float(np.percentile(hurst_samples, 75) - np.percentile(hurst_samples, 25))
    return h_median, h_iqr, bool((h_iqr < 0.15) and (abs(h_median - 0.5) > 0.1))

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df['Close'], df['Volume'], df['Open'], df['High'], df['Low'] = df['Close'].ffill(), df['Volume'].ffill(), df['Open'].ffill(), df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_150'] = df['Close'].rolling(150).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    rs = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean() / (-delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean() + 1e-10)
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
    clv_num = (df['Close'].values - df['Low'].values) - (df['High'].values - df['Close'].values)
    df['CMF'] = (pd.Series((clv_num / hl_diff * dollar_vol)).rolling(20).sum() / (pd.Series(dollar_vol).rolling(20).sum() + 1e-10)).clip(-1.0, 1.0).fillna(0.0)

    df['Range'] = df['High'] - df['Low']
    df['NR7'] = (df['Range'] <= df['Range'].rolling(7).min())
    df['Inside_Bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))

    close_shift = np.roll(close_arr, 1)
    close_shift[0] = np.nan
    df['VPT_Cum'] = (np.where((close_shift == 0) | np.isnan(close_shift), 0.0, (close_arr - close_shift) / close_shift) * df['Volume'].values).cumsum()
    vpt_std50 = np.where((np.isnan(df['VPT_Cum'].rolling(50).std().values)) | (df['VPT_Cum'].rolling(50).std().values == 0), 1e-6, df['VPT_Cum'].rolling(50).std().values)
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

    # 🚀 典藏金版防线：彻底斩断 Inf 毒药，保证机器学习矩阵的纯净度
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df

def _get_transformer_seq(df_ind: pd.DataFrame, end_idx: int = -1) -> np.ndarray:
    if end_idx == -1: end_idx = len(df_ind)
    start_idx = end_idx - 60
    if start_idx < 0: return np.zeros((60, 49), dtype=np.float32)
    df = df_ind.iloc[start_idx:end_idx].copy()
    c = df['Close'].values + 1e-10
    seq = np.zeros((60, 49), dtype=np.float32)
    features = [
        df['RSI'].values / 100.0, df['MACD'].values / c, df['Signal_Line'].values / c, df['ATR'].values / c,
        (df['KC_Upper'].values - df['KC_Lower'].values) / c, (df['BB_Upper'].values - df['BB_Lower'].values) / c,
        df['Tenkan'].values / c - 1.0, df['Kijun'].values / c - 1.0, df['SenkouA'].values / c - 1.0, df['SenkouB'].values / c - 1.0,
        df['SuperTrend_Up'].values, df['CMF'].values, df['Range'].values / c, df['NR7'].astype(float).values,
        df['Inside_Bar'].astype(float).values, df['VPT_ZScore'].values, df['VPT_Accel'].values,
        df['VWAP_20'].values / c - 1.0, np.where(df['AVWAP'].isna(), 0, df['AVWAP'].values / c - 1.0),
        df['Volume'].values / (df['Max_Down_Vol_10'].values + 1e-10),
        np.where(df['OB_High'].isna(), 0, df['OB_High'].values / c - 1.0), np.where(df['OB_Low'].isna(), 0, df['OB_Low'].values / c - 1.0),
        np.where(df['Swing_Low_20'].isna(), 0, df['Swing_Low_20'].values / c - 1.0),
        df['Range_60'].values / c, df['Range_20'].values / c, df['Price_High_20'].values / c - 1.0,
        df['Volume'].values / (df['Vol_MA20'].values + 1e-10), df['CVD_Trend'].values, df['CVD_Divergence'].values,
        df['Highest_22'].values / c - 1.0, df['ATR_22'].values / c, df['Chandelier_Exit'].values / c - 1.0,
        df['Smart_Money_Flow'].values, df['Recent_Price_Surge_3d'].values / 100.0, df['Recent_Vol_Surge_3d'].values,
        df['Amihud'].values, np.where(df['Dist_52W_High'].isna(), 0, df['Dist_52W_High'].values),
        df['Close'].values / (df['SMA_50'].values + 1e-10) - 1.0, df['Close'].values / (df['SMA_150'].values + 1e-10) - 1.0,
        df['Close'].values / (df['SMA_200'].values + 1e-10) - 1.0, df['Close'].values / (df['EMA_20'].values + 1e-10) - 1.0,
        df['Close'].values / (df['EMA_50'].values + 1e-10) - 1.0, df['Above_Cloud'].values,
        df['Open'].values / c - 1.0, df['High'].values / c - 1.0, df['Low'].values / c - 1.0,
        df['Close'].pct_change().fillna(0).values, df['Volume'].pct_change().fillna(0).values,
        (df['Close'].values - df['Open'].values) / (df['High'].values - df['Low'].values + 1e-10)
    ]
    for idx, vals in enumerate(features): seq[:, idx] = vals
    return np.nan_to_num(seq, nan=0.0, posinf=5.0, neginf=-5.0)

def _extract_complex_features(stock: StockData, ctx: MarketContext) -> ComplexFeatures:
    """🚀 满血复原：完整的机构级跨时空与微观结构特征萃取"""
    target_dt = pd.to_datetime(stock.curr.name).tz_localize('UTC') if stock.curr.name.tzinfo is None else pd.to_datetime(stock.curr.name).tz_convert('UTC') if hasattr(stock.curr.name, 'tzinfo') else None
    
    aligned_w = stock.df_w.copy()
    if target_dt is not None and not aligned_w.empty:
        aligned_w.index = aligned_w.index.tz_localize('UTC') if aligned_w.index.tzinfo is None else aligned_w.index.tz_convert('UTC')
        aligned_w = aligned_w[aligned_w.index <= target_dt + pd.Timedelta(days=7)]
        
    weekly_bullish, weekly_macd_res = False, 0.0
    if len(aligned_w) >= 40:
        if aligned_w['Close'].iloc[-1] > aligned_w['Close'].rolling(40).mean().iloc[-1]:
            ema10, ema30 = aligned_w['Close'].ewm(span=10).mean(), aligned_w['Close'].ewm(span=30).mean()
            weekly_bullish = (aligned_w['Close'].iloc[-1] > ema10.iloc[-1]) and (ema10.iloc[-1] > ema30.iloc[-1])
        hist_w = (aligned_w['Close'].ewm(span=12).mean() - aligned_w['Close'].ewm(span=26).mean()) - (aligned_w['Close'].ewm(span=12).mean() - aligned_w['Close'].ewm(span=26).mean()).ewm(span=9).mean()
        if len(hist_w) >= 2 and hist_w.iloc[-1] > 0 and hist_w.iloc[-1] > hist_w.iloc[-2]: weekly_macd_res = 1.0

    fvg_lower, fvg_upper = 0.0, 0.0
    if len(stock.df) >= 22:
        lows, highs = stock.df['Low'].values, stock.df['High'].values
        valid_idx = np.where(lows[-20:-1] > highs[-22:-3])[0]
        if len(valid_idx) > 0: fvg_lower, fvg_upper = highs[valid_idx[-1]-2], lows[valid_idx[-1]]

    kde_breakout_score = 0.0
    if KDE_AVAILABLE and len(stock.df.iloc[-60:]) >= 20 and np.std(stock.df.iloc[-60:]['Close'].values) > 1e-5:
        try:
            prices = stock.df.iloc[-60:]['Close'].values
            kde = gaussian_kde(prices, weights=stock.df.iloc[-60:]['Volume'].values, bw_method='silverman')
            density_current = kde.evaluate(prices[-1])[0]
            densities = kde.evaluate(np.linspace(prices.min(), prices.max(), 200))
            if density_current <= np.percentile(densities, 50):
                kde_breakout_score = min(1.0, 0.5 + 0.5 * (np.percentile(densities, 50) - density_current) / (np.percentile(densities, 50) + 1e-10))
        except Exception: pass

    fft_ensemble_score = _robust_fft_ensemble(stock.df['Close'].values)
    hurst_med, hurst_iqr, hurst_reliable = _robust_hurst(stock.df['Close'].values)
    
    monthly_inst_flow, rsi_60m_bounce = 0.0, 0.0
    aligned_m = stock.df_m.copy()
    if target_dt is not None and not aligned_m.empty:
        aligned_m.index = aligned_m.index.tz_localize('UTC') if aligned_m.index.tzinfo is None else aligned_m.index.tz_convert('UTC')
        aligned_m = aligned_m[aligned_m.index <= target_dt + pd.Timedelta(days=31)] 
    if not aligned_m.empty and len(aligned_m) >= 3:
        m_flow = (aligned_m['Close'] - aligned_m['Open']) / (aligned_m['High'] - aligned_m['Low'] + 1e-10) * aligned_m['Volume']
        if m_flow.iloc[-1] > 0 and m_flow.iloc[-2] > 0 and m_flow.iloc[-3] > 0:
            monthly_inst_flow = 1.0

    aligned_60m = stock.df_60m.copy()
    if target_dt is not None and not aligned_60m.empty:
        aligned_60m.index = aligned_60m.index.tz_localize('UTC') if aligned_60m.index.tzinfo is None else aligned_60m.index.tz_convert('UTC')
        end_of_day = target_dt.replace(hour=23, minute=59, second=59)
        aligned_60m = aligned_60m[aligned_60m.index <= end_of_day]
    if not aligned_60m.empty and len(aligned_60m) >= 15:
        delta = aligned_60m['Close'].diff()
        up = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
        down = -delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean()
        rs = up / (down + 1e-10)
        rsi_60m = 100 - (100 / (1 + rs))
        if len(rsi_60m) >= 2 and rsi_60m.iloc[-1] > rsi_60m.iloc[-2] and rsi_60m.iloc[-2] < 40:
            rsi_60m_bounce = 1.0

    beta_60d, tlt_corr, dxy_corr, vrp, pure_alpha = 1.0, 0.0, 0.0, float((ctx.vix_current - 15.0) / max(ctx.vix_current, 1.0)), 0.0
    rs_20 = 0.0
    if not ctx.qqq_df.empty:
        m_df = pd.merge(stock.df[['Close']], ctx.qqq_df[['Close']], left_index=True, right_index=True, how='inner')
        if len(m_df) >= 60:
            rs_20 = (m_df['Close_x'].iloc[-1] / m_df['Close_x'].iloc[-20]) / max(m_df['Close_y'].iloc[-1] / m_df['Close_y'].iloc[-20], 0.5)
            cov_matrix = np.cov(m_df['Close_x'].pct_change().dropna().iloc[-60:], m_df['Close_y'].pct_change().dropna().iloc[-60:])
            beta_local = cov_matrix[0,1] / (cov_matrix[1,1] + 1e-10) if cov_matrix[1,1] > 0 else 1.0
            pure_alpha = (m_df['Close_x'].pct_change().dropna().iloc[-5:].mean() - beta_local * m_df['Close_y'].pct_change().dropna().iloc[-5:].mean()) * 252

    return ComplexFeatures(weekly_bullish, fvg_lower, fvg_upper, kde_breakout_score, fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable, monthly_inst_flow, weekly_macd_res, rsi_60m_bounce, beta_60d, tlt_corr, dxy_corr, vrp, rs_20, pure_alpha)

def _extract_ml_features(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData, alpha_vec: np.ndarray = None) -> dict:
    macd_cross_strength = safe_div(stock.curr['MACD'] - stock.curr['Signal_Line'], abs(stock.curr['Close']) * 0.01)
    vol_surge_ratio = safe_div(stock.curr['Volume'], stock.curr['Vol_MA20'], cap=50.0)
    hurst_score = max(0.0, min(1.0, (cf.hurst_med - 0.5) * 2.0)) if (cf.hurst_reliable and cf.hurst_med > Config.Params.HURST_RELIABLE) else 0.0

    feat_dict = {
        "米奈尔维尼": safe_div(stock.curr['SMA_50'] - stock.curr['SMA_200'], stock.curr['SMA_200']),
        "强相对强度": cf.rs_20, "MACD金叉": macd_cross_strength,
        "TTM Squeeze ON": safe_div((stock.curr['KC_Upper'] - stock.curr['KC_Lower']) - (stock.curr['BB_Upper'] - stock.curr['BB_Lower']), stock.curr['ATR'] + 1e-10),
        "一目多头": safe_div(stock.curr['Close'] - max(stock.curr['SenkouA'], stock.curr['SenkouB']), stock.curr['Close'] * 0.01),
        "强势回踩": safe_div(stock.curr['Close'] - stock.curr['EMA_20'], stock.curr['EMA_20'] * 0.01),
        "机构控盘(CMF)": stock.curr['CMF'], "突破缺口": safe_div(stock.curr['Open'] - stock.prev['Close'], stock.prev['Close'] * 0.01),
        "VWAP突破": safe_div(stock.curr['Close'] - stock.curr['VWAP_20'], stock.curr['VWAP_20'] * 0.01),
        "AVWAP突破": safe_div(stock.curr['Close'] - stock.curr['AVWAP'], stock.curr['AVWAP'] * 0.01) if pd.notna(stock.curr['AVWAP']) else 0.0,
        "SMC失衡区": safe_div(stock.curr['Close'] - cf.fvg_lower, stock.curr['Close'] * 0.01) if cf.fvg_lower > 0 else 0.0,
        "流动性扫盘": safe_div(stock.curr['Swing_Low_20'] - stock.curr['Low'], stock.curr['Low'] * 0.01) if pd.notna(stock.curr['Swing_Low_20']) else 0.0,
        "聪明钱抢筹": stock.curr['Smart_Money_Flow'], "巨量滞涨": vol_surge_ratio,
        "放量长阳": safe_div(stock.curr['Close'] - stock.curr['Open'], stock.curr['Open'] * 0.01),
        "口袋支点": safe_div(stock.curr['Volume'], stock.curr['Max_Down_Vol_10'], cap=50.0),
        "VCP收缩": safe_div(stock.curr['Range_20'], stock.curr['Range_60'] + 1e-10),
        "量子概率云(KDE)": cf.kde_breakout_score, "特性改变(ChoCh)": safe_div(stock.curr['Close'] - stock.swing_high_10, stock.swing_high_10 * 0.01),
        "订单块(OB)": safe_div(stock.curr['Close'] - stock.curr['OB_Low'], stock.curr['OB_High'] - stock.curr['OB_Low'] + 1e-10) if pd.notna(stock.curr['OB_High']) else 0.0,
        "AMD操盘": safe_div(min(stock.curr['Open'], stock.curr['Close']) - stock.curr['Low'], stock.curr['TR'] + 1e-10),
        "跨时空共振(周线)": 1.0 if cf.weekly_bullish else 0.0, "CVD筹码净流入": float(stock.curr['CVD_Trend'] * (0.3 if stock.curr['CVD_Divergence'] == 1 else 1.0)),
        "独立Alpha(脱钩)": cf.pure_alpha, "NR7极窄突破": safe_div(stock.curr['Range'], stock.curr['ATR'] + 1e-10),
        "VPT量价共振": (1.0 / (1.0 + np.exp(-stock.curr['VPT_ZScore']))) if stock.curr['VPT_Accel'] > 0 else 0.0,
        "带量金叉(交互)": macd_cross_strength * vol_surge_ratio, "量价吸筹(交互)": stock.curr['CMF'] * stock.curr['Smart_Money_Flow'],
        "近3日突破(滞后)": stock.curr['Recent_Price_Surge_3d'] if pd.notna(stock.curr['Recent_Price_Surge_3d']) else 0.0,
        "近3日巨量(滞后)": stock.curr['Recent_Vol_Surge_3d'] if pd.notna(stock.curr['Recent_Vol_Surge_3d']) else 0.0,
        "稳健赫斯特(Hurst)": hurst_score, "FFT多窗共振(动能)": cf.fft_ensemble_score,
        "大周期保护小周期(MACD共振)": 1.0 if (cf.weekly_macd_res == 1.0 and check_macd_cross(stock.curr, stock.prev)) else 0.0,
        "聪明钱月度净流入(月线)": cf.monthly_inst_flow, "60分钟级精准校准(RSI反弹)": cf.rsi_60m_bounce,
        "大盘Beta(宏观调整)": cf.beta_60d, "利率敏感度(TLT相关性)": cf.tlt_corr, "汇率传导(DXY相关性)": cf.dxy_corr,
        "Amihud非流动性(冲击成本)": stock.curr['Amihud'] if pd.notna(stock.curr['Amihud']) else 0.0,
        "52周高点距离(动能延续)": stock.curr['Dist_52W_High'] if pd.notna(stock.curr['Dist_52W_High']) else 0.0,
        "波动率风险溢价(VRP)": cf.vrp, "期权PutCall情绪(PCR)": alt.pcr, "隐含波动率偏度(IV Skew)": alt.iv_skew,
        "做空兴趣突变(轧空)": alt.short_change if alt.short_float > Config.Params.SHORT_SQZ_FLT else 0.0,
        "内部人集群净买入(Insider)": alt.insider_net_buy, "分析师修正动量(Analyst)": alt.analyst_mom,
        "舆情NLP情感极值(News_NLP)": alt.nlp_score, "散户热度加速度(WSB_Accel)": alt.wsb_accel
    }
    if alpha_vec is not None and len(alpha_vec) == 16:
        for i, val in enumerate(alpha_vec): feat_dict[f"Alpha_T{i+1:02d}"] = float(val)
    return {f: float(np.nan_to_num(feat_dict.get(f, 0.0), nan=0.0, posinf=20.0, neginf=-20.0)) for f in Config.ALL_FACTORS}

def _evaluate_omni_matrix(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData) -> Tuple[int, List[str], List[str], bool]:
    """🚀 还原：49 维全息打分矩阵 (包含所有技术形态与防御降权)"""
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

    gap_pct = (stock.curr['Open'] - stock.prev['Close']) / (stock.prev['Close'] + 1e-10)
    atr_pct = (stock.curr['ATR'] / (stock.prev['Close'] + 1e-10)) * 100
    day_chg = (stock.curr['Close'] - stock.curr['Open']) / (stock.curr['Open'] + 1e-10) * 100
    tr_val = stock.curr['High'] - stock.curr['Low'] + 1e-10

    if pd.notna(stock.curr['SMA_200']) and stock.curr['Close'] > stock.curr['SMA_50'] > stock.curr['SMA_150'] > stock.curr['SMA_200']:
        m_str = (stock.curr['SMA_50'] - stock.curr['SMA_200']) / (stock.curr['SMA_200'] + 1e-10)
        add_t("米奈尔维尼", f"🏆 米奈尔维尼模板形成 (强度:{m_str*100:.1f}% 权:{{fw:.2f}}x)", 8 + int(m_str*20), "TREND")
        
    if cf.rs_20 > 1.0 + (stock.curr['ATR'] / (stock.curr['Close'] + 1e-10)) * 2.0:
        add_t("强相对强度", f"⚡ 动能超越波动率动态阈值 (权:{{fw:.2f}}x)", 7 if stock.is_vol else 4, "TREND")
    
    if check_macd_cross(stock.curr, stock.prev):
        add_t("MACD金叉", f"🔥 MACD起爆 (权:{{fw:.2f}}x)", 12 if stock.curr['MACD'] > 0 else 8, "TREND")

    if stock.curr['Above_Cloud'] == 1 and stock.curr['Tenkan'] > stock.curr['Kijun']: add_t("一目多头", "🌥️ 一目均衡表云上多头共振 (权:{fw:.2f}x)", 6, "TREND")
    if pd.notna(stock.curr['AVWAP']) and stock.curr['Close'] > stock.curr['AVWAP'] and stock.prev['Close'] <= stock.curr['AVWAP']: add_t("AVWAP突破", "⚓ 强势站上AVWAP核心区 (权:{fw:.2f}x)", 12, "TREND")
        
    kc_w, bb_w = stock.curr['KC_Upper'] - stock.curr['KC_Lower'], stock.curr['BB_Upper'] - stock.curr['BB_Lower']
    if bb_w < kc_w: add_t("TTM Squeeze ON", f"📦 TTM Squeeze 激活 (权:{{fw:.2f}}x)", 8 + int(((kc_w - bb_w) / (stock.curr['ATR'] + 1e-10))*10), "VOLATILITY")
    
    if stock.prev['NR7'] and stock.prev['Inside_Bar'] and stock.curr['Close'] > stock.prev['High']: add_t("NR7极窄突破", "🎯 7日极窄压缩孕线向上爆破 (权:{fw:.2f}x)", 12, "VOLATILITY")
    
    if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Low'] < stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.curr['Swing_Low_20']:
        add_t("流动性扫盘", "🧹 刺穿前低扫掉散户止损后诱空反转 (权:{fw:.2f}x)", 15, "REVERSAL")
    
    if pd.notna(stock.curr['OB_High']) and stock.curr['Low'] <= stock.curr['OB_High'] and stock.curr['Close'] >= stock.curr['OB_Low'] and stock.curr['Close'] > stock.curr['Open']:
        add_t("订单块(OB)", "🧱 触达历史起爆底仓区 (权:{fw:.2f}x)", 15, "REVERSAL")

    if cf.kde_breakout_score > Config.Params.KDE_BREAKOUT: add_t("量子概率云(KDE)", f"☁️ KDE 揭示上行阻力极度稀薄 (权:{{fw:.2f}}x)", 15 * cf.kde_breakout_score, "QUANTUM")
    if cf.hurst_reliable and cf.hurst_med > Config.Params.HURST_RELIABLE: add_t("稳健赫斯特(Hurst)", f"⏳ R/S Bootstrap 确认强抗噪持续性 (权:{{fw:.2f}}x)", 15 * ((cf.hurst_med - 0.5) * 2.0), "QUANTUM")
    if alt.insider_net_buy > Config.Params.INSIDER_BUY: add_t("内部人集群净买入(Insider)", "👔 SEC披露高管集群大额买入 (权:{fw:.2f}x)", 20, "QUANTUM")
    if alt.analyst_mom > Config.Params.ANALYST_UP: add_t("分析师修正动量(Analyst)", f"📊 分析师评级修正突破 (权:{{fw:.2f}}x)", 8, "QUANTUM")

    lower_wick = stock.curr['Open'] - stock.curr['Low'] if stock.curr['Close'] > stock.curr['Open'] else stock.curr['Close'] - stock.curr['Low']
    if stock.curr['Close'] > stock.curr['Open'] and (lower_wick / tr_val) > 0.3 and (stock.curr['High'] - stock.curr['Close'])/tr_val < 0.15: add_t("AMD操盘", "🎭 深度开盘诱空下杀后拉升派发 (权:{fw:.2f}x)", 12, "REVERSAL")

    if stock.curr['Volume'] > stock.curr['Vol_MA20'] * 2.0 and abs(stock.curr['Close'] - stock.curr['Open']) < stock.curr['ATR'] * 0.5:
        if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Close'] < stock.curr['Swing_Low_20'] * 1.05: add_t("巨量滞涨", "🛑 底部巨量滞涨机构冰山吸筹 (权:{fw:.2f}x)", 12, "QUANTUM")
        
    if gap_pct * 100 > max(1.5, atr_pct * 0.3) and gap_pct < 0.06: add_t("突破缺口", "💥 放量跳空底部突破缺口 (权:{fw:.2f}x)", 8, "VOLATILITY")

    if alt.pcr > Config.Params.PCR_BEAR or alt.iv_skew > Config.Params.IV_SKEW_BEAR: 
        black_swan_risk = True
        add_t("期权PutCall情绪(PCR)", "⚠️ 期权市场极度看空避险 (权:{fw:.2f}x)", -15, "VOLATILITY")

    sum_s = sum(50.0 * (1 - np.exp(-v / 25.0)) for v in theme_scores.values())
    return int(100.0 * (1 - np.exp(-sum_s / 50.0))), triggered_list, factors_list, black_swan_risk

def _apply_market_filters(curr: pd.Series, prev: pd.Series, sym: str, base_score: int, sig: List[str]) -> Tuple[int, bool, List[str]]:
    total_score = base_score
    is_bearish_div = False
    if curr['Close'] >= curr['Price_High_20'] * 0.98 and curr['RSI'] < 60.0 and prev['RSI'] > 60.0:
        is_bearish_div = True
        total_score = 0 
        sig.append(f"🩸 [顶背离危局] 价格近高但动能显著衰竭，严禁追高")
    if curr['RSI'] > 85.0:
        total_score = max(0, total_score - 30)
        sig.append(f"⚠️ [极端超买] RSI 高达 {curr['RSI']:.1f} (严防获利盘踩踏)")
    bias_20 = (curr['Close'] - curr['EMA_20']) / (curr['EMA_20'] + 1e-10)
    if bias_20 > 0.08:
        total_score = max(0, total_score - int((bias_20 - 0.08) * 100))
        sig.append(f"⚠️ [短线高估] 偏离20日均线 +{bias_20*100:.1f}% (防追高扣减)")
        
    # 🚀 还原：RRG 板块相对动能轮动加权
    sym_sec = Config.get_sector_etf(sym)
    leading_sectors = ['XLK', 'XLC'] 
    lagging_sectors = ['XLV', 'XLP']
    
    if sym_sec in leading_sectors:
        total_score = int(total_score * 1.15)
        sig.append(f"🔥 [RRG象限跃迁] 所属板块 {sym_sec} 相对动能加速领跑 (+15%)")
    elif sym_sec in lagging_sectors:
        total_score = int(total_score * 0.85)
        sig.append(f"🧊 [RRG象限坠落] 所属板块 {sym_sec} 相对动能加速衰退 (-15%)")
        
    return total_score, is_bearish_div, sig

def _apply_ai_inference(raw_reports: List[dict], ctx: MarketContext) -> List[dict]:
    if not os.path.exists(Config.MODEL_FILE):
        logger.warning("未找到左脑树模型 scoring_model.pkl，退化为纯军规打分。")
        for r in raw_reports: r['ai_prob'] = 0.5
        return raw_reports
    try:
        import pickle
        with open(Config.MODEL_FILE, 'rb') as f: model_pack = pickle.load(f)
        active_factors = model_pack.get('active_factors', Config.ALL_FACTORS)
        scaler = model_pack['scaler']
        base_A, base_B, meta_clf = model_pack.get('base_A'), model_pack.get('base_B'), model_pack['meta']
        active_A, active_B = model_pack.get('active_A', []), model_pack.get('active_B', [])
        
        df_feats = pd.DataFrame([r['ml_features'] for r in raw_reports]).fillna(0.0)
        for f in active_factors:
            if f not in df_feats.columns: df_feats[f] = 0.0
        X_scaled = scaler.transform(df_feats[active_factors])
        X_scaled_df = pd.DataFrame(X_scaled, columns=active_factors)
        
        prob_A = base_A.predict_proba(X_scaled_df[active_A].values)[:, 1] if base_A and active_A else np.full(len(raw_reports), 0.5)
        prob_B = base_B.predict_proba(X_scaled_df[active_B].values)[:, 1] if base_B and active_B else np.full(len(raw_reports), 0.5)
        
        X_meta = np.column_stack([prob_A, prob_B, np.full(len(raw_reports), ctx.vix_current / 20.0), np.full(len(raw_reports), ctx.credit_spread_mom), np.full(len(raw_reports), ctx.vix_term_structure - 1.0), np.full(len(raw_reports), ctx.market_pcr - 1.0)])
        final_probs = meta_clf.predict_proba(X_meta)[:, 1]
        
        for i, r in enumerate(raw_reports):
            r['ai_prob'] = float(final_probs[i])
            if r['ai_prob'] > 0.65:
                r['total_score'] = int(r['total_score'] * 1.2)
                r['sig'].append(f"🧠 [AI 确认] Meta-Learner 胜率极高 ({r['ai_prob']:.1%})，权重上调")
            elif r['ai_prob'] < 0.40:
                r['total_score'] = int(r['total_score'] * 0.5)
                r['sig'].append(f"⚠️ [AI 否决] Meta-Learner 胜率低迷 ({r['ai_prob']:.1%})，权重削减")
    except Exception as e: logger.error(f"AI 推理管线崩溃: {e}")
    return raw_reports

def _apply_kelly_cluster_optimization(reports: List[dict], price_history_dict: dict, total_exposure: float, ctx: MarketContext) -> List[dict]:
    """🚀 还原：Kelly-CVaR 风险平价与凸优化阵型"""
    reports.sort(key=lambda x: (x["score"], x["ai_prob"]), reverse=True)
    candidate_pool = reports[:15]
    if not candidate_pool: return []
    if len(candidate_pool) == 1:
        candidate_pool[0]['pos_advice'] = f"✅ 组合配置权重: {total_exposure * 100:.1f}% (极简化单目标资产)"
        candidate_pool[0]['opt_weight'] = 1.0
        return candidate_pool
    try:
        syms = [r['symbol'] for r in candidate_pool]
        ret_df = pd.DataFrame({sym: price_history_dict[sym] for sym in syms}).ffill().pct_change().dropna()
        cov_matrix = LedoitWolf().fit(ret_df).covariance_
        
        cvars = []
        for sym in syms:
            rets = ret_df[sym].values
            var_95 = np.percentile(rets, 5) if len(rets) > 0 else 0
            cvars.append(abs(min(rets[rets <= var_95].mean() if len(rets[rets <= var_95]) > 0 else var_95, 0.0)))
            
        norm_scores = np.array([r['score'] for r in candidate_pool]) / (np.max([r['score'] for r in candidate_pool]) + 1e-10)
        risk_aversion = max(2.0, 5.0 - ctx.health_score * 3.0)
        
        def objective(w): return -(np.dot(w, norm_scores) - risk_aversion * np.dot(w, cvars))
        bounds = tuple((0.02, 0.15) for _ in range(len(syms))) 
        init_w = np.ones(len(syms)) / len(syms)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: (0.25 / np.sqrt(252)) ** 2 - np.dot(w.T, np.dot(cov_matrix, w))}]
        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_weights = res.x if res.success else (norm_scores * (1.0 / (np.array(cvars) + 1e-5))) / np.sum(norm_scores * (1.0 / (np.array(cvars) + 1e-5)))
        for i, r in enumerate(candidate_pool):
            r['opt_weight'] = opt_weights[i]
            r['cvar_95'] = cvars[i]
            r['pos_advice'] = f"✅ 组合配置权重: {opt_weights[i] * total_exposure * 100:.1f}% (Kelly Cluster 分配)"
        return candidate_pool
    except Exception: return candidate_pool

def _calculate_position_size(stock: StockData, ctx: MarketContext, ai_prob: float, is_bearish_div: bool, black_swan: bool) -> Tuple[float, float, float, str]:
    # 🚀 满血还原：VIX 波动率标量与最高价棘轮吊灯止损 (Chandelier Exit)
    atr_mult_sl = 1.0 if ctx.vix_inv else 1.5
    atr_mult_tp = 2.0 if ctx.vix_inv else 3.0
    
    tp_val = stock.curr['Close'] + atr_mult_tp * ctx.vix_scalar * stock.curr['ATR']
    
    # 优先使用形态止损线，否则回退到 ATR 动态止损
    sl_chandelier = stock.curr['Chandelier_Exit'] if pd.notna(stock.curr['Chandelier_Exit']) else (stock.curr['Close'] - atr_mult_sl * ctx.vix_scalar * stock.curr['ATR'])
    sl_val = max(sl_chandelier, stock.curr['Close'] - atr_mult_sl * ctx.vix_scalar * stock.curr['ATR'])
    
    kelly = ai_prob - (1.0 - ai_prob) / 2.0
    
    if black_swan: return tp_val, sl_val, 0.0, "❌ 黑天鹅极险"
    if kelly <= 0 or is_bearish_div: return tp_val, sl_val, 0.0, "❌ 盈亏比劣势"
    return tp_val, sl_val, max(0.0, kelly), ""

def _route_orders_to_gateway(final_reports: List[dict], ctx: MarketContext) -> None:
    if not final_reports: return
    ledger = OrderLedger(Config.ORDER_DB_PATH)
    
    # 🚀 终极防线：持仓与挂单对账，防止无限重复买入
    existing_orders = ledger.fetch_orders_by_status(['PENDING_SUBMIT', 'SUBMITTED', 'OPEN', 'PARTIALLY_FILLED', 'FILLED'])
    existing_symbols = set(existing_orders['symbol'].tolist()) if not existing_orders.empty else set()
    
    routed_count = 0
    for r in final_reports:
        if r['symbol'] in existing_symbols:
            logger.info(f"⏭️ [网关路由] {r['symbol']} 已在账本中存在持仓或挂单，跳过重复发单。")
            continue
        qty = round((Config.Params.PORTFOLIO_VALUE * ctx.total_market_exposure * r.get('opt_weight', 0.1)) / max(r['curr_close'], 1e-10), 4)
        if qty > 0:
            ledger.insert_order(r['symbol'], 'BUY', qty, r['curr_close'])
            routed_count += 1
    logger.info(f"✅ [网关路由] 成功推送 {routed_count} 笔实盘发单指令到 SQLite。")

# ================= 6. 核心管线重构 (四大支柱) =================

def run_matrix():
    """支柱 1：每日决策主脑"""
    logger.info("🚀 启动 Matrix 决策主脑...")
    ctx = _build_market_context()
    
    prepared_data, price_dict = _prepare_universe_data(ctx)
    if not prepared_data: return
    
    if getattr(ctx, 'transformer_model', None) is not None:
        seqs = np.array([d['seq'] for d in prepared_data])
        alpha_vecs = ctx.transformer_model.extract_alpha(seqs)
        for i, d in enumerate(prepared_data): d['alpha_vec'] = alpha_vecs[i]
    else:
        for d in prepared_data: d['alpha_vec'] = np.zeros(16)

    raw_reports = []
    for d in prepared_data:
        ml_feat = _extract_ml_features(d['stock'], ctx, d['cf'], d['alt'], d['alpha_vec'])
        base_score, sig, factors, black_swan = _evaluate_omni_matrix(d['stock'], ctx, d['cf'], d['alt'])
        score, is_bearish_div, sig = _apply_market_filters(d['curr'], d['prev'], d['sym'], base_score, sig)
        raw_reports.append({'sym': d['sym'], 'curr': d['curr'], 'prev': d['prev'], 'score': score, 'sig': sig, 'factors': factors, 'ml_features': ml_feat, 'news': d['news'], 'sym_sec': Config.get_sector_etf(d['sym']), 'is_bearish_div': is_bearish_div, 'black_swan_risk': black_swan, 'total_score': score, 'is_untradeable': False, 'ai_prob': 0.0})

    raw_reports = _apply_ai_inference(raw_reports, ctx)
    reports = []
    background_pool = []
    for r in raw_reports:
        tp, sl, kelly, advice = _calculate_position_size(StockData(r['sym'], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), r['curr'], r['prev'], False, 0.0), ctx, r['ai_prob'], r['is_bearish_div'], r['black_swan_risk'])
        if r['total_score'] > 0 and kelly > 0:
            reports.append({'symbol': r['sym'], 'score': r['total_score'], 'ai_prob': r['ai_prob'], 'signals': r['sig'][:8], 'factors': r['factors'], 'ml_features': r['ml_features'], 'curr_close': float(r['curr']['Close']), 'tp': tp, 'sl': sl, 'news': r['news'], 'sector': r['sym_sec'], 'pos_advice': advice, 'kelly_fraction': kelly})
        else:
            background_pool.append({'symbol': r['sym'], 'score': r['total_score'], 'ai_prob': r['ai_prob'], 'factors': r['factors'], 'ml_features': r['ml_features']})

    if reports:
        ctx.dynamic_min_score = max(Config.Params.MIN_SCORE_THRESHOLD, np.percentile([r['score'] for r in reports], 85))
        reports = [r for r in reports if r['score'] >= ctx.dynamic_min_score]
        
        # 🚀 还原：同质化板块拥挤度惩罚防线
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
        
        for sec, stks in groups.items():
            if sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.Params.CROWDING_MIN_STOCKS:
                pen = max(0.6, min(0.9, Config.Params.CROWDING_PENALTY * (1.0 + ctx.health_score * 0.3)))
                for s in stks[1:]: 
                    s["score"] = int(s["score"] * pen)
                    s["signals"].append(f"⚠️ [防拥挤] 同板块标的过多，触发多元化防线降权 (-{(1-pen)*100:.0f}%)")

        final_reports = _apply_kelly_cluster_optimization(reports, price_dict, ctx.total_market_exposure, ctx)
        for r in final_reports: set_alerted(r["symbol"])
        
        # 🚀 还原：生成暗影背景池 (Shadow Pool) 供 MLOps 提取负样本
        final_symbols = {r['symbol'] for r in final_reports}
        unselected_bg = [s for s in background_pool if s['symbol'] not in final_symbols]
        random.shuffle(unselected_bg)
        final_shadow_pool = unselected_bg[:150]
        for r in final_shadow_pool: set_alerted(r["symbol"], is_shadow=True, shadow_data=r)
            
        _route_orders_to_gateway(final_reports, ctx)
        
        msg = "\n\n".join([f"**{r['symbol']}** | 评分: {r['score']} | AI胜率: {r.get('ai_prob',0):.1%} | 现价: {r['curr_close']:.2f}\n" + "\n".join(r['signals']) for r in final_reports[:10]])
        
        try:
            with open(Config.get_current_log_file(), "a", encoding="utf-8") as f:
                log_entry = {
                    "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), 
                    "macro_meta": {"vix": ctx.vix_current, "credit_spread_mom": ctx.credit_spread_mom, "vix_term_structure": ctx.vix_term_structure, "market_pcr": ctx.market_pcr},
                    "top_picks": [{"symbol": r.get("symbol"), "score": r.get("score"), "signals": r.get("signals"), "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "ai_prob": r.get("ai_prob", 0.0), "tp": r.get("tp"), "sl": r.get("sl")} for r in final_reports],
                    "shadow_pool": [{"symbol": r.get("symbol"), "score": r.get("score"), "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "ai_prob": r.get("ai_prob", 0.0)} for r in final_shadow_pool]
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e: logger.error(f"写入全息日志崩溃: {e}")

        send_alert("Matrix 扫盘决断战报", f"今日 VIX: {ctx.vix_current:.1f}\n\n{msg}")
    logger.info("✅ Matrix 决策管线执行完毕。")

def run_backtest():
    """支柱 2：周末复盘与回测引擎 (TimeSeriesSplit + Rank IC)"""
    logger.info("⏳ 启动历史日志回测与模型代谢管线...")
    log_files = [f for f in os.listdir(Config.DATA_DIR) if f.startswith(Config.LOG_PREFIX) and f.endswith('.jsonl')]
    if not log_files: 
        logger.warning("未找到任何历史日志，回测取消。")
        return
        
    trades = []
    for lf in log_files:
        try:
            with open(os.path.join(Config.DATA_DIR, lf), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log = json.loads(line.strip())
                        
                        # 🚀 满血还原：合并正样本(top_picks)与反事实负样本(shadow_pool)
                        daily_samples = log.get('top_picks', []) + log.get('shadow_pool', [])
                        
                        for p in daily_samples:
                            trades.append({'date': log['date'], 'vix': log.get('macro_meta', {}).get('vix', 18.0), 'symbol': p['symbol'], 'ml_features': p.get('ml_features', {}), 'factors': p.get('factors', []), 'sl': p.get('sl', 0.0), 'tp': p.get('tp', float('inf'))})
                    except Exception: pass
        except Exception: pass
            
    if not trades: return
    syms = list(set([t['symbol'] for t in trades]))
    
    dfs = []
    logger.info(f"下载 {len(syms)} 支标的近1年历史轨迹...")
    for i in range(0, len(syms), 40):
        try:
            chunk = yf.download(syms[i:i+40], period="1y", progress=False, threads=False, timeout=15)
            if not chunk.empty: dfs.append(chunk)
        except Exception: pass
        time.sleep(1)
        
    if not dfs: return
    df_all = pd.concat(dfs, axis=1)
    df_c = df_all['Close'] if isinstance(df_all.columns, pd.MultiIndex) else df_all[['Close']].rename(columns={'Close': syms[0]})
    df_o = df_all['Open'] if isinstance(df_all.columns, pd.MultiIndex) else df_all[['Open']].rename(columns={'Open': syms[0]})
    df_h = df_all['High'] if isinstance(df_all.columns, pd.MultiIndex) else df_all[['High']].rename(columns={'High': syms[0]})
    df_l = df_all['Low'] if isinstance(df_all.columns, pd.MultiIndex) else df_all[['Low']].rename(columns={'Low': syms[0]})
    df_c.index = df_c.index.strftime('%Y-%m-%d')
    df_o.index = df_o.index.strftime('%Y-%m-%d')
    df_h.index = df_h.index.strftime('%Y-%m-%d')
    df_l.index = df_l.index.strftime('%Y-%m-%d')
    
    SLIPPAGE, COMMISSION = Config.Params.SLIPPAGE, Config.Params.COMMISSION
    stats_period = {'T+1': [], 'T+3': [], 'T+5': []}
    mae_mfe = {'T+1': [], 'T+3': [], 'T+5': []}
    trades_with_ret = []
    
    transformer_X, transformer_Y = [], []
    cached_inds = {}
    
    for t in trades:
        sym, r_dt = t['symbol'], t['date']
        initial_sl, tp_price = t.get('sl', 0.0), t.get('tp', float('inf'))
        if sym not in df_c.columns: continue
        valid = df_c.index[df_c.index >= r_dt]
        if len(valid) == 0: continue
        e_idx = df_c.index.get_loc(valid[0])
        if e_idx + 1 >= len(df_c): continue
        
        entry = df_o.iloc[e_idx + 1][sym]
        if pd.isna(entry) or entry <= 0: continue
        entry_cost = entry * (1 + SLIPPAGE + COMMISSION)
        trail_dist = max(entry * 0.02, entry - initial_sl) if initial_sl > 0 else entry * 0.05
        
        for d in [1, 3, 5]:
            if e_idx + d < len(df_c):
                exit_revenue, dyn_sl, highest_px = None, initial_sl, entry
                max_h, min_l = entry, entry
                for i in range(1, d + 1):
                    curr_idx = e_idx + i
                    d_o, d_h, d_l, d_c = df_o.iloc[curr_idx][sym], df_h.iloc[curr_idx][sym], df_l.iloc[curr_idx][sym], df_c.iloc[curr_idx][sym]
                    if pd.isna(d_o): continue
                    max_h, min_l = max(max_h, d_h), min(min_l, d_l)
                    
                    if dyn_sl > 0 and d_o < dyn_sl: exit_revenue = d_o * (1 - SLIPPAGE*5 - COMMISSION); break
                    if dyn_sl > 0 and d_l <= dyn_sl: exit_revenue = dyn_sl * (1 - SLIPPAGE*3 - COMMISSION); break
                    if tp_price > 0 and d_h >= tp_price: exit_revenue = tp_price * (1 - SLIPPAGE - COMMISSION); break
                    
                    highest_px = max(highest_px, d_h)
                    dyn_sl = max(dyn_sl, highest_px - trail_dist)
                    if i == d: exit_revenue = d_c * (1 - SLIPPAGE - COMMISSION)
                
                if exit_revenue is not None:
                    ret = (exit_revenue - entry_cost) / entry_cost
                    stats_period[f'T+{d}'].append(ret)
                    mae_mfe[f'T+{d}'].append({'mfe': (max_h - entry)/entry, 'mae': (min_l - entry)/entry})
                    if d == 3: 
                        trades_with_ret.append({**t, 'ret': ret})
                        if TRANSFORMER_AVAILABLE:
                            try:
                                if sym not in cached_inds:
                                    sym_df = pd.DataFrame({'Close': df_c[sym], 'Open': df_o[sym], 'High': df_h[sym], 'Low': df_l[sym], 'Volume': 1000000})
                                    cached_inds[sym] = calculate_indicators(sym_df)
                                ind_df = cached_inds[sym]
                                valid_dates = ind_df.index[ind_df.index <= r_dt]
                                if len(valid_dates) > 0:
                                    t_idx = ind_df.index.get_loc(valid_dates[-1])
                                    if isinstance(t_idx, slice): t_idx = t_idx.stop - 1
                                    elif isinstance(t_idx, np.ndarray): t_idx = np.where(t_idx)[0][-1]
                                    seq = _get_transformer_seq(ind_df, end_idx=t_idx + 1)
                                    if not np.all(seq == 0):
                                        transformer_X.append(seq)
                                        transformer_Y.append(ret)
                            except Exception: pass
            
    res = {}
    for p, r in stats_period.items():
        if not r: continue
        ret_arr = np.array(r)
        wins, losses = ret_arr[ret_arr > 0], ret_arr[ret_arr < 0]
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 99.0
        win_maes = [rec['mae'] for rec, x in zip(mae_mfe[p], ret_arr) if x > 0]
        res[p] = {'win_rate': len(wins)/len(ret_arr), 'avg_ret': np.mean(ret_arr), 'sharpe': np.mean(ret_arr)/(np.std(ret_arr)+1e-10), 'profit_factor': pf, 'total_trades': len(ret_arr), 'avg_win_mae': np.mean(win_maes) if win_maes else 0.0}

    if len(trades_with_ret) >= 30:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.linear_model import LogisticRegression
            import pickle
            
            trade_df = pd.DataFrame(trades_with_ret).sort_values('date')
            X_all_df = pd.DataFrame(trade_df['ml_features'].tolist()).fillna(0.0)
            for f in Config.ALL_FACTORS:
                if f not in X_all_df.columns: X_all_df[f] = 0.0
            
            y_all_cont = trade_df['ret'].values
            y_all_class = (y_all_cont > 0.015).astype(int)
            
            ic_records = {f: [] for f in Config.ALL_FACTORS}
            for d_str in np.unique(trade_df['date'].values):
                mask = trade_df['date'].values == d_str
                if np.sum(mask) < 5 or np.std(y_all_cont[mask]) < 1e-6: continue
                for f in Config.ALL_FACTORS:
                    ic, _ = stats.spearmanr(X_all_df[mask][f].values, y_all_cont[mask])
                    ic_records[f].append(float(ic) if not np.isnan(ic) else 0.0)
                    
            active_factors_list = [f for f, ics in ic_records.items() if len(ics)>5 and abs(np.mean(ics)/(np.std(ics)/np.sqrt(len(ics))+1e-10)) > 1.0]
            if len(active_factors_list) < 5: active_factors_list = Config.ALL_FACTORS[:10]
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_all_df[active_factors_list])
            
            tscv = TimeSeriesSplit(n_splits=3)
            oof_pred = np.zeros(len(y_all_class))
            lgbm = LGBMClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, class_weight='balanced', random_state=42)
            
            for train_idx, val_idx in tscv.split(X_scaled):
                lgbm.fit(X_scaled[train_idx], y_all_class[train_idx])
                oof_pred[val_idx] = lgbm.predict_proba(X_scaled[val_idx])[:, 1]
                
            meta = LogisticRegression(class_weight='balanced', random_state=42)
            meta.fit(np.column_stack([oof_pred, np.zeros(len(X_scaled)), np.zeros(len(X_scaled)), np.zeros(len(X_scaled)), np.zeros(len(X_scaled)), np.zeros(len(X_scaled))]), y_all_class)
            
            lgbm.fit(X_scaled, y_all_class)
            
            clf_model = {'version': Config.MODEL_VERSION, 'active_factors': active_factors_list, 'active_A': active_factors_list, 'active_B': [], 'scaler': scaler, 'base_A': lgbm, 'base_B': None, 'meta': meta}
            with open(Config.MODEL_FILE, 'wb') as f: pickle.dump(clf_model, f)
            logger.info(f"✅ Rank IC 代谢完成，保留 {len(active_factors_list)} 维高能特征，LightGBM 元学习器重训落盘。")
        except Exception as e: logger.error(f"模型训练崩溃: {e}")

    if TRANSFORMER_AVAILABLE and len(transformer_X) >= 64:
        logger.info(f"🧠 [MLOps] 已捕获 {len(transformer_X)} 笔三元组时序切片，正压入训练数据共享缓冲区...")
        try:
            X_arr, Y_arr = np.array(transformer_X, dtype=np.float32), np.array(transformer_Y, dtype=np.float32)
            if os.path.exists(Config.TRAINING_BUFFER):
                try:
                    existing = np.load(Config.TRAINING_BUFFER)
                    X_arr, Y_arr = np.concatenate([existing['X'], X_arr]), np.concatenate([existing['Y'], Y_arr])
                except Exception: pass
            tmp_buf = Config.TRAINING_BUFFER + ".tmp"
            np.savez(tmp_buf, X=X_arr, Y=Y_arr)
            os.replace(tmp_buf, Config.TRAINING_BUFFER)
            logger.info(f"✅ 样本集成功落盘 (当前总样本池: {len(X_arr)} 笔)。")
        except Exception as e: logger.error(f"⚠️ 缓冲区落盘失败: {e}")

    report_md = [f"# 📈 自动量化战报与 AI 透视\n**更新:** {datetime_to_str()}\n\n## ⚔️ 核心表现评估\n| 周期 | 原始胜率 | 均收益 | 盈亏比 | Sharpe | 胜单平均抗压(MAE) | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for p in ['T+1', 'T+3', 'T+5']:
        d = res.get(p, {'win_rate':0,'avg_ret':0,'profit_factor':0,'sharpe':0,'avg_win_mae':0,'total_trades':0})
        report_md.append(f"| {p} | {d['win_rate']*100:.1f}% | {d['avg_ret']*100:+.2f}% | {d['profit_factor']:.2f} | {d['sharpe']:.2f} | {d['avg_win_mae']*100:.1f}% | {d['total_trades']} |")
    
    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_md))
    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump({"overall": res}, f, indent=4)
    
    del dfs, df_all, trades_with_ret, transformer_X, transformer_Y, cached_inds
    gc.collect()

    send_alert("周末反思代谢报告", f"本周完成 {res.get('T+3', {}).get('total_trades', 0)} 笔严格动态滑点与吊灯止损轨迹回测。\n\n**T+3 胜率**: {res.get('T+3', {}).get('win_rate', 0)*100:.1f}%\n**盈亏比**: {res.get('T+3', {}).get('profit_factor', 0):.2f}\n\n已生成全息 Markdown 战报，Rank IC 与 Meta-Learner 参数已更新。")

def run_gateway():
    """支柱 3：实盘执行网关"""
    logger.info("📡 启动 Execution Gateway...")
    
    if Config.ALPACA_API_KEY and Config.ALPACA_API_SECRET:
        logger.info("🔗 检测到 Alpaca API 密钥，切换至【真实/纸面券商环境】!")
        broker = AlpacaGateway(Config.ALPACA_API_KEY, Config.ALPACA_API_SECRET, Config.ALPACA_BASE_URL)
    else:
        logger.warning("⚠️ 未配置 Alpaca API 密钥，当前运行在【Mock 模拟沙盒模式】!")
        broker = MockAlpacaGateway()
        
    ExecutionEngine(broker).run()

def _decompose_and_perturb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_phase, df_noise = df.copy(), df.copy()
    c = df['Close']
    trend = c.ewm(span=60, adjust=False).mean()
    high_freq_noise = c - c.ewm(span=5, adjust=False).mean()
    seasonal = c.ewm(span=5, adjust=False).mean() - trend
    fft_vals = np.fft.rfft(seasonal.values)
    angles = np.random.uniform(0, 2 * np.pi, len(fft_vals))
    angles[0] = 0.0 
    randomized_fft = fft_vals * np.exp(1j * angles)
    shifted_seasonal = pd.Series(np.fft.irfft(randomized_fft, n=len(seasonal.values)), index=seasonal.index)
    c_phase = trend + shifted_seasonal + high_freq_noise
    c_noise = trend + seasonal + high_freq_noise * 1.5
    ratio_phase = c_phase / (c + 1e-10)
    df_phase['Close'], df_phase['Open'], df_phase['High'], df_phase['Low'] = c_phase, df['Open']*ratio_phase, df['High']*ratio_phase, df['Low']*ratio_phase
    ratio_noise = c_noise / (c + 1e-10)
    df_noise['Close'], df_noise['Open'], df_noise['High'], df_noise['Low'] = c_noise, df['Open']*ratio_noise, df['High']*ratio_noise, df['Low']*ratio_noise
    return df_phase, df_noise

def run_synthetic_stress_test() -> None:
    """支柱 4：合成数据对抗压测"""
    logger.info("🌪️ 启动合成数据对抗压测引擎...")
    active_pool = get_filtered_watchlist(max_stocks=30) 
    metrics = {'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}}
    ctx = _build_market_context()
    
    def _worker(sym):
        df_raw = safe_get_history(sym, "5y", "1d", fast_mode=True)
        if len(df_raw) < 500: return None
        df_phase, df_noise = _decompose_and_perturb(df_raw)
        datasets = {'Original': df_raw, 'Phase_Chaos': df_phase, 'Noise_Explosion': df_noise}
        local_metrics = {'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}}
        for env_name, df_env in datasets.items():
            try:
                df_ind = calculate_indicators(df_env)
                for i in range(len(df_ind) - 252, len(df_ind) - 5):
                    curr, prev = df_ind.iloc[i], df_ind.iloc[i-1]
                    if (curr['ATR'] / curr['Close'] > 0.15) or (pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200']): continue
                    stock_data = StockData(sym, df_ind.iloc[:i+1], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), curr, prev, False, curr['High'])
                    score, _, _, _ = _evaluate_omni_matrix(stock_data, ctx, ComplexFeatures(False,0,0,0,0,0.5,0,False,0,0,0,1.0,0,0,0,0,0), AltData(0,0,0,0,0,0,0,0))
                    if score >= Config.Params.MIN_SCORE_THRESHOLD:
                        ret = (df_ind['Close'].iloc[i+4] - df_ind['Open'].iloc[i+1]) / df_ind['Open'].iloc[i+1]
                        local_metrics[env_name]['trades'] += 1
                        local_metrics[env_name]['ret_sum'] += ret
                        if ret > 0: local_metrics[env_name]['wins'] += 1
            except Exception: pass
        return local_metrics

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.Params.MAX_WORKERS) as executor:
        for res in executor.map(_worker, active_pool[:10]):
            if res:
                for env in metrics:
                    metrics[env]['trades'] += res[env]['trades']
                    metrics[env]['wins'] += res[env]['wins']
                    metrics[env]['ret_sum'] += res[env]['ret_sum']

    report = ["### 🌪️ **对抗样本压力测试**\n"]
    for env, m in metrics.items():
        wr = (m['wins']/m['trades'])*100 if m['trades']>0 else 0
        avg_ret = (m['ret_sum']/m['trades'])*100 if m['trades']>0 else 0
        report.append(f"**{env}**: 交易 {m['trades']} 笔 | 胜率 {wr:.1f}% | 均利 {avg_ret:.2f}%")
    send_alert("对抗压测战报", "\n".join(report))

# ================= 7. 全局调度 CLI 路由与 TDD 测谎仪 =================
class QuantEngineTests(unittest.TestCase):
    """🚀 还原：14 道地狱级防爆墙与形态测谎仪"""
    def setUp(self):
        np.random.seed(42)
        periods = 150
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        base_trend = np.linspace(100, 250, periods)
        noise = np.random.normal(0, 1.5, periods)
        close_px = base_trend + noise
        self.mock_df = pd.DataFrame({
            'Open': close_px - np.random.uniform(0.5, 2.0, periods),
            'High': close_px + np.random.uniform(0.5, 3.0, periods),
            'Low': close_px - np.random.uniform(0.5, 3.0, periods),
            'Close': close_px, 'Volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)

        self.test_db = ".quantbot_data/test_order_state.db"
        if os.path.exists(self.test_db): os.remove(self.test_db)
        if os.path.exists(self.test_db + "-wal"): os.remove(self.test_db + "-wal")
        if os.path.exists(self.test_db + "-shm"): os.remove(self.test_db + "-shm")
        self.ledger = OrderLedger(self.test_db)
        self.broker = MockAlpacaGateway()
        self.engine = ExecutionEngine(self.broker, self.test_db)
        
        self.orig_tca = Config.TCA_LOG_PATH
        Config.TCA_LOG_PATH = ".quantbot_data/test_tca_history.jsonl"
        if os.path.exists(Config.TCA_LOG_PATH): os.remove(Config.TCA_LOG_PATH)

    def tearDown(self):
        Config.TCA_LOG_PATH = self.orig_tca

    def _get_mock_context(self) -> MarketContext:
        return MarketContext(
            regime="bull", regime_desc="", w_mul=1.0, xai_weights={},
            vix_current=15.0, vix_desc="", vix_scalar=1.0, max_risk=0.015,
            macro_gravity=False, is_credit_risk_high=False, vix_inv=False,
            qqq_df=pd.DataFrame(), macro_data={}, total_market_exposure=1.0,
            health_score=1.0, pain_warning="", credit_spread_mom=0.0,
            vix_term_structure=1.0, market_pcr=1.0, dynamic_min_score=8.0
        )

    def _eval_factors_for_df(self, df: pd.DataFrame) -> list:
        df_ind = calculate_indicators(df.copy(deep=True))
        ctx = self._get_mock_context()
        curr, prev = df_ind.iloc[-1], df_ind.iloc[-2]
        is_vol = curr['Volume'] > curr['Vol_MA20'] * 1.5
        swing_high_10 = df_ind['High'].iloc[-11:-1].max() if len(df_ind) >= 11 else curr['High']
        stock = StockData("TEST", df_ind, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), curr, prev, is_vol, swing_high_10)
        cf = _extract_complex_features(stock, ctx)
        alt = AltData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        _, _, factors, _ = _evaluate_omni_matrix(stock, ctx, cf, alt)
        return factors

    # ---------------- 核心主脑特征算法测谎 ----------------
    def test_indicator_shapes_and_nans(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        self.assertEqual(len(df_res), len(self.mock_df), "输出的 DataFrame 长度发生截断缩水！")
        for col in ['RSI', 'MACD', 'ATR', 'CMF', 'VPT_ZScore']: self.assertIn(col, df_res.columns)

    def test_indicator_value_bounds_parameterized(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        bounds = [('RSI', 0.0, 100.0), ('CMF', -1.0, 1.0), ('ATR', 0.0, None), ('SuperTrend_Up', 0.0, 1.0)]
        for col, min_v, max_v in bounds:
            valid_vals = df_res[col].dropna()
            if min_v is not None: self.assertTrue((valid_vals >= min_v - 1e-5).all())
            if max_v is not None: self.assertTrue((valid_vals <= max_v + 1e-5).all())
            
    def test_rsi_logic_on_uptrend(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        self.assertTrue((df_res['RSI'].iloc[-30:] > 50).all(), "严格构造的单调上涨序列中，RSI 居然跌破 50！")
        
    def test_macd_cross_detection(self):
        close = np.linspace(100, 70, 59).tolist()
        close.append(110.0) 
        df = pd.DataFrame({'Close': close, 'Open': [c*0.99 for c in close], 'High': [c*1.05 for c in close], 'Low': [c*0.95 for c in close], 'Volume': 1000000}, index=pd.date_range('2023-01-01', periods=60))
        res = calculate_indicators(df)
        self.assertTrue(res.iloc[-2]['MACD'] < res.iloc[-2]['Signal_Line'])
        self.assertTrue(res.iloc[-1]['MACD'] > res.iloc[-1]['Signal_Line'])

    def test_vwap_and_avwap_logic(self):
        periods = 100
        base_px = np.linspace(10, 20, periods)
        df = pd.DataFrame({'Open': base_px*0.98, 'High': base_px*1.05, 'Low': base_px*0.95, 'Close': base_px, 'Volume': 1000000}, index=pd.date_range('2023-01-01', periods=periods))
        df.iloc[50, df.columns.get_loc('Volume')] = 50000000 
        tp_anchor = (df.iloc[50]['High'] + df.iloc[50]['Low'] + df.iloc[50]['Close']) / 3.0
        res = calculate_indicators(df)
        self.assertAlmostEqual(res['AVWAP'].iloc[50], tp_anchor, places=4, msg="AVWAP 锚点归零重启逻辑失效！")

    def test_chandelier_exit_logic(self):
        res = calculate_indicators(self.mock_df.copy(deep=True))
        self.assertTrue((res['Chandelier_Exit'].iloc[-50:] < res['Highest_22'].iloc[-50:]).all())

    def test_factor_amd_and_spring(self):
        close = np.linspace(100, 90, 30)  
        df = pd.DataFrame({'Close': close, 'Open': close, 'High': close+1, 'Low': close-1, 'Volume': 1000000}, index=pd.date_range('2023-01-01', periods=30))
        df.iloc[-1, df.columns.get_loc('Open')] = 91.0
        df.iloc[-1, df.columns.get_loc('Low')] = 85.0
        df.iloc[-1, df.columns.get_loc('Close')] = 95.0
        df.iloc[-1, df.columns.get_loc('High')] = 95.5
        factors = self._eval_factors_for_df(df)
        self.assertIn("流动性扫盘", factors, "猎杀止损的【流动性扫盘】被漏判！")
        self.assertIn("AMD操盘", factors, "机构诱空拉升的【AMD操盘模型】被漏判！")

    def test_nan_and_zero_handling(self):
        df = pd.DataFrame({'Open': 100.0, 'High': 105.0, 'Low': 95.0, 'Close': 100.0, 'Volume': 1000000.0}, index=pd.date_range('2023-01-01', periods=250))
        df.iloc[10:15, :] = np.nan
        df.iloc[80:85, df.columns.get_loc('Volume')] = 0.0
        res = calculate_indicators(df)
        self.assertFalse(np.isinf(res.select_dtypes(include=[np.number])).values.any(), "输入含 NaN/0 导致矩阵出现 Inf 致命异常！")

    # ---------------- IPC 异步执行网关与状态机测谎 ----------------
    def test_fsm_state_transitions(self):
        """测谎 1: 验证 PENDING -> OPEN -> FILLED 的黄金流程路径"""
        oid = self.ledger.insert_order("NVDA", "BUY", 10.0, 500.0)
        self.engine._process_queue()
        df = self.ledger.fetch_orders_by_status(['OPEN'])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['status'], 'OPEN')
        self.assertIsNotNone(df.iloc[0]['broker_oid'])
            
        self.engine._sync_open_orders()
        df_filled = self.ledger.fetch_orders_by_status(['FILLED'])
        self.assertEqual(len(df_filled), 1)
        self.assertEqual(df_filled.iloc[0]['filled_qty'], 10.0)

    def test_crash_recovery_consistency(self):
        """测谎 2: 验证网关崩溃后的“最终一致性”恢复能力"""
        with self.ledger._get_conn() as conn:
            conn.execute("INSERT INTO orders (client_oid, symbol, side, qty, order_type, arrival_price, status, broker_oid) VALUES ('CRASH_TEST_1', 'AMD', 'BUY', 5.0, 'MKT', 100.0, 'OPEN', 'MOCK_BROKER_ID_1')")
        
        self.broker._mock_exchange['MOCK_BROKER_ID_1'] = {"status": "FILLED", "filled_qty": 5.0, "avg_fill_price": 101.5, "qty": 5.0, "price": 100.0}
        self.engine._recover_state()
        
        df = self.ledger.fetch_orders_by_status(['FILLED'])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['client_oid'], 'CRASH_TEST_1')
        self.assertEqual(df.iloc[0]['avg_fill_price'], 101.5)

    def test_fat_finger_kill_switch(self):
        """测谎 3: 验证胖手指防线，防止错下天量订单"""
        oid = self.ledger.insert_order("AAPL", "BUY", 100000.0, 100.0) 
        self.engine._process_queue()
        df = self.ledger.fetch_orders_by_status(['REJECTED'])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['client_oid'], oid)

    def test_tca_slippage_math_guard(self):
        """测谎 4: 验证 TCA 滑点运算中的 `1e-10` 毒药防御"""
        oid = self.ledger.insert_order("GME", "BUY", 10.0, 0.0)
        self.engine._process_queue()
        self.engine._sync_open_orders()
        self.assertTrue(os.path.exists(Config.TCA_LOG_PATH))
        with open(Config.TCA_LOG_PATH, 'r') as f:
            logs = [json.loads(line) for line in f]
            self.assertEqual(len(logs), 1)
            self.assertTrue(logs[0]['slippage_bps'] is None)

def main():
    initialize_directories()
    parser = argparse.ArgumentParser(description="Quant Engine (Singularity Edition)")
    parser.add_argument('mode', choices=['matrix', 'backtest', 'gateway', 'stress', 'test'], help="运行模式")
    args = parser.parse_args()

    if args.mode == 'test':
        sys.argv = [sys.argv[0]]
        unittest.main(verbosity=2)
    elif args.mode == 'gateway': run_gateway()
    elif args.mode == 'matrix': run_matrix()
    elif args.mode == 'backtest': run_backtest()
    elif args.mode == 'stress': run_synthetic_stress_test()

if __name__ == "__main__":
    main()
