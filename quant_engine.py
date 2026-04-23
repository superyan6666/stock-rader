# 存储路径: quant_engine.py
import yfinance as yf
import requests
import os
import sys
import pandas as pd
import numpy as np
import time
import random
import logging
import re
import json
import warnings
import sqlite3     
import uuid        
import scipy.stats as stats
import concurrent.futures
import threading
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass, field

try:
    from quant_transformer import QuantAlphaTransformer, train_alpha_model
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

try:
    from scipy.stats import gaussian_kde
    KDE_AVAILABLE = True
except ImportError:
    KDE_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ================= 1. 日志与全局配置管理 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantBot")

_GLOBAL_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}

class Config:
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    DINGTALK_KEYWORD: str = "AI"
    
    INDEX_ETF: str = "QQQ" 
    VIX_INDEX: str = "^VIX" 
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'TXN', 'INTU', 'AMAT', 'MU', 'LRCX', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'NXPI', 'MRVL', 'MCHP', 'FTNT', 'CRWD'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD', 'ROST', 'EBAY', 'TSCO', 'PDD', 'DASH', 'CPRT', 'PCAR'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA', 'TTWO', 'WBD', 'SIRI', 'CHTR'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM', 'IDXX', 'MRNA', 'ALGN', 'BMRN', 'GEHC'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'MNST', 'WBA']
    }
    
    LOG_PREFIX: str = "backtest_log_"
    STATS_FILE: str = "strategy_stats.json"
    REPORT_FILE: str = "backtest_report.md"
    CACHE_FILE: str = "tickers_cache.json"
    ALERT_CACHE_FILE: str = "alert_history.json"
    MODEL_FILE: str = "scoring_model.pkl"
    WSB_CACHE_FILE: str = "wsb_history.json"  
    CUSTOM_CONFIG_FILE: str = "quantbot_config.json" 
    
    DATA_DIR: str = ".quantbot_data"
    EXT_CACHE_FILE: str = os.path.join(DATA_DIR, "daily_ext_cache.json")
    ORDER_DB_PATH: str = os.path.join(DATA_DIR, "order_state.db") 
    TCA_LOG_PATH: str = os.path.join(DATA_DIR, "tca_history.jsonl")
    MODEL_VERSION: str = "1.0" 

    CORE_WATCHLIST: list = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "QCOM",
        "JPM", "V", "MA", "UNH", "JNJ", "XOM", "PG", "HD", "COST", "ABBV",
        "CRM", "NFLX", "ADBE", "NOW", "UBER", "INTU", "IBM", "ISRG", "SYK",
        "SPY", "QQQ", "IWM", "DIA"
    ]
    CROWDING_EXCLUDE_SECTORS: list = [INDEX_ETF] 

    CORE_FACTORS = [
        "米奈尔维尼", "强相对强度", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)",
        "突破缺口", "VWAP突破", "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", "口袋支点", 
        "VCP收缩", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "跨时空共振(周线)"
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

        PCR_BEAR = 1.5           
        PCR_BULL = 0.5           
        IV_SKEW_BEAR = 0.08      
        IV_SKEW_BULL = -0.05     
        SHORT_SQZ_CHG = 0.10     
        SHORT_SQZ_FLT = 0.05     
        VRP_EXTREME = 0.25       
        WSB_ACCEL = 20.0         
        ANALYST_UP = 1.5         
        ANALYST_DN = -1.5        
        NLP_BULL = 0.5           
        NLP_BEAR = -0.5          
        INSIDER_BUY = 0.5        
        AMIHUD_ILLIQ = 0.5       
        DIST_52W = -0.05         
        
        KDE_BREAKOUT = 0.5       
        HURST_RELIABLE = 0.65    
        FFT_RESONANCE = 0.71     
        VCP_BULL = 0.5           
        VCP_BEAR = 0.4           
        MACD_WATERLINE = 0.0     

        @classmethod
        def load_overrides(cls):
            if os.path.exists(Config.CUSTOM_CONFIG_FILE):
                try:
                    with open(Config.CUSTOM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                        custom_cfg = json.load(f)
                        for k, v in custom_cfg.items():
                            if hasattr(cls, k):
                                orig_val = getattr(cls, k)
                                if isinstance(orig_val, bool):
                                    if isinstance(v, str): cast_v = v.lower() in ['true', '1', 'yes', 't']
                                    else: cast_v = bool(v)
                                    setattr(cls, k, cast_v)
                                elif isinstance(orig_val, (int, float, str)):
                                    try: setattr(cls, k, type(orig_val)(v))
                                    except (ValueError, TypeError): pass
                                else: setattr(cls, k, v)
                except Exception: pass

    @classmethod
    def get_current_log_file(cls) -> str:
        return f"{cls.LOG_PREFIX}{datetime.now(timezone.utc).strftime('%Y_%m')}.jsonl"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF

def datetime_to_str():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def validate_config():
    if not Config.WEBHOOK_URL and not Config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未配置任何推送渠道。")
        sys.exit(1)
    
    Config.Params.load_overrides() 
    
    curr_log = Config.get_current_log_file()
    if not os.path.exists(curr_log): open(curr_log, 'a').close()
    if not os.path.exists(Config.REPORT_FILE): open(Config.REPORT_FILE, 'a').close()
    if not os.path.exists(Config.STATS_FILE):
        with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: f.write("{}")
    if not os.path.exists(Config.ALERT_CACHE_FILE):
        with open(Config.ALERT_CACHE_FILE, 'w', encoding='utf-8') as f: 
            json.dump({"matrix": {}, "shadow_pool": {}}, f)
            
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR, exist_ok=True)
            
    logger.info("✅ 环境与占位文件校验通过")

# ================= 2. 线程安全缓存与数据抽象化模型 =================

_KLINE_CACHE = {}
_KLINE_LOCK = threading.Lock()

_SENTIMENT_CACHE = {} 
_SENTIMENT_LOCK = threading.Lock()

_ALT_DATA_CACHE = {}  
_ALT_DATA_LOCK = threading.Lock()

_DAILY_EXT_CACHE = {}
_DAILY_EXT_LOCK = threading.Lock()

def _init_ext_cache():
    global _DAILY_EXT_CACHE
    with _DAILY_EXT_LOCK:
        if not _DAILY_EXT_CACHE:
            if os.path.exists(Config.EXT_CACHE_FILE):
                try:
                    with open(Config.EXT_CACHE_FILE, 'r', encoding='utf-8') as f:
                        _DAILY_EXT_CACHE = json.load(f)
                except Exception: pass
            
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if _DAILY_EXT_CACHE.get("date") != today:
                _DAILY_EXT_CACHE = {"date": today, "sentiment": {}, "alt": {}}

def _save_ext_cache():
    with _DAILY_EXT_LOCK:
        try:
            temp_ext = f"{Config.EXT_CACHE_FILE}.{threading.get_ident()}.tmp"
            with open(temp_ext, 'w', encoding='utf-8') as f:
                json.dump(_DAILY_EXT_CACHE, f)
            os.replace(temp_ext, Config.EXT_CACHE_FILE)
        except Exception: 
            try:
                if os.path.exists(temp_ext): os.remove(temp_ext)
            except Exception: pass

@dataclass
class MarketContext:
    regime: str; regime_desc: str; w_mul: float; xai_weights: dict; vix_current: float
    vix_desc: str; vix_scalar: float; max_risk: float; macro_gravity: bool
    is_credit_risk_high: bool; vix_inv: bool; qqq_df: pd.DataFrame; macro_data: dict
    total_market_exposure: float; health_score: float; pain_warning: str
    credit_spread_mom: float = 0.0; vix_term_structure: float = 1.0; market_pcr: float = 1.0
    dynamic_min_score: float = 8.0; global_wsb_data: dict = field(default_factory=dict)
    meta_weights: dict = field(default_factory=dict); transformer_model: 'Any' = None  

@dataclass
class StockData:
    sym: str; df: pd.DataFrame; df_w: pd.DataFrame; df_m: pd.DataFrame
    df_60m: pd.DataFrame; curr: pd.Series; prev: pd.Series
    is_vol: bool; swing_high_10: float

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

def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    cache_key = f"{symbol}_{interval}"
    cache_file = os.path.join(Config.DATA_DIR, f"{cache_key}.pkl")
    archive_dir = os.path.join(Config.DATA_DIR, "archive", symbol)
    
    with _KLINE_LOCK:
        if cache_key in _KLINE_CACHE:
            return _KLINE_CACHE[cache_key].copy()
            
    df = pd.DataFrame()
    try:
        if os.path.exists(cache_file):
            df = pd.read_pickle(cache_file)
    except Exception: pass
    
    now_utc = datetime.now(timezone.utc)
    download_period = period
    needs_full = True
    
    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
        elif df.index.tzinfo != timezone.utc: df.index = df.index.tz_convert('UTC')
            
        last_date = df.index[-1]
        days_diff = (now_utc - last_date).days
        
        min_required_len = 200 if period in ["1y", "2y", "5y"] else 0
        if len(df) >= min_required_len:
            if days_diff < 1: download_period, needs_full = "5d", False
            elif days_diff < 30: download_period, needs_full = "1mo", False
            elif days_diff < 90: download_period, needs_full = "3mo", False
        
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(0.1, 0.3) if fast_mode else random.uniform(1.0, 2.0)
            time.sleep(sleep_sec)
            
            new_df = yf.Ticker(symbol).history(period=download_period, interval=interval, auto_adjust=auto_adjust, timeout=10)
            
            if not new_df.empty:
                new_df.index = pd.to_datetime(new_df.index, utc=True)
                
                if not df.empty and not needs_full:
                    overlap = df.index.intersection(new_df.index)
                    if len(overlap) > 0:
                        old_close, new_close = df.loc[overlap[-1], 'Close'], new_df.loc[overlap[-1], 'Close']
                        if abs(old_close - new_close) / (old_close + 1e-10) > 0.05:
                            new_df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=auto_adjust, timeout=10)
                            new_df.index = pd.to_datetime(new_df.index, utc=True)
                            df = new_df
                        else:
                            df = pd.concat([df[~df.index.isin(new_df.index)], new_df]).sort_index()
                    else:
                        df = pd.concat([df, new_df]).sort_index()
                else:
                    df = new_df
                    
                df = df[~df.index.duplicated(keep='last')]
                
                hot_cache_limit = 750
                if len(df) > hot_cache_limit:
                    df_to_archive = df.iloc[:-hot_cache_limit]
                    try:
                        os.makedirs(archive_dir, exist_ok=True)
                        for (year, month), group in df_to_archive.groupby([df_to_archive.index.year, df_to_archive.index.month]):
                            arch_file = os.path.join(archive_dir, f"{year}_{month:02d}_{interval}.pkl")
                            if os.path.exists(arch_file):
                                combined_arch = pd.concat([pd.read_pickle(arch_file), group])
                                combined_arch = combined_arch[~combined_arch.index.duplicated(keep='last')]
                            else: combined_arch = group
                            tmp_arch = f"{arch_file}.{threading.get_ident()}.tmp"
                            combined_arch.to_pickle(tmp_arch)
                            os.replace(tmp_arch, arch_file)
                    except Exception: pass
                    df = df.iloc[-hot_cache_limit:] 
                
                with _KLINE_LOCK: _KLINE_CACHE[cache_key] = df.copy()
                try:
                    temp_cache_file = f"{cache_file}.{threading.get_ident()}.tmp"
                    df.to_pickle(temp_cache_file)
                    os.replace(temp_cache_file, cache_file)
                except Exception: pass
                return df
        except Exception as e:
            if attempt == retries - 1: return df
            time.sleep((10 + attempt * 5) if "429" in str(e).lower() else (2 + attempt * 2))
            
    return df

def fetch_global_wsb_data() -> Dict[str, float]:
    with _ALT_DATA_LOCK:
        if "WSB_ACCEL_GLOBAL" in _ALT_DATA_CACHE: return _ALT_DATA_CACHE["WSB_ACCEL_GLOBAL"]
        
    cache_file = Config.WSB_CACHE_FILE
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    history = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f: history = json.load(f)
        except Exception: pass
        
    current_data = {}
    for attempt in range(3):
        try:
            url = "https://tradestie.com/api/v1/apps/reddit"
            response = requests.get(url, headers=_GLOBAL_HEADERS, timeout=10)
            if response.status_code == 200:
                for item in response.json():
                    tk = item.get('ticker')
                    if tk and item.get('sentiment') == 'Bullish': 
                        current_data[tk] = item.get('no_of_comments', 0)
                break
        except Exception:
            time.sleep(2.0)
            
    if current_data: history[today_str] = current_data
        
    sorted_dates = sorted(history.keys())
    if len(sorted_dates) > 5:
        for d in sorted_dates[:-5]: del history[d]
        
    try:
        temp_wsb = f"{cache_file}.{threading.get_ident()}.tmp"
        with open(temp_wsb, "w", encoding="utf-8") as f: json.dump(history, f)
        os.replace(temp_wsb, cache_file)
    except Exception: pass
    
    wsb_accel_dict = {}
    dates = sorted(history.keys())
    if len(dates) >= 3:
        d0, d1, d2 = dates[-1], dates[-2], dates[-3]
        for tk in history[d0].keys():
            v0, v1, v2 = history[d0].get(tk, 0), history[d1].get(tk, 0), history[d2].get(tk, 0)
            wsb_accel_dict[tk] = float((v0 - v1) - (v1 - v2))
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
        
    insider_net_buy, analyst_mom, nlp_score, news_summary = 0.0, 0.0, 0.0, ""
    
    try:
        tk = yf.Ticker(symbol)
        try:
            insiders = tk.insider_transactions
            if insiders is not None and not insiders.empty:
                recent_insiders = insiders.head(20) 
                buys, sells = 0, 0
                if 'Shares' in recent_insiders.columns:
                    if 'Text' in recent_insiders.columns:
                        buys = recent_insiders[recent_insiders['Text'].str.contains('Buy|Purchase', case=False, na=False)]['Shares'].sum()
                        sells = recent_insiders[recent_insiders['Text'].str.contains('Sell|Sale', case=False, na=False)]['Shares'].abs().sum()
                    else:
                        buys = recent_insiders[recent_insiders['Shares'] > 0]['Shares'].sum()
                        sells = recent_insiders[recent_insiders['Shares'] < 0]['Shares'].abs().sum()
                if (buys + sells) > 0: insider_net_buy = (buys - sells) / (buys + sells)
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
                        current_score = monthly_scores.iloc[-1]
                        hist_scores = monthly_scores.iloc[:-1]
                        analyst_mom = (current_score - hist_scores.mean()) / (hist_scores.std() + 1e-5)
                    else: analyst_mom = recent_1y['score'].sum() * 0.1 
        except Exception: pass
    except Exception: pass

    with _DAILY_EXT_LOCK: _DAILY_EXT_CACHE["alt"][symbol] = (insider_net_buy, analyst_mom, nlp_score, news_summary)
    _save_ext_cache()
    return insider_net_buy, analyst_mom, nlp_score, news_summary

def check_earnings_risk(symbol: str) -> bool:
    try:
        tk = yf.Ticker(symbol)
        now_date = datetime.now(timezone.utc).date()
        try:
            cal = tk.calendar
            if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
                ed = cal['Earnings Date']
                if ed and hasattr(ed[0], 'date') and 0 <= (ed[0].date() - now_date).days <= 5: return True
        except Exception: pass
        try:
            ed_df = tk.earnings_dates
            if ed_df is not None and not ed_df.empty:
                for d in ed_df.index:
                    if hasattr(d, 'date') and d.date() >= now_date and 0 <= (d.date() - now_date).days <= 5: return True
        except Exception: pass
    except Exception: pass
    return False

def get_filtered_watchlist(max_stocks: int = 150) -> list:
    return list(Config.CORE_WATCHLIST)[:max_stocks]

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

def get_alert_cache() -> dict:
    cache_data = {"matrix": {}, "shadow_pool": {}}
    try:
        if os.path.exists(Config.ALERT_CACHE_FILE):
            with open(Config.ALERT_CACHE_FILE, 'r', encoding='utf-8') as f:
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
        if shadow_data: shadow_data['_ts'] = now_ts; shadow_pool[sym] = shadow_data
        else: shadow_pool[sym] = {"_ts": now_ts}
    try:
        temp_file = f"{Config.ALERT_CACHE_FILE}.{threading.get_ident()}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f: json.dump(cache, f)
        os.replace(temp_file, Config.ALERT_CACHE_FILE)
    except Exception: pass

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    req_headers = _GLOBAL_HEADERS.copy()
    req_headers["Content-Type"] = "application/json"
    
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": f"【{Config.DINGTALK_KEYWORD}】{title}", "text": f"## 🤖 【{Config.DINGTALK_KEYWORD}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            threading.Thread(target=lambda u, p: requests.post(u, json=p, headers=req_headers, timeout=10) if True else None, args=(url, payload)).start()
                
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        html_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = re.sub(r'### (.*?)\n', r'<b>\1</b>\n', html_content)
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_content)
        html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
        html_content = html_content.replace('\n---', '\n━━━━━━━━━━━━━━━━━━')
        
        tg_text = f"🤖 <b>【量化监控】{html_title}</b>\n\n{html_content}\n\n⏱️ <i>{formatted_time}</i>"
        threading.Thread(
            target=lambda tk, chat, txt: requests.post(f"https://api.telegram.org/bot{tk}/sendMessage", json={"chat_id": chat, "text": txt[:4000], "parse_mode": "HTML", "disable_web_page_preview": True}, headers=req_headers, timeout=10) if True else None, 
            args=(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID, tg_text)
        ).start()

# ================= 4. 大盘感知与特征工程模块 =================
def get_vix_level(qqq_df_for_shadow: pd.DataFrame = None) -> Tuple[float, str]:
    df = safe_get_history(Config.VIX_INDEX, period="5d", interval="1d", retries=3, auto_adjust=False, fast_mode=True)
    vix, is_simulated = 18.0, False
    if not df.empty and len(df) >= 1: vix = df['Close'].ffill().iloc[-1]
    else:
        if qqq_df_for_shadow is not None and len(qqq_df_for_shadow) >= 20:
            realized_vol = qqq_df_for_shadow['Close'].pct_change().dropna().ewm(span=20).std().iloc[-1] * (252 ** 0.5) * 100
            vix, is_simulated = min(max(realized_vol * 1.15 + 4.0, 9.0), 85.0), True
            
    prefix = "影子VIX" if is_simulated else "VIX"
    if vix > 30: return vix, f"🚨 极其恐慌 ({prefix}: {vix:.2f})"
    if vix > 25: return vix, f"⚠️ 市场恐慌 ({prefix}: {vix:.2f})"
    if vix < 15: return vix, f"✅ 市场平静 ({prefix}: {vix:.2f})"
    return vix, f"⚖️ 正常波动 ({prefix}: {vix:.2f})"

def get_market_regime(active_pool: List[str] = None) -> Tuple[str, str, pd.DataFrame, bool, bool]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", auto_adjust=False, fast_mode=True)
    if len(df) < 200: return "range", "数据不足，默认震荡", df, False, False
    
    c_close = df['Close'].ffill().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    ma50_curr = df['Close'].rolling(50).mean().iloc[-1]
    ma50_prev = df['Close'].rolling(50).mean().iloc[-20]
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    credit_risk_alert, credit_desc = False, ""
    try:
        hyg = safe_get_history("HYG", "3mo", "1d", fast_mode=True)
        ief = safe_get_history("IEF", "3mo", "1d", fast_mode=True)
        if not hyg.empty and not ief.empty:
            ratio = hyg['Close'] / ief['Close']
            if ratio.iloc[-1] < ratio.rolling(20).mean().iloc[-1] and ratio.iloc[-1] < ratio.iloc[-10]:
                credit_risk_alert, credit_desc = True, "\n- 🚨 **宏观信用风控**: 高收益债流出避险，市场警报！"
    except Exception: pass
    
    macro_gravity, gravity_desc = False, ""
    try:
        dxy = safe_get_history("DX-Y.NYB", "1mo", "1d", fast_mode=True)
        tnx = safe_get_history("^TNX", "1mo", "1d", fast_mode=True)
        if not dxy.empty and not tnx.empty:
            if (dxy['Close'].iloc[-1] / dxy['Close'].iloc[-10] - 1) > 0.015 and (tnx['Close'].iloc[-1] / tnx['Close'].iloc[-10] - 1) > 0.04:
                macro_gravity, gravity_desc = True, "\n- 🌑 **宏观引力波**: 美元与美债收益率双飙，流动性黑洞来袭，系统极度承压！"
    except Exception: pass
    
    breadth_desc = ""
    if c_close > ma200:
        if trend_20d > 0.02: return "bull", f"🐂 牛市主升阶段{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        else: return "range", f"⚖️ 牛市高位震荡{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
    else:
        if c_close > ma50_curr and ma50_curr > ma50_prev and trend_20d > 0.04:
            return "rebound", f"🦅 熊市超跌反弹 (V反){breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        elif trend_20d < -0.02: return "bear", f"🐻 熊市回调阶段{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        else: return "range", f"⚖️ 熊市底部震荡{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity

def _robust_fft_ensemble(close_prices: np.ndarray, base_length=120, ensemble_count=7) -> float:
    if len(close_prices) < base_length + (ensemble_count // 2) * 5: return 0.0
    votes = []
    for offset in range(ensemble_count):
        win_len = base_length + (offset - ensemble_count//2) * 5
        segment = close_prices[-win_len:]
        if np.isnan(segment).any() or np.all(segment == segment[0]): continue
        fft_res = np.fft.fft(segment - np.mean(segment))
        freqs = np.fft.fftfreq(win_len)
        pos_mask = (freqs > 0.01) & (freqs < 0.2)
        if not np.any(pos_mask): continue
        pos_idx = np.where(pos_mask)[0]
        peak_idx = pos_idx[np.argmax(np.abs(fft_res[pos_idx]))]
        phase = np.angle(fft_res[peak_idx])
        t = win_len - 1
        dom_freq = freqs[peak_idx]
        current_phase = (2 * np.pi * dom_freq * t + phase) % (2 * np.pi)
        if 0 < current_phase < np.pi: votes.append(1.0)
        else: votes.append(-1.0)
    if not votes: return 0.0
    return float(sum(votes) / len(votes))

def _robust_hurst(close_prices: np.ndarray, min_window=30, n_bootstrap=100) -> Tuple[float, float, bool]:
    safe_prices = np.maximum(close_prices, 1e-10)
    log_ret = np.diff(np.log(safe_prices[-121:])) if len(safe_prices) > 120 else np.diff(np.log(safe_prices))
    if len(log_ret) <= min_window: return 0.5, 0.0, False
        
    np.random.seed(42)
    hurst_samples = []
    for _ in range(n_bootstrap):
        sub_len = np.random.randint(min_window, len(log_ret))
        start = np.random.randint(0, len(log_ret) - sub_len)
        sub_ret = log_ret[start:start+sub_len]
        S = np.std(sub_ret)
        if S == 0: continue
        R = np.max(np.cumsum(sub_ret - np.mean(sub_ret))) - np.min(np.cumsum(sub_ret - np.mean(sub_ret)))
        if R <= 0: continue
        hurst_samples.append(np.log(R/S) / np.log(len(sub_ret)))
        
    if len(hurst_samples) < 20: return 0.5, 0.0, False
    h_median, h_iqr = float(np.median(hurst_samples)), float(np.percentile(hurst_samples, 75) - np.percentile(hurst_samples, 25))
    is_reliable = bool((h_iqr < 0.15) and (abs(h_median - 0.5) > 0.1))
    return h_median, h_iqr, is_reliable

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['Open'], df['High'], df['Low'] = df['Open'].ffill(), df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'], df['SMA_150'], df['SMA_200'] = df['Close'].rolling(50).mean(), df['Close'].rolling(150).mean(), df['Close'].rolling(200).mean()
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    up, down = delta.where(delta > 0, 0.0), -delta.where(delta < 0, 0.0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean() + 1e-10)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    atr20 = df['TR'].rolling(window=20).mean()
    df['KC_Upper'], df['KC_Lower'] = df['EMA_20'] + 1.5 * atr20, df['EMA_20'] - 1.5 * atr20
    bb_ma, bb_std = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Upper'], df['BB_Lower'] = bb_ma + 2 * bb_std, bb_ma - 2 * bb_std
    
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    
    df['SuperTrend_Up'] = (df['Close'] > df['EMA_20']).astype(int) 
    
    hl_diff = np.maximum(df['High'].values - df['Low'].values, 1e-10)
    dollar_vol = df['Close'].values * df['Volume'].values
    clv_num = (df['Close'].values - df['Low'].values) - (df['High'].values - df['Close'].values)
    df['CMF'] = pd.Series((clv_num / hl_diff * dollar_vol)).rolling(20).sum().values / (pd.Series(dollar_vol).rolling(20).sum().values + 1e-10)
    df['CMF'] = df['CMF'].clip(lower=-1.0, upper=1.0).fillna(0.0)

    df['Range'] = df['High'] - df['Low']
    df['NR7'] = (df['Range'] <= df['Range'].rolling(7).min())
    df['Inside_Bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))

    close_shift = np.roll(df['Close'].values, 1)
    close_shift[0] = np.nan
    vpt_base = np.where((close_shift == 0) | np.isnan(close_shift), 0.0, (df['Close'].values - close_shift) / close_shift)
    df['VPT_Cum'] = (vpt_base * df['Volume'].values).cumsum()
    vpt_ma50, vpt_std50 = df['VPT_Cum'].rolling(50).mean().values, df['VPT_Cum'].rolling(50).std().values
    df['VPT_ZScore'] = (df['VPT_Cum'].values - vpt_ma50) / np.where((np.isnan(vpt_std50)) | (vpt_std50 == 0), 1e-6, vpt_std50)
    df['VPT_Accel'] = np.gradient(np.nan_to_num(df['VPT_ZScore']))

    df['VWAP_20'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)
    
    is_new_anchor = df['Volume'] >= df['Volume'].rolling(window=120, min_periods=1).max()
    anchor_groups = is_new_anchor.cumsum() 
    df['AVWAP'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).groupby(anchor_groups).cumsum() / (df['Volume'].groupby(anchor_groups).cumsum() + 1e-10)
    
    df['Max_Down_Vol_10'] = df['Volume'].where(df['Close'] < df['Close'].shift(), 0).shift(1).rolling(10).max()
    surge = (df['Close'] > df['Close'].shift(1) * 1.04) & (df['Volume'] > df['Volume'].rolling(20).mean())
    df['OB_High'] = df['High'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['OB_Low'] = df['Low'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Range_60'] = df['High'].rolling(60).max() - df['Low'].rolling(60).min()
    df['Range_20'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Price_High_20'] = df['High'].rolling(20).max()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    df['CVD_Smooth'] = ((df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)).cumsum()).ewm(span=5).mean()
    cvd_ma10, cvd_ma30 = df['CVD_Smooth'].rolling(10).mean(), df['CVD_Smooth'].rolling(30).mean()
    df['CVD_Trend'] = np.where(cvd_ma10 > cvd_ma30, 1.0, np.where(cvd_ma10 < cvd_ma30, -1.0, 0.0))
    df['CVD_Divergence'] = ((df['Close'] >= df['Price_High_20'] * 0.99) & (df['CVD_Smooth'] < df['CVD_Smooth'].rolling(20).max() * 0.95)).astype(int)
    
    df['Highest_22'] = df['High'].rolling(window=22).max()
    df['ATR_22'] = df['TR'].rolling(window=22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    df['Smart_Money_Flow'] = (clv_num / hl_diff).rolling(window=10).mean()

    df['Recent_Price_Surge_3d'] = (df['Close'] / df['Open'] - 1).rolling(3).max().shift(1) * 100
    df['Recent_Vol_Surge_3d'] = (df['Volume'] / (df['Vol_MA20']+1)).rolling(3).max().shift(1)
    df['Amihud'] = (df['Close'].pct_change().abs() / (df['Close'] * df['Volume'] + 1e-10)).rolling(20).mean() * 1e6
    high_52w = df['High'].rolling(252, min_periods=20).max()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-10)

    return df

def _get_transformer_seq(df_ind: pd.DataFrame, end_idx: int = -1) -> np.ndarray:
    if end_idx == -1: end_idx = len(df_ind)
    start_idx = end_idx - 60
    if start_idx < 0: return np.zeros((60, 49), dtype=np.float32)
    df = df_ind.iloc[start_idx:end_idx].copy()
    c = df['Close'].values + 1e-10
    seq = np.zeros((60, 49), dtype=np.float32)
    feature_mapping = [
        (0, df['RSI'].values / 100.0), (1, df['MACD'].values / c), (2, df['Signal_Line'].values / c), (3, df['ATR'].values / c),
        (4, (df['KC_Upper'].values - df['KC_Lower'].values) / c), (5, (df['BB_Upper'].values - df['BB_Lower'].values) / c),
        (6, df['Tenkan'].values / c - 1.0), (7, df['Kijun'].values / c - 1.0), (8, df['SenkouA'].values / c - 1.0),
        (9, df['SenkouB'].values / c - 1.0), (10, df['SuperTrend_Up'].values), (11, df['CMF'].values),
        (12, df['Range'].values / c), (13, df['NR7'].astype(float).values), (14, df['Inside_Bar'].astype(float).values),
        (15, df['VPT_ZScore'].values), (16, df['VPT_Accel'].values), (17, df['VWAP_20'].values / c - 1.0),
        (18, np.where(df['AVWAP'].isna(), 0, df['AVWAP'].values / c - 1.0)), (19, df['Volume'].values / (df['Max_Down_Vol_10'].values + 1e-10)),
        (20, np.where(df['OB_High'].isna(), 0, df['OB_High'].values / c - 1.0)), (21, np.where(df['OB_Low'].isna(), 0, df['OB_Low'].values / c - 1.0)),
        (22, np.where(df['Swing_Low_20'].isna(), 0, df['Swing_Low_20'].values / c - 1.0)), (23, df['Range_60'].values / c),
        (24, df['Range_20'].values / c), (25, df['Price_High_20'].values / c - 1.0), (26, df['Volume'].values / (df['Vol_MA20'].values + 1e-10)),
        (27, df['CVD_Trend'].values), (28, df['CVD_Divergence'].values), (29, df['Highest_22'].values / c - 1.0),
        (30, df['ATR_22'].values / c), (31, df['Chandelier_Exit'].values / c - 1.0), (32, df['Smart_Money_Flow'].values),
        (33, df['Recent_Price_Surge_3d'].values / 100.0), (34, df['Recent_Vol_Surge_3d'].values), (35, df['Amihud'].values),
        (36, np.where(df['Dist_52W_High'].isna(), 0, df['Dist_52W_High'].values)), (37, df['Close'].values / (df['SMA_50'].values + 1e-10) - 1.0),
        (38, df['Close'].values / (df['SMA_150'].values + 1e-10) - 1.0), (39, df['Close'].values / (df['SMA_200'].values + 1e-10) - 1.0),
        (40, df['Close'].values / (df['EMA_20'].values + 1e-10) - 1.0), (41, df['Close'].values / (df['EMA_50'].values + 1e-10) - 1.0),
        (42, df['Above_Cloud'].values), (43, df['Open'].values / c - 1.0), (44, df['High'].values / c - 1.0),
        (45, df['Low'].values / c - 1.0), (46, df['Close'].pct_change().fillna(0).values), (47, df['Volume'].pct_change().fillna(0).values),
        (48, (df['Close'].values - df['Open'].values) / (df['High'].values - df['Low'].values + 1e-10))
    ]
    for idx, vals in feature_mapping: seq[:, idx] = vals
    return np.nan_to_num(seq, nan=0.0, posinf=5.0, neginf=-5.0)

def _extract_complex_features(stock: StockData, ctx: MarketContext) -> ComplexFeatures:
    weekly_bullish, weekly_macd_res, fvg_lower, fvg_upper = False, 0.0, 0.0, 0.0
    aligned_w = stock.df_w
    if not aligned_w.empty and len(aligned_w) >= 40:
        if aligned_w['Close'].iloc[-1] > aligned_w['Close'].rolling(40).mean().iloc[-1]:
            ema10, ema30 = aligned_w['Close'].ewm(span=10, adjust=False).mean(), aligned_w['Close'].ewm(span=30, adjust=False).mean()
            weekly_bullish = (aligned_w['Close'].iloc[-1] > ema10.iloc[-1]) and (ema10.iloc[-1] > ema30.iloc[-1])
        macd_w = aligned_w['Close'].ewm(span=12, adjust=False).mean() - aligned_w['Close'].ewm(span=26, adjust=False).mean()
        hist_w = macd_w - macd_w.ewm(span=9, adjust=False).mean()
        if len(hist_w) >= 2 and hist_w.iloc[-1] > 0 and hist_w.iloc[-1] > hist_w.iloc[-2]: weekly_macd_res = 1.0
            
    n = len(stock.df)
    if n >= 22:
        lows, highs = stock.df['Low'].values, stock.df['High'].values
        valid_idx = np.where(lows[n-20:n-1] > highs[n-22:n-3])[0]
        if len(valid_idx) > 0:
            last_i = valid_idx[-1] + n - 20
            fvg_lower, fvg_upper = highs[last_i-2], lows[last_i]
    
    kde_breakout_score = 0.0
    if KDE_AVAILABLE and len(stock.df) >= 60:
        try:
            prices, volumes = stock.df['Close'].iloc[-60:].values, stock.df['Volume'].iloc[-60:].values
            if np.std(prices) > 1e-5:
                kde = gaussian_kde(prices, weights=volumes, bw_method='silverman')
                den_curr = kde.evaluate(prices[-1])[0]
                densities = kde.evaluate(np.linspace(prices.min(), prices.max(), 200))
                den_50 = np.percentile(densities, 50)
                if den_curr <= den_50:
                    kde_breakout_score = min(1.0, 0.5 + 0.5 * (den_50 - den_curr) / (den_50 + 1e-10))
        except Exception: pass
        
    fft_ensemble_score = _robust_fft_ensemble(stock.df['Close'].values, base_length=120, ensemble_count=7)
    hurst_med, hurst_iqr, hurst_reliable = _robust_hurst(stock.df['Close'].values)
    
    monthly_inst_flow = 0.0
    if not stock.df_m.empty and len(stock.df_m) >= 3:
        m_flow = (stock.df_m['Close'] - stock.df_m['Open']) / (stock.df_m['High'] - stock.df_m['Low'] + 1e-10) * stock.df_m['Volume']
        if m_flow.iloc[-1] > 0 and m_flow.iloc[-2] > 0 and m_flow.iloc[-3] > 0: monthly_inst_flow = 1.0

    rsi_60m_bounce = 0.0
    if not stock.df_60m.empty and len(stock.df_60m) >= 15:
        delta = stock.df_60m['Close'].diff()
        rs = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean() / (-delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean() + 1e-10)
        rsi_60m = 100 - (100 / (1 + rs))
        if len(rsi_60m) >= 2 and rsi_60m.iloc[-1] > rsi_60m.iloc[-2] and rsi_60m.iloc[-2] < 40: rsi_60m_bounce = 1.0
            
    stock_ret = stock.df['Close'].pct_change().fillna(0)
    beta_60d, spy_realized_vol = 1.0, 15.0 
    if 'spy' in ctx.macro_data and not ctx.macro_data['spy'].empty and len(stock_ret) >= 60:
        spy_ret = ctx.macro_data['spy']['Close'].pct_change().reindex(stock_ret.index).fillna(0)
        cov_mat = np.cov(stock_ret.iloc[-60:], spy_ret.iloc[-60:])
        if cov_mat[1,1] > 0: beta_60d = cov_mat[0, 1] / cov_mat[1, 1]
        spy_realized_vol = spy_ret.iloc[-20:].std() * np.sqrt(252) * 100
        
    tlt_corr, dxy_corr = 0.0, 0.0
    if 'tlt' in ctx.macro_data and not ctx.macro_data['tlt'].empty and len(stock_ret) >= 90:
        tlt_ret = ctx.macro_data['tlt']['Close'].pct_change().reindex(stock_ret.index).fillna(0)
        tlt_corr = stock_ret.iloc[-90:].corr(tlt_ret.iloc[-90:])
    if 'dxy' in ctx.macro_data and not ctx.macro_data['dxy'].empty and len(stock_ret) >= 20:
        dxy_ret = ctx.macro_data['dxy']['Close'].pct_change().reindex(stock_ret.index).fillna(0)
        dxy_corr = stock_ret.iloc[-20:].corr(dxy_ret.iloc[-20:])
        
    rs_20, pure_alpha = 0.0, 0.0
    if not ctx.qqq_df.empty:
        m_df = pd.merge(stock.df[['Close']], ctx.qqq_df[['Close']], left_index=True, right_index=True, how='inner')
        if len(m_df) >= 60:
            qqq_ret = max(m_df['Close_y'].iloc[-1] / m_df['Close_y'].iloc[-20], 0.5)
            rs_20 = (m_df['Close_x'].iloc[-1] / m_df['Close_x'].iloc[-20]) / qqq_ret
            ret_stock, ret_qqq = m_df['Close_x'].pct_change().dropna(), m_df['Close_y'].pct_change().dropna()
            cov_matrix = np.cov(ret_stock.iloc[-60:], ret_qqq.iloc[-60:])
            beta_local = cov_matrix[0,1] / (cov_matrix[1,1] + 1e-10) if cov_matrix[1,1] > 0 else 1.0
            pure_alpha = (ret_stock.iloc[-5:].mean() - beta_local * ret_qqq.iloc[-5:].mean()) * 252
        
    return ComplexFeatures(
        weekly_bullish=weekly_bullish, fvg_lower=fvg_lower, fvg_upper=fvg_upper, 
        kde_breakout_score=kde_breakout_score, fft_ensemble_score=fft_ensemble_score, 
        hurst_med=hurst_med, hurst_iqr=hurst_iqr, hurst_reliable=hurst_reliable, 
        monthly_inst_flow=monthly_inst_flow, weekly_macd_res=weekly_macd_res, 
        rsi_60m_bounce=rsi_60m_bounce, beta_60d=float(beta_60d), tlt_corr=float(tlt_corr if pd.notna(tlt_corr) else 0), 
        dxy_corr=float(dxy_corr if pd.notna(dxy_corr) else 0), vrp=float((ctx.vix_current - spy_realized_vol) / max(ctx.vix_current, 1.0)), 
        rs_20=float(rs_20), pure_alpha=float(pure_alpha)
    )

def _extract_ml_features(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData, alpha_vec: np.ndarray = None) -> dict:
    macd_cross_strength = safe_div(stock.curr['MACD'] - stock.curr['Signal_Line'], abs(stock.curr['Close']) * 0.01)
    vol_surge_ratio = safe_div(stock.curr['Volume'], stock.curr['Vol_MA20'], cap=50.0)
    cmf_val, smf_val = stock.curr['CMF'], stock.curr['Smart_Money_Flow']
    hurst_score = max(0.0, min(1.0, (cf.hurst_med - 0.5) * 2.0)) if (cf.hurst_reliable and cf.hurst_med > Config.Params.HURST_RELIABLE) else 0.0

    feat_dict = {
        "米奈尔维尼": safe_div(stock.curr['SMA_50'] - stock.curr['SMA_200'], stock.curr['SMA_200']),
        "强相对强度": cf.rs_20, "MACD金叉": macd_cross_strength,
        "TTM Squeeze ON": safe_div((stock.curr['KC_Upper'] - stock.curr['KC_Lower']) - (stock.curr['BB_Upper'] - stock.curr['BB_Lower']), stock.curr['ATR'] + 1e-10),
        "一目多头": safe_div(stock.curr['Close'] - max(stock.curr['SenkouA'], stock.curr['SenkouB']), stock.curr['Close'] * 0.01),
        "强势回踩": safe_div(stock.curr['Close'] - stock.curr['EMA_20'], stock.curr['EMA_20'] * 0.01),
        "机构控盘(CMF)": cmf_val, "突破缺口": safe_div(stock.curr['Open'] - stock.prev['Close'], stock.prev['Close'] * 0.01),
        "VWAP突破": safe_div(stock.curr['Close'] - stock.curr['VWAP_20'], stock.curr['VWAP_20'] * 0.01),
        "AVWAP突破": safe_div(stock.curr['Close'] - stock.curr['AVWAP'], stock.curr['AVWAP'] * 0.01) if pd.notna(stock.curr['AVWAP']) else 0.0,
        "SMC失衡区": safe_div(stock.curr['Close'] - cf.fvg_lower, stock.curr['Close'] * 0.01) if cf.fvg_lower > 0 else 0.0,
        "流动性扫盘": safe_div(stock.curr['Swing_Low_20'] - stock.curr['Low'], stock.curr['Low'] * 0.01) if pd.notna(stock.curr['Swing_Low_20']) else 0.0,
        "聪明钱抢筹": smf_val, "巨量滞涨": vol_surge_ratio, "放量长阳": safe_div(stock.curr['Close'] - stock.curr['Open'], stock.curr['Open'] * 0.01),
        "口袋支点": safe_div(stock.curr['Volume'], stock.curr['Max_Down_Vol_10'], cap=50.0),
        "VCP收缩": safe_div(stock.curr['Range_20'], stock.curr['Range_60'] + 1e-10),
        "量子概率云(KDE)": cf.kde_breakout_score, "特性改变(ChoCh)": safe_div(stock.curr['Close'] - stock.swing_high_10, stock.swing_high_10 * 0.01),
        "订单块(OB)": safe_div(stock.curr['Close'] - stock.curr['OB_Low'], stock.curr['OB_High'] - stock.curr['OB_Low'] + 1e-10) if pd.notna(stock.curr['OB_High']) else 0.0,
        "AMD操盘": safe_div(min(stock.curr['Open'], stock.curr['Close']) - stock.curr['Low'], stock.curr['TR'] + 1e-10),
        "跨时空共振(周线)": 1.0 if cf.weekly_bullish else 0.0, "CVD筹码净流入": float(stock.curr['CVD_Trend'] * (0.3 if stock.curr['CVD_Divergence'] == 1 else 1.0)),
        "独立Alpha(脱钩)": cf.pure_alpha, "NR7极窄突破": safe_div(stock.curr['Range'], stock.curr['ATR'] + 1e-10),
        "VPT量价共振": (1.0 / (1.0 + np.exp(-stock.curr['VPT_ZScore']))) if stock.curr['VPT_Accel'] > 0 else 0.0,
        "带量金叉(交互)": macd_cross_strength * vol_surge_ratio, "量价吸筹(交互)": cmf_val * smf_val,
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
    else:
        for i in range(1, 17): feat_dict[f"Alpha_T{i:02d}"] = 0.0
    return {f: float(np.nan_to_num(feat_dict.get(f, 0.0), nan=0.0, posinf=20.0, neginf=-20.0)) for f in Config.ALL_FACTORS}

def _evaluate_omni_matrix(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData) -> Tuple[int, List[str], List[str], bool]:
    triggered_list, factors_list = [], []
    theme_scores = {'TREND': 0.0, 'VOLATILITY': 0.0, 'REVERSAL': 0.0, 'QUANTUM': 0.0}
    black_swan_risk = False
    
    def get_fw(tag_name: str) -> float: return ctx.xai_weights.get(tag_name, 1.0)
    def add_trigger(tag, text, pts, theme):
        fw = get_fw(tag)
        if fw > 0:
            adj_pts = pts * ctx.w_mul * fw
            if ctx.regime in ["bear", "hidden_bear"]:
                if theme in ["TREND", "VOLATILITY"]: adj_pts *= 0.6  
                elif theme == "REVERSAL": adj_pts *= 1.4
            elif ctx.regime in ["bull", "rebound"]:
                if theme in ["TREND", "VOLATILITY", "QUANTUM"]: adj_pts *= 1.2
            theme_scores[theme] += adj_pts
            triggered_list.append(text.format(fw=fw)); factors_list.append(tag)

    gap_pct = (stock.curr['Open'] - stock.prev['Close']) / (stock.prev['Close'] + 1e-10)
    atr_pct = (stock.curr['ATR'] / (stock.prev['Close'] + 1e-10)) * 100
    day_chg = (stock.curr['Close'] - stock.curr['Open']) / (stock.curr['Open'] + 1e-10) * 100
    tr_val = stock.curr['High'] - stock.curr['Low'] + 1e-10

    if pd.notna(stock.curr['SMA_200']) and stock.curr['Close'] > stock.curr['SMA_50'] > stock.curr['SMA_150'] > stock.curr['SMA_200']:
        m_str = (stock.curr['SMA_50'] - stock.curr['SMA_200']) / (stock.curr['SMA_200'] + 1e-10)
        add_trigger("米奈尔维尼", f"🏆 [主升趋势] 米奈尔维尼模板形成 (强度:{m_str*100:.1f}% 权:{{fw:.2f}}x)", 8 + int(m_str*20), "TREND")
        
    if cf.rs_20 > 0: 
        dynamic_rs_thresh = 1.0 + (stock.curr['ATR'] / (stock.curr['Close'] + 1e-10)) * 2.0
        if cf.rs_20 > dynamic_rs_thresh: add_trigger("强相对强度", f"⚡ [相对强度] 动能超越波动率动态阈值 (阈值:{dynamic_rs_thresh:.2f} 权:{{fw:.2f}}x)", 7 if stock.is_vol else 4, "TREND")
    
    macd_crossed = check_macd_cross(stock.curr, stock.prev)
    if macd_crossed:
        is_above_water = stock.curr['MACD'] > Config.Params.MACD_WATERLINE
        add_trigger("MACD金叉", f"🔥 [经典动能] MACD{'水上金叉' if is_above_water else '水下金叉'}起爆 (权:{{fw:.2f}}x)", 12 if is_above_water else 8, "TREND")

    if stock.curr['Above_Cloud'] == 1 and stock.curr['Tenkan'] > stock.curr['Kijun']: add_trigger("一目多头", "🌥️ [趋势确认] 一目均衡表云上多头共振 (权:{fw:.2f}x)", 6, "TREND")
    if stock.curr['Close'] > stock.curr['VWAP_20'] and stock.prev['Close'] <= stock.curr['VWAP_20']: add_trigger("VWAP突破", "🌊 [量价突破] 放量逾越近20日VWAP机构均价线 (权:{fw:.2f}x)", 8, "TREND")
    if pd.notna(stock.curr['AVWAP']) and stock.curr['Close'] > stock.curr['AVWAP'] and stock.prev['Close'] <= stock.curr['AVWAP']: add_trigger("AVWAP突破", "⚓ [筹码夺回] 强势站上AVWAP锚定成本核心区 (权:{fw:.2f}x)", 12, "TREND")

    kc_w, bb_w = stock.curr['KC_Upper'] - stock.curr['KC_Lower'], stock.curr['BB_Upper'] - stock.curr['BB_Lower']
    if bb_w < kc_w: 
        s_ratio = (kc_w - bb_w) / (stock.curr['ATR'] + 1e-10)
        add_trigger("TTM Squeeze ON", f"📦 [波动压缩] TTM Squeeze 挤流状态激活 (比率:{s_ratio:.2f} 权:{{fw:.2f}}x)", 8 + int(s_ratio*10), "VOLATILITY")

    if stock.prev['NR7'] and stock.prev['Inside_Bar'] and stock.curr['Close'] > stock.prev['High']: add_trigger("NR7极窄突破", "🎯 [极度压缩] 7日极窄压缩孕线完成向上爆破 (权:{fw:.2f}x)", 12, "VOLATILITY")

    vcp_th = Config.Params.VCP_BEAR if ctx.regime in ["bear", "hidden_bear"] else Config.Params.VCP_BULL
    if stock.curr['Range_20'] > 0 and stock.curr['Range_20'] < stock.curr['Range_60'] * vcp_th and stock.curr['Close'] > stock.curr['SMA_50']:
        add_trigger("VCP收缩", f"🌪️ [VCP形态] 极度价格波动压缩后的放量突破 (阈值:{vcp_th} 权:{{fw:.2f}}x)", 15, "VOLATILITY")

    if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Low'] < stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.curr['Swing_Low_20']: add_trigger("流动性扫盘", "🧹 [止损猎杀] 刺穿前低扫掉散户止损后迅速诱空反转 (权:{fw:.2f}x)", 15, "REVERSAL")
    if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Low'] > stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.swing_high_10: add_trigger("特性改变(ChoCh)", "🔀 [结构破坏] 突破近期反弹高点，完成 ChoCh 趋势逆转确认 (权:{fw:.2f}x)", 15, "REVERSAL")
    if pd.notna(stock.curr['OB_High']) and stock.curr['Low'] <= stock.curr['OB_High'] and stock.curr['Close'] >= stock.curr['OB_Low'] and stock.curr['Close'] > stock.curr['Open']: add_trigger("订单块(OB)", "🧱 [机构订单块] 触达历史起爆底仓区并收出企稳阳线 (权:{fw:.2f}x)", 15, "REVERSAL")

    if cf.kde_breakout_score > Config.Params.KDE_BREAKOUT: add_trigger("量子概率云(KDE)", f"☁️ [真空逃逸] KDE 自适应带宽揭示上行阻力极度稀薄 (得分:{cf.kde_breakout_score:.2f} 权:{{fw:.2f}}x)", 15 * cf.kde_breakout_score, "QUANTUM")
    if cf.hurst_reliable and cf.hurst_med > Config.Params.HURST_RELIABLE: add_trigger("稳健赫斯特(Hurst)", f"⏳ [稳健记忆] R/S Bootstrap 确认强抗噪持续性 (Hurst={cf.hurst_med:.2f} 权:{{fw:.2f}}x)", 15 * (cf.hurst_med - 0.5) * 2.0, "QUANTUM")

    if alt.insider_net_buy > Config.Params.INSIDER_BUY: add_trigger("内部人集群净买入(Insider)", "👔 [内幕天眼] SEC Form 4 披露高管集群式大额净买入 (权:{fw:.2f}x)", 20, "QUANTUM")
    if alt.analyst_mom > Config.Params.ANALYST_UP: add_trigger("分析师修正动量(Analyst)", f"📊 [投行护航] 分析师评级修正动量突破 Z-Score={alt.analyst_mom:.1f} (权:{{fw:.2f}}x)", 8, "QUANTUM")
    elif alt.analyst_mom < Config.Params.ANALYST_DN: add_trigger("分析师修正动量(Analyst)", f"⚠️ [投行抛售] 分析师评级下调动量爆表 Z-Score={alt.analyst_mom:.1f}，已被系统降权 (权:{{fw:.2f}}x)", -10, "TREND")

    if macd_crossed and "带量金叉(交互)" in factors_list: theme_scores['TREND'] -= min(get_fw("MACD金叉")*8, get_fw("带量金叉(交互)")*12) * 0.5 
    if cf.pure_alpha > 0.8: add_trigger("独立Alpha(脱钩)", "🪐 [独立Alpha] 强势剥离大盘Beta，爆发特质动能 (权:{fw:.2f}x)", 22, "TREND")
    if stock.curr['SuperTrend_Up'] == 1 and stock.curr['Close'] < stock.curr['EMA_20'] * 1.02: add_trigger("强势回踩", "🟢 [低吸点位] 超级趋势主升轨精准回踩 (权:{fw:.2f}x)", 10, "REVERSAL")
    if gap_pct * 100 > max(1.5, atr_pct * 0.3) and gap_pct < 0.06: add_trigger("突破缺口", "💥 [动能爆发] 放量跳空，留下底部突破缺口 (权:{fw:.2f}x)", 8, "VOLATILITY")
    if cf.fvg_lower > 0 and stock.curr['Low'] <= cf.fvg_upper and stock.curr['Close'] > cf.fvg_lower: add_trigger("SMC失衡区", "🧲 [SMC交易法] 精准回踩并测试前期机构失衡区(FVG) (权:{fw:.2f}x)", 15, "REVERSAL")

    if stock.curr['Volume'] > stock.curr['Vol_MA20'] * 2.0 and abs(stock.curr['Close'] - stock.curr['Open']) < stock.curr['ATR'] * 0.5:
        if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Close'] < stock.curr['Swing_Low_20'] * 1.05: add_trigger("巨量滞涨", "🛑 [冰山吸筹] 底部巨量滞涨，极大概率为机构冰山挂单吸货 (权:{fw:.2f}x)", 12, "QUANTUM")
        elif stock.curr['Close'] > stock.swing_high_10 * 0.95: triggered_list.append("⚠️ [高位派发] 高位巨量滞涨，警惕机构冰山挂单出货 (已被系统降权)")
        
    if day_chg > max(3.0, atr_pct * 0.6) and stock.curr['Volume'] > stock.curr['Vol_MA20'] * 1.5: add_trigger("放量长阳", "⚡ [动能脉冲] 强劲的日内放量大实体阳线 (权:{fw:.2f}x)", 12, "QUANTUM")
    if stock.curr['Close'] > stock.prev['Close'] and stock.curr['Volume'] > stock.curr['Max_Down_Vol_10'] > 0 and stock.curr['Close'] >= stock.curr['EMA_50'] and stock.prev['Close'] <= stock.curr['EMA_50'] * 1.02: add_trigger("口袋支点", "💎 [口袋支点] 放量阳线成交量完全吞噬近期最大阴量 (权:{fw:.2f}x)", 12, "REVERSAL")

    lower_wick = stock.curr['Open'] - stock.curr['Low'] if stock.curr['Close'] > stock.curr['Open'] else stock.curr['Close'] - stock.curr['Low']
    upper_wick = stock.curr['High'] - stock.curr['Close'] if stock.curr['Close'] > stock.curr['Open'] else stock.curr['High'] - stock.curr['Open']
    if stock.curr['Close'] > stock.curr['Open'] and (lower_wick / tr_val) > 0.3 and (upper_wick / tr_val) < 0.15: add_trigger("AMD操盘", "🎭 [AMD诱空] 深度开盘诱空下杀后，全天拉升派发的操盘模型 (权:{fw:.2f}x)", 12, "REVERSAL")
    if cf.weekly_bullish and (stock.curr['Close'] > stock.curr['Highest_22'] * 0.95): add_trigger("跨时空共振(周线)", "🌌 [多周期共振] 周线级别主升浪与日线级别放量的强力双重共振 (权:{fw:.2f}x)", 20, "QUANTUM")

    if stock.curr['CVD_Trend'] == 1.0 and stock.prev['CVD_Trend'] <= 0.0:
        if stock.curr['CVD_Divergence'] == 0: add_trigger("CVD筹码净流入", "🧬 [微观筹码] CVD 双均线平滑多头确立且无量价背离，真实买盘涌入 (权:{fw:.2f}x)", 12, "QUANTUM")
        else: triggered_list.append("⚠️ [微观筹码] 价格近高但 CVD 出现顶背离，尾盘动能涉嫌虚假欺骗 (已被AI降权)")

    if stock.curr['VPT_ZScore'] > 0.5 and stock.curr['VPT_Accel'] > 0 and stock.prev['VPT_ZScore'] <= 0.5: add_trigger("VPT量价共振", "📈 [量价归一] VPT Z-Score突破且动能加速，真实买盘绝对共振 (权:{fw:.2f}x)", 10, "TREND")
    if macd_crossed: add_trigger("带量金叉(交互)", "🔥 [交互共振] MACD金叉与成交量激增产生乘数效应 (权:{fw:.2f}x)", 12, "TREND")
    if stock.curr['CMF'] > 0.15 and stock.curr['Smart_Money_Flow'] > 0.4: add_trigger("量价吸筹(交互)", "🏦 [交互共振] 蔡金资金流与微观聪明钱同向深度吸筹 (权:{fw:.2f}x)", 10, "QUANTUM")
    if cf.fft_ensemble_score >= Config.Params.FFT_RESONANCE: add_trigger("FFT多窗共振(动能)", "🌊 [频域共振] 多窗口 FFT 阵列一致确认，主导周期处于强劲上升象限 (权:{fw:.2f}x)", 15, "QUANTUM")
    if cf.weekly_macd_res == 1.0 and macd_crossed: add_trigger("大周期保护小周期(MACD共振)", "🛡️ [多维时空] 周线MACD动能发散，日线精准金叉，大周期绝对战役保护 (权:{fw:.2f}x)", 15, "TREND")
    if cf.monthly_inst_flow == 1.0: add_trigger("聪明钱月度净流入(月线)", "🏛️ [战略定调] 月线级别连续3个月大单资金净流入，暗池机构底仓坚如磐石 (权:{fw:.2f}x)", 10, "QUANTUM")
    if cf.rsi_60m_bounce == 1.0: add_trigger("60分钟级精准校准(RSI反弹)", "⏱️ [战术执行] 60分钟线 RSI 触底反弹，日内高频微操切入绝佳滑点位置 (权:{fw:.2f}x)", 8, "REVERSAL")

    if cf.beta_60d > 1.2 and not ctx.macro_gravity: add_trigger("大盘Beta(宏观调整)", "📈 [宏观Beta动能] 宏观低压期，高Beta(>1.2)特质赋予极强上行弹性 (权:{fw:.2f}x)", 7, "TREND")
    elif cf.beta_60d > 1.2 and ctx.macro_gravity: add_trigger("大盘Beta(宏观调整)", "⚠️ [宏观Beta反噬] 宏观引力波高压期，高Beta特质面临深度回撤风险，降权防御 (权:{fw:.2f}x)", -10, "TREND")

    if cf.tlt_corr > 0.4: add_trigger("利率敏感度(TLT相关性)", "🏦 [宏观映射] 与长期国债(TLT)高度正相关，受益于无风险利率见顶预期 (权:{fw:.2f}x)", 8, "TREND")
    elif cf.tlt_corr < -0.4: add_trigger("利率敏感度(TLT相关性)", "🛡️ [宏观防御] 与长期国债(TLT)高度负相关，具备抗息避险属性 (权:{fw:.2f}x)", 6, "TREND")
    if cf.dxy_corr < -0.4: add_trigger("汇率传导(DXY相关性)", "💱 [宏观映射] 与美元指数(DXY)强负相关，受惠于弱美元与全球流动性释放 (权:{fw:.2f}x)", 8, "QUANTUM")
    elif cf.dxy_corr > 0.4: add_trigger("汇率传导(DXY相关性)", "💵 [宏观映射] 与美元指数(DXY)强正相关，具备强汇率避险属性 (权:{fw:.2f}x)", 6, "QUANTUM")
        
    dist_52w = stock.curr['Dist_52W_High'] if pd.notna(stock.curr['Dist_52W_High']) else 0.0
    if dist_52w > Config.Params.DIST_52W and cf.weekly_bullish: add_trigger("52周高点距离(动能延续)", "🏔️ [动能延续] 逼近 52 周新高且周线多头，上方无抛压阻力真空区 (权:{fw:.2f}x)", 10, "TREND")
        
    amihud_val = stock.curr['Amihud'] if pd.notna(stock.curr['Amihud']) else 0.0
    if amihud_val > Config.Params.AMIHUD_ILLIQ and ctx.macro_gravity: add_trigger("Amihud非流动性(冲击成本)", "⚠️ [流动性枯竭] 宏观高压下 Amihud 冲击成本显著放大，极易发生踩踏被降权 (权:{fw:.2f}x)", -10, "VOLATILITY")
    if cf.vrp > Config.Params.VRP_EXTREME: add_trigger("波动率风险溢价(VRP)", f"🌋 [风险溢价] VRP归一化极度飙升(溢价率>{cf.vrp*100:.1f}%)，期权市场定价极端恐慌，捕捉极值底 (权:{{fw:.2f}}x)", 12, "QUANTUM")

    if alt.nlp_score > Config.Params.NLP_BULL: add_trigger("舆情NLP情感极值(News_NLP)", f"📰 [舆情引擎] VADER-Lite 分析判定近期新闻呈现极度狂热 (复合得分:{alt.nlp_score:.2f} 权:{{fw:.2f}}x)", 6, "VOLATILITY")
    elif alt.nlp_score < Config.Params.NLP_BEAR: add_trigger("舆情NLP情感极值(News_NLP)", f"⚠️ [舆情崩塌] 探测到新闻包含密集利空/诉讼词汇 (复合得分:{alt.nlp_score:.2f} 权:{{fw:.2f}}x)", -10, "VOLATILITY")
    if alt.wsb_accel > Config.Params.WSB_ACCEL: add_trigger("散户热度加速度(WSB_Accel)", f"🔥 [散户加速] Reddit/WSB 提及数二阶导数飙升 (加速度:{alt.wsb_accel:.0f}/日²)，游资加速抬轿 (权:{{fw:.2f}}x)", 8, "VOLATILITY")

    if alt.pcr > Config.Params.PCR_BEAR: add_trigger("期权PutCall情绪(PCR)", "⚠️ [极端避险] Put/Call Ratio 爆表，期权市场极度看空避险 (权:{fw:.2f}x)", -15, "VOLATILITY"); black_swan_risk = True
    elif 0 < alt.pcr < Config.Params.PCR_BULL: add_trigger("期权PutCall情绪(PCR)", "🔥 [极端贪婪] Put/Call Ratio 极低，期权市场聪明钱疯狂做多 (权:{fw:.2f}x)", 10, "QUANTUM")
    if alt.iv_skew > Config.Params.IV_SKEW_BEAR: add_trigger("隐含波动率偏度(IV Skew)", "🚨 [黑天鹅警报] 看跌期权隐含波动率畸高，内幕资金正疯狂买入下行保护！ (权:{fw:.2f}x)", -20, "VOLATILITY"); black_swan_risk = True
    elif alt.iv_skew < Config.Params.IV_SKEW_BULL: add_trigger("隐含波动率偏度(IV Skew)", "🚀 [上行爆发] 看涨期权隐含波动率压倒性占优，主力预期剧烈向上重估 (权:{fw:.2f}x)", 12, "QUANTUM")
    if alt.short_change > Config.Params.SHORT_SQZ_CHG and alt.short_float > Config.Params.SHORT_SQZ_FLT and stock.curr['Close'] > stock.curr['EMA_20']: add_trigger("做空兴趣突变(轧空)", "💥 [轧空引擎] 近期做空兴趣激增且技术面多头，随时触发空头踩踏平仓的 Short Squeeze (权:{fw:.2f}x)", 15, "QUANTUM")

    saturated_theme_sum = sum([50.0 * (1 - np.exp(-raw_s / 25.0)) for raw_s in theme_scores.values()])
    final_score_saturated = 100.0 * (1 - np.exp(-saturated_theme_sum / 50.0))
    if alt.pcr > Config.Params.PCR_BEAR or alt.iv_skew > Config.Params.IV_SKEW_BEAR: black_swan_risk = True
        
    return int(final_score_saturated), triggered_list, factors_list, black_swan_risk

def _apply_market_filters(curr: pd.Series, prev: pd.Series, sym: str, base_score: int, sig: List[str], 
                          black_hole_sectors: List[str], leading_sectors: List[str], lagging_sectors: List[str]) -> Tuple[int, bool, List[str]]:
    total_score = base_score
    is_bearish_div = False
    if curr['Close'] >= curr['Price_High_20'] * 0.98 and curr['RSI'] < 60.0 and prev['RSI'] > 60.0:
        is_bearish_div, total_score = True, 0 
        sig.append(f"🩸 [顶背离危局] 价格近高但动能显著衰竭，严禁追高")

    if curr['RSI'] > 85.0:
        total_score = max(0, total_score - 30)
        sig.append(f"⚠️ [极端超买] RSI 高达 {curr['RSI']:.1f} (严防获利盘踩踏)")
        
    bias_20 = (curr['Close'] - curr['EMA_20']) / (curr['EMA_20'] + 1e-10)
    if bias_20 > 0.08:
        total_score = max(0, total_score - int((bias_20 - 0.08) * 100))
        sig.append(f"⚠️ [短线高估] 偏离20日均线 +{bias_20*100:.1f}% (防追高扣减)")

    sym_sec = Config.get_sector_etf(sym)
    if sym_sec in black_hole_sectors:
        total_score = int(total_score * 1.30)
        sig.append(f"🕳️ [流动性黑洞] 所属板块 {sym_sec} 正疯狂吸血全市场资金 (+30%)")
    elif sym_sec in leading_sectors:
        total_score = int(total_score * 1.15)
        sig.append(f"🔥 [RRG象限跃迁] 所属板块 {sym_sec} 相对动能呈现加速领跑 (+15%)")
    elif sym_sec in lagging_sectors:
        total_score = int(total_score * 0.85)
        sig.append(f"🧊 [RRG象限坠落] 所属板块 {sym_sec} 相对动能呈现加速衰退 (-15%)")
        
    return total_score, is_bearish_div, sig

def _build_market_context() -> MarketContext:
    logger.info("🌐 正在构建宏观市场感知雷达 (Market Context)...")
    qqq_df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", fast_mode=True)
    macro_data = {
        'spy': safe_get_history("SPY", period="1y", interval="1d", fast_mode=True),
        'tlt': safe_get_history("TLT", period="1y", interval="1d", fast_mode=True),
        'dxy': safe_get_history("DX-Y.NYB", period="1y", interval="1d", fast_mode=True)
    }

    vix_current, vix_desc = get_vix_level(qqq_df)
    vix_scalar = 1.0 if vix_current < 20 else (20.0 / vix_current)
    vix_inv = vix_current > 30

    active_pool = get_filtered_watchlist(max_stocks=100)
    regime, regime_desc, _, is_credit_risk_high, macro_gravity = get_market_regime(active_pool)

    xai_weights, meta_weights, health_score, pain_warning = {}, {}, 1.0, ""
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                xai_data = stats_data.get("xai_importances", {})
                if xai_data:
                    avg_imp = 1.0 / len(Config.ALL_FACTORS)
                    for tag, imp in xai_data.items(): xai_weights[tag] = 0.0 if imp < avg_imp * 0.25 else max(0.5, min(3.0, float(imp) / avg_imp))
                meta_weights = stats_data.get("meta_weights", {})
                t3_stats = stats_data.get("overall", {}).get("T+3", {})
                max_cons_loss = t3_stats.get("max_cons_loss", 0)
                if max_cons_loss >= 4:
                    health_score = 0.5
                    pain_warning = f" | 🩸 痛觉激活: 历史探测到 {max_cons_loss} 连亏，系统强制降杠杆防守"
    except Exception: pass

    w_mul, max_risk = 1.0, Config.Params.BASE_MAX_RISK
    if regime in ["bear", "hidden_bear"]: w_mul *= 0.6; max_risk *= 0.6
    elif regime in ["bull", "rebound"]: w_mul *= 1.2; max_risk *= 1.2
    if macro_gravity: w_mul *= 0.5; max_risk *= 0.5
    if is_credit_risk_high: w_mul *= 0.7

    w_mul *= health_score; max_risk *= health_score

    total_market_exposure = 1.0
    if vix_current > 25: total_market_exposure = 0.6
    if vix_current > 35: total_market_exposure = 0.3
    if macro_gravity: total_market_exposure = min(total_market_exposure, 0.5)

    wsb_data = fetch_global_wsb_data()

    ctx = MarketContext(
        regime=regime, regime_desc=regime_desc, w_mul=w_mul, xai_weights=xai_weights,
        vix_current=vix_current, vix_desc=vix_desc, vix_scalar=vix_scalar, max_risk=max_risk,
        macro_gravity=macro_gravity, is_credit_risk_high=is_credit_risk_high, vix_inv=vix_inv,
        qqq_df=qqq_df, macro_data=macro_data, total_market_exposure=total_market_exposure,
        health_score=health_score, pain_warning=pain_warning,
        credit_spread_mom=0.0, vix_term_structure=1.0, market_pcr=1.0,
        dynamic_min_score=Config.Params.MIN_SCORE_THRESHOLD,
        global_wsb_data=wsb_data, meta_weights=meta_weights
    )
    logger.info(f"✅ 宏观上下文雷达构建完毕: Regime={regime.upper()} | 敞口上限={total_market_exposure*100:.0f}% | VIX={vix_current:.1f}")
    return ctx

def _prepare_universe_data(ctx: MarketContext) -> Tuple[List[dict], dict]:
    active_pool = get_filtered_watchlist(max_stocks=200)
    prepared_data, price_history_dict = [], {}
    
    def _worker(sym: str) -> Optional[dict]:
        df = safe_get_history(sym, "1y", "1d", fast_mode=True)
        if df.empty or len(df) < 120: return None
        df_w = df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        df_m = df.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        df_ind = calculate_indicators(df)
        curr, prev = df_ind.iloc[-1], df_ind.iloc[-2]
        
        stock = StockData(sym, df_ind, df_w, df_m, pd.DataFrame(), curr, prev, curr['Volume'] > curr['Vol_MA20'] * 1.5, df_ind['High'].iloc[-11:-1].max() if len(df_ind) >= 11 else curr['High'])
        pcr, iv_skew, sc, sf = safe_get_sentiment_data(sym)
        inb, am, nlp, news = safe_get_alt_data(sym)
        alt = AltData(pcr, iv_skew, sc, sf, inb, am, nlp, 0.0)
        
        return {'sym': sym, 'stock': stock, 'alt': alt, 'cf': _extract_complex_features(stock, ctx), 'seq': _get_transformer_seq(df_ind), 'curr': curr, 'prev': prev, 'news': news, 'close_history': df['Close'].tail(60).values}

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.Params.MAX_WORKERS) as executor:
        for future in concurrent.futures.as_completed([executor.submit(_worker, sym) for sym in active_pool]):
            res = future.result()
            if res: prepared_data.append(res); price_history_dict[res['sym']] = res['close_history']
    return prepared_data, price_history_dict

def _apply_ai_inference(raw_reports: List[dict], ctx: MarketContext) -> List[dict]:
    if not os.path.exists(Config.MODEL_FILE):
        logger.warning("未找到左脑树模型 scoring_model.pkl，退化为纯军规打分。")
        for r in raw_reports: r['ai_prob'] = 0.5
        return raw_reports
    try:
        import pickle
        with open(Config.MODEL_FILE, 'rb') as f: model_pack = pickle.load(f)
        active_factors, scaler = model_pack.get('active_factors', Config.ALL_FACTORS), model_pack['scaler']
        base_A, base_B, meta_clf = model_pack.get('base_A'), model_pack.get('base_B'), model_pack['meta']
        active_A, active_B = model_pack.get('active_A', []), model_pack.get('active_B', [])
        
        df_feats = pd.DataFrame([r['ml_features'] for r in raw_reports]).fillna(0.0)
        for f in active_factors:
            if f not in df_feats.columns: df_feats[f] = 0.0
        X_scaled_df = pd.DataFrame(scaler.transform(df_feats[active_factors]), columns=active_factors)
        
        prob_A = base_A.predict_proba(X_scaled_df[active_A].values)[:, 1] if base_A and active_A else np.full(len(raw_reports), 0.5)
        prob_B = base_B.predict_proba(X_scaled_df[active_B].values)[:, 1] if base_B and active_B else np.full(len(raw_reports), 0.5)
        
        X_meta = np.column_stack([prob_A, prob_B, np.full(len(raw_reports), ctx.vix_current / 20.0), np.full(len(raw_reports), ctx.credit_spread_mom), np.full(len(raw_reports), ctx.vix_term_structure - 1.0), np.full(len(raw_reports), ctx.market_pcr - 1.0)])
        final_probs = meta_clf.predict_proba(X_meta)[:, 1]
        
        for i, r in enumerate(raw_reports):
            r['ai_prob'] = float(final_probs[i])
            if r['ai_prob'] > 0.65: r['total_score'] = int(r['total_score'] * 1.2); r['sig'].append(f"🧠 [AI 确认] Meta-Learner 胜率极高 ({r['ai_prob']:.1%})，权重上调")
            elif r['ai_prob'] < 0.40: r['total_score'] = int(r['total_score'] * 0.5); r['sig'].append(f"⚠️ [AI 否决] Meta-Learner 胜率低迷 ({r['ai_prob']:.1%})，权重削减")
    except Exception as e: logger.error(f"AI 推理管线崩溃: {e}")
    return raw_reports

def _calculate_position_size(stock: StockData, ctx: MarketContext, ai_prob: float, is_bearish_div: bool, black_swan_risk: bool) -> Tuple[float, float, float, str]:
    atr_mult_sl, atr_mult_tp = (1.0, 2.0) if ctx.vix_inv else (1.5, 3.0)
    tp_val = stock.curr['Close'] + atr_mult_tp * ctx.vix_scalar * stock.curr['ATR']
    sl_val = max(stock.curr['Chandelier_Exit'] if pd.notna(stock.curr['Chandelier_Exit']) else (stock.curr['Close'] - atr_mult_sl * ctx.vix_scalar * stock.curr['ATR']), stock.curr['Close'] - atr_mult_sl * ctx.vix_scalar * stock.curr['ATR'])
    kelly_fraction = ai_prob - (1.0 - ai_prob) / 2.0
    if black_swan_risk: return tp_val, sl_val, 0.0, "❌ 强制熔断 (期权隐含波动率/情绪探测到黑天鹅巨险)"
    if kelly_fraction <= 0 or is_bearish_div: return tp_val, sl_val, 0.0, "❌ 放弃建仓 (AI盈亏比劣势 或 顶背离确认)"
    return tp_val, sl_val, kelly_fraction, ""

def _apply_kelly_cluster_optimization(reports: List[dict], price_history_dict: dict, total_exposure: float, ctx: MarketContext) -> List[dict]:
    candidate_pool = sorted(reports, key=lambda x: (x["score"], x["ai_prob"]), reverse=True)[:15]
    if not candidate_pool: return []
    if len(candidate_pool) == 1: candidate_pool[0]['pos_advice'], candidate_pool[0]['opt_weight'] = f"✅ 组合配置权重: {total_exposure * 100:.1f}% (极简化单目标资产)", 1.0; return candidate_pool
        
    try:
        syms = [r['symbol'] for r in candidate_pool]
        ret_df = pd.DataFrame({sym: price_history_dict[sym] for sym in syms}).ffill().pct_change().dropna()
        cov_matrix = LedoitWolf().fit(ret_df).covariance_
        
        cvars = []
        for sym in syms:
            rets = ret_df[sym].values
            if len(rets) > 0:
                var_95 = np.percentile(rets, 5) 
                cvars.append(abs(min(rets[rets <= var_95].mean() if len(rets[rets <= var_95]) > 0 else var_95, 0.0)))
            else: cvars.append(0.05) 
        cvars = np.array(cvars)
        
        scores = np.array([r['score'] for r in candidate_pool])
        norm_scores = scores / (np.max(scores) + 1e-10)
        risk_aversion = max(2.0, 5.0 - ctx.health_score * 3.0)
        
        def objective(w): return -(np.dot(w, norm_scores) - risk_aversion * np.dot(w, cvars))
        sectors, bounds, init_w = np.array([r['sector'] for r in candidate_pool]), tuple((0.02, 0.15) for _ in range(len(syms))), np.ones(len(syms)) / len(syms)

        def optimize_with_constraints(sec_limit, vol_limit):
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}] 
            for sec in np.unique(sectors): constraints.append({'type': 'ineq', 'fun': lambda w, m=(sectors == sec).astype(float), lim=sec_limit: lim - np.dot(w, m)})
            constraints.append({'type': 'ineq', 'fun': lambda w, targ=(vol_limit / np.sqrt(252)) ** 2: targ - np.dot(w.T, np.dot(cov_matrix, w))})
            return minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)

        res = optimize_with_constraints(0.40, 0.25)
        if res.success: opt_weights = res.x
        else:
            logger.warning(f"⚠️ 严格凸优化约束冲突，尝试放宽...")
            res_relaxed = optimize_with_constraints(0.60, 0.35)
            if res_relaxed.success: opt_weights = res_relaxed.x
            else:
                logger.warning(f"🚨 优化器熔断，回退至得分加权。")
                raw_w = norm_scores * (1.0 / (cvars + 1e-5))
                opt_weights = np.clip((raw_w / np.sum(raw_w)), 0.02, 0.15)
                opt_weights = opt_weights / np.sum(opt_weights)
            
        for i, r in enumerate(candidate_pool):
            r['opt_weight'], r['cvar_95'], r['pos_advice'] = opt_weights[i], cvars[i], f"✅ 组合配置权重: {opt_weights[i] * total_exposure * 100:.1f}% (Kelly Cluster 分配)"
        return candidate_pool
    except Exception: return candidate_pool

def _generate_and_send_matrix_report(final_reports: List[dict], final_shadow_pool: List[dict], ctx: MarketContext) -> None:
    txts = []
    for idx, r in enumerate(final_reports):
        icon = ['🥇', '🥈', '🥉'][idx] if idx < 3 else '🔸'
        txts.append(f"### {icon} **{r['symbol']}** | 🤖 分层元学习胜率: {f'🔥 **{r.get('ai_prob', 0):.1%}**' if r.get('ai_prob', 0) > 0.60 else f'{r.get('ai_prob', 0):.1%}'} | 🌟 终极评级: {r['score']}分\n**💡 机构交易透视:**\n" + "\n".join([f"- {s}" for s in r["signals"]]) + (f"\n- 📰 {r['news']}" if r['news'] else "") + f"\n\n**💰 绝对风控界限:**\n- 💵 现价: `{r['curr_close']:.2f}`\n- ⚖️ {r.get('pos_advice', '✅ 缺省仓位')}\n- 🎯 建议止盈: **${r['tp']:.2f}**\n- 🛡️ 吊灯止损: **${r['sl']:.2f} (最高价回落保护)**\n- 📈 离场纪律: **跌破止损防线请无条件市价清仓！**")

    perf = load_strategy_performance_tag()
    header = f"**📊 宏观引力与决策体系状态:**\n- {ctx.vix_desc}\n- {ctx.regime_desc}{ctx.pain_warning}\n- ⚔️ 今日截面淘汰线: **{ctx.dynamic_min_score:.1f}分**"
    meta_desc = f"\n\n**🧠 Stacking Meta-Learner 状态:**\n- 传统大局观 (Group A) 权重: **{ctx.meta_weights.get('Group_A', 0)*100:.1f}%**\n- 高级微观组 (Group B) 权重: **{ctx.meta_weights.get('Group_B', 0)*100:.1f}%**" if ctx.meta_weights else ""

    send_alert("量化诸神之战 (防特征错位版)", (f"{perf}\n\n{header}\n\n---\n\n" if perf else f"{header}\n\n---\n\n") + "\n\n---\n\n".join(txts) + meta_desc + f"\n\n*(防范特征错位: 已引入严格的训练-推理对齐机制，确保宏观因子正确穿透至元学习器！)*")

    try:
        with open(Config.get_current_log_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), "macro_meta": {"vix": ctx.vix_current, "credit_spread_mom": ctx.credit_spread_mom, "vix_term_structure": ctx.vix_term_structure, "market_pcr": ctx.market_pcr}, "top_picks": [{"symbol": r.get("symbol"), "score": r.get("score"), "signals": r.get("signals"), "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "ai_prob": r.get("ai_prob", 0.0), "tp": r.get("tp"), "sl": r.get("sl")} for r in final_reports], "shadow_pool": [{"symbol": r.get("symbol"), "score": r.get("score"), "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "ai_prob": r.get("ai_prob", 0.0)} for r in final_shadow_pool]}, ensure_ascii=False) + "\n")
    except Exception as e: logger.error(f"严重：写入矩阵全息日志时发生崩溃: {e}")

def _route_orders_to_gateway(final_reports: List[dict], ctx: MarketContext) -> None:
    if not final_reports: return
    os.makedirs(os.path.dirname(Config.ORDER_DB_PATH), exist_ok=True)
    try:
        with sqlite3.connect(Config.ORDER_DB_PATH, timeout=10.0) as conn:
            conn.execute('pragma journal_mode=wal')
            conn.execute('''CREATE TABLE IF NOT EXISTS orders (client_oid TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL, qty REAL NOT NULL, order_type TEXT NOT NULL, limit_price REAL, arrival_price REAL NOT NULL, status TEXT NOT NULL, filled_qty REAL DEFAULT 0.0, avg_fill_price REAL DEFAULT 0.0, retry_count INTEGER DEFAULT 0, broker_oid TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            for r in final_reports:
                if r.get('opt_weight', 0.0) > 0.0:
                    alloc_usd = Config.Params.PORTFOLIO_VALUE * ctx.total_market_exposure * r['opt_weight']
                    qty = round(alloc_usd / max(r['curr_close'], 1e-10), 4)
                    if qty > 0:
                        conn.execute('''INSERT INTO orders (client_oid, symbol, side, qty, order_type, arrival_price, status) VALUES (?, ?, 'BUY', ?, 'MKT', ?, 'PENDING_SUBMIT')''', (f"QB_{uuid.uuid4().hex[:8]}", r['symbol'], qty, r['curr_close']))
                        logger.info(f"📡 [网关路由] 成功下发 {r['symbol']} BUY 指令: {qty} 股 -> PENDING_SUBMIT")
        logger.info(f"✅ [网关路由] 本轮矩阵扫盘批次生成完毕，共 {len(final_reports)} 笔指令已推送至底层执行状态机。")
    except Exception as e: logger.error(f"❌ [网关路由] IPC 致命错误，写入底层账本失败: {e}")

# ================= 5. 实盘执行网关 (Execution Gateway & Singularity) =================
@dataclass
class BrokerOrder:
    broker_oid: str
    status: str
    filled_qty: float
    avg_fill_price: float

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float
    unrealized_pnl: float

class BaseBrokerGateway:
    def __init__(self): self.is_connected = False
    def connect(self) -> bool: return True
    def disconnect(self): pass
    def get_account_summary(self) -> Dict[str, float]: raise NotImplementedError
    def get_positions(self) -> List[Position]: raise NotImplementedError
    def cancel_order(self, broker_oid: str) -> bool: raise NotImplementedError
    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None, client_oid: str = None) -> BrokerOrder: raise NotImplementedError
    def fetch_order(self, broker_oid: str) -> BrokerOrder: raise NotImplementedError

class MockAlpacaGateway(BaseBrokerGateway):
    def __init__(self):
        super().__init__()
        self._mock_exchange = {}
        self._cash = 100000.0
        self._positions = {}

    def connect(self) -> bool:
        self.is_connected = True
        logger.info("🟢 Mock Broker 虚拟专线已连接，进入纸面交易模式。")
        return True

    def disconnect(self):
        self.is_connected = False
        logger.info("🔴 Mock Broker 虚拟专线已断开。")

    def get_account_summary(self) -> Dict[str, float]:
        return {"total_cash": self._cash, "buying_power": self._cash * 2, "net_liquidation": self._cash + sum(q * 100.0 for q in self._positions.values())}

    def get_positions(self) -> List[Position]:
        return [Position(sym, q, 100.0, q*100.0, 0.0) for sym, q in self._positions.items() if q != 0]

    def cancel_order(self, broker_oid: str) -> bool:
        if broker_oid in self._mock_exchange and self._mock_exchange[broker_oid]["status"] == "OPEN":
            self._mock_exchange[broker_oid]["status"] = "CANCELED"
            return True
        return False

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None, client_oid: str = None) -> BrokerOrder:
        broker_oid = f"ALPACA_{uuid.uuid4().hex[:12]}"
        self._mock_exchange[broker_oid] = {
            "status": "OPEN", "filled_qty": 0.0, "avg_fill_price": 0.0, "qty": qty, 
            "price": limit_price or 100.0, "side": side, "symbol": symbol, "created_ts": time.time()
        }
        return BrokerOrder(broker_oid, "OPEN", 0.0, 0.0)

    def fetch_order(self, broker_oid: str) -> BrokerOrder:
        if broker_oid not in self._mock_exchange: raise ValueError("Order not found")
        order = self._mock_exchange[broker_oid]
        if order["status"] == "OPEN" and time.time() - order["created_ts"] > 0.5:
            order["status"] = "FILLED"
            order["filled_qty"] = order["qty"]
            slip_pct = np.random.normal(1.5, 1.0) / 10000.0
            order["avg_fill_price"] = order["price"] * (1 + slip_pct) if order["side"] == "BUY" else order["price"] * (1 - slip_pct)
            cost = order["avg_fill_price"] * order["qty"]
            if order["side"] == "BUY":
                self._cash -= cost
                self._positions[order["symbol"]] = self._positions.get(order["symbol"], 0) + order["qty"]
            else:
                self._cash += cost
                self._positions[order["symbol"]] = self._positions.get(order["symbol"], 0) - order["qty"]
        return BrokerOrder(broker_oid, order["status"], order["filled_qty"], order["avg_fill_price"])

class AlpacaGateway(BaseBrokerGateway):
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret, "Content-Type": "application/json"}

    def connect(self) -> bool:
        try:
            if requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10).status_code == 200:
                self.is_connected = True; logger.info("🟢 Alpaca 真实/纸面专线连接成功！"); return True
            return False
        except Exception as e: logger.error(f"🔴 Alpaca 连接异常: {e}"); return False

    def disconnect(self): self.is_connected = False; logger.info("🔴 Alpaca 专线已断开。")

    def get_account_summary(self) -> Dict[str, float]:
        resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
        if resp.status_code == 200: return {"total_cash": float(resp.json().get("cash", 0.0)), "buying_power": float(resp.json().get("buying_power", 0.0)), "net_liquidation": float(resp.json().get("portfolio_value", 0.0))}
        return {"total_cash": 0.0, "buying_power": 0.0, "net_liquidation": 0.0}

    def get_positions(self) -> List[Position]:
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
        if resp.status_code == 200: return [Position(p["symbol"], float(p["qty"]), float(p["avg_entry_price"]), float(p["market_value"]), float(p["unrealized_pl"])) for p in resp.json()]
        return []

    def cancel_order(self, broker_oid: str) -> bool:
        if requests.delete(f"{self.base_url}/v2/orders/{broker_oid}", headers=self.headers, timeout=10).status_code in [200, 204]:
            logger.info(f"🛑 [Alpaca] 撤单成功: {broker_oid}"); return True
        return False

    def _map_status(self, alpaca_status: str) -> str:
        return {"new": "OPEN", "accepted": "OPEN", "partially_filled": "PARTIALLY_FILLED", "filled": "FILLED", "done_for_day": "FILLED", "canceled": "CANCELED", "expired": "EXPIRED", "replaced": "OPEN", "pending_cancel": "OPEN", "pending_replace": "OPEN", "rejected": "REJECTED", "suspended": "REJECTED"}.get(alpaca_status, "OPEN")

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None, client_oid: str = None) -> BrokerOrder:
        payload = {"symbol": symbol, "qty": str(qty), "side": side.lower(), "type": "market" if order_type == "MARKET" else "limit", "time_in_force": "day"}
        if limit_price and payload["type"] == "limit": payload["limit_price"] = str(round(limit_price, 2))
        if client_oid: payload["client_order_id"] = client_oid # 👈 注入强一致性幂等标识，彻底免疫网络波动导致的多重扣款
        resp = requests.post(f"{self.base_url}/v2/orders", json=payload, headers=self.headers, timeout=10)
        if resp.status_code in [200, 201]: return BrokerOrder(resp.json()["id"], self._map_status(resp.json()["status"]), float(resp.json()["filled_qty"]), float(resp.json()["filled_avg_price"] or 0.0))
        logger.error(f"❌ [Alpaca] 委托拒绝: {resp.text}"); return None

    def fetch_order(self, broker_oid: str) -> BrokerOrder:
        resp = requests.get(f"{self.base_url}/v2/orders/{broker_oid}", headers=self.headers, timeout=10)
        if resp.status_code == 200: return BrokerOrder(resp.json()["id"], self._map_status(resp.json()["status"]), float(resp.json()["filled_qty"]), float(resp.json()["filled_avg_price"] or 0.0))
        raise RuntimeError(f"Alpaca Fetch Error: {resp.text}")

class OrderLedger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute('pragma journal_mode=wal')
            conn.execute('''CREATE TABLE IF NOT EXISTS orders (client_oid TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL, qty REAL NOT NULL, order_type TEXT NOT NULL, limit_price REAL, arrival_price REAL NOT NULL, status TEXT NOT NULL, filled_qty REAL DEFAULT 0.0, avg_fill_price REAL DEFAULT 0.0, retry_count INTEGER DEFAULT 0, broker_oid TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON orders(status)')

    def fetch_orders_by_status(self, statuses: list) -> pd.DataFrame:
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            return pd.read_sql_query(f"SELECT * FROM orders WHERE status IN ({','.join(['?']*len(statuses))})", conn, params=statuses).copy(deep=True)

    def update_order_status(self, client_oid: str, status: str, filled_qty: float = 0.0, avg_fill_price: float = 0.0, broker_oid: str = None):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn: conn.execute('''UPDATE orders SET status = ?, filled_qty = ?, avg_fill_price = ?, broker_oid = COALESCE(?, broker_oid), updated_at = CURRENT_TIMESTAMP WHERE client_oid = ?''', (status, filled_qty, avg_fill_price, broker_oid, client_oid))
            
    def increment_retry(self, client_oid: str):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn: conn.execute('UPDATE orders SET retry_count = retry_count + 1 WHERE client_oid = ?', (client_oid,))

    def insert_mock_order(self, symbol: str, side: str, qty: float, arrival_price: float) -> str:
        client_oid = f"QB_{uuid.uuid4().hex[:8]}"
        with sqlite3.connect(self.db_path, timeout=10.0) as conn: conn.execute('''INSERT INTO orders (client_oid, symbol, side, qty, order_type, arrival_price, status) VALUES (?, ?, ?, ?, 'MKT', ?, 'PENDING_SUBMIT')''', (client_oid, symbol, side, qty, arrival_price))
        return client_oid

class ExecutionEngine:
    def __init__(self, broker: BaseBrokerGateway, db_path: str = Config.ORDER_DB_PATH):
        self.broker = broker
        self.ledger = OrderLedger(db_path)
        self.is_running = False

    def _write_tca_log_atomic(self, order_row: pd.Series):
        arr_px = np.maximum(float(order_row['arrival_price']), 1e-10)
        slippage_bps = (float(order_row['avg_fill_price']) - arr_px) / arr_px * 10000.0
        if order_row['side'] == "SELL": slippage_bps = -slippage_bps
        tca_record = {"client_oid": order_row['client_oid'], "symbol": order_row['symbol'], "side": order_row['side'], "qty": float(order_row['filled_qty']), "arrival_price": arr_px, "execution_price": float(order_row['avg_fill_price']), "timestamp": datetime_to_str(), "slippage_bps": float(slippage_bps)}
        tmp_path = f"{Config.TCA_LOG_PATH}.{threading.get_ident()}.tmp"
        try:
            history = []
            if os.path.exists(Config.TCA_LOG_PATH):
                with open(Config.TCA_LOG_PATH, 'r') as f: history = [json.loads(line) for line in f]
            history.append(tca_record)
            with open(tmp_path, 'w') as f:
                for h in history: f.write(json.dumps(h) + '\n')
            os.replace(tmp_path, Config.TCA_LOG_PATH)
            logger.info(f"📊 [TCA] 订单 {tca_record['client_oid']} 归因完成。滑点: {slippage_bps:.2f} bps")
        except Exception as e:
            logger.error(f"TCA 日志原子写入崩溃: {e}")
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def _process_queue(self):
        for _, row in self.ledger.fetch_orders_by_status(['PENDING_SUBMIT']).iterrows():
            oid = row['client_oid']
            if row['qty'] * row['arrival_price'] > 50000.0 or row['retry_count'] >= 3:
                self.ledger.update_order_status(oid, 'REJECTED')
                continue
            try:
                logger.info(f"📤 向券商提交订单 {oid}: {row['side']} {row['qty']} {row['symbol']}")
                # 🚀 将策略端生成的唯一 OID 送入网关，构建防重下屏障
                b_order = self.broker.submit_order(row['symbol'], row['side'], float(row['qty']), "MARKET" if row['order_type'] == "MKT" else "LIMIT", float(row['limit_price']) if pd.notna(row['limit_price']) else None, client_oid=oid)
                if b_order: self.ledger.update_order_status(oid, b_order.status, b_order.filled_qty, b_order.avg_fill_price, b_order.broker_oid)
                else: self.ledger.update_order_status(oid, 'REJECTED')
            except Exception as e:
                logger.error(f"提交订单 {oid} 时异常: {e}")
                self.ledger.increment_retry(oid)

    def _sync_open_orders(self):
        for _, row in self.ledger.fetch_orders_by_status(['SUBMITTED', 'OPEN', 'PARTIALLY_FILLED']).iterrows():
            if pd.notna(row['broker_oid']):
                try:
                    b_order = self.broker.fetch_order(row['broker_oid'])
                    if b_order.status != row['status'] or b_order.filled_qty != row['filled_qty']:
                        logger.info(f"🔄 订单 {row['client_oid']} 状态跃迁: {row['status']} -> {b_order.status}")
                        self.ledger.update_order_status(row['client_oid'], b_order.status, b_order.filled_qty, b_order.avg_fill_price)
                        if b_order.status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED'] and b_order.filled_qty > 0:
                            row_copy = row.copy(deep=True); row_copy['filled_qty'] = b_order.filled_qty; row_copy['avg_fill_price'] = b_order.avg_fill_price
                            self._write_tca_log_atomic(row_copy)
                except Exception as e: logger.error(f"同步订单 {row['client_oid']} 失败: {e}")

    def run(self):
        self.is_running = True
        logger.info("🚀 Execution Gateway 守护进程启动，开始监听 IPC...")
        if not self.broker.is_connected: self.broker.connect()
        try:
            while self.is_running:
                self._process_queue()
                self._sync_open_orders()
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("🛑 网关安全下线。")
            self.broker.disconnect()
            self.is_running = False

def run_gateway():
    api_key, api_secret, base_url = os.environ.get('ALPACA_API_KEY', ''), os.environ.get('ALPACA_API_SECRET', ''), os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    if api_key and api_secret: logger.info("🔗 初始化 Alpaca 真实交易网关..."); broker = AlpacaGateway(api_key, api_secret, base_url)
    else: logger.warning("⚠️ 未检测到 Alpaca API 凭证，降级为 Mock 纸面模拟网关！"); broker = MockAlpacaGateway()
    ExecutionEngine(broker).run()

# ================= 6. 主程序入口路由 (Runners) =================
def run_tech_matrix() -> None:
    ctx = _build_market_context()
    
    if TRANSFORMER_AVAILABLE:
        t_path = os.path.join(Config.DATA_DIR, "transformer_production.pth")
        if os.path.exists(t_path):
            try:
                import torch
                load_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ctx.transformer_model = QuantAlphaTransformer.load_checkpoint(t_path, device=load_device)
                logger.info(f"🧠 深度学习右脑成功唤醒，准备接收 Batch 批量张量！(设备: {load_device})")
            except Exception as e: logger.error(f"⚠️ 右脑唤醒失败: {e}")
    
    prepared_data, price_history_dict = _prepare_universe_data(ctx)
    if not prepared_data: logger.info("📭 扫描池为空，提前终止。"); return
        
    if getattr(ctx, 'transformer_model', None) is not None:
        try:
            seqs = np.array([d['seq'] for d in prepared_data]) 
            alpha_vecs = ctx.transformer_model.extract_alpha(seqs) 
            for i, d in enumerate(prepared_data): d['alpha_vec'] = alpha_vecs[i]
        except Exception as e:
            for d in prepared_data: d['alpha_vec'] = np.zeros(16)
    else:
        for d in prepared_data: d['alpha_vec'] = np.zeros(16)

    raw_reports = []
    for d in prepared_data:
        ml_features = _extract_ml_features(d['stock'], ctx, d['cf'], d['alt'], d['alpha_vec'])
        base_score, sig, factors, black_swan = _evaluate_omni_matrix(d['stock'], ctx, d['cf'], d['alt'])
        score, is_bearish_div, sig = _apply_market_filters(d['curr'], d['prev'], d['sym'], base_score, sig, [], [], [])
        raw_reports.append({'sym': d['sym'], 'curr': d['curr'], 'prev': d['prev'], 'score': score, 'sig': sig, 'factors': factors, 'ml_features': ml_features, 'news': d['news'], 'sym_sec': Config.get_sector_etf(d['sym']), 'is_bearish_div': is_bearish_div, 'black_swan_risk': black_swan, 'total_score': score, 'is_untradeable': False, 'ai_prob': 0.0})

    raw_reports = _apply_ai_inference(raw_reports, ctx)
    
    reports, background_pool, all_raw_scores = [], [], []
    for r in raw_reports:
        if r['total_score'] > 0: all_raw_scores.append(r['total_score'])
        tp_val, sl_val, kelly_fraction, basic_advice = _calculate_position_size(StockData(r['sym'], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), r['curr'], r['prev'], False, 0.0), ctx, r['ai_prob'], r['is_bearish_div'], r['black_swan_risk'])
        stock_data_pack = {"symbol": r['sym'], "score": r['total_score'], "ai_prob": r['ai_prob'], "signals": r['sig'][:8], "factors": r['factors'], "ml_features": r['ml_features'], "curr_close": float(r['curr']['Close']), "tp": float(tp_val), "sl": float(sl_val), "news": r['news'], "sector": r['sym_sec'], "pos_advice": basic_advice, "kelly_fraction": kelly_fraction}

        if not r['is_untradeable'] and r['total_score'] > 0 and kelly_fraction > 0:
            if check_earnings_risk(r['sym']):
                r['sig'].append("💣 [财报雷区] 近5日发财报,风险极高")
                r['total_score'] = int(r['total_score'] * 0.5)
                stock_data_pack["score"] = r['total_score']
            reports.append(stock_data_pack)
        else: background_pool.append(stock_data_pack)

    if reports and all_raw_scores:
        ctx.dynamic_min_score = max(Config.Params.MIN_SCORE_THRESHOLD, np.percentile(all_raw_scores, 85))
        reports = [r for r in reports if r['score'] >= ctx.dynamic_min_score]
        
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
        
        for sec, stks in groups.items():
            if hasattr(Config, 'CROWDING_EXCLUDE_SECTORS') and sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.Params.CROWDING_MIN_STOCKS:
                pen = max(0.6, min(0.9, Config.Params.CROWDING_PENALTY * (1.0 + ctx.health_score * 0.3)))
                for s in stks[1:]: s["score"] = int(s["score"] * pen)

        final_reports = _apply_kelly_cluster_optimization(reports, price_history_dict, ctx.total_market_exposure, ctx)
        for r in final_reports: set_alerted(r["symbol"])
        
        final_symbols = {r['symbol'] for r in final_reports}
        unselected_background = [s for s in background_pool if s['symbol'] not in final_symbols]
        random.shuffle(unselected_background)
        final_shadow_pool = unselected_background[:150]
        
        for r in final_shadow_pool: set_alerted(r["symbol"], is_shadow=True, shadow_data=r)
            
        _generate_and_send_matrix_report(final_reports, final_shadow_pool, ctx)
        _route_orders_to_gateway(final_reports, ctx)
    else: logger.info("📭 本次矩阵扫描无标的突破 Top 15% 截面排位，宁缺毋滥，保持静默。")

def run_backtest_engine() -> None:
    log_files = [f for f in os.listdir('.') if f.startswith('backtest_log') and f.endswith('.jsonl')]
    if not log_files: logger.warning("未找到历史日志，回测取消。"); return
        
    trades = []
    for lf in log_files:
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log = json.loads(line.strip())
                        daily_trades = log.get('top_picks', []) + log.get('shadow_pool', [])
                        macro = log.get('macro_meta', {})
                        for p in daily_trades:
                            raw_ml = p.get('ml_features', {})
                            if isinstance(raw_ml, list): ml_feats = {Config.ALL_FACTORS[i]: val for i, val in enumerate(raw_ml + [0.0] * (len(Config.ALL_FACTORS) - len(raw_ml)))}
                            else: ml_feats = raw_ml
                            trades.append({'date': log['date'], 'vix': macro.get('vix', log.get('vix', 18.0)), 'cred': macro.get('credit_spread_mom', 0.0), 'term': macro.get('vix_term_structure', 1.0), 'pcr': macro.get('market_pcr', 1.0), 'symbol': p['symbol'], 'signals': p.get('signals', []), 'factors': p.get('factors', []), 'ml_features': ml_feats, 'ai_prob': p.get('ai_prob', 0.0), 'tp': p.get('tp', float('inf')), 'sl': p.get('sl', 0)})
                    except Exception: pass
        except Exception: pass
            
    if not trades: return
    syms = list(set([t['symbol'] for t in trades]))
    
    try:
        dfs = []
        logger.info(f"⏳ 启动回测引擎：正在拉取历史轨迹数据...")
        for i in range(0, len(syms), 40):
            chunk = syms[i:i + 40]
            for attempt in range(3):
                try:
                    chunk_df = yf.download(chunk, period="1y", progress=False, threads=False, timeout=15)
                    if not chunk_df.empty: 
                        if len(chunk) == 1 and not isinstance(chunk_df.columns, pd.MultiIndex): chunk_df.columns = pd.MultiIndex.from_product([chunk_df.columns, chunk])
                        dfs.append(chunk_df); break
                    else: time.sleep((attempt + 1) * 2.0)
                except Exception: time.sleep((attempt + 1) * 3.0)
            if i + 40 < len(syms): time.sleep(random.uniform(2.0, 3.5))

        if not dfs: return
        df_all = pd.concat(dfs, axis=1)

        if isinstance(df_all.columns, pd.MultiIndex): df_c, df_o, df_h, df_l = df_all['Close'], df_all['Open'], df_all['High'], df_all['Low']
        else: df_c, df_o, df_h, df_l = df_all[['Close']].rename(columns={'Close': syms[0]}), df_all[['Open']].rename(columns={'Open': syms[0]}), df_all[['High']].rename(columns={'High': syms[0]}), df_all[['Low']].rename(columns={'Low': syms[0]})
            
        df_c.index, df_o.index, df_h.index, df_l.index = df_c.index.strftime('%Y-%m-%d'), df_o.index.strftime('%Y-%m-%d'), df_h.index.strftime('%Y-%m-%d'), df_l.index.strftime('%Y-%m-%d')
    except Exception: return
    
    stats_period, factor_rets, trades_with_ret, mae_mfe_records = {'T+1': [], 'T+3': [], 'T+5': []}, {}, [], {'T+1': [], 'T+3': [], 'T+5': []}
    ai_filtered_wins, ai_filtered_total, transformer_X, transformer_Y, cached_inds = 0, 0, [], [], {}
    
    for t in trades:
        sym, r_dt, initial_sl, tp_price = t['symbol'], t['date'], t.get('sl', 0) if not pd.isna(t.get('sl', 0)) else 0, t.get('tp', float('inf')) if not pd.isna(t.get('tp', float('inf'))) else float('inf')
        if sym not in df_c.columns or sym not in df_o.columns: continue
        valid = df_c.index[df_c.index >= r_dt]
        if len(valid) == 0: continue
        
        e_idx = df_c.index.get_loc(valid[0])
        if e_idx + 1 >= len(df_c): continue
        e_px_raw, prev_c_px = df_o.iloc[e_idx + 1][sym], df_c.iloc[e_idx][sym]
        if pd.isna(e_px_raw) or e_px_raw <= 0: continue
        
        gap_up = (e_px_raw - prev_c_px) / (prev_c_px + 1e-10)
        entry_cost = e_px_raw * (1 + Config.Params.SLIPPAGE * 3 + Config.Params.COMMISSION) if gap_up > 0.03 else e_px_raw * (1 + Config.Params.SLIPPAGE + Config.Params.COMMISSION)
        trail_distance = max(e_px_raw * 0.02, e_px_raw - initial_sl) if initial_sl > 0 else e_px_raw * 0.05
        
        for d in [1, 3, 5]:
            if e_idx + d < len(df_c):
                exit_revenue, highest_seen_px, dynamic_sl, max_high_during_trade, min_low_during_trade = None, e_px_raw, initial_sl, e_px_raw, e_px_raw
                
                for i in range(1, d + 1):
                    check_idx = e_idx + i
                    day_open, day_low, day_high, day_close = df_o.iloc[check_idx][sym], df_l.iloc[check_idx][sym], df_h.iloc[check_idx][sym], df_c.iloc[check_idx][sym]
                    if pd.isna(day_open) or pd.isna(day_low) or pd.isna(day_high): continue
                    
                    max_high_during_trade, min_low_during_trade = max(max_high_during_trade, day_high), min(min_low_during_trade, day_low)
                    
                    if dynamic_sl > 0 and day_open < dynamic_sl: exit_revenue = day_open * (1 - Config.Params.SLIPPAGE * 5 - Config.Params.COMMISSION); break
                    if dynamic_sl > 0 and day_low <= dynamic_sl: exit_revenue = dynamic_sl * (1 - Config.Params.SLIPPAGE * 3 - Config.Params.COMMISSION); break
                    if tp_price > 0 and day_high >= tp_price: exit_revenue = tp_price * (1 - Config.Params.SLIPPAGE - Config.Params.COMMISSION); break
                        
                    highest_seen_px = max(highest_seen_px, day_high)
                    dynamic_sl = max(dynamic_sl, highest_seen_px - trail_distance)
                    
                    if i == d: exit_revenue = day_close * (1 - (Config.Params.SLIPPAGE * 5 if abs((day_close - (df_c.iloc[check_idx-1][sym] if check_idx > e_idx else e_px_raw)) / (df_c.iloc[check_idx-1][sym] if check_idx > e_idx else e_px_raw + 1e-10)) > 0.15 else Config.Params.SLIPPAGE) - Config.Params.COMMISSION)
                
                if exit_revenue is not None:
                    ret = (exit_revenue - entry_cost) / entry_cost * (0.5 if gap_up > 0.03 else 1.0)
                    
                    if TRANSFORMER_AVAILABLE:
                        try:
                            if sym not in cached_inds: cached_inds[sym] = calculate_indicators(df_all.xs(sym, level=1, axis=1) if isinstance(df_all.columns, pd.MultiIndex) else pd.DataFrame({'Close': df_c[sym], 'Open': df_o[sym], 'High': df_h[sym], 'Low': df_l[sym], 'Volume': 1000000}))
                            ind_df = cached_inds[sym]
                            valid_dates = ind_df.index[ind_df.index <= r_dt]
                            if len(valid_dates) > 0:
                                t_idx = np.where(ind_df.index.get_loc(valid_dates[-1]))[0][-1] if isinstance(ind_df.index.get_loc(valid_dates[-1]), np.ndarray) else (ind_df.index.get_loc(valid_dates[-1]).stop - 1 if isinstance(ind_df.index.get_loc(valid_dates[-1]), slice) else ind_df.index.get_loc(valid_dates[-1]))
                                seq = _get_transformer_seq(ind_df, end_idx=t_idx + 1)
                                if not np.all(seq == 0): transformer_X.append(seq); transformer_Y.append(ret if d == 3 else 0.0) 
                        except Exception: pass
                    
                    mae_mfe_records[f'T+{d}'].append({'ret': ret, 'mfe': (max_high_during_trade - entry_cost) / entry_cost, 'mae': (min_low_during_trade - entry_cost) / entry_cost})
                    stats_period[f'T+{d}'].append(ret)
                    
                    if d == 3:
                        factor_list = t.get('factors', [])
                        if not factor_list:
                            for sig_txt in t.get('signals', []):
                                m = re.search(r'\[(.*?)\]', sig_txt)
                                if m: factor_list.append(m.group(1).split(" ")[0])
                        for f_name in factor_list: factor_rets.setdefault(f"[{f_name}]", []).append(ret)
                            
                        if t.get('ai_prob', 0.0) >= 0.50:  
                            ai_filtered_total += 1
                            if ret > 0: ai_filtered_wins += 1
                            
                        if t.get('ml_features', {}): trades_with_ret.append({'date': t['date'], 'vix': t['vix'], 'cred': t['cred'], 'term': t['term'], 'pcr': t['pcr'], 'ml_features': t.get('ml_features', {}), 'factors': factor_list, 'ret': ret})
    
    feature_importances_dict, meta_weights_dict, factor_ic_report, trade_df = {}, {}, {}, pd.DataFrame(trades_with_ret)
    
    if len(trade_df) >= 30:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.linear_model import LogisticRegression
            import pickle
            
            trade_df = trade_df.sort_values('date')
            X_all_df = pd.DataFrame(trade_df['ml_features'].tolist()).fillna(0.0)
            for f in Config.ALL_FACTORS:
                if f not in X_all_df.columns: X_all_df[f] = 0.0
            X_all_df = X_all_df[Config.ALL_FACTORS]
            
            y_all_cont, dates_all = trade_df['ret'].values, trade_df['date'].values
            ic_records = {f: [] for f in Config.ALL_FACTORS}
            
            for d_str in np.unique(dates_all):
                mask = dates_all == d_str
                if np.sum(mask) < 5 or np.std(y_all_cont[mask]) < 1e-6: continue
                for factor_name in Config.ALL_FACTORS:
                    f_vals = X_all_df[mask][factor_name].values
                    ic, _ = stats.spearmanr(f_vals, y_all_cont[mask]) if np.std(f_vals) >= 1e-6 else (np.nan, 0)
                    ic_records[factor_name].append(float(ic) if not np.isnan(ic) else 0.0)
            
            active_candidates = []
            for f in Config.ALL_FACTORS:
                ic_arr = np.array(ic_records[f])
                mean_ic, ir, t_stat = (float(np.mean(ic_arr)), float(np.mean(ic_arr)) / (float(np.std(ic_arr)) + 1e-10), float(np.mean(ic_arr)) / ((float(np.std(ic_arr)) + 1e-10) / np.sqrt(len(ic_arr)))) if len(ic_arr) > 5 else (0.0, 0.0, 0.0)
                factor_ic_report[f] = {'mean_ic': mean_ic, 'ir': ir, 't_stat': t_stat}
                if abs(t_stat) > Config.Params.MIN_T_STAT: active_candidates.append((f, abs(t_stat)))
            
            active_factors_list = [x[0] for x in sorted(active_candidates, key=lambda x: x[1], reverse=True)]
            if len(active_factors_list) < 5: active_factors_list = [x[0] for x in sorted(factor_ic_report.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True)[:5]]
                
            scaler = RobustScaler()
            X_scaled_df = pd.DataFrame(scaler.fit_transform(X_all_df[active_factors_list]), columns=active_factors_list)
            active_A, active_B = [f for f in active_factors_list if f in Config.GROUP_A_FACTORS], [f for f in active_factors_list if f in Config.GROUP_B_FACTORS]
            X_A, X_B = X_scaled_df[active_A].values if active_A else np.zeros((len(X_scaled_df), 0)), X_scaled_df[active_B].values if active_B else np.zeros((len(X_scaled_df), 0))
            
            y_all_class, vix_all, cred_all, term_all, pcr_all = (y_all_cont > 0.015).astype(int), trade_df['vix'].values / 20.0, trade_df['cred'].values, trade_df['term'].values - 1.0, trade_df['pcr'].values - 1.0
            oof_pred_A, oof_pred_B = np.zeros(len(y_all_class)), np.zeros(len(y_all_class))
            lgbm_params = dict(n_estimators=60, max_depth=3, learning_rate=0.05, class_weight='balanced', random_state=42)
            
            for train_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_scaled_df):
                if active_A: oof_pred_A[val_idx] = LGBMClassifier(**lgbm_params).fit(X_A[train_idx], y_all_class[train_idx]).predict_proba(X_A[val_idx])[:, 1]
                else: oof_pred_A[val_idx] = 0.5
                if active_B: oof_pred_B[val_idx] = LGBMClassifier(**lgbm_params).fit(X_B[train_idx], y_all_class[train_idx]).predict_proba(X_B[val_idx])[:, 1]
                else: oof_pred_B[val_idx] = 0.5
                
            valid_meta_idx = np.where(oof_pred_A > 0)[0]
            if len(valid_meta_idx) > 10:
                meta_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42).fit(np.column_stack([oof_pred_A[valid_meta_idx], oof_pred_B[valid_meta_idx], vix_all[valid_meta_idx], cred_all[valid_meta_idx], term_all[valid_meta_idx], pcr_all[valid_meta_idx]]), y_all_class[valid_meta_idx])
                exp_wA, exp_wB = np.exp(meta_clf.coef_[0][0]), np.exp(meta_clf.coef_[0][1])
                meta_weights_dict['Group_A'], meta_weights_dict['Group_B'] = exp_wA / (exp_wA + exp_wB), exp_wB / (exp_wA + exp_wB)
            else:
                meta_clf = LogisticRegression(class_weight='balanced').fit(np.column_stack([oof_pred_A, oof_pred_B, vix_all, cred_all, term_all, pcr_all]), y_all_class)

            final_base_A, final_base_B = LGBMClassifier(**lgbm_params).fit(X_A, y_all_class) if active_A else None, LGBMClassifier(**lgbm_params).fit(X_B, y_all_class) if active_B else None
            with open(Config.MODEL_FILE, 'wb') as f: pickle.dump({'version': Config.MODEL_VERSION, 'active_factors': active_factors_list, 'active_A': active_A, 'active_B': active_B, 'scaler': scaler, 'base_A': final_base_A, 'base_B': final_base_B, 'meta': meta_clf, 'n_features_in_': len(active_factors_list)}, f)
            
            combined_imp = np.zeros(len(Config.ALL_FACTORS))
            if final_base_A and hasattr(final_base_A, 'feature_importances_'):
                for i, active_f in enumerate(active_A): combined_imp[Config.ALL_FACTORS.index(active_f)] += (final_base_A.feature_importances_[i] / (np.sum(final_base_A.feature_importances_) + 1e-10)) * meta_weights_dict.get('Group_A', 0.5)
            if final_base_B and hasattr(final_base_B, 'feature_importances_'):
                for i, active_f in enumerate(active_B): combined_imp[Config.ALL_FACTORS.index(active_f)] += (final_base_B.feature_importances_[i] / (np.sum(final_base_B.feature_importances_) + 1e-10)) * meta_weights_dict.get('Group_B', 0.5)
            for i, factor in enumerate(Config.ALL_FACTORS):
                if combined_imp[i] > 0: feature_importances_dict[factor] = float(combined_imp[i])
        except ImportError: pass
        except Exception: pass
            
    res = {}
    for p, r in stats_period.items():
        if not r: continue
        ret_arr = np.array(r)
        win_returns, loss_returns = ret_arr[ret_arr > 0], ret_arr[ret_arr < 0]
        
        max_cons_loss, curr_cons_loss = 0, 0
        for ret in ret_arr:
            if ret < 0: curr_cons_loss += 1; max_cons_loss = max(max_cons_loss, curr_cons_loss)
            else: curr_cons_loss = 0
                
        res_data = {'win_rate': len(win_returns) / len(ret_arr), 'avg_ret': np.mean(ret_arr), 'sharpe': np.mean(ret_arr) / (np.std(ret_arr) + 1e-10), 'worst_trade': np.min(ret_arr), 'total_trades': len(r), 'profit_factor': np.sum(win_returns) / abs(np.sum(loss_returns)) if len(loss_returns) > 0 else 99.0, 'max_cons_loss': max_cons_loss, 'avg_win_mae': np.mean([rec['mae'] for rec in mae_mfe_records[p] if rec['ret'] > 0]) if [rec['mae'] for rec in mae_mfe_records[p] if rec['ret'] > 0] else 0.0}
        if p == 'T+3' and ai_filtered_total > 0: res_data['ai_win_rate'] = ai_filtered_wins / ai_filtered_total
        res[p] = res_data

    f_res = {}
    for t, r in factor_rets.items():
        if len(r) >= 2:
            win_returns, loss_returns = np.array(r)[np.array(r) > 0], np.array(r)[np.array(r) < 0]
            f_res[t] = {'win_rate': len(win_returns)/len(r), 'avg_ret': sum(r)/len(r), 'count': len(r), 'profit_factor': np.sum(win_returns) / abs(np.sum(loss_returns)) if len(loss_returns) > 0 else 99.0}
            
    attr_report = {}
    if len(trades_with_ret) >= 30:
        adv_factors = ["FFT多窗共振(动能)", "稳健赫斯特(Hurst)", "VPT量价共振", "CVD筹码净流入", "量子概率云(KDE)", "Amihud非流动性(冲击成本)", "波动率风险溢价(VRP)"]
        attr_df = pd.DataFrame([{'ret': tr['ret'], **{f: 1 if f in tr.get('factors', []) else 0 for f in adv_factors + ["MACD金叉"]}} for tr in trades_with_ret])
        for f in adv_factors:
            trig, not_trig = attr_df[attr_df[f] == 1], attr_df[attr_df[f] == 0]
            attr_report[f] = {'premium_bps': float((trig['ret'].median() - not_trig['ret'].median()) * 10000) if len(trig) > 0 and len(not_trig) > 0 else 0.0, 'corr_with_baseline': float(attr_df[f].corr(attr_df["MACD金叉"]) if len(attr_df) > 1 and not pd.isna(attr_df[f].corr(attr_df["MACD金叉"])) else 0.0), 'trigger_rate': float(len(trig) / len(attr_df)) if len(attr_df) > 0 else 0.0}

    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump({"overall": res, "factors": f_res, "xai_importances": feature_importances_dict, "meta_weights": meta_weights_dict, "attribution": attr_report, "factor_ic": factor_ic_report}, f, indent=4)
    
    report_md = [f"# 📈 自动量化战报与 AI 透视\n**更新:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n## ⚔️ 核心表现评估\n| 周期 | 原始胜率 | ⚡代谢演化过滤 | 均收益 | 盈亏比 | Sharpe | 胜单平均抗压(MAE) | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for p in ['T+1', 'T+3', 'T+5']:
        d = res.get(p, {'win_rate':0,'avg_ret':0,'profit_factor':0,'sharpe':0,'avg_win_mae':0,'max_cons_loss':0,'total_trades':0})
        report_md.append(f"| {p} | {d['win_rate']*100:.1f}% | {f'**{d.get('ai_win_rate', 0.0)*100:.1f}%**' if 'ai_win_rate' in d else '-'} | {d['avg_ret']*100:+.2f}% | {d['profit_factor']:.2f} | {d['sharpe']:.2f} | {d['avg_win_mae']*100:.1f}% | {d['total_trades']} |")
    
    if factor_ic_report:
        report_md.append("\n## 🧬 因子动物园 Rank IC 淘汰赛排行榜 (Top 10)\n| 因子特征 | 均值 IC | T-Statistic (绝对值) | 状态 |\n|:---|:---:|:---:|:---:|")
        active_names = [x[0] for x in sorted(factor_ic_report.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True) if abs(x[1]['t_stat']) > Config.Params.MIN_T_STAT]
        if len(active_names) < 5: active_names = [x[0] for x in sorted(factor_ic_report.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True)[:5]]
        for tag, data in sorted(factor_ic_report.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True)[:10]: report_md.append(f"| {tag} | {data['mean_ic']:.4f} | {abs(data['t_stat']):.2f} | {'🟢 激活' if tag in active_names else '🧊 冷宫'} |")

    if feature_importances_dict:
        report_md.append("\n## 🧠 XAI (解释性人工智能) - 驱动当期市场的核心因子权重\n| 因子特征 | AI 分配重要性 (元学习器赋权) |\n|:---|:---:|")
        for tag, imp in sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True): report_md.append(f"| {tag} | {imp*100:.1f}% |")
    
    if attr_report:
        report_md.append(f"\n## 🔬 高级微观因子归因仪表盘 (Alpha Attribution)\n| 高级因子 | 纯因子溢价 (BPS) | 与传统动能耦合度 (Corr) | 触发频率 | 归因诊断 |\n|:---|:---:|:---:|:---:|:---:|")
        for f, data in attr_report.items(): report_md.append(f"| {f} | {data['premium_bps']:+.1f} bps | {data['corr_with_baseline']:.2f} | {data['trigger_rate']*100:.1f}% | {'⚠️ 负溢价' if data['premium_bps'] < 0 else ('⚖️ 高度耦合' if data['corr_with_baseline'] > 0.6 else ('💎 纯净 Alpha' if data['premium_bps'] > 50 and data['corr_with_baseline'] < 0.4 else '✅ 有效增益'))} |")

    if TRANSFORMER_AVAILABLE and len(transformer_X) >= 64:
        logger.info(f"🧠 [架构解耦] 已捕获 {len(transformer_X)} 笔三元组时序切片，正压入训练数据共享缓冲区...")
        try:
            X_arr, Y_arr, buf_path = np.array(transformer_X, dtype=np.float32), np.array(transformer_Y, dtype=np.float32), os.path.join(Config.DATA_DIR, "training_buffer.npz")
            if os.path.exists(buf_path):
                try: existing = np.load(buf_path); X_arr, Y_arr = np.concatenate([existing['X'], X_arr]), np.concatenate([existing['Y'], Y_arr])
                except Exception: pass
            temp_buf_path = buf_path + ".tmp"
            np.savez(temp_buf_path, X=X_arr, Y=Y_arr); os.replace(temp_buf_path, buf_path)
            report_md.append(f"\n## 🌌 深度学习右脑 (Transformer) 数据积淀战报\n- **状态**: ✅ 已将 {len(transformer_X)} 笔新样本压入 `.npz` 缓冲区\n- **架构声明**: 训练进程已被物理剥离，请稍后执行 `python train_transformer.py` 唤醒独立算力节点进行闭环代谢。")
            logger.info(f"✅ 样本集成功落盘 (当前总样本池: {len(X_arr)} 笔)。")
        except Exception as e: logger.error(f"⚠️ 缓冲区落盘失败: {e}")

    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_md))

    send_alert("策略终极回测战报 (代谢进化版)", "\n".join(["### 📊 **机构级回测报表 (含代谢淘汰赛)**"] + [f"- **{p}:** 原始胜率 {d['win_rate']*100:.1f}%{f' | ⚡代谢演化过滤: **{d.get('ai_win_rate', 0.0)*100:.1f}%**' if 'ai_win_rate' in d else ''} | 盈亏比 {d['profit_factor']:.2f}" for p, d in res.items()]))

def _decompose_and_perturb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_phase, df_noise = df.copy(), df.copy()
    c = df['Close']
    trend, high_freq_noise, seasonal = c.ewm(span=60, adjust=False).mean(), c - c.ewm(span=5, adjust=False).mean(), c.ewm(span=5, adjust=False).mean() - c.ewm(span=60, adjust=False).mean()

    angles = np.random.uniform(0, 2 * np.pi, len(np.fft.rfft(seasonal.values)))
    angles[0] = 0.0 
    if len(seasonal.values) % 2 == 0: angles[-1] = 0.0 
    
    shifted_seasonal = pd.Series(np.fft.irfft(np.fft.rfft(seasonal.values) * np.exp(1j * angles), n=len(seasonal.values)), index=seasonal.index)
    c_phase, c_noise = trend + shifted_seasonal + high_freq_noise, trend + seasonal + high_freq_noise * 1.5

    ratio_phase, ratio_noise = c_phase / (c + 1e-10), c_noise / (c + 1e-10)
    df_phase['Close'], df_phase['Open'], df_phase['High'], df_phase['Low'] = c_phase, df['Open']*ratio_phase, df['High']*ratio_phase, df['Low']*ratio_phase
    df_noise['Close'], df_noise['Open'], df_noise['High'], df_noise['Low'] = c_noise, df['Open']*ratio_noise, df['High']*ratio_noise, df['Low']*ratio_noise

    return df_phase, df_noise

def run_synthetic_stress_test() -> None:
    logger.info("🌪️ 启动合成数据对抗压测引擎 (Synthetic Adversarial Test)...")
    xai_weights = {}
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                xai_data = json.load(f).get("xai_importances", {})
                if xai_data:
                    avg_imp = 1.0 / len(Config.ALL_FACTORS)
                    for tag, imp in xai_data.items(): xai_weights[tag] = 0.0 if imp < avg_imp * 0.25 else max(0.5, min(3.0, float(imp) / avg_imp))
    except Exception: pass
    
    metrics = {'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}}
    market_ctx = MarketContext(regime="bull", regime_desc="", w_mul=1.0, xai_weights=xai_weights, vix_current=18.0, vix_desc="", vix_scalar=1.0, max_risk=0.015, macro_gravity=False, is_credit_risk_high=False, vix_inv=False, qqq_df=pd.DataFrame(), macro_data={'spy': pd.DataFrame(), 'tlt': pd.DataFrame(), 'dxy': pd.DataFrame()}, total_market_exposure=1.0, health_score=1.0, pain_warning="", dynamic_min_score=8.0)
    
    def _stress_test_worker(sym):
        df_raw = safe_get_history(sym, "5y", "1d", fast_mode=True)
        if len(df_raw) < 500: return None
        df_phase, df_noise = _decompose_and_perturb(df_raw)
        
        local_metrics = {'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}}
        
        for env_name, df_env in {'Original': df_raw, 'Phase_Chaos': df_phase, 'Noise_Explosion': df_noise}.items():
            try:
                df_w, df_m = df_env.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna(), df_env.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                df_ind = calculate_indicators(df_env)
                
                for i in range(len(df_ind) - 252, len(df_ind) - 5):
                    curr, prev = df_ind.iloc[i], df_ind.iloc[i-1]
                    if (curr['ATR'] / curr['Close'] > 0.15) or (pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']): continue
                    is_vol, swing_high_10 = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open']), df_ind['High'].iloc[i-10:i].max() if i >= 10 else curr['High']
                    stock_data = StockData(sym, df_ind.iloc[:i+1], df_w, df_m, pd.DataFrame(), curr, prev, is_vol, swing_high_10)
                    
                    if _evaluate_omni_matrix(stock_data, market_ctx, _extract_complex_features(stock_data, market_ctx), AltData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))[0] >= Config.Params.MIN_SCORE_THRESHOLD:
                        ret = (df_ind['Close'].iloc[i+4] - df_ind['Open'].iloc[i+1]) / df_ind['Open'].iloc[i+1]
                        local_metrics[env_name]['trades'] += 1
                        local_metrics[env_name]['ret_sum'] += ret
                        if ret > 0: local_metrics[env_name]['wins'] += 1
            except Exception: pass
        return local_metrics

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.Params.MAX_WORKERS) as executor:
        for res in [future.result() for future in concurrent.futures.as_completed([executor.submit(_stress_test_worker, sym) for sym in get_filtered_watchlist(max_stocks=30)[:20]]) if future.result()]:
            for env_name in metrics: metrics[env_name]['trades'] += res[env_name]['trades']; metrics[env_name]['wins'] += res[env_name]['wins']; metrics[env_name]['ret_sum'] += res[env_name]['ret_sum']

    def _fmt(m): return f"交易笔数: {m['trades']} | 胜率: {(m['wins']/m['trades'])*100 if m['trades']>0 else 0:.1f}% | 单笔均利: {(m['ret_sum']/m['trades'])*100 if m['trades']>0 else 0:.2f}%"
    orig, phase, noise = metrics['Original'], metrics['Phase_Chaos'], metrics['Noise_Explosion']
    report_lines = ["### 🌪️ **对抗样本压力测试报告 (Adversarial Stress Test)**\n", f"**🟢 原始环境 (Original History)**\n- {_fmt(orig)}\n", f"**🌌 平行宇宙A：相位错乱 (Phase Shifted)**\n- {_fmt(phase)}"]
    
    report_lines.append("- 📉 **归因诊断**: 胜率急剧崩塌！当前模型对 FFT 等周期相位产生了**严重虚假依赖**。" if (orig['wins']/orig['trades'] if orig['trades']>0 else 0) - (phase['wins']/phase['trades'] if phase['trades']>0 else 0) > 0.15 else "- 🛡️ **归因诊断**: 胜率坚挺。模型未陷入周期的死记硬背，对时序变异具备极强鲁棒性！")
    report_lines.append(f"\n**🌋 平行宇宙B：噪音爆炸 (Noise Amplified 1.5x)**\n- {_fmt(noise)}")
    send_alert("终极实战压测报告", "\n".join(report_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if m == "matrix": run_tech_matrix()
    elif m == "gateway": run_gateway()
    elif m == "backtest": run_backtest_engine()
    elif m == "stress": run_synthetic_stress_test()
    elif m == "test": send_alert("连通性测试", "全维宏观 Meta 跃迁完成！系统已开启 6 维上帝视野。")
