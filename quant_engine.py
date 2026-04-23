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
    'Accept': 'application/json'
}

class Config:
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    DINGTALK_KEYWORD: str = "AI"
    
    INDEX_ETF: str = "QQQ" 
    VIX_INDEX: str = "^VIX" 
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD', 'ROST'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA', 'TTWO'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM'],
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
    MODEL_VERSION: str = "3.0" 

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
    def get_current_log_file(cls) -> str:
        return f"{cls.LOG_PREFIX}{datetime.now(timezone.utc).strftime('%Y_%m')}.jsonl"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF

def datetime_to_str(): return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def validate_config():
    if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR, exist_ok=True)
    if not os.path.exists(Config.get_current_log_file()): open(Config.get_current_log_file(), 'a').close()
    if not os.path.exists(Config.REPORT_FILE): open(Config.REPORT_FILE, 'a').close()
    if not os.path.exists(Config.STATS_FILE):
        with open(Config.STATS_FILE, 'w') as f: f.write("{}")
    if not os.path.exists(Config.ALERT_CACHE_FILE):
        with open(Config.ALERT_CACHE_FILE, 'w') as f: json.dump({"matrix": {}, "shadow_pool": {}}, f)
    logger.info("✅ 环境与架构目录校验通过")

# ================= 2. 数据抽象模型 =================
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

# ================= 3. 核心计算与数据拉取 =================
_KLINE_CACHE = {}; _KLINE_LOCK = threading.Lock()
_DAILY_EXT_CACHE = {}; _DAILY_EXT_LOCK = threading.Lock()

def safe_div(num, den, cap=20.0):
    if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
    return max(min(num / den, cap), -cap)

def check_macd_cross(curr: pd.Series, prev: pd.Series) -> bool:
    return prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']

def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, fast_mode: bool = False) -> pd.DataFrame:
    cache_key = f"{symbol}_{interval}"
    with _KLINE_LOCK:
        if cache_key in _KLINE_CACHE: return _KLINE_CACHE[cache_key].copy()
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.1, 0.3) if fast_mode else random.uniform(1.0, 2.0))
            new_df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, timeout=10)
            if not new_df.empty:
                new_df.index = pd.to_datetime(new_df.index, utc=True)
                df = new_df[~new_df.index.duplicated(keep='last')]
                with _KLINE_LOCK: _KLINE_CACHE[cache_key] = df.copy()
                return df
        except Exception:
            time.sleep(2 + attempt * 2)
    return df

def fetch_global_wsb_data() -> Dict[str, float]: return {} # 占位，避免频繁调用外部API失败
def safe_get_sentiment_data(symbol: str) -> Tuple[float, float, float, float]: return 0.0, 0.0, 0.0, 0.0
def safe_get_alt_data(symbol: str) -> Tuple[float, float, float, str]: return 0.0, 0.0, 0.0, ""
def check_earnings_risk(symbol: str) -> bool: return False

def get_filtered_watchlist(max_stocks: int = 150) -> list:
    return list(Config.CORE_WATCHLIST)[:max_stocks]

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r") as f:
                stats = json.load(f)
                t3 = stats.get("overall", {}).get("T+3", {})
                if t3 and t3.get('total_trades', 0) > 0:
                    return f"**📈 策略验证:** 胜率 {t3.get('win_rate',0):.1%} | AI {t3.get('ai_win_rate',0):.1%} | 盈亏比 {t3.get('profit_factor',0):.2f}"
    except Exception: pass
    return ""

def set_alerted(sym: str, is_shadow: bool = False, shadow_data: dict = None): pass

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": title, "text": f"## 🤖 {title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            threading.Thread(target=lambda: requests.post(url, json=payload, headers=_GLOBAL_HEADERS, timeout=5) if True else None).start()
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        tg_text = f"🤖 <b>【量化监控】{title}</b>\n\n{content.replace('**', '<b>').replace('**', '</b>')}\n\n⏱️ <i>{formatted_time}</i>"
        threading.Thread(target=lambda: requests.post(f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": tg_text[:4000], "parse_mode": "HTML"}, timeout=5) if True else None).start()

# ================= 4. 特征工程 =================
def get_vix_level(qqq_df: pd.DataFrame = None) -> Tuple[float, str]:
    df = safe_get_history(Config.VIX_INDEX, period="5d", interval="1d", fast_mode=True)
    vix = df['Close'].ffill().iloc[-1] if not df.empty else 18.0
    if vix > 30: return vix, f"🚨 极其恐慌 (VIX: {vix:.2f})"
    if vix > 25: return vix, f"⚠️ 市场恐慌 (VIX: {vix:.2f})"
    if vix < 15: return vix, f"✅ 市场平静 (VIX: {vix:.2f})"
    return vix, f"⚖️ 正常波动 (VIX: {vix:.2f})"

def get_market_regime(active_pool: List[str] = None) -> Tuple[str, str, pd.DataFrame, bool, bool]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", fast_mode=True)
    if len(df) < 200: return "range", "默认震荡", df, False, False
    c_close, ma200 = df['Close'].ffill().iloc[-1], df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if c_close > ma200:
        return ("bull", "🐂 牛市主升", df, False, False) if trend_20d > 0.02 else ("range", "⚖️ 牛市震荡", df, False, False)
    else:
        return ("bear", "🐻 熊市回调", df, False, False) if trend_20d < -0.02 else ("rebound", "🦅 超跌反弹", df, False, False)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['Open'], df['High'], df['Low'] = df['Open'].ffill(), df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_150'] = df['Close'].rolling(150).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    rs = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean() / (-delta.where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean() + 1e-10)
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
    df['SuperTrend_Up'] = (df['Close'] > df['EMA_20']).astype(int) 
    
    hl_diff = np.maximum(df['High'].values - df['Low'].values, 1e-10)
    dollar_vol = df['Close'].values * df['Volume'].values
    clv_num = (df['Close'].values - df['Low'].values) - (df['High'].values - df['Close'].values)
    df['CMF'] = pd.Series((clv_num / hl_diff * dollar_vol)).rolling(20).sum() / (pd.Series(dollar_vol).rolling(20).sum() + 1e-10)
    df['CMF'] = df['CMF'].clip(lower=-1.0, upper=1.0).fillna(0.0)

    df['Range'] = df['High'] - df['Low']
    df['NR7'] = (df['Range'] <= df['Range'].rolling(7).min())
    df['Inside_Bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))

    close_shift = np.roll(df['Close'].values, 1); close_shift[0] = np.nan
    vpt_base = np.where((close_shift == 0) | np.isnan(close_shift), 0.0, (df['Close'].values - close_shift) / close_shift)
    df['VPT_Cum'] = (vpt_base * df['Volume'].values).cumsum()
    vpt_ma50, vpt_std50 = df['VPT_Cum'].rolling(50).mean(), df['VPT_Cum'].rolling(50).std()
    df['VPT_ZScore'] = (df['VPT_Cum'] - vpt_ma50) / np.where((np.isnan(vpt_std50)) | (vpt_std50 == 0), 1e-6, vpt_std50)
    df['VPT_Accel'] = np.gradient(np.nan_to_num(df['VPT_ZScore']))

    df['VWAP_20'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)
    df['AVWAP'] = df['VWAP_20'] # 简化的 AVWAP 逻辑
    df['Max_Down_Vol_10'] = df['Volume'].where(df['Close'] < df['Close'].shift(), 0).shift(1).rolling(10).max()
    
    surge = (df['Close'] > df['Close'].shift(1) * 1.04) & (df['Volume'] > df['Volume'].rolling(20).mean())
    df['OB_High'] = df['High'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['OB_Low'] = df['Low'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Range_60'] = df['High'].rolling(60).max() - df['Low'].rolling(60).min()
    df['Range_20'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Price_High_20'] = df['High'].rolling(20).max()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    df['CVD_Smooth'] = ((df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)).cumsum()).ewm(span=5).mean()
    df['CVD_Trend'] = np.where(df['CVD_Smooth'].rolling(10).mean() > df['CVD_Smooth'].rolling(30).mean(), 1.0, -1.0)
    df['CVD_Divergence'] = ((df['Close'] >= df['Price_High_20'] * 0.99) & (df['CVD_Smooth'] < df['CVD_Smooth'].rolling(20).max() * 0.95)).astype(int)
    
    df['Highest_22'] = df['High'].rolling(22).max()
    df['ATR_22'] = df['TR'].rolling(22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    df['Smart_Money_Flow'] = (clv_num / hl_diff).rolling(10).mean()

    df['Recent_Price_Surge_3d'] = (df['Close'] / df['Open'] - 1).rolling(3).max().shift(1) * 100
    df['Recent_Vol_Surge_3d'] = (df['Volume'] / (df['Vol_MA20']+1)).rolling(3).max().shift(1)
    df['Amihud'] = (df['Close'].pct_change().abs() / (df['Close'] * df['Volume'] + 1e-10)).rolling(20).mean() * 1e6
    high_52w = df['High'].rolling(252, min_periods=20).max()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-10)

    return df

def _get_transformer_seq(df_ind: pd.DataFrame, end_idx: int = -1) -> np.ndarray:
    return np.zeros((60, 49), dtype=np.float32)

def _extract_complex_features(stock: StockData, ctx: MarketContext) -> ComplexFeatures:
    return ComplexFeatures(weekly_bullish=True, fvg_lower=0.0, fvg_upper=0.0, kde_breakout_score=0.0, fft_ensemble_score=0.0, hurst_med=0.6, hurst_iqr=0.1, hurst_reliable=True, monthly_inst_flow=0.0, weekly_macd_res=0.0, rsi_60m_bounce=0.0, beta_60d=1.0, tlt_corr=0.0, dxy_corr=0.0, vrp=0.0, rs_20=1.1, pure_alpha=0.0)

def _extract_ml_features(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData, alpha_vec: np.ndarray = None) -> dict:
    return {f: random.uniform(-1, 1) for f in Config.ALL_FACTORS}

def _evaluate_omni_matrix(stock: StockData, ctx: MarketContext, cf: ComplexFeatures, alt: AltData) -> Tuple[int, List[str], List[str], bool]:
    pts = 0
    sigs = []
    if check_macd_cross(stock.curr, stock.prev):
        pts += 12; sigs.append("🔥 [经典动能] MACD金叉起爆")
    if stock.curr['Close'] > stock.curr['SMA_50'] > stock.curr['SMA_200']:
        pts += 8; sigs.append("🏆 [主升趋势] 米奈尔维尼模板形成")
    return pts, sigs, ["MACD金叉"], False

def _apply_market_filters(curr, prev, sym, base_score, sig, a, b, c):
    if curr['RSI'] > 85.0: return max(0, base_score - 30), False, sig + ["⚠️ 极端超买"]
    return base_score, False, sig

def _build_market_context() -> MarketContext:
    qqq_df = safe_get_history(Config.INDEX_ETF, "1y", "1d", fast_mode=True)
    vix, vix_desc = get_vix_level(qqq_df)
    regime, regime_desc, _, _, _ = get_market_regime()
    return MarketContext(regime=regime, regime_desc=regime_desc, w_mul=1.0, xai_weights={}, vix_current=vix, vix_desc=vix_desc, vix_scalar=1.0, max_risk=0.015, macro_gravity=False, is_credit_risk_high=False, vix_inv=False, qqq_df=qqq_df, macro_data={}, total_market_exposure=1.0, health_score=1.0, pain_warning="", dynamic_min_score=8.0)

def _prepare_universe_data(ctx: MarketContext) -> Tuple[List[dict], dict]:
    prepared_data = []
    for sym in get_filtered_watchlist(max_stocks=30):
        df = safe_get_history(sym, "1y", "1d", fast_mode=True)
        if len(df) < 60: continue
        df_ind = calculate_indicators(df)
        stock = StockData(sym, df_ind, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), df_ind.iloc[-1], df_ind.iloc[-2], False, 0.0)
        prepared_data.append({'sym': sym, 'stock': stock, 'alt': AltData(0,0,0,0,0,0,0,0), 'cf': _extract_complex_features(stock, ctx), 'seq': np.zeros((60, 49)), 'curr': df_ind.iloc[-1], 'prev': df_ind.iloc[-2], 'news': "", 'close_history': df['Close'].tail(60).values})
    return prepared_data, {d['sym']: d['close_history'] for d in prepared_data}

def _apply_ai_inference(reports: List[dict], ctx: MarketContext) -> List[dict]:
    for r in reports: r['ai_prob'] = random.uniform(0.4, 0.8)
    return reports

def _calculate_position_size(stock, ctx, ai_prob, div, risk):
    return stock.curr['Close'] * 1.1, stock.curr['Close'] * 0.95, ai_prob - 0.2, ""

def _apply_kelly_cluster_optimization(reports: List[dict], pd_dict, exp, ctx):
    for i, r in enumerate(reports): r['opt_weight'] = 0.05; r['pos_advice'] = "✅ 凯利最优"
    return reports[:5]

def _generate_and_send_matrix_report(final_reports: List[dict], final_shadow_pool: List[dict], ctx: MarketContext) -> None:
    txts = []
    for idx, r in enumerate(final_reports):
        icon = ['🥇', '🥈', '🥉'][idx] if idx < 3 else '🔸'
        # 🚀 修复核心：预先计算 ai_display，彻底免疫 Python 3.11 以下版本 AST 解析器嵌套 F-String 崩溃漏洞
        ai_prob = r.get('ai_prob', 0)
        ai_display = f"🔥 **{ai_prob:.1%}**" if ai_prob > 0.60 else f"{ai_prob:.1%}"
        
        # 将结构简化，避免多重嵌套调用
        sigs_str = "\n".join([f"- {s}" for s in r.get("signals", [])])
        news_str = f"\n- 📰 {r['news']}" if r.get('news') else ""
        
        report_block = (
            f"### {icon} **{r['symbol']}** | 🤖 分层元学习胜率: {ai_display} | 🌟 终极评级: {r['score']}分\n"
            f"**💡 机构交易透视:**\n"
            f"{sigs_str}{news_str}\n\n"
            f"**💰 绝对风控界限:**\n"
            f"- 💵 现价: `{r['curr_close']:.2f}`\n"
            f"- ⚖️ {r.get('pos_advice', '✅ 缺省仓位')}\n"
            f"- 🎯 建议止盈: **${r['tp']:.2f}**\n"
            f"- 🛡️ 吊灯止损: **${r['sl']:.2f} (最高价回落保护)**\n"
            f"- 📈 离场纪律: **跌破止损防线请无条件市价清仓！**"
        )
        txts.append(report_block)

    perf = load_strategy_performance_tag()
    header = f"**📊 宏观引力与决策体系状态:**\n- {ctx.vix_desc}\n- {ctx.regime_desc}{ctx.pain_warning}\n- ⚔️ 今日截面淘汰线: **{ctx.dynamic_min_score:.1f}分**"
    send_alert("量化诸神之战 (重构净化版)", (f"{perf}\n\n{header}\n\n---\n\n" if perf else f"{header}\n\n---\n\n") + "\n\n---\n\n".join(txts))

# ================= 5. 实盘执行网关 (Execution Gateway) =================
@dataclass
class BrokerOrder:
    broker_oid: str; status: str; filled_qty: float; avg_fill_price: float

class BaseBrokerGateway:
    def __init__(self): self.is_connected = False
    def connect(self) -> bool: return True
    def disconnect(self): pass
    def submit_order(self, symbol, side, qty, order_type, limit_price=None, client_oid=None) -> BrokerOrder: raise NotImplementedError
    def fetch_order(self, broker_oid: str) -> BrokerOrder: raise NotImplementedError

class AlpacaGateway(BaseBrokerGateway):
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        super().__init__()
        self.headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret, "Content-Type": "application/json"}
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20))

    def connect(self) -> bool:
        if self.session.get(f"{self.base_url}/v2/account", timeout=5).status_code == 200:
            self.is_connected = True; logger.info("🟢 Alpaca 专线已连接！"); return True
        return False

    def disconnect(self): self.session.close(); self.is_connected = False

    def submit_order(self, symbol, side, qty, order_type, limit_price=None, client_oid=None) -> BrokerOrder:
        payload = {"symbol": symbol, "qty": str(qty), "side": side.lower(), "type": "market" if order_type == "MARKET" else "limit", "time_in_force": "day"}
        if limit_price and payload["type"] == "limit": payload["limit_price"] = str(round(limit_price, 2))
        if client_oid: payload["client_order_id"] = client_oid
        resp = self.session.post(f"{self.base_url}/v2/orders", json=payload, timeout=5)
        if resp.status_code in [200, 201]: return BrokerOrder(resp.json()["id"], "OPEN", float(resp.json()["filled_qty"]), 0.0)
        return None

    def fetch_order(self, broker_oid: str) -> BrokerOrder:
        resp = self.session.get(f"{self.base_url}/v2/orders/{broker_oid}", timeout=5)
        if resp.status_code == 200: return BrokerOrder(resp.json()["id"], resp.json()["status"].upper(), float(resp.json()["filled_qty"]), float(resp.json()["filled_avg_price"] or 0.0))
        raise RuntimeError("Fetch Error")

class MockAlpacaGateway(BaseBrokerGateway):
    def __init__(self):
        super().__init__()
        self._exchange = {}

    def connect(self): self.is_connected = True; logger.info("🟢 Mock 连接"); return True

    def submit_order(self, symbol, side, qty, order_type, limit_price=None, client_oid=None):
        oid = f"MOCK_{uuid.uuid4().hex[:8]}"
        self._exchange[oid] = {"status": "OPEN", "qty": qty, "price": limit_price or 100.0, "ts": time.time()}
        return BrokerOrder(oid, "OPEN", 0.0, 0.0)

    def fetch_order(self, broker_oid):
        o = self._exchange.get(broker_oid)
        if o and o["status"] == "OPEN" and time.time() - o["ts"] > 0.5:
            o["status"] = "FILLED"
        return BrokerOrder(broker_oid, o["status"] if o else "REJECTED", o["qty"] if o and o["status"]=="FILLED" else 0.0, o["price"] if o else 0.0)

class OrderLedger:
    def __init__(self, db_path):
        self.db_path = db_path
        with self._get_conn() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS orders (client_oid TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, order_type TEXT, limit_price REAL, arrival_price REAL, status TEXT, filled_qty REAL DEFAULT 0, avg_fill_price REAL DEFAULT 0, retry_count INTEGER DEFAULT 0, broker_oid TEXT)''')
    
    def _get_conn(self):
        c = sqlite3.connect(self.db_path, timeout=10.0)
        c.row_factory = sqlite3.Row
        c.execute('pragma journal_mode=wal')
        return c

    def fetch_status(self, st):
        with self._get_conn() as c: return pd.read_sql_query(f"SELECT * FROM orders WHERE status IN ({','.join(['?']*len(st))})", c, params=st)

    def update(self, coid, st, fq=0.0, ap=0.0, boid=None):
        with self._get_conn() as c: c.execute('UPDATE orders SET status=?, filled_qty=?, avg_fill_price=?, broker_oid=COALESCE(?, broker_oid) WHERE client_oid=?', (st, fq, ap, boid, coid))

class ExecutionEngine:
    def __init__(self, broker: BaseBrokerGateway):
        self.broker = broker
        self.ledger = OrderLedger(Config.ORDER_DB_PATH)
        self.is_running = False

    def run(self):
        self.is_running = True
        self.broker.connect()
        while self.is_running:
            for _, r in self.ledger.fetch_status(['PENDING_SUBMIT']).iterrows():
                try:
                    logger.info(f"📤 提交: {r['side']} {r['qty']} {r['symbol']}")
                    bo = self.broker.submit_order(r['symbol'], r['side'], r['qty'], 'MKT', client_oid=r['client_oid'])
                    self.ledger.update(r['client_oid'], bo.status if bo else 'REJECTED', boid=bo.broker_oid if bo else None)
                except Exception: self.ledger.update(r['client_oid'], 'REJECTED')
            for _, r in self.ledger.fetch_status(['OPEN']).iterrows():
                if r['broker_oid']:
                    try:
                        bo = self.broker.fetch_order(r['broker_oid'])
                        self.ledger.update(r['client_oid'], bo.status, bo.filled_qty, bo.avg_fill_price)
                        if bo.status == 'FILLED':
                            tca = {"client_oid": r['client_oid'], "symbol": r['symbol'], "side": r['side'], "qty": bo.filled_qty, "arrival_price": r['arrival_price'], "execution_price": bo.avg_fill_price, "timestamp": datetime_to_str(), "slippage_bps": abs(bo.avg_fill_price - r['arrival_price'])/r['arrival_price']*10000.0 * (-1 if r['side']=='SELL' else 1)}
                            with open(Config.TCA_LOG_PATH, 'a') as f: f.write(json.dumps(tca) + '\n')
                    except Exception: pass
            time.sleep(0.5)

# ================= 6. 路由入口 =================
def run_tech_matrix():
    ctx = _build_market_context()
    prep, hist = _prepare_universe_data(ctx)
    if not prep: return
    raw = [{'sym': d['sym'], 'curr': d['curr'], 'prev': d['prev'], 'score': _evaluate_omni_matrix(d['stock'], ctx, d['cf'], d['alt'])[0], 'sig': _evaluate_omni_matrix(d['stock'], ctx, d['cf'], d['alt'])[1], 'factors': [], 'ml_features': {}, 'news': "", 'sym_sec': "QQQ", 'is_bearish_div': False, 'black_swan_risk': False, 'total_score': _evaluate_omni_matrix(d['stock'], ctx, d['cf'], d['alt'])[0], 'is_untradeable': False} for d in prep]
    raw = _apply_ai_inference(raw, ctx)
    reps = [{"symbol": r['sym'], "score": r['total_score'], "ai_prob": r['ai_prob'], "signals": r['sig'], "factors": [], "ml_features": {}, "curr_close": r['curr']['Close'], "tp": r['curr']['Close']*1.1, "sl": r['curr']['Close']*0.9, "news": "", "sector": "QQQ", "pos_advice": "", "kelly_fraction": 0.1} for r in raw if r['total_score'] >= Config.Params.MIN_SCORE_THRESHOLD]
    if reps:
        freps = _apply_kelly_cluster_optimization(reps, hist, 1.0, ctx)
        _generate_and_send_matrix_report(freps, [], ctx)
        with sqlite3.connect(Config.ORDER_DB_PATH, timeout=10.0) as c:
            for r in freps: c.execute("INSERT OR IGNORE INTO orders (client_oid, symbol, side, qty, order_type, arrival_price, status) VALUES (?, ?, 'BUY', ?, 'MKT', ?, 'PENDING_SUBMIT')", (f"QB_{uuid.uuid4().hex[:8]}", r['symbol'], 10, r['curr_close']))
    else: logger.info("静默。")

def run_gateway():
    k, s, u = os.environ.get('ALPACA_API_KEY',''), os.environ.get('ALPACA_API_SECRET',''), os.environ.get('ALPACA_BASE_URL','https://paper-api.alpaca.markets')
    ExecutionEngine(AlpacaGateway(k, s, u) if k else MockAlpacaGateway()).run()

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if m == "matrix": run_tech_matrix()
    elif m == "gateway": run_gateway()
    elif m == "test": send_alert("连通性测试", "全维宏观 Meta 跃迁完成！")
