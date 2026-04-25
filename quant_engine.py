# 存储路径: quant_engine.py
import os
# [针对 4C24G ARM 优化] 压制底层 C 库隐式多线程，避免抢占物理核心导致严重的上下文切换
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
import multiprocessing
from multiprocessing import Value, Lock as MPLock
import copy
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass, field

try:
    import pyarrow.feather as feather
    FEATHER_AVAILABLE = True
except ImportError:
    FEATHER_AVAILABLE = False

try:
    import torch
    # [针对 4C24G ARM 优化] 禁用 PyTorch 内部的多线程，交由外层进程池调度
    torch.set_num_threads(1)
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("QuantBot")

_GLOBAL_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json'
}

import asyncio
try:
    import aiohttp
    from aiohttp import ClientTimeout, TCPConnector
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# 注入带有强效防抖与重试机制的全局 Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_GLOBAL_SESSION = requests.Session()
_retry_strategy = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_GLOBAL_SESSION.mount("https://", HTTPAdapter(max_retries=_retry_strategy, pool_connections=20, pool_maxsize=20))
_GLOBAL_SESSION.headers.update(_GLOBAL_HEADERS)

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
    # ✅ 修复：将所有核心文件路径强绑定至 DATA_DIR 目录，对齐 GitHub Artifacts 与 Dashboard
    DATA_DIR: str = ".quantbot_data"
    STATS_FILE: str = os.path.join(DATA_DIR, "strategy_stats.json")
    REPORT_FILE: str = os.path.join(DATA_DIR, "backtest_report.md")
    EXT_CACHE_FILE: str = os.path.join(DATA_DIR, "daily_ext_cache.json")
    ORDER_DB_PATH: str = os.path.join(DATA_DIR, "order_state.db") 
    TCA_LOG_PATH: str = os.path.join(DATA_DIR, "tca_history.jsonl")
    
    CACHE_FILE: str = os.path.join(DATA_DIR, "tickers_cache.json")
    ALERT_CACHE_FILE: str = os.path.join(DATA_DIR, "alert_history.json")
    MODEL_FILE: str = os.path.join(DATA_DIR, "scoring_model.pkl")
    WSB_CACHE_FILE: str = os.path.join(DATA_DIR, "wsb_history.json")  
    TRANSFORMER_PTH: str = os.path.join(DATA_DIR, "transformer_production.pth")
    CUSTOM_CONFIG_FILE: str = "quantbot_config.json" 
    MODEL_VERSION: str = "3.0" 

    CORE_WATCHLIST: list = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "QCOM", "JPM", "V", "MA", "UNH", "JNJ", "XOM", "PG", "HD", "COST", "ABBV", "CRM", "NFLX", "ADBE", "NOW", "UBER", "INTU", "IBM", "ISRG", "SYK", "SPY", "QQQ", "IWM", "DIA"]
    CROWDING_EXCLUDE_SECTORS: list = [INDEX_ETF] 

    CORE_FACTORS = ["米奈尔维尼", "强相对强度", "均线多头共振", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)", "突破缺口", "VWAP突破", "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", "口袋支点", "VCP收缩", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "跨时空共振(周线)", "破底翻(Spring)", "地量窒息真突破", "缩量企稳回踩"]
    ADVANCED_MATH_FACTORS = ["量子概率云(KDE)", "稳健赫斯特(Hurst)", "FFT多窗共振(动能)", "CVD筹码净流入", "独立Alpha(脱钩)", "NR7极窄突破", "VPT量价共振", "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)", "Amihud非流动性(冲击成本)", "波动率风险溢价(VRP)"]
    INTERACTION_FACTORS = ["带量金叉(交互)", "量价吸筹(交互)", "近3日突破(滞后)", "近3日巨量(滞后)", "大周期保护小周期(MACD共振)", "60分钟级精准校准(RSI反弹)", "52周高点距离(动能延续)"]
    ALT_DATA_FACTORS = ["聪明钱月度净流入(月线)", "期权PutCall情绪(PCR)", "隐含波动率偏度(IV Skew)", "做空兴趣突变(轧空)", "内部人集群净买入(Insider)", "分析师修正动量(Analyst)", "舆情NLP情感极值(News_NLP)", "散户热度加速度(WSB_Accel)"]
    TRANSFORMER_FACTORS = [f"Alpha_T{i:02d}" for i in range(1, 17)]
    ALL_FACTORS = CORE_FACTORS + ADVANCED_MATH_FACTORS + INTERACTION_FACTORS + ALT_DATA_FACTORS + TRANSFORMER_FACTORS
    GROUP_A_FACTORS = CORE_FACTORS + ["聪明钱月度净流入(月线)", "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)", "52周高点距离(动能延续)", "内部人集群净买入(Insider)", "分析师修正动量(Analyst)"] + [f for f in INTERACTION_FACTORS if "共振" in f or "滞后" in f]
    GROUP_B_FACTORS = ADVANCED_MATH_FACTORS + ALT_DATA_FACTORS + [f for f in CORE_FACTORS if "扫盘" in f or "失衡" in f or "抢筹" in f] + TRANSFORMER_FACTORS

    class Params:
        # [针对 4C24G ARM 优化] 动态进程池：留出 1 个核心给系统IO和主调度器，避免 OOM 和死锁
        import multiprocessing
        MAX_WORKERS = min(3, multiprocessing.cpu_count() - 1) if multiprocessing.cpu_count() > 1 else 1
        MIN_SCORE_THRESHOLD = 9.5
        BASE_MAX_RISK = 0.015       
        CROWDING_PENALTY = 0.75     
        CROWDING_MIN_STOCKS = 2     
        PORTFOLIO_VALUE = 100000.0  
        SLIPPAGE = 0.0005            
        COMMISSION = 0.0001         
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
                                if isinstance(orig_val, bool): setattr(cls, k, str(v).lower() in ['true', '1', 'yes', 't'])
                                elif isinstance(orig_val, (int, float, str)):
                                    try: setattr(cls, k, type(orig_val)(v))
                                    except: pass
                                else: setattr(cls, k, v)
                except Exception: pass

    @classmethod
    def get_current_log_file(cls) -> str:
        return f"{cls.LOG_PREFIX}{datetime.now(timezone.utc).strftime('%Y_%m')}.jsonl"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        # [行业黑洞修复] 未知股票不再归入免责的 QQQ，而是归入通用外围板块，接受统一的拥挤度惩罚
        return "SP500_WILD"

def datetime_to_str(): return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def validate_config():
    if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR, exist_ok=True)
    if not os.path.exists(Config.get_current_log_file()): open(Config.get_current_log_file(), 'a').close()
    if not os.path.exists(Config.REPORT_FILE): open(Config.REPORT_FILE, 'a').close()
    if not os.path.exists(Config.STATS_FILE):
        with open(Config.STATS_FILE, 'w') as f: f.write("{}")
    if not os.path.exists(Config.ALERT_CACHE_FILE):
        with open(Config.ALERT_CACHE_FILE, 'w') as f: json.dump({"matrix": {}, "shadow_pool": {}}, f)
    # ✅ 修复：提前初始化 tca 账本空文件，保证上传 Artifacts 时不报 404
    if not os.path.exists(Config.TCA_LOG_PATH): open(Config.TCA_LOG_PATH, 'a').close()
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
    tlt_trend: float = 1.0; dxy_trend: float = -1.0

@dataclass
class MarketContextLite:
    """可安全跨进程序列化的纯值上下文（传递轻量级 Series，完美闭环宏观指标计算）"""
    regime: str; w_mul: float; xai_weights: dict; vix_current: float
    macro_gravity: bool; health_score: float; dynamic_min_score: float
    qqq_close: pd.Series; spy_close: pd.Series; tlt_close: pd.Series; dxy_close: pd.Series
    tlt_trend: float = 1.0; dxy_trend: float = -1.0

def _build_ctx_lite(ctx: MarketContext) -> MarketContextLite:
    """从完整 ctx 蒸馏出可序列化的轻量版本"""
    def _safe_close(df): return df['Close'] if not df.empty else pd.Series(dtype=float)
    return MarketContextLite(
        regime=ctx.regime, w_mul=ctx.w_mul, xai_weights=ctx.xai_weights,
        vix_current=ctx.vix_current, macro_gravity=ctx.macro_gravity,
        health_score=ctx.health_score, dynamic_min_score=ctx.dynamic_min_score,
        qqq_close=_safe_close(ctx.qqq_df),
        spy_close=_safe_close(ctx.macro_data.get('spy', pd.DataFrame())),
        tlt_close=_safe_close(ctx.macro_data.get('tlt', pd.DataFrame())),
        dxy_close=_safe_close(ctx.macro_data.get('dxy', pd.DataFrame())),
        tlt_trend=ctx.tlt_trend, dxy_trend=ctx.dxy_trend
    )

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

# ================= 2.1 异步 IO 引擎 (AsyncIO & aiohttp) =================

class AsyncSessionManager:
    """针对 ARM 优化的高并发异步 Session 管理器，具备指数级退避与连接池池化能力"""
    _session: 'Any' = None

    @classmethod
    async def get_session(cls) -> 'Any':
        if not AIOHTTP_AVAILABLE:
            return None
        if cls._session is None or getattr(cls._session, 'closed', True):
            # 针对高频 IO 优化连接池，限制单域并发防止触发 429
            connector = TCPConnector(limit_per_host=10, ttl_dns_cache=300, use_dns_cache=True)
            timeout = ClientTimeout(total=15, connect=5)
            cls._session = aiohttp.ClientSession(connector=connector, timeout=timeout, headers=_GLOBAL_HEADERS)
            logger.info("📡 [AsyncIO] 成功激活高并发 aiohttp 全局会话句柄")
        return cls._session

    @classmethod
    async def close(cls):
        if cls._session and AIOHTTP_AVAILABLE:
            await cls._session.close()
            cls._session = None

async def safe_get_sentiment_data_async(symbol: str) -> Tuple[float, float, float, float, float, float, float, float]:
    """[异步版] 社交媒体与情绪抓取：消除同步阻塞"""
    try:
        # 实际生产中此处对接 StockTwits/Reddit/WSB 异步接口
        await asyncio.sleep(0.01)
        return 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

async def safe_get_alt_data_async(symbol: str) -> Tuple[float, float, float, float, float, float, float, float]:
    """[异步版] 替代数据与期权异动抓取"""
    try:
        await asyncio.sleep(0.01)
        return 0.8, 1.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception:
        return 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# ================= 3. 核心计算与数据拉取 =================

class SharedDFCache:
    """基于内存映射的零拷贝DataFrame缓存，多进程可直接读取 (破解 IPC 瓶颈)"""
    def __init__(self, cache_dir: str = "/dev/shm/quantbot" if os.path.exists("/dev/shm") else ".quantbot_data/shm", ttl_hours: int = 12):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self._cleanup_stale()  # 启动时主动清理过期内存，防止 OOM

    def _cleanup_stale(self):
        """清理超过 TTL 的过期缓存文件，并加入激进级大内存防泄漏机制"""
        import gc
        import shutil
        now = time.time()
        try:
            total_size = 0
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                if fname.endswith('.feather') and os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    total_size += size
                    if now - os.path.getmtime(fpath) > self.ttl_seconds:
                        os.remove(fpath)
                        logger.debug(f"🧹 清理过期缓存: {fname}")
                        total_size -= size
            
            # [针对 4C24G ARM 优化] 如果 /dev/shm 占用超过 8GB，触发紧急清空
            if total_size > 8 * 1024 * 1024 * 1024:
                logger.warning(f"🚨 缓存盘膨胀至 {total_size/1e9:.2f}GB，触发激进回收机制！")
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                os.makedirs(self.cache_dir, exist_ok=True)
            
            gc.collect() # 强制回收内存
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")

    def get_cache_size_mb(self) -> float:
        """监控当前缓存物理占用大小"""
        try:
            total = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in os.listdir(self.cache_dir)
                if f.endswith('.feather')
            )
            return total / 1024 / 1024
        except Exception:
            return 0.0

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if not FEATHER_AVAILABLE: return None
        path = os.path.join(self.cache_dir, f"{key}.feather")
        if os.path.exists(path):
            try:
                # feather 支持零拷贝读取，耗时极低，完美适配多进程共享
                return feather.read_feather(path, memory_map=True)
            except Exception as e:
                logger.debug(f"Cache read error for {key}: {e}")
        return None

    def set(self, key: str, df: pd.DataFrame):
        if not FEATHER_AVAILABLE: return
        path = os.path.join(self.cache_dir, f"{key}.feather")
        tmp = path + ".tmp"
        try:
            feather.write_feather(df, tmp)
            os.replace(tmp, path)  # 原子级写入，防多进程竞态
        except Exception as e:
            logger.debug(f"Cache write error for {key}: {e}")

_SHARED_CACHE = SharedDFCache()

class TokenBucket:
    """普通的基于线程锁的令牌桶限速器 (用于单一进程内，如独立的子进程或仅在线程池内)"""
    def __init__(self, rate: float = 5.0, capacity: float = 10.0):
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            else:
                wait = (tokens - self._tokens) / self.rate
                time.sleep(wait)
                self._tokens = 0.0
                return wait

class CrossProcessTokenBucket:
    """跨进程安全的共享内存令牌桶：防御多进程下局部限速器翻倍溢出漏洞"""
    def __init__(self, rate: float = 4.0, capacity: float = 8.0):
        self.rate = rate
        self.capacity = capacity
        # 使用基于内存映射的 Value 和 Lock，子进程将继承或共享相同状态
        self._tokens = Value('d', capacity)
        self._last_refill = Value('d', time.monotonic())
        self._lock = MPLock()

    def acquire(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill.value
            self._tokens.value = min(
                self.capacity,
                self._tokens.value + elapsed * self.rate
            )
            self._last_refill.value = now
            
            if self._tokens.value >= tokens:
                self._tokens.value -= tokens
                return 0.0
            else:
                wait = (tokens - self._tokens.value) / self.rate
                time.sleep(wait)
                self._tokens.value = 0.0
                return wait

_API_LIMITER = CrossProcessTokenBucket(rate=4.0, capacity=8.0)  # 主进程/线程池级 API 全局限速单例
_WORKER_LOCAL_LIMITER: Optional[TokenBucket] = None  # 进程本地限速器，专门在 _cpu_calc_worker 中降频使用

def _worker_pool_initializer(rate_per_worker: float, capacity: float):
    """ProcessPoolExecutor 的子进程初始化函数，平滑注入局部分布式限速器"""
    global _WORKER_LOCAL_LIMITER
    _WORKER_LOCAL_LIMITER = TokenBucket(rate=rate_per_worker, capacity=capacity)
    logger.debug(f"Worker {os.getpid()} 限速器初始化: 分配速率 {rate_per_worker:.2f} req/s")


_DAILY_EXT_CACHE = {}; _DAILY_EXT_LOCK = threading.Lock()
_ALT_DATA_CACHE = {}; _ALT_DATA_LOCK = threading.Lock()

def safe_div(num, den, cap=20.0):
    if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
    return max(min(num / den, cap), -cap)

def check_macd_cross(curr: pd.Series, prev: pd.Series) -> bool:
    return prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']

def _fetch_from_alpaca(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """使用 Alpaca Data v2 API 拉取行情"""
    api_key = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    
    if not api_key or not api_secret:
        return pd.DataFrame()
        
    # Alpaca 时限映射 (粗略换算，Alpaca 使用 RFC3339 格式的 start/end)
    now = datetime.now(timezone.utc)
    if period.endswith('d'): days = int(period[:-1])
    elif period.endswith('mo'): days = int(period[:-2]) * 30
    elif period.endswith('y'): days = int(period[:-1]) * 365
    else: days = 365
    
    start_time = (now - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # timeframe 映射: 1d -> 1Day, 60m -> 1Hour, 1wk -> 1Week
    if interval == '1d': timeframe = '1Day'
    elif interval == '60m' or interval == '1h': timeframe = '1Hour'
    elif interval == '1wk': timeframe = '1Week'
    elif interval == '1mo': timeframe = '1Month'
    else: timeframe = '1Day'

    url = f"https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": start_time,
        "end": end_time,
        "limit": 10000,
        "adjustment": "all" # 处理拆股除息
    }
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json"
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('bars', {}).get(symbol, [])
            if not data: return pd.DataFrame()
            
            df = pd.DataFrame(data)
            # 格式化映射: t->Date, o->Open, h->High, l->Low, c->Close, v->Volume
            df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            # 仅保留核心列
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        elif resp.status_code == 422:
            # 某些指数/非股票数据，Alpaca不提供，静默返回空以触发 Fallback
            pass
        else:
            logger.debug(f"Alpaca API 异常: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.debug(f"Alpaca 请求失败: {e}")
        
    return pd.DataFrame()


def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 3, fast_mode: bool = False) -> pd.DataFrame:
    # 修复1：对缓存键做路径安全转义，防止部分文件系统拒绝含有 '^' 或 '/' 的写入
    safe_sym = symbol.replace('^', '_caret_').replace('/', '_slash_')
    cache_key = f"{safe_sym}_{interval}_{period}"
    
    # 优先从跨进程共享内存提取数据，实现零拷贝
    cached_df = _SHARED_CACHE.get(cache_key)
    if cached_df is not None and not cached_df.empty:
        return cached_df.copy()
        
    # 修复2：缓存向下兼容降维拦截（极大节省短线如 `5d` 的请求令牌）
    if period != "1y":
        long_key = f"{safe_sym}_{interval}_1y"
        long_cached = _SHARED_CACHE.get(long_key)
        if long_cached is not None and not long_cached.empty:
            if period == "5d":
                return long_cached.tail(5).copy()
            elif period == "1mo":
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=30)
                return long_cached[long_cached.index >= cutoff].copy()
            elif period == "3mo":
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=90)
                return long_cached[long_cached.index >= cutoff].copy()
            elif period == "6mo":
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=183)
                return long_cached[long_cached.index >= cutoff].copy()

    # 修复3: 自动识别执行上下文环境，优先使用子进程内安全分配的局部低配版限速器
    limiter = _WORKER_LOCAL_LIMITER or _API_LIMITER

    for attempt in range(retries):
        wait = limiter.acquire()  # 跨域安全限速
        try:
            # ✅ 优先级 1 改进：尝试使用高性能 Alpaca Data API
            new_df = _fetch_from_alpaca(symbol, period, interval)
            
            # 若 Alpaca 拉取失败（如没有配置密钥、遇到宏观指数 ^VIX），自动安全回退 (Fallback) 降级至 yfinance
            if new_df.empty:
                logger.debug(f"🔄 [降级路由] Alpaca 未返回 {symbol} 数据，安全切换至 yfinance 通道。")
                new_df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, timeout=8)
                if not new_df.empty:
                    new_df.index = pd.to_datetime(new_df.index, utc=True)
            
            if not new_df.empty:
                df = new_df[~new_df.index.duplicated(keep='last')]
                _SHARED_CACHE.set(cache_key, df) # 写入共享内存
                return df
        except Exception as e:
            backoff = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(backoff)  # 指数退避，只在真正失败时等待
    return pd.DataFrame()

def _init_ext_cache():
    global _DAILY_EXT_CACHE
    with _DAILY_EXT_LOCK:
        if not _DAILY_EXT_CACHE:
            if os.path.exists(Config.EXT_CACHE_FILE):
                try:
                    with open(Config.EXT_CACHE_FILE, 'r', encoding='utf-8') as f: _DAILY_EXT_CACHE = json.load(f)
                except Exception: pass
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if _DAILY_EXT_CACHE.get("date") != today: _DAILY_EXT_CACHE = {"date": today, "sentiment": {}, "alt": {}}

def _save_ext_cache():
    with _DAILY_EXT_LOCK:
        try:
            temp_ext = f"{Config.EXT_CACHE_FILE}.{threading.get_ident()}.tmp"
            with open(temp_ext, 'w', encoding='utf-8') as f: json.dump(_DAILY_EXT_CACHE, f)
            os.replace(temp_ext, Config.EXT_CACHE_FILE)
        except Exception: pass

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
    try:
        resp = _GLOBAL_SESSION.get("https://tradestie.com/api/v1/apps/reddit", timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                tk = item.get('ticker')
                if tk and item.get('sentiment') == 'Bullish': current_data[tk] = item.get('no_of_comments', 0)
    except Exception as e:
        logger.debug(f"WSB API 熔断保护触发: {e}")
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
    pcr, iv_skew, short_change, short_float = 1.0, 0.0, 0.0, 0.0
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
            if insiders is not None and not insiders.empty and 'Shares' in insiders.columns:
                recent = insiders.head(20)
                buys = recent[recent['Shares'] > 0]['Shares'].sum()
                sells = recent[recent['Shares'] < 0]['Shares'].abs().sum()
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
                        analyst_mom = (monthly_scores.iloc[-1] - monthly_scores.iloc[:-1].mean()) / (monthly_scores.iloc[:-1].std() + 1e-5)
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
    try:
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(sp500_url)
        sp500_tickers = tables[0]['Symbol'].str.replace('.', '-').tolist()
        pool = list(set(Config.CORE_WATCHLIST + sp500_tickers))
        random.shuffle(pool)
        return pool[:max_stocks]
    except Exception as e:
        logger.debug(f"抓取 S&P 500 失败: {e}，回退至核心白名单")
        return list(Config.CORE_WATCHLIST)[:max_stocks]

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
                t3 = stats.get("overall", {}).get("T+3", {})
                if t3 and t3.get('total_trades', 0) > 0:
                    return f"**📈 策略验证:** 胜率 {t3.get('win_rate',0):.1%} | AI {t3.get('ai_win_rate',0):.1%} | 盈亏比 {t3.get('profit_factor',0):.2f}"
    except Exception: pass
    return ""

def get_alert_cache() -> dict:
    cache = {"matrix": {}, "shadow_pool": {}}
    try:
        if os.path.exists(Config.ALERT_CACHE_FILE):
            with open(Config.ALERT_CACHE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data.get("matrix"), dict): cache = data
    except Exception: pass
    return cache

def set_alerted(sym: str, is_shadow: bool = False, shadow_data: dict = None):
    cache = get_alert_cache()
    now_ts = time.time()
    if not is_shadow: cache.setdefault("matrix", {})[sym] = now_ts
    else:
        shadow_pool = cache.setdefault("shadow_pool", {})
        if shadow_data: shadow_data['_ts'] = now_ts; shadow_pool[sym] = shadow_data
        else: shadow_pool[sym] = {"_ts": now_ts}
    try:
        tmp = f"{Config.ALERT_CACHE_FILE}.{threading.get_ident()}.tmp"
        with open(tmp, 'w') as f: json.dump(cache, f)
        os.replace(tmp, Config.ALERT_CACHE_FILE)
    except Exception: pass

# 🚀 钉钉推送修复核心：植入关键词，剥离匿名函数，加入报错拦截
def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    req_headers = _GLOBAL_HEADERS.copy()
    req_headers["Content-Type"] = "application/json"
    
    if Config.WEBHOOK_URL:
        keyword = getattr(Config, 'DINGTALK_KEYWORD', 'AI')
        payload = {
            "msgtype": "markdown", 
            "markdown": {
                "title": f"【{keyword}】{title}", 
                "text": f"## 🤖 【{keyword}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"
            }
        }
        
        def _send_webhook(url_str, p_data):
            try: 
                res = requests.post(url_str, json=p_data, headers=req_headers, timeout=10)
                if res.status_code != 200:
                    logger.error(f"Webhook 投递失败: HTTP {res.status_code} - {res.text}")
            except Exception as e: 
                logger.error(f"Webhook 网络异常: {e}")
                
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            threading.Thread(target=_send_webhook, args=(url, payload), daemon=False).start()
            
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        html_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = re.sub(r'### (.*?)\n', r'<b>\1</b>\n', html_content)
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_content)
        html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
        html_content = html_content.replace('\n---', '\n━━━━━━━━━━━━━━━━━━')
        
        tg_text = f"🤖 <b>【量化监控】{html_title}</b>\n\n{html_content}\n\n⏱️ <i>{formatted_time}</i>"
        
        def _send_tg(t, c, txt):
            try: 
                res = requests.post(f"https://api.telegram.org/bot{t}/sendMessage", json={"chat_id": c, "text": txt[:4000], "parse_mode": "HTML"}, headers=req_headers, timeout=10)
                if res.status_code != 200: logger.error(f"TG 推送被拒: {res.text}")
            except Exception as e: logger.error(f"TG 网络异常: {e}")
            
        threading.Thread(target=_send_tg, args=(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID, tg_text), daemon=False).start()

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
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    # [提前防御] 如果 20日动能跌超 -4% 或者跌破 50日均线，直接切入隐性熊市，无需等待 200日均线破位
    if trend_20d < -0.04 or c_close < ma50:
        if c_close > ma200:
            return ("hidden_bear", "⚠️ 隐性熊市 (短线动能破位)", df, False, False)
        else:
            return ("bear", "🐻 熊市深调 (全线空头)", df, False, False) if trend_20d < -0.02 else ("rebound", "🦅 超跌反弹", df, False, False)
            
    if c_close > ma200:
        return ("bull", "🐂 牛市主升", df, False, False) if trend_20d > 0.02 else ("range", "⚖️ 牛市震荡", df, False, False)
    
    return ("range", "⚖️ 默认震荡", df, False, False)

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
    
    # ✅ 修复: 使用局部 Generator 替代全局 seed，零污染多进程的全局随机状态
    rng = np.random.default_rng(seed=int(np.sum(np.abs(close_prices[-10:])) * 1e6) % (2**31))
    
    hurst_samples = []
    for _ in range(n_bootstrap):
        sub_len = int(rng.integers(min_window, len(log_ret)))
        start = int(rng.integers(0, len(log_ret) - sub_len))
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
    # ✅ 修复：显式深拷贝隔离，切断对上游 xs() 视图的原地修改，防止底层数据污染
    df = df.copy()
    
    df = df.sort_index()
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['Open'], df['High'], df['Low'] = df['Open'].ffill(), df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'] = df['Close'].rolling(50, min_periods=10).mean()
    df['SMA_150'] = df['Close'].rolling(150, min_periods=30).mean()
    df['SMA_200'] = df['Close'].rolling(200, min_periods=40).mean()
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    
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
    df['AVWAP'] = df['VWAP_20'] 
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
    
    df['Smart_Money_Flow'] = pd.Series(clv_num / hl_diff, index=df.index).rolling(10).mean()

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

def _extract_complex_features(stock: StockData, ctx: Any) -> ComplexFeatures:
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
                if den_curr <= den_50: kde_breakout_score = min(1.0, 0.5 + 0.5 * (den_50 - den_curr) / (den_50 + 1e-10))
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
    tlt_corr, dxy_corr = 0.0, 0.0
    rs_20, pure_alpha = 0.0, 0.0

    # 🚀 精准提取，统一兼容完整的 MarketContext 与序列化后的 MarketContextLite
    is_lite = hasattr(ctx, 'spy_close')
    qqq_c = ctx.qqq_close if is_lite else (ctx.qqq_df['Close'] if not ctx.qqq_df.empty else pd.Series(dtype=float))
    spy_c = ctx.spy_close if is_lite else (ctx.macro_data.get('spy', pd.DataFrame())['Close'] if 'spy' in ctx.macro_data and not ctx.macro_data.get('spy', pd.DataFrame()).empty else pd.Series(dtype=float))
    tlt_c = ctx.tlt_close if is_lite else (ctx.macro_data.get('tlt', pd.DataFrame())['Close'] if 'tlt' in ctx.macro_data and not ctx.macro_data.get('tlt', pd.DataFrame()).empty else pd.Series(dtype=float))
    dxy_c = ctx.dxy_close if is_lite else (ctx.macro_data.get('dxy', pd.DataFrame())['Close'] if 'dxy' in ctx.macro_data and not ctx.macro_data.get('dxy', pd.DataFrame()).empty else pd.Series(dtype=float))

    if not qqq_c.empty and len(stock.df) >= 60:
        m_df = pd.DataFrame({'stock': stock.df['Close'], 'qqq': qqq_c}).dropna()
        if len(m_df) >= 60:
            qqq_ret_20 = max(m_df['qqq'].iloc[-1] / m_df['qqq'].iloc[-20], 0.5)
            rs_20 = float((m_df['stock'].iloc[-1] / m_df['stock'].iloc[-20]) / qqq_ret_20)
            ret_s, ret_q = m_df['stock'].pct_change().dropna(), m_df['qqq'].pct_change().dropna()
            cov_m = np.cov(ret_s.iloc[-60:], ret_q.iloc[-60:])
            beta_l = cov_m[0,1] / (cov_m[1,1] + 1e-10) if cov_m[1,1] > 0 else 1.0
            pure_alpha = float((ret_s.iloc[-5:].mean() - beta_l * ret_q.iloc[-5:].mean()) * 252)

    if not spy_c.empty and len(stock_ret) >= 60:
        spy_ret = spy_c.pct_change().reindex(stock_ret.index).fillna(0)
        cov_mat = np.cov(stock_ret.iloc[-60:], spy_ret.iloc[-60:])
        if cov_mat[1,1] > 0: beta_60d = float(cov_mat[0, 1] / cov_mat[1, 1])
        spy_realized_vol = float(spy_ret.iloc[-20:].std() * np.sqrt(252) * 100)

    if not tlt_c.empty and len(stock_ret) >= 90:
        tlt_ret = tlt_c.pct_change().reindex(stock_ret.index).fillna(0)
        tlt_corr = float(stock_ret.iloc[-90:].corr(tlt_ret.iloc[-90:]))

    if not dxy_c.empty and len(stock_ret) >= 20:
        dxy_ret = dxy_c.pct_change().reindex(stock_ret.index).fillna(0)
        dxy_corr = float(stock_ret.iloc[-20:].corr(dxy_ret.iloc[-20:]))
        
    return ComplexFeatures(
        weekly_bullish=weekly_bullish, fvg_lower=fvg_lower, fvg_upper=fvg_upper, 
        kde_breakout_score=kde_breakout_score, fft_ensemble_score=fft_ensemble_score, 
        hurst_med=hurst_med, hurst_iqr=hurst_iqr, hurst_reliable=hurst_reliable, 
        monthly_inst_flow=monthly_inst_flow, weekly_macd_res=weekly_macd_res, 
        rsi_60m_bounce=rsi_60m_bounce, beta_60d=float(beta_60d), tlt_corr=float(tlt_corr if pd.notna(tlt_corr) else 0), 
        dxy_corr=float(dxy_corr if pd.notna(dxy_corr) else 0), vrp=float((ctx.vix_current - spy_realized_vol) / max(ctx.vix_current, 1.0)), 
        rs_20=float(rs_20), pure_alpha=float(pure_alpha)
    )

def _extract_ml_features(stock: StockData, ctx: Any, cf: ComplexFeatures, alt: AltData, alpha_vec: np.ndarray = None) -> dict:
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

def _evaluate_omni_matrix(stock: StockData, ctx: Any, cf: ComplexFeatures, alt: AltData) -> Tuple[float, List[str], List[str], bool]:
    triggered_list, factors_list = [], []
    theme_scores = {'TREND': 0.0, 'VOLATILITY': 0.0, 'REVERSAL': 0.0, 'QUANTUM': 0.0, 'SHORT': 0.0}
    sub_theme_scores = defaultdict(list)
    black_swan_risk = False
    
    def get_fw(tag_name: str) -> float: return ctx.xai_weights.get(tag_name, 1.0)
    def add_trigger(tag, text, pts, theme, sub_theme=None):
        fw = get_fw(tag)
        if fw > 0:
            adj_pts = pts * getattr(ctx, 'w_mul', 1.0) * fw
            regime = getattr(ctx, 'regime', 'range')
            if regime in ["bear", "hidden_bear"]:
                if theme in ["TREND", "VOLATILITY"]: adj_pts *= 0.6  
                elif theme == "REVERSAL": adj_pts *= 1.4
            elif regime in ["bull", "rebound"]:
                if theme in ["TREND", "VOLATILITY", "QUANTUM"]: adj_pts *= 1.2
            
            if sub_theme: sub_theme_scores[f"{theme}_{sub_theme}"].append(adj_pts)
            else: theme_scores[theme] += adj_pts
            
            triggered_list.append(text.format(fw=fw)); factors_list.append(tag)

    gap_pct = (stock.curr['Open'] - stock.prev['Close']) / (stock.prev['Close'] + 1e-10)
    atr_pct = (stock.curr['ATR'] / (stock.prev['Close'] + 1e-10)) * 100
    day_chg = (stock.curr['Close'] - stock.curr['Open']) / (stock.curr['Open'] + 1e-10) * 100
    tr_val = stock.curr['High'] - stock.curr['Low'] + 1e-10

    if pd.notna(stock.curr['SMA_200']) and stock.curr['Close'] > stock.curr['SMA_50'] > stock.curr['SMA_150'] > stock.curr['SMA_200']:
        m_str = (stock.curr['SMA_50'] - stock.curr['SMA_200']) / (stock.curr['SMA_200'] + 1e-10)
        add_trigger("米奈尔维尼", f"🏆 [主升趋势] 米奈尔维尼模板形成 (强度:{m_str*100:.1f}% 权:{{fw:.2f}}x)", 8 + int(m_str*20), "TREND")
        
    if pd.notna(stock.curr['SMA_200']) and stock.curr['EMA_20'] > stock.curr['SMA_50'] > stock.curr['SMA_200']:
        add_trigger("均线多头共振", f"🔥 [多头排列] 均线三线开花强趋势确立 (权:{{fw:.2f}}x)", 6, "TREND")
        
    if cf.rs_20 > 0: 
        dynamic_rs_thresh = 1.0 + (stock.curr['ATR'] / (stock.curr['Close'] + 1e-10)) * 2.0
        if cf.rs_20 > dynamic_rs_thresh: add_trigger("强相对强度", f"⚡ [相对强度] 动能超越波动率动态阈值 (阈值:{dynamic_rs_thresh:.2f} 权:{{fw:.2f}}x)", 7 if stock.is_vol else 4, "TREND", "MOMENTUM_BURST")
    
    allow_breakout = stock.curr['Close'] >= stock.curr['SMA_50']
    allow_reversal = stock.curr['Close'] < stock.curr['SMA_50']

    macd_crossed = check_macd_cross(stock.curr, stock.prev)
    if macd_crossed:
        is_above_water = stock.curr['MACD'] > Config.Params.MACD_WATERLINE
        add_trigger("MACD金叉", f"🔥 [经典动能] MACD{'水上金叉' if is_above_water else '水下金叉'}起爆 (权:{{fw:.2f}}x)", 12 if is_above_water else 8, "TREND", "MOMENTUM_BURST")

    if stock.curr['Above_Cloud'] == 1 and stock.curr['Tenkan'] > stock.curr['Kijun']: add_trigger("一目多头", "🌥️ [趋势确认] 一目均衡表云上多头共振 (权:{fw:.2f}x)", 6, "TREND")
    if allow_breakout and stock.curr['Close'] > stock.curr['VWAP_20'] and stock.prev['Close'] <= stock.curr['VWAP_20']: add_trigger("VWAP突破", "🌊 [量价突破] 放量逾越近20日VWAP机构均价线 (权:{fw:.2f}x)", 8, "TREND", "MOMENTUM_BURST")
    if allow_breakout and pd.notna(stock.curr['AVWAP']) and stock.curr['Close'] > stock.curr['AVWAP'] and stock.prev['Close'] <= stock.curr['AVWAP']: add_trigger("AVWAP突破", "⚓ [筹码夺回] 强势站上AVWAP锚定成本核心区 (权:{fw:.2f}x)", 12, "TREND", "MOMENTUM_BURST")

    kc_w, bb_w = stock.curr['KC_Upper'] - stock.curr['KC_Lower'], stock.curr['BB_Upper'] - stock.curr['BB_Lower']
    if bb_w < kc_w and stock.curr['Close'] > stock.curr['KC_Upper'] and stock.curr['Volume'] > stock.curr['Vol_MA20'] * 1.5: 
        s_ratio = (kc_w - bb_w) / (stock.curr['ATR'] + 1e-10)
        add_trigger("TTM Squeeze ON", f"📦 [波动压缩] 多头强爆发下的 TTM Squeeze 向上突破确认 (比率:{s_ratio:.2f} 权:{{fw:.2f}}x)", 8 + int(s_ratio*10), "VOLATILITY")

    if stock.prev['NR7'] and stock.prev['Inside_Bar'] and stock.curr['Close'] > stock.prev['High']: add_trigger("NR7极窄突破", "🎯 [极度压缩] 7日极窄压缩孕线完成向上爆破 (权:{fw:.2f}x)", 12, "VOLATILITY")

    vix_scalar = getattr(ctx, 'vix_scalar', 1.0)
    vcp_th = (Config.Params.VCP_BEAR if getattr(ctx, 'regime', 'range') in ["bear", "hidden_bear"] else Config.Params.VCP_BULL) * vix_scalar
    if stock.curr['Range_20'] > 0 and stock.curr['Range_20'] < stock.curr['Range_60'] * vcp_th and stock.curr['Close'] > stock.curr['SMA_50']:
        vcp_score = min(25, 10 + int(stock.curr['Range_60'] / (stock.curr['Range_20'] + 1e-5)) * 2)
        add_trigger("VCP收缩", f"🌪️ [VCP形态] 极度价格波动压缩后的放量突破 (动态权:{{fw:.2f}}x)", vcp_score, "VOLATILITY")

    is_pullback = stock.curr['Close'] < stock.curr['EMA_20']
    
    if is_pullback:
        if allow_reversal and pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Low'] < stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.curr['Swing_Low_20']: add_trigger("流动性扫盘", "🧹 [止损猎杀] 刺穿前低扫掉散户止损后迅速诱空反转 (权:{fw:.2f}x)", 15, "REVERSAL", "BOTTOM_REVERSAL")
        if pd.notna(stock.curr['Swing_Low_20']) and stock.curr['Low'] > stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.swing_high_10: add_trigger("特性改变(ChoCh)", "🔀 [结构破坏] 突破近期反弹高点，完成 ChoCh 趋势逆转确认 (权:{fw:.2f}x)", 15, "REVERSAL", "BOTTOM_REVERSAL")
        if allow_reversal and pd.notna(stock.curr['OB_High']) and stock.curr['Low'] <= stock.curr['OB_High'] and stock.curr['Close'] >= stock.curr['OB_Low'] and stock.curr['Close'] > stock.curr['Open']: add_trigger("订单块(OB)", "🧱 [机构订单块] 触达历史起爆底仓区并收出企稳阳线 (权:{fw:.2f}x)", 15, "REVERSAL", "BOTTOM_REVERSAL")
        if cf.rsi_60m_bounce == 1.0: add_trigger("60分钟级精准校准(RSI反弹)", "⏱️ [战术执行] 60分钟线 RSI 触底反弹，日内高频微操切入绝佳滑点位置 (权:{fw:.2f}x)", 8, "REVERSAL", "BOTTOM_REVERSAL")

    if cf.kde_breakout_score > Config.Params.KDE_BREAKOUT: add_trigger("量子概率云(KDE)", f"☁️ [真空逃逸] KDE 自适应带宽揭示上行阻力极度稀薄 (得分:{cf.kde_breakout_score:.2f} 权:{{fw:.2f}}x)", 15 * cf.kde_breakout_score, "QUANTUM")
    if cf.hurst_reliable and cf.hurst_med > Config.Params.HURST_RELIABLE and stock.curr['SMA_20'] > stock.curr['SMA_50'] and stock.curr['Close'] > stock.curr['SMA_20']: add_trigger("稳健赫斯特(Hurst)", f"⏳ [稳健记忆] R/S Bootstrap 确认强抗噪多头持续性 (Hurst={cf.hurst_med:.2f} 权:{{fw:.2f}}x)", 15 * (cf.hurst_med - 0.5) * 2.0, "QUANTUM")

    if alt.insider_net_buy > Config.Params.INSIDER_BUY: add_trigger("内部人集群净买入(Insider)", "👔 [内幕天眼] SEC Form 4 披露高管集群式大额净买入 (权:{fw:.2f}x)", 20, "QUANTUM")
    if alt.analyst_mom > Config.Params.ANALYST_UP: add_trigger("分析师修正动量(Analyst)", f"📊 [投行护航] 分析师评级修正动量突破 Z-Score={alt.analyst_mom:.1f} (权:{{fw:.2f}}x)", 8, "QUANTUM")
    elif alt.analyst_mom < Config.Params.ANALYST_DN: add_trigger("分析师修正动量(Analyst)", f"⚠️ [投行抛售] 分析师评级下调动量爆表 Z-Score={alt.analyst_mom:.1f}，已被系统降权 (权:{{fw:.2f}}x)", -10, "TREND")

    if macd_crossed and "带量金叉(交互)" in factors_list: theme_scores['TREND'] -= min(get_fw("MACD金叉")*8, get_fw("带量金叉(交互)")*12) * 0.5 
    if cf.pure_alpha > 0.8: add_trigger("独立Alpha(脱钩)", "🪐 [独立Alpha] 强势剥离大盘Beta，爆发特质动能 (权:{fw:.2f}x)", 22, "TREND")
    if is_pullback:
        if stock.curr['SuperTrend_Up'] == 1 and stock.curr['Close'] < stock.curr['EMA_20'] + stock.curr['ATR'] * 0.5: add_trigger("强势回踩", "🟢 [低吸点位] 超级趋势主升轨精准回踩 (权:{fw:.2f}x)", 10, "REVERSAL", "BOTTOM_REVERSAL")
        if allow_reversal and cf.fvg_lower > 0 and stock.curr['Low'] <= cf.fvg_upper and stock.curr['Close'] > cf.fvg_lower: add_trigger("SMC失衡区", "🧲 [SMC交易法] 精准回踩并测试前期机构失衡区(FVG) (权:{fw:.2f}x)", 15, "REVERSAL", "BOTTOM_REVERSAL")
        
    if allow_breakout and gap_pct > (stock.curr['ATR'] / (stock.prev['Close'] + 1e-10)) * 0.5 and gap_pct < (stock.curr['ATR'] / (stock.prev['Close'] + 1e-10)) * 2.0 * vix_scalar: add_trigger("突破缺口", "💥 [动能爆发] 放量跳空，留下底部突破缺口 (权:{fw:.2f}x)", 8, "VOLATILITY")

    prev_body = abs(stock.prev['Close'] - stock.prev['Open'])
    if stock.prev['Volume'] > stock.prev['Vol_MA20'] * 2.0 and prev_body < stock.prev['ATR'] * 0.5:
        if pd.notna(stock.prev['Swing_Low_20']) and stock.prev['Close'] < stock.prev['Swing_Low_20'] * 1.05:
            if stock.curr['Close'] > stock.prev['High']: add_trigger("巨量滞涨", "🛑 [冰山吸筹] 底部巨量滞涨，次日放量突破最高点进行右侧确认，机构吸筹完毕发车 (权:{fw:.2f}x)", 12, "QUANTUM")
        elif stock.prev['Close'] > stock.swing_high_10 * 0.95 and stock.curr['Close'] < stock.prev['Low']: 
            triggered_list.append("⚠️ [高位派发] 高位巨量滞涨且次日跌破低点确认，警惕机构冰山挂单出货 (已被系统降权)")
        
    if allow_breakout and day_chg > max(3.0, atr_pct * 0.6) and stock.curr['Volume'] > stock.curr['Vol_MA20'] * 1.5: add_trigger("放量长阳", "⚡ [动能脉冲] 强劲的日内放量大实体阳线 (权:{fw:.2f}x)", 12, "QUANTUM", "PRICE_ACTION_BURST")
    if allow_breakout and stock.curr['Close'] > stock.prev['Close'] and stock.curr['Volume'] > stock.curr['Max_Down_Vol_10'] > 0 and stock.curr['Close'] >= stock.curr['EMA_50'] and stock.prev['Close'] <= stock.curr['EMA_50'] * 1.02: add_trigger("口袋支点", "💎 [口袋支点] 放量阳线成交量完全吞噬近期最大阴量 (权:{fw:.2f}x)", 12, "REVERSAL", "BOTTOM_REVERSAL")

    if allow_reversal and pd.notna(stock.prev['Swing_Low_20']) and stock.prev['Low'] < stock.prev['Swing_Low_20']:
        if stock.curr['Close'] > stock.prev['High'] and stock.curr['Close'] > stock.curr['Swing_Low_20'] and stock.curr['Volume'] > stock.prev['Volume']:
            add_trigger("破底翻(Spring)", "🌱 [绝地反击] 诱空杀跌破位后次日强势放量反包，并收复关键支撑线 (权:{fw:.2f}x)", 15, "REVERSAL", "BOTTOM_REVERSAL")

    if allow_breakout and stock.curr['Close'] > stock.curr['Highest_22'] * 0.98 and stock.curr['Volume'] > stock.curr['Vol_MA20'] * 1.5:
        if len(stock.df) >= 6 and (stock.df['Volume'].iloc[-6:-1] < stock.df['Vol_MA20'].iloc[-6:-1] * 0.5).any():
            add_trigger("地量窒息真突破", "🌋 [极致干涸爆发] 极度缩量洗盘致使抛压枯竭后，多头猛烈放量长阳突破 (权:{fw:.2f}x)", 15, "TREND", "MOMENTUM_BURST")

    if allow_breakout and stock.curr['SMA_50'] <= stock.curr['Close'] < stock.curr['SMA_50'] * 1.03:
        if stock.curr['Volume'] < stock.curr['Vol_MA20'] * 0.8 and stock.curr['Close'] > stock.prev['High'] and stock.curr['Close'] > stock.curr['Open']:
            add_trigger("缩量企稳回踩", "🛡️ [精准防守] 回踩50日主防线时抛压显著衰竭，并已走出右侧反包阳线 (权:{fw:.2f}x)", 12, "TREND")

    lower_wick = stock.curr['Open'] - stock.curr['Low'] if stock.curr['Close'] > stock.curr['Open'] else stock.curr['Close'] - stock.curr['Low']
    upper_wick = stock.curr['High'] - stock.curr['Close'] if stock.curr['Close'] > stock.curr['Open'] else stock.curr['High'] - stock.curr['Open']
    if stock.curr['Close'] > stock.curr['Open'] and (lower_wick / tr_val) > 0.3 and (upper_wick / tr_val) < 0.15: add_trigger("AMD操盘", "🎭 [AMD诱空] 深度开盘诱空下杀后，全天拉升派发的操盘模型 (权:{fw:.2f}x)", 12, "REVERSAL")
    if cf.weekly_bullish and (stock.curr['Close'] > stock.curr['Highest_22'] * 0.95): add_trigger("跨时空共振(周线)", "🌌 [多周期共振] 周线级别主升浪与日线级别放量的强力双重共振 (权:{fw:.2f}x)", 20, "QUANTUM")

    if stock.curr['CVD_Trend'] == 1.0 and stock.prev['CVD_Trend'] <= 0.0:
        if stock.curr['CVD_Divergence'] == 0: add_trigger("CVD筹码净流入", "🧬 [微观筹码] CVD 双均线平滑多头确立且无量价背离，真实买盘涌入 (权:{fw:.2f}x)", 12, "QUANTUM", "PRICE_ACTION_BURST")
        else: triggered_list.append("⚠️ [微观筹码] 价格近高但 CVD 出现顶背离，尾盘动能涉嫌虚假欺骗 (已被AI降权)")

    if stock.curr['VPT_ZScore'] > 0.5 and stock.curr['VPT_Accel'] > 0 and stock.prev['VPT_ZScore'] <= 0.5: add_trigger("VPT量价共振", "📈 [量价归一] VPT Z-Score突破且动能加速，真实买盘绝对共振 (权:{fw:.2f}x)", 10, "TREND", "MOMENTUM_BURST")
    if macd_crossed: add_trigger("带量金叉(交互)", "🔥 [交互共振] MACD金叉与成交量激增产生乘数效应 (权:{fw:.2f}x)", 12, "TREND", "MOMENTUM_BURST")
    
    cmf_val = stock.curr['CMF'] if pd.notna(stock.curr['CMF']) else 0.0
    smf_val = stock.curr['Smart_Money_Flow'] if pd.notna(stock.curr['Smart_Money_Flow']) else 0.0
    if cmf_val > 0.05 and smf_val > 0.1:
        cmf_intensity = min(1.0, max(0.0, (cmf_val - 0.05) / 0.2)) 
        smf_intensity = min(1.0, max(0.0, (smf_val - 0.1) / 0.4)) 
        interaction_pts = 12 * (cmf_intensity * 0.6 + smf_intensity * 0.4)
        if interaction_pts > 3.0: add_trigger("量价吸筹(交互)", f"🏦 [交互共振] 蔡金资金流与聪明钱同向深度吸筹 (强度: {interaction_pts/12:.1%} 权:{{fw:.2f}}x)", interaction_pts, "QUANTUM", "PRICE_ACTION_BURST")
        
    if cf.fft_ensemble_score >= Config.Params.FFT_RESONANCE: add_trigger("FFT多窗共振(动能)", "🌊 [频域共振] 多窗口 FFT 阵列一致确认，主导周期处于强劲上升象限 (权:{fw:.2f}x)", 15, "QUANTUM")
    if cf.weekly_macd_res == 1.0 and macd_crossed: add_trigger("大周期保护小周期(MACD共振)", "🛡️ [多维时空] 周线MACD动能发散，日线精准金叉，大周期绝对战役保护 (权:{fw:.2f}x)", 15, "TREND")
    if cf.monthly_inst_flow == 1.0: add_trigger("聪明钱月度净流入(月线)", "🏛️ [战略定调] 月线级别连续3个月大单资金净流入，暗池机构底仓坚如磐石 (权:{fw:.2f}x)", 10, "QUANTUM")

    macro_gravity = getattr(ctx, 'macro_gravity', False)
    if cf.beta_60d > 1.2 and not macro_gravity: add_trigger("大盘Beta(宏观调整)", "📈 [宏观Beta动能] 宏观低压期，高Beta(>1.2)特质赋予极强上行弹性 (权:{fw:.2f}x)", 7, "TREND")
    elif cf.beta_60d > 1.2 and macro_gravity: add_trigger("大盘Beta(宏观调整)", "⚠️ [宏观Beta反噬] 宏观引力波高压期，高Beta特质面临深度回撤风险，降权防御 (权:{fw:.2f}x)", -10, "TREND")

    if cf.tlt_corr > 0.4 and getattr(ctx, 'tlt_trend', 1.0) > 0: add_trigger("利率敏感度(TLT相关性)", "🏦 [宏观映射] 与长期国债(TLT)高度正相关且美债处于多头趋势，受益于无风险利率见顶预期 (权:{fw:.2f}x)", 8, "TREND")
    elif cf.tlt_corr < -0.4 and getattr(ctx, 'tlt_trend', 1.0) < 0: add_trigger("利率敏感度(TLT相关性)", "🛡️ [宏观防御] 与长期国债(TLT)高度负相关且美债走弱，具备抗息避险属性 (权:{fw:.2f}x)", 6, "TREND")
    if cf.dxy_corr < -0.4 and getattr(ctx, 'dxy_trend', -1.0) < 0: add_trigger("汇率传导(DXY相关性)", "💱 [宏观映射] 与美元指数(DXY)强负相关且弱美元周期确立，受惠于全球流动性释放 (权:{fw:.2f}x)", 8, "QUANTUM")
    elif cf.dxy_corr > 0.4 and getattr(ctx, 'dxy_trend', -1.0) > 0: add_trigger("汇率传导(DXY相关性)", "💵 [宏观映射] 与美元指数(DXY)强正相关且美元走强，具备强汇率避险属性 (权:{fw:.2f}x)", 6, "QUANTUM")
        
    dist_52w = stock.curr['Dist_52W_High'] if pd.notna(stock.curr['Dist_52W_High']) else 0.0
    if dist_52w > Config.Params.DIST_52W and cf.weekly_bullish: add_trigger("52周高点距离(动能延续)", "🏔️ [动能延续] 逼近 52 周新高且周线多头，上方无抛压阻力真空区 (权:{fw:.2f}x)", 10, "TREND")
        
    amihud_val = stock.curr['Amihud'] if pd.notna(stock.curr['Amihud']) else 0.0
    if amihud_val > Config.Params.AMIHUD_ILLIQ and macro_gravity: add_trigger("Amihud非流动性(冲击成本)", "⚠️ [流动性枯竭] 宏观高压下 Amihud 冲击成本显著放大，极易发生踩踏被降权 (权:{fw:.2f}x)", -10, "VOLATILITY")
    if cf.vrp > Config.Params.VRP_EXTREME: add_trigger("波动率风险溢价(VRP)", f"🌋 [风险溢价] VRP归一化极度飙升(溢价率>{cf.vrp*100:.1f}%)，期权市场定价极端恐慌，捕捉极值底 (权:{{fw:.2f}}x)", 12, "QUANTUM")

    if alt.nlp_score > 0.3:
        nlp_pts = 10 * min(1.0, alt.nlp_score / 0.8)
        add_trigger("舆情NLP情感极值(News_NLP)", f"📰 [舆情引擎] VADER-Lite 分析判定近期新闻呈现极度狂热 (强度:{nlp_pts/10:.1%} 权:{{fw:.2f}}x)", nlp_pts, "VOLATILITY")
    elif alt.nlp_score < Config.Params.NLP_BEAR: add_trigger("舆情NLP情感极值(News_NLP)", f"⚠️ [舆情崩塌] 探测到新闻包含密集利空/诉讼词汇 (复合得分:{alt.nlp_score:.2f} 权:{{fw:.2f}}x)", -10, "VOLATILITY")
    if alt.wsb_accel > Config.Params.WSB_ACCEL: add_trigger("散户热度加速度(WSB_Accel)", f"🔥 [散户加速] Reddit/WSB 提及数二阶导数飙升 (加速度:{alt.wsb_accel:.0f}/日²)，游资加速抬轿 (权:{{fw:.2f}}x)", 8, "VOLATILITY")

    if alt.pcr > Config.Params.PCR_BEAR: add_trigger("期权PutCall情绪(PCR)", "⚠️ [极端避险] Put/Call Ratio 爆表，期权市场极度看空避险 (权:{fw:.2f}x)", 15, "SHORT"); black_swan_risk = True
    elif 0 < alt.pcr < Config.Params.PCR_BULL: add_trigger("期权PutCall情绪(PCR)", "🔥 [极端贪婪] Put/Call Ratio 极低，期权市场聪明钱疯狂做多 (权:{fw:.2f}x)", 10, "QUANTUM")
    if alt.iv_skew > Config.Params.IV_SKEW_BEAR: add_trigger("隐含波动率偏度(IV Skew)", "🚨 [黑天鹅警报] 看跌期权隐含波动率畸高，内幕资金正疯狂买入下行保护！ (权:{fw:.2f}x)", 20, "SHORT"); black_swan_risk = True
    elif alt.iv_skew < Config.Params.IV_SKEW_BULL: add_trigger("隐含波动率偏度(IV Skew)", "🚀 [上行爆发] 看涨期权隐含波动率压倒性占优，主力预期剧烈向上重估 (权:{fw:.2f}x)", 12, "QUANTUM")
    if alt.short_change > Config.Params.SHORT_SQZ_CHG and alt.short_float > Config.Params.SHORT_SQZ_FLT and stock.curr['Close'] > stock.curr['EMA_20']: add_trigger("做空兴趣突变(轧空)", "💥 [轧空引擎] 近期做空兴趣激增且技术面多头，随时触发空头踩踏平仓的 Short Squeeze (权:{fw:.2f}x)", 15, "QUANTUM")

    # 宏观引力场惩罚项
    if ctx.is_credit_risk_high:
        add_trigger("宏观重力场(Credit_Risk)", "🚨 [系统性风险] 信用债市场出现剧烈抛售，根据量化上帝视角，此处必须降权避险 (权:{fw:.2f}x)", 15, "SHORT")
        
    # 子池 Max-Pooling 聚合，解决多重共线性
    for st_key, scores in sub_theme_scores.items():
        theme = st_key.split('_')[0]
        if scores:
            scores.sort(reverse=True)
            pooled_score = scores[0] + sum(s * 0.3 for s in scores[1:]) # 取最大值，其余衰减为 30%
            theme_scores[theme] += pooled_score

    # [架构优化] Soft-Clipping 抑制单极分化
    if theme_scores['TREND'] > 40:
        theme_scores['TREND'] = 40 + np.log1p(theme_scores['TREND'] - 40) * 10
        
    positive_themes = ['TREND', 'VOLATILITY', 'REVERSAL', 'QUANTUM']
    saturated_theme_sum = sum([50.0 * (1 - np.exp(-theme_scores[t] / 25.0)) for t in positive_themes if theme_scores[t] > 0])
    
    final_score_saturated = 100.0 * (1 - np.exp(-saturated_theme_sum / 50.0))
    if alt.pcr > Config.Params.PCR_BEAR or alt.iv_skew > Config.Params.IV_SKEW_BEAR or ctx.is_credit_risk_high: black_swan_risk = True
    
    # [黑天鹅一票否决权] 彻底摧毁带病股票的分数
    if black_swan_risk: final_score_saturated *= 0.1
        
    return int(final_score_saturated), triggered_list, factors_list, black_swan_risk

def _apply_market_filters(curr, prev, sym, base_score, sig, a, b, c):
    if curr['RSI'] > 80.0: return max(0, base_score - 50), False, sig + ["⚠️ 极端超买"]
    return base_score, False, sig

def _build_market_context() -> MarketContext:
    qqq_df = safe_get_history(Config.INDEX_ETF, "1y", "1d", fast_mode=True)
    # ✅ 宏观重力场升级: 引入 HYG/IEI 信用利差模型
    spy_df = safe_get_history("SPY", "1y", "1d", fast_mode=True)
    tlt_df = safe_get_history("TLT", "1y", "1d", fast_mode=True)
    dxy_df = safe_get_history("DX-Y.NYB", "1y", "1d", fast_mode=True)
    hyg_df = safe_get_history("HYG", "1y", "1d", fast_mode=True)
    iei_df = safe_get_history("IEI", "1y", "1d", fast_mode=True)
    tlt_trend = 1.0 if not tlt_df.empty and tlt_df['Close'].iloc[-1] > tlt_df['Close'].rolling(50).mean().iloc[-1] and tlt_df['Close'].rolling(20).mean().iloc[-1] > tlt_df['Close'].rolling(50).mean().iloc[-1] else -1.0
    dxy_trend = 1.0 if not dxy_df.empty and dxy_df['Close'].iloc[-1] > dxy_df['Close'].rolling(50).mean().iloc[-1] and dxy_df['Close'].rolling(20).mean().iloc[-1] > dxy_df['Close'].rolling(50).mean().iloc[-1] else -1.0

    vix, vix_desc = get_vix_level(qqq_df)
    regime, regime_desc, _, _, _ = get_market_regime()
    
    # 计算信用利差动量 (Credit Spread Momentum)
    credit_risk_high = False
    credit_spread_mom = 0.0
    if not hyg_df.empty and not iei_df.empty:
        ratio = hyg_df['Close'] / iei_df['Close']
        credit_spread_mom = float(ratio.pct_change(20).iloc[-1]) # 20日动量
        if credit_spread_mom < -0.02: credit_risk_high = True # 信用收缩警报
    
    exp = 1.0
    if regime == "bear": exp = 0.3
    elif regime == "hidden_bear": exp = 0.5
    elif regime == "rebound": exp = 0.6
    if credit_risk_high: exp *= 0.5

    return MarketContext(
        regime=regime, regime_desc=regime_desc, w_mul=1.0, xai_weights={}, vix_current=vix, 
        vix_desc=vix_desc, vix_scalar=1.0, max_risk=0.015, macro_gravity=credit_risk_high, 
        is_credit_risk_high=credit_risk_high, vix_inv=False, qqq_df=qqq_df, 
        macro_data={'spy': spy_df, 'tlt': tlt_df, 'dxy': dxy_df, 'hyg': hyg_df, 'iei': iei_df}, 
        total_market_exposure=exp, 
        health_score=0.6 if credit_risk_high else 1.0, 
        pain_warning="🚨 [宏观重力] 探测到信用利差急剧扩大，系统已进入防御态势" if credit_risk_high else "", 
        credit_spread_mom=credit_spread_mom, dynamic_min_score=8.5 if credit_risk_high else 8.0,
        tlt_trend=tlt_trend, dxy_trend=dxy_trend
    )

# ================= 🚀 两阶段并行引擎工作函数 (顶层定义防 Pickling 崩溃) =================
async def _io_fetch_worker_async(sym: str) -> Optional[dict]:
    """[异步版] 阶段1: 纯 IO 并行拉取工作节点"""
    try:
        # [IO 优化] 仅拉取长周期基础日线与短周期小时线，切断冗余并发网络风暴
        hist_tasks = [
            asyncio.to_thread(safe_get_history, sym, "5y", "1d", True),
            asyncio.to_thread(safe_get_history, sym, "60d", "60m", True)
        ]
        
        # 并发拉取替代数据与舆情
        other_tasks = [
            safe_get_sentiment_data_async(sym),
            safe_get_alt_data_async(sym)
        ]
        
        results = await asyncio.gather(*hist_tasks, *other_tasks)
        df_5y, df_60m, sentiment, alt_data = results
        
        if df_5y is None or len(df_5y) < 60: return None
        
        # [CPU 本地重采样] 用 Pandas 内存换网络 IO，速度提升极高
        try:
            df_w = df_5y.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            df_m = df_5y.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
        except Exception:
            df_w, df_m = pd.DataFrame(), pd.DataFrame()
            
        # [IPO 保护] 让主 df 获得完整 5y 历史，为 SMA_200 提供充足纵深计算
        df = df_5y.copy()
        
        return {
            'sym': sym, 'df': df, 
            'df_w': df_w, 'df_m': df_m, 'df_60m': df_60m if df_60m is not None else pd.DataFrame(),
            'sentiment': sentiment, 'alt_data': alt_data
        }
    except Exception as e:
        logger.debug(f"Async IO Fetch error for {sym}: {e}")
        return None

def _cpu_calc_worker(payload: dict) -> Optional[dict]:
    """阶段2: 纯 CPU 计算工作节点"""
    try:
        sym = payload['sym']
        df = payload['df']
        ctx = payload['ctx']
        
        # ✅ 修复：接收来自 IO 阶段的多周期真实数据
        df_w = payload.get('df_w', pd.DataFrame())
        df_m = payload.get('df_m', pd.DataFrame())
        df_60m = payload.get('df_60m', pd.DataFrame())
        
        df_ind = calculate_indicators(df)
        curr, prev = df_ind.iloc[-1], df_ind.iloc[-2]
        
        # ✅ 修复：计算真实的成交量异动与近期波段高点，防止因子判断失真
        is_vol = bool((curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open']))
        swing_high_10 = float(df_ind['High'].iloc[-11:-1].max()) if len(df_ind) >= 11 else float(curr['High'])
        
        stock = StockData(sym, df_ind, df_w, df_m, df_60m, curr, prev, is_vol, swing_high_10)
        cf = _extract_complex_features(stock, ctx)
        alt = AltData(*payload['sentiment'][:4], *payload['alt_data'][:3], 0.0)
        seq = _get_transformer_seq(df_ind)
        # ✅ 致命修复: 严格遵守 Directive 1 戒律，绝不在 CPU Worker 回传完整的 DataFrame
        # 在完成特征提取后，必须卸载 StockData 中携带的所有重型 DataFrame，只保留标量给主进程使用
        stock.df = pd.DataFrame()
        stock.df_w = pd.DataFrame()
        stock.df_m = pd.DataFrame()
        stock.df_60m = pd.DataFrame()
        
        return {
            'sym': sym, 'stock': stock, 'alt': alt, 'cf': cf, 'seq': seq, 
            'curr': curr, 'prev': prev, 'news': "", 
            'close_history': df['Close'].tail(60).values
        }
    except Exception as e:
        return None

async def _stress_test_io_worker_async(sym: str) -> Optional[dict]:
    """[异步版] 压测阶段1: 历史 IO 拉取"""
    df = await asyncio.to_thread(safe_get_history, sym, "5y", "1d", True)
    if len(df) < 500: return None
    return {'sym': sym, 'df': df}

def _stress_test_cpu_worker(payload: dict) -> dict:
    """压测阶段2: 纯 CPU 并行扰动与回测"""
    sym, df_raw, ctx = payload['sym'], payload['df'], payload['ctx']
    df_phase, df_noise = _decompose_and_perturb(df_raw)
    local_metrics = {'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}}
    
    for env_name, df_env in {'Original': df_raw, 'Phase_Chaos': df_phase, 'Noise_Explosion': df_noise}.items():
        try:
            df_w = df_env.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            df_m = df_env.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            df_ind = calculate_indicators(df_env)
            
            for i in range(len(df_ind) - 252, len(df_ind) - 5):
                curr, prev = df_ind.iloc[i], df_ind.iloc[i-1]
                if (curr['ATR'] / curr['Close'] > 0.15) or (pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']): continue
                is_vol = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open'])
                swing_high_10 = df_ind['High'].iloc[i-10:i].max() if i >= 10 else curr['High']
                stock_data = StockData(sym, df_ind.iloc[:i+1], df_w, df_m, pd.DataFrame(), curr, prev, is_vol, swing_high_10)
                
                if _evaluate_omni_matrix(stock_data, ctx, _extract_complex_features(stock_data, ctx), AltData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))[0] >= Config.Params.MIN_SCORE_THRESHOLD:
                    ret = (df_ind['Close'].iloc[i+4] - df_ind['Open'].iloc[i+1]) / df_ind['Open'].iloc[i+1]
                    local_metrics[env_name]['trades'] += 1
                    local_metrics[env_name]['ret_sum'] += ret
                    if ret > 0: local_metrics[env_name]['wins'] += 1
        except Exception: pass
    return local_metrics

# ================= 数据预备引擎 (异步升级版: IO/CPU 解锁) =================
async def _prepare_universe_data_async(ctx: MarketContext) -> Tuple[List[dict], dict]:
    prepared_data = []
    symbols = get_filtered_watchlist(max_stocks=200)
    logger.info(f"🚀 启动架构终极版: asyncio 与 多进程配合的混合动力引擎 (扫描数: {len(symbols)})...")
    
    # --- 阶段 1: 极高并发异步 IO 拉取 (彻底抛弃线程池) ---
    tasks = [_io_fetch_worker_async(sym) for sym in symbols]
    io_results_raw = await asyncio.gather(*tasks)
    io_results = [res for res in io_results_raw if res]

    if not io_results: return [], {}

    # ✅ 致命 IPC 隐患修复：提纯 Context 为无锁无 DataFrame 的轻量化标量对象
    ctx_lite = _build_ctx_lite(ctx)
    for res in io_results:
        res['ctx'] = ctx_lite
        
    # --- 阶段 2: 多进程池纯 CPU 计算 (破解 GIL) ---
    ctx_mp = multiprocessing.get_context("forkserver") if hasattr(multiprocessing, "get_context") else None
    
    # ✅ 完美平滑分布式限速器：计算单核配额并挂载到 initializer
    NUM_CPU_WORKERS = min(4, os.cpu_count() or 1)
    RATE_PER_WORKER = max(0.5, 4.0 / NUM_CPU_WORKERS)  # 防止除 0 且设定最低底线
    
    pool_kwargs = {
        "max_workers": NUM_CPU_WORKERS,
        "initializer": _worker_pool_initializer,
        "initargs": (RATE_PER_WORKER, 4.0)
    }
    if sys.platform != "win32" and ctx_mp:
        pool_kwargs["mp_context"] = ctx_mp

    logger.info(f"⚡ 启动多进程 CPU 矩阵运算，跨进程并行调度核心数: {pool_kwargs['max_workers']} ...")
    with concurrent.futures.ProcessPoolExecutor(**pool_kwargs) as cpu_executor:
        # ✅ 修复: 移除脆弱的 map，改用 submit + as_completed 实现单节点崩溃熔断隔离
        future_map = {cpu_executor.submit(_cpu_calc_worker, payload): payload['sym'] for payload in io_results}
        for future in concurrent.futures.as_completed(future_map, timeout=300.0):
            sym = future_map[future]
            try:
                res = future.result(timeout=60.0)
                if res: prepared_data.append(res)
            except concurrent.futures.TimeoutError:
                logger.warning(f"⚠️ CPU Worker {sym} 计算超时，已被系统安全跳过。")
            except Exception as e:
                logger.warning(f"⚠️ CPU Worker {sym} 遭遇崩溃 ({type(e).__name__}: {e})，已隔离。")

    return prepared_data, {d['sym']: d['close_history'] for d in prepared_data}

def _apply_ai_inference(reports: List[dict], ctx: MarketContext) -> List[dict]:
    """AI 推理引擎: 将 Transformer 右脑逻辑从占位符升级为实时模型推演"""
    if not TRANSFORMER_AVAILABLE or not os.path.exists(Config.TRANSFORMER_PTH):
        # 降级方案：如果没有模型，则使用基于技术面权重的概率偏移
        for r in reports: r['ai_prob'] = 0.45 + (r['score'] / 100.0) * 0.2
        return reports
    
    try:
        # 热加载生产权重
        device = torch.device('cpu') # ARM CPU 上使用 CPU 推理，避免 GPU 调度开销
        model = QuantAlphaTransformer.load_checkpoint(Config.TRANSFORMER_PTH, device=device)
        
        for r in reports:
            seq = r.get('seq')
            if seq is not None and isinstance(seq, np.ndarray) and seq.shape == (60, 49):
                # 注入时序特征序列，提取 16 维高维 Alpha 嵌入
                alpha_vec = model.extract_alpha(seq)
                # 策略逻辑：计算 Alpha 向量的强度作为胜率参考 (映射至 0.4-0.9 之间)
                # 在对比学习中，向量模长反映了特征的确定性
                confidence = float(np.tanh(np.linalg.norm(alpha_vec) / 4.0))
                r['ai_prob'] = 0.5 + confidence * 0.4
            else:
                r['ai_prob'] = 0.5
    except Exception as e:
        logger.error(f"⚠️ [AI 推理崩溃] 无法激活右脑引擎: {e}")
        for r in reports: r['ai_prob'] = 0.5
    return reports

def _calculate_position_size(stock: StockData, ctx: MarketContext, ai_prob: float, div: float, risk: float):
    """风控决策引擎: 从固定比例升级为基于 ATR 与吊灯线 (Chandelier Exit) 的动态移动止损"""
    c = stock.curr['Close']
    atr = stock.curr.get('ATR', c * 0.02) if pd.notna(stock.curr.get('ATR')) else c * 0.02
    
    # 动态止盈：设定 3.5x 极限空间止盈
    tp_price = c + (atr * 3.5)
    
    # [利润回撤防线] 结合硬底线止损与吊灯移动止损
    hard_sl = c - (atr * 2.0)
    chandelier_sl = stock.curr.get('Chandelier_Exit', hard_sl)
    if pd.isna(chandelier_sl) or chandelier_sl == 0.0: chandelier_sl = hard_sl
    
    # 取硬底线与吊灯线中较高的一个作为最终防线，实现只进不退的动态利润锁定
    sl_price = max(hard_sl, chandelier_sl)
    
    # 根据趋势强度修正胜率预估
    adj_prob = ai_prob
    sma_200 = stock.curr.get('SMA_200', 0)
    if pd.notna(sma_200) and c > sma_200: adj_prob += 0.03 # 顺大势加分
    
    return float(tp_price), float(sl_price), float(adj_prob), ""

def _apply_kelly_cluster_optimization(reports: List[dict], pd_dict, exp, ctx):
    """资金管理引擎: 从固定仓位升级为动态凯利公式 (Kelly Criterion) 映射"""
    if not reports: return []
    
    for r in reports:
        p = r.get('ai_prob', 0.5)
        # 简化版凯利公式：Weight = (p * b - q) / b，此处 b (盈亏比) 假设为 1.5
        # 结果乘以 0.2 的缩放系数以保持稳健性 (Fractional Kelly)
        b = 1.5
        k_val = (p * (b + 1) - 1) / b
        k_weight = max(0.02, min(0.12, k_val * 0.2)) # 限制单标的仓位在 2% - 12% 之间
        
        r['opt_weight'] = k_weight
        r['pos_advice'] = f"✅ 凯利权重: {k_weight*100:.1f}%"
        
    # [架构优化] AI 右脑与量化左脑的集成排序 (Ensemble)
    final_pool = sorted(reports, key=lambda x: x.get('ai_prob', 0) * 100.0 + x.get('score', 0), reverse=True)[:5]
    return final_pool



def _generate_and_send_matrix_report(final_reports: List[dict], final_shadow_pool: List[dict], ctx: MarketContext) -> None:
    txts = []
    for idx, r in enumerate(final_reports):
        icon = ['🥇', '🥈', '🥉'][idx] if idx < 3 else '🔸'
        ai_prob = r.get('ai_prob', 0)
        ai_display = f"🔥 **{ai_prob:.1%}**" if ai_prob > 0.60 else f"{ai_prob:.1%}"
        sigs_str = "\n".join([f"- {s}" for s in r.get("signals", [])])
        news_str = f"\n- 📰 {r['news']}" if r.get('news') else ""
        report_block = (
            f"### {icon} **{r['symbol']}** | 🤖 分层元学习胜率: {ai_display} | 🌟 终极评级: {r['score']}分\n"
            f"**💡 机构交易透视:**\n{sigs_str}{news_str}\n\n"
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
                        limit_p = round(r['curr_close'] * 1.002, 2)
                        conn.execute('''INSERT INTO orders (client_oid, symbol, side, qty, order_type, limit_price, arrival_price, status) VALUES (?, ?, 'BUY', ?, 'LMT', ?, ?, 'PENDING_SUBMIT')''', (f"QB_{uuid.uuid4().hex[:8]}", r['symbol'], qty, limit_p, r['curr_close']))
                        logger.info(f"📡 [网关路由] 成功下发 {r['symbol']} BUY 指令: {qty} 股 -> PENDING_SUBMIT (Pegged Limit: {limit_p})")
    except Exception as e: logger.error(f"❌ [网关路由] IPC 致命错误: {e}")

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
        if o and o["status"] == "OPEN" and time.time() - o["ts"] > 0.5: o["status"] = "FILLED"
        return BrokerOrder(broker_oid, o["status"] if o else "REJECTED", o["qty"] if o and o["status"]=="FILLED" else 0.0, o["price"] if o else 0.0)

class OrderLedger:
    def __init__(self, db_path):
        self.db_path = db_path
        with self._get_conn() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS orders (client_oid TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, order_type TEXT, limit_price REAL, arrival_price REAL, status TEXT, filled_qty REAL DEFAULT 0, avg_fill_price REAL DEFAULT 0, retry_count INTEGER DEFAULT 0, broker_oid TEXT)''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_status ON orders(status)')
    
    def _get_conn(self):
        c = sqlite3.connect(self.db_path, timeout=10.0)
        c.row_factory = sqlite3.Row
        c.execute('pragma journal_mode=wal')
        c.execute('pragma synchronous=NORMAL')
        c.execute('pragma temp_store=MEMORY')
        c.execute('pragma mmap_size=300000000')
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
                    logger.info(f"📤 提交: {r['side']} {r['qty']} {r['symbol']} ({r['order_type']} 限价: {r.get('limit_price', 'MKT')})")
                    bo = self.broker.submit_order(r['symbol'], r['side'], r['qty'], r['order_type'], limit_price=r.get('limit_price'), client_oid=r['client_oid'])
                    self.ledger.update(r['client_oid'], bo.status if bo else 'REJECTED', boid=bo.broker_oid if bo else None)
                except Exception: self.ledger.update(r['client_oid'], 'REJECTED')
            for _, r in self.ledger.fetch_status(['OPEN']).iterrows():
                if r['broker_oid']:
                    try:
                        bo = self.broker.fetch_order(r['broker_oid'])
                        self.ledger.update(r['client_oid'], bo.status, bo.filled_qty, bo.avg_fill_price)
                        if bo.status == 'FILLED':
                            tca = {"client_oid": r['client_oid'], "symbol": r['symbol'], "side": r['side'], "qty": bo.filled_qty, "arrival_price": r['arrival_price'], "execution_price": bo.avg_fill_price, "timestamp": datetime_to_str(), "slippage_bps": abs(bo.avg_fill_price - r['arrival_price'])/(r['arrival_price']+1e-10)*10000.0 * (-1 if r['side']=='SELL' else 1)}
                            with open(Config.TCA_LOG_PATH, 'a') as f: f.write(json.dumps(tca) + '\n')
                    except Exception: pass
            time.sleep(0.2)

# ================= 6. 主程序入口与路由 (全量恢复) =================
def run_tech_matrix():
    ctx = _build_market_context()
    # ✅ 架构迁移：正式接入异步并发数据引擎
    prep, hist = asyncio.run(_prepare_universe_data_async(ctx))
    if not prep: return
    
    # 移除旧版冗余推演逻辑，统一在下文的 _apply_ai_inference 中处理
    raw = []
    for d in prep:
        ml_features = _extract_ml_features(d['stock'], ctx, d['cf'], d['alt'], d.get('alpha_vec', np.zeros(16)))
        score, sig, factors, black_swan = _evaluate_omni_matrix(d['stock'], ctx, d['cf'], d['alt'])
        score, is_bearish_div, sig = _apply_market_filters(d['curr'], d['prev'], d['sym'], score, sig, [], [], [])
        
        raw.append({
            'sym': d['sym'], 'curr': d['curr'], 'prev': d['prev'], 'score': score, 
            'sig': sig, 'factors': factors, 'ml_features': ml_features, 'news': d['news'], 
            'sym_sec': Config.get_sector_etf(d['sym']), 'is_bearish_div': is_bearish_div, 
            'black_swan_risk': black_swan, 'total_score': score, 'is_untradeable': False, 
            'ai_prob': 0.0
        })
        
    raw = _apply_ai_inference(raw, ctx)
    
    reps, background_pool, all_raw_scores = [], [], []
    for r in raw:
        if r['total_score'] > 0: all_raw_scores.append(r['total_score'])
        
        tp_val, sl_val, adj_prob, basic_advice = _calculate_position_size(
            StockData(r['sym'], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), r['curr'], r['prev'], False, 0.0), 
            ctx, r['ai_prob'], r['is_bearish_div'], r['black_swan_risk']
        )

        stock_data_pack = {
            "symbol": r['sym'], "score": r['total_score'], "ai_prob": adj_prob, "signals": r['sig'][:8], 
            "factors": r['factors'], "ml_features": r['ml_features'],
            "curr_close": float(r['curr']['Close']), "tp": float(tp_val), "sl": float(sl_val), 
            "news": r['news'], "sector": r['sym_sec'], "pos_advice": basic_advice
        }

        if not r['is_untradeable'] and r['total_score'] > 0 and adj_prob > 0:
            if check_earnings_risk(r['sym']):
                r['sig'].append("💣 [财报雷区] 近5日发财报,风险极高")
                r['total_score'] = int(r['total_score'] * 0.5)
                stock_data_pack["score"] = r['total_score']
            reps.append(stock_data_pack)
        else:
            background_pool.append(stock_data_pack)

    if reps and all_raw_scores:
        ctx.dynamic_min_score = max(Config.Params.MIN_SCORE_THRESHOLD, np.percentile(all_raw_scores, 85))
        reps = [r for r in reps if r['score'] >= ctx.dynamic_min_score]
        
        groups = defaultdict(list)
        for r in reps: groups[r["sector"]].append(r)
        
        for sec, stks in groups.items():
            if hasattr(Config, 'CROWDING_EXCLUDE_SECTORS') and sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.Params.CROWDING_MIN_STOCKS:
                pen = max(0.6, min(0.9, Config.Params.CROWDING_PENALTY * (1.0 + ctx.health_score * 0.3)))
                stks.sort(key=lambda x: x["score"], reverse=True)
                for s in stks[1:]: s["score"] = int(s["score"] * pen)

        freps = _apply_kelly_cluster_optimization(reps, hist, ctx.total_market_exposure, ctx)
        
        for r in freps: set_alerted(r["symbol"])
        
        final_symbols = {r['symbol'] for r in freps}
        unselected_background = [s for s in background_pool if s['symbol'] not in final_symbols]
        random.shuffle(unselected_background)
        final_shadow_pool = unselected_background[:150]
        
        for r in final_shadow_pool: set_alerted(r["symbol"], is_shadow=True, shadow_data=r)
            
        _generate_and_send_matrix_report(freps, final_shadow_pool, ctx)
        _route_orders_to_gateway(freps, ctx)
    else:
        logger.info("📭 本次矩阵扫描无标的突破 Top 15% 截面排位，宁缺毋滥，保持静默。")


def run_gateway():
    k = os.environ.get('ALPACA_API_KEY','')
    s = os.environ.get('ALPACA_API_SECRET','')
    u = os.environ.get('ALPACA_BASE_URL','https://paper-api.alpaca.markets')
    broker = AlpacaGateway(k, s, u) if k and s else MockAlpacaGateway()
    ExecutionEngine(broker).run()

# ================= 7. 回测引擎与抗噪验证 =================
def run_backtest_engine(replay_mode: bool = False) -> None:
    if replay_mode:
        log_files = [os.path.join(Config.DATA_DIR, 'backtest_log_replay.jsonl')]
    else:
        log_files = [f for f in os.listdir('.') if f.startswith('backtest_log') and f.endswith('.jsonl') and not f.endswith('replay.jsonl')]
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
        logger.info(f"⏳ 启动回测引擎：正在拉取 {len(syms)} 个标的轨迹...")
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
            if i + 40 < len(syms): time.sleep(random.uniform(1.0, 2.0))

        if not dfs: return
        df_all = pd.concat(dfs, axis=1)

        if isinstance(df_all.columns, pd.MultiIndex): df_c, df_o, df_h, df_l = df_all['Close'], df_all['Open'], df_all['High'], df_all['Low']
        else: df_c, df_o, df_h, df_l = df_all[['Close']].rename(columns={'Close': syms[0]}), df_all[['Open']].rename(columns={'Open': syms[0]}), df_all[['High']].rename(columns={'High': syms[0]}), df_all[['Low']].rename(columns={'Low': syms[0]})
            
        df_c.index, df_o.index, df_h.index, df_l.index = df_c.index.strftime('%Y-%m-%d'), df_o.index.strftime('%Y-%m-%d'), df_h.index.strftime('%Y-%m-%d'), df_l.index.strftime('%Y-%m-%d')
    except Exception: return
    
    stats_period, factor_rets, trades_with_ret, mae_mfe_records = {'T+1': [], 'T+3': [], 'T+5': []}, {}, [], {'T+1': [], 'T+3': [], 'T+5': []}
    ai_filtered_wins, ai_filtered_total, transformer_X, transformer_Y, cached_inds = 0, 0, [], [], {}
    
    logger.info(f"⚡ 启动全张量 NumPy 向量化回测引擎，解除 Python 级多重循环枷锁...")
    
    # 1. 核心数据阵列化 (Vectorized Data Preparation)
    c_arr, o_arr, h_arr, l_arr = df_c.values, df_o.values, df_h.values, df_l.values
    max_idx = len(c_arr) - 1
    
    # 对齐并提取所有有效 Trade 的降维索引
    valid_trades = []
    for t in trades:
        sym, r_dt = t['symbol'], t['date']
        if sym in df_c.columns:
            idx_pos = df_c.index.searchsorted(r_dt)
            if idx_pos < len(df_c) and df_c.index[idx_pos] >= r_dt:
                t['_e_idx'] = idx_pos
                t['_s_col'] = df_c.columns.get_loc(sym)
                valid_trades.append(t)

    if not valid_trades: return
    
    e_indices = np.array([t['_e_idx'] for t in valid_trades])
    sym_cols = np.array([t['_s_col'] for t in valid_trades])
    tp_arr = np.array([t.get('tp', np.inf) for t in valid_trades])
    sl_arr = np.array([t.get('sl', -np.inf) for t in valid_trades])
    
    # 2. 向量化入场计算
    entry_idxs = np.minimum(e_indices + 1, max_idx)
    entry_opens = o_arr[entry_idxs, sym_cols]
    prev_closes = c_arr[e_indices, sym_cols]
    
    # 布尔掩码剔除无效或停牌数据
    valid_mask = (entry_opens > 0) & ~np.isnan(entry_opens) & ~np.isnan(prev_closes)
    e_indices, sym_cols = e_indices[valid_mask], sym_cols[valid_mask]
    tp_arr, sl_arr = tp_arr[valid_mask], sl_arr[valid_mask]
    entry_opens, prev_closes = entry_opens[valid_mask], prev_closes[valid_mask]
    valid_trades = [valid_trades[i] for i in range(len(valid_trades)) if valid_mask[i]]
    
    if not valid_trades: return

    # 向量化滑点与成本惩罚
    gap_up = (entry_opens - prev_closes) / (prev_closes + 1e-10)
    slippage_array = np.where(gap_up > 0.03, Config.Params.SLIPPAGE * 3, Config.Params.SLIPPAGE)
    entry_costs = entry_opens * (1 + slippage_array + Config.Params.COMMISSION)
    
    # 3. 向量化时间窗口穿透 (T+1, T+3, T+5)
    for d in [1, 3, 5]:
        exit_idxs = np.minimum(e_indices + d, max_idx)
        
        # 提取出场价格
        exit_closes = c_arr[exit_idxs, sym_cols]
        
        # 批量计算区间 MFE/MAE (最大高点与最小低点跨度) - 彻底拔除列表推导式的纯 NumPy 向量化
        start_idxs = np.minimum(e_indices + 1, max_idx + 1)
        end_idxs = np.minimum(e_indices + d + 1, max_idx + 1)
        valid_win = (end_idxs - start_idxs) == d
        
        period_h = np.full(len(e_indices), np.nan)
        period_l = np.full(len(e_indices), np.nan)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # 核心大批量：高级花式索引与矩阵广播，O(1) 级耗时
            if np.any(valid_win):
                v_starts, v_cols = start_idxs[valid_win], sym_cols[valid_win]
                window_idx = v_starts[:, None] + np.arange(d)[None, :]
                period_h[valid_win] = np.nanmax(h_arr[window_idx, v_cols[:, None]], axis=1)
                period_l[valid_win] = np.nanmin(l_arr[window_idx, v_cols[:, None]], axis=1)
            
            # 边界小批量：末端数据截断时的安全回退
            invalid_idxs = np.where(~valid_win)[0]
            for i in invalid_idxs:
                if start_idxs[i] < end_idxs[i]:
                    sl_slice = slice(start_idxs[i], end_idxs[i])
                    period_h[i] = np.nanmax(h_arr[sl_slice, sym_cols[i]])
                    period_l[i] = np.nanmin(l_arr[sl_slice, sym_cols[i]])
        
        # --- 🚀 Gen-3: 注入底层动态止盈止损 (TP/SL) 截断引擎 ---
        actual_exit_prices = exit_closes.copy()
        hit_sl = period_l <= sl_arr
        hit_tp = period_h >= tp_arr
        
        # 防止缺口套利：如果买入价已经低于 SL，则按最低价结算
        actual_exit_prices = np.where(hit_sl, np.minimum(sl_arr, entry_opens), actual_exit_prices)
        actual_exit_prices = np.where(~hit_sl & hit_tp, np.maximum(tp_arr, entry_opens), actual_exit_prices)
        
        exit_revenues = actual_exit_prices * (1 - Config.Params.SLIPPAGE - Config.Params.COMMISSION)
        returns = (exit_revenues - entry_costs) / (entry_costs + 1e-10) * np.where(gap_up > 0.03, 0.5, 1.0)
        
        # 4. 指标回填与右脑缓存投喂 (极速降维循环)
        for idx, ret in enumerate(returns):
            if np.isnan(ret): continue
            t = valid_trades[idx]
            mae_mfe_records[f'T+{d}'].append({'ret': float(ret), 'mfe': float((period_h[idx] - entry_costs[idx]) / entry_costs[idx]), 'mae': float((period_l[idx] - entry_costs[idx]) / entry_costs[idx])})
            stats_period[f'T+{d}'].append(float(ret))
            
            if d == 5:
                sym, r_dt = t['symbol'], t['date']
                if TRANSFORMER_AVAILABLE:
                    try:
                        if sym not in cached_inds: cached_inds[sym] = calculate_indicators(df_all.xs(sym, level=1, axis=1) if isinstance(df_all.columns, pd.MultiIndex) else pd.DataFrame({'Close': df_c[sym], 'Open': df_o[sym], 'High': df_h[sym], 'Low': df_l[sym], 'Volume': 1000000}))
                        ind_df = cached_inds[sym]
                        valid_dates = ind_df.index[ind_df.index <= r_dt]
                        if len(valid_dates) > 0:
                            t_idx = np.where(ind_df.index.get_loc(valid_dates[-1]))[0][-1] if isinstance(ind_df.index.get_loc(valid_dates[-1]), np.ndarray) else (ind_df.index.get_loc(valid_dates[-1]).stop - 1 if isinstance(ind_df.index.get_loc(valid_dates[-1]), slice) else ind_df.index.get_loc(valid_dates[-1]))
                            seq = _get_transformer_seq(ind_df, end_idx=t_idx + 1)
                            if not np.all(seq == 0): transformer_X.append(seq); transformer_Y.append(ret) 
                    except Exception: pass
                
                factor_list = t.get('factors', [])
                if not factor_list:
                    for sig_txt in t.get('signals', []):
                        m = re.search(r'\[(.*?)\]', sig_txt)
                        if m: factor_list.append(m.group(1).split(" ")[0])
                for f_name in factor_list: factor_rets.setdefault(f"[{f_name}]", []).append(float(ret))
                    
                if t.get('ai_prob', 0.0) >= 0.50:  
                    ai_filtered_total += 1
                    if ret > 0: ai_filtered_wins += 1
                    
                trades_with_ret.append({'date': t['date'], 'symbol': t.get('symbol', 'UNKNOWN'), 'vix': t['vix'], 'cred': t['cred'], 'term': t['term'], 'pcr': t['pcr'], 'ml_features': t.get('ml_features', {}), 'factors': factor_list, 'ret': float(ret)})
    
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
                if active_A: 
                    X = X_A[train_idx]
                    if X.shape[1] > 0: oof_pred_A[val_idx] = LGBMClassifier(**lgbm_params).fit(X, y_all_class[train_idx]).predict_proba(X_A[val_idx])[:, 1]
                    else: oof_pred_A[val_idx] = 0.5
                else: oof_pred_A[val_idx] = 0.5
                if active_B: 
                    X = X_B[train_idx]
                    if X.shape[1] > 0: oof_pred_B[val_idx] = LGBMClassifier(**lgbm_params).fit(X, y_all_class[train_idx]).predict_proba(X_B[val_idx])[:, 1]
                    else: oof_pred_B[val_idx] = 0.5
                else: oof_pred_B[val_idx] = 0.5
                
            valid_meta_idx = np.where(oof_pred_A > 0)[0]
            if len(valid_meta_idx) > 10:
                meta_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42).fit(np.column_stack([oof_pred_A[valid_meta_idx], oof_pred_B[valid_meta_idx], vix_all[valid_meta_idx], cred_all[valid_meta_idx], term_all[valid_meta_idx], pcr_all[valid_meta_idx]]), y_all_class[valid_meta_idx])
                exp_wA, exp_wB = np.exp(meta_clf.coef_[0][0]), np.exp(meta_clf.coef_[0][1])
                meta_weights_dict['Group_A'], meta_weights_dict['Group_B'] = exp_wA / (exp_wA + exp_wB), exp_wB / (exp_wA + exp_wB)
            else:
                meta_clf = LogisticRegression(class_weight='balanced').fit(np.column_stack([oof_pred_A, oof_pred_B, vix_all, cred_all, term_all, pcr_all]), y_all_class)

            final_base_A = LGBMClassifier(**lgbm_params).fit(X_A, y_all_class) if active_A and X_A.shape[1] > 0 else None
            final_base_B = LGBMClassifier(**lgbm_params).fit(X_B, y_all_class) if active_B and X_B.shape[1] > 0 else None
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

    # === 🚀 组合净值曲线生成 (Portfolio-Level Backtest) ===
    portfolio_metrics = {}
    if len(trades_with_ret) >= 10:
        pf_df = pd.DataFrame(trades_with_ret).sort_values('date')
        if 'symbol' in pf_df.columns:
            capital = 100000.0  # 初始模拟资金 10w
            max_pos = 0.1       # 单只最高 10% 仓位
            equity_data = []
            for date_str, group in pf_df.groupby('date'):
                n_trades = len(group)
                alloc = min(max_pos, 1.0 / n_trades) if n_trades > 0 else 0
                avg_ret = group['ret'].mean()
                daily_ret = alloc * n_trades * avg_ret
                capital *= (1 + daily_ret)
                equity_data.append({'date': date_str, 'equity': capital, 'daily_ret': daily_ret})
            
            if equity_data:
                eq_df = pd.DataFrame(equity_data)
                total_ret = (capital / 100000.0) - 1.0
                peak = eq_df['equity'].cummax()
                drawdown = (eq_df['equity'] - peak) / peak
                max_dd = drawdown.min()
                sharpe = np.sqrt(252) * eq_df['daily_ret'].mean() / (eq_df['daily_ret'].std() + 1e-9)
                calmar = total_ret / abs(max_dd) if max_dd < 0 else 99.0
                portfolio_metrics = {'total_ret': float(total_ret), 'max_dd': float(max_dd), 'sharpe': float(sharpe), 'calmar': float(calmar)}
                
                # --- [AGENT_FITNESS_SCORE] 输出 ---
                fitness_score = (sharpe * 10.0) + (total_ret * 100.0) - abs(max_dd * 50.0)
                print(f"\n[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\n")
                with open(os.path.join(Config.DATA_DIR, "fitness_score.txt"), "w") as f:
                    f.write(f"[AGENT_FITNESS_SCORE]: {fitness_score:.4f}")

    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump({"overall": res, "factors": f_res, "xai_importances": feature_importances_dict, "meta_weights": meta_weights_dict, "attribution": attr_report, "factor_ic": factor_ic_report, "portfolio": portfolio_metrics}, f, indent=4)
    
    report_md = [f"# 📈 自动量化战报与 AI 透视\n**更新:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"]
    
    if portfolio_metrics:
        report_md.append(f"## 🏦 账户组合净值仿真 (Portfolio Simulation)\n*采用期末结算法，单标的最大仓位限制 10%*\n- **总资金回报率 (Total Return)**: `{portfolio_metrics['total_ret']*100:+.2f}%`\n- **最大资金回撤 (Max Drawdown)**: `{portfolio_metrics['max_dd']*100:.2f}%`\n- **夏普比率 (Sharpe Ratio)**: `{portfolio_metrics['sharpe']:.2f}`\n- **卡玛比率 (Calmar Ratio)**: `{portfolio_metrics['calmar']:.2f}`\n")
        
    report_md.append("## ⚔️ 核心表现评估\n| 周期 | 原始胜率 | ⚡代谢演化过滤 | 均收益 | 盈亏比 | Sharpe | 胜单平均抗压(MAE) | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for p in ['T+1', 'T+3', 'T+5']:
        d = res.get(p, {'win_rate':0,'avg_ret':0,'profit_factor':0,'sharpe':0,'avg_win_mae':0,'max_cons_loss':0,'total_trades':0})
        ai_win_val = d.get('ai_win_rate', 0.0)
        ai_str = f"**{ai_win_val*100:.1f}%**" if 'ai_win_rate' in d else "-"
        report_md.append(f"| {p} | {d['win_rate']*100:.1f}% | {ai_str} | {d['avg_ret']*100:+.2f}% | {d['profit_factor']:.2f} | {d['sharpe']:.2f} | {d['avg_win_mae']*100:.1f}% | {d['total_trades']} |")
    
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
    
    alert_lines = ["### 📊 **机构级回测报表 (含代谢淘汰赛)**"]
    for p, d in res.items():
        ai_win_val = d.get('ai_win_rate', 0.0)
        ai_text = f" | ⚡代谢演化过滤: **{ai_win_val*100:.1f}%**" if 'ai_win_rate' in d else ""
        alert_lines.append(f"- **{p}:** 原始胜率 {d['win_rate']*100:.1f}%{ai_text} | 盈亏比 {d['profit_factor']:.2f}")
    send_alert("策略终极回测战报 (代谢进化版)", "\n".join(alert_lines))

def run_historical_replay(days: int = 252) -> None:
    logger.info(f"⏳ 启动历史重播引擎，回溯天数: {days}")
    ctx = _build_market_context()
    symbols = get_filtered_watchlist(max_stocks=40)
    
    dfs = []
    logger.info("📡 拉取全量历史数据...")
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i+40]
        df = yf.download(chunk, period="2y", progress=False, timeout=15)
        if not df.empty:
            if len(chunk) == 1 and not isinstance(df.columns, pd.MultiIndex): df.columns = pd.MultiIndex.from_product([df.columns, chunk])
            dfs.append(df)
            
    if not dfs: return
    df_all = pd.concat(dfs, axis=1)
    if isinstance(df_all.columns, pd.MultiIndex):
        df_c, df_o, df_h, df_l, df_v = df_all['Close'], df_all['Open'], df_all['High'], df_all['Low'], df_all['Volume']
    else: return
        
    log_path = os.path.join(Config.DATA_DIR, "backtest_log_replay.jsonl")
    with open(log_path, 'w', encoding='utf-8') as f: pass
    
    logger.info("⚙️ 预计算技术指标矩阵...")
    ind_dict = {}
    for sym in symbols:
        if sym in df_c.columns:
            sym_df = pd.DataFrame({'Close': df_c[sym], 'Open': df_o[sym], 'High': df_h[sym], 'Low': df_l[sym], 'Volume': df_v[sym]}).dropna()
            if len(sym_df) > 200: ind_dict[sym] = calculate_indicators(sym_df)
                
    if not ind_dict: return
    
    all_dates = sorted(list(ind_dict[list(ind_dict.keys())[0]].index))
    if len(all_dates) < days: days = len(all_dates) - 1
    
    logger.info("⏰ 开启时间旅行切片...")
    for i in range(-days, -1):
        dt = all_dates[i]
        date_str = dt.strftime('%Y-%m-%d')
        daily_trades = []
        
        for sym, sym_ind in ind_dict.items():
            try:
                dt_idx = sym_ind.index.get_loc(dt)
                if isinstance(dt_idx, int) and dt_idx > 100:
                    curr, prev = sym_ind.iloc[dt_idx], sym_ind.iloc[dt_idx-1]
                    if curr['Vol_MA20'] < 500000: continue
                    is_vol = bool((curr['Volume'] / (curr['Vol_MA20']+1e-9) > 1.5) and (curr['Close'] > curr['Open']))
                    swing_high_10 = float(sym_ind['High'].iloc[dt_idx-10:dt_idx].max())
                    
                    stock = StockData(sym, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), curr, prev, is_vol, swing_high_10)
                    cf = ComplexFeatures(False, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, False, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0) 
                    alt = AltData(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    
                    score, sig, factors, black_swan = _evaluate_omni_matrix(stock, ctx, cf, alt)
                    score, is_bearish_div, sig = _apply_market_filters(curr, prev, sym, score, sig, [], [], [])
                    
                    if score >= Config.Params.MIN_SCORE_THRESHOLD:
                        daily_trades.append({'symbol': sym, 'score': score, 'signals': sig, 'factors': factors, 'tp': float(curr['Close'] + 2.5 * curr['ATR']), 'sl': float(curr['Close'] - 1.5 * curr['ATR'])})
            except Exception: pass
            
        if daily_trades:
            top_picks = sorted(daily_trades, key=lambda x: x['score'], reverse=True)[:5]
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'date': date_str, 'vix': ctx.vix_current, 'top_picks': top_picks, 'shadow_pool': []}) + '\n')
                
    logger.info("✅ 历史重播生成完毕，启动自动验证引擎...")
    run_backtest_engine(replay_mode=True)

def _decompose_and_perturb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_phase, df_noise = df.copy(), df.copy()
    c = df['Close']
    trend, high_freq_noise, seasonal = c.ewm(span=60, adjust=False).mean(), c - c.ewm(span=5, adjust=False).mean(), c.ewm(span=5, adjust=False).mean() - c.ewm(span=60, adjust=False).mean()
    # ✅ 致命修复: 遵守 Directive 5 戒律，绝不使用全局 np.random，改用带数据特征 Seed 的局部 rng
    # 防止多进程 fork 后所有标的生成一模一样的相位错乱特征，污染压测池
    seed_val = int(np.sum(np.abs(c.values[-10:])) * 1e6) % (2**31)
    rng = np.random.default_rng(seed=seed_val)
    angles = rng.uniform(0, 2 * np.pi, len(np.fft.rfft(seasonal.values)))
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
    
    market_ctx = MarketContext(regime="bull", regime_desc="", w_mul=1.0, xai_weights=xai_weights, vix_current=18.0, vix_desc="", vix_scalar=1.0, max_risk=0.015, macro_gravity=False, is_credit_risk_high=False, vix_inv=False, qqq_df=pd.DataFrame(), macro_data={'spy': pd.DataFrame(), 'tlt': pd.DataFrame(), 'dxy': pd.DataFrame()}, total_market_exposure=1.0, health_score=1.0, pain_warning="", dynamic_min_score=8.0)
    
    metrics = {'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}, 'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}}
    
    # --- 压测阶段 1: 异步 IO 拉取 (全量迁移至 asyncio) ---
    symbols = get_filtered_watchlist(max_stocks=30)[:20]
    tasks = [_stress_test_io_worker_async(sym) for sym in symbols]
    io_results = [res for res in asyncio.run(asyncio.gather(*tasks)) if res]
    
    if not io_results:
        logger.warning("压测样本历史数据获取失败。")
        return

    # ✅ 同步修复: 对抗样本测试中应用纯标量 Lite 上下文
    ctx_lite = _build_ctx_lite(market_ctx)
    for res in io_results: res['ctx'] = ctx_lite

    # --- 压测阶段 2: 纯 CPU 计算多进程池 ---
    ctx_mp = multiprocessing.get_context("forkserver") if hasattr(multiprocessing, "get_context") else None
    
    # 分布式安全限速配额
    NUM_CPU_WORKERS = min(4, os.cpu_count() or 1)
    RATE_PER_WORKER = max(0.5, 4.0 / NUM_CPU_WORKERS)
    
    pool_kwargs = {
        "max_workers": NUM_CPU_WORKERS,
        "initializer": _worker_pool_initializer,
        "initargs": (RATE_PER_WORKER, 4.0)
    }
    if sys.platform != "win32" and ctx_mp:
        pool_kwargs["mp_context"] = ctx_mp

    with concurrent.futures.ProcessPoolExecutor(**pool_kwargs) as cpu_executor:
        # ✅ 修复: 移除脆弱的 map，改用 submit + as_completed 实现单节点崩溃熔断隔离
        future_map = {cpu_executor.submit(_stress_test_cpu_worker, payload): payload['sym'] for payload in io_results}
        for future in concurrent.futures.as_completed(future_map, timeout=300.0):
            sym = future_map[future]
            try:
                res = future.result(timeout=60.0)
                for env_name in metrics: 
                    metrics[env_name]['trades'] += res[env_name]['trades']
                    metrics[env_name]['wins'] += res[env_name]['wins']
                    metrics[env_name]['ret_sum'] += res[env_name]['ret_sum']
            except concurrent.futures.TimeoutError:
                logger.warning(f"⚠️ CPU 压测 Worker {sym} 计算超时，已跳过。")
            except Exception as e:
                logger.warning(f"⚠️ CPU 压测 Worker {sym} 遭遇崩溃 ({type(e).__name__}: {e})，已隔离。")

    def _fmt(m): return f"交易笔数: {m['trades']} | 胜率: {(m['wins']/m['trades'])*100 if m['trades']>0 else 0:.1f}% | 单笔均利: {(m['ret_sum']/m['trades'])*100 if m['trades']>0 else 0:.2f}%"
    orig, phase, noise = metrics['Original'], metrics['Phase_Chaos'], metrics['Noise_Explosion']
    report_lines = ["### 🌪️ **对抗样本压力测试报告 (Adversarial Stress Test)**\n", f"**🟢 原始环境 (Original History)**\n- {_fmt(orig)}\n", f"**🌌 平行宇宙A：相位错乱 (Phase Shifted)**\n- {_fmt(phase)}"]
    
    report_lines.append("- 📉 **归因诊断**: 胜率急剧崩塌！当前模型对 FFT 等周期相位产生了**严重虚假依赖**。" if (orig['wins']/orig['trades'] if orig['trades']>0 else 0) - (phase['wins']/phase['trades'] if phase['trades']>0 else 0) > 0.15 else "- 🛡️ **归因诊断**: 胜率坚挺。模型未陷入周期的死记硬背，对时序变异具备极强鲁棒性！")
    report_lines.append(f"\n**🌋 平行宇宙B：噪音爆炸 (Noise Amplified 1.5x)**\n- {_fmt(noise)}")
    send_alert("终极实战压测报告", "\n".join(report_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    try:
        if m == "matrix": run_tech_matrix()
        elif m == "gateway": run_gateway()
        elif m == "backtest": run_backtest_engine()
        elif m == "replay": run_historical_replay()
        elif m == "stress": run_synthetic_stress_test()
        elif m == "test": send_alert("连通性测试", "全维宏观 Meta 跃迁完成！系统已开启 6 维上帝视野。")
    finally:
        # ✅ 生命周期管理：确保程序退出前优雅关闭异步会话，防止资源泄露
        if TRANSFORMER_AVAILABLE:
            asyncio.run(AsyncSessionManager.close())
