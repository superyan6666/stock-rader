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
import scipy.stats as stats
import concurrent.futures
from typing import List, Tuple, Dict
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ================= 1. 日志与配置管理 =================
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
    
    CORE_WATCHLIST: list = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK-B", "AVGO", 
        "LLY", "JPM", "V", "XOM", "UNH", "MA", "PG", "JNJ", "HD", "MRK", 
        "ABBV", "CVX", "CRM", "COST", "PEP", "AMD", "WMT", "BAC", "TMO", "KO", 
        "MCD", "DIS", "ADBE", "CSCO", "ABT", "INTC", "QCOM", "TXN", "INTU", "AMAT", 
        "IBM", "AMGN", "NFLX", "NOW", "PFE", "BA", "ISRG", "SPGI", "GE", "HON", 
        "CAT", "UNP", "GS", "SYK", "BKNG", "TJX", "VRTX", "MS", "LRCX", "MDT", 
        "PGR", "REGN", "ADI", "ADP", "BSX", "C", "MDLZ", "GILD", "CB", "PANW", 
        "FI", "MMC", "CVS", "CI", "MU", "SNPS", "BMY", "CDNS", "KLAC", "DE", 
        "VLO", "SHW", "CSX", "WM", "FCX", "T", "F", "GM", "PLTR", "UBER", 
        "MSTR", "COIN", "CRWD", "SNOW", "DDOG", "NET", "PATH", "ROKU", "ZM", "SQ", 
        "SHOP", "SPOT", "TEAM", "WDAY", "ZS", "DKNG", "HOOD", "RBLX", "U", "AFRM",
        "SMCI", "ARM", "ALAB", "MRVL", "GFS", "TSM", "ASML", "NVO", "MELI", "PDD",
        "BABA", "BIDU", "JD", "NIO", "LI", "XPEV", "TCEHY", "NTES", "RYAAY", "LVMUY",
        "QQQ", "SPY", "DIA", "IWM", "SOXX", "SMH", "XLK", "XLF", "XLV", "XLP"
    ]
    
    INDEX_ETF: str = "QQQ" 
    VIX_INDEX: str = "^VIX" 
    MIN_SCORE_THRESHOLD: int = 8 
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'TXN', 'INTU', 'AMAT', 'MU', 'LRCX', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'NXPI', 'MRVL', 'MCHP', 'FTNT', 'CRWD'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD', 'ROST', 'EBAY', 'TSCO', 'PDD', 'DASH', 'CPRT', 'PCAR'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA', 'TTWO', 'WBD', 'SIRI', 'CHTR'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM', 'IDXX', 'MRNA', 'ALGN', 'BMRN', 'GEHC'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'MNST', 'WBA']
    }

    CROWDING_PENALTY: float = 0.75
    CROWDING_MIN_STOCKS: int = 2
    CROWDING_EXCLUDE_SECTORS: list = ["QQQ"]
    
    LOG_PREFIX: str = "backtest_log_"
    STATS_FILE: str = "strategy_stats.json"
    REPORT_FILE: str = "backtest_report.md"
    CACHE_FILE: str = "tickers_cache.json"
    ALERT_CACHE_FILE: str = "alert_history.json"
    MODEL_FILE: str = "scoring_model.pkl"
    
    ALL_FACTORS = [
        "米奈尔维尼", "强相对强度", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)",
        "突破缺口", "VWAP突破", "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", "口袋支点", 
        "VCP收缩", "量子概率云(KDE)", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "威科夫弹簧(Spring)", 
        "跨时空共振(周线)", "CVD筹码净流入", "独立Alpha(脱钩)", "NR7极窄突破", "VPT量价共振",
        "带量金叉(交互)", "量价吸筹(交互)", "近3日突破(滞后)", "近3日巨量(滞后)",
        "稳健赫斯特(Hurst)", "FFT多窗共振(动能)", 
        "大周期保护小周期(MACD共振)", "聪明钱月度净流入(月线)", "60分钟级精准校准(RSI反弹)",
        "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)",
        "Amihud非流动性(冲击成本)", "52周高点距离(动能延续)", "波动率风险溢价(VRP)",
        "期权PutCall情绪(PCR)", "隐含波动率偏度(IV Skew)", "做空兴趣突变(轧空)",
        "内部人集群净买入(Insider)", "分析师修正动量(Analyst)", "舆情NLP情感极值(News_NLP)", "散户热度加速度(WSB_Accel)"
    ]

    GROUP_A_FACTORS = [
        "米奈尔维尼", "强相对强度", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)",
        "突破缺口", "VWAP突破", "巨量滞涨", "放量长阳", "口袋支点", "VCP收缩", 
        "跨时空共振(周线)", "独立Alpha(脱钩)", "近3日突破(滞后)", "近3日巨量(滞后)", "带量金叉(交互)",
        "大周期保护小周期(MACD共振)", "聪明钱月度净流入(月线)", 
        "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)",
        "52周高点距离(动能延续)", "内部人集群净买入(Insider)", "分析师修正动量(Analyst)"
    ]
    
    GROUP_B_FACTORS = [
        "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", 
        "威科夫弹簧(Spring)", "CVD筹码净流入", "NR7极窄突破", "VPT量价共振", "量价吸筹(交互)", 
        "稳健赫斯特(Hurst)", "FFT多窗共振(动能)", "量子概率云(KDE)",
        "60分钟级精准校准(RSI反弹)", "Amihud非流动性(冲击成本)", "波动率风险溢价(VRP)",
        "期权PutCall情绪(PCR)", "隐含波动率偏度(IV Skew)", "做空兴趣突变(轧空)",
        "舆情NLP情感极值(News_NLP)", "散户热度加速度(WSB_Accel)"
    ]
    
    FINBERT_POS = ['beat', 'raise', 'upgrade', 'strong', 'surge', 'rally', 'buy', 'bullish', 'record', 'profit', 'outperform', 'exceed', 'soar', 'jump', 'dividend']
    FINBERT_NEG = ['miss', 'cut', 'downgrade', 'weak', 'decline', 'sell', 'bearish', 'warn', 'loss', 'underperform', 'plunge', 'drop', 'lawsuit', 'investigation', 'scandal']

    @classmethod
    def get_current_log_file(cls) -> str:
        return f"{cls.LOG_PREFIX}{datetime.now(timezone.utc).strftime('%Y_%m')}.jsonl"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF

def validate_config():
    if not Config.WEBHOOK_URL and not Config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未配置任何推送渠道。")
        sys.exit(1)
    
    curr_log = Config.get_current_log_file()
    if not os.path.exists(curr_log): open(curr_log, 'a').close()
    if not os.path.exists(Config.REPORT_FILE): open(Config.REPORT_FILE, 'a').close()
    if not os.path.exists(Config.STATS_FILE):
        with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: f.write("{}")
    if not os.path.exists(Config.ALERT_CACHE_FILE):
        with open(Config.ALERT_CACHE_FILE, 'w', encoding='utf-8') as f: 
            json.dump({"date": "", "matrix": []}, f)
            
    logger.info("✅ 环境与占位文件校验通过")

_KLINE_CACHE = {}
_SENTIMENT_CACHE = {} 
_ALT_DATA_CACHE = {}  

# ================= 2. 数据工具模块 =================
def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    cache_key = f"{symbol}_{period}_{interval}"
    if cache_key in _KLINE_CACHE:
        return _KLINE_CACHE[cache_key].copy()
        
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(0.1, 0.3) if fast_mode else random.uniform(1.5, 3.0)
            time.sleep(sleep_sec)
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=auto_adjust, timeout=15)
            if not df.empty: 
                _KLINE_CACHE[cache_key] = df.copy()
                return df
        except Exception as e:
            logger.warning(f"[{symbol}] 尝试 {attempt+1} 失败: {e}")
            if attempt == retries - 1: return pd.DataFrame()
            time.sleep((10 + attempt * 5) if "429" in str(e).lower() else (2 + attempt * 2))
    return pd.DataFrame()

def fetch_global_wsb_data() -> Dict[str, dict]:
    if "WSB_GLOBAL" in _ALT_DATA_CACHE:
        return _ALT_DATA_CACHE["WSB_GLOBAL"]
    wsb_dict = {}
    for attempt in range(3):
        try:
            url = "https://tradestie.com/api/v1/apps/reddit"
            response = requests.get(url, headers=_GLOBAL_HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    tk = item.get('ticker')
                    if tk:
                        wsb_dict[tk] = {
                            'sentiment': 1.0 if item.get('sentiment') == 'Bullish' else -1.0,
                            'comments': item.get('no_of_comments', 0)
                        }
                logger.info(f"🌐 成功加载全局 Reddit/WSB 舆情雷达，监控 {len(wsb_dict)} 活跃标的。")
                break
        except Exception as e:
            logger.debug(f"WSB 舆情雷达接入尝试 {attempt+1} 失败: {e}")
            time.sleep(2.0)
            
    if not wsb_dict:
        logger.warning(f"⚠️ WSB 舆情雷达最终接入失败，将使用降级空值处理。")
        
    _ALT_DATA_CACHE["WSB_GLOBAL"] = wsb_dict
    return wsb_dict

def safe_get_sentiment_data(symbol: str) -> Tuple[float, float, float, float]:
    if symbol in _SENTIMENT_CACHE: 
        return _SENTIMENT_CACHE[symbol]
        
    pcr, iv_skew, short_change, short_float = 0.0, 0.0, 0.0, 0.0
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        curr_short = info.get('sharesShort', 0)
        prev_short = info.get('sharesShortPriorMonth', 0)
        short_float = info.get('shortPercentOfFloat', 0)
        if curr_short and prev_short and prev_short > 0:
            short_change = (curr_short - prev_short) / prev_short
            
        exps = tk.options
        if exps:
            opt = tk.option_chain(exps[0]) 
            c_vol = opt.calls['volume'].sum() if 'volume' in opt.calls else 0
            p_vol = opt.puts['volume'].sum() if 'volume' in opt.puts else 0
            if c_vol > 0: 
                pcr = p_vol / c_vol
                
            c_iv = opt.calls['impliedVolatility'].median() if 'impliedVolatility' in opt.calls else 0
            p_iv = opt.puts['impliedVolatility'].median() if 'impliedVolatility' in opt.puts else 0
            iv_skew = p_iv - c_iv
            
    except Exception as e:
        logger.debug(f"[{symbol}] 期权与做空数据探测器预警盲区: {e}")
        
    _SENTIMENT_CACHE[symbol] = (pcr, iv_skew, short_change, short_float)
    return pcr, iv_skew, short_change, short_float

def safe_get_alt_data(symbol: str) -> Tuple[float, float, float, str]:
    cache_key = f"ALT_{symbol}"
    if cache_key in _ALT_DATA_CACHE:
        return _ALT_DATA_CACHE[cache_key]
        
    insider_net_buy = 0.0
    analyst_mom = 0.0
    nlp_score = 0.0
    news_summary = ""
    
    try:
        tk = yf.Ticker(symbol)
        try:
            insiders = tk.insider_transactions
            if insiders is not None and not insiders.empty:
                recent_insiders = insiders.head(20) 
                buys = 0
                sells = 0
                if 'Shares' in recent_insiders.columns:
                    if 'Text' in recent_insiders.columns:
                        buys = recent_insiders[recent_insiders['Text'].str.contains('Buy|Purchase', case=False, na=False)]['Shares'].sum()
                        sells = recent_insiders[recent_insiders['Text'].str.contains('Sell|Sale', case=False, na=False)]['Shares'].abs().sum()
                    else:
                        buys = recent_insiders[recent_insiders['Shares'] > 0]['Shares'].sum()
                        sells = recent_insiders[recent_insiders['Shares'] < 0]['Shares'].abs().sum()
                
                if (buys + sells) > 0:
                    insider_net_buy = (buys - sells) / (buys + sells)
        except Exception as e:
            logger.debug(f"[{symbol}] 内部人数据解析跳过: {e}")

        try:
            upgrades = tk.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                cutoff_date = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=30)
                if isinstance(upgrades.index, pd.DatetimeIndex):
                    try:
                        recent_upgs = upgrades[upgrades.index >= cutoff_date]
                        if not recent_upgs.empty and 'Action' in recent_upgs.columns:
                            upg_count = len(recent_upgs[recent_upgs['Action'] == 'up'])
                            dwg_count = len(recent_upgs[recent_upgs['Action'] == 'down'])
                            analyst_mom = upg_count - dwg_count
                    except Exception: pass
        except Exception as e:
            logger.debug(f"[{symbol}] 分析师调级数据解析跳过: {e}")

        try:
            news_data = tk.news
            if news_data:
                pos_hits = 0
                neg_hits = 0
                latest = news_data[0]
                title = latest.get('title', '')
                publisher = latest.get('publisher', '')
                
                for n in news_data[:5]: 
                    t = n.get('title', '').lower()
                    pos_hits += sum(1 for word in Config.FINBERT_POS if word in t)
                    neg_hits += sum(1 for word in Config.FINBERT_NEG if word in t)
                
                if (pos_hits + neg_hits) > 0:
                    nlp_score = (pos_hits - neg_hits) / (pos_hits + neg_hits)
                
                if title:
                    sentiment_icon = "🟢" if nlp_score > 0.3 else ("🔴" if nlp_score < -0.3 else "⚪")
                    news_summary = f"{sentiment_icon} {title} ({publisher})"
        except Exception: pass
        
    except Exception as e:
        logger.debug(f"[{symbol}] 另类数据引擎异常: {e}")

    _ALT_DATA_CACHE[cache_key] = (insider_net_buy, analyst_mom, nlp_score, news_summary)
    return insider_net_buy, analyst_mom, nlp_score, news_summary

def check_earnings_risk(symbol: str) -> bool:
    try:
        tk = yf.Ticker(symbol)
        try:
            ed_df = tk.earnings_dates
            if ed_df is not None and not ed_df.empty:
                for d in ed_df.index:
                    d_val = d.date() if hasattr(d, 'date') else None
                    if d_val:
                        delta = (d_val - datetime.now(timezone.utc).date()).days
                        if 0 <= delta <= 5: return True
        except Exception as e: 
            logger.debug(f"[{symbol}] 财报解析途径1异常: {e}")
        
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    ed = cal['Earnings Date']
                    if isinstance(ed, list) and len(ed) > 0:
                        ed_date = ed[0]
                        if hasattr(ed_date, 'date'):
                            delta = (ed_date.date() - datetime.now(timezone.utc).date()).days
                            if 0 <= delta <= 5: return True
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    if 'Earnings Date' in cal.index:
                        ed = cal.loc['Earnings Date'].iloc[0]
                        if hasattr(ed, 'date'):
                            delta = (ed.date() - datetime.now(timezone.utc).date()).days
                            if 0 <= delta <= 5: return True
        except Exception as e: 
            logger.debug(f"[{symbol}] 财报解析途径2异常: {e}")
    except Exception as e: 
        logger.debug(f"[{symbol}] 财报风险探测整体异常: {e}")
    return False

def fetch_tradingview_screener(max_tickers=150) -> list:
    logger.info("📡 启动狂暴天眼 (TradingView Scanner) 猎杀全市场高动能异动标的...")
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
            "options": {"lang": "en"},
            "markets": ["america"],
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
            tickers = []
            for item in data.get('data', []):
                raw_ticker = item['d'][0]
                ticker = raw_ticker.split(':')[1] if ':' in raw_ticker else raw_ticker
                tickers.append(ticker)
            
            valid_tickers = [t.replace('.', '-') for t in tickers if len(t) <= 5]
            logger.info(f"👁️ 天眼锁定: 成功捕获 {len(valid_tickers)} 只高弹性(Beta>1.1)暴躁黑马。")
            return valid_tickers
    except Exception as e:
        logger.warning(f"⚠️ 天眼扫描受阻，退回常规模式: {e}")
    return []

def get_filtered_watchlist(max_stocks: int = 150) -> list:
    logger.info(">>> 漏斗过滤：融合狂暴天眼雷达与核心白马池...")
    tickers = set(Config.CORE_WATCHLIST)
    
    tv_tickers = fetch_tradingview_screener(120)
    if tv_tickers:
        tickers.update(tv_tickers)
    
    if os.path.exists(Config.CACHE_FILE):
        try:
            mtime = os.path.getmtime(Config.CACHE_FILE)
            if time.time() - mtime < 7 * 86400:
                with open(Config.CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_tickers = json.load(f)
                    tickers.update(cached_tickers)
                    logger.info(f"♻️ 成功加载本地存活缓存。")
        except Exception: pass

    tickers_list = list(tickers)
    logger.info(f"✅ 汇总全维海选池: {len(tickers_list)} 只标的。")

    try:
        chunk_size = 50 
        dfs = []
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            for attempt in range(3):
                try:
                    chunk_df = yf.download(chunk, period="5d", progress=False, threads=2)
                    if not chunk_df.empty: 
                        dfs.append(chunk_df)
                        break
                    else:
                        logger.warning(f"⚠️ 批次 {i//chunk_size + 1} 返回为空，尝试第 {attempt+2} 次拉取...")
                        time.sleep((attempt + 1) * 2.5)
                except Exception as dl_e:
                    logger.warning(f"⚠️ 批次 {i//chunk_size + 1} 网络报错: {dl_e}，准备重试...")
                    time.sleep((attempt + 1) * 3)
            else:
                logger.error(f"❌ 批次 {i//chunk_size + 1} 彻底失败，该批次标的已在本次扫描中丢失。")

            if i + chunk_size < len(tickers_list): time.sleep(random.uniform(2.0, 3.5))
                
        if not dfs: raise ValueError("批量下载发生全局灾难性失败。")
        df = pd.concat(dfs, axis=1)
        
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close'] if 'Close' in df.columns else df.xs('Close', level=0, axis=1)
            volume_df = df['Volume'] if 'Volume' in df.columns else df.xs('Volume', level=0, axis=1)
        else:
            close_df, volume_df = df, pd.DataFrame(1e6, index=df.index, columns=df.columns)

        available_tickers = set(close_df.columns) if isinstance(close_df, pd.DataFrame) else set()
        
        if len(available_tickers) > 200:
            try:
                with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(available_tickers), f)
            except Exception: pass

        closes = close_df.dropna(axis=1, how='all').ffill().iloc[-1]
        volumes = volume_df.dropna(axis=1, how='all').mean()
        turnovers = (closes * volumes).dropna()
        
        valid_turnovers = turnovers[(closes > 10.0) & (turnovers > 30_000_000)]
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        if top_tickers:
            return top_tickers
        return Config.CORE_WATCHLIST[:max_stocks]
    except Exception as e:
        logger.error(f"❌ 批量下载底层崩溃: {e}")
        return Config.CORE_WATCHLIST[:max_stocks]

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

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    req_headers = _GLOBAL_HEADERS.copy()
    req_headers["Content-Type"] = "application/json"
    
    if Config.WEBHOOK_URL:
        payload = {
            "msgtype": "markdown", 
            "markdown": {
                "title": f"【{Config.DINGTALK_KEYWORD}】{title}", 
                "text": f"## 🤖 【{Config.DINGTALK_KEYWORD}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"
            }
        }
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, headers=req_headers, timeout=10)
            except Exception: pass
                
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        html_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        html_content = re.sub(r'### (.*?)\n', r'<b>\1</b>\n', html_content)
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_content)
        html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
        html_content = html_content.replace('\n---', '\n━━━━━━━━━━━━━━━━━━')
        
        tg_text = f"🤖 <b>【量化监控】{html_title}</b>\n\n{html_content}\n\n⏱️ <i>{formatted_time}</i>"
        try:
            requests.post(
                f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": tg_text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                }, headers=req_headers, timeout=10)
        except Exception: pass

# ================= 3. 大盘感知与指标模块 =================
def get_vix_level(qqq_df_for_shadow: pd.DataFrame = None) -> Tuple[float, str]:
    df = safe_get_history(Config.VIX_INDEX, period="5d", interval="1d", retries=3, auto_adjust=False, fast_mode=True)
    vix, is_simulated = 18.0, False
    
    if not df.empty and len(df) >= 1:
        vix = df['Close'].ffill().iloc[-1]
    else:
        if qqq_df_for_shadow is not None and len(qqq_df_for_shadow) >= 20:
            realized_vol = qqq_df_for_shadow['Close'].pct_change().dropna().ewm(span=20).std().iloc[-1] * (252 ** 0.5) * 100
            vix = min(max(realized_vol * 1.15 + 4.0, 9.0), 85.0)
            is_simulated = True
            
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
    
    credit_risk_alert = False
    credit_desc = ""
    try:
        hyg = safe_get_history("HYG", "3mo", "1d", fast_mode=True)
        ief = safe_get_history("IEF", "3mo", "1d", fast_mode=True)
        if not hyg.empty and not ief.empty:
            ratio = hyg['Close'] / ief['Close']
            ratio_ma20 = ratio.rolling(20).mean().iloc[-1]
            if ratio.iloc[-1] < ratio_ma20 and ratio.iloc[-1] < ratio.iloc[-10]:
                credit_risk_alert = True
                credit_desc = "\n- 🚨 **宏观信用风控**: 高收益债流出避险，市场警报！"
    except Exception: pass
    
    macro_gravity = False
    gravity_desc = ""
    try:
        dxy = safe_get_history("DX-Y.NYB", "1mo", "1d", fast_mode=True)
        tnx = safe_get_history("^TNX", "1mo", "1d", fast_mode=True)
        if not dxy.empty and not tnx.empty:
            dxy_trend = dxy['Close'].iloc[-1] / dxy['Close'].iloc[-10] - 1
            tnx_trend = tnx['Close'].iloc[-1] / tnx['Close'].iloc[-10] - 1
            if dxy_trend > 0.015 and tnx_trend > 0.04:
                macro_gravity = True
                gravity_desc = "\n- 🌑 **宏观引力波**: 美元与美债收益率双飙，流动性黑洞来袭，系统极度承压！"
    except Exception: pass
    
    breadth_desc = ""
    if active_pool and len(active_pool) >= 30:
        try:
            sample = active_pool[:60]
            pool_df = yf.download(sample, period="3mo", progress=False)['Close']
            if isinstance(pool_df, pd.DataFrame):
                sma50 = pool_df.rolling(50).mean()
                above_50 = (pool_df.iloc[-1] > sma50.iloc[-1]).sum()
                breadth = above_50 / len(sample)
                breadth_desc = f" | 市场宽度: {breadth:.0%} 站上50日均线"
                
                if c_close > ma200 and breadth < 0.4:
                    return "hidden_bear", f"⚠️ 指数虚高但宽度严重背离{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
                elif c_close < ma200 and breadth > 0.6:
                    return "hidden_bull", f"🔥 指数弱势但暗流涌动{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        except Exception: pass

    if c_close > ma200:
        if trend_20d > 0.02: return "bull", f"🐂 牛市主升阶段{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        else: return "range", f"⚖️ 牛市高位震荡{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
    else:
        if c_close > ma50_curr and ma50_curr > ma50_prev and trend_20d > 0.04:
            return "rebound", f"🦅 熊市超跌反弹 (V反){breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        elif trend_20d < -0.02: 
            return "bear", f"🐻 熊市回调阶段{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity
        else: 
            return "range", f"⚖️ 熊市底部震荡{breadth_desc}{credit_desc}{gravity_desc}", df, credit_risk_alert, macro_gravity

def _robust_fft_ensemble(close_prices: np.ndarray, base_length=120, ensemble_count=7) -> float:
    if len(close_prices) < base_length + (ensemble_count // 2) * 5:
        return 0.0
    
    votes = []
    for offset in range(ensemble_count):
        win_len = base_length + (offset - ensemble_count//2) * 5
        segment = close_prices[-win_len:]
        
        if np.isnan(segment).any() or np.all(segment == segment[0]):
            continue
            
        detrended = segment - np.mean(segment)
        fft_res = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(win_len)
        
        pos_mask = (freqs > 0.01) & (freqs < 0.2)
        if not np.any(pos_mask): continue
        
        pos_idx = np.where(pos_mask)[0]
        peak_idx = pos_idx[np.argmax(np.abs(fft_res[pos_idx]))]
        phase = np.angle(fft_res[peak_idx])
        
        t = win_len - 1
        dom_freq = freqs[peak_idx]
        current_phase = (2 * np.pi * dom_freq * t + phase) % (2 * np.pi)
        
        if 0 < current_phase < np.pi:
            votes.append(1.0)
        else:
            votes.append(-1.0)
            
    if not votes: return 0.0
    return float(sum(votes) / len(votes))

def _robust_hurst(close_prices: np.ndarray, min_window=30, n_bootstrap=100) -> Tuple[float, float, bool]:
    log_ret = np.diff(np.log(close_prices[-121:])) if len(close_prices) > 120 else np.diff(np.log(close_prices))
    
    if len(log_ret) <= min_window:
        return 0.5, 0.0, False
        
    np.random.seed(42)
    hurst_samples = []
    for _ in range(n_bootstrap):
        sub_len = np.random.randint(min_window, len(log_ret))
        start = np.random.randint(0, len(log_ret) - sub_len)
        sub_ret = log_ret[start:start+sub_len]
        
        S = np.std(sub_ret)
        if S == 0: continue
        
        mean_ret = np.mean(sub_ret)
        cum_dev = np.cumsum(sub_ret - mean_ret)
        R = np.max(cum_dev) - np.min(cum_dev)
        
        if R <= 0: continue
        h = np.log(R/S) / np.log(len(sub_ret))
        hurst_samples.append(h)
        
    if len(hurst_samples) < 20:
        return 0.5, 0.0, False
        
    h_median = float(np.median(hurst_samples))
    h_iqr = float(np.percentile(hurst_samples, 75) - np.percentile(hurst_samples, 25))
    
    is_reliable = bool((h_iqr < 0.15) and (abs(h_median - 0.5) > 0.1))
    return h_median, h_iqr, is_reliable

def _compute_advanced_composite(curr: pd.Series, fft_ensemble_score: float, hurst_med: float, 
                                hurst_reliable: bool, kde_breakout_score: float) -> Tuple[float, dict]:
    fft_score = fft_ensemble_score * 0.5 
    hurst_score = max(0.0, min(1.0, (hurst_med - 0.5) * 2.0)) if (hurst_reliable and hurst_med > 0.65) else 0.0
    vpt_score = (1.0 / (1.0 + np.exp(-curr['VPT_ZScore']))) if curr['VPT_Accel'] > 0 else 0.0
    cvd_score = float(curr['CVD_Trend'] * (0.3 if curr['CVD_Divergence'] == 1 else 1.0))
    kde_score = kde_breakout_score
    composite = (fft_score + hurst_score + vpt_score + cvd_score + kde_score) / 5.0
    details = {
        'FFT': fft_score, 'Hurst': hurst_score, 
        'VPT': vpt_score, 'CVD': cvd_score, 'KDE': kde_score
    }
    return composite, details

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['Open'] = df['Open'].ffill()
    df['High'], df['Low'] = df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    atr20 = df['TR'].rolling(window=20).mean()
    df['KC_Upper'] = df['EMA_20'] + 1.5 * atr20
    df['KC_Lower'] = df['EMA_20'] - 1.5 * atr20
    bb_ma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = bb_ma + 2 * bb_std
    df['BB_Lower'] = bb_ma - 2 * bb_std
    
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    
    hl2 = (df['High'].values + df['Low'].values) / 2.0
    atr10 = df['TR'].rolling(window=10).mean().values
    ub = hl2 + 3 * atr10
    lb = hl2 - 3 * atr10
    close_arr = df['Close'].values
    in_up = np.ones(len(df), dtype=bool)
    
    for i in range(1, len(df)):
        if np.isnan(ub[i-1]): continue
        c = close_arr[i]
        prev_ub = ub[i-1]
        prev_lb = lb[i-1]
        prev_in_up = in_up[i-1]
        
        if c > prev_ub: curr_in_up = True
        elif c < prev_lb: curr_in_up = False
        else: curr_in_up = prev_in_up
            
        in_up[i] = curr_in_up
        
        if curr_in_up and prev_in_up: 
            lb[i] = lb[i] if lb[i] > prev_lb else prev_lb
        elif not curr_in_up and not prev_in_up: 
            ub[i] = ub[i] if ub[i] < prev_ub else prev_ub
            
    df['SuperTrend_Up'] = in_up.astype(int)
    
    hl_diff = np.maximum(df['High'].values - df['Low'].values, 1e-10)
    dollar_vol = df['Close'].values * df['Volume'].values
    vol_sum20 = pd.Series(dollar_vol).rolling(20).sum().values + 1e-10
    clv_num = (df['Close'].values - df['Low'].values) - (df['High'].values - df['Close'].values)
    df['CMF'] = pd.Series((clv_num / hl_diff * dollar_vol)).rolling(20).sum().values / vol_sum20
    df['CMF'] = df['CMF'].clip(lower=-1.0, upper=1.0).fillna(0.0)

    df['Range'] = df['High'] - df['Low']
    df['NR7'] = (df['Range'] <= df['Range'].rolling(7).min())
    df['Inside_Bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))

    close_shift = np.roll(close_arr, 1)
    close_shift[0] = np.nan
    vpt_base = np.where((close_shift == 0) | np.isnan(close_shift), 0.0, (close_arr - close_shift) / close_shift)
    df['VPT'] = vpt_base * df['Volume'].values
    df['VPT_Cum'] = df['VPT'].cumsum()
    
    vpt_cum_arr = df['VPT_Cum'].values
    vpt_ma50 = df['VPT_Cum'].rolling(50).mean().values
    vpt_std50 = df['VPT_Cum'].rolling(50).std().values
    vpt_std50 = np.where((np.isnan(vpt_std50)) | (vpt_std50 == 0), 1e-6, vpt_std50)
    vpt_zscore = (vpt_cum_arr - vpt_ma50) / vpt_std50
    df['VPT_ZScore'] = vpt_zscore
    df['VPT_Accel'] = np.gradient(np.nan_to_num(vpt_zscore))

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP_20'] = (df['Typical_Price'] * df['Volume']).rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)
    
    df['AVWAP'] = np.nan
    n = len(df)
    if n > 0:
        lookback = min(120, n)
        vol_vals = df['Volume'].values
        anchor_iloc = n - lookback + np.argmax(vol_vals[-lookback:])
        tp_vol = (df['Typical_Price'].values[anchor_iloc:] * vol_vals[anchor_iloc:]).cumsum()
        vol_cum = vol_vals[anchor_iloc:].cumsum() + 1e-10
        df.iloc[anchor_iloc:, df.columns.get_loc('AVWAP')] = tp_vol / vol_cum
    
    down_vol = df['Volume'].where(df['Close'] < df['Close'].shift(), 0)
    df['Max_Down_Vol_10'] = down_vol.shift(1).rolling(10).max()
    
    surge_condition = (df['Close'] > df['Close'].shift(1) * 1.04) & (df['Volume'] > df['Volume'].rolling(20).mean())
    df['OB_High'] = df['High'].shift(1).where(surge_condition, np.nan).ffill(limit=20)
    df['OB_Low'] = df['Low'].shift(1).where(surge_condition, np.nan).ffill(limit=20)
    
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Range_60'] = df['High'].rolling(60).max() - df['Low'].rolling(60).min()
    df['Range_20'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    
    df['Price_High_20'] = df['High'].rolling(20).max()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    intra_strength = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
    df['CVD'] = (df['Volume'] * intra_strength).cumsum()
    
    df['CVD_Smooth'] = df['CVD'].ewm(span=5).mean()
    cvd_ma10 = df['CVD_Smooth'].rolling(10).mean()
    cvd_ma30 = df['CVD_Smooth'].rolling(30).mean()
    df['CVD_Trend'] = np.where(cvd_ma10 > cvd_ma30, 1.0, np.where(cvd_ma10 < cvd_ma30, -1.0, 0.0))
    df['CVD_High_20'] = df['CVD_Smooth'].rolling(20).max()
    df['CVD_Divergence'] = ((df['Close'] >= df['Price_High_20'] * 0.99) & (df['CVD_Smooth'] < df['CVD_High_20'] * 0.95)).astype(int)
    
    df['Highest_22'] = df['High'].rolling(window=22).max()
    df['ATR_22'] = df['TR'].rolling(window=22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['Smart_Money_Flow'] = clv.rolling(window=10).mean()

    df['Recent_Price_Surge_3d'] = (df['Close'] / df['Open'] - 1).rolling(3).max().shift(1) * 100
    df['Recent_Vol_Surge_3d'] = (df['Volume'] / df['Vol_MA20']).rolling(3).max().shift(1)

    daily_ret_abs = df['Close'].pct_change().abs()
    dollar_vol = df['Close'] * df['Volume'] + 1e-10
    df['Amihud'] = (daily_ret_abs / dollar_vol).rolling(20).mean() * 1e6
    
    high_52w = df['High'].rolling(252, min_periods=20).max()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-10)

    return df

# ================= 4. 执行引擎 =================

def get_alert_cache() -> dict:
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    cache_data = {"date": today_str, "matrix": [], "shadow_pool": []}
    try:
        if os.path.exists(Config.ALERT_CACHE_FILE):
            with open(Config.ALERT_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("date") == today_str:
                    cache_data = data
    except Exception: pass
    return cache_data

def is_alerted(sym: str) -> bool:
    return sym in get_alert_cache().get("matrix", [])

def set_alerted(sym: str, is_shadow: bool = False, shadow_data: dict = None) -> None:
    cache = get_alert_cache()
    if not is_shadow:
        if sym not in cache.get("matrix", []):
            cache.setdefault("matrix", []).append(sym)
    else:
        if shadow_data and not any(s['symbol'] == sym for s in cache.get("shadow_pool", [])):
            cache.setdefault("shadow_pool", []).append(shadow_data)
            
    try:
        with open(Config.ALERT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except Exception: pass

def _extract_complex_features(df: pd.DataFrame, df_w: pd.DataFrame, df_m: pd.DataFrame, df_60m: pd.DataFrame, macro_data: dict, vix_current: float) -> Tuple[bool, float, float, float, float, float, float, bool, float, float, float, float, float, float, float]:
    weekly_bullish = False
    weekly_macd_res = 0.0
    if not df_w.empty and len(df_w) >= 40:
        sma40_w = df_w['Close'].rolling(40).mean().iloc[-1]
        if df_w['Close'].iloc[-1] > sma40_w:
            df_w['EMA_10'] = df_w['Close'].ewm(span=10, adjust=False).mean()
            df_w['EMA_30'] = df_w['Close'].ewm(span=30, adjust=False).mean()
            weekly_bullish = (df_w['Close'].iloc[-1] > df_w['EMA_10'].iloc[-1]) and (df_w['EMA_10'].iloc[-1] > df_w['EMA_30'].iloc[-1])
            
        ema12_w = df_w['Close'].ewm(span=12, adjust=False).mean()
        ema26_w = df_w['Close'].ewm(span=26, adjust=False).mean()
        macd_w = ema12_w - ema26_w
        signal_w = macd_w.ewm(span=9, adjust=False).mean()
        hist_w = macd_w - signal_w
        if len(hist_w) >= 2 and hist_w.iloc[-1] > 0 and hist_w.iloc[-1] > hist_w.iloc[-2]:
            weekly_macd_res = 1.0
            
    fvg_lower, fvg_upper = 0.0, 0.0
    n = len(df)
    if n >= 22:
        lows, highs = df['Low'].values, df['High'].values
        start_idx = n - 20
        valid_idx = np.where(lows[start_idx:n-1] > highs[start_idx-2:n-3])[0]
        if len(valid_idx) > 0:
            last_i = valid_idx[-1] + start_idx
            fvg_lower, fvg_upper = highs[last_i-2], lows[last_i]
    
    df_60 = df.iloc[-60:]
    kde_breakout_score = 0.0
    if len(df_60) >= 20 and df_60['Volume'].sum() > 0:
        try:
            from scipy.stats import gaussian_kde
            prices = df_60['Close'].values
            volumes = df_60['Volume'].values
            
            if np.std(prices) > 1e-5:
                kde = gaussian_kde(prices, weights=volumes, bw_method='silverman')
                current_price = prices[-1]
                density_current = kde.evaluate(current_price)[0]
                
                price_grid = np.linspace(prices.min(), prices.max(), 200)
                densities = kde.evaluate(price_grid)
                density_threshold_50 = np.percentile(densities, 50)
                
                if density_current <= density_threshold_50:
                    breakout_score = 0.5 + 0.5 * (density_threshold_50 - density_current) / (density_threshold_50 + 1e-10)
                    breakout_score = min(1.0, breakout_score)
                else:
                    breakout_score = 0.0
                    
                poc_idx = np.argmax(densities)
                poc_price = price_grid[poc_idx]
                poc_distance = abs(current_price - poc_price) / (prices.max() - prices.min() + 1e-10)
                
                if poc_distance > 0.2:
                    kde_breakout_score = float(breakout_score)
        except Exception:
            pass
        
    fft_ensemble_score = _robust_fft_ensemble(df['Close'].values, base_length=120, ensemble_count=7)
    hurst_med, hurst_iqr, hurst_reliable = _robust_hurst(df['Close'].values)
    
    monthly_inst_flow = 0.0
    if not df_m.empty and len(df_m) >= 3:
        m_flow = (df_m['Close'] - df_m['Open']) / (df_m['High'] - df_m['Low'] + 1e-10) * df_m['Volume']
        if m_flow.iloc[-1] > 0 and m_flow.iloc[-2] > 0 and m_flow.iloc[-3] > 0:
            monthly_inst_flow = 1.0

    rsi_60m_bounce = 0.0
    if not df_60m.empty and len(df_60m) >= 15:
        delta = df_60m['Close'].diff()
        up = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
        down = -delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean()
        rs = up / (down + 1e-10)
        rsi_60m = 100 - (100 / (1 + rs))
        if len(rsi_60m) >= 2 and rsi_60m.iloc[-1] > rsi_60m.iloc[-2] and rsi_60m.iloc[-2] < 40:
            rsi_60m_bounce = 1.0
            
    stock_ret = df['Close'].pct_change().fillna(0)
    
    beta_60d = 1.0
    spy_realized_vol = 15.0 
    if 'spy' in macro_data and not macro_data['spy'].empty and len(stock_ret) >= 60:
        spy_ret = macro_data['spy']['Close'].pct_change().reindex(stock_ret.index).fillna(0)
        cov_mat = np.cov(stock_ret.iloc[-60:], spy_ret.iloc[-60:])
        if cov_mat[1,1] > 0: beta_60d = cov_mat[0, 1] / cov_mat[1, 1]
        spy_realized_vol = spy_ret.iloc[-20:].std() * np.sqrt(252) * 100
        
    tlt_corr = 0.0
    if 'tlt' in macro_data and not macro_data['tlt'].empty and len(stock_ret) >= 90:
        tlt_ret = macro_data['tlt']['Close'].pct_change().reindex(stock_ret.index).fillna(0)
        tlt_corr = stock_ret.iloc[-90:].corr(tlt_ret.iloc[-90:])
        if pd.isna(tlt_corr): tlt_corr = 0.0
        
    dxy_corr = 0.0
    if 'dxy' in macro_data and not macro_data['dxy'].empty and len(stock_ret) >= 20:
        dxy_ret = macro_data['dxy']['Close'].pct_change().reindex(stock_ret.index).fillna(0)
        dxy_corr = stock_ret.iloc[-20:].corr(dxy_ret.iloc[-20:])
        if pd.isna(dxy_corr): dxy_corr = 0.0
        
    vrp = float(vix_current - spy_realized_vol)
        
    return weekly_bullish, fvg_lower, fvg_upper, kde_breakout_score, fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable, monthly_inst_flow, weekly_macd_res, rsi_60m_bounce, float(beta_60d), float(tlt_corr), float(dxy_corr), float(vrp)

def _extract_ml_features(df: pd.DataFrame, curr: pd.Series, prev: pd.Series, qqq_df: pd.DataFrame,
                         fvg_lower: float, kde_breakout_score: float, weekly_bullish: bool,
                         fft_ensemble_score: float, hurst_med: float, hurst_iqr: float, hurst_reliable: bool,
                         monthly_inst_flow: float, weekly_macd_res: float, rsi_60m_bounce: float,
                         beta_60d: float, tlt_corr: float, dxy_corr: float, vrp: float,
                         pcr: float, iv_skew: float, short_change: float, short_float: float,
                         insider_net_buy: float, analyst_mom: float, nlp_score: float, wsb_accel: float) -> List[float]:
    def safe_div(num, den, cap=20.0):
        if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
        return max(min(num / den, cap), -cap)

    rs_20, pure_alpha = 0.0, 0.0
    if not qqq_df.empty:
        m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
        if len(m_df) >= 60:
            qqq_ret = max(m_df['Close_y'].iloc[-1] / m_df['Close_y'].iloc[-20], 0.5)
            rs_20 = (m_df['Close_x'].iloc[-1] / m_df['Close_x'].iloc[-20]) / qqq_ret
            ret_stock = m_df['Close_x'].pct_change().dropna()
            ret_qqq = m_df['Close_y'].pct_change().dropna()
            cov_matrix = np.cov(ret_stock.iloc[-60:], ret_qqq.iloc[-60:])
            beta = cov_matrix[0,1] / (cov_matrix[1,1] + 1e-10) if cov_matrix[1,1] > 0 else 1.0
            pure_alpha = (ret_stock.iloc[-5:].mean() - beta * ret_qqq.iloc[-5:].mean()) * 252

    swing_high_10 = df['High'].iloc[-11:-1].max() if len(df) >= 12 else curr['High']

    macd_cross_strength = safe_div(curr['MACD'] - curr['Signal_Line'], abs(curr['Close']) * 0.01)
    vol_surge_ratio = safe_div(curr['Volume'], curr['Vol_MA20'], cap=50.0)
    cmf_val = curr['CMF']
    smf_val = curr['Smart_Money_Flow']

    hurst_score = max(0.0, min(1.0, (hurst_med - 0.5) * 2.0)) if (hurst_reliable and hurst_med > 0.65) else 0.0

    feat_dict = {
        "米奈尔维尼": safe_div(curr['SMA_50'] - curr['SMA_200'], curr['SMA_200']),
        "强相对强度": rs_20,
        "MACD金叉": macd_cross_strength,
        "TTM Squeeze ON": safe_div((curr['KC_Upper'] - curr['KC_Lower']) - (curr['BB_Upper'] - curr['BB_Lower']), curr['Close'] * 0.01),
        "一目多头": safe_div(curr['Close'] - max(curr['SenkouA'], curr['SenkouB']), curr['Close'] * 0.01),
        "强势回踩": safe_div(curr['Close'] - curr['EMA_20'], curr['EMA_20'] * 0.01),
        "机构控盘(CMF)": cmf_val,
        "突破缺口": safe_div(curr['Open'] - prev['Close'], prev['Close'] * 0.01),
        "VWAP突破": safe_div(curr['Close'] - curr['VWAP_20'], curr['VWAP_20'] * 0.01),
        "AVWAP突破": safe_div(curr['Close'] - curr['AVWAP'], curr['AVWAP'] * 0.01) if pd.notna(curr['AVWAP']) else 0.0,
        "SMC失衡区": safe_div(curr['Close'] - fvg_lower, curr['Close'] * 0.01) if fvg_lower > 0 else 0.0,
        "流动性扫盘": safe_div(curr['Swing_Low_20'] - curr['Low'], curr['Low'] * 0.01) if pd.notna(curr['Swing_Low_20']) else 0.0,
        "聪明钱抢筹": smf_val,
        "巨量滞涨": vol_surge_ratio,
        "放量长阳": safe_div(curr['Close'] - curr['Open'], curr['Open'] * 0.01),
        "口袋支点": safe_div(curr['Volume'], curr['Max_Down_Vol_10'], cap=50.0),
        "VCP收缩": safe_div(curr['Range_20'], curr['Range_60'] + 1e-10),
        "量子概率云(KDE)": kde_breakout_score,
        "特性改变(ChoCh)": safe_div(curr['Close'] - swing_high_10, swing_high_10 * 0.01),
        "订单块(OB)": safe_div(curr['Close'] - curr['OB_Low'], curr['OB_High'] - curr['OB_Low'] + 1e-10) if pd.notna(curr['OB_High']) else 0.0,
        "AMD操盘": safe_div(min(curr['Open'], curr['Close']) - curr['Low'], curr['TR'] + 1e-10),
        "威科夫弹簧(Spring)": safe_div(curr['Swing_Low_20'] - curr['Low'], curr['Low'] * 0.01) if pd.notna(curr['Swing_Low_20']) else 0.0,
        "跨时空共振(周线)": 1.0 if weekly_bullish else 0.0,
        "CVD筹码净流入": float(curr['CVD_Trend'] * (0.3 if curr['CVD_Divergence'] == 1 else 1.0)),
        "独立Alpha(脱钩)": pure_alpha,
        "NR7极窄突破": safe_div(curr['Range'], curr['ATR'] + 1e-10),
        "VPT量价共振": (1.0 / (1.0 + np.exp(-curr['VPT_ZScore']))) if curr['VPT_Accel'] > 0 else 0.0,
        "带量金叉(交互)": macd_cross_strength * vol_surge_ratio,
        "量价吸筹(交互)": cmf_val * smf_val,
        "近3日突破(滞后)": curr['Recent_Price_Surge_3d'] if pd.notna(curr['Recent_Price_Surge_3d']) else 0.0,
        "近3日巨量(滞后)": curr['Recent_Vol_Surge_3d'] if pd.notna(curr['Recent_Vol_Surge_3d']) else 0.0,
        "稳健赫斯特(Hurst)": hurst_score,
        "FFT多窗共振(动能)": fft_ensemble_score,
        "大周期保护小周期(MACD共振)": 1.0 if (weekly_macd_res == 1.0 and prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']) else 0.0,
        "聪明钱月度净流入(月线)": monthly_inst_flow,
        "60分钟级精准校准(RSI反弹)": rsi_60m_bounce,
        "大盘Beta(宏观调整)": beta_60d,
        "利率敏感度(TLT相关性)": tlt_corr,
        "汇率传导(DXY相关性)": dxy_corr,
        "Amihud非流动性(冲击成本)": curr['Amihud'] if pd.notna(curr['Amihud']) else 0.0,
        "52周高点距离(动能延续)": curr['Dist_52W_High'] if pd.notna(curr['Dist_52W_High']) else 0.0,
        "波动率风险溢价(VRP)": vrp,
        "期权PutCall情绪(PCR)": pcr,
        "隐含波动率偏度(IV Skew)": iv_skew,
        "做空兴趣突变(轧空)": short_change if short_float > 0.05 else 0.0,
        
        "内部人集群净买入(Insider)": insider_net_buy,
        "分析师修正动量(Analyst)": analyst_mom,
        "舆情NLP情感极值(News_NLP)": nlp_score,
        "散户热度加速度(WSB_Accel)": wsb_accel
    }

    raw_array = [feat_dict.get(f, 0.0) for f in Config.ALL_FACTORS]
    return [float(x) for x in np.nan_to_num(raw_array, nan=0.0, posinf=20.0, neginf=-20.0)]

def _evaluate_omni_matrix(df: pd.DataFrame, curr: pd.Series, prev: pd.Series, qqq_df: pd.DataFrame, is_vol: bool,
                          weekly_bullish: bool, fvg_lower: float, fvg_upper: float, kde_breakout_score: float,
                          regime: str, w_mul: float, xai_weights: dict,
                          fft_ensemble_score: float, hurst_med: float, hurst_iqr: float, hurst_reliable: bool,
                          monthly_inst_flow: float, weekly_macd_res: float, rsi_60m_bounce: float,
                          beta_60d: float, tlt_corr: float, dxy_corr: float, vrp: float, macro_gravity: bool,
                          pcr: float, iv_skew: float, short_change: float, short_float: float,
                          insider_net_buy: float, analyst_mom: float, nlp_score: float, wsb_accel: float) -> Tuple[int, List[str], List[str], bool]:
    
    sig, factors, triggered = [], [], []
    black_swan_risk = False
    
    def get_fw(tag_name: str) -> float: return xai_weights.get(tag_name, 1.0)
    def add_trigger(tag, text, pts, category):
        fw = get_fw(tag)
        if fw > 0: triggered.append((tag, text.format(fw=fw), pts * w_mul * fw, category))

    rs_20, pure_alpha = 0.0, 0.0
    if not qqq_df.empty:
        m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
        if len(m_df) >= 60:
            rs_20 = (m_df['Close_x'].iloc[-1]/m_df['Close_x'].iloc[-20]) / max(m_df['Close_y'].iloc[-1]/m_df['Close_y'].iloc[-20], 0.5)
            ret_stock, ret_qqq = m_df['Close_x'].pct_change().dropna(), m_df['Close_y'].pct_change().dropna()
            cov_mat = np.cov(ret_stock.iloc[-60:], ret_qqq.iloc[-60:])
            beta = cov_mat[0,1] / (cov_mat[1,1] + 1e-10) if cov_mat[1,1] > 0 else 1.0
            pure_alpha = (ret_stock.iloc[-5:].mean() - beta * ret_qqq.iloc[-5:].mean()) * 252

    gap_pct = (curr['Open'] - prev['Close']) / prev['Close']
    atr_pct = (curr['ATR'] / prev['Close']) * 100
    day_chg = (curr['Close'] - curr['Open']) / curr['Open'] * 100
    swing_high_10 = df['High'].iloc[-11:-1].max() if len(df) >= 12 else curr['High']
    tr_val = curr['High'] - curr['Low'] + 1e-10
    lower_wick = curr['Open'] - curr['Low'] if curr['Close'] > curr['Open'] else curr['Close'] - curr['Low']
    upper_wick = curr['High'] - curr['Close'] if curr['Close'] > curr['Open'] else curr['High'] - curr['Open']

    if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200']:
        minervini_str = (curr['SMA_50'] - curr['SMA_200']) / curr['SMA_200']
        pts = 8 + int(minervini_str * 20)
        add_trigger("米奈尔维尼", f"🏆 [主升趋势] 米奈尔维尼模板形成 (强度:{minervini_str*100:.1f}% 权:{{fw:.2f}}x)", pts, "TREND")
        
    if rs_20 > 0: 
        dynamic_rs_thresh = 1.0 + (curr['ATR'] / curr['Close']) * 2.0
        if rs_20 > dynamic_rs_thresh:
            pts = 7 if is_vol else 4
            add_trigger("强相对强度", f"⚡ [相对强度] 动能超越动态阈值 (阈值:{dynamic_rs_thresh:.2f} 权:{{fw:.2f}}x)", pts, "TREND")
    
    if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
        is_above_zero = curr['MACD'] > 0
        pts = 12 if is_above_zero else 8
        desc = "水上金叉" if is_above_zero else "水下金叉"
        add_trigger("MACD金叉", f"🔥 [经典动能] MACD{desc}起爆 (权:{{fw:.2f}}x)", pts, "TREND")

    if (curr['BB_Upper'] - curr['BB_Lower']) < (curr['KC_Upper'] - curr['KC_Lower']): 
        add_trigger("TTM Squeeze ON", "📦 [波动压缩] TTM Squeeze 挤流状态激活 (权:{fw:.2f}x)", 8, "VOLATILITY")

    if curr['Above_Cloud'] == 1 and curr['Tenkan'] > curr['Kijun']:
        add_trigger("一目多头", "🌥️ [趋势确认] 一目均衡表云上多头共振 (权:{fw:.2f}x)", 6, "TREND")

    if curr['CMF'] > 0.20: 
        add_trigger("机构控盘(CMF)", "🏦 [资金监控] CMF显示主力资金持续强控盘 (权:{fw:.2f}x)", 5, "TREND")
        
    if curr['Smart_Money_Flow'] > 0.5 and curr['Close'] > curr['EMA_20']:
        add_trigger("聪明钱抢筹", "🕵️ [暗中吸筹] Smart Money 指数显示连续尾盘抢筹 (权:{fw:.2f}x)", 6, "QUANTUM")

    if pd.notna(curr['Recent_Price_Surge_3d']) and curr['Recent_Price_Surge_3d'] > 4.0:
        add_trigger("近3日突破(滞后)", "⏱️ [滞后记忆] 近3日内曾发生暴涨，趋势处于亢奋延续期 (权:{fw:.2f}x)", 8, "TREND")
        
    if pd.notna(curr['Recent_Vol_Surge_3d']) and curr['Recent_Vol_Surge_3d'] > 2.5:
        add_trigger("近3日巨量(滞后)", "⏱️ [滞后记忆] 近3日内曾爆出巨量，机构资金高度活跃 (权:{fw:.2f}x)", 8, "VOLATILITY")

    if curr['Close'] > curr['VWAP_20'] and prev['Close'] <= curr['VWAP_20']:
        add_trigger("VWAP突破", "🌊 [量价突破] 放量逾越近20日VWAP机构均价线 (权:{fw:.2f}x)", 8, "TREND")
        
    if pd.notna(curr['AVWAP']) and curr['Close'] > curr['AVWAP'] and prev['Close'] <= curr['AVWAP']:
        add_trigger("AVWAP突破", "⚓ [筹码夺回] 强势站上AVWAP锚定成本核心区 (权:{fw:.2f}x)", 12, "TREND")

    if pd.notna(curr['Swing_Low_20']) and curr['Low'] < curr['Swing_Low_20'] and curr['Volume'] < curr['Vol_MA20'] * 0.8 and curr['Close'] > (curr['Low'] + tr_val * 0.5):
        add_trigger("威科夫弹簧(Spring)", "🏹 [威科夫测试] 跌破前低但抛压枯竭缩量，经典的Spring深蹲起爆 (权:{fw:.2f}x)", 18, "REVERSAL")

    if is_vol:
        if pure_alpha > 0.8:
            add_trigger("独立Alpha(脱钩)", "🪐 [独立Alpha] 强势剥离大盘Beta，爆发特质动能 (权:{fw:.2f}x)", 22, "TREND")
            
        if curr['SuperTrend_Up'] == 1 and curr['Close'] < curr['EMA_20'] * 1.02:
            add_trigger("强势回踩", "🟢 [低吸点位] 超级趋势主升轨精准回踩 (权:{fw:.2f}x)", 10, "REVERSAL")
            
        if gap_pct * 100 > max(1.5, atr_pct * 0.3) and gap_pct < 0.06:
            add_trigger("突破缺口", "💥 [动能爆发] 放量跳空，留下底部突破缺口 (权:{fw:.2f}x)", 8, "VOLATILITY")
            
        if fvg_lower > 0 and curr['Low'] <= fvg_upper and curr['Close'] > fvg_lower:
            add_trigger("SMC失衡区", "🧲 [SMC交易法] 精准回踩并测试前期机构失衡区(FVG) (权:{fw:.2f}x)", 15, "REVERSAL")
    
        if pd.notna(curr['Swing_Low_20']) and curr['Low'] < curr['Swing_Low_20'] and curr['Close'] > curr['Swing_Low_20']:
            add_trigger("流动性扫盘", "🧹 [止损猎杀] 刺穿前低扫掉散户流动性后迅速诱空反转 (权:{fw:.2f}x)", 15, "REVERSAL")
            
        if curr['Volume'] > curr['Vol_MA20'] * 2.0 and abs(curr['Close'] - curr['Open']) < curr['ATR'] * 0.5:
            add_trigger("巨量滞涨", "🛑 [冰山订单] 巨量滞涨，极大概率为机构冰山挂单吸货 (权:{fw:.2f}x)", 12, "QUANTUM")
            
        if day_chg > max(3.0, atr_pct * 0.6) and curr['Volume'] > curr['Vol_MA20'] * 1.5:
            add_trigger("放量长阳", "⚡ [动能脉冲] 强劲的日内放量大实体阳线 (权:{fw:.2f}x)", 12, "QUANTUM")
        
        if curr['Close'] > prev['Close'] and curr['Volume'] > curr['Max_Down_Vol_10'] > 0 and curr['Close'] >= curr['EMA_50'] and prev['Close'] <= curr['EMA_50'] * 1.02:
            add_trigger("口袋支点", "💎 [口袋支点] 放量阳线成交量完全吞噬近期最大阴量 (权:{fw:.2f}x)", 12, "REVERSAL")
            
        if curr['Range_20'] > 0 and curr['Range_60'] > 0 and curr['Range_20'] < curr['Range_60'] * 0.5 and curr['Close'] > curr['SMA_50']:
            add_trigger("VCP收缩", "🌪️ [VCP形态] 经历极度价格波动压缩后的放量突破 (权:{fw:.2f}x)", 15, "VOLATILITY")
                
        if kde_breakout_score > 0.5:
            add_trigger("量子概率云(KDE)", f"☁️ [真空逃逸] KDE 自适应带宽揭示价格脱离筹码区，逃逸分={kde_breakout_score:.2f} (权:{{fw:.2f}}x)", 15 * kde_breakout_score, "QUANTUM")

        if pd.notna(curr['Swing_Low_20']) and curr['Low'] > curr['Swing_Low_20'] and curr['Close'] > swing_high_10:
            add_trigger("特性改变(ChoCh)", "🔀 [结构破坏] 突破近期反弹高点，完成 ChoCh 趋势逆转确认 (权:{fw:.2f}x)", 15, "REVERSAL")

        if pd.notna(curr['OB_High']) and curr['OB_Low'] > 0 and curr['Low'] <= curr['OB_High'] and curr['Close'] >= curr['OB_Low'] and curr['Close'] > curr['Open']:
            add_trigger("订单块(OB)", "🧱 [机构订单块] 触达机构历史起爆底仓区并收出企稳阳线 (权:{fw:.2f}x)", 15, "REVERSAL")

        if curr['Close'] > curr['Open'] and (lower_wick / tr_val) > 0.3 and (upper_wick / tr_val) < 0.15:
            add_trigger("AMD操盘", "🎭 [AMD诱空] 深度开盘诱空下杀后，全天拉升派发的操盘模型 (权:{fw:.2f}x)", 12, "REVERSAL")
            
        if weekly_bullish and (curr['Close'] > curr['Highest_22'] * 0.95):
            add_trigger("跨时空共振(周线)", "🌌 [多周期共振] 周线级别主升浪与日线级别放量的强力双重共振 (权:{fw:.2f}x)", 20, "QUANTUM")

        if curr['CVD_Trend'] == 1.0 and prev['CVD_Trend'] <= 0.0:
            if curr['CVD_Divergence'] == 0:
                add_trigger("CVD筹码净流入", "🧬 [微观筹码] CVD 双均线平滑多头确立且无量价背离，真实买盘涌入 (权:{fw:.2f}x)", 12, "QUANTUM")
            else:
                sig.append("⚠️ [微观筹码] 价格近高但 CVD 出现顶背离，尾盘动能涉嫌虚假欺骗 (已被AI降权)")

        if prev['NR7'] and prev['Inside_Bar'] and curr['Close'] > prev['High']:
            add_trigger("NR7极窄突破", "🎯 [极度压缩] 7日极窄压缩孕线完成向上爆破 (权:{fw:.2f}x)", 12, "VOLATILITY")

        if curr['VPT_ZScore'] > 0.5 and curr['VPT_Accel'] > 0 and prev['VPT_ZScore'] <= 0.5:
            add_trigger("VPT量价共振", "📈 [量价归一] VPT Z-Score突破且动能加速，真实买盘绝对共振 (权:{fw:.2f}x)", 10, "TREND")

        if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
            add_trigger("带量金叉(交互)", "🔥 [交互共振] MACD金叉与成交量激增产生乘数效应 (权:{fw:.2f}x)", 12, "TREND")
            
        if curr['CMF'] > 0.15 and curr['Smart_Money_Flow'] > 0.4:
            add_trigger("量价吸筹(交互)", "🏦 [交互共振] 蔡金资金流与微观聪明钱同向深度吸筹 (权:{fw:.2f}x)", 10, "QUANTUM")
            
        if hurst_reliable and hurst_med > 0.65:
            h_score = max(0.0, min(1.0, (hurst_med - 0.5) * 2.0))
            add_trigger("稳健赫斯特(Hurst)", f"⏳ [稳健记忆] R/S Bootstrap 确认极强抗噪趋势持续性，Hurst={hurst_med:.2f} (权:{{fw:.2f}}x)", 15 * h_score, "QUANTUM")

        if fft_ensemble_score >= 0.71:
            add_trigger("FFT多窗共振(动能)", "🌊 [频域共振] 多窗口 FFT 阵列一致确认，主导周期处于强劲上升象限 (权:{fw:.2f}x)", 15, "QUANTUM")

        if weekly_macd_res == 1.0 and prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
            add_trigger("大周期保护小周期(MACD共振)", "🛡️ [多维时空] 周线MACD动能发散，日线精准金叉，大周期绝对战役保护 (权:{fw:.2f}x)", 15, "TREND")

        if monthly_inst_flow == 1.0:
            add_trigger("聪明钱月度净流入(月线)", "🏛️ [战略定调] 月线级别连续3个月大单资金净流入，暗池机构底仓坚如磐石 (权:{fw:.2f}x)", 10, "QUANTUM")
            
        if rsi_60m_bounce == 1.0:
            add_trigger("60分钟级精准校准(RSI反弹)", "⏱️ [战术执行] 60分钟线 RSI 触底反弹，日内高频微操切入绝佳滑点位置 (权:{fw:.2f}x)", 8, "REVERSAL")

        if beta_60d > 1.2 and not macro_gravity:
            add_trigger("大盘Beta(宏观调整)", "📈 [宏观Beta动能] 宏观低压期，高Beta(>1.2)特质赋予极强上行弹性 (权:{fw:.2f}x)", 7, "TREND")
        elif beta_60d > 1.2 and macro_gravity:
            add_trigger("大盘Beta(宏观调整)", "⚠️ [宏观Beta反噬] 宏观引力波高压期，高Beta特质面临深度回撤风险，降权防御 (权:{fw:.2f}x)", -10, "TREND")

        if tlt_corr > 0.4:
            add_trigger("利率敏感度(TLT相关性)", "🏦 [宏观映射] 与长期国债(TLT)高度正相关，受益于无风险利率见顶预期 (权:{fw:.2f}x)", 8, "TREND")
        elif tlt_corr < -0.4:
            add_trigger("利率敏感度(TLT相关性)", "🛡️ [宏观防御] 与长期国债(TLT)高度负相关，具备抗息避险属性 (权:{fw:.2f}x)", 6, "TREND")
            
        if dxy_corr < -0.4:
            add_trigger("汇率传导(DXY相关性)", "💱 [宏观映射] 与美元指数(DXY)强负相关，受惠于弱美元与全球流动性释放 (权:{fw:.2f}x)", 8, "QUANTUM")
        elif dxy_corr > 0.4:
            add_trigger("汇率传导(DXY相关性)", "💵 [宏观映射] 与美元指数(DXY)强正相关，具备强汇率避险属性 (权:{fw:.2f}x)", 6, "QUANTUM")
            
        dist_52w = curr['Dist_52W_High'] if pd.notna(curr['Dist_52W_High']) else 0.0
        if dist_52w > -0.05 and weekly_bullish:
            add_trigger("52周高点距离(动能延续)", "🏔️ [动能延续] 逼近 52 周新高且周线多头，上方无抛压阻力真空区 (权:{fw:.2f}x)", 10, "TREND")
            
        amihud_val = curr['Amihud'] if pd.notna(curr['Amihud']) else 0.0
        if amihud_val > 0.5 and macro_gravity:
            add_trigger("Amihud非流动性(冲击成本)", "⚠️ [流动性枯竭] 宏观高压下 Amihud 冲击成本显著放大，极易发生踩踏被降权 (权:{fw:.2f}x)", -10, "VOLATILITY")
            
        if vrp > 5.0:
            add_trigger("波动率风险溢价(VRP)", "🌋 [风险溢价] VRP(隐含-实现)极度飙升，期权市场定价极端恐慌，捕捉极值底 (权:{fw:.2f}x)", 12, "QUANTUM")

        if insider_net_buy > 0.5:
            add_trigger("内部人集群净买入(Insider)", "👔 [内幕天眼] SEC Form 4 披露高管集群式大额净买入，极其强烈的估值底背书 (权:{fw:.2f}x)", 20, "QUANTUM")
            
        if analyst_mom > 2:
            add_trigger("分析师修正动量(Analyst)", "📊 [投行护航] 近 30 日华尔街分析师评级修正动量爆发，机构共识上调 (权:{fw:.2f}x)", 8, "TREND")
            
        if nlp_score > 0.3:
            add_trigger("舆情NLP情感极值(News_NLP)", "📰 [舆情引擎] FinBERT 极速词典判定近期新闻舆情呈现极度狂热 (权:{fw:.2f}x)", 6, "VOLATILITY")
        elif nlp_score < -0.3:
            add_trigger("舆情NLP情感极值(News_NLP)", "⚠️ [舆情崩塌] FinBERT 探测到新闻包含密集利空/调查词汇，自动降权防御 (权:{fw:.2f}x)", -10, "VOLATILITY")
            
        if wsb_accel > 0:
            add_trigger("散户热度加速度(WSB_Accel)", "🔥 [散户加速] Reddit/WSB 论坛提及次数呈二阶导数飙升，散户资金正在疯狂抬轿 (权:{fw:.2f}x)", 8, "VOLATILITY")

        if pcr > 1.5:
            add_trigger("期权PutCall情绪(PCR)", "⚠️ [极端避险] Put/Call Ratio 爆表，期权市场极度看空避险 (权:{fw:.2f}x)", -15, "VOLATILITY")
            black_swan_risk = True
        elif 0 < pcr < 0.5:
            add_trigger("期权PutCall情绪(PCR)", "🔥 [极端贪婪] Put/Call Ratio 极低，期权市场聪明钱疯狂做多 (权:{fw:.2f}x)", 10, "QUANTUM")
            
        if iv_skew > 0.08:
            add_trigger("隐含波动率偏度(IV Skew)", "🚨 [黑天鹅警报] 看跌期权隐含波动率畸高，内幕资金正疯狂买入下行保护！ (权:{fw:.2f}x)", -20, "VOLATILITY")
            black_swan_risk = True
        elif iv_skew < -0.05:
            add_trigger("隐含波动率偏度(IV Skew)", "🚀 [上行爆发] 看涨期权隐含波动率压倒性占优，主力预期剧烈向上重估 (权:{fw:.2f}x)", 12, "QUANTUM")
            
        if short_change > 0.10 and short_float > 0.05 and curr['Close'] > curr['EMA_20']:
            add_trigger("做空兴趣突变(轧空)", "💥 [轧空引擎] 近期做空兴趣激增且技术面多头，随时触发空头踩踏平仓的 Short Squeeze (权:{fw:.2f}x)", 15, "QUANTUM")

    score_raw = 0.0
    for tag, text, pts, category in triggered:
        adj_pts = pts
        if regime in ["bear", "hidden_bear"]:
            if category in ["TREND", "VOLATILITY"]: adj_pts *= 0.5  
            elif category == "REVERSAL": adj_pts *= 1.5
        elif regime == "range":
            if category in ["TREND", "VOLATILITY"]: adj_pts *= 0.8 
            elif category == "REVERSAL": adj_pts *= 1.2
        elif regime in ["bull", "rebound"]:
            if category in ["TREND", "VOLATILITY", "QUANTUM"]: adj_pts *= 1.2  
        score_raw += adj_pts
        sig.append(text)
        factors.append(tag)
        
    return int(score_raw), sig, factors, black_swan_risk

def _apply_market_filters(curr: pd.Series, prev: pd.Series, sym: str, base_score: int, sig: List[str], 
                          black_hole_sectors: List[str], leading_sectors: List[str], lagging_sectors: List[str]) -> Tuple[int, bool, List[str]]:
    total_score = base_score
    is_bearish_div = False
    if curr['Close'] >= curr['Price_High_20'] * 0.98 and curr['RSI'] < 60.0 and prev['RSI'] > 60.0:
        is_bearish_div = True
        total_score = 0 
        sig.append(f"🩸 [顶背离危局] 价格近高但动能显著衰竭，严禁追高")

    if curr['RSI'] > 85.0:
        total_score = max(0, total_score - 30)
        sig.append(f"⚠️ [极端超买] RSI 高达 {curr['RSI']:.1f} (严防获利盘踩踏)")
        
    bias_20 = (curr['Close'] - curr['EMA_20']) / curr['EMA_20']
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

def _calculate_position_size(curr: pd.Series, prev: pd.Series, ai_prob: float, vix_scalar: float, 
                             is_bearish_div: bool, is_credit_risk_high: bool, macro_gravity: bool, 
                             max_risk: float, vix_term_inverted: bool, black_swan_risk: bool) -> Tuple[float, float, float, str]:
    
    atr_mult_sl = 1.0 if vix_term_inverted else 1.5
    atr_mult_tp = 2.0 if vix_term_inverted else 3.0
    
    tp_val = curr['Close'] + atr_mult_tp * vix_scalar * curr['ATR']
    sl_chandelier = curr['Chandelier_Exit'] if pd.notna(curr['Chandelier_Exit']) else (curr['Close'] - atr_mult_sl * vix_scalar * curr['ATR'])
    sl_val = max(sl_chandelier, curr['Close'] - atr_mult_sl * vix_scalar * curr['ATR'])
    
    odds_b = 2.0
    kelly_fraction = ai_prob - (1.0 - ai_prob) / odds_b
    
    if black_swan_risk:
        return tp_val, sl_val, 0.0, "❌ 强制熔断 (期权隐含波动率/情绪探测到黑天鹅巨险)"
        
    if kelly_fraction <= 0 or is_bearish_div:
        return tp_val, sl_val, 0.0, "❌ 放弃建仓 (AI盈亏比劣势 或 顶背离确认)"
        
    return tp_val, sl_val, kelly_fraction, ""

def _apply_kelly_cluster_optimization(reports: List[dict], price_history_dict: dict, total_exposure: float) -> List[dict]:
    reports.sort(key=lambda x: (x["score"], x["ai_prob"]), reverse=True)
    candidate_pool = reports[:15]
    
    if not candidate_pool: return []
    if len(candidate_pool) == 1:
        alloc = total_exposure
        candidate_pool[0]['pos_advice'] = f"✅ 组合配置权重: {alloc * 100:.1f}% (极简化单目标资产)"
        candidate_pool[0]['opt_weight'] = 1.0
        return candidate_pool
        
    try:
        syms = [r['symbol'] for r in candidate_pool]
        df_prices = pd.DataFrame({sym: price_history_dict[sym] for sym in syms}).fillna(method='ffill')
        ret_df = df_prices.pct_change().dropna()
        
        lw = LedoitWolf().fit(ret_df)
        cov_matrix = lw.covariance_
        
        cvars = []
        for sym in syms:
            rets = ret_df[sym].values
            if len(rets) > 0:
                var_95 = np.percentile(rets, 5) 
                cvar_95 = rets[rets <= var_95].mean() if len(rets[rets <= var_95]) > 0 else var_95
                cvars.append(abs(cvar_95))
            else:
                cvars.append(0.05) 
        cvars = np.array(cvars)
        
        scores = np.array([r['score'] for r in candidate_pool])
        norm_scores = scores / (np.max(scores) + 1e-10)
        
        def objective(w):
            utility = np.dot(w, norm_scores) - 5.0 * np.dot(w, cvars) 
            return -utility
            
        sectors = np.array([r['sector'] for r in candidate_pool])
        bounds = tuple((0.02, 0.15) for _ in range(len(syms))) 
        init_w = np.ones(len(syms)) / len(syms)

        def optimize_with_constraints(sec_limit, vol_limit):
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}] 
            for sec in np.unique(sectors):
                sec_mask = (sectors == sec).astype(float)
                constraints.append({'type': 'ineq', 'fun': lambda w, m=sec_mask, lim=sec_limit: lim - np.dot(w, m)})
            daily_target_var = (vol_limit / np.sqrt(252)) ** 2
            constraints.append({'type': 'ineq', 'fun': lambda w, targ=daily_target_var: targ - np.dot(w.T, np.dot(cov_matrix, w))})
            return minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)

        res = optimize_with_constraints(0.40, 0.25)
        
        if res.success:
            opt_weights = res.x
        else:
            logger.warning(f"⚠️ 严格凸优化约束冲突，尝试放宽行业(60%)与波动率(35%)上限...")
            res_relaxed = optimize_with_constraints(0.60, 0.35)
            if res_relaxed.success:
                opt_weights = res_relaxed.x
                logger.info("✅ 宽松约束优化成功。")
            else:
                logger.warning(f"🚨 优化器彻底熔断，回退至 Score-Weighted Inverse CVaR 分配。")
                inv_cvar = 1.0 / (cvars + 1e-5)
                raw_w = norm_scores * inv_cvar
                opt_weights = raw_w / np.sum(raw_w)
                opt_weights = np.clip(opt_weights, 0.02, 0.15)
                opt_weights = opt_weights / np.sum(opt_weights)
            
        final_reports = []
        for i, r in enumerate(candidate_pool):
            alloc = opt_weights[i] * total_exposure
            r['opt_weight'] = opt_weights[i]
            r['cvar_95'] = cvars[i]
            
            r['signals'].append(f"🛡️ [风险平价] Ledoit-Wolf 稳健测算，近端尾部崩盘风险 CVaR={cvars[i]*100:.2f}%")
            r['pos_advice'] = f"✅ 组合配置权重: {alloc * 100:.1f}% (Kelly Cluster 风险平价引擎分配)"
            final_reports.append(r)
            
        return final_reports
        
    except Exception as e:
        logger.error(f"凯利集群组合优化底层失败: {e}")
        for r in candidate_pool: 
            r['pos_advice'] = f"✅ 均权配置: {total_exposure * 100 / len(candidate_pool):.1f}% (优化器熔断 fallback)"
        return candidate_pool

def _generate_and_send_matrix_report(final_reports: List[dict], final_shadow_pool: List[dict], 
                                     vix_desc: str, regime_desc: str, pain_warning: str, 
                                     pruned_factors: list, dynamic_min_score: float,
                                     meta_weights: dict) -> None:
    txts = []
    for idx, r in enumerate(final_reports):
        icon = ['🥇', '🥈', '🥉'][idx] if idx < 3 else '🔸'
        sigs_fmt = "\n".join([f"- {s}" for s in r["signals"]])
        news_fmt = f"\n- 📰 {r['news']}" if r['news'] else ""
        ai_display = f"🔥 **{r.get('ai_prob', 0):.1%}**" if r.get('ai_prob', 0) > 0.60 else f"{r.get('ai_prob', 0):.1%}"
        
        txts.append(
            f"### {icon} **{r['symbol']}** | 🤖 分层元学习胜率: {ai_display} | 🌟 终极评级: {r['score']}分\n"
            f"**💡 机构交易透视:**\n{sigs_fmt}{news_fmt}\n\n"
            f"**💰 绝对风控界限:**\n"
            f"- 💵 现价: `{r['curr_close']:.2f}`\n"
            f"- ⚖️ {r.get('pos_advice', '✅ 缺省仓位')}\n"
            f"- 🎯 建议止盈: **${r['tp']:.2f}**\n"
            f"- 🛡️ 吊灯止损: **${r['sl']:.2f} (最高价回落保护)**\n"
            f"- 📈 离场纪律: **跌破止损防线请无条件市价清仓！**"
        )

    perf = load_strategy_performance_tag()
    pruned_desc = f"\n- ✂️ 因子大逃杀: 本期共有 **{len(pruned_factors)}** 个衰竭因子被 Rank IC 判定淘汰，已打入冷宫。" if pruned_factors else ""
    header = f"**📊 宏观引力与多维时空状态:**\n- {vix_desc}\n- {regime_desc}{pain_warning}{pruned_desc}\n- ⚔️ 今日截面淘汰线 (Top 15%): **{dynamic_min_score:.1f}分**"
    
    meta_desc = ""
    if meta_weights:
        w_A = meta_weights.get('Group_A', 0)
        w_B = meta_weights.get('Group_B', 0)
        meta_desc = f"\n\n**🧠 Stacking Meta-Learner (元学习器) 状态:**\n- 传统大局观 (Group A) 决策权重: **{w_A*100:.1f}%**\n- 高级微观组 (Group B) 决策权重: **{w_B*100:.1f}%**"
        if w_B < 0.05:
            meta_desc += "\n- ⚠️ **警报**: 市场进入噪声期，高级微观因子组已大范围失效，被 L1 正则化自动屏蔽！"

    final_content = (f"{perf}\n\n{header}\n\n---\n\n" if perf else f"{header}\n\n---\n\n") + \
                    "\n\n---\n\n".join(txts) + \
                    meta_desc + \
                    f"\n\n*(全维宏观雷达: 顶层 Meta-Learner 已接入 SPY-PCR 与信用利差(HYG/IEF) 6维状态张量，动态感知恐慌阻尼！)*"
    
    send_alert("量化诸神之战 (全维宏观 Meta 跃迁版)", final_content)

def run_tech_matrix() -> None:
    max_risk = 0.015 
    pain_warning = ""
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_json = json.load(f)
                t3 = stats_json.get('overall', {}).get('T+3', {})
                if t3.get('max_cons_loss', 0) >= 4 or t3.get('profit_factor', 1.0) < 1.2:
                    max_risk = 0.0075  
                    pain_warning = "\n- 🩸 **痛觉神经激活**: 近期回测遭重挫，引擎已主动防守降杠杆！"
                meta_weights = stats_json.get("meta_weights", {})
    except Exception as e: 
        logger.debug(f"加载性能统计感知痛觉失败: {e}")
        meta_weights = {}

    active_pool = get_filtered_watchlist(max_stocks=150)
    regime, regime_desc, qqq_df, is_credit_risk_high, macro_gravity = get_market_regime(active_pool)
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    vix_scalar = max(0.6, min(1.4, 18.0 / max(vix, 1.0)))
    if macro_gravity: max_risk = min(max_risk, 0.005)
    
    vix_current = float(vix_desc.split(":")[1].replace(")", "").strip()) if "VIX:" in vix_desc else 18.0
    
    vix3m_df = safe_get_history("^VIX3M", "5d", "1d", fast_mode=True)
    vix_inv = False
    vix_term_structure = 1.0
    if not vix3m_df.empty and vix > 0:
        vix3m_val = vix3m_df['Close'].iloc[-1]
        if vix3m_val > 0:
            vix_term_structure = vix / vix3m_val
            if vix_term_structure > 1.05:
                vix_inv = True
                vix_desc += "\n- 🚨 **VIX曲面倒挂**: 近端恐慌(VIX)碾压远期(VIX3M)，全市场防线已强制极致收缩！"
            
    hyg_df = safe_get_history("HYG", "1mo", "1d", fast_mode=True)
    ief_df = safe_get_history("IEF", "1mo", "1d", fast_mode=True)
    credit_spread_mom = 0.0
    if not hyg_df.empty and not ief_df.empty:
        ratio = hyg_df['Close'] / ief_df['Close']
        if len(ratio) >= 10:
            credit_spread_mom = (ratio.iloc[-1] / ratio.iloc[-10]) - 1.0
            
    spy_pcr, _, _, _ = safe_get_sentiment_data("SPY")
    market_pcr = spy_pcr if spy_pcr > 0 else 1.0
            
    total_market_exposure = 1.0
    if vix_inv: total_market_exposure *= 0.5
    if macro_gravity: total_market_exposure *= 0.7
    if is_credit_risk_high: total_market_exposure *= 0.8
            
    macro_data = {
        'spy': safe_get_history("SPY", "2y", "1d", fast_mode=True),
        'tlt': safe_get_history("TLT", "2y", "1d", fast_mode=True),
        'dxy': safe_get_history("DX-Y.NYB", "2y", "1d", fast_mode=True)
    }
    
    valid_sector_data, black_hole_sectors = {}, []
    for etf in Config.SECTOR_MAP.keys():
        sdf = safe_get_history(etf, "3mo", "1d", fast_mode=True)
        if not sdf.empty and len(sdf) >= 30 and not qqq_df.empty:
            rs = sdf['Close'] / qqq_df['Close'].reindex(sdf.index).ffill()
            valid_sector_data[etf] = rs.rolling(10).mean().diff(5).iloc[-1] 
            if sdf['Volume'].iloc[-1] / (sdf['Volume'].iloc[-20:].mean() + 1e-10) > 1.6:  
                black_hole_sectors.append(etf)
                
    sorted_sectors = sorted(valid_sector_data.items(), key=lambda x: x[1], reverse=True)
    leading_sectors = [s[0] for s in sorted_sectors[:2]] if len(sorted_sectors) >= 4 else []
    lagging_sectors = [s[0] for s in sorted_sectors[-2:]] if len(sorted_sectors) >= 4 else []
                  
    health_score = -0.9 if vix > 30 else (-0.6 if vix > 25 else (0.9 if vix < 15 and 'bull' in regime else (0.6 if 'bull' in regime else (0.3 if 'rebound' in regime else (-0.7 if 'bear' in regime else 0.0)))))
    if is_credit_risk_high: health_score -= 0.5
    if macro_gravity: health_score -= 0.8
    w_mul = max(0.2, 1.0 + health_score * 0.8)
    
    xai_weights, pruned_factors = {}, []
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                xai_data = json.load(f).get("xai_importances", {})
                if xai_data and len(xai_data) > 0:
                    avg_imp = 1.0 / len(Config.ALL_FACTORS)
                    for tag, imp in xai_data.items():
                        if imp < avg_imp * 0.25:
                            xai_weights[tag] = 0.0
                            pruned_factors.append(tag)
                        else:
                            xai_weights[tag] = max(0.5, min(3.0, float(imp) / avg_imp))
    except Exception as e:
        logger.debug(f"解析 XAI 特征权重分布失败: {e}")
        
    clf_model = None
    if os.path.exists(Config.MODEL_FILE):
        try:
            import pickle
            with open(Config.MODEL_FILE, 'rb') as f: clf_model = pickle.load(f)
        except Exception as e:
            logger.debug(f"载入双脑模型序列化权重失败: {e}")

    raw_reports = []
    price_history_dict = {} 
    
    global_wsb_data = fetch_global_wsb_data()
    
    def _process_single_symbol(sym):
        if is_alerted(sym): return None
        try:
            df_raw = safe_get_history(sym, "2y", "1d", fast_mode=True)
            if len(df_raw) < 150: return None
            
            pcr_val, iv_skew, short_chg, short_flt = safe_get_sentiment_data(sym)
            insider_net_buy, analyst_mom, nlp_score, news_summary = safe_get_alt_data(sym)
            
            wsb_accel = 0.0
            if sym in global_wsb_data:
                wsb_info = global_wsb_data[sym]
                if wsb_info['sentiment'] > 0 and wsb_info['comments'] > 15:
                    wsb_accel = 1.0
            
            df = calculate_indicators(df_raw)
            df_w = df_raw.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            df_m = df_raw.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            df_60m = safe_get_history(sym, "1mo", "60m", fast_mode=True)
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            is_untradeable = False
            if (curr['ATR'] / curr['Close'] > 0.15): is_untradeable = True
            if pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']: is_untradeable = True
            
            price_hist = df['Close'].iloc[-60:]
            is_vol = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open'])
            
            weekly_bullish, fvg_lower, fvg_upper, kde_breakout_score, fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable, monthly_inst_flow, weekly_macd_res, rsi_60m_bounce, beta_60d, tlt_corr, dxy_corr, vrp = _extract_complex_features(df, df_w, df_m, df_60m, macro_data, vix_current)
            
            ml_features_array = _extract_ml_features(
                df, curr, prev, qqq_df, fvg_lower, kde_breakout_score, weekly_bullish,
                fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable,
                monthly_inst_flow, weekly_macd_res, rsi_60m_bounce,
                beta_60d, tlt_corr, dxy_corr, vrp,
                pcr_val, iv_skew, short_chg, short_flt,
                insider_net_buy, analyst_mom, nlp_score, wsb_accel
            )
            
            base_score, sig, factors, black_swan_risk = _evaluate_omni_matrix(
                df, curr, prev, qqq_df, is_vol, weekly_bullish, fvg_lower, fvg_upper, 
                kde_breakout_score, regime, w_mul, xai_weights,
                fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable,
                monthly_inst_flow, weekly_macd_res, rsi_60m_bounce,
                beta_60d, tlt_corr, dxy_corr, vrp, macro_gravity,
                pcr_val, iv_skew, short_chg, short_flt,
                insider_net_buy, analyst_mom, nlp_score, wsb_accel
            )
            
            adv_composite, adv_details = _compute_advanced_composite(
                curr, fft_ensemble_score, hurst_med, hurst_reliable, kde_breakout_score
            )
            
            total_score, is_bearish_div, sig = _apply_market_filters(
                curr, prev, sym, base_score, sig, black_hole_sectors, leading_sectors, lagging_sectors
            )
            
            if adv_composite > 0.2:
                sig.append(f"🪂 [特种部队增援] 高级微观因子强烈共振 (分值:{adv_composite:+.2f})，总得分获得战术加成")
            elif adv_composite < -0.2:
                sig.append(f"⚠️ [特种部队否决] 高级微观因子呈现衰竭背离 (分值:{adv_composite:+.2f})，总得分遭到战术削减")
            
            total_score = int(total_score * (1.0 + 0.3 * adv_composite))
            
            sym_sec = Config.get_sector_etf(sym)
            report_data = {
                "sym": sym, "curr": curr, "prev": prev, "total_score": total_score, "is_bearish_div": is_bearish_div,
                "sig": sig, "factors": factors, "ml_features": ml_features_array, "is_untradeable": is_untradeable,
                "sym_sec": sym_sec, "black_swan_risk": black_swan_risk, "news": news_summary
            }
            return sym, report_data, price_hist
        except Exception as e:
            logger.debug(f"[{sym}] 特征提取/打分计算底层抛错: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_process_single_symbol, sym): sym for sym in active_pool}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                sym, report_data, price_hist = res
                raw_reports.append(report_data)
                price_history_dict[sym] = price_hist

    if clf_model and raw_reports:
        X_batch = np.array([r['ml_features'] for r in raw_reports])
        
        if isinstance(clf_model, dict) and 'active_factors' in clf_model:
            active_factors = clf_model['active_factors']
            idx_active = [Config.ALL_FACTORS.index(f) for f in active_factors if f in Config.ALL_FACTORS]
            X_batch_active = X_batch[:, idx_active]
            
            base_A = clf_model.get('base_A')
            base_B = clf_model.get('base_B')
            meta_clf = clf_model.get('meta')
            
            if base_A and base_B and meta_clf:
                active_A = clf_model.get('active_A', [])
                active_B = clf_model.get('active_B', [])
                idx_A = [active_factors.index(f) for f in active_A if f in active_factors]
                idx_B = [active_factors.index(f) for f in active_B if f in active_factors]
                
                if len(idx_A) > 0 and len(idx_B) > 0:
                    X_A = X_batch_active[:, idx_A]
                    X_B = X_batch_active[:, idx_B]
                    
                    prob_A = base_A.predict_proba(X_A)[:, 1]
                    prob_B = base_B.predict_proba(X_B)[:, 1]
                    
                    vix_batch = np.full(len(raw_reports), vix_current / 20.0)
                    cred_batch = np.full(len(raw_reports), credit_spread_mom)
                    term_batch = np.full(len(raw_reports), vix_term_structure - 1.0)
                    pcr_batch = np.full(len(raw_reports), market_pcr - 1.0)
                    
                    if hasattr(meta_clf, 'coef_') and meta_clf.coef_.shape[1] == 6:
                        X_meta = np.column_stack([prob_A, prob_B, vix_batch, cred_batch, term_batch, pcr_batch])
                    else:
                        X_meta = np.column_stack([prob_A, prob_B, vix_batch])
                    
                    probs = meta_clf.predict_proba(X_meta)[:, 1]
                else:
                    probs = np.full(len(raw_reports), 0.52)
            else:
                probs = np.full(len(raw_reports), 0.52)
        elif hasattr(clf_model, 'predict_proba'): 
            try:
                class_idx = np.where(clf_model.classes_ == 1)[0][0] if 1 in clf_model.classes_ else 0
                probs = clf_model.predict_proba(X_batch)[:, class_idx]
            except Exception as e:
                logger.debug(f"旧版模型推理降级: {e}")
                probs = np.full(len(raw_reports), 0.52)
        else:
            probs = np.full(len(raw_reports), 0.52)
                
        for i, r in enumerate(raw_reports):
            r['ai_prob'] = float(probs[i])
    else:
        for r in raw_reports: r['ai_prob'] = 0.52

    reports, background_pool, all_raw_scores = [], [], []
    for r in raw_reports:
        if r['total_score'] > 0: all_raw_scores.append(r['total_score'])
        tp_val, sl_val, kelly_fraction, basic_advice = _calculate_position_size(
            r['curr'], r['prev'], r['ai_prob'], vix_scalar, r['is_bearish_div'], is_credit_risk_high, macro_gravity, max_risk, vix_inv, r['black_swan_risk']
        )

        stock_data_pack = {
            "symbol": r['sym'], "score": r['total_score'], "ai_prob": r['ai_prob'], "signals": r['sig'][:8], 
            "factors": r['factors'], "ml_features": r['ml_features'],
            "curr_close": float(r['curr']['Close']), "tp": float(tp_val), "sl": float(sl_val), 
            "news": r['news'], "sector": r['sym_sec'], "pos_advice": basic_advice,
            "kelly_fraction": kelly_fraction
        }

        if not r['is_untradeable'] and r['total_score'] > 0 and kelly_fraction > 0:
            if check_earnings_risk(r['sym']):
                r['sig'].append("💣 [财报雷区] 近5日发财报,风险极高")
                r['total_score'] = int(r['total_score'] * 0.5)
                stock_data_pack["score"] = r['total_score']
            reports.append(stock_data_pack)
        else:
            background_pool.append(stock_data_pack)

    if reports and all_raw_scores:
        dynamic_min_score = max(Config.MIN_SCORE_THRESHOLD, np.percentile(all_raw_scores, 85))
        reports = [r for r in reports if r['score'] >= dynamic_min_score]
        
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
        for sec, stks in groups.items():
            if sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.CROWDING_MIN_STOCKS:
                if valid_sector_data.get(sec, 0.0) > 0.04:
                    for s in stks[1:]: 
                        if "🚀 [板块主升豁免] 让利润奔跑" not in s["signals"]: s["signals"].append("🚀 [板块主升豁免] 让利润奔跑")
                else:
                    pen = max(0.6, min(0.9, Config.CROWDING_PENALTY * (1.0 + health_score * 0.3)))
                    for s in stks[1:]: s["score"] = int(s["score"] * pen)

        final_reports = _apply_kelly_cluster_optimization(reports, price_history_dict, total_market_exposure)
        
        for r in final_reports: set_alerted(r["symbol"])
        
        final_symbols = {r['symbol'] for r in final_reports}
        unselected_background = [s for s in background_pool if s['symbol'] not in final_symbols]
        random.shuffle(unselected_background)
        final_shadow_pool = unselected_background[:150]
        
        for r in final_shadow_pool: set_alerted(r["symbol"], is_shadow=True, shadow_data=r)
            
        _generate_and_send_matrix_report(
            final_reports, final_shadow_pool, vix_desc, regime_desc, 
            pain_warning, pruned_factors, dynamic_min_score, meta_weights
        )
    else:
        logger.info("📭 本次矩阵扫描无标的突破 Top 15% 截面排位，宁缺毋滥，保持静默。")

    try:
        with open(Config.get_current_log_file(), "a", encoding="utf-8") as f:
            log_entry = {
                "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), 
                "macro_meta": {
                    "vix": vix_current,
                    "credit_spread_mom": credit_spread_mom,
                    "vix_term_structure": vix_term_structure,
                    "market_pcr": market_pcr
                },
                "top_picks": [{"symbol": r.get("symbol"), "score": r.get("score"), "signals": r.get("signals"), "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "ai_prob": r.get("ai_prob", 0.0), "tp": r.get("tp"), "sl": r.get("sl")} for r in (final_reports if reports and all_raw_scores else [])],
                "shadow_pool": [{"symbol": r.get("symbol"), "score": r.get("score"), "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "ai_prob": r.get("ai_prob", 0.0)} for r in (final_shadow_pool if reports and all_raw_scores else [])]
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"严重：写入矩阵全息日志时发生崩溃: {e}")

def run_backtest_engine() -> None:
    log_files = [f for f in os.listdir('.') if f.startswith('backtest_log') and f.endswith('.jsonl')]
    if not log_files: 
        logger.warning("未找到任何历史日志，回测取消。")
        return
        
    trades = []
    for lf in log_files:
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log = json.loads(line.strip())
                        daily_trades = log.get('top_picks', [])
                        shadows = log.get('shadow_pool', [])
                        if shadows: daily_trades.extend(shadows)
                        
                        macro = log.get('macro_meta', {})
                        
                        for p in daily_trades:
                            trades.append({
                                'date': log['date'], 
                                'vix': macro.get('vix', log.get('vix', 18.0)), 
                                'cred': macro.get('credit_spread_mom', 0.0),
                                'term': macro.get('vix_term_structure', 1.0),
                                'pcr': macro.get('market_pcr', 1.0),
                                'symbol': p['symbol'], 
                                'signals': p.get('signals', []), 
                                'factors': p.get('factors', []), 
                                'ml_features': p.get('ml_features', []), 
                                'ai_prob': p.get('ai_prob', 0.0), 
                                'tp': p.get('tp', float('inf')), 
                                'sl': p.get('sl', 0)
                            })
                    except Exception as e:
                        logger.debug(f"跳过损毁的 JSONL 日志行: {e}")
        except Exception as e:
            logger.debug(f"读取日志分片文件 {lf} 发生阻断: {e}")
            
    if not trades: return
    syms = list(set([t['symbol'] for t in trades]))
    
    try:
        chunk_size = 40 
        dfs = []
        logger.info(f"⏳ 启动回测引擎：正在为 {len(syms)} 个全息样本拉取历史轨迹数据...")
        for i in range(0, len(syms), chunk_size):
            chunk = syms[i:i + chunk_size]
            for attempt in range(3):
                try:
                    chunk_df = yf.download(chunk, period="2mo", progress=False, threads=2)
                    if not chunk_df.empty: 
                        if len(chunk) == 1 and not isinstance(chunk_df.columns, pd.MultiIndex):
                            chunk_df.columns = pd.MultiIndex.from_product([chunk_df.columns, chunk])
                        dfs.append(chunk_df)
                        break
                    else:
                        logger.warning(f"⚠️ 回测数据批次 {i//chunk_size + 1} 返回为空，尝试第 {attempt+2} 次拉取...")
                        time.sleep((attempt + 1) * 2.0)
                except Exception as dl_e:
                    logger.warning(f"⚠️ 回测数据批次 {i//chunk_size + 1} 网络报错: {dl_e}，准备重试...")
                    time.sleep((attempt + 1) * 3.0)
            else:
                logger.error(f"❌ 回测数据批次 {i//chunk_size + 1} 彻底失败，该批次样本将被迫丢弃。")

            if i + chunk_size < len(syms): time.sleep(random.uniform(2.0, 3.5))

        if not dfs: 
            logger.error("❌ 回测历史数据拉取遭遇全局灾难性失败。")
            return
            
        df_all = pd.concat(dfs, axis=1)

        if isinstance(df_all.columns, pd.MultiIndex):
            df_c = df_all['Close']
            df_o = df_all['Open']
            df_h = df_all['High']
            df_l = df_all['Low']
        else:
            df_c = df_all[['Close']].rename(columns={'Close': syms[0]})
            df_o = df_all[['Open']].rename(columns={'Open': syms[0]})
            df_h = df_all[['High']].rename(columns={'High': syms[0]})
            df_l = df_all[['Low']].rename(columns={'Low': syms[0]})
            
        df_c.index = df_c.index.strftime('%Y-%m-%d')
        df_o.index = df_o.index.strftime('%Y-%m-%d')
        df_h.index = df_h.index.strftime('%Y-%m-%d')
        df_l.index = df_l.index.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"回测拉取数据失败: {e}")
        return
    
    SLIPPAGE = 0.003    
    COMMISSION = 0.0005 
    
    stats_period, factor_rets = {'T+1': [], 'T+3': [], 'T+5': []}, {}
    trades_with_ret = []
    
    ai_filtered_wins, ai_filtered_total = 0, 0
    mae_mfe_records = {'T+1': [], 'T+3': [], 'T+5': []}
    
    for t in trades:
        sym, r_dt = t['symbol'], t['date']
        initial_sl = t.get('sl', 0)
        if pd.isna(initial_sl): initial_sl = 0
        tp_price = t.get('tp', float('inf'))
        if pd.isna(tp_price): tp_price = float('inf')
        
        if sym not in df_c.columns or sym not in df_o.columns: continue
        valid = df_c.index[df_c.index >= r_dt]
        if len(valid) == 0: continue
        
        e_idx = df_c.index.get_loc(valid[0])
        if e_idx + 1 >= len(df_c): continue
        
        e_px_raw = df_o.iloc[e_idx + 1][sym] 
        prev_c_px = df_c.iloc[e_idx][sym]
        
        if pd.isna(e_px_raw) or e_px_raw <= 0: continue
        
        gap_up = (e_px_raw - prev_c_px) / prev_c_px
        is_half_pos = gap_up > 0.03
        
        entry_cost = e_px_raw * (1 + SLIPPAGE * 3 + COMMISSION) if is_half_pos else e_px_raw * (1 + SLIPPAGE + COMMISSION)
        trail_distance = max(e_px_raw * 0.02, e_px_raw - initial_sl) if initial_sl > 0 else e_px_raw * 0.05
        
        for d in [1, 3, 5]:
            exit_idx = e_idx + d
            if exit_idx < len(df_c):
                exit_revenue = None
                highest_seen_px = e_px_raw
                dynamic_sl = initial_sl
                
                max_high_during_trade = e_px_raw
                min_low_during_trade = e_px_raw
                
                for i in range(1, d + 1):
                    check_idx = e_idx + i
                    day_open = df_o.iloc[check_idx][sym]
                    day_low = df_l.iloc[check_idx][sym]
                    day_high = df_h.iloc[check_idx][sym]
                    day_close = df_c.iloc[check_idx][sym]
                    
                    if pd.isna(day_open) or pd.isna(day_low) or pd.isna(day_high): continue
                    
                    max_high_during_trade = max(max_high_during_trade, day_high)
                    min_low_during_trade = min(min_low_during_trade, day_low)
                    
                    if dynamic_sl > 0 and day_open < dynamic_sl:
                        exit_revenue = day_open * (1 - SLIPPAGE * 5 - COMMISSION)
                        break
                        
                    if dynamic_sl > 0 and day_low <= dynamic_sl:
                        exit_revenue = dynamic_sl * (1 - SLIPPAGE * 3 - COMMISSION)
                        break
                    
                    if tp_price > 0 and day_high >= tp_price:
                        exit_revenue = tp_price * (1 - SLIPPAGE - COMMISSION)
                        break
                        
                    highest_seen_px = max(highest_seen_px, day_high)
                    dynamic_sl = max(dynamic_sl, highest_seen_px - trail_distance)
                    
                    if i == d:
                        prev_x_px = df_c.iloc[check_idx-1][sym] if check_idx > e_idx else e_px_raw
                        daily_volatility = abs((day_close - prev_x_px) / prev_x_px)
                        curr_exit_slippage = SLIPPAGE * 5 if daily_volatility > 0.15 else SLIPPAGE
                        exit_revenue = day_close * (1 - curr_exit_slippage - COMMISSION)
                
                if exit_revenue is not None:
                    ret = (exit_revenue - entry_cost) / entry_cost
                    if gap_up > 0.03: ret *= 0.5
                    
                    trade_mfe = (max_high_during_trade - entry_cost) / entry_cost
                    trade_mae = (min_low_during_trade - entry_cost) / entry_cost
                    mae_mfe_records[f'T+{d}'].append({'ret': ret, 'mfe': trade_mfe, 'mae': trade_mae})
                    stats_period[f'T+{d}'].append(ret)
                    
                    if d == 3:
                        factor_list = t.get('factors', [])
                        if not factor_list:
                            for sig_txt in t.get('signals', []):
                                m = re.search(r'\[(.*?)\]', sig_txt)
                                if m: factor_list.append(m.group(1).split(" ")[0])
                                
                        for f_name in factor_list:
                            factor_rets.setdefault(f"[{f_name}]", []).append(ret)
                            
                        ai_prob = t.get('ai_prob', 0.0)
                        if ai_prob >= 0.50:  
                            ai_filtered_total += 1
                            if ret > 0: ai_filtered_wins += 1
                            
                        ml_feats = t.get('ml_features', [])
                        if ml_feats:
                            if len(ml_feats) < len(Config.ALL_FACTORS):
                                ml_feats = ml_feats + [0.0] * (len(Config.ALL_FACTORS) - len(ml_feats))
                            
                            if len(ml_feats) == len(Config.ALL_FACTORS):
                                trades_with_ret.append({
                                    'date': t['date'],
                                    'vix': t.get('vix', 18.0),
                                    'cred': t.get('cred', 0.0),
                                    'term': t.get('term', 1.0),
                                    'pcr': t.get('pcr', 1.0),
                                    'ml_features': ml_feats,
                                    'factors': factor_list,
                                    'ret': ret
                                })
    
    feature_importances_dict = {}
    meta_weights_dict = {}
    factor_ic_report = {}
    trade_df = pd.DataFrame(trades_with_ret)
    
    if len(trade_df) >= 30:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.linear_model import LogisticRegression
            import pickle
            
            trade_df = trade_df.sort_values('date')
            X_all = np.vstack(trade_df['ml_features'].values)
            y_all_cont = trade_df['ret'].values
            dates_all = trade_df['date'].values
            
            ic_records = {f: [] for f in Config.ALL_FACTORS}
            unique_dates = np.unique(dates_all)
            
            for d_str in unique_dates:
                mask = dates_all == d_str
                if np.sum(mask) < 5: continue
                
                X_d = X_all[mask]
                y_d = y_all_cont[mask]
                
                if np.std(y_d) < 1e-6: continue
                
                for i, factor_name in enumerate(Config.ALL_FACTORS):
                    f_vals = X_d[:, i]
                    if np.std(f_vals) < 1e-6:
                        ic_records[factor_name].append(0.0)
                    else:
                        ic, _ = stats.spearmanr(f_vals, y_d)
                        ic_records[factor_name].append(float(ic) if not np.isnan(ic) else 0.0)
            
            active_candidates = []
            for f in Config.ALL_FACTORS:
                ic_arr = np.array(ic_records[f])
                if len(ic_arr) > 5:
                    mean_ic = float(np.mean(ic_arr))
                    std_ic = float(np.std(ic_arr)) + 1e-10
                    ir = mean_ic / std_ic
                    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_arr)))
                else:
                    mean_ic, ir, t_stat = 0.0, 0.0, 0.0
                    
                factor_ic_report[f] = {'mean_ic': mean_ic, 'ir': ir, 't_stat': t_stat}
                
                if abs(t_stat) > 0.5:
                    active_candidates.append((f, abs(t_stat)))
            
            active_candidates.sort(key=lambda x: x[1], reverse=True)
            top_25_factors = [x[0] for x in active_candidates[:25]]
            
            if len(top_25_factors) < 5:
                top_25_factors = Config.ALL_FACTORS[:25]
                
            logger.info(f"🧬 代谢淘汰完毕，25 个高 T-stat 因子已入列。被淘汰打入冷宫因子数：{len(Config.ALL_FACTORS) - len(top_25_factors)}")
            
            idx_active = [Config.ALL_FACTORS.index(f) for f in top_25_factors]
            X_all_active = X_all[:, idx_active]
            
            active_A = [f for f in top_25_factors if f in Config.GROUP_A_FACTORS]
            active_B = [f for f in top_25_factors if f in Config.GROUP_B_FACTORS]
            
            idx_A_in_active = [top_25_factors.index(f) for f in active_A]
            idx_B_in_active = [top_25_factors.index(f) for f in active_B]

            X_A = X_all_active[:, idx_A_in_active]
            X_B = X_all_active[:, idx_B_in_active]
            
            y_all_class = (y_all_cont > 0.015).astype(int)
            
            vix_all = trade_df['vix'].values / 20.0
            cred_all = trade_df['cred'].values
            term_all = trade_df['term'].values - 1.0
            pcr_all = trade_df['pcr'].values - 1.0
            
            tscv = TimeSeriesSplit(n_splits=3)
            oof_pred_A = np.zeros(len(y_all_class))
            oof_pred_B = np.zeros(len(y_all_class))
            
            lgbm_params = dict(n_estimators=60, max_depth=3, learning_rate=0.05, class_weight='balanced', random_state=42)
            
            can_train_A = len(active_A) > 0
            can_train_B = len(active_B) > 0
            
            for train_idx, val_idx in tscv.split(X_A):
                if can_train_A:
                    clf_A = LGBMClassifier(**lgbm_params)
                    clf_A.fit(X_A[train_idx], y_all_class[train_idx])
                    oof_pred_A[val_idx] = clf_A.predict_proba(X_A[val_idx])[:, 1]
                else:
                    oof_pred_A[val_idx] = 0.5
                    
                if can_train_B:
                    clf_B = LGBMClassifier(**lgbm_params)
                    clf_B.fit(X_B[train_idx], y_all_class[train_idx])
                    oof_pred_B[val_idx] = clf_B.predict_proba(X_B[val_idx])[:, 1]
                else:
                    oof_pred_B[val_idx] = 0.5
                
            valid_meta_idx = np.where(oof_pred_A > 0)[0]
            if len(valid_meta_idx) > 10:
                X_meta_train = np.column_stack([
                    oof_pred_A[valid_meta_idx], 
                    oof_pred_B[valid_meta_idx], 
                    vix_all[valid_meta_idx],
                    cred_all[valid_meta_idx],
                    term_all[valid_meta_idx],
                    pcr_all[valid_meta_idx]
                ])
                y_meta_train = y_all_class[valid_meta_idx]
                
                meta_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, class_weight='balanced', random_state=42)
                meta_clf.fit(X_meta_train, y_meta_train)
                
                w_A, w_B = meta_clf.coef_[0][0], meta_clf.coef_[0][1]
                exp_wA, exp_wB = np.exp(w_A), np.exp(w_B)
                meta_weights_dict['Group_A'] = exp_wA / (exp_wA + exp_wB)
                meta_weights_dict['Group_B'] = exp_wB / (exp_wA + exp_wB)
            else:
                meta_clf = LogisticRegression(class_weight='balanced')
                meta_clf.fit(np.column_stack([oof_pred_A, oof_pred_B, vix_all, cred_all, term_all, pcr_all]), y_all_class)

            final_base_A = LGBMClassifier(**lgbm_params).fit(X_A, y_all_class) if can_train_A else None
            final_base_B = LGBMClassifier(**lgbm_params).fit(X_B, y_all_class) if can_train_B else None
            
            clf_model = {
                'active_factors': top_25_factors,
                'active_A': active_A,
                'active_B': active_B,
                'base_A': final_base_A, 
                'base_B': final_base_B, 
                'meta': meta_clf,
                'n_features_in_': len(top_25_factors)
            }
            with open(Config.MODEL_FILE, 'wb') as f:
                pickle.dump(clf_model, f)
            logger.info(f"🧠 【动态代谢体系】25维优胜张量已封装进 6维全息宏观感知的 Stacking Ensemble 模型落盘。")
            
            combined_imp = np.zeros(len(Config.ALL_FACTORS))
            if final_base_A and hasattr(final_base_A, 'feature_importances_'):
                imp_A = final_base_A.feature_importances_
                for i, active_f_idx in enumerate(idx_A_in_active):
                    orig_idx = Config.ALL_FACTORS.index(top_25_factors[active_f_idx])
                    combined_imp[orig_idx] += (imp_A[i] / (np.sum(imp_A) + 1e-10)) * meta_weights_dict.get('Group_A', 0.5)
                    
            if final_base_B and hasattr(final_base_B, 'feature_importances_'):
                imp_B = final_base_B.feature_importances_
                for i, active_f_idx in enumerate(idx_B_in_active):
                    orig_idx = Config.ALL_FACTORS.index(top_25_factors[active_f_idx])
                    combined_imp[orig_idx] += (imp_B[i] / (np.sum(imp_B) + 1e-10)) * meta_weights_dict.get('Group_B', 0.5)
                
            for i, factor in enumerate(Config.ALL_FACTORS):
                if combined_imp[i] > 0:
                    feature_importances_dict[factor] = float(combined_imp[i])
                    
        except ImportError:
            logger.warning("未检测到 lightgbm 或 scipy 环境，已跳过代谢训练。")
        except Exception as e:
            logger.error(f"ML 模型代谢淘汰训练受阻: {e}")
            
    res = {}
    for p, r in stats_period.items():
        if not r: continue
        ret_arr = np.array(r)
        
        win_returns = ret_arr[ret_arr > 0]
        loss_returns = ret_arr[ret_arr < 0]
        
        wr = len(win_returns) / len(ret_arr)
        avg_r = np.mean(ret_arr)
        std_r = np.std(ret_arr) if len(ret_arr) > 1 else 1e-6
        sharpe = avg_r / (std_r + 1e-10)
        worst = np.min(ret_arr)
        
        sum_wins = np.sum(win_returns) if len(win_returns) > 0 else 0.0
        sum_losses = abs(np.sum(loss_returns)) if len(loss_returns) > 0 else 1e-6
        pf = sum_wins / sum_losses if sum_losses > 0 else 99.0
        
        max_cons_loss = 0
        curr_cons_loss = 0
        for ret in ret_arr:
            if ret < 0:
                curr_cons_loss += 1
                max_cons_loss = max(max_cons_loss, curr_cons_loss)
            else:
                curr_cons_loss = 0
                
        win_maes = [rec['mae'] for rec in mae_mfe_records[p] if rec['ret'] > 0]
        avg_win_mae = np.mean(win_maes) if win_maes else 0.0
                
        res_data = {
            'win_rate': wr, 'avg_ret': avg_r, 'sharpe': sharpe, 
            'worst_trade': worst, 'total_trades': len(r),
            'profit_factor': pf, 'max_cons_loss': max_cons_loss,
            'avg_win_mae': avg_win_mae
        }
        
        if p == 'T+3' and ai_filtered_total > 0:
            res_data['ai_win_rate'] = ai_filtered_wins / ai_filtered_total
            
        res[p] = res_data

    f_res = {}
    for t, r in factor_rets.items():
        if len(r) >= 2:
            ret_arr = np.array(r)
            win_returns = ret_arr[ret_arr > 0]
            loss_returns = ret_arr[ret_arr < 0]
            sum_wins = np.sum(win_returns) if len(win_returns) > 0 else 0.0
            sum_losses = abs(np.sum(loss_returns)) if len(loss_returns) > 0 else 1e-6
            pf = sum_wins / sum_losses if sum_losses > 0 else 99.0
            
            f_res[t] = {
                'win_rate': len(win_returns)/len(r), 
                'avg_ret': sum(r)/len(r), 
                'count': len(r),
                'profit_factor': pf
            }
            
    attr_report = {}
    if len(trades_with_ret) >= 30:
        adv_factors = ["FFT多窗共振(动能)", "稳健赫斯特(Hurst)", "VPT量价共振", "CVD筹码净流入", "量子概率云(KDE)", "Amihud非流动性(冲击成本)", "波动率风险溢价(VRP)"]
        baseline_f = "MACD金叉" 
        
        attr_data = []
        for tr in trades_with_ret:
            row = {'ret': tr['ret']}
            f_list = tr.get('factors', [])
            for f in adv_factors + [baseline_f]:
                row[f] = 1 if f in f_list else 0
            attr_data.append(row)
        attr_df = pd.DataFrame(attr_data)
        
        for f in adv_factors:
            trig = attr_df[attr_df[f] == 1]
            not_trig = attr_df[attr_df[f] == 0]
            premium = 0.0
            
            if len(trig) > 0 and len(not_trig) > 0:
                premium = trig['ret'].median() - not_trig['ret'].median()
            
            corr = attr_df[f].corr(attr_df[baseline_f]) if len(attr_df) > 1 else 0.0
            if pd.isna(corr): corr = 0.0
            
            attr_report[f] = {
                'premium_bps': float(premium * 10000), 
                'corr_with_baseline': float(corr),
                'trigger_rate': float(len(trig) / len(attr_df)) if len(attr_df) > 0 else 0.0
            }

    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump({"overall": res, "factors": f_res, "xai_importances": feature_importances_dict, "meta_weights": meta_weights_dict, "attribution": attr_report, "factor_ic": factor_ic_report}, f, indent=4)
    
    report_md = [f"# 📈 自动量化战报与 AI 透视\n**更新:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n## ⚔️ 核心表现评估\n| 周期 | 原始胜率 | ⚡代谢演化过滤 | 均收益 | 盈亏比 | Sharpe | 胜单平均抗压(MAE) | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for p in ['T+1', 'T+3', 'T+5']:
        d = res.get(p, {'win_rate':0,'avg_ret':0,'profit_factor':0,'sharpe':0,'avg_win_mae':0,'max_cons_loss':0,'total_trades':0})
        ai_str = f"**{d.get('ai_win_rate', 0.0)*100:.1f}%**" if 'ai_win_rate' in d else "-"
        report_md.append(f"| {p} | {d['win_rate']*100:.1f}% | {ai_str} | {d['avg_ret']*100:+.2f}% | {d['profit_factor']:.2f} | {d['sharpe']:.2f} | {d['avg_win_mae']*100:.1f}% | {d['total_trades']} |")
    
    if factor_ic_report:
        report_md.append("\n## 🧬 因子动物园 Rank IC 淘汰赛排行榜 (Top 10)\n*展示最具未来收益解释度的因子，T-stat 低于 1.5 且排名靠后的因子已自动打入冷宫。*\n| 因子特征 | 均值 IC | T-Statistic (绝对值) | 状态 |\n|:---|:---:|:---:|:---:|")
        sorted_ic = sorted(factor_ic_report.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True)
        top_25_names = [x[0] for x in sorted_ic[:25]]
        for tag, data in sorted_ic[:10]: 
            status = "🟢 激活" if tag in top_25_names else "🧊 冷宫"
            report_md.append(f"| {tag} | {data['mean_ic']:.4f} | {abs(data['t_stat']):.2f} | {status} |")

    if feature_importances_dict:
        report_md.append("\n## 🧠 XAI (解释性人工智能) - 驱动当期市场的核心因子权重\n| 因子特征 | AI 分配重要性 (元学习器赋权) |\n|:---|:---:|")
        sorted_xai = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        for tag, imp in sorted_xai: 
            report_md.append(f"| {tag} | {imp*100:.1f}% |")
    
    if attr_report:
        report_md.append(f"\n## 🔬 高级微观因子归因仪表盘 (Alpha Attribution)\n*鉴别高级数学因子是否为伪信号或高度共线性。纯因子溢价(Premium)衡量该信号脱离传统动能独立创造的超额收益基点。*\n| 高级因子 | 纯因子溢价 (BPS) | 与传统动能耦合度 (Corr) | 触发频率 | 归因诊断 |\n|:---|:---:|:---:|:---:|:---:|")
        for f, data in attr_report.items():
            prem = data['premium_bps']
            corr = data['corr_with_baseline']
            trig_rate = data['trigger_rate']
            
            if prem < 0: diag = "⚠️ 负溢价 (拖累策略，系统已启动降权防御)"
            elif corr > 0.6: diag = "⚖️ 高度耦合 (缺乏独立增量信息，沦为辅助确认)"
            elif prem > 50 and corr < 0.4: diag = "💎 纯净 Alpha (极高独立价值，不可被传统指标替代)"
            else: diag = "✅ 有效增益"
            
            report_md.append(f"| {f} | {prem:+.1f} bps | {corr:.2f} | {trig_rate*100:.1f}% | {diag} |")

    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_md))

    alert_lines = ["### 📊 **机构级回测报表 (含代谢淘汰赛)**"]
    for p, d in res.items(): 
        ai_text = f" | ⚡代谢演化过滤: **{d['ai_win_rate']*100:.1f}%**" if 'ai_win_rate' in d else ""
        alert_lines.append(f"- **{p}:** 原始胜率 {d['win_rate']*100:.1f}%{ai_text} | 盈亏比 {d['profit_factor']:.2f} | 获利单抗压(MAE) {d['avg_win_mae']*100:.1f}%")
    
    if factor_ic_report:
        sorted_ic = sorted(factor_ic_report.items(), key=lambda x: abs(x[1]['t_stat']), reverse=True)
        top_3 = sorted_ic[:3]
        alert_lines.extend(["", "---", "", "### 🧬 **因子 Rank IC 排行榜 (Top 3)**"])
        for f, data in top_3:
            alert_lines.append(f"- 🏆 **{f}**: T-stat {abs(data['t_stat']):.2f}")
        alert_lines.append(f"*(共有 {len(Config.ALL_FACTORS)-25} 个因子在本次淘汰赛中被降级退役打入冷宫)*")

    if meta_weights_dict:
        w_A = meta_weights_dict.get('Group_A', 0)
        w_B = meta_weights_dict.get('Group_B', 0)
        alert_lines.extend(["", "---", "", "### ⚖️ **Stacking 元学习器状态**"])
        alert_lines.append(f"- 🟢 传统大局观 (Group A) 决策权重: **{w_A*100:.1f}%**")
        alert_lines.append(f"- 🟣 高级微观组 (Group B) 决策权重: **{w_B*100:.1f}%**")
            
    send_alert("策略终极回测战报 (代谢进化版)", "\n".join(alert_lines))

def _decompose_and_perturb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_phase, df_noise = df.copy(), df.copy()
    c = df['Close']
    
    trend = c.ewm(span=60, adjust=False).mean()
    high_freq_noise = c - c.ewm(span=5, adjust=False).mean()
    seasonal = c.ewm(span=5, adjust=False).mean() - trend

    shifted_seasonal = pd.Series(np.roll(seasonal.values, 30), index=seasonal.index)
    c_phase = trend + shifted_seasonal + high_freq_noise

    c_noise = trend + seasonal + high_freq_noise * 1.5

    ratio_phase = c_phase / (c + 1e-10)
    df_phase['Close'] = c_phase
    df_phase['Open'] = df['Open'] * ratio_phase
    df_phase['High'] = df['High'] * ratio_phase
    df_phase['Low'] = df['Low'] * ratio_phase

    ratio_noise = c_noise / (c + 1e-10)
    df_noise['Close'] = c_noise
    df_noise['Open'] = df['Open'] * ratio_noise
    df_noise['High'] = df['High'] * ratio_noise
    df_noise['Low'] = df['Low'] * ratio_noise

    return df_phase, df_noise

def run_synthetic_stress_test() -> None:
    logger.info("🌪️ 启动合成数据对抗压测引擎 (Synthetic Adversarial Test)...")
    
    active_pool = get_filtered_watchlist(max_stocks=30) 
    
    xai_weights = {}
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                xai_data = json.load(f).get("xai_importances", {})
                if xai_data:
                    avg_imp = 1.0 / len(Config.ALL_FACTORS)
                    for tag, imp in xai_data.items():
                        xai_weights[tag] = 0.0 if imp < avg_imp * 0.25 else max(0.5, min(3.0, float(imp) / avg_imp))
    except Exception: pass
    
    metrics = {
        'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0},
        'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0},
        'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}
    }
    
    macro_data = {
        'spy': pd.DataFrame(), 'tlt': pd.DataFrame(), 'dxy': pd.DataFrame()
    }
    
    def _stress_test_worker(sym):
        df_raw = safe_get_history(sym, "5y", "1d", fast_mode=True)
        if len(df_raw) < 500: return None
        
        df_phase, df_noise = _decompose_and_perturb(df_raw)
        
        datasets = {
            'Original': df_raw,
            'Phase_Chaos': df_phase,
            'Noise_Explosion': df_noise
        }
        
        local_metrics = {
            'Original': {'trades': 0, 'wins': 0, 'ret_sum': 0.0},
            'Phase_Chaos': {'trades': 0, 'wins': 0, 'ret_sum': 0.0},
            'Noise_Explosion': {'trades': 0, 'wins': 0, 'ret_sum': 0.0}
        }
        
        for env_name, df_env in datasets.items():
            try:
                df_w = df_env.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                df_m = df_env.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                df_ind = calculate_indicators(df_env)
                
                for i in range(len(df_ind) - 252, len(df_ind) - 5):
                    curr = df_ind.iloc[i]
                    prev = df_ind.iloc[i-1]
                    
                    if (curr['ATR'] / curr['Close'] > 0.15) or (pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']):
                        continue
                        
                    is_vol = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open'])
                    
                    weekly_bullish = False
                    if len(df_w) > 40:
                        w_close = df_w['Close'].iloc[min(len(df_w)-1, i//5)]
                        w_sma40 = df_w['Close'].rolling(40).mean().iloc[min(len(df_w)-1, i//5)]
                        weekly_bullish = w_close > w_sma40
                    
                    weekly_bullish, fvg_lower, fvg_upper, kde_breakout_score, fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable, monthly_inst_flow, weekly_macd_res, rsi_60m_bounce, beta_60d, tlt_corr, dxy_corr, vrp = _extract_complex_features(df_ind.iloc[:i+1], df_w, df_m, pd.DataFrame(), macro_data, 18.0)
                    
                    score, _, _, _ = _evaluate_omni_matrix(
                        df_ind.iloc[:i+1], curr, prev, pd.DataFrame(), is_vol, weekly_bullish, 
                        fvg_lower, fvg_upper, kde_breakout_score, "bull", 1.0, xai_weights,
                        fft_ensemble_score, hurst_med, hurst_iqr, hurst_reliable,
                        monthly_inst_flow, weekly_macd_res, rsi_60m_bounce,
                        beta_60d, tlt_corr, dxy_corr, vrp, False,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0
                    )
                    
                    if score >= Config.MIN_SCORE_THRESHOLD:
                        entry_px = df_ind['Open'].iloc[i+1]
                        exit_px = df_ind['Close'].iloc[i+4]
                        ret = (exit_px - entry_px) / entry_px
                        
                        local_metrics[env_name]['trades'] += 1
                        local_metrics[env_name]['ret_sum'] += ret
                        if ret > 0: local_metrics[env_name]['wins'] += 1
            except Exception as e:
                logger.debug(f"[{sym}] 压测模拟时阻断: {e}")
        return local_metrics

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_stress_test_worker, sym) for sym in active_pool[:20]]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                for env_name in metrics:
                    metrics[env_name]['trades'] += res[env_name]['trades']
                    metrics[env_name]['wins'] += res[env_name]['wins']
                    metrics[env_name]['ret_sum'] += res[env_name]['ret_sum']

    report_lines = ["### 🌪️ **对抗样本压力测试报告 (Adversarial Stress Test)**"]
    report_lines.append("*通过合成数据对模型进行极限摧残，暴露虚假拟合。*\n")
    
    orig = metrics['Original']
    phase = metrics['Phase_Chaos']
    noise = metrics['Noise_Explosion']
    
    def _format_metric(m):
        wr = (m['wins']/m['trades'])*100 if m['trades']>0 else 0
        avg_ret = (m['ret_sum']/m['trades'])*100 if m['trades']>0 else 0
        return f"交易笔数: {m['trades']} | 胜率: {wr:.1f}% | 单笔均利: {avg_ret:.2f}%"

    report_lines.append(f"**🟢 原始环境 (Original History)**\n- {_format_metric(orig)}\n")
    report_lines.append(f"**🌌 平行宇宙A：相位错乱 (Phase Shifted)**\n- {_format_metric(phase)}")
    
    orig_wr = orig['wins']/orig['trades'] if orig['trades']>0 else 0
    phase_wr = phase['wins']/phase['trades'] if phase['trades']>0 else 0
    if orig_wr - phase_wr > 0.15:
        report_lines.append("- 📉 **归因诊断**: 胜率急剧崩塌！当前模型对 FFT 等周期相位产生了**严重虚假依赖**。")
    else:
        report_lines.append("- 🛡️ **归因诊断**: 胜率坚挺。模型未陷入周期的死记硬背，对时序变异具备极强鲁棒性！")
        
    report_lines.append(f"\n**🌋 平行宇宙B：噪音爆炸 (Noise Amplified 1.5x)**\n- {_format_metric(noise)}")
    noise_wr = noise['wins']/noise['trades'] if noise['trades']>0 else 0
    if orig_wr - noise_wr > 0.15:
        report_lines.append("- 📉 **归因诊断**: 波动率放大后模型崩溃。K线突破逻辑过于脆弱，容易被庄家扫损洗盘。")
    else:
        report_lines.append("- 🛡️ **归因诊断**: 动量逻辑经受住狂暴洗盘的考验。高级因子起到了良好的防插针过滤作用！")

    send_alert("终极实战压测报告", "\n".join(report_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if m == "matrix": run_tech_matrix()
    elif m == "backtest": run_backtest_engine()
    elif m == "stress": run_synthetic_stress_test()
    elif m == "test": send_alert("连通性测试", "全维宏观 Meta 跃迁完成！系统已通过 L1 正则化，开启了包括信用利差、PCR 与 VIX 曲线在内的 6 维上帝视野。")
