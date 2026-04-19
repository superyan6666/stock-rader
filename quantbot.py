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
from typing import List, Tuple, Dict
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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
    
    # 🪒 31 维全息张量：引入交互特征(Interaction)与滞后特征(Lag)
    ALL_FACTORS = [
        "米奈尔维尼", "强相对强度", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)",
        "突破缺口", "VWAP突破", "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", "口袋支点", 
        "VCP收缩", "筹码峰突破", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "威科夫弹簧(Spring)", 
        "跨时空共振(周线)", "CVD筹码净流入", "独立Alpha(脱钩)", "NR7极窄突破", "VPT量价共振",
        "带量金叉(交互)", "量价吸筹(交互)", "近3日突破(滞后)", "近3日巨量(滞后)"
    ]

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

# ================= 2. 数据工具模块 =================
def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(0.1, 0.3) if fast_mode else random.uniform(1.5, 3.0)
            time.sleep(sleep_sec)
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=auto_adjust, timeout=15)
            if not df.empty: return df
        except Exception as e:
            logger.warning(f"[{symbol}] 尝试 {attempt+1} 失败: {e}")
            if attempt == retries - 1: return pd.DataFrame()
            time.sleep((10 + attempt * 5) if "429" in str(e).lower() else (2 + attempt * 2))
    return pd.DataFrame()

def get_latest_news(symbol: str) -> str:
    try:
        news_data = yf.Ticker(symbol).news
        if news_data:
            latest = news_data[0]
            title, publisher = latest.get('title', ''), latest.get('publisher', '')
            if title:
                lower_title = title.lower()
                if any(kw in lower_title for kw in ['beat', 'raise', 'upgrade', 'strong', 'surge', 'rally', 'buy', 'bullish', 'record', 'profit']):
                    sentiment = "🟢 [利好]"
                elif any(kw in lower_title for kw in ['miss', 'cut', 'downgrade', 'weak', 'decline', 'sell', 'bearish', 'warn', 'loss']):
                    sentiment = "🔴 [利空]"
                else:
                    sentiment = "⚪ [中性]"
                return f"{sentiment} {title} ({publisher})"
    except Exception: pass
    return ""

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
        except Exception: pass
        
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
        except Exception: pass
    except Exception: pass
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Content-Type": "application/json"
        }
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
                    ai_wr_str = f" | ⚡LTR截面过滤 {t3['ai_win_rate']:.1%}" if 'ai_win_rate' in t3 else ""
                    return f"**📈 策略基底验证 (T+3):** 原始胜率 {t3['win_rate']:.1%}{ai_wr_str}{pf_str}"
    except Exception: pass
    return ""

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    if Config.WEBHOOK_URL:
        payload = {
            "msgtype": "markdown", 
            "markdown": {
                "title": f"【{Config.DINGTALK_KEYWORD}】{title}", 
                "text": f"## 🤖 【{Config.DINGTALK_KEYWORD}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"
            }
        }
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, timeout=10)
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
                }, timeout=10)
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
        if close_arr[i] > ub[i-1]: in_up[i] = True
        elif close_arr[i] < lb[i-1]: in_up[i] = False
        else: in_up[i] = in_up[i-1]
        
        if in_up[i] and in_up[i-1]: lb[i] = lb[i] if lb[i] > lb[i-1] else lb[i-1]
        if not in_up[i] and not in_up[i-1]: ub[i] = ub[i] if ub[i] < ub[i-1] else ub[i-1]
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
    df['VPT_MA20'] = df['VPT_Cum'].rolling(20).mean()

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
    
    delta = df['Close'].diff()
    up = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    down = -delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean()
    rs = up / (down + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Price_High_20'] = df['High'].rolling(20).max()
    
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    intra_strength = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
    df['CVD'] = (df['Volume'] * intra_strength).cumsum()
    df['CVD_MA20'] = df['CVD'].rolling(20).mean()
    
    df['Highest_22'] = df['High'].rolling(window=22).max()
    df['ATR_22'] = df['TR'].rolling(window=22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['Smart_Money_Flow'] = clv.rolling(window=10).mean()

    df['Recent_Price_Surge_3d'] = (df['Close'] / df['Open'] - 1).rolling(3).max().shift(1) * 100
    df['Recent_Vol_Surge_3d'] = (df['Volume'] / df['Vol_MA20']).rolling(3).max().shift(1)

    hist_vol_rolling = df['Volume'].shift(10).rolling(window=50, min_periods=10).quantile(0.8)
    vol_10d_mean = df['Volume'].rolling(window=10, min_periods=1).mean()
    vol_spike = (vol_10d_mean / (hist_vol_rolling + 1e-10)) > 2.5
    price_spike = df['Close'].pct_change().abs().rolling(window=5, min_periods=1).max() > 0.08
    df['Event_Risk'] = 0.0
    df.loc[vol_spike, 'Event_Risk'] += 0.6
    df.loc[price_spike, 'Event_Risk'] += 0.4
    df['Event_Risk'] = df['Event_Risk'].clip(upper=1.0)

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

def _extract_complex_features(df: pd.DataFrame, df_w: pd.DataFrame) -> Tuple[bool, float, float, float]:
    weekly_bullish = False
    if not df_w.empty and len(df_w) >= 40:
        sma40_w = df_w['Close'].rolling(40).mean().iloc[-1]
        if df_w['Close'].iloc[-1] > sma40_w:
            df_w['EMA_10'] = df_w['Close'].ewm(span=10, adjust=False).mean()
            df_w['EMA_30'] = df_w['Close'].ewm(span=30, adjust=False).mean()
            weekly_bullish = (df_w['Close'].iloc[-1] > df_w['EMA_10'].iloc[-1]) and (df_w['EMA_10'].iloc[-1] > df_w['EMA_30'].iloc[-1])
            
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
    poc_price = 0.0
    if not df_60.empty and df_60['Volume'].sum() > 0:
        counts, bins = np.histogram(df_60['Close'], bins=12, weights=df_60['Volume'])
        poc_price = (bins[np.argmax(counts)] + bins[np.argmax(counts)+1]) / 2.0
        
    return weekly_bullish, fvg_lower, fvg_upper, poc_price

def _extract_ml_features(df: pd.DataFrame, curr: pd.Series, prev: pd.Series, qqq_df: pd.DataFrame,
                         fvg_lower: float, poc_price: float, weekly_bullish: bool) -> List[float]:
    def safe_div(num, den, cap=20.0):
        if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
        return max(min(num / den, cap), -cap)

    rs_20, pure_alpha = 0.0, 0.0
    if not qqq_df.empty:
        m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
        if len(m_df) >= 40:
            qqq_ret = max(m_df['Close_y'].iloc[-1] / m_df['Close_y'].iloc[-20], 0.5)
            rs_20 = (m_df['Close_x'].iloc[-1] / m_df['Close_x'].iloc[-20]) / qqq_ret
            ret_stock = m_df['Close_x'].pct_change().dropna()
            ret_qqq = m_df['Close_y'].pct_change().dropna()
            cov_matrix = np.cov(ret_stock.iloc[-20:], ret_qqq.iloc[-20:])
            beta = cov_matrix[0,1] / (cov_matrix[1,1] + 1e-10) if cov_matrix[1,1] > 0 else 1.0
            pure_alpha = (ret_stock.iloc[-5:].mean() - beta * ret_qqq.iloc[-5:].mean()) * 252

    swing_high_10 = df['High'].iloc[-11:-1].max() if len(df) >= 12 else curr['High']

    macd_cross_strength = safe_div(curr['MACD'] - curr['Signal_Line'], abs(curr['Close']) * 0.01)
    vol_surge_ratio = safe_div(curr['Volume'], curr['Vol_MA20'], cap=50.0)
    cmf_val = curr['CMF']
    smf_val = curr['Smart_Money_Flow']

    feat_dict = {
        "米奈尔维尼": safe_div(curr['Close'] - curr['SMA_200'], curr['SMA_200']),
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
        "筹码峰突破": safe_div(curr['Close'] - poc_price, poc_price * 0.01) if poc_price > 0 else 0.0,
        "特性改变(ChoCh)": safe_div(curr['Close'] - swing_high_10, swing_high_10 * 0.01),
        "订单块(OB)": safe_div(curr['Close'] - curr['OB_Low'], curr['OB_High'] - curr['OB_Low'] + 1e-10) if pd.notna(curr['OB_High']) else 0.0,
        "AMD操盘": safe_div(min(curr['Open'], curr['Close']) - curr['Low'], curr['TR'] + 1e-10),
        "威科夫弹簧(Spring)": safe_div(curr['Swing_Low_20'] - curr['Low'], curr['Low'] * 0.01) if pd.notna(curr['Swing_Low_20']) else 0.0,
        "跨时空共振(周线)": 1.0 if weekly_bullish else 0.0,
        "CVD筹码净流入": safe_div(curr['CVD'] - curr['CVD_MA20'], abs(curr['CVD_MA20']) + 1e-10),
        "独立Alpha(脱钩)": pure_alpha,
        "NR7极窄突破": safe_div(curr['Range'], curr['ATR'] + 1e-10),
        "VPT量价共振": safe_div(curr['VPT_Cum'] - curr['VPT_MA20'], abs(curr['VPT_MA20']) + 1e-10),
        "带量金叉(交互)": macd_cross_strength * vol_surge_ratio,
        "量价吸筹(交互)": cmf_val * smf_val,
        "近3日突破(滞后)": curr['Recent_Price_Surge_3d'] if pd.notna(curr['Recent_Price_Surge_3d']) else 0.0,
        "近3日巨量(滞后)": curr['Recent_Vol_Surge_3d'] if pd.notna(curr['Recent_Vol_Surge_3d']) else 0.0
    }

    raw_array = [feat_dict.get(f, 0.0) for f in Config.ALL_FACTORS]
    return [float(x) for x in np.nan_to_num(raw_array, nan=0.0, posinf=20.0, neginf=-20.0)]

def _evaluate_omni_matrix(df: pd.DataFrame, curr: pd.Series, prev: pd.Series, qqq_df: pd.DataFrame, is_vol: bool,
                          weekly_bullish: bool, fvg_lower: float, fvg_upper: float, poc_price: float,
                          regime: str, w_mul: float, xai_weights: dict) -> Tuple[int, List[str], List[str]]:
    def get_fw(tag_name: str) -> float: return xai_weights.get(tag_name, 1.0)
    sig, factors, triggered = [], [], []
    
    if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200']:
        fw = get_fw("米奈尔维尼")
        if fw > 0: triggered.append(("米奈尔维尼", f"🏆 [主升趋势] 米奈尔维尼模板形成 (权:{fw:.2f}x)", 8 * w_mul * fw, "TREND"))
        
    if not qqq_df.empty:
        m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
        if len(m_df) >= 40:
            rs_20 = (m_df['Close_x'].iloc[-1]/m_df['Close_x'].iloc[-20]) / max(m_df['Close_y'].iloc[-1]/m_df['Close_y'].iloc[-20], 0.5)
            if rs_20 > 1.08: 
                fw = get_fw("强相对强度")
                if fw > 0: triggered.append(("强相对强度", f"⚡ [相对强度] 近20日动能大幅跑赢纳指 (权:{fw:.2f}x)", (7 if is_vol else 4) * w_mul * fw, "TREND"))
            
            ret_stock = m_df['Close_x'].pct_change().dropna()
            ret_qqq = m_df['Close_y'].pct_change().dropna()
            cov_matrix = np.cov(ret_stock.iloc[-20:], ret_qqq.iloc[-20:])
            beta = cov_matrix[0,1] / (cov_matrix[1,1] + 1e-10) if cov_matrix[1,1] > 0 else 1.0
            pure_alpha = (ret_stock.iloc[-5:].mean() - beta * ret_qqq.iloc[-5:].mean()) * 252
            if pure_alpha > 0.8 and is_vol:
                fw = get_fw("独立Alpha(脱钩)")
                if fw > 0: triggered.append(("独立Alpha(脱钩)", f"🪐 [独立Alpha] 强势剥离大盘Beta，爆发特质动能 (权:{fw:.2f}x)", 22 * w_mul * fw, "TREND"))
    
    if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
        fw = get_fw("MACD金叉")
        if fw > 0: triggered.append(("MACD金叉", f"🔥 [经典动能] MACD水上金叉起爆 (权:{fw:.2f}x)", 10 * w_mul * fw, "TREND"))
        
    if (curr['BB_Upper'] - curr['BB_Lower']) < (curr['KC_Upper'] - curr['KC_Lower']): 
        fw = get_fw("TTM Squeeze ON")
        if fw > 0: triggered.append(("TTM Squeeze ON", f"📦 [波动压缩] TTM Squeeze 挤流状态激活 (权:{fw:.2f}x)", 8 * w_mul * fw, "VOLATILITY"))

    if curr['Above_Cloud'] == 1 and curr['Tenkan'] > curr['Kijun']:
        fw = get_fw("一目多头")
        if fw > 0: triggered.append(("一目多头", f"🌥️ [趋势确认] 一目均衡表云上多头共振 (权:{fw:.2f}x)", 6 * w_mul * fw, "TREND"))

    if curr['SuperTrend_Up'] == 1 and curr['Close'] < curr['EMA_20'] * 1.02 and is_vol:
        fw = get_fw("强势回踩")
        if fw > 0: triggered.append(("强势回踩", f"🟢 [低吸点位] 超级趋势主升轨精准回踩 (权:{fw:.2f}x)", 10 * w_mul * fw, "REVERSAL"))
        
    if curr['CMF'] > 0.20: 
        fw = get_fw("机构控盘(CMF)")
        if fw > 0: triggered.append(("机构控盘(CMF)", f"🏦 [资金监控] CMF显示主力资金持续强控盘 (权:{fw:.2f}x)", 5 * w_mul * fw, "TREND"))
        
    gap_pct = (curr['Open'] - prev['Close']) / prev['Close']
    if gap_pct * 100 > max(1.5, (curr['ATR'] / prev['Close']) * 100 * 0.3) and gap_pct < 0.06 and is_vol:
        fw = get_fw("突破缺口")
        if fw > 0: triggered.append(("突破缺口", f"💥 [动能爆发] 放量跳空，留下底部突破缺口 (权:{fw:.2f}x)", 8 * w_mul * fw, "VOLATILITY"))
        
    if curr['Close'] > curr['VWAP_20'] and prev['Close'] <= curr['VWAP_20']:
        fw = get_fw("VWAP突破")
        if fw > 0: triggered.append(("VWAP突破", f"🌊 [量价突破] 放量逾越近20日VWAP机构均价线 (权:{fw:.2f}x)", 8 * w_mul * fw, "TREND"))
        
    if pd.notna(curr['AVWAP']) and curr['Close'] > curr['AVWAP'] and prev['Close'] <= curr['AVWAP']:
        fw = get_fw("AVWAP突破")
        if fw > 0: triggered.append(("AVWAP突破", f"⚓ [筹码夺回] 强势站上AVWAP锚定成本核心区 (权:{fw:.2f}x)", 12 * w_mul * fw, "TREND"))
        
    if fvg_lower > 0 and curr['Low'] <= fvg_upper and curr['Close'] > fvg_lower and is_vol:
        fw = get_fw("SMC失衡区")
        if fw > 0: triggered.append(("SMC失衡区", f"🧲 [SMC交易法] 精准回踩并测试前期机构失衡区(FVG) (权:{fw:.2f}x)", 15 * w_mul * fw, "REVERSAL"))
    
    if pd.notna(curr['Swing_Low_20']) and curr['Low'] < curr['Swing_Low_20'] and curr['Close'] > curr['Swing_Low_20'] and is_vol:
        fw = get_fw("流动性扫盘")
        if fw > 0: triggered.append(("流动性扫盘", f"🧹 [止损猎杀] 刺穿前低扫掉散户流动性后迅速诱空反转 (权:{fw:.2f}x)", 15 * w_mul * fw, "REVERSAL"))
        
    if curr['Smart_Money_Flow'] > 0.5 and curr['Close'] > curr['EMA_20']:
        fw = get_fw("聪明钱抢筹")
        if fw > 0: triggered.append(("聪明钱抢筹", f"🕵️ [暗中吸筹] Smart Money 指数显示连续尾盘抢筹 (权:{fw:.2f}x)", 6 * w_mul * fw, "QUANTUM"))
        
    if curr['Volume'] > curr['Vol_MA20'] * 2.0 and abs(curr['Close'] - curr['Open']) < curr['ATR'] * 0.5:
        fw = get_fw("巨量滞涨")
        if fw > 0: triggered.append(("巨量滞涨", f"🛑 [冰山订单] 巨量滞涨，极大概率为机构冰山挂单吸货 (权:{fw:.2f}x)", 12 * w_mul * fw, "QUANTUM"))
        
    atr_pct = (curr['ATR'] / prev['Close']) * 100
    day_chg = (curr['Close'] - curr['Open']) / curr['Open'] * 100
    if day_chg > max(3.0, atr_pct * 0.6) and curr['Volume'] > curr['Vol_MA20'] * 1.5:
        fw = get_fw("放量长阳")
        if fw > 0: triggered.append(("放量长阳", f"⚡ [动能脉冲] 强劲的日内放量大实体阳线 (权:{fw:.2f}x)", 12 * w_mul * fw, "QUANTUM"))
    
    if curr['Close'] > prev['Close'] and curr['Volume'] > curr['Max_Down_Vol_10'] > 0 and curr['Close'] >= curr['EMA_50'] and prev['Close'] <= curr['EMA_50'] * 1.02:
        fw = get_fw("口袋支点")
        if fw > 0: triggered.append(("口袋支点", f"💎 [口袋支点] 放量阳线成交量完全吞噬近期最大阴量 (权:{fw:.2f}x)", 12 * w_mul * fw, "REVERSAL"))
        
    if curr['Range_20'] > 0 and curr['Range_60'] > 0 and curr['Range_20'] < curr['Range_60'] * 0.5 and curr['Close'] > curr['SMA_50'] and is_vol:
        fw = get_fw("VCP收缩")
        if fw > 0: triggered.append(("VCP收缩", f"🌪️ [VCP形态] 经历极度价格波动压缩后的放量突破 (权:{fw:.2f}x)", 15 * w_mul * fw, "VOLATILITY"))
            
    if poc_price > 0 and prev['Close'] <= poc_price < curr['Close'] and is_vol:
        fw = get_fw("筹码峰突破")
        if fw > 0: triggered.append(("筹码峰突破", f"🏔️ [结构突破] 强势跨越 60日最密集筹码交易峰区 (权:{fw:.2f}x)", 12 * w_mul * fw, "TREND"))

    swing_high_10 = df['High'].iloc[-11:-1].max()
    if pd.notna(curr['Swing_Low_20']) and curr['Low'] > curr['Swing_Low_20'] and curr['Close'] > swing_high_10 and is_vol:
        fw = get_fw("特性改变(ChoCh)")
        if fw > 0: triggered.append(("特性改变(ChoCh)", f"🔀 [结构破坏] 突破近期反弹高点，完成 ChoCh 趋势逆转确认 (权:{fw:.2f}x)", 15 * w_mul * fw, "REVERSAL"))

    if pd.notna(curr['OB_High']) and curr['OB_Low'] > 0:
        if curr['Low'] <= curr['OB_High'] and curr['Close'] >= curr['OB_Low'] and curr['Close'] > curr['Open']:
            fw = get_fw("订单块(OB)")
            if fw > 0: triggered.append(("订单块(OB)", f"🧱 [机构订单块] 触达机构历史起爆底仓区并收出企稳阳线 (权:{fw:.2f}x)", 15 * w_mul * fw, "REVERSAL"))

    tr_val = curr['High'] - curr['Low'] + 1e-10
    lower_wick = curr['Open'] - curr['Low'] if curr['Close'] > curr['Open'] else curr['Close'] - curr['Low']
    upper_wick = curr['High'] - curr['Close'] if curr['Close'] > curr['Open'] else curr['High'] - curr['Open']
    if curr['Close'] > curr['Open'] and (lower_wick / tr_val) > 0.3 and (upper_wick / tr_val) < 0.15 and is_vol:
        fw = get_fw("AMD操盘")
        if fw > 0: triggered.append(("AMD操盘", f"🎭 [AMD诱空] 深度开盘诱空下杀后，全天拉升派发的操盘模型 (权:{fw:.2f}x)", 12 * w_mul * fw, "REVERSAL"))
        
    if pd.notna(curr['Swing_Low_20']) and curr['Low'] < curr['Swing_Low_20']:
        if curr['Volume'] < curr['Vol_MA20'] * 0.8 and curr['Close'] > (curr['Low'] + tr_val * 0.5):
            fw = get_fw("威科夫弹簧(Spring)")
            if fw > 0: triggered.append(("威科夫弹簧(Spring)", f"🏹 [威科夫测试] 跌破前低但抛压枯竭缩量，经典的Spring深蹲起爆 (权:{fw:.2f}x)", 18 * w_mul * fw, "REVERSAL"))
    
    if weekly_bullish and is_vol and (curr['Close'] > curr['Highest_22'] * 0.95):
        fw = get_fw("跨时空共振(周线)")
        if fw > 0: triggered.append(("跨时空共振(周线)", f"🌌 [多周期共振] 周线级别主升浪与日线级别放量的强力双重共振 (权:{fw:.2f}x)", 20 * w_mul * fw, "QUANTUM"))

    if curr['CVD'] > curr['CVD_MA20'] and prev['CVD'] <= prev['CVD_MA20'] and is_vol:
        fw = get_fw("CVD筹码净流入")
        if fw > 0: triggered.append(("CVD筹码净流入", f"🧬 [微观筹码] CVD(累积量价差) 突破均线，揭示真实的日内买盘压制 (权:{fw:.2f}x)", 12 * w_mul * fw, "QUANTUM"))

    if prev['NR7'] and prev['Inside_Bar'] and curr['Close'] > prev['High'] and is_vol:
        fw = get_fw("NR7极窄突破")
        if fw > 0: triggered.append(("NR7极窄突破", f"🎯 [极度压缩] 7日极窄压缩孕线完成向上爆破 (权:{fw:.2f}x)", 12 * w_mul * fw, "VOLATILITY"))

    if curr['VPT_Cum'] > curr['VPT_MA20'] and prev['VPT_Cum'] <= prev['VPT_MA20'] and is_vol:
        fw = get_fw("VPT量价共振")
        if fw > 0: triggered.append(("VPT量价共振", f"📈 [真实动能] VPT量价趋势线突破均线，量价配合绝对健康 (权:{fw:.2f}x)", 10 * w_mul * fw, "TREND"))

    if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line'] and is_vol:
        fw = get_fw("带量金叉(交互)")
        if fw > 0: triggered.append(("带量金叉(交互)", f"🔥 [交互共振] MACD水上金叉与成交量激增产生乘数效应 (权:{fw:.2f}x)", 12 * w_mul * fw, "TREND"))
        
    if curr['CMF'] > 0.15 and curr['Smart_Money_Flow'] > 0.4:
        fw = get_fw("量价吸筹(交互)")
        if fw > 0: triggered.append(("量价吸筹(交互)", f"🏦 [交互共振] 蔡金资金流与微观聪明钱同向深度吸筹 (权:{fw:.2f}x)", 10 * w_mul * fw, "QUANTUM"))
        
    if pd.notna(curr['Recent_Price_Surge_3d']) and curr['Recent_Price_Surge_3d'] > 4.0:
        fw = get_fw("近3日突破(滞后)")
        if fw > 0: triggered.append(("近3日突破(滞后)", f"⏱️ [滞后记忆] 近3日内曾发生暴涨，趋势处于亢奋延续期 (权:{fw:.2f}x)", 8 * w_mul * fw, "TREND"))
        
    if pd.notna(curr['Recent_Vol_Surge_3d']) and curr['Recent_Vol_Surge_3d'] > 2.5:
        fw = get_fw("近3日巨量(滞后)")
        if fw > 0: triggered.append(("近3日巨量(滞后)", f"⏱️ [滞后记忆] 近3日内曾爆出巨量，机构资金高度活跃 (权:{fw:.2f}x)", 8 * w_mul * fw, "VOLATILITY"))

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
        
    return int(score_raw), sig, factors

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
                             max_risk: float) -> Tuple[float, float, float, str]:
    tp_val = curr['Close'] + 3.0 * vix_scalar * curr['ATR']
    sl_chandelier = curr['Chandelier_Exit'] if pd.notna(curr['Chandelier_Exit']) else (curr['Close'] - 1.5 * vix_scalar * curr['ATR'])
    sl_val = max(sl_chandelier, curr['Close'] - 1.5 * vix_scalar * curr['ATR'])
    sl_pct_distance = max(0.01, (curr['Close'] - sl_val) / curr['Close']) 
    
    odds_b = 2.0
    kelly_fraction = ai_prob - (1.0 - ai_prob) / odds_b
    if kelly_fraction <= 0 or is_bearish_div:
        return tp_val, sl_val, 0.0, "❌ 放弃建仓 (AI盈亏比劣势 或 顶背离确认)"
        
    pos_percentage = min(min(0.20, kelly_fraction / 2.0), max_risk / sl_pct_distance)
    if is_credit_risk_high: pos_percentage *= 0.6
    if macro_gravity: pos_percentage *= 0.3 
    
    gap_pct = (curr['Open'] - prev['Close']) / prev['Close']
    if gap_pct > 0.03:
        pos_percentage *= 0.5 
        advice = f"⚠️ 建议仓位 {pos_percentage:.1%} (已防备跳空高开，单笔风险锁死)"
    elif (curr['Close'] - curr['EMA_20']) / curr['EMA_20'] > 0.12:
        pos_percentage = min(0.05, pos_percentage * 0.3)
        advice = f"⚠️ 建议仓位 {pos_percentage:.1%} (乖离过大，极小仓摸奖)"
    else:
        advice = f"✅ 建议仓位 {pos_percentage:.1%} (波动率平价 & 凯利双校验)"
        
    return tp_val, sl_val, pos_percentage, advice

def _apply_markowitz_decorrelation(reports: List[dict], price_history_dict: dict) -> List[dict]:
    reports.sort(key=lambda x: (x["ai_prob"], x["score"]), reverse=True)
    candidate_pool = reports[:25]
    final_reports = []
    if candidate_pool:
        try:
            corr_df_data = {r['symbol']: price_history_dict[r['symbol']] for r in candidate_pool if r['symbol'] in price_history_dict}
            if corr_df_data:
                corr_df = pd.DataFrame(corr_df_data).fillna(method='ffill').pct_change().corr()
                for candidate in candidate_pool:
                    if len(final_reports) >= 15: break
                    sym = candidate['symbol']
                    is_redundant = False
                    for accepted in final_reports:
                        acc_sym = accepted['symbol']
                        if sym in corr_df.columns and acc_sym in corr_df.columns and corr_df.loc[sym, acc_sym] > 0.80:
                            is_redundant = True
                            logger.info(f"🛡️ 反脆弱机制：踢除 {sym}，因为与已选高分股 {acc_sym} 相关性达 {corr_df.loc[sym, acc_sym]:.2f}。")
                            break
                    if not is_redundant: final_reports.append(candidate)
            else: final_reports = candidate_pool[:15]
        except Exception as e:
            logger.warning(f"反脆弱协方差矩阵构建失败: {e}")
            final_reports = candidate_pool[:15]
    return final_reports

def run_tech_matrix() -> None:
    max_risk = 0.015 
    pain_warning = ""
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                t3 = json.load(f).get('overall', {}).get('T+3', {})
                if t3.get('max_cons_loss', 0) >= 4 or t3.get('profit_factor', 1.0) < 1.2:
                    max_risk = 0.0075  
                    pain_warning = "\n- 🩸 **痛觉神经激活**: 近期回测遭重挫，引擎已主动防守降杠杆！"
    except Exception: pass

    active_pool = get_filtered_watchlist(max_stocks=150)
    regime, regime_desc, qqq_df, is_credit_risk_high, macro_gravity = get_market_regime(active_pool)
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    vix_scalar = max(0.6, min(1.4, 18.0 / max(vix, 1.0)))
    if macro_gravity: max_risk = min(max_risk, 0.005)
    
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
                            xai_weights[tag], pruned_factors.append(tag) = 0.0, tag
                        else:
                            xai_weights[tag] = max(0.5, min(3.0, float(imp) / avg_imp))
    except Exception: pass
        
    clf_model = None
    if os.path.exists(Config.MODEL_FILE):
        try:
            import pickle
            with open(Config.MODEL_FILE, 'rb') as f: clf_model = pickle.load(f)
        except Exception: pass

    raw_reports = []
    price_history_dict = {} 
    
    for sym in active_pool:
        if is_alerted(sym): continue
        try:
            df_w = safe_get_history(sym, "2y", "1wk", fast_mode=True)
            df = safe_get_history(sym, "8mo", "1d", fast_mode=True) 
            if len(df) < 150: continue
            df = calculate_indicators(df)
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            is_untradeable = False
            if (curr['ATR'] / curr['Close'] > 0.15): is_untradeable = True
            if pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']: is_untradeable = True
            
            price_history_dict[sym] = df['Close'].iloc[-60:]
            is_vol = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open'])
            
            weekly_bullish, fvg_lower, fvg_upper, poc_price = _extract_complex_features(df, df_w)
            ml_features_array = _extract_ml_features(df, curr, prev, qqq_df, fvg_lower, poc_price, weekly_bullish)
            
            base_score, sig, factors = _evaluate_omni_matrix(
                df, curr, prev, qqq_df, is_vol, weekly_bullish, fvg_lower, fvg_upper, 
                poc_price, regime, w_mul, xai_weights
            )
            
            total_score, is_bearish_div, sig = _apply_market_filters(
                curr, prev, sym, base_score, sig, black_hole_sectors, leading_sectors, lagging_sectors
            )
            
            sym_sec = Config.get_sector_etf(sym)
            raw_reports.append({
                "sym": sym, "curr": curr, "prev": prev, "total_score": total_score, "is_bearish_div": is_bearish_div,
                "sig": sig, "factors": factors, "ml_features": ml_features_array, "is_untradeable": is_untradeable,
                "sym_sec": sym_sec
            })
        except Exception: pass

    if clf_model and raw_reports:
        X_batch = np.array([r['ml_features'] for r in raw_reports])
        if hasattr(clf_model, 'predict_proba'):
            try:
                class_idx = np.where(clf_model.classes_ == 1)[0][0] if 1 in clf_model.classes_ else 0
                probs = clf_model.predict_proba(X_batch)[:, class_idx]
            except Exception:
                probs = np.full(len(raw_reports), 0.52)
        else:
            raw_scores = clf_model.predict(X_batch)
            if len(raw_scores) > 1:
                z_scores = (raw_scores - np.mean(raw_scores)) / (np.std(raw_scores) + 1e-10)
                probs = 1 / (1 + np.exp(-z_scores))
            else:
                probs = np.full(len(raw_reports), 0.6)
                
        for i, r in enumerate(raw_reports):
            r['ai_prob'] = float(probs[i])
    else:
        for r in raw_reports: r['ai_prob'] = 0.52

    reports, background_pool, all_raw_scores = [], [], []
    for r in raw_reports:
        if r['total_score'] > 0: all_raw_scores.append(r['total_score'])
        tp_val, sl_val, pos_percentage, pos_advice = _calculate_position_size(
            r['curr'], r['prev'], r['ai_prob'], vix_scalar, r['is_bearish_div'], is_credit_risk_high, macro_gravity, max_risk
        )

        stock_data_pack = {
            "symbol": r['sym'], "score": r['total_score'], "ai_prob": r['ai_prob'], "signals": r['sig'][:8], 
            "factors": r['factors'], "ml_features": r['ml_features'],
            "curr_close": float(r['curr']['Close']), "tp": float(tp_val), "sl": float(sl_val), 
            "news": get_latest_news(r['sym']), "sector": r['sym_sec'], "pos_advice": pos_advice
        }

        if not r['is_untradeable'] and r['total_score'] > 0 and pos_percentage > 0:
            if check_earnings_risk(r['sym']):
                r['sig'].append("💣 [财报雷区] 近5日发财报,风险极高")
                r['total_score'] = int(r['total_score'] * 0.5)
                pos_percentage *= 0.2
                stock_data_pack["pos_advice"] = f"⚠️ 建议仓位 {pos_percentage:.1%} (财报赌博极限风控)"
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

        final_reports = _apply_markowitz_decorrelation(reports, price_history_dict)
        for r in final_reports: set_alerted(r["symbol"])
        
        final_symbols = {r['symbol'] for r in final_reports}
        unselected_background = [s for s in background_pool if s['symbol'] not in final_symbols]
        random.shuffle(unselected_background)
        final_shadow_pool = unselected_background[:150]
        
        for r in final_shadow_pool: set_alerted(r["symbol"], is_shadow=True, shadow_data=r)
            
        txts = []
        for idx, r in enumerate(final_reports):
            icon = ['🥇', '🥈', '🥉'][idx] if idx < 3 else '🔸'
            sigs_fmt = "\n".join([f"- {s}" for s in r["signals"]])
            news_fmt = f"\n- 📰 {r['news']}" if r['news'] else ""
            ai_display = f"🔥 **{r.get('ai_prob', 0):.1%}**" if r.get('ai_prob', 0) > 0.60 else f"{r.get('ai_prob', 0):.1%}"
            
            txts.append(
                f"### {icon} **{r['symbol']}** | 🤖 LTR 截面排序概率: {ai_display} | 🌟 逻辑共振度: {r['score']}分\n"
                f"**💡 机构交易透视:**\n{sigs_fmt}{news_fmt}\n\n"
                f"**💰 绝对风控界限:**\n"
                f"- 💵 现价: `{r['curr_close']:.2f}`\n"
                f"- ⚖️ {r.get('pos_advice', '✅ 标准仓位')}\n"
                f"- 🎯 建议止盈: **${r['tp']:.2f}**\n"
                f"- 🛡️ 吊灯止损: **${r['sl']:.2f} (最高价回落保护)**\n"
                f"- 📈 离场纪律: **跌破止损防线请无条件市价清仓！**"
            )

        perf = load_strategy_performance_tag()
        pruned_desc = f"\n- ✂️ 神经突触修剪: 已成功忘却 **{len(pruned_factors)}** 个低效因子，达成绝对至简" if pruned_factors else ""
        header = f"**📊 宏观引力与系统状态:**\n- {vix_desc}\n- {regime_desc}{pain_warning}{pruned_desc}\n- ⚔️ 今日截面淘汰线 (Top 15%): **{dynamic_min_score:.1f}分**"
        
        final_content = (f"{perf}\n\n{header}\n\n---\n\n" if perf else f"{header}\n\n---\n\n") + \
                        "\n\n---\n\n".join(txts) + \
                        f"\n\n*(防穿越净化: 系统已搭载 Purged Walk-Forward 交叉验证机制，严防信息泄露，锁定最优迭代边界。)*"
        
        send_alert("量化诸神之战 (Purged防穿越版)", final_content)
        
        with open(Config.get_current_log_file(), "a", encoding="utf-8") as f:
            log_entry = {
                "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), 
                "top_picks": [{"symbol": r["symbol"], "score": r["score"], "signals": r["signals"], "factors": r.get("factors", []), "ml_features": r.get("ml_features", []), "tp": r.get("tp"), "sl": r.get("sl")} for r in final_reports],
                "shadow_pool": [{"symbol": r["symbol"], "score": r["score"], "factors": r.get("factors", []), "ml_features": r.get("ml_features", [])} for r in final_shadow_pool]
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    else:
        logger.info("📭 本次矩阵扫描无标的突破 Top 15% 截面排位，宁缺毋滥，保持静默。")

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
                        
                        for p in daily_trades:
                            trades.append({'date': log['date'], 'symbol': p['symbol'], 'signals': p.get('signals', []), 'factors': p.get('factors', []), 'ml_features': p.get('ml_features', []), 'tp': p.get('tp', float('inf')), 'sl': p.get('sl', 0)})
                    except: pass
        except Exception as e:
            logger.debug(f"读取日志分片 {lf} 失败: {e}")
            
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
    
    stats, factor_rets = {'T+1': [], 'T+3': [], 'T+5': []}, {}
    trades_with_ret = []
    
    ai_filtered_wins, ai_filtered_total = 0, 0
    mae_mfe_records = {'T+1': [], 'T+3': [], 'T+5': []}
    
    clf_model = None
    if os.path.exists(Config.MODEL_FILE):
        try:
            import pickle
            with open(Config.MODEL_FILE, 'rb') as f:
                clf_model = pickle.load(f)
        except Exception: pass
    
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
                    stats[f'T+{d}'].append(ret)
                    
                    if d == 3:
                        factor_list = t.get('factors', [])
                        if not factor_list:
                            for sig_txt in t.get('signals', []):
                                m = re.search(r'\[(.*?)\]', sig_txt)
                                if m: factor_list.append(m.group(1).split(" ")[0])
                                
                        for f_name in factor_list:
                            factor_rets.setdefault(f"[{f_name}]", []).append(ret)
                            
                        ml_feats = t.get('ml_features', [])
                        if ml_feats and len(ml_feats) == len(Config.ALL_FACTORS):
                            trades_with_ret.append({
                                'date': t['date'],
                                'ml_features': ml_feats,
                                'ret': ret
                            })
                            
                            if clf_model and hasattr(clf_model, 'predict'):
                                try:
                                    score = clf_model.predict([ml_feats])[0]
                                    if score > 0: 
                                        ai_filtered_total += 1
                                        if ret > 0: ai_filtered_wins += 1
                                except Exception: pass
    
    feature_importances_dict = {}
    trade_df = pd.DataFrame(trades_with_ret)
    if len(trade_df) >= 30:
        try:
            from lightgbm import LGBMRanker, early_stopping
            import pickle
            
            trade_df = trade_df.sort_values('date')
            
            relevance_list = []
            for date, group in trade_df.groupby('date', sort=False):
                if len(group) < 3:
                    relevance_list.append(pd.Series(0, index=group.index))
                else:
                    try:
                        relevance_list.append(pd.qcut(group['ret'], 5, labels=False, duplicates='drop'))
                    except Exception:
                        relevance_list.append(pd.Series(0, index=group.index))
            trade_df['relevance'] = pd.concat(relevance_list)
            
            X_train = np.vstack(trade_df['ml_features'].values)
            y_train = trade_df['relevance'].values
            groups = trade_df.groupby('date', sort=False).size().values
            
            base_lgbm_params = dict(
                objective="lambdarank",
                metric="ndcg",
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                n_estimators=150
            )
            
            unique_dates = trade_df['date'].unique()
            # 🚀 时间序列交叉验证 (Purged Walk-Forward CV) 核心逻辑：
            # 针对 GitHub Actions 短时间重训的特性，采用单折验证动态寻找 Early Stopping
            if len(unique_dates) > 30:
                # 划分 80% 训练集, 20% 验证集
                val_size = max(1, int(len(unique_dates) * 0.2))
                train_dates = unique_dates[:-val_size]
                val_dates = unique_dates[-val_size:]
                
                # 🚀 Purge 净化：强行剔除训练集末尾 5 个交易日，切断 T+5 标签重叠导致的未来函数泄露！
                purge_days = 5
                train_dates = train_dates[:-purge_days] if len(train_dates) > purge_days else train_dates
                
                train_mask = trade_df['date'].isin(train_dates)
                val_mask = trade_df['date'].isin(val_dates)
                
                X_train_cv, y_train_cv = X_train[train_mask], y_train[train_mask]
                groups_train_cv = trade_df[train_mask].groupby('date', sort=False).size().values
                
                X_val_cv, y_val_cv = X_train[val_mask], y_train[val_mask]
                groups_val_cv = trade_df[val_mask].groupby('date', sort=False).size().values
                
                clf_cv = LGBMRanker(**base_lgbm_params)
                # 💡 使用被 Purge 洗净的验证集进行 Early Stopping，防范过拟合
                clf_cv.fit(
                    X_train_cv, y_train_cv, group=groups_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)], eval_group=[groups_val_cv],
                    callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
                )
                
                best_iters = clf_cv.best_iteration_ if clf_cv.best_iteration_ else 100
                logger.info(f"🛡️ Purged CV 防穿越净化完成：验证集防过拟合最佳树迭代次数为 {best_iters} 棵。")
                
                # 将最佳参数带回全量数据重新拟合
                clf = LGBMRanker(**base_lgbm_params)
                clf.set_params(n_estimators=best_iters)
                clf.fit(X_train, y_train, group=groups)
            else:
                # 数据不足 30 天，暂时保守拟合
                clf = LGBMRanker(**base_lgbm_params)
                clf.fit(X_train, y_train, group=groups)
            
            with open(Config.MODEL_FILE, 'wb') as f:
                pickle.dump(clf, f)
            logger.info(f"🧠 【排序学习跃迁】搭载 Purged Walk-Forward 的全息排序模型已重训落盘。")
            
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                for factor, imp in zip(Config.ALL_FACTORS, importances):
                    feature_importances_dict[factor] = float(imp)
        except ImportError:
            logger.warning("未检测到 lightgbm 环境，已跳过 Ranker 训练。请确保在 requirements.txt 中添加了 lightgbm。")
        except Exception as e:
            logger.error(f"ML 模型训练受阻: {e}")
            
    res = {}
    for p, r in stats.items():
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
            
    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump({"overall": res, "factors": f_res, "xai_importances": feature_importances_dict}, f, indent=4)
    
    report_md = [f"# 📈 自动量化战报与 AI 透视\n**更新:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n## ⚔️ 核心表现评估\n| 周期 | 原始胜率 | ⚡LTR截面过滤 | 均收益 | 盈亏比 | Sharpe | 胜单平均抗压(MAE) | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for p in ['T+1', 'T+3', 'T+5']:
        d = res.get(p, {'win_rate':0,'avg_ret':0,'profit_factor':0,'sharpe':0,'avg_win_mae':0,'max_cons_loss':0,'total_trades':0})
        ai_str = f"**{d.get('ai_win_rate', 0.0)*100:.1f}%**" if 'ai_win_rate' in d else "-"
        report_md.append(f"| {p} | {d['win_rate']*100:.1f}% | {ai_str} | {d['avg_ret']*100:+.2f}% | {d['profit_factor']:.2f} | {d['sharpe']:.2f} | {d['avg_win_mae']*100:.1f}% | {d['total_trades']} |")
    
    if feature_importances_dict:
        report_md.append("\n## 🧠 XAI (解释性人工智能) - 驱动当期市场的核心因子权重\n| 因子特征 | AI 分配重要性 (Feature Importance) |\n|:---|:---:|")
        sorted_xai = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        for tag, imp in sorted_xai: 
            report_md.append(f"| {tag} | {imp*100:.1f}% |")

    if f_res:
        report_md.append(f"\n## 🧬 终极 {len(Config.ALL_FACTORS)} 维高解释度矩阵因子群 (T+3)\n| 因子 | 胜率 | 盈亏比 | 触发次数 |\n|:---|:---:|:---:|:---:|")
        sorted_f = sorted(f_res.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        for tag, d in sorted_f: report_md.append(f"| {tag} | {d['win_rate']*100:.1f}% | {d['profit_factor']:.2f} | {d['count']} |")
    
    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_md))

    alert_lines = ["### 📊 **机构级回测报表 (含 MAE/MFE 归因分析)**"]
    for p, d in res.items(): 
        ai_text = f" | ⚡LTR截面过滤: **{d['ai_win_rate']*100:.1f}%**" if 'ai_win_rate' in d else ""
        alert_lines.append(f"- **{p}:** 原始胜率 {d['win_rate']*100:.1f}%{ai_text} | 盈亏比 {d['profit_factor']:.2f} | 获利单抗压(MAE) {d['avg_win_mae']*100:.1f}%")
    
    if feature_importances_dict:
        alert_lines.extend(["", "---", "", "### 🧠 **XAI 市场驱动因子 (LightGBM Ranker)**"])
        sorted_xai = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        for idx, (tag, imp) in enumerate(sorted_xai[:3]):
            icon = ['🔥','🔥','🔥'][idx]
            alert_lines.append(f"- {icon} **{tag}**: 贡献度 {imp*100:.1f}%")
            
    send_alert("策略终极回测战报 (Purged CV版)", "\n".join(alert_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if m == "matrix": run_tech_matrix()
    elif m == "backtest": run_backtest_engine()
    elif m == "test": send_alert("连通性测试", "防穿越武装完毕！系统已搭载 Purged Walk-Forward 切割器，验证集将前置剥离最后5天以阻断未来函数，锁定真实最优参数。")
