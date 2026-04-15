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
from typing import List, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# 忽略第三方库引发的各种杂音警告，保持 GitHub Actions 日志清爽
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
    
    CORE_WATCHLIST: List[str] = [
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
    CROWDING_EXCLUDE_SECTORS: List[str] = ["QQQ"]
    
    LOG_PREFIX: str = "backtest_log_"
    STATS_FILE: str = "strategy_stats.json"
    REPORT_FILE: str = "backtest_report.md"
    CACHE_FILE: str = "tickers_cache.json"

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

def get_filtered_watchlist(max_stocks: int = 120) -> List[str]:
    logger.info(">>> 漏斗过滤：从多维度数据源拉取名单 (含纳指与中盘)...")
    tickers = set(Config.CORE_WATCHLIST)
    
    if os.path.exists(Config.CACHE_FILE):
        try:
            mtime = os.path.getmtime(Config.CACHE_FILE)
            if time.time() - mtime < 7 * 86400:
                with open(Config.CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_tickers = json.load(f)
                    tickers.update(cached_tickers)
                    logger.info(f"♻️ 成功加载本地存活缓存: {len(cached_tickers)} 只。")
            else:
                logger.info("🗑️ 本地缓存已过期(>7天)，将执行全局深度清洗重建。")
        except Exception as e:
            logger.warning(f"⚠️ 缓存读取失败: {e}")

    online_fetched = 0
    try:
        sp500_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        resp = requests.get(sp500_url, headers=_GLOBAL_HEADERS, timeout=10)
        if resp.status_code == 200:
            from io import StringIO
            df_sp500 = pd.read_csv(StringIO(resp.text))
            if 'Symbol' in df_sp500.columns:
                fetched = df_sp500['Symbol'].dropna().astype(str).str.replace('.', '-').tolist()
                tickers.update(fetched)
                online_fetched += len(fetched)
    except Exception as e:
        logger.warning(f"⚠️ 标普500动态拉取受限: {e}")

    try:
        from io import StringIO
        wiki_urls = [
            'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
            'https://en.wikipedia.org/wiki/Nasdaq-100'
        ]
        for w_url in wiki_urls:
            w_resp = requests.get(w_url, headers=_GLOBAL_HEADERS, timeout=10)
            if w_resp.status_code == 200:
                for table in pd.read_html(StringIO(w_resp.text), flavor='lxml'):
                    col = next((c for c in ['Symbol', 'Ticker symbol', 'Ticker'] if c in table.columns), None)
                    if col:
                        fetched = table[col].dropna().astype(str).str.replace('.', '-').tolist()
                        tickers.update(fetched)
                        online_fetched += len(fetched)
                        break
    except Exception as e:
        logger.warning(f"⚠️ Wiki 拓展列表获取受限: {e}")
        
    if online_fetched == 0 and len(tickers) <= len(Config.CORE_WATCHLIST):
        logger.warning("🔴 外部数据源全部脆断，将依靠【护城河名单】与【本地自洁缓存】安全运行！")
    
    tickers_list = list(tickers)
    logger.info(f"✅ 汇总海选池: {len(tickers_list)} 只标的。")

    try:
        chunk_size = 50 
        dfs = []
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            chunk_df = yf.download(chunk, period="5d", progress=False, threads=2)
            if not chunk_df.empty: dfs.append(chunk_df)
            if i + chunk_size < len(tickers_list): time.sleep(random.uniform(2.0, 3.5))
                
        if not dfs: raise ValueError("批量下载失败")
        df = pd.concat(dfs, axis=1)
        
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close'] if 'Close' in df.columns else df.xs('Close', level=0, axis=1)
            volume_df = df['Volume'] if 'Volume' in df.columns else df.xs('Volume', level=0, axis=1)
        else:
            close_df, volume_df = df, pd.DataFrame(1e6, index=df.index, columns=df.columns)

        available_tickers = set(close_df.columns) if isinstance(close_df, pd.DataFrame) else set()
        missing_tickers = set(tickers_list) - available_tickers
        
        if missing_tickers:
            logger.info(f"🛡️ 数据自净: 发现 {len(missing_tickers)} 只失效/退市标的，已将其从本次计算与缓存中永久抹除。")
            
        if len(available_tickers) > 200:
            try:
                with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(available_tickers), f)
            except Exception as e:
                logger.debug(f"写入缓存失败: {e}")

        closes = close_df.dropna(axis=1, how='all').ffill().iloc[-1]
        volumes = volume_df.dropna(axis=1, how='all').mean()
        turnovers = (closes * volumes).dropna()
        
        valid_turnovers = turnovers[(closes > 10.0) & (turnovers > 30_000_000)]
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        if top_tickers:
            logger.info(f"✅ 漏斗完成！锁定极高流动性标的: {len(top_tickers)} 只。")
            return top_tickers
        return Config.CORE_WATCHLIST[:max_stocks]
    except Exception as e:
        logger.error(f"❌ 批量下载崩溃: {e}")
        return Config.CORE_WATCHLIST[:max_stocks]

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                t3 = stats_data.get("overall", {}).get("T+3") if "overall" in stats_data else stats_data.get("T+3")
                if t3 and t3.get('total_trades', 0) > 0:
                    return f"**📈 策略表现 (T+3):** 胜率 {t3['win_rate']:.1%} | 均收益 {t3['avg_ret']:+.2%} "
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
        qqq_df = qqq_df_for_shadow if qqq_df_for_shadow is not None else safe_get_history(Config.INDEX_ETF, period="2mo", interval="1d", fast_mode=True)
        if qqq_df is not None and len(qqq_df) >= 20:
            vix = qqq_df['Close'].pct_change().dropna().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 * 1.2
            is_simulated = True
            
    prefix = "影子VIX" if is_simulated else "VIX"
    if vix > 30: return vix, f"🚨 极其恐慌 ({prefix}: {vix:.2f})"
    if vix > 25: return vix, f"⚠️ 市场恐慌 ({prefix}: {vix:.2f})"
    if vix < 15: return vix, f"✅ 市场平静 ({prefix}: {vix:.2f})"
    return vix, f"⚖️ 正常波动 ({prefix}: {vix:.2f})"

def get_market_regime() -> Tuple[str, str, pd.DataFrame]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", auto_adjust=False, fast_mode=True)
    if len(df) < 200: return "range", "数据不足，默认震荡", df
    
    c_close = df['Close'].ffill().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    ma50_curr = df['Close'].rolling(50).mean().iloc[-1]
    ma50_prev = df['Close'].rolling(50).mean().iloc[-20]
    
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if c_close > ma200:
        if trend_20d > 0.02: return "bull", "🐂 牛市主升阶段", df
        else: return "range", "⚖️ 牛市高位震荡", df
    else:
        if c_close > ma50_curr and ma50_curr > ma50_prev and trend_20d > 0.04:
            return "rebound", "🦅 熊市超跌反弹 (V反)", df
        elif trend_20d < -0.02: 
            return "bear", "🐻 熊市回调阶段", df
        else: 
            return "range", "⚖️ 熊市底部震荡", df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['Open'] = df['Open'].ffill()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['High_52W'] = df['High'].rolling(window=252, min_periods=120).max()
    df['Low_52W'] = df['Low'].rolling(window=252, min_periods=120).min()
    df['OBV'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()
    
    delta = df['Close'].diff()
    up = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    down = -delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean()
    rs = up / (down + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['RSI_P20'] = df['RSI'].shift(1).rolling(window=120, min_periods=60).quantile(0.20).fillna(30.0)

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    hl2, atr10 = (df['High'] + df['Low']) / 2, df['TR'].rolling(window=10).mean()
    ub, lb = (hl2 + 3 * atr10).values, (hl2 - 3 * atr10).values
    
    in_up = np.zeros(len(df), dtype=bool)
    in_up[0] = True 
    
    for i in range(1, len(df)):
        if pd.isna(ub[i-1]) or pd.isna(lb[i-1]): 
            in_up[i] = in_up[i-1]
            continue
        in_up[i] = True if df['Close'].values[i] > ub[i-1] else (False if df['Close'].values[i] < lb[i-1] else in_up[i-1])
        if in_up[i] and in_up[i-1]: lb[i] = max(lb[i], lb[i-1])
        if not in_up[i] and not in_up[i-1]: ub[i] = min(ub[i], ub[i-1])
    df['SuperTrend_Up'] = in_up.astype(int)

    atr20 = df['TR'].rolling(window=20).mean()
    df['KC_Upper'], df['KC_Lower'] = df['EMA_20'] + 1.5 * atr20, df['EMA_20'] - 1.5 * atr20
    bb_ma, bb_std = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Upper'], df['BB_Lower'] = bb_ma + 2 * bb_std, bb_ma - 2 * bb_std

    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    df['Cloud_Twist'] = (df['SenkouA'] > df['SenkouB']).astype(int)

    hl_diff = (df['High'] - df['Low']).replace(0, 1e-10)
    dollar_vol = df['Close'] * df['Volume']
    vol_sum20 = dollar_vol.rolling(20).sum() + 1e-10
    df['CMF'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_diff * dollar_vol).rolling(20).sum() / vol_sum20
    df['CMF'] = df['CMF'].clip(lower=-1.0, upper=1.0).fillna(0.0)

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
def run_volatility_sentinel() -> None:
    curr_hour = datetime.now(timezone.utc).hour
    if not (13 <= curr_hour <= 21): 
        logger.info("💤 当前非美股交易核心时段 (UTC 13-21)，哨兵按风控逻辑保持休眠。")
        return
        
    active_pool = get_filtered_watchlist(max_stocks=40)
    alerts = []
    
    vix, _ = get_vix_level()
    vix_scalar = max(0.6, min(1.4, 18.0 / max(vix, 1.0)))
    
    for sym in active_pool:
        try:
            df_h = safe_get_history(sym, period="2d", interval="1h", fast_mode=True)
            if len(df_h) < 2: continue
            curr_px = df_h['Close'].ffill().iloc[-1]
            hr_chg = (curr_px - df_h['Open'].iloc[-1]) / df_h['Open'].iloc[-1] * 100
            
            is_alert = False
            alert_reason = ""
            if abs(hr_chg) > 3.5: 
                is_valid_vol = True
                if len(df_h) >= 5:
                    vol_last_hour = df_h['Volume'].iloc[-1]
                    vol_ma_5h = df_h['Volume'].rolling(5).mean().iloc[-1]
                    if vol_last_hour < vol_ma_5h * 1.2:
                        is_valid_vol = False  
                
                if is_valid_vol:
                    is_alert = True
                    alert_reason = f"- ⚡ 1h脉冲: **{hr_chg:+.2f}%** (放量确认)"

            df_d = safe_get_history(sym, period="1mo", interval="1d", fast_mode=True)
            trade_plan = ""
            if len(df_d) >= 15:
                df_d['TR'] = pd.concat([df_d['High']-df_d['Low'], (df_d['High']-df_d['Close'].shift()).abs(), (df_d['Low']-df_d['Close'].shift()).abs()], axis=1).max(axis=1)
                atr = df_d['TR'].rolling(14).mean().iloc[-1]
                
                tp, sl = curr_px + 3.0 * vix_scalar * atr, curr_px - 1.5 * vix_scalar * atr
                
                trade_plan = (
                    f"**💰 交易计划:**\n"
                    f"- 💵 现价: `{curr_px:.2f}`\n"
                    f"- 🎯 止盈: **${tp:.2f}**\n"
                    f"- 🛡️ 止损: **${sl:.2f}**"
                )
                gap = (df_d['Open'].iloc[-1] - df_d['Close'].iloc[-2]) / df_d['Close'].iloc[-2] * 100
                if abs(gap) > 4:
                    is_alert = True
                    reason = f"- 💥 跳空缺口: **{gap:+.2f}%**"
                    alert_reason = f"{alert_reason}\n{reason}" if alert_reason else reason
            
            if is_alert:
                news = get_latest_news(sym)
                news_block = f"\n\n**📰 最新资讯:**\n- {news}" if news else ""
                
                msg = (
                    f"### 🚨 {sym} | 盘中异动 {'🚀' if hr_chg>0 else '🩸'}\n"
                    f"**💡 异动捕捉:**\n{alert_reason}\n\n"
                    f"{trade_plan}{news_block}"
                )
                alerts.append(msg)
        except Exception: pass
        
    if alerts: 
        send_alert("高频哨兵预警", "\n\n---\n\n".join(alerts))
    else:
        logger.info("📭 本次扫描未发现符合脉冲与放量条件的标的，保持静默。")

def run_tech_matrix() -> None:
    regime, regime_desc, qqq_df = get_market_regime()
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    
    vix_scalar = max(0.6, min(1.4, 18.0 / max(vix, 1.0)))
    
    sector_data = {etf: (sdf['Close'].ffill().iloc[-1] / sdf['Close'].iloc[-20]) - 1 
                  for etf in Config.SECTOR_MAP.keys() 
                  if not (sdf := safe_get_history(etf, "2mo", "1d", fast_mode=True)).empty and len(sdf) >= 20}
                  
    health_score = -0.9 if vix > 30 else (-0.6 if vix > 25 else (0.9 if vix < 15 and regime == 'bull' else (0.6 if regime == 'bull' else (0.3 if regime == 'rebound' else (-0.7 if regime == 'bear' else 0.0)))))
    w_mul, min_score = 1.0 + health_score * 0.8, max(Config.MIN_SCORE_THRESHOLD, 8 + round(health_score * -6))
    
    factor_weights = {}
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                f_stats = json.load(f).get("factors", {})
                for tag, data in f_stats.items():
                    wr = data.get("win_rate", 0.5)
                    avg_ret = data.get("avg_ret", 0.0)
                    cnt = data.get("count", 0)
                    
                    MIN_SAMPLES = 5.0        
                    PRIOR_WIN_RATE = 0.52    
                    PRIOR_AVG_RET = 0.0      
                    
                    alpha = min(1.0, cnt / MIN_SAMPLES)
                    shrunk_wr = alpha * wr + (1.0 - alpha) * PRIOR_WIN_RATE
                    shrunk_avg_ret = alpha * avg_ret + (1.0 - alpha) * PRIOR_AVG_RET
                    
                    w = 1.0 + (shrunk_wr - 0.5) * 2.0 + (shrunk_avg_ret * 20) 
                    factor_weights[tag] = max(0.2, min(2.0, w)) 
    except Exception as e:
        logger.warning(f"动态权重解析跳过: {e}")

    def get_fw(tag_name: str) -> float:
        return factor_weights.get(f"[{tag_name}]", 1.0)

    reports = []
    for sym in get_filtered_watchlist():
        try:
            df = safe_get_history(sym, "1y", "1d", fast_mode=True)
            if len(df) < 150: continue
            df = calculate_indicators(df)
            curr, prev = df.iloc[-1], df.iloc[-2]
            if curr['Event_Risk'] > 0.85 or (curr['ATR'] / curr['Close'] > 0.10): continue
            
            if pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']: 
                continue
            
            sig, factors, st_cnt = [], [], 0
            
            is_vol = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open'])
            is_st = curr['SuperTrend_Up'] == 1
            
            triggered = []
            cloud_penalty = False

            if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200']:
                fw = get_fw("米奈尔维尼")
                triggered.append(("trend_align", "米奈尔维尼", f"🏆 [米奈尔维尼] 主升形态 (权:{fw:.1f}x)", 8 * w_mul * fw, True))
                
            if len(df) >= 40 and df['Close'].iloc[-20:].min() < df['Close'].iloc[-40:-20].min() and df['OBV'].iloc[-20:].min() > df['OBV'].iloc[-40:-20].min():
                fw = get_fw("OBV底背离")
                triggered.append(("flow", "OBV底背离", f"🌊 [OBV底背离] 资金潜伏 (权:{fw:.1f}x)", 7 * w_mul * fw, True))
                
            if not qqq_df.empty:
                m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
                if len(m_df) >= 20:
                    rs_20 = (m_df['Close_x'].iloc[-1]/m_df['Close_x'].iloc[-20]) / max(m_df['Close_y'].iloc[-1]/m_df['Close_y'].iloc[-20], 0.5)
                    if rs_20 > 1.08: 
                        fw = get_fw("强相对强度")
                        triggered.append(("flow", "强相对强度", f"⚡ [强相对强度] 跑赢大盘 (权:{fw:.1f}x)", (7 if is_vol else 4) * w_mul * fw, False))
                        
            dyn_rsi = curr['RSI_P20']
            if curr['RSI'] < dyn_rsi and prev['RSI'] >= dyn_rsi and is_st:
                fw = get_fw("强势回踩")
                triggered.append(("reversal", "强势回踩", f"🟢 [强势回踩] RSI:{curr['RSI']:.1f} (权:{fw:.1f}x)", 10 * w_mul * fw, True))
                
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                fw = get_fw("MACD金叉")
                triggered.append(("trend_confirm", "MACD金叉", f"🔥 [MACD金叉] (权:{fw:.1f}x)", 10 * w_mul * fw, True))
                
            is_sqz = (curr['BB_Upper'] - curr['BB_Lower']) < (curr['KC_Upper'] - curr['KC_Lower'])
            if is_sqz: 
                fw = get_fw("TTM Squeeze ON")
                triggered.append(("squeeze", "TTM Squeeze ON", f"📦 [TTM Squeeze ON] (权:{fw:.1f}x)", 6 * w_mul * fw, False))
                
            cmf = curr['CMF']
            if cmf > 0.20: 
                fw = get_fw("机构控盘")
                triggered.append(("flow", "机构控盘", f"🏦 [机构控盘] CMF:{cmf:.2f} (权:{fw:.1f}x)", 5 * w_mul * fw, True))
                
            if curr['Above_Cloud'] == 0: 
                cloud_penalty = True
            elif curr['Tenkan'] > curr['Kijun'] and curr['Cloud_Twist'] == 1:
                fw = get_fw("一目多头")
                triggered.append(("trend_confirm", "一目多头", f"🌥️ [一目多头] (权:{fw:.1f}x)", 6 * w_mul * fw, True))
                
            gap_pct = (curr['Open'] - prev['Close']) / prev['Close']
            if 0.015 < gap_pct < 0.06 and is_vol:
                fw = get_fw("突破缺口")
                triggered.append(("reversal", "突破缺口", f"💥 [突破缺口] +{gap_pct*100:.1f}% (权:{fw:.1f}x)", 6 * w_mul * fw, True))
            
            cat_scores = defaultdict(float)
            cat_counts = defaultdict(int)
            
            for cat, tag, text, pts, is_s in triggered:
                cat_scores[cat] += pts
                cat_counts[cat] += 1
                sig.append(text)
                factors.append(tag)
                if is_s: st_cnt += 1
                
            raw_score = 0.0
            for cat, pts in cat_scores.items():
                k = cat_counts[cat]
                discount = max(0.65, 1.0 / (1 + 0.2 * (k - 1))) if k >= 2 else 1.0
                raw_score += pts * discount
                
            if cloud_penalty:
                raw_score *= 0.4
                sig.append("☁️ [云下压制]")
                
            score = int(raw_score)

            if score > 0 and st_cnt < 1 and not is_vol: score = int(score * 0.3)
            if st_cnt >= 2: score += 5

            position_advice = "✅ 标准仓位"
            if gap_pct > 0.03:
                score = int(score * 0.7)
                position_advice = "⚠️ 半仓观察 (今日跳空偏高，提防回落)"

            if sig and score >= min_score:
                if check_earnings_risk(sym):
                    sig.append("💣 [财报雷区] 近5日发财报,风险极高")
                    score = int(score * 0.5)
                    position_advice = "⚠️ 极小仓位 (财报赌博风险)"

                if score >= min_score:
                    news = get_latest_news(sym)
                    reports.append({
                        "symbol": sym, "score": score, "signals": sig[:8], "factors": factors,
                        "curr_close": curr['Close'], 
                        "tp": curr['Close'] + 3.0 * vix_scalar * curr['ATR'], 
                        "sl": curr['Close'] - 1.5 * vix_scalar * curr['ATR'], 
                        "news": news,
                        "sector": Config.get_sector_etf(sym), "raw_score": score,
                        "pos_advice": position_advice
                    })
        except Exception: pass

    if reports:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
        for sec, stks in groups.items():
            if sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.CROWDING_MIN_STOCKS:
                stks.sort(key=lambda x: x["score"], reverse=True)
                sec_mom = sector_data.get(sec, 0.0)
                
                if sec_mom > 0.04:
                    for s in stks[1:]: 
                        if "🚀 [板块主升豁免] 让利润奔跑" not in s["signals"]:
                            s["signals"].append("🚀 [板块主升豁免] 让利润奔跑")
                else:
                    base_pen = max(0.6, min(0.9, Config.CROWDING_PENALTY * (1.0 + health_score * 0.3)))
                    top_score = stks[0]["raw_score"]
                    for s in stks[1:]:
                        if top_score - s["raw_score"] <= 5:
                            s["score"] = int(s["raw_score"] * min(0.95, base_pen + 0.15))
                        else:
                            s["score"] = int(s["raw_score"] * base_pen)

        reports.sort(key=lambda x: x["score"], reverse=True)
        medals = ['🥇', '🥈', '🥉']
        txts = []
        for idx, r in enumerate(reports[:15]):
            icon = medals[idx] if idx < 3 else '🔸'
            sigs_fmt = "\n".join([f"- {s}" for s in r["signals"]])
            news_fmt = f"\n- 📰 {r['news']}" if r['news'] else ""
            
            card = (
                f"### {icon} **{r['symbol']}** | 🌟 评分: {r['score']}\n"
                f"**💡 触发共振:**\n{sigs_fmt}{news_fmt}\n\n"
                f"**💰 交易计划:**\n"
                f"- 💵 现价: `{r['curr_close']:.2f}`\n"
                f"- ⚖️ 建议仓位: **{r.get('pos_advice', '✅ 标准仓位')}**\n"
                f"- 🎯 建议止盈: **${r['tp']:.2f}**\n"
                f"- 🛡️ 初始止损: **${r['sl']:.2f}**\n"
                f"- 📈 移动防守: **新高后按 Max(最高价 - {1.5 * vix_scalar:.1f}*ATR) 追踪**\n"
                f"- ⚠️ 缺口策略: **若次日跳空 > 3%, 建议缩减半仓或等30分钟回踩**"
            )
            txts.append(card)

        perf = load_strategy_performance_tag()
        header = f"**📊 环境感知:**\n- {vix_desc}\n- {regime_desc}"
        
        final_content = (f"{perf}\n\n{header}\n\n---\n\n" if perf else f"{header}\n\n---\n\n") + \
                        "\n\n---\n\n".join(txts) + \
                        f"\n\n*(门槛: {min_score}分 | 正交降权与 EV风控已激活)*"
        
        send_alert("多因子优选 (矩阵版)", final_content)
        
        with open(Config.get_current_log_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), "top_picks": [{"symbol": r["symbol"], "score": r["score"], "signals": r["signals"], "factors": r.get("factors", [])} for r in reports[:15]]}, ensure_ascii=False) + "\n")
    else:
        logger.info("📭 本次矩阵扫描未发现符合硬性门槛的强共振标的，不发送推送。")

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
                        trades.extend([{'date': log['date'], 'symbol': p['symbol'], 'signals': p.get('signals', []), 'factors': p.get('factors', [])} for p in log.get('top_picks', [])])
                    except: pass
        except Exception as e:
            logger.debug(f"读取日志分片 {lf} 失败: {e}")
            
    if not trades: return
    syms = list(set([t['symbol'] for t in trades]))
    
    try:
        df_all = yf.download(syms, period="2mo", progress=False, threads=2)
        if isinstance(df_all.columns, pd.MultiIndex):
            df_c = df_all['Close']
            df_o = df_all['Open']
        else:
            df_c = df_all[['Close']].rename(columns={'Close': syms[0]})
            df_o = df_all[['Open']].rename(columns={'Open': syms[0]})
            
        df_c.index = df_c.index.strftime('%Y-%m-%d')
        df_o.index = df_o.index.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"回测拉取数据失败: {e}")
        return
    
    SLIPPAGE = 0.003    
    COMMISSION = 0.0005 
    
    stats, factor_rets = {'T+1': [], 'T+3': [], 'T+5': []}, {}
    for t in trades:
        sym, r_dt = t['symbol'], t['date']
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
        
        if is_half_pos:
            entry_cost = e_px_raw * (1 + SLIPPAGE * 3 + COMMISSION) 
        else:
            entry_cost = e_px_raw * (1 + SLIPPAGE + COMMISSION)
        
        for d in [1, 3, 5]:
            exit_idx = e_idx + d
            if exit_idx < len(df_c):
                x_px_raw = df_c.iloc[exit_idx][sym] 
                if not np.isnan(x_px_raw):
                    
                    prev_x_px = df_c.iloc[exit_idx-1][sym] if exit_idx > 0 else e_px_raw
                    daily_volatility = abs((x_px_raw - prev_x_px) / prev_x_px)
                    curr_exit_slippage = SLIPPAGE * 5 if daily_volatility > 0.15 else SLIPPAGE
                    
                    exit_revenue = x_px_raw * (1 - curr_exit_slippage - COMMISSION)
                    ret = (exit_revenue - entry_cost) / entry_cost
                    
                    if is_half_pos:
                        ret = ret * 0.5
                        
                    stats[f'T+{d}'].append(ret)
                    
                    if d == 3:
                        factor_list = t.get('factors', [])
                        if not factor_list:
                            for sig_txt in t.get('signals', []):
                                m = re.search(r'\[(.*?)\]', sig_txt)
                                if m: factor_list.append(m.group(1).split(" ")[0])
                                
                        for f_name in factor_list:
                            factor_rets.setdefault(f"[{f_name}]", []).append(ret)
    
    res = {}
    for p, r in stats.items():
        if not r: continue
        ret_arr = np.array(r)
        wr = len(ret_arr[ret_arr > 0]) / len(ret_arr)
        avg_r = np.mean(ret_arr)
        std_r = np.std(ret_arr) if len(ret_arr) > 1 else 1e-6
        sharpe = avg_r / (std_r + 1e-10)
        worst = np.min(ret_arr)
        res[p] = {'win_rate': wr, 'avg_ret': avg_r, 'sharpe': sharpe, 'worst_trade': worst, 'total_trades': len(r)}

    f_res = {t: {'win_rate': sum(1 for x in r if x > 0)/len(r), 'avg_ret': sum(r)/len(r), 'count': len(r)} for t, r in factor_rets.items() if len(r) >= 2}
    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump({"overall": res, "factors": f_res}, f, indent=4)
    
    report_md = [f"# 📈 自动量化战报\n**更新:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n| 周期 | 胜率 | 均收益 | Sharpe | 单笔极亏 | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for p in ['T+1', 'T+3', 'T+5']:
        d = res.get(p, {'win_rate':0,'avg_ret':0,'sharpe':0,'worst_trade':0,'total_trades':0})
        report_md.append(f"| {p} | {d['win_rate']*100:.1f}% | {d['avg_ret']*100:+.2f}% | {d['sharpe']:.2f} | {d['worst_trade']*100:.1f}% | {d['total_trades']} |")
    
    if f_res:
        report_md.append("\n## 🧬 因子排行 (T+3)\n| 因子 | 胜率 | 次数 |\n|:---|:---:|:---:|")
        sorted_f = sorted(f_res.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        for tag, d in sorted_f: report_md.append(f"| {tag} | {d['win_rate']*100:.1f}% | {d['count']} |")
    
    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_md))

    alert_lines = ["### 📊 **周期胜率总览 (含严格摩擦与踩踏滑点)**"]
    for p, d in res.items(): alert_lines.append(f"- **{p}:** 胜率 {d['win_rate']*100:.1f}% | 均收益 {d['avg_ret']*100:+.2%} | Sharpe {d['sharpe']:.2f} | 极回撤 {d['worst_trade']*100:.1f}%")
    
    if f_res:
        alert_lines.extend(["", "---", "", "### 🧬 **核心因子排行 (T+3)**"])
        for idx, (tag, d) in enumerate(sorted_f[:3]):
            icon = ['1️⃣','2️⃣','3️⃣'][idx]
            alert_lines.append(f"- {icon} **{tag}**: 胜率 {d['win_rate']*100:.1f}%")
            
    send_alert("策略终极回测战报", "\n".join(alert_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    if m == "sentinel": run_volatility_sentinel()
    elif m == "matrix": run_tech_matrix()
    elif m == "backtest": run_backtest_engine()
    elif m == "test": send_alert("连通性测试", "系统已完成终极底层优化！正交拥挤度折扣 (Orthogonal Discount) 已上线。")
