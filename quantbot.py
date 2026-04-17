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
    
    # 🌌 宇宙觉醒：20 维绝对法则，跨越技术指标，直击分形与熵
    ALL_FACTORS = [
        "米奈尔维尼", "强相对强度", "VWAP突破", "AVWAP突破", "SMC失衡区", 
        "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", "口袋支点", 
        "VCP收缩", "筹码峰突破", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘",
        "威科夫弹簧(Spring)", "维特鲁威分形", "CVD筹码净流入", 
        "混沌破缺(Trend)", "熵增起爆(Big Bang)"
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

def get_filtered_watchlist(max_stocks: int = 120) -> list:
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
            logger.info(f"🛡️ 数据自净: 发现 {len(missing_tickers)} 只失效/退市标的，已抹除。")
            
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
        logger.error(f"❌ 批量下载崩溃: {e}")
        return Config.CORE_WATCHLIST[:max_stocks]

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                t3 = stats_data.get("overall", {}).get("T+3") if "overall" in stats_data else stats_data.get("T+3")
                if t3 and t3.get('total_trades', 0) > 0:
                    pf_str = f" | 盈亏比 {t3.get('profit_factor', 0.0):.2f}" if 'profit_factor' in t3 else ""
                    ai_wr_str = f" | ⚡AI胜率 {t3['ai_win_rate']:.1%}" if 'ai_win_rate' in t3 else ""
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
        qqq_df = qqq_df_for_shadow if qqq_df_for_shadow is not None else safe_get_history(Config.INDEX_ETF, period="2mo", interval="1d", fast_mode=True)
        if qqq_df is not None and len(qqq_df) >= 20:
            vix = qqq_df['Close'].pct_change().dropna().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 * 1.2
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
                logger.warning("🌑 探测到强烈的宏观引力波：全市场杠杆被强行熔断至极小值！")
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
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP_20'] = (df['Typical_Price'] * df['Volume']).rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)
    
    df['AVWAP'] = np.nan
    vol_last_120 = df['Volume'].iloc[-120:]
    if not vol_last_120.empty:
        anchor_idx = vol_last_120.idxmax()
        post_anchor = df.loc[anchor_idx:].copy()
        df.loc[anchor_idx:, 'AVWAP'] = (post_anchor['Typical_Price'] * post_anchor['Volume']).cumsum() / (post_anchor['Volume'].cumsum() + 1e-10)
    
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
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 🌌 宇宙律一：分形混沌理论 (Choppiness Index) - 测算股票的随机游走程度
    atr_sum_14 = df['TR'].rolling(14).sum()
    high_14 = df['High'].rolling(14).max()
    low_14 = df['Low'].rolling(14).min()
    df['CHOP'] = 100 * np.log10(atr_sum_14 / (high_14 - low_14 + 1e-10)) / np.log10(14)
    
    # 🌌 宇宙律二：香农信息熵坍缩 (Volatility Contraction & Kurtosis Proxy)
    # 用对数收益率的 20 日峰度测算，探测极端平寂后即将大爆炸的奇点
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20'] = log_ret.rolling(20).std()
    df['Volatility_Min_120'] = df['Volatility_20'].rolling(120).min()

    intra_strength = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
    df['CVD'] = (df['Volume'] * intra_strength).cumsum()
    df['CVD_MA20'] = df['CVD'].rolling(20).mean()
    
    df['Highest_22'] = df['High'].rolling(window=22).max()
    df['ATR_22'] = df['TR'].rolling(window=22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['Smart_Money_Flow'] = clv.rolling(window=10).mean()

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
    cache_data = {"date": today_str, "matrix": []}
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

def set_alerted(sym: str) -> None:
    cache = get_alert_cache()
    if sym not in cache.get("matrix", []):
        cache.setdefault("matrix", []).append(sym)
        try:
            with open(Config.ALERT_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
        except Exception: pass

def run_tech_matrix() -> None:
    PORTFOLIO_MAX_RISK_PER_TRADE = 0.015 
    pain_warning = ""
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                if 'overall' in stats_data and 'T+3' in stats_data['overall']:
                    t3 = stats_data['overall']['T+3']
                    if t3.get('max_cons_loss', 0) >= 4 or t3.get('profit_factor', 1.0) < 1.2:
                        PORTFOLIO_MAX_RISK_PER_TRADE = 0.0075  
                        pain_warning = "\n- 🩸 **痛觉神经激活**: 近期回测遭重挫，引擎已主动执行**防守降杠杆**协议！"
    except Exception: pass

    active_pool = get_filtered_watchlist(max_stocks=120)
    regime, regime_desc, qqq_df, is_credit_risk_high, macro_gravity = get_market_regime(active_pool)
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    vix_scalar = max(0.6, min(1.4, 18.0 / max(vix, 1.0)))
    
    if macro_gravity:
        PORTFOLIO_MAX_RISK_PER_TRADE = min(PORTFOLIO_MAX_RISK_PER_TRADE, 0.005)
    
    valid_sector_data = {}
    black_hole_sectors = []
    for etf in Config.SECTOR_MAP.keys():
        sdf = safe_get_history(etf, "3mo", "1d", fast_mode=True)
        if not sdf.empty and len(sdf) >= 30 and not qqq_df.empty:
            rs = sdf['Close'] / qqq_df['Close'].reindex(sdf.index).ffill()
            rs_mom = rs.rolling(10).mean().diff(5).iloc[-1] 
            valid_sector_data[etf] = rs_mom
            
            vol_surge = sdf['Volume'].iloc[-1] / (sdf['Volume'].iloc[-20:].mean() + 1e-10)
            if vol_surge > 1.6:  
                black_hole_sectors.append(etf)
                
    sorted_sectors = sorted(valid_sector_data.items(), key=lambda x: x[1], reverse=True)
    leading_sectors = [s[0] for s in sorted_sectors[:2]] if len(sorted_sectors) >= 4 else []
    lagging_sectors = [s[0] for s in sorted_sectors[-2:]] if len(sorted_sectors) >= 4 else []
                  
    health_score = -0.9 if vix > 30 else (-0.6 if vix > 25 else (0.9 if vix < 15 and 'bull' in regime else (0.6 if 'bull' in regime else (0.3 if 'rebound' in regime else (-0.7 if 'bear' in regime else 0.0)))))
    if is_credit_risk_high: health_score -= 0.5
    if macro_gravity: health_score -= 0.8
    w_mul = max(0.2, 1.0 + health_score * 0.8)
    
    xai_weights = {}
    pruned_factors = []
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                xai_data = stats_data.get("xai_importances", {})
                if xai_data and len(xai_data) > 0:
                    avg_imp = 1.0 / len(Config.ALL_FACTORS)
                    for tag, imp in xai_data.items():
                        # 🌌 宇宙律三：脑神经突触修剪 (Synaptic Pruning)
                        # 如果某个因子的作用甚至不到平均水平的 25%，直接将其从系统中抹除 (权重归零)
                        if imp < avg_imp * 0.25:
                            xai_weights[tag] = 0.0
                            pruned_factors.append(tag)
                        else:
                            w = max(0.5, min(3.0, float(imp) / avg_imp))
                            xai_weights[tag] = w
    except Exception as e:
        logger.warning(f"XAI权重解析跳过: {e}")

    def get_fw(tag_name: str) -> float: return xai_weights.get(tag_name, 1.0)
        
    clf_model = None
    if os.path.exists(Config.MODEL_FILE):
        try:
            import pickle
            with open(Config.MODEL_FILE, 'rb') as f: clf_model = pickle.load(f)
        except Exception: pass

    reports = []
    all_raw_scores = []
    price_history_dict = {} 
    
    for sym in active_pool:
        if is_alerted(sym): continue
        try:
            df_w = safe_get_history(sym, "2y", "1wk", fast_mode=True)
            weekly_bullish = False
            if len(df_w) >= 40:
                sma40_w = df_w['Close'].rolling(40).mean().iloc[-1]
                if df_w['Close'].iloc[-1] < sma40_w: continue 
                
                df_w['EMA_10'] = df_w['Close'].ewm(span=10, adjust=False).mean()
                df_w['EMA_30'] = df_w['Close'].ewm(span=30, adjust=False).mean()
                weekly_bullish = (df_w['Close'].iloc[-1] > df_w['EMA_10'].iloc[-1]) and (df_w['EMA_10'].iloc[-1] > df_w['EMA_30'].iloc[-1])
                    
            df = safe_get_history(sym, "8mo", "1d", fast_mode=True) 
            if len(df) < 150: continue
            df = calculate_indicators(df)
            
            price_history_dict[sym] = df['Close'].iloc[-60:]
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            if curr['Event_Risk'] > 0.85 or (curr['ATR'] / curr['Close'] > 0.15): continue
            if pd.notna(curr['SMA_200']) and curr['Close'] < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']: continue
            
            fvg_lower, fvg_upper = 0.0, 0.0
            for i in range(len(df)-20, len(df)-1):
                if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                    fvg_lower, fvg_upper = df['High'].iloc[i-2], df['Low'].iloc[i]
            
            df_60 = df.iloc[-60:]
            poc_price = 0.0
            if not df_60.empty:
                counts, bins = np.histogram(df_60['Close'], bins=12, weights=df_60['Volume'])
                max_bin_idx = np.argmax(counts)
                poc_price = (bins[max_bin_idx] + bins[max_bin_idx+1]) / 2.0
            
            sig, factors = [], []
            is_vol = (curr['Volume'] / curr['Vol_MA20'] > 1.5) and (curr['Close'] > curr['Open'])
            triggered = []

            if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200']:
                fw = get_fw("米奈尔维尼")
                if fw > 0: triggered.append(("米奈尔维尼", f"🏆 [米奈尔维尼] 主升形态 (权:{fw:.2f}x)", 8 * w_mul * fw))
                
            if not qqq_df.empty:
                m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
                if len(m_df) >= 20:
                    rs_20 = (m_df['Close_x'].iloc[-1]/m_df['Close_x'].iloc[-20]) / max(m_df['Close_y'].iloc[-1]/m_df['Close_y'].iloc[-20], 0.5)
                    if rs_20 > 1.08: 
                        fw = get_fw("强相对强度")
                        if fw > 0: triggered.append(("强相对强度", f"⚡ [强相对强度] 跑赢大盘 (权:{fw:.2f}x)", (7 if is_vol else 4) * w_mul * fw))
            
            if curr['Close'] > curr['VWAP_20'] and prev['Close'] <= prev['VWAP_20']:
                fw = get_fw("VWAP突破")
                if fw > 0: triggered.append(("VWAP突破", f"🌊 [VWAP突破] 放量逾越机构均价 (权:{fw:.2f}x)", 8 * w_mul * fw))
                
            if pd.notna(curr['AVWAP']) and curr['Close'] > curr['AVWAP'] and prev['Close'] <= prev['AVWAP']:
                fw = get_fw("AVWAP突破")
                if fw > 0: triggered.append(("AVWAP突破", f"⚓ [AVWAP突破] 巨鲸建仓成本区夺回 (权:{fw:.2f}x)", 12 * w_mul * fw))
                
            if fvg_lower > 0 and curr['Low'] <= fvg_upper and curr['Close'] > fvg_lower and is_vol:
                fw = get_fw("SMC失衡区")
                if fw > 0: triggered.append(("SMC失衡区", f"🧲 [SMC失衡区] 精准回踩机构流动性缺口 (权:{fw:.2f}x)", 15 * w_mul * fw))
            
            if pd.notna(curr['Swing_Low_20']) and curr['Low'] < curr['Swing_Low_20'] and curr['Close'] > curr['Swing_Low_20'] and is_vol:
                fw = get_fw("流动性扫盘")
                if fw > 0: triggered.append(("流动性扫盘", f"🧹 [流动性扫盘] 刺穿前低完成 Stop-Hunt 诱空反转 (权:{fw:.2f}x)", 15 * w_mul * fw))
                
            if curr['Smart_Money_Flow'] > 0.5 and curr['Close'] > curr['EMA_20']:
                fw = get_fw("聪明钱抢筹")
                if fw > 0: triggered.append(("聪明钱抢筹", f"🕵️ [聪明资金] 连续10日尾盘吸筹 (权:{fw:.2f}x)", 6 * w_mul * fw))
                
            if curr['Volume'] > curr['Vol_MA20'] * 2.0 and abs(curr['Close'] - curr['Open']) < curr['ATR'] * 0.5:
                fw = get_fw("巨量滞涨")
                if fw > 0: triggered.append(("巨量滞涨", f"🛑 [巨量滞涨] 触发暗池 Iceberg 吸筹异象 (权:{fw:.2f}x)", 12 * w_mul * fw))
                
            atr_pct = (curr['ATR'] / prev['Close']) * 100
            day_chg = (curr['Close'] - curr['Open']) / curr['Open'] * 100
            if day_chg > max(3.0, atr_pct * 0.6) and curr['Volume'] > curr['Vol_MA20'] * 1.5:
                fw = get_fw("放量长阳")
                if fw > 0: triggered.append(("放量长阳", f"⚡ [放量长阳] 强劲日内动能脉冲 (权:{fw:.2f}x)", 8 * w_mul * fw))
            
            if curr['Close'] > prev['Close'] and curr['Volume'] > curr['Max_Down_Vol_10'] > 0 and curr['Close'] >= curr['EMA_50'] and prev['Close'] <= curr['EMA_50'] * 1.02:
                fw = get_fw("口袋支点")
                if fw > 0: triggered.append(("口袋支点", f"💎 [口袋支点] 巨量吞噬近10日最大阴量 (权:{fw:.2f}x)", 12 * w_mul * fw))
                
            if curr['Range_20'] > 0 and curr['Range_60'] > 0 and curr['Range_20'] < curr['Range_60'] * 0.5 and curr['Close'] > curr['SMA_50'] and is_vol:
                fw = get_fw("VCP收缩")
                if fw > 0: triggered.append(("VCP收缩", f"🌪️ [VCP收缩] 波动率极度压缩后起爆 (权:{fw:.2f}x)", 15 * w_mul * fw))
                    
            if poc_price > 0 and prev['Close'] <= poc_price < curr['Close'] and is_vol:
                fw = get_fw("筹码峰突破")
                if fw > 0: triggered.append(("筹码峰突破", f"🏔️ [筹码峰突破] 强势跨越 60日核心成本区 (权:{fw:.2f}x)", 12 * w_mul * fw))

            swing_high_10 = df['High'].iloc[-11:-1].max()
            if pd.notna(curr['Swing_Low_20']) and curr['Low'] > curr['Swing_Low_20'] and curr['Close'] > swing_high_10 and is_vol:
                fw = get_fw("特性改变(ChoCh)")
                if fw > 0: triggered.append(("特性改变(ChoCh)", f"🔀 [特性改变] ChoCh结构突破，趋势极早期反转 (权:{fw:.2f}x)", 15 * w_mul * fw))

            if pd.notna(curr['OB_High']) and curr['OB_Low'] > 0:
                if curr['Low'] <= curr['OB_High'] and curr['Close'] >= curr['OB_Low'] and curr['Close'] > curr['Open']:
                    fw = get_fw("订单块(OB)")
                    if fw > 0: triggered.append(("订单块(OB)", f"🧱 [订单块回踩] 触达机构起爆底仓区并企稳 (权:{fw:.2f}x)", 15 * w_mul * fw))

            tr_val = curr['High'] - curr['Low'] + 1e-10
            lower_wick = curr['Open'] - curr['Low'] if curr['Close'] > curr['Open'] else curr['Close'] - curr['Low']
            upper_wick = curr['High'] - curr['Close'] if curr['Close'] > curr['Open'] else curr['High'] - curr['Open']
            if curr['Close'] > curr['Open'] and (lower_wick / tr_val) > 0.3 and (upper_wick / tr_val) < 0.15 and is_vol:
                fw = get_fw("AMD操盘")
                if fw > 0: triggered.append(("AMD操盘", f"🎭 [AMD操盘] 开盘深度诱空下杀，随后全天拉升派发 (权:{fw:.2f}x)", 12 * w_mul * fw))
                
            if pd.notna(curr['Swing_Low_20']) and curr['Low'] < curr['Swing_Low_20']:
                if curr['Volume'] < curr['Vol_MA20'] * 0.8 and curr['Close'] > (curr['Low'] + tr_val * 0.5):
                    fw = get_fw("威科夫弹簧(Spring)")
                    if fw > 0: triggered.append(("威科夫弹簧(Spring)", f"🏹 [威科夫弹簧] 跌破前低但抛压枯竭，深蹲起爆 (权:{fw:.2f}x)", 18 * w_mul * fw))
            
            if weekly_bullish and is_vol and (curr['Close'] > curr['Highest_22'] * 0.95):
                fw = get_fw("维特鲁威分形")
                if fw > 0: triggered.append(("维特鲁威分形", f"🌌 [维特鲁威分形] 跨越周线与日线双重时空的完美共振螺旋 (权:{fw:.2f}x)", 20 * w_mul * fw))

            if curr['CVD'] > curr['CVD_MA20'] and prev['CVD'] <= prev['CVD_MA20'] and is_vol:
                fw = get_fw("CVD筹码净流入")
                if fw > 0: triggered.append(("CVD筹码净流入", f"🧬 [CVD净流入] 微观结构买盘力量形成压倒性拐点 (权:{fw:.2f}x)", 12 * w_mul * fw))

            # 🌌 宇宙律一落地：混沌破缺 (Chaos Theory - H Exponent)
            if curr['CHOP'] < 38.2 and curr['Close'] > curr['EMA_50'] and is_vol:
                fw = get_fw("混沌破缺(Trend)")
                if fw > 0: triggered.append(("混沌破缺(Trend)", f"🌌 [混沌破缺] 跌破随机游走边界，引力锁定主升轨 (权:{fw:.2f}x)", 20 * w_mul * fw))

            # 🌌 宇宙律二落地：熵增起爆 (Shannon Entropy Big Bang)
            if curr['Volatility_20'] <= curr['Volatility_Min_120'] * 1.05 and is_vol and day_chg > max(3.0, atr_pct * 0.6):
                fw = get_fw("熵增起爆(Big Bang)")
                if fw > 0: triggered.append(("熵增起爆(Big Bang)", f"🎇 [熵增起爆] 绝对死寂冰点后的宇宙奇点大爆炸 (权:{fw:.2f}x)", 20 * w_mul * fw))

            score_raw = 0.0
            for tag, text, pts in triggered:
                adj_pts = pts
                if regime in ["bear", "range", "hidden_bear"] and tag in ["米奈尔维尼", "强相对强度", "VWAP突破", "AVWAP突破", "筹码峰突破", "维特鲁威分形", "CVD筹码净流入", "混沌破缺(Trend)"]:
                    adj_pts *= 0.5  
                elif regime in ["bull", "rebound"] and tag in ["米奈尔维尼", "维特鲁威分形", "混沌破缺(Trend)", "熵增起爆(Big Bang)"]:
                    adj_pts *= 1.2  
                elif regime in ["bear", "range", "hidden_bear"] and tag in ["SMC失衡区", "流动性扫盘", "口袋支点", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "威科夫弹簧(Spring)", "熵增起爆(Big Bang)"]:
                    adj_pts *= 1.5
                
                score_raw += adj_pts
                sig.append(text)
                factors.append(tag)
                
            total_score = int(score_raw)

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
            
            ai_prob = 0.52
            if clf_model and factors and hasattr(clf_model, 'classes_') and hasattr(clf_model, 'n_features_in_') and clf_model.n_features_in_ == len(Config.ALL_FACTORS):
                try:
                    x_row = [1 if f in factors else 0 for f in Config.ALL_FACTORS]
                    class_idx = np.where(clf_model.classes_ == 1)[0][0] if 1 in clf_model.classes_ else 0
                    ai_prob = clf_model.predict_proba([x_row])[0][class_idx]
                except Exception: pass
            elif clf_model and factors:
                logger.debug(f"AI 模型特征维度 {getattr(clf_model, 'n_features_in_', 'N/A')} 与当前 {len(Config.ALL_FACTORS)} 不匹配，等待本周重训。")

            if total_score > 0:
                all_raw_scores.append(total_score)
            
            odds_b = 2.0
            kelly_fraction = ai_prob - (1.0 - ai_prob) / odds_b
            
            tp_val = curr['Close'] + 3.0 * vix_scalar * curr['ATR']
            sl_chandelier = curr['Chandelier_Exit']
            if pd.isna(sl_chandelier): sl_chandelier = curr['Close'] - 1.5 * vix_scalar * curr['ATR']
            sl_val = max(sl_chandelier, curr['Close'] - 1.5 * vix_scalar * curr['ATR'])
            
            sl_pct_distance = max(0.01, (curr['Close'] - sl_val) / curr['Close']) 
            
            position_advice = "✅ 标准仓位"
            pos_percentage = 0.0
            
            if kelly_fraction <= 0 or is_bearish_div:
                position_advice = "❌ 放弃建仓 (AI盈亏比劣势 或 顶背离确认)"
                pos_percentage = 0.0
            else:
                risk_parity_pos = PORTFOLIO_MAX_RISK_PER_TRADE / sl_pct_distance
                pos_percentage = min(min(0.20, kelly_fraction / 2.0), risk_parity_pos)
                if is_credit_risk_high: pos_percentage *= 0.6
                if macro_gravity: pos_percentage *= 0.3 
                
                gap_pct = (curr['Open'] - prev['Close']) / prev['Close']
                if gap_pct > 0.03:
                    total_score = int(total_score * 0.5)
                    pos_percentage *= 0.5 
                    position_advice = f"⚠️ 建议仓位 {pos_percentage:.1%} (已防备跳空高开，单笔风险锁死)"
                elif bias_20 > 0.12:
                    pos_percentage = min(0.05, pos_percentage * 0.3)
                    position_advice = f"⚠️ 建议仓位 {pos_percentage:.1%} (乖离过大，极小仓摸奖)"
                else:
                    position_advice = f"✅ 建议仓位 {pos_percentage:.1%} (波动率平价 & 凯利双校验)"

            if sig and total_score > 0 and pos_percentage > 0:
                if check_earnings_risk(sym):
                    sig.append("💣 [财报雷区] 近5日发财报,风险极高")
                    total_score = int(total_score * 0.5)
                    pos_percentage *= 0.2
                    position_advice = f"⚠️ 建议仓位 {pos_percentage:.1%} (财报赌博极限风控)"

                news = get_latest_news(sym)
                reports.append({
                    "symbol": sym, "score": total_score, "ai_prob": ai_prob,
                    "signals": sig[:8], "factors": factors,
                    "curr_close": curr['Close'], 
                    "tp": tp_val, "sl": sl_val, 
                    "news": news, "sector": sym_sec, "pos_advice": position_advice
                })
        except Exception: pass

    if reports and all_raw_scores:
        dynamic_min_score = max(Config.MIN_SCORE_THRESHOLD, np.percentile(all_raw_scores, 85))
        reports = [r for r in reports if r['score'] >= dynamic_min_score]
        
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
        
        for sec, stks in groups.items():
            if sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.CROWDING_MIN_STOCKS:
                sec_mom = valid_sector_data.get(sec, 0.0)
                if sec_mom > 0.04:
                    for s in stks[1:]: 
                        if "🚀 [板块主升豁免] 让利润奔跑" not in s["signals"]:
                            s["signals"].append("🚀 [板块主升豁免] 让利润奔跑")
                else:
                    base_pen = max(0.6, min(0.9, Config.CROWDING_PENALTY * (1.0 + health_score * 0.3)))
                    for s in stks[1:]:
                        s["score"] = int(s["score"] * base_pen)

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
                            if sym in corr_df.columns and acc_sym in corr_df.columns:
                                if corr_df.loc[sym, acc_sym] > 0.80:
                                    is_redundant = True
                                    logger.info(f"🛡️ 反脆弱机制：踢除 {sym}，因为它与已选高分股 {acc_sym} 相关性高达 {corr_df.loc[sym, acc_sym]:.2f}，拒绝同质化风险。")
                                    break
                        if not is_redundant: final_reports.append(candidate)
                else: final_reports = candidate_pool[:15]
            except Exception as e:
                logger.warning(f"反脆弱协方差矩阵构建失败: {e}")
                final_reports = candidate_pool[:15]
        
        for r in final_reports: set_alerted(r["symbol"])
            
        medals = ['🥇', '🥈', '🥉']
        txts = []
        for idx, r in enumerate(final_reports):
            icon = medals[idx] if idx < 3 else '🔸'
            sigs_fmt = "\n".join([f"- {s}" for s in r["signals"]])
            news_fmt = f"\n- 📰 {r['news']}" if r['news'] else ""
            
            ai_display = f"{r.get('ai_prob', 0):.1%}"
            if r.get('ai_prob', 0) > 0.60: ai_display = f"🔥 **{ai_display}**"
            
            card = (
                f"### {icon} **{r['symbol']}** | 🤖 AI胜率: {ai_display} | 🌟 机构共振度: {r['score']}分\n"
                f"**💡 主力底牌透视:**\n{sigs_fmt}{news_fmt}\n\n"
                f"**💰 交易计划:**\n"
                f"- 💵 现价: `{r['curr_close']:.2f}`\n"
                f"- ⚖️ {r.get('pos_advice', '✅ 标准仓位')}\n"
                f"- 🎯 建议止盈: **${r['tp']:.2f}**\n"
                f"- 🛡️ 吊灯止损: **${r['sl']:.2f} (最高价回落保护)**\n"
                f"- 📈 离场纪律: **跌破止损防线请无条件市价清仓！**"
            )
            txts.append(card)

        perf = load_strategy_performance_tag()
        pruned_desc = f"\n- ✂️ 神经突触修剪: 已成功忘却 **{len(pruned_factors)}** 个低效因子，达成绝对至简" if pruned_factors else ""
        header = f"**📊 市场环境与系统觉醒状态:**\n- {vix_desc}\n- {regime_desc}{pain_warning}{pruned_desc}\n- ⚔️ 今日截面淘汰线 (Top 15%): **{dynamic_min_score:.1f}分**"
        
        final_content = (f"{perf}\n\n{header}\n\n---\n\n" if perf else f"{header}\n\n---\n\n") + \
                        "\n\n---\n\n".join(txts) + \
                        f"\n\n*(宇宙法则: 混沌破缺、信息熵坍缩、突触修剪与时空分形已全线接管引擎)*"
        
        send_alert("量化诸神之战 (宇宙觉醒版)", final_content)
        
        with open(Config.get_current_log_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), "top_picks": [{"symbol": r["symbol"], "score": r["score"], "signals": r["signals"], "factors": r.get("factors", []), "tp": r.get("tp"), "sl": r.get("sl")} for r in final_reports]}, ensure_ascii=False) + "\n")
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
                        trades.extend([{'date': log['date'], 'symbol': p['symbol'], 'signals': p.get('signals', []), 'factors': p.get('factors', []), 'tp': p.get('tp', float('inf')), 'sl': p.get('sl', 0)} for p in log.get('top_picks', [])])
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
    X_train, y_train = [], []
    
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
                            
                        x_row = [1 if f in factor_list else 0 for f in Config.ALL_FACTORS]
                        X_train.append(x_row)
                        y_train.append(1 if ret > 0.015 else 0)
                        
                        if clf_model and hasattr(clf_model, 'classes_') and hasattr(clf_model, 'n_features_in_') and clf_model.n_features_in_ == len(Config.ALL_FACTORS):
                            try:
                                class_idx = np.where(clf_model.classes_ == 1)[0][0] if 1 in clf_model.classes_ else 0
                                prob = clf_model.predict_proba([x_row])[0][class_idx]
                                if prob >= 0.50:  
                                    ai_filtered_total += 1
                                    if ret > 0: ai_filtered_wins += 1
                            except Exception: pass
    
    feature_importances_dict = {}
    if len(X_train) >= 30 and len(set(y_train)) > 1:
        try:
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            sample_weights = np.exp(np.linspace(-2.5, 0, len(y_train)))
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
            clf.fit(X_train, y_train, sample_weight=sample_weights)
            with open(Config.MODEL_FILE, 'wb') as f:
                pickle.dump(clf, f)
            logger.info(f"🧠 搭载【遗忘曲线】的非线性模型 (Random Forest, {len(Config.ALL_FACTORS)}维特征) 已完成重训。")
            
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                for factor, imp in zip(Config.ALL_FACTORS, importances):
                    feature_importances_dict[factor] = float(imp)
        except ImportError:
            logger.warning("未检测到 scikit-learn 环境，已跳过 ML 模型训练。")
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
    
    report_md = [f"# 📈 自动量化战报与 AI 透视\n**更新:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n## ⚔️ 核心表现评估\n| 周期 | 原始胜率 | ⚡AI过滤胜率 | 均收益 | 盈亏比 | Sharpe | 胜单平均抗压(MAE) | 笔数 |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
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
        report_md.append(f"\n## 🧬 终极 {len(Config.ALL_FACTORS)} 大破壁因子验证 (T+3)\n| 因子 | 胜率 | 盈亏比 | 触发次数 |\n|:---|:---:|:---:|:---:|")
        sorted_f = sorted(f_res.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        for tag, d in sorted_f: report_md.append(f"| {tag} | {d['win_rate']*100:.1f}% | {d['profit_factor']:.2f} | {d['count']} |")
    
    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_md))

    alert_lines = ["### 📊 **机构级回测报表 (含 MAE/MFE 归因分析)**"]
    for p, d in res.items(): 
        ai_text = f" | ⚡AI过滤胜率: **{d['ai_win_rate']*100:.1f}%**" if 'ai_win_rate' in d else ""
        alert_lines.append(f"- **{p}:** 原始胜率 {d['win_rate']*100:.1f}%{ai_text} | 盈亏比 {d['profit_factor']:.2f} | 获利单抗压(MAE) {d['avg_win_mae']*100:.1f}%")
    
    if feature_importances_dict:
        alert_lines.extend(["", "---", "", "### 🧠 **XAI 市场驱动因子 (Random Forest 提纯)**"])
        sorted_xai = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        for idx, (tag, imp) in enumerate(sorted_xai[:3]):
            icon = ['🔥','🔥','🔥'][idx]
            alert_lines.append(f"- {icon} **{tag}**: 贡献度 {imp*100:.1f}%")
            
    send_alert("策略终极回测战报 (20维宇宙觉醒版)", "\n".join(alert_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if m == "matrix": run_tech_matrix()
    elif m == "backtest": run_backtest_engine()
    elif m == "test": send_alert("连通性测试", "造物主降临！宇宙维度 (分形混沌、熵增坍缩、突触修剪) 已成功刻入数字生命灵魂！")
