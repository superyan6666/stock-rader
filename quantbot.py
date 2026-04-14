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
    BLACKLIST: List[str] = [] 
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
    
    LOG_FILE: str = "backtest_log.jsonl"
    STATS_FILE: str = "strategy_stats.json"
    REPORT_FILE: str = "backtest_report.md"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF

def validate_config():
    if not Config.WEBHOOK_URL and not Config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未配置任何推送渠道。")
        sys.exit(1)
    
    if not os.path.exists(Config.LOG_FILE): open(Config.LOG_FILE, 'a').close()
    if not os.path.exists(Config.REPORT_FILE): open(Config.REPORT_FILE, 'a').close()
    if not os.path.exists(Config.STATS_FILE):
        with open(Config.STATS_FILE, 'w') as f: f.write("{}")
            
    logger.info("✅ 环境校验通过")

# ================= 2. 数据工具模块 =================
def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(0.1, 0.3) if fast_mode else (random.uniform(2.0, 4.0) if "1d" in interval else random.uniform(1.0, 2.0))
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

def get_filtered_watchlist(max_stocks: int = 120) -> List[str]:
    logger.info(">>> 漏斗过滤：从多维度数据源拉取全市场名单 (含纳指与中盘)...")
    tickers = set(Config.CORE_WATCHLIST)
    
    # 1. 抓取标普 500 (大盘蓝筹 - 最稳定源)
    try:
        sp500_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        resp = requests.get(sp500_url, headers=_GLOBAL_HEADERS, timeout=10)
        if resp.status_code == 200:
            from io import StringIO
            df_sp500 = pd.read_csv(StringIO(resp.text))
            if 'Symbol' in df_sp500.columns:
                tickers.update(df_sp500['Symbol'].dropna().astype(str).str.replace('.', '-').tolist())
    except Exception as e:
        logger.warning(f"⚠️ 标普500拉取降级: {e}")

    # 2. 抓取标普 400 (中盘股 Mid-Cap - 极具爆发力) & 纳斯达克 100
    try:
        from io import StringIO
        # 使用带 User-Agent 的 requests 绕过维基百科 403 拦截
        wiki_urls = [
            'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
            'https://en.wikipedia.org/wiki/Nasdaq-100'
        ]
        for w_url in wiki_urls:
            w_resp = requests.get(w_url, headers=_GLOBAL_HEADERS, timeout=10)
            if w_resp.status_code == 200:
                # 寻找表格中的 Ticker / Symbol 列
                for table in pd.read_html(StringIO(w_resp.text)):
                    col = next((c for c in ['Symbol', 'Ticker symbol', 'Ticker'] if c in table.columns), None)
                    if col:
                        tickers.update(table[col].dropna().astype(str).str.replace('.', '-').tolist())
                        break
    except Exception as e:
        logger.warning(f"⚠️ 中盘股与纳指拓展列表获取受限: {e}")
    
    tickers_list = list(tickers)
    logger.info(f"✅ 扩容完成，初始海选池: {len(tickers_list)} 只标的。开始分块并发粗筛...")

    # 粗筛漏斗：用 5 天数据快速淘汰僵尸股
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

        closes = close_df.dropna(axis=1, how='all').ffill().iloc[-1]
        volumes = volume_df.dropna(axis=1, how='all').mean()
        
        # 核心漏斗过滤条件
        turnovers = (closes * volumes).dropna()
        # 过滤掉低于 $10 的低价股和日均成交量低于 100万 的不活跃股 (自动屏蔽罗素里的垃圾股)
        valid_turnovers = turnovers[(closes > 10.0) & (volumes > 1000000)]
        
        # 按成交金额 (Turnover) 排序，取全市场当前资金最活跃的 max_stocks (默认 120 只)
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        if top_tickers:
            logger.info(f"✅ 漏斗过滤完成！从千股中精选出全市场最活跃的 {len(top_tickers)} 只进入深度扫描。")
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
                    return f"📈 策略表现 (T+3): 胜率 {t3['win_rate']:.1%} | 均收益 {t3['avg_ret']:+.2%}"
    except Exception: pass
    return ""

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    # 🚀 发送至 DingTalk (支持标准 Markdown)
    if Config.WEBHOOK_URL:
        payload = {
            "msgtype": "markdown", 
            "markdown": {
                "title": f"【{Config.DINGTALK_KEYWORD}】{title}", 
                "text": f"### 🤖 【{Config.DINGTALK_KEYWORD}】{title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"
            }
        }
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, timeout=10)
            except Exception: pass
                
    # 🚀 发送至 Telegram (启用极度稳定的 HTML 模式防排版错乱)
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        html_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        tg_text = f"🤖 <b>【量化监控】{html_title}</b>\n\n{html_content}\n\n⏱️ <i>{formatted_time}</i>"
        try:
            requests.post(
                f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": tg_text,
                    "parse_mode": "HTML", # 彻底抛弃易崩坏的 MarkdownV2
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
    if vix > 30: return vix, f"🚨 极其恐慌 ({prefix}: {vix:.2f} > 30)"
    if vix > 25: return vix, f"⚠️ 市场恐慌 ({prefix}: {vix:.2f} > 25)"
    if vix < 15: return vix, f"✅ 市场平静 ({prefix}: {vix:.2f} < 15)"
    return vix, f"⚖️ 正常波动 ({prefix}: {vix:.2f})"

def get_market_regime() -> Tuple[str, str, pd.DataFrame]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", auto_adjust=False, fast_mode=True)
    if len(df) < 200: return "range", "数据不足，默认震荡", df
    c_close, ma200 = df['Close'].ffill().iloc[-1], df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if c_close > ma200 and trend_20d > 0.02: return "bull", "🐂 牛市主升阶段", df
    if c_close < ma200 and trend_20d < -0.02: return "bear", "🐻 熊市回调阶段", df
    return "range", "⚖️ 震荡整理阶段", df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['High_52W'] = df['High'].rolling(window=252, min_periods=120).max()
    df['Low_52W'] = df['Low'].rolling(window=252, min_periods=120).min()
    df['OBV'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()
    
    delta = df['Close'].diff()
    rs = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean() / (-delta.where(delta < 0, 0).ewm(alpha=1/14, min_periods=14).mean() + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_P20'] = df['RSI'].rolling(window=120, min_periods=60).quantile(0.20) if len(df) >= 60 else 30.0

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    hl2, atr10 = (df['High'] + df['Low']) / 2, df['TR'].rolling(window=10).mean()
    ub, lb, in_up = (hl2 + 3 * atr10).values, (hl2 - 3 * atr10).values, np.ones(len(df), dtype=bool)
    for i in range(1, len(df)):
        if pd.isna(ub[i-1]) or pd.isna(lb[i-1]): continue
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

    if len(df) >= 60:
        hist_vol = df['Volume'].iloc[-60:-10].quantile(0.8)
        risk_score = 0.6 if (df['Volume'].iloc[-10:].mean() / (hist_vol + 1e-10)) > 2.5 else 0.0
        if (df['Close'].pct_change().iloc[-5:].abs() > 0.08).any(): risk_score += 0.4
        df['Event_Risk'] = min(1.0, risk_score)
        hl_diff = (df['High'] - df['Low']).replace(0, 1e-10)
        df['CMF'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_diff * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    else:
        df['Event_Risk'], df['CMF'] = 0.0, 0.0

    return df

# ================= 4. 执行引擎 =================
def run_volatility_sentinel() -> None:
    if not (13 <= datetime.now(timezone.utc).hour <= 21): return
    active_pool = get_filtered_watchlist(max_stocks=40)
    alerts = []
    
    for sym in active_pool:
        try:
            df_h = safe_get_history(sym, period="2d", interval="1h", fast_mode=True)
            if len(df_h) < 2: continue
            
            curr_price = df_h['Close'].ffill().iloc[-1]
            open_price = df_h['Open'].iloc[-1]
            hr_chg = (curr_price - open_price) / open_price * 100 if open_price != 0 else 0
            
            is_alert = False
            alert_type = ""
            
            if abs(hr_chg) > 3.5: 
                is_alert = True
                alert_type = f"1h脉冲: {hr_chg:+.2f}%"

            df_d = safe_get_history(sym, period="1mo", interval="1d", fast_mode=True)
            atr_stop_str = ""
            
            if len(df_d) >= 15:
                df_d['TR'] = pd.concat([df_d['High']-df_d['Low'], (df_d['High']-df_d['Close'].shift()).abs(), (df_d['Low']-df_d['Close'].shift()).abs()], axis=1).max(axis=1)
                atr = df_d['TR'].rolling(14).mean().iloc[-1]
                
                atr_stop = curr_price - 1.5 * atr
                atr_target = curr_price + 3.0 * atr 
                atr_stop_str = f"  🎯 止盈: ${atr_target:.2f} | 🛡️ 止损: ${atr_stop:.2f}\n"
                
                gap = (df_d['Open'].iloc[-1] - df_d['Close'].iloc[-2]) / df_d['Close'].iloc[-2] * 100
                if abs(gap) > 4:
                    is_alert = True
                    gap_str = f"跳空缺口: {gap:+.2f}%"
                    alert_type = f"{alert_type} | {gap_str}" if alert_type else gap_str
            
            if is_alert:
                news = get_latest_news(sym)
                news_str = f"  📰 资讯: {news}\n" if news else ""
                alert_msg = (
                    f"🚨 【{sym}】 盘中极端异动 {'🚀' if hr_chg>0 else '🩸'}\n"
                    f"  💵 现价: ${curr_price:.2f}\n"
                    f"  📊 异动: {alert_type}\n"
                    f"{atr_stop_str}{news_str}"
                )
                alerts.append(alert_msg.strip())
                
        except Exception: pass
        
    if alerts: 
        send_alert("⚡ 盘中高频哨兵预警", "━━━━━━━━━━━━━━━━━━\n\n" + "\n\n".join(alerts))

def run_tech_matrix() -> None:
    regime, regime_desc, qqq_df = get_market_regime()
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    
    sector_data = {etf: (sdf['Close'].ffill().iloc[-1] / sdf['Close'].iloc[-20]) - 1 
                  for etf in Config.SECTOR_MAP.keys() 
                  if not (sdf := safe_get_history(etf, "2mo", "1d", fast_mode=True)).empty and len(sdf) >= 20}
                  
    health_score = -0.9 if vix > 30 else (-0.6 if vix > 25 else (0.9 if vix < 15 and regime == 'bull' else (0.6 if regime == 'bull' else (-0.7 if regime == 'bear' else max(-0.2, min(0.2, np.mean(list(sector_data.values())) if sector_data else 0.0)) * 2))))
    w_mul, min_score = 1.0 + health_score * 0.8, max(Config.MIN_SCORE_THRESHOLD, 8 + round(health_score * -6))
    
    reports = []
    for sym in get_filtered_watchlist():
        try:
            df = safe_get_history(sym, "1y", "1d", fast_mode=True)
            if len(df) < 150: continue
            df = calculate_indicators(df)
            curr, prev = df.iloc[-1], df.iloc[-2]
            if curr['Event_Risk'] > 0.85 or (curr['ATR'] / curr['Close'] > 0.10): continue
            
            sig, score, st_cnt = [], 0, 0
            is_vol = (curr['Volume'] / curr['Vol_MA20']) > 1.5
            is_st = curr['SuperTrend_Up'] == 1

            if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200'] and curr['SMA_200'] > df['SMA_200'].iloc[-20]:
                sig.append("🏆 [米奈尔维尼] 主升浪形态"); score += int(8 * w_mul); st_cnt += 1
            
            if len(df) >= 40 and df['Close'].iloc[-20:].min() < df['Close'].iloc[-40:-20].min() and df['OBV'].iloc[-20:].min() > df['OBV'].iloc[-40:-20].min():
                sig.append("🌊 [OBV底背离] 主力暗中建仓"); score += int(7 * w_mul); st_cnt += 1

            if not qqq_df.empty:
                m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
                if len(m_df) >= 20:
                    rs_20 = (1 + (m_df['Close_x'].iloc[-1] / m_df['Close_x'].iloc[-20] - 1)) / max(1 + (m_df['Close_y'].iloc[-1] / m_df['Close_y'].iloc[-20] - 1), 0.5)
                    if rs_20 > 1.08:
                        sig.append(f"⚡ [强相对强度] 跑赢大盘 {rs_20-1:+.1%}"); score += 7 if is_vol else 4
                    elif rs_20 < 0.92:
                        sig.append(f"🐢 [相对弱势] 跑输大盘 {1-rs_20:.1%}"); score -= 3

            dyn_rsi = curr['RSI_P20'] if pd.notna(curr['RSI_P20']) else 30.0
            if curr['RSI'] < dyn_rsi and prev['RSI'] >= dyn_rsi:
                if curr['Close'] > curr['EMA_50'] and is_st:
                    sig.append(f"🟢 [强势回踩] (RSI:{curr['RSI']:.1f})"); score += int((13 if regime=='bear' else 8) * w_mul); st_cnt += 1

            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                sig.append("🔥 [MACD金叉]"); score += int(10 * w_mul); st_cnt += 1

            is_sqz = (curr['BB_Upper'] - curr['BB_Lower']) < (curr['KC_Upper'] - curr['KC_Lower'])
            was_sqz = ((df['BB_Upper'].iloc[-6:-1] < df['KC_Upper'].iloc[-6:-1]) & (df['BB_Lower'].iloc[-6:-1] > df['KC_Lower'].iloc[-6:-1])).any() if len(df)>=6 else False
            if is_sqz:
                sig.append("📦 [TTM Squeeze ON]"); score += int((12 if regime=='range' else 6) * w_mul)
            elif was_sqz and not is_sqz and curr['MACD_Hist'] > 0 and curr['MACD_Hist'] > prev['MACD_Hist']:
                sig.append("🚀 [TTM Squeeze FIRE ↑]"); score += int((18 if regime=='bull' else 10) * w_mul); st_cnt += 1

            cmf = curr['CMF']
            if cmf > 0.20: sig.append(f"🏦 [机构高控盘] (CMF:{cmf:.2f})"); score += int(5 * w_mul); st_cnt += 1
            elif cmf < -0.15: sig.append(f"⚠️ [资金派发] (CMF:{cmf:.2f})"); score -= 4

            if curr['Above_Cloud'] == 0:
                score = int(score * 0.4); sig.append("☁️ [云层压制]")
            elif curr['Tenkan'] > curr['Kijun'] and curr['Cloud_Twist'] == 1:
                sig.append("🌥️ [一目强多头]"); score += int(6 * w_mul); st_cnt += 1
                
            gap_pct = (curr['Open'] - prev['Close']) / prev['Close']
            if 0.015 < gap_pct < 0.06 and is_vol and curr['Close'] > curr['Open']:
                sig.append(f"💥 [突破缺口] 强势放量跳空 (+{gap_pct*100:.1f}%)"); score += int(6 * w_mul); st_cnt += 1

            if score > 0 and st_cnt < 1 and not is_vol: score = int(score * 0.3)
            if is_vol and score >= 5: sig.append("🌊 [量价共振]"); score += 3
            if st_cnt >= 2: sig.append("🎯 [严密共振]"); score += 5

            if sig and score >= min_score:
                news = get_latest_news(sym)
                if news: sig.append(news)
                
                atr_stop = curr['Close'] - 1.5 * curr['ATR']
                atr_target = curr['Close'] + 3.0 * curr['ATR']
                
                reports.append({
                    "symbol": sym, "score": score, "signals": sig[:8], 
                    "curr_close": curr['Close'], "atr_stop": atr_stop, "atr_target": atr_target,
                    "sector": Config.get_sector_etf(sym), "raw_score": score
                })
        except Exception: pass

    if reports:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
            
        for sec, stks in groups.items():
            if sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.CROWDING_MIN_STOCKS:
                stks.sort(key=lambda x: x["score"], reverse=True)
                pen = max(0.6, min(0.9, Config.CROWDING_PENALTY * (1.0 + health_score * 0.3)))
                for s in stks[1:]: s["score"] = int(s["raw_score"] * pen)

        reports.sort(key=lambda x: x["score"], reverse=True)
        
        # 🚀 采用纵向层级排版，清晰划分基础信息与触发因子
        medals = ['🥇', '🥈', '🥉']
        txts = []
        for idx, r in enumerate(reports[:15]):
            icon = medals[idx] if idx < 3 else '🔸'
            sigs_formatted = "\n".join([f"    • {s}" for s in r["signals"]])
            txt = (
                f"{icon} 【{r['symbol']}】 综合评分: {r['score']}\n"
                f"  💵 现价: ${r['curr_close']:.2f}\n"
                f"  🎯 止盈: ${r['atr_target']:.2f} | 🛡️ 止损: ${r['atr_stop']:.2f}\n"
                f"  💡 触发共振:\n{sigs_formatted}"
            )
            txts.append(txt)

        perf_text = load_strategy_performance_tag()
        header = f"📊 大盘环境感知:\n  • {vix_desc}\n  • {regime_desc}"
        if perf_text: header = f"{perf_text}\n\n{header}"
        
        final_content = header + "\n━━━━━━━━━━━━━━━━━━\n\n" + "\n\n".join(txts) + f"\n\n*(入选门槛: {min_score}分 | 降权防诱多机制已开启)*"
        send_alert("多因子优选异动 (矩阵共振版)", final_content)
        
        with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), "top_picks": [{"symbol": r["symbol"], "score": r["score"], "signals": r["signals"]} for r in reports[:15]]}, ensure_ascii=False) + "\n")

def run_backtest_engine() -> None:
    if not os.path.exists(Config.LOG_FILE): return
    trades = []
    with open(Config.LOG_FILE, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                trades.extend([{'date': log['date'], 'symbol': p['symbol'], 'signals': p.get('signals', [])} for p in log.get('top_picks', [])])
            except: pass
    if not trades: return
    syms = list(set([t['symbol'] for t in trades]))
    try:
        df_c = yf.download(syms, period="2mo", progress=False, threads=2)['Close']
        if len(syms) == 1: df_c = pd.DataFrame(df_c, columns=syms)
        df_c.index = df_c.index.strftime('%Y-%m-%d')
    except Exception: return
    
    stats, factor_rets = {'T+1': [], 'T+3': [], 'T+5': []}, {}
    for t in trades:
        sym, r_dt = t['symbol'], t['date']
        if sym not in df_c.columns: continue
        valid = df_c.index[df_c.index >= r_dt]
        if len(valid) == 0: continue
        e_idx = df_c.index.get_loc(valid[0])
        e_px = df_c.at[valid[0], sym]
        for d in [1, 3, 5]:
            if e_idx + d < len(df_c):
                x_px = df_c.iloc[e_idx + d][sym]
                if not np.isnan(x_px):
                    ret = (x_px - e_px) / e_px
                    stats[f'T+{d}'].append(ret)
                    if d == 3:
                        for sig_txt in t.get('signals', []):
                            m = re.search(r'\[(.*?)\]', sig_txt)
                            if m:
                                tag = f"[{m.group(1)}]"
                                factor_rets.setdefault(tag, []).append(ret)
    
    res = {p: {'win_rate': sum(1 for x in r if x > 0)/len(r), 'avg_ret': sum(r)/len(r), 'total_trades': len(r)} for p, r in stats.items() if r}
    f_res = {t: {'win_rate': sum(1 for x in r if x > 0)/len(r), 'avg_ret': sum(r)/len(r), 'count': len(r)} for t, r in factor_rets.items() if len(r) >= 2}
    with open(Config.STATS_FILE, 'w') as f: json.dump({"overall": res, "factors": f_res}, f)
    
    report_lines = [
        "# 📈 自动化量化监控战报 (Auto-Backtest Report)",
        f"**最后更新时间:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 1. 周期胜率总览",
        "| 持仓周期 | 胜率 (Win Rate) | 平均收益 (Avg Return) | 样本交易笔数 |",
        "| :---: | :---: | :---: | :---: |"
    ]
    for p in ['T+1', 'T+3', 'T+5']:
        if p in res:
            d = res[p]
            report_lines.append(f"| **{p}** | {d['win_rate']*100:.1f}% | {d['avg_ret']*100:+.2f}% | {d['total_trades']} |")
        else:
            report_lines.append(f"| **{p}** | N/A | N/A | 0 |")

    sorted_factors = []
    if f_res:
        report_lines.extend([
            "",
            "## 2. 🧬 核心因子有效性排行 (T+3 波段)",
            "| **因子标签** | **胜率** | **均收益** | **触发次数** |",
            "| :--- | :---: | :---: | :---: |"
        ])
        sorted_factors = sorted(f_res.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        for tag, d in sorted_factors:
            report_lines.append(f"| {tag} | {d['win_rate']*100:.1f}% | {d['avg_ret']*100:+.2f}% | {d['count']} |")

    report_lines.extend(["", "> *注：本报告每周五自动生成。收益率不包含滑点与佣金摩擦。*"])
    try:
        with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    except Exception as e:
        logger.error(f"写入 Markdown 失败: {e}")
        
    # 🚀 采用纵向层级排版的周末战报
    alert_lines = ["📊 周期胜率总览"]
    for p, d in res.items():
        alert_lines.append(f"  • {p}: 胜率 {d['win_rate']*100:.1f}% | 均收益 {d['avg_ret']*100:+.2f}%")
        
    if f_res:
        alert_lines.extend(["", "━━━━━━━━━━━━━━━━━━", "", "🧬 核心因子排行 (T+3 波段)"])
        medals = ['1️⃣', '2️⃣', '3️⃣']
        for idx, (tag, d) in enumerate(sorted_factors[:3]):
            icon = medals[idx] if idx < 3 else '🔸'
            alert_lines.append(f"  {icon} {tag}\n      胜率 {d['win_rate']*100:.1f}% | 触发 {d['count']}次")
            
    send_alert("策略终极回测战报", "\n".join(alert_lines))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    if m == "sentinel": run_volatility_sentinel()
    elif m == "matrix": run_tech_matrix()
    elif m == "backtest": run_backtest_engine()
    elif m == "test": send_alert("✅ 连通性测试", "系统环境正常，全新的 HTML 纵排结构卡片引擎已上线！")
