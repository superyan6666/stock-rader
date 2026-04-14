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
logging.getLogger('yfinance').setLevel(logging.CRITICAL) # 🛡️ 屏蔽 yfinance 报错刷屏

# ================= 1. 日志与配置管理 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantBot")

# 全局网络伪装头 (保留给下载 GitHub CSV 开源名单使用)
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
    
    # --- 🚀 [护城河] 绝对稳定的核心资产名单 ---
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
    
    # 自动补齐 Git 需要的占位文件，防止 Action 提交时报错
    if not os.path.exists(Config.LOG_FILE): open(Config.LOG_FILE, 'a').close()
    if not os.path.exists(Config.REPORT_FILE): open(Config.REPORT_FILE, 'a').close()
    if not os.path.exists(Config.STATS_FILE):
        with open(Config.STATS_FILE, 'w') as f: f.write("{}")
            
    logger.info("✅ 环境校验通过")

# ================= 2. 数据工具模块 =================
def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            # 🚀 提速模式：压缩休眠
            sleep_sec = random.uniform(0.1, 0.3) if fast_mode else (random.uniform(2.0, 4.0) if "1d" in interval else random.uniform(1.0, 2.0))
            time.sleep(sleep_sec)
            
            # 使用新版 yfinance 自动接管反爬伪装
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
                return f"📰 {sentiment} **{title}** ({publisher})"
    except Exception: pass
    return ""

def get_filtered_watchlist(max_stocks: int = 120) -> List[str]:
    logger.info(">>> 漏斗过滤：从稳定数据源拉取全市场名单...")
    tickers = set(Config.CORE_WATCHLIST)
    
    try:
        sp500_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        resp = requests.get(sp500_url, headers=_GLOBAL_HEADERS, timeout=15)
        if resp.status_code == 200:
            from io import StringIO
            df_sp500 = pd.read_csv(StringIO(resp.text))
            if 'Symbol' in df_sp500.columns:
                tickers.update(df_sp500['Symbol'].dropna().astype(str).str.replace('.', '-').tolist())
    except Exception as e:
        logger.warning(f"⚠️ CSV 降级运行: {e}")
    
    tickers_list = list(tickers)
    logger.info(f"✅ 待筛名单: {len(tickers_list)} 只。开始分块巡逻...")

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
        
        turnovers = (closes * volumes).dropna()
        valid_turnovers = turnovers[(closes > 10.0) & (volumes > 1000000)]
        
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        if top_tickers:
            logger.info(f"✅ 漏斗完成！精选 {len(top_tickers)} 只进入深度扫描。")
            return top_tickers
        return Config.CORE_WATCHLIST[:max_stocks]
    except Exception as e:
        logger.error(f"❌ 批量下载崩溃: {e}")
        return Config.CORE_WATCHLIST[:max_stocks]

def escape_md_v2(text: str) -> str:
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                t3 = stats_data.get("overall", {}).get("T+3") if "overall" in stats_data else stats_data.get("T+3")
                if t3 and t3.get('total_trades', 0) > 0:
                    return f"📈 **策略历史表现 (T+3)**: 胜率 **{t3['win_rate']:.1%}** | 均收益 **{t3['avg_ret']:+.2%}**\n\n"
    except Exception: pass
    return ""

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": f"【{Config.DINGTALK_KEYWORD}】{title}", "text": f"### 🤖 【{Config.DINGTALK_KEYWORD}量化监控】\n#### {title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, timeout=10)
            except Exception: pass
                
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": f"🤖 *量化监控系统*\n\n*{escape_md_v2(title)}*\n\n{escape_md_v2(content)}\n\n⏱️ _{escape_md_v2(formatted_time)}_",
                    "parse_mode": "MarkdownV2",
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
    
    # RSI
    delta = df['Close'].diff()
    rs = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean() / (-delta.where(delta < 0, 0).ewm(alpha=1/14, min_periods=14).mean() + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_P20'] = df['RSI'].rolling(window=120, min_periods=60).quantile(0.20) if len(df) >= 60 else 30.0

    # MACD & ATR
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # SuperTrend & Channels
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

    # Ichimoku
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    df['Cloud_Twist'] = (df['SenkouA'] > df['SenkouB']).astype(int)

    # Risk & CMF
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
            df = safe_get_history(sym, period="2d", interval="1h", fast_mode=True)
            if len(df) < 2: continue
            hr_chg = (df['Close'].ffill().iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100
            if abs(hr_chg) > 3.5: 
                alerts.append(f"> **{sym}** 盘中异动 {'🚀' if hr_chg>0 else '🩸'} | 波动: **{hr_chg:+.2f}%**")
        except Exception: pass
    if alerts: send_alert("⚡ 盘中哨兵预警", "\n\n".join(alerts))

def run_tech_matrix() -> None:
    regime, regime_desc, qqq_df = get_market_regime()
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    health_score = -0.9 if vix > 30 else (0.9 if vix < 15 and regime == 'bull' else 0.0)
    w_mul, min_score = 1.0 + health_score * 0.8, max(Config.MIN_SCORE_THRESHOLD, 8)
    
    reports = []
    for sym in get_filtered_watchlist():
        try:
            df = safe_get_history(sym, "1y", "1d", fast_mode=True)
            if len(df) < 150: continue
            df = calculate_indicators(df)
            curr, prev = df.iloc[-1], df.iloc[-2]
            if curr['Event_Risk'] > 0.85: continue
            
            sig, score, st_cnt = [], 0, 0
            is_vol = (curr['Volume'] / curr['Vol_MA20']) > 1.5
            is_st = curr['SuperTrend_Up'] == 1

            # 因子1: 米奈尔维尼模板
            if curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200'] and curr['SMA_200'] > df['SMA_200'].iloc[-20]:
                sig.append("🏆 **[米奈尔维尼模板]** 主升浪形态"); score += int(8 * w_mul); st_cnt += 1
            
            # 因子2: OBV 底背离
            if df['Close'].iloc[-20:].min() < df['Close'].iloc[-40:-20].min() and df['OBV'].iloc[-20:].min() > df['OBV'].iloc[-40:-20].min():
                sig.append("🌊 **[OBV底背离]** 主力暗中建仓"); score += int(7 * w_mul); st_cnt += 1

            # 因子3: MACD 金叉
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                sig.append("🔥 **[MACD金叉]**"); score += int(10 * w_mul); st_cnt += 1

            # 因子4: TTM Squeeze
            if (curr['BB_Upper'] - curr['BB_Lower']) < (curr['KC_Upper'] - curr['KC_Lower']):
                sig.append("📦 **[TTM Squeeze]**"); score += int(6 * w_mul)

            if sig and score >= min_score:
                news = get_latest_news(sym)
                if news: sig.append(news)
                atr_stop = curr['Close'] - 1.5 * curr['ATR']
                reports.append({"symbol": sym, "score": score, "signals": sig[:8], "curr_close": curr['Close'], "atr_stop": atr_stop})
        except Exception: pass

    if reports:
        reports.sort(key=lambda x: x["score"], reverse=True)
        txts = [f"**{r['symbol']}** (${r['curr_close']:.2f} | 🛡️ 止损: ${r['atr_stop']:.2f} | 🌟 {r['score']})\n> " + "\n> ".join(r["signals"]) for r in reports[:15]]
        send_alert("📊 多因子扫描共振", f"{load_strategy_performance_tag()}*{vix_desc}*\n*{regime_desc}*\n\n" + "\n\n".join(txts))
        with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), "top_picks": [{"symbol": r["symbol"], "score": r["score"], "signals": r["signals"]} for r in reports[:15]]}) + "\n")

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
    f_res = {t: {'win_rate': sum(1 for x in r if x > 0)/len(r), 'count': len(r)} for t, r in factor_rets.items() if len(r) >= 2}
    with open(Config.STATS_FILE, 'w') as f: json.dump({"overall": res, "factors": f_res}, f)
    send_alert("📅 策略回测报告", "\n".join([f"> **{p}**: 胜率 {d['win_rate']*100:.1f}%" for p, d in res.items()]))

if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    if m == "sentinel": run_volatility_sentinel()
    elif m == "matrix": run_tech_matrix()
    elif m == "backtest": run_backtest_engine()
    elif m == "test": send_alert("✅ 测试", "系统环境正常，极简架构已对齐。")
