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
    
    # --- 🚀 [护城河升级] 绝对稳定的核心资产名单 ---
    # 彻底告别对外部网页的依赖。这 150 只高流动性标的足以覆盖美股核心 Alpha 收益。
    # 即使所有网页获取失效，系统仍将以此名单为基础完美运转数年。
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
        "QQQ", "SPY", "DIA", "IWM", "SOXX", "SMH", "XLK", "XLF", "XLV", "XLE"
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
        logger.error("❌ 未配置任何推送渠道。请检查 GitHub Secrets。")
        sys.exit(1)
    logger.info("✅ 环境变量与配置校验通过")

# ================= 2. 数据与网络工具模块 =================
def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(0.2, 0.5) if fast_mode else (random.uniform(2.0, 4.5) if "1d" in interval else random.uniform(1.2, 2.5))
            time.sleep(sleep_sec)
            
            # 移除 session，让新版 yfinance 自动接管底层反爬伪装
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=auto_adjust, timeout=15)
            if not df.empty: return df
        except Exception as e:
            logger.warning(f"[{symbol}] 尝试 {attempt+1} 失败: {e}")
            if attempt == retries - 1: return pd.DataFrame()
            time.sleep((12 + attempt * 10) if "429" in str(e).lower() else (4 + attempt * 3))
    return pd.DataFrame()

def get_latest_news(symbol: str) -> str:
    try:
        # 移除 session
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
    except Exception as e:
        logger.debug(f"[{symbol}] 新闻拉取失败: {e}")
    return ""

def get_filtered_watchlist(max_stocks: int = 120) -> List[str]:
    logger.info(">>> 漏斗过滤：尝试从稳定的开源数据源拉取全市场名单...")
    tickers = set(Config.CORE_WATCHLIST)
    
    # --- 修复：补回漏掉的 GitHub CSV 拉取与变量声明 ---
    try:
        sp500_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        resp = requests.get(sp500_url, headers=_GLOBAL_HEADERS, timeout=15)
        if resp.status_code == 200:
            from io import StringIO
            df_sp500 = pd.read_csv(StringIO(resp.text))
            if 'Symbol' in df_sp500.columns:
                tickers.update(df_sp500['Symbol'].dropna().astype(str).str.replace('.', '-').tolist())
    except Exception as e:
        logger.warning(f"⚠️ 开源 CSV 拉取失败，将完全使用内置的硬核股票池进行扫描。 ({e})")
    
    tickers_list = list(tickers)
    logger.info(f"✅ 获取待筛名单: {len(tickers_list)} 只。开始分块拉取粗筛...")
    # ---------------------------------------------------

    try:
        chunk_size = 50  # 降低分块大小，防止触发 Yahoo DDoS 防护
        dfs = []
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            # 移除 session，让新版 yfinance 自动接管防封禁
            chunk_df = yf.download(chunk, period="5d", progress=False, threads=2)
            if not chunk_df.empty: dfs.append(chunk_df)
            if i + chunk_size < len(tickers_list): time.sleep(random.uniform(2.5, 4.0))
                
        if not dfs: raise ValueError("所有分块批量下载均失败")
        df = pd.concat(dfs, axis=1)
        
        # 处理单维或多维的 DataFrame 结构
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close'] if 'Close' in df.columns else df.xs('Close', level=0, axis=1)
            volume_df = df['Volume'] if 'Volume' in df.columns else df.xs('Volume', level=0, axis=1)
        else:
            close_df = df
            volume_df = pd.DataFrame(1e6, index=df.index, columns=df.columns) # Fallback

        closes = close_df.dropna(axis=1, how='all').ffill().iloc[-1]
        volumes = volume_df.dropna(axis=1, how='all').mean()
        
        turnovers = (closes * volumes).dropna()
        valid_turnovers = turnovers[(closes > 10.0) & (volumes > 1000000)]
        
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        if top_tickers:
            logger.info(f"✅ 漏斗完成！精选 {len(top_tickers)} 只标的进入深度扫描。")
            return top_tickers
        raise ValueError("过滤后名单为空")
    except Exception as e:
        logger.error(f"❌ 批量拉取或过滤彻底失败: {e}。直接启用内置硬核名单降级运行。")
        # 直接使用 Config 中精挑细选的头等舱名单，确保系统永不停转
        return Config.CORE_WATCHLIST[:max_stocks]

def escape_md_v2(text: str) -> str:
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)

def load_strategy_performance_tag() -> str:
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                t3 = json.load(f).get("T+3")
                if t3 and t3.get('total_trades', 0) > 0:
                    return f"📈 **策略历史表现 (T+3)**: 胜率 **{t3['win_rate']:.1%}** | 均收益 **{t3['avg_ret']:+.2%}**\n\n"
    except Exception as e:
        logger.debug(f"无回测统计: {e}")
    return ""

def send_alert(title: str, content: str) -> None:
    if not content.strip(): return
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": f"【{Config.DINGTALK_KEYWORD}】{title}", "text": f"### 🤖 【{Config.DINGTALK_KEYWORD}量化监控】\n#### {title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, timeout=10)
            except Exception as e: logger.error(f"Webhook 推送异常: {e}")
                
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
        except Exception as e:
            logger.error(f"Telegram 推送异常: {e}")

# ================= 3. 大盘感知与指标模块 =================
def get_vix_level(qqq_df_for_shadow: pd.DataFrame = None) -> Tuple[float, str]:
    df = safe_get_history(Config.VIX_INDEX, period="5d", interval="1d", retries=3, auto_adjust=False, fast_mode=True)
    vix, is_simulated = 18.0, False
    
    if not df.empty and len(df) >= 1:
        vix = df['Close'].ffill().iloc[-1]
    else:
        qqq_df = qqq_df_for_shadow if qqq_df_for_shadow is not None else safe_get_history(Config.INDEX_ETF, period="2mo", interval="1d", retries=2, auto_adjust=True, fast_mode=True)
        if qqq_df is not None and len(qqq_df) >= 20:
            vix = qqq_df['Close'].pct_change().dropna().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 * 1.2
            is_simulated = True
            
    prefix = "影子VIX" if is_simulated else "VIX"
    if vix > 30: return vix, f"🚨 极其恐慌 ({prefix}: {vix:.2f} > 30)，切勿盲目抄底！"
    if vix > 25: return vix, f"⚠️ 市场恐慌 ({prefix}: {vix:.2f} > 25)，防接飞刀全开！"
    if vix < 15: return vix, f"✅ 市场平静 ({prefix}: {vix:.2f} < 15)，信号可靠性高。"
    return vix, f"⚖️ 正常波动 ({prefix}: {vix:.2f})"

def get_market_regime() -> Tuple[str, str, pd.DataFrame]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", auto_adjust=False, fast_mode=True)
    if len(df) < 200: return "range", "大盘数据不足，默认震荡市", df
    c_close, ma200 = df['Close'].ffill().iloc[-1], df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if c_close > ma200 and trend_20d > 0.02: return "bull", "🐂 牛市主升阶段", df
    if c_close < ma200 and trend_20d < -0.02: return "bear", "🐻 熊市回调阶段", df
    return "range", "⚖️ 震荡整理阶段", df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    
    # 基础均线计算
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['High_52W'] = df['High'].rolling(window=252, min_periods=120).max()
    df['Low_52W'] = df['Low'].rolling(window=252, min_periods=120).min()
    
    # RSI
    delta = df['Close'].diff()
    rs = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean() / (-delta.where(delta < 0, 0).ewm(alpha=1/14, min_periods=14).mean() + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_P20'] = df['RSI'].rolling(window=120, min_periods=60).quantile(0.20) if len(df) >= 60 else 30.0

    # MACD & EMA
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    # ATR & SuperTrend & Channels
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
    df['BB_Upper'], df['BB_Lower'], df['BB_Width'] = bb_ma + 2 * bb_std, bb_ma - 2 * bb_std, (bb_ma + 2 * bb_std - (bb_ma - 2 * bb_std)) / bb_ma

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
        if (df['Volume'].iloc[-5:] > hist_vol * 1.5).sum() >= 3: risk_score += 0.3
        if df['Close'].iloc[-2] != 0 and abs((df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) > 0.06: risk_score += 0.35
        df['Event_Risk'] = min(1.0, risk_score)
        
        hl_diff = (df['High'] - df['Low']).replace(0, 1e-10)
        df['CMF'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_diff * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    else:
        df['Event_Risk'], df['CMF'] = 0.0, 0.0

    return df

# ================= 4. 业务执行引擎 =================
def run_volatility_sentinel() -> None:
    if not (13 <= datetime.now(timezone.utc).hour <= 21): return logger.info("💤 非交易时段跳过高频扫描。")
    
    logger.info(">>> 启动高频哨兵：获取全市场近期最高活跃度异动池 (Top 40)...")
    active_pool = get_filtered_watchlist(max_stocks=40)
    
    alerts = []
    for sym in active_pool:
        if sym in Config.BLACKLIST: continue
        try:
            df = safe_get_history(sym, period="2d", interval="1h", fast_mode=True)
            if len(df) < 2: continue
            
            curr_price = df['Close'].ffill().iloc[-1]
            open_price = df['Open'].iloc[-1]
            hr_chg = (curr_price - open_price) / open_price * 100 if open_price != 0 else 0
            
            if abs(hr_chg) > 3.5: 
                alerts.append(f"> **{sym}** 盘中极端异动 {'🚀' if hr_chg>0 else '🩸'} | 1h 波动: **{hr_chg:+.2f}%** (现价: ${curr_price:.2f})")
            
            df_d = safe_get_history(sym, period="5d", interval="1d", fast_mode=True)
            if len(df_d) >= 2:
                gap = (df_d['Open'].iloc[-1] - df_d['Close'].iloc[-2]) / df_d['Close'].iloc[-2] * 100
                if abs(gap) > 4: 
                    alerts.append(f"> **{sym}** 日线跳空确认 {'💥' if gap>0 else '⚠️'} | 缺口: **{gap:+.2f}%**")
                    
        except Exception as e: 
            logger.debug(f"[{sym}] 哨兵巡逻跳过: {e}")
            
    if alerts: 
        send_alert("⚡ 高活跃游资/机构标的 盘中异动监控", "\n\n".join(alerts))
    else:
        logger.info("哨兵巡视完毕，当前活跃池无极端异动。")

def run_tech_matrix() -> None:
    regime, regime_desc, qqq_df = get_market_regime()
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    
    sector_data = {etf: (sdf['Close'].ffill().iloc[-1] / sdf['Close'].iloc[-20]) - 1 
                  for etf in Config.SECTOR_MAP.keys() 
                  if not (sdf := safe_get_history(etf, "2mo", "1d", fast_mode=True)).empty and len(sdf) >= 20}
                  
    health_score = -0.9 if vix > 30 else (-0.6 if vix > 25 else (0.9 if vix < 15 and regime == 'bull' else (0.6 if regime == 'bull' else (-0.7 if regime == 'bear' else max(-0.2, min(0.2, np.mean(list(sector_data.values())) if sector_data else 0.0)) * 2))))
    w_mul, min_score = 1.0 + health_score * 0.8, max(Config.MIN_SCORE_THRESHOLD, 8 + round(health_score * -6))
    
    reports = []
    for idx, sym in enumerate(get_filtered_watchlist()):
        if sym in Config.BLACKLIST: continue
        try:
            df = safe_get_history(sym, "1y", "1d")
            if len(df) < 150 or df['Volume'].tail(20).mean() < 1e6: continue
            
            df = calculate_indicators(df)
            curr, prev, ev_risk = df.iloc[-1], df.iloc[-2], df['Event_Risk'].iloc[-1]
            if ev_risk > 0.85 or (curr['ATR'] / curr['Close'] > 0.10): continue
            
            sig, score, st_cnt = [], 0, 0
            is_vol = (curr['Volume'] / curr['Vol_MA20'] if curr['Vol_MA20'] > 0 else 1.0) > 1.5
            is_st = curr['SuperTrend_Up'] == 1

            if pd.notna(curr['SMA_200']) and pd.notna(curr['Low_52W']):
                minervini_pass = (
                    curr['Close'] > curr['SMA_50'] > curr['SMA_150'] > curr['SMA_200'] and
                    curr['SMA_200'] > df['SMA_200'].iloc[-20] and
                    curr['Close'] > curr['Low_52W'] * 1.25 and
                    curr['Close'] > curr['High_52W'] * 0.75
                )
                if minervini_pass:
                    sig.append("🏆 **[米奈尔维尼模板]** 绝对主升浪形态确认，胜率极高！")
                    score += int(8 * w_mul)
                    st_cnt += 1

            if not qqq_df.empty:
                m_df = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner')
                if len(m_df) >= 20:
                    rs_20 = (1 + (m_df['Close_x'].iloc[-1] / m_df['Close_x'].iloc[-20] - 1)) / max(1 + (m_df['Close_y'].iloc[-1] / m_df['Close_y'].iloc[-20] - 1), 0.5)
                    if rs_20 > 1.08:
                        sig.append(f"⚡ **[强相对强度]** 跑赢大盘 {rs_20-1:+.1%}"); score += 7 if is_vol else 4
                    elif rs_20 < 0.92:
                        sig.append(f"🐢 **[相对弱势]** 跑输大盘 {1-rs_20:.1%}"); score -= 3

            dyn_rsi = curr['RSI_P20'] if pd.notna(curr['RSI_P20']) else 30.0
            if curr['RSI'] < dyn_rsi and prev['RSI'] >= dyn_rsi:
                if curr['Close'] > curr['EMA_50'] and is_st:
                    sig.append(f"🟢 **[强势回踩]** (RSI:{curr['RSI']:.1f})"); score += int((13 if regime=='bear' else 8) * w_mul); st_cnt += 1
                else:
                    sig.append(f"⚠️ **[弱势超卖]**"); score += int((13 if regime=='bear' else 8) * 0.4)

            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                w = 10 if regime == 'bull' else 6
                if curr['MACD'] > 0: sig.append("🔥 **[零上金叉]**"); score += int(w * w_mul); st_cnt += 1
                else: sig.append("🔸 **[零下金叉]**"); score += int(w * 0.6)

            is_sqz = (curr['BB_Upper'] - curr['BB_Lower']) < (curr['KC_Upper'] - curr['KC_Lower'])
            was_sqz = ((df['BB_Upper'].iloc[-6:-1] < df['KC_Upper'].iloc[-6:-1]) & (df['BB_Lower'].iloc[-6:-1] > df['KC_Lower'].iloc[-6:-1])).any() if len(df)>=6 else False
            
            if is_sqz:
                sig.append("📦 **[TTM Squeeze ON]**"); score += int((12 if regime=='range' else 6) * w_mul)
            elif was_sqz and not is_sqz and curr['MACD_Hist'] > 0 and curr['MACD_Hist'] > prev['MACD_Hist']:
                sig.append("🚀 **[TTM Squeeze FIRE ↑]**"); score += int((18 if regime=='bull' else 10) * w_mul); st_cnt += 1

            if prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
                if is_st and is_vol: sig.append("📈 **[有效突破]** 站上50日线"); score += int((9 if regime=='bull' else 5) * w_mul); st_cnt += 1
                else: sig.append("📉 **[疲软突破]**"); score -= 3

            cmf = curr['CMF']
            if cmf > 0.20: sig.append(f"🏦 **[机构高控盘]** (CMF:{cmf:.2f})"); score += int(5 * w_mul); st_cnt += 1
            elif cmf < -0.15: sig.append(f"⚠️ **[资金派发]** (CMF:{cmf:.2f})"); score -= 4

            if curr['Above_Cloud'] == 0:
                score = int(score * 0.4); sig.append("☁️ **[云层压制]**")
            elif curr['Tenkan'] > curr['Kijun'] and curr['Cloud_Twist'] == 1:
                sig.append("🌥️ **[一目强多头]**"); score += int(6 * w_mul); st_cnt += 1

            if score > 0 and st_cnt < 1 and not is_vol: score = int(score * 0.3)
            if is_vol and score >= 5: sig.append("🌊 **[量价共振]**"); score += 3
            if st_cnt >= 2: sig.append("🎯 **[严密共振]**"); score += 5

            if sig and score >= min_score:
                news = get_latest_news(sym)
                if news: sig.append(news)
                
                atr_stop = curr['Close'] - 1.5 * curr['ATR']
                
                reports.append({
                    "symbol": sym, 
                    "score": score, 
                    "signals": sig[:8], 
                    "curr_close": curr['Close'], 
                    "atr_stop": atr_stop,          
                    "turnover": curr['Close']*curr['Volume'], 
                    "sector": Config.get_sector_etf(sym), 
                    "raw_score": score
                })
        except Exception as e: logger.debug(f"[{sym}] 扫描静默错误: {e}")

    if reports:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in reports: groups[r["sector"]].append(r)
            
        for sec, stks in groups.items():
            if sec not in Config.CROWDING_EXCLUDE_SECTORS and len(stks) >= Config.CROWDING_MIN_STOCKS:
                stks.sort(key=lambda x: x["score"], reverse=True)
                pen = max(0.6, min(0.9, Config.CROWDING_PENALTY * (1.0 + health_score * 0.3)))
                for s in stks[1:]: s["score"] = int(s["raw_score"] * pen); s["is_crowded"] = True; s["crowding_penalty"] = pen

        reports.sort(key=lambda x: x["score"], reverse=True)
        txts = [f"**{r['symbol']}** (${r['curr_close']:.2f} | 🛡️ 止损: ${r['atr_stop']:.2f} | 🌟 {r['score']})\n> " + "\n> ".join(r["signals"]) for r in reports[:15]]
        
        send_alert("📊 全市场优选异动", f"{load_strategy_performance_tag()}*{vix_desc}*\n*{regime_desc}*\n\n" + "\n\n".join(txts) + f"\n\n*(门槛: {min_score}分 | 降权机制开启)*")
        
        try:
            with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"date": datetime.now(timezone.utc).strftime('%Y-%m-%d'), "vix": round(vix, 2), "regime": regime, "top_picks": [{"symbol": r["symbol"], "score": r["score"]} for r in reports[:15]]}, ensure_ascii=False) + "\n")
        except Exception as e: logger.error(f"写入回测日志失败: {e}")
    else: logger.info("个股未达动态门槛。")

def run_backtest_engine() -> None:
    logger.info(">>> 启动终极绩效回测...")
    if not os.path.exists(Config.LOG_FILE): return logger.warning("无历史日志，跳过回测。")
    
    trades = []
    with open(Config.LOG_FILE, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                trades.extend([{'date': log['date'], 'symbol': p['symbol']} for p in log.get('top_picks', [])])
            except: pass

    if not trades: return
    syms = list(set([t['symbol'] for t in trades]))
    start_dt = (datetime.strptime(min([t['date'] for t in trades]), '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')

    try:
        # 回测批量下载移除 session
        df_c = yf.download(syms, start=start_dt, progress=False, threads=2)['Close']
        if len(syms) == 1: df_c = pd.DataFrame(df_c, columns=syms)
        df_c.index = df_c.index.strftime('%Y-%m-%d')
    except Exception as e: return logger.error(f"批量回测拉取失败: {e}")

    stats = {'T+1': [], 'T+3': [], 'T+5': []}
    for t in trades:
        sym, r_dt = t['symbol'], t['date']
        if sym not in df_c.columns: continue
        valid = df_c.index[df_c.index >= r_dt]
        if len(valid) == 0: continue
        
        e_idx, e_px = df_c.index.get_loc(valid[0]), df_c.at[valid[0], sym]
        if pd.isna(e_px) or e_px <= 0: continue

        for d in [1, 3, 5]:
            if e_idx + d < len(df_c):
                x_px = df_c.iloc[e_idx + d][sym]
                if not pd.isna(x_px): stats[f'T+{d}'].append((x_px - e_px) / e_px)

    res = {p: {'win_rate': sum(1 for x in r if x > 0)/len(r), 'avg_ret': sum(r)/len(r), 'total_trades': len(r)} for p, r in stats.items() if r}
    if not res: return logger.info("闭环数据不足。")

    try:
        with open(Config.STATS_FILE, 'w') as f: json.dump(res, f)
        open(Config.REPORT_FILE, 'w').write('\n'.join(["# 📈 回测战报", f"**时间:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"] + [f"| **{p}** | {d['win_rate']*100:.1f}% | {d['avg_ret']*100:+.2f}% | {d['total_trades']} |" for p, d in res.items()]))
    except: pass
    
    send_alert("📅 策略历史回测周报", "\n\n".join([f"> ⏱️ **{p}**: 胜率 **{d['win_rate']*100:.1f}%** | 均收益 **{d['avg_ret']*100:+.2f}%**" for p, d in res.items()]))

# ================= 5. 统一入口 =================
if __name__ == "__main__":
    validate_config()
    m = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    if m == "sentinel": run_volatility_sentinel()
    elif m == "matrix": run_tech_matrix()
    elif m == "daily": send_alert("📝 占位", "每日复盘逻辑合并至 Matrix")
    elif m == "backtest": run_backtest_engine()
    elif m == "test": send_alert("✅ 测试", "引擎响应正常！已载入绝对降级护城河与动态止损系统。")
