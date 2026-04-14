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

class Config:
    # Webhook 支持钉钉/飞书/企业微信
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    # 新增 Telegram 原生多渠道推送支持
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    
    DINGTALK_KEYWORD: str = "AI"
    CORE_WATCHLIST: List[str] = ["NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]
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

    # 板块拥挤度过滤强度（0.6~0.9 之间，数值越小惩罚越狠）
    CROWDING_PENALTY: float = 0.75
    CROWDING_MIN_STOCKS: int = 2
    CROWDING_EXCLUDE_SECTORS: List[str] = ["QQQ"]
    
    # 统一管理回测与日志的文件路径
    LOG_FILE: str = "backtest_log.jsonl"
    STATS_FILE: str = "strategy_stats.json"
    REPORT_FILE: str = "backtest_report.md"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols:
                return etf
        return Config.INDEX_ETF

def validate_config():
    if not Config.WEBHOOK_URL and not Config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未配置任何推送渠道 (WEBHOOK_URL 或 TELEGRAM_BOT_TOKEN)。请检查 GitHub Secrets。")
        sys.exit(1)
    logger.info("✅ 环境变量与配置校验通过")

# ================= 2. 数据与网络工具模块 =================

_NEWS_CACHE = {}

def safe_get_history(symbol: str, period: str = "6mo", interval: str = "1d", retries: int = 5, auto_adjust: bool = True, fast_mode: bool = False) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            if fast_mode:
                sleep_sec = random.uniform(0.2, 0.5)
            else:
                sleep_sec = random.uniform(2.0, 4.5) if "1d" in interval else random.uniform(1.2, 2.5)
                
            time.sleep(sleep_sec)
            
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=auto_adjust, timeout=15)
            if not df.empty:
                return df
                
        except Exception as e:
            logger.warning(f"[{symbol}] 第 {attempt+1} 次获取数据失败: {e}")
            if attempt == retries - 1:
                logger.error(f"[{symbol}] 最终获取失败，已放弃")
                return pd.DataFrame()
            
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg:
                time.sleep(12 + attempt * 10)
            else:
                time.sleep(4 + attempt * 3)
            
    return pd.DataFrame()

def get_latest_news(symbol: str) -> str:
    current_time = time.time()
    
    if symbol in _NEWS_CACHE:
        cached_time, cached_news = _NEWS_CACHE[symbol]
        if current_time - cached_time < 1800:
            return cached_news

    try:
        news_data = yf.Ticker(symbol).news
        if news_data and len(news_data) > 0:
            latest = news_data[0]
            title = latest.get('title', '')
            publisher = latest.get('publisher', '')
            if title:
                lower_title = title.lower()
                if any(kw in lower_title for kw in ['beat', 'raise', 'upgrade', 'strong', 'surge', 'rally', 'buy', 'bullish', 'record', 'profit', 'revenue growth']):
                    sentiment = "🟢 [利好]"
                elif any(kw in lower_title for kw in ['miss', 'cut', 'downgrade', 'weak', 'decline', 'sell', 'bearish', 'warn', 'loss', 'recall']):
                    sentiment = "🔴 [利空]"
                else:
                    sentiment = "⚪ [中性]"
                
                result = f"📰 {sentiment} **{title}** ({publisher})"
                _NEWS_CACHE[symbol] = (current_time, result)
                return result
    except Exception as e:
        logger.debug(f"[{symbol}] 新闻拉取失败: {e}")
        
    _NEWS_CACHE[symbol] = (current_time, "")
    return ""

def get_filtered_watchlist(max_stocks: int = 120) -> List[str]:
    logger.info(">>> 启动漏斗过滤：获取全市场基础名单 (S&P 500 + S&P 400 + Nasdaq 100)...")
    tickers = set(Config.CORE_WATCHLIST)
    
    try:
        url_sp500 = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables_sp500 = pd.read_html(url_sp500)
        for table in tables_sp500:
            if 'Symbol' in table.columns:
                sp500_tickers = table['Symbol'].dropna().astype(str).str.replace('.', '-').tolist()
                tickers.update(sp500_tickers)
                break
                
        url_sp400 = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        tables_sp400 = pd.read_html(url_sp400)
        for table in tables_sp400:
            col = 'Symbol' if 'Symbol' in table.columns else ('Ticker symbol' if 'Ticker symbol' in table.columns else None)
            if col:
                sp400_tickers = table[col].dropna().astype(str).str.replace('.', '-').tolist()
                tickers.update(sp400_tickers)
                break
        
        url_ndx = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables_ndx = pd.read_html(url_ndx, match='Ticker|Symbol')
        for table in tables_ndx:
            col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
            ndx_tickers = table[col].dropna().astype(str).str.replace('.', '-').tolist()
            tickers.update(ndx_tickers)
            
    except Exception as e:
        logger.warning(f"⚠️ 获取维基百科名单部分失败: {e}")
        
    tickers_list = list(tickers)
    logger.info(f"✅ 成功获取基础名单 {len(tickers_list)} 只股票(已自动去重)。开始分块获取近期数据进行漏斗粗筛...")
    
    try:
        chunk_size = 300
        dfs = []
        
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            logger.info(f"⏳ 正在拉取第 {i//chunk_size + 1} 批基础数据 (包含 {len(chunk)} 只股票)...")
            
            chunk_df = yf.download(chunk, period="5d", progress=False)
            if not chunk_df.empty:
                dfs.append(chunk_df)
                
            if i + chunk_size < len(tickers_list):
                time.sleep(random.uniform(2.0, 3.5))
                
        if not dfs:
            raise ValueError("所有分块批量下载均失败或返回空数据")
            
        df = pd.concat(dfs, axis=1)
        
        if df.empty or not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("分块拼接后的数据异常或结构非 MultiIndex")
            
        close_df = df['Close'].dropna(axis=1, how='all')
        volume_df = df['Volume'].dropna(axis=1, how='all')
        
        closes = close_df.ffill().iloc[-1]
        volumes = volume_df.mean()
        
        turnovers = (closes * volumes).dropna()
        valid_turnovers = turnovers[(closes > 10.0) & (volumes > 1000000)]
        
        top_tickers = valid_turnovers.sort_values(ascending=False).head(max_stocks).index.tolist()
        
        if top_tickers:
            logger.info(f"✅ 漏斗过滤完成！精选出全市场最活跃的 {len(top_tickers)} 只股票进入深度分析矩阵。")
            return top_tickers
        else:
            raise ValueError("过滤后满足条件的名单为空")
            
    except Exception as e:
        logger.error(f"❌ 批量拉取或漏斗过滤失败: {e}。启动降级安全方案。")
        return Config.CORE_WATCHLIST

def escape_md_v2(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def load_strategy_performance_tag() -> str:
    """[协同功能] 读取回测系统生成的最新胜率数据，作为推送战绩标签"""
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
                t3 = stats.get("T+3")
                if t3 and t3.get('total_trades', 0) > 0:
                    return f"📈 **策略历史表现 (T+3)**: 胜率 **{t3['win_rate']:.1%}** | 均收益 **{t3['avg_ret']:+.2%}**\n\n"
    except Exception as e:
        logger.debug(f"读取回测统计失败，可能是第一次运行还未生成: {e}")
    return ""

def send_alert(title: str, content: str) -> None:
    if not content.strip():
        logger.warning(f"⚠️ 警告内容为空，跳过推送：{title}")
        return

    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    if Config.WEBHOOK_URL:
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"【{Config.DINGTALK_KEYWORD}盯盘】{title}",
                "text": f"### 🤖 【{Config.DINGTALK_KEYWORD}量化监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {formatted_time}*"
            }
        }
        urls = [url.strip() for url in Config.WEBHOOK_URL.split(',') if url.strip()]
        for url in urls:
            try:
                resp = requests.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                logger.info(f"✅ 成功推送至 Webhook")
            except Exception as e:
                logger.error(f"Webhook 推送异常: {e}")
                
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        tg_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        safe_title = escape_md_v2(title)
        safe_content = escape_md_v2(content)
        safe_time = escape_md_v2(formatted_time)
        
        tg_text = f"🤖 *量化监控系统*\n\n*{safe_title}*\n\n{safe_content}\n\n⏱️ _{safe_time}_"
        
        tg_payload = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": tg_text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(tg_url, json=tg_payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"✅ 成功推送至 Telegram (MarkdownV2)")
        except Exception as e:
            logger.error(f"Telegram 推送异常: {e}")

# ================= 3. 大盘风控感知系统 (Market & Risk Regime) =================
def get_vix_level(qqq_df_for_shadow: pd.DataFrame = None) -> Tuple[float, str]:
    logger.info(">>> 正在获取 VIX 恐慌指数...")
    df = safe_get_history(Config.VIX_INDEX, period="5d", interval="1d", retries=3, auto_adjust=False, fast_mode=True)
    
    vix = 18.0
    is_simulated = False
    
    if not df.empty and len(df) >= 1:
        vix = df['Close'].ffill().iloc[-1]
    else:
        logger.warning("⚠️ 雅虎财经拒绝返回 ^VIX 数据，启动影子波动率合成引擎...")
        qqq_df = qqq_df_for_shadow if qqq_df_for_shadow is not None else safe_get_history(Config.INDEX_ETF, period="2mo", interval="1d", retries=2, auto_adjust=True, fast_mode=True)
        if qqq_df is not None and not qqq_df.empty and len(qqq_df) >= 20:
            daily_pct_change = qqq_df['Close'].pct_change().dropna()
            realized_volatility = daily_pct_change.rolling(window=20).std().iloc[-1]
            vix = realized_volatility * (252 ** 0.5) * 100 * 1.2
            is_simulated = True
        else:
            return 18.0, "⚠️ VIX 与大盘双双获取失败，启用中性防守模式 (默认 VIX=18.0)"

    prefix = "影子VIX" if is_simulated else "VIX"
    
    if vix > 30:
        return vix, f"🚨 极其恐慌 ({prefix}: {vix:.2f} > 30)，切勿盲目抄底，防守降权全开！"
    elif vix > 25:
        return vix, f"⚠️ 市场恐慌 ({prefix}: {vix:.2f} > 25)，系统已自动启动【防接飞刀】降权机制！"
    elif vix < 15:
        return vix, f"✅ 市场平静 ({prefix}: {vix:.2f} < 15)，技术信号可靠性较高。"
    else:
        return vix, f"⚖️ 正常波动 ({prefix}: {vix:.2f})"

def get_market_regime() -> Tuple[str, str, pd.DataFrame]:
    logger.info(f">>> 正在分析大盘环境 ({Config.INDEX_ETF})...")
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", auto_adjust=False, fast_mode=True)
    
    if len(df) < 200:
        return "range", "大盘数据不足，默认震荡市", df
        
    current_close = df['Close'].ffill().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (current_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if current_close > ma200 and trend_20d > 0.02:
        return "bull", "🐂 牛市主升阶段 (在200日均线之上且趋势向上)", df
    elif current_close < ma200 and trend_20d < -0.02:
        return "bear", "🐻 熊市回调阶段 (在200日均线之下且趋势向下)", df
    else:
        return "range", "⚖️ 震荡整理阶段 (缺乏明确的单边趋势)", df

def get_weekly_trend(symbol: str) -> str:
    df_week = safe_get_history(symbol, period="2y", interval="1wk", retries=2, fast_mode=True)
    if len(df_week) < 55: return "neutral"
        
    price = df_week['Close'].ffill().iloc[-1]
    ma50w = df_week['Close'].rolling(50).mean().iloc[-1]
    if pd.isna(ma50w): return "neutral"
        
    if price > ma50w: return "bullish"
    elif price < ma50w * 0.95: return "bearish"
    return "neutral"

# ================= 4. 核心量化指标模块 =================

def _add_rsi(df: pd.DataFrame) -> None:
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    if len(df) >= 60:
        df['RSI_P20'] = df['RSI'].rolling(window=120, min_periods=60).quantile(0.20)
    else:
        df['RSI_P20'] = 30.0

def _add_macd_and_emas(df: pd.DataFrame) -> None:
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line'] 

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

def _add_atr_and_supertrend(df: pd.DataFrame) -> None:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df['TR'] = tr
    df['ATR'] = tr.rolling(window=14).mean()
    
    # SuperTrend (10, 3)
    hl2 = (df['High'] + df['Low']) / 2
    atr10 = tr.rolling(window=10).mean()
    ub = (hl2 + 3 * atr10).values
    lb = (hl2 - 3 * atr10).values
    close = df['Close'].values
    
    in_uptrend = np.ones(len(df), dtype=bool)
    for i in range(1, len(df)):
        if pd.isna(ub[i-1]) or pd.isna(lb[i-1]):
            in_uptrend[i] = True
            continue
            
        if close[i] > ub[i-1]:
            in_uptrend[i] = True
        elif close[i] < lb[i-1]:
            in_uptrend[i] = False
        else:
            in_uptrend[i] = in_uptrend[i-1]
            
        if in_uptrend[i] and in_uptrend[i-1]:
            lb[i] = max(lb[i], lb[i-1])
        if not in_uptrend[i] and not in_uptrend[i-1]:
            ub[i] = min(ub[i], ub[i-1])
            
    df['SuperTrend_Up'] = in_uptrend.astype(int) 

def _add_channels(df: pd.DataFrame) -> None:
    df['ATR_20'] = df['TR'].rolling(window=20).mean()
        
    df['KC_Upper'] = df['EMA_20'] + 1.5 * df['ATR_20']
    df['KC_Lower'] = df['EMA_20'] - 1.5 * df['ATR_20']
    
    df['BB_MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA20'] + 2 * df['BB_STD20']
    df['BB_Lower'] = df['BB_MA20'] - 2 * df['BB_STD20']
    
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_MA20']

def _add_ichimoku(df: pd.DataFrame) -> None:
    high9 = df['High'].rolling(9).max()
    low9 = df['Low'].rolling(9).min()
    df['Tenkan'] = (high9 + low9) / 2
    
    high26 = df['High'].rolling(26).max()
    low26 = df['Low'].rolling(26).min()
    df['Kijun'] = (high26 + low26) / 2
    
    high52 = df['High'].rolling(52).max()
    low52 = df['Low'].rolling(52).min()
    
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((high52 + low52) / 2).shift(26)
    
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    df['Cloud_Twist'] = (df['SenkouA'] > df['SenkouB']).astype(int)

def _add_event_risk(df: pd.DataFrame) -> None:
    if len(df) < 60:
        df['Event_Risk'] = 0.0
        return
        
    recent_vol = df['Volume'].iloc[-10:].mean()
    hist_vol = df['Volume'].iloc[-60:-10].quantile(0.8)
    vol_spike = recent_vol / (hist_vol + 1e-10) if hist_vol > 0 else 1.0
        
    recent_changes = df['Close'].pct_change().iloc[-5:].abs()
    gap_or_spike = (recent_changes > 0.08).any()
    
    consecutive_high_vol = (df['Volume'].iloc[-5:] > hist_vol * 1.5).sum()
        
    risk_score = 0.0
    if vol_spike > 2.5:
        risk_score += 0.6
    if gap_or_spike:
        risk_score += 0.4
    if consecutive_high_vol >= 3:
        risk_score += 0.3
        
    prev_close = df['Close'].iloc[-2]
    gap_today = abs((df['Open'].iloc[-1] - prev_close) / prev_close) if prev_close != 0 else 0
    if gap_today > 0.06:
        risk_score += 0.35
        
    df['Event_Risk'] = min(1.0, risk_score)

def _add_institutional_factor(df: pd.DataFrame) -> None:
    hl_diff = df['High'] - df['Low']
    hl_diff = hl_diff.replace(0, 1e-10)
    
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_diff
    mfv = mfm * df['Volume']
    
    if len(df) >= 20:
        df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    else:
        df['CMF'] = 0.0

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['Close'] = df['Close'].ffill()
    df['Volume'] = df['Volume'].ffill()
    
    _add_rsi(df)
    _add_macd_and_emas(df)
    _add_atr_and_supertrend(df)
    _add_channels(df)
    _add_ichimoku(df)
    _add_event_risk(df) 
    _add_institutional_factor(df) 
    return df

# ================= 5. 业务策略模块 =================
def run_volatility_sentinel() -> None:
    logger.info(">>> 启动高频异动哨兵模式...")
    now_utc = datetime.now(timezone.utc)
    if not (13 <= now_utc.hour <= 21):
        logger.info(f"💤 非主要交易时段，跳过高频扫描。")
        return

    alerts = []
    for symbol in Config.CORE_WATCHLIST:
        if symbol in Config.BLACKLIST:
            continue
            
        try:
            df = safe_get_history(symbol, period="2d", interval="1h")
            if len(df) < 2: continue
            
            curr_price = df['Close'].ffill().iloc[-1]
            open_price = df['Open'].iloc[-1] 
            
            hour_change = (curr_price - open_price) / open_price * 100 if open_price != 0 else 0.0
            if abs(hour_change) > 3:
                alerts.append(f"> **{symbol}** 短线异动 {'🚀' if hour_change > 0 else '🩸'} \n> 1小时内波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            df_daily = safe_get_history(symbol, period="5d", interval="1d")
            if not df_daily.empty and len(df_daily) >= 2:
                gap_daily_pct = (df_daily['Open'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2] * 100
                if abs(gap_daily_pct) > 4:
                    alerts.append(f"> **{symbol}** 日线跳空 {'💥' if gap_daily_pct > 0 else '⚠️'}\n> 真实缺口: **{gap_daily_pct:+.2f}%**")
                    
        except Exception as e:
            logger.debug(f"[{symbol}] 哨兵模式分析跳过: {e}")

    if alerts:
        send_alert("⚡ 极端异动警告", "\n\n".join(alerts))

def run_tech_matrix() -> None:
    logger.info(">>> 启动复合技术指标诊断...")
    
    regime, regime_desc, qqq_df = get_market_regime()
    vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
    logger.info(vix_desc)
    logger.info(f"当前大盘状态: {regime_desc}")
    
    qqq_ret_20d, qqq_ret_5d = 0.0, 0.0
    if not qqq_df.empty and len(qqq_df) > 20:
        qqq_ret_20d = (qqq_df['Close'].ffill().iloc[-1] / qqq_df['Close'].iloc[-20]) - 1
        qqq_ret_5d = (qqq_df['Close'].ffill().iloc[-1] / qqq_df['Close'].iloc[-5]) - 1

    logger.info(">>> 预加载 Sector ETF 板块数据 (Fast Mode)...")
    sector_etfs = list(Config.SECTOR_MAP.keys())
    sector_data = {}
    for etf in sector_etfs:
        sdf = safe_get_history(etf, period="2mo", interval="1d", auto_adjust=False, retries=2, fast_mode=True)
        if not sdf.empty and len(sdf) >= 20:
            sector_data[etf] = (sdf['Close'].ffill().iloc[-1] / sdf['Close'].iloc[-20]) - 1

    avg_sector_strength = np.mean(list(sector_data.values())) if sector_data else 0.0
    health_score = 0.0
    
    if vix > 30:
        health_score = -0.9
    elif vix > 25:
        health_score = -0.6
    elif vix < 15 and regime == 'bull':
        health_score = 0.9
    elif regime == 'bull':
        health_score = 0.6
    elif regime == 'bear':
        health_score = -0.7
    else:
        sector_strength_capped = max(-0.2, min(0.2, float(avg_sector_strength)))
        health_score = 0.0 if sector_strength_capped < 0.01 else sector_strength_capped * 2

    health_score = max(-1.0, min(1.0, float(health_score)))

    weight_multiplier = 1.0 + health_score * 0.8
    min_score_dynamic = max(Config.MIN_SCORE_THRESHOLD, 8 + round(health_score * -6))
    
    logger.info(f"市场健康度: {health_score:.2f} | 权重倍增器: {weight_multiplier:.2f} | 动态门槛: {min_score_dynamic}")
            
    target_list = get_filtered_watchlist()
    total_stocks = len(target_list)
    reports = []
    
    for idx, symbol in enumerate(target_list):
        if symbol in Config.BLACKLIST:
            continue
            
        if idx % 10 == 0:
            logger.info(f"[进度 {idx}/{total_stocks}] 正在进行严密共振扫描...")
            
        try:
            df = safe_get_history(symbol, period="6mo", interval="1d")
            if len(df) < 100: continue
            if df['Volume'].tail(20).mean() < 1000000: continue

            df = calculate_indicators(df)
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            event_risk = curr.get('Event_Risk', 0.0)
            if event_risk > 0.85:
                logger.debug(f"[{symbol}] 检测到极高事件风险(量价突变)，防雷网直接跳过。")
                continue  
                
            signals = []
            score = 0
            
            atr_pct = curr['ATR'] / curr['Close']
            if atr_pct > 0.10: continue 

            is_strong_trend = curr.get('SuperTrend_Up', 0) == 1 
            dynamic_rsi_threshold = curr.get('RSI_P20', 30.0)
            if pd.isna(dynamic_rsi_threshold): dynamic_rsi_threshold = 30.0
            
            vol_ratio = curr['Volume'] / curr['Vol_MA20'] if curr['Vol_MA20'] > 0 else 1.0
            is_volume_confirmed = vol_ratio > 1.5

            strong_signals_count = 0

            rs_20, rs_5 = 1.0, 1.0
            if not qqq_df.empty:
                merged = pd.merge(df[['Close']], qqq_df[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_stock', '_qqq'))
                if len(merged) >= 20:
                    stock_ret_20d = (merged['Close_stock'].ffill().iloc[-1] / merged['Close_stock'].iloc[-20]) - 1
                    qqq_ret_20d_merged = (merged['Close_qqq'].ffill().iloc[-1] / merged['Close_qqq'].iloc[-20]) - 1
                    
                    denom_qqq = max(1 + qqq_ret_20d_merged, 0.5)
                    rs_20 = (1 + stock_ret_20d) / denom_qqq
                    
                    stock_ret_5d = (merged['Close_stock'].ffill().iloc[-1] / merged['Close_stock'].iloc[-5]) - 1
                    qqq_ret_5d_merged = (merged['Close_qqq'].ffill().iloc[-1] / merged['Close_qqq'].iloc[-5]) - 1
                    rs_5 = (1 + stock_ret_5d) / max(1 + qqq_ret_5d_merged, 0.5)
                    
                    target_sector = Config.get_sector_etf(symbol)
                    sector_ret_20d = sector_data.get(target_sector, qqq_ret_20d_merged)
                    
                    denom_sector = max(1 + sector_ret_20d, 0.5)
                    stock_sector_rs = (1 + stock_ret_20d) / denom_sector

                    if rs_20 > 1.08 or (rs_5 > 1.05 and rs_20 > 1.03):
                        signals.append(f"⚡ **[强相对强度]** 跑赢大盘 {rs_20-1:+.1%} (5日{rs_5-1:+.1%})")
                        score += 7 if is_volume_confirmed else 4
                        
                    if stock_sector_rs > 1.06 and target_sector != Config.INDEX_ETF:
                        signals.append(f"🏆 **[板块领涨]** 强势吸金，跑赢所在板块({target_sector}) {stock_sector_rs-1:.1%}")
                        score += 5
                        strong_signals_count += 1
                        
            if rs_20 < 0.92:
                signals.append(f"🐢 **[相对弱势]** 跑输大盘 {(1-rs_20):.1%}")
                score -= 3

            if curr['RSI'] < dynamic_rsi_threshold and prev['RSI'] >= dynamic_rsi_threshold:
                weight = 13 if regime == 'bear' else 8 
                if curr['Close'] > curr['EMA_50'] and is_strong_trend:
                    signals.append(f"🟢 **[强势回踩超卖]** 趋势+自适应买点 (RSI:{curr['RSI']:.1f})")
                    score += int(weight * weight_multiplier)
                    strong_signals_count += 1
                else:
                    signals.append(f"⚠️ **[弱势超卖]** 逆势防范接飞刀")
                    score += int(weight * 0.4)
            
            price_low_5 = df['Close'].iloc[-5:].min()
            idx_low_5 = df['Close'].iloc[-5:].idxmin()
            rsi_at_low_5 = df['RSI'].loc[idx_low_5]
            price_low_prev = df['Close'].iloc[-15:-5].min()
            idx_low_prev = df['Close'].iloc[-15:-5].idxmin()
            rsi_at_low_prev = df['RSI'].loc[idx_low_prev]
            
            if price_low_5 < price_low_prev and rsi_at_low_5 > rsi_at_low_prev and curr['RSI'] < 40:
                weight = 12 if regime == 'bear' else 8
                signals.append("🔍 **[RSI底背离]** 下跌动能衰竭")
                score += int(weight * weight_multiplier)
                strong_signals_count += 1
            
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                weight = 10 if regime == 'bull' else 6 
                if curr['MACD'] < 0:
                    signals.append("🔸 **[零下金叉]** 弱反弹预期")
                    score += int(weight * 0.6)
                else:
                    signals.append("🔥 **[零上金叉]** 主升浪高概率启动")
                    score += int(weight * weight_multiplier)
                    strong_signals_count += 1

            bb_abs = curr['BB_Upper'] - curr['BB_Lower']
            kc_abs = curr['KC_Upper'] - curr['KC_Lower']
            is_squeeze_on = bb_abs < kc_abs
            
            was_squeeze = False
            if len(df) >= 6:
                past_squeeze = (df['BB_Upper'].iloc[-6:-1] < df['KC_Upper'].iloc[-6:-1]) & \
                               (df['BB_Lower'].iloc[-6:-1] > df['KC_Lower'].iloc[-6:-1])
                was_squeeze = past_squeeze.any()
            
            macd_hist = curr['MACD_Hist']
            prev_macd_hist = prev['MACD_Hist']
            
            is_squeeze_fire_up = was_squeeze and (not is_squeeze_on) and (macd_hist > 0) and (macd_hist > prev_macd_hist)
            is_squeeze_fire_down = was_squeeze and (not is_squeeze_on) and (macd_hist < 0) and (macd_hist < prev_macd_hist)
            price_above_bb_upper = curr['Close'] > curr['BB_Upper']

            if is_squeeze_on:
                weight = 12 if regime == 'range' else 6
                signals.append("📦 **[TTM Squeeze ON]** 极致压缩，等待爆发")
                score += int(weight * weight_multiplier)
            elif is_squeeze_fire_up:
                weight = 18 if regime == 'bull' else 10
                if price_above_bb_upper and is_strong_trend:
                    signals.append("🚀 **[TTM Squeeze FIRE ↑]** 破位放量，动能强劲确认点火！")
                    score += int((weight + 4) * weight_multiplier) 
                    strong_signals_count += 1
                elif is_strong_trend:
                    signals.append("🔥 **[TTM Squeeze FIRE ↑]** 挤压向上释放 (趋势确认)")
                    score += int(weight * weight_multiplier)
                    strong_signals_count += 1
                else:
                    signals.append("🔥 **[TTM Squeeze FIRE ↑]** 挤压向上释放 (需防假突破)")
                    score += int(weight * 0.6)
            elif is_squeeze_fire_down:
                signals.append("🔻 **[TTM Squeeze FIRE ↓]** 空头动量释放，警惕下杀")
                score -= 8 

            if prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
                weight = 9 if regime == 'bull' else 5
                if is_strong_trend and is_volume_confirmed:
                    signals.append(f"📈 **[SuperTrend + 有效突破]** 携趋势与巨量站上50日均线")
                    score += int((weight + 2) * weight_multiplier)
                    strong_signals_count += 1
                else:
                    signals.append(f"📉 **[疲软突破]** 站上50日均线 (缺乏量能或大级别趋势配合)")
                    score -= 3 

            cmf = curr.get('CMF', 0.0)
            if cmf > 0.20:
                signals.append(f"🏦 **[机构高度控盘]** 资金呈强净流入态势 (CMF: {cmf:.2f})")
                score += int(5 * weight_multiplier)
                strong_signals_count += 1
            elif cmf > 0.05:
                signals.append(f"💰 **[资金温和吸筹]** 机构资金呈流入迹象 (CMF: {cmf:.2f})")
                score += 2
            elif cmf < -0.15:
                signals.append(f"⚠️ **[资金派发中]** 机构疑似撤退，动能不足 (CMF: {cmf:.2f})")
                score -= 4
                
            bb_width = curr.get('BB_Width', 1.0)
            if len(df) >= 60:
                past_width_min = df['BB_Width'].iloc[-60:-1].min()
                if bb_width < 0.06 or bb_width <= past_width_min * 1.1:
                    signals.append(f"🗜️ **[布林带极致收口]** 波动率降至冰点 (宽幅 {bb_width:.1%})，即将迎来大级别变盘！")
                    score += 4

            above_cloud = curr.get('Above_Cloud', 0) == 1
            cloud_twist = curr.get('Cloud_Twist', 0) == 1
            tenkan_above_kijun = curr.get('Tenkan', 0) > curr.get('Kijun', 0)
            
            if not above_cloud:
                score = int(score * 0.4)
                if strong_signals_count >= 1:
                    signals.append("☁️ **[云层压制]** 价格处于一目均衡云下方，长线趋势承压")
            else:
                if tenkan_above_kijun and cloud_twist:
                    signals.append("🌥️ **[一目均衡强多头]** 稳居云上 + 腾落线金叉，多头格局确立")
                    score += int(6 * weight_multiplier)
                    strong_signals_count += 1

            event_risk = curr.get('Event_Risk', 0.0)
            if event_risk > 0.85:
                score = int(score * 0.2)  
                signals.append("🚨 **[极高事件风险]** 发生极端量价突变，技术面随时失效！")
            elif event_risk > 0.5:
                score = int(score * (1 - event_risk))
                signals.append("⚠️ **[事件风险预警]** 近期量价异常(疑似事件窗口)，防守降权")

            if score > 0 and strong_signals_count < 1 and not is_volume_confirmed:
                score = int(score * 0.3) 

            if is_volume_confirmed and score >= 5:
                signals.append("🌊 **[量价共振]** 成交量显著放大(>1.5x)，确认资金介入")
                score += 3
                
            if strong_signals_count >= 2:
                signals.append("🎯 **[严密共振确认]** 多维度技术面与趋势共振达成！")
                score += 5

            if signals and score >= 15:
                weekly_trend = get_weekly_trend(symbol)
                if weekly_trend == "bullish":
                    signals.append("🌟 **[周线共振]** 长期趋势多头，胜率倍增")
                    score += 5 
                elif weekly_trend == "bearish":
                    signals.append("⚠️ **[周线逆风]** 长期趋势空头，严防诱多陷阱")
                    score -= 4 

            if signals:
                if score < min_score_dynamic:
                    continue

                news_context = get_latest_news(symbol)
                if news_context:
                    signals.append(news_context)

                if len(signals) > 8:
                    signals = signals[:8] + [f"*(...还有 {len(signals)-8} 个强趋势印证)*"]
                    
                turnover = curr['Close'] * curr['Volume']
                
                reports.append({
                    "symbol": symbol,
                    "score": score,
                    "signals": signals,
                    "curr_close": curr['Close'],
                    "turnover": turnover,
                    "sector": Config.get_sector_etf(symbol), 
                    "raw_score": score
                })
                
        except Exception as e:
            logger.debug(f"[{symbol}] 分析发生静默错误: {e}")

    if reports:
        from collections import defaultdict
        sector_groups = defaultdict(list)
        for r in reports:
            sector_groups[r["sector"]].append(r)
            
        for sector, stocks in sector_groups.items():
            sector_ret = sector_data.get(sector, 0.0)
            is_weak_sector = sector_ret < -0.02 and sector not in Config.CROWDING_EXCLUDE_SECTORS
            momentum_penalty = 0.7 if health_score < 0 else 0.85
            
            if is_weak_sector:
                for stock in stocks:
                    stock["score"] = int(stock["raw_score"] * momentum_penalty)
                    stock["is_weak_sector"] = True
                    stock["momentum_penalty"] = momentum_penalty
                    
            if sector not in Config.CROWDING_EXCLUDE_SECTORS and len(stocks) >= Config.CROWDING_MIN_STOCKS:
                stocks.sort(key=lambda x: x["score"], reverse=True)
                
                dynamic_penalty = Config.CROWDING_PENALTY * (1.0 + health_score * 0.3)
                dynamic_penalty = max(0.6, min(0.9, dynamic_penalty))
                
                for stock in stocks[1:]:
                    if not stock.get("is_weak_sector"):
                        stock["score"] = int(stock["raw_score"] * dynamic_penalty)
                        stock["is_crowded"] = True
                        stock["crowding_penalty"] = dynamic_penalty

        reports.sort(key=lambda x: x["score"], reverse=True)
        top_reports_text = []
        
        for r in reports[:15]:
            turnover_str = f"${r['turnover']/1e9:.2f}B" if r['turnover'] >= 1e9 else f"${r['turnover']/1e6:.2f}M"
            
            score_display = f"**{r['score']}**"
            if r.get("is_crowded"):
                score_display += f"（板块拥挤降权×{r.get('crowding_penalty', 1.0):.2f}）"
            elif r.get("is_weak_sector"):
                score_display += f"（弱势板块降权×{r.get('momentum_penalty', 1.0):.2f}）"
            
            text = f"**{r['symbol']}** (${r['curr_close']:.2f} | 额: {turnover_str} | 🌟动态评分: {score_display})\n> " + "\n> ".join(r["signals"])
            top_reports_text.append(text)
        
        perf_tag = load_strategy_performance_tag()
        
        final_report = f"{perf_tag}*{vix_desc}*\n*{regime_desc}*\n\n" + "\n\n".join(top_reports_text)
        
        if len(reports) > 15:
            final_report += f"\n\n*(已过滤低质信号 + 拥挤/弱势动态降权，为您优选展示最高分 Top 15)*"
        else:
            final_report += f"\n\n*(动态门槛: {min_score_dynamic} 分 | 权重因子: {weight_multiplier:.2f} | 已应用拥挤/弱势动态降权)*"
            
        send_alert("📊 全市场优选异动池", final_report)
        
        try:
            log_data = {
                "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                "vix": round(float(vix), 2),
                "regime": regime,
                "health_score": round(float(health_score), 2),
                "top_picks": [
                    {
                        "symbol": r["symbol"],
                        "score": r["score"],
                        "raw_score": r["raw_score"],
                        "sector": r["sector"],
                        "close": round(float(r["curr_close"]), 2),
                        "crowded": r.get("is_crowded", False),
                        "weak_sector": r.get("is_weak_sector", False)
                    } for r in reports[:15]
                ]
            }
            with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            logger.info(f"✅ 历史回测日志已追加至 {Config.LOG_FILE}")
        except Exception as e:
            logger.error(f"写入回测日志失败: {e}")
            
    else:
        logger.info(f"所有个股未通过 Ichimoku 趋势云过滤，或得分低于动态门槛 ({min_score_dynamic}分)。")

def run_daily_screener() -> None:
    logger.info(">>> 启动动态优选全量扫盘报告...")
    target_list = get_filtered_watchlist()
    bullish, bearish = [], []
    total_stocks = len(target_list)
    
    for idx, symbol in enumerate(target_list):
        if symbol in Config.BLACKLIST:
            continue
            
        if idx % 10 == 0:
            logger.info(f"[进度 {idx}/{total_stocks}] 正在进行每日全景复盘...")
        try:
            df = safe_get_history(symbol, period="6mo", interval="1d")
            if len(df) < 50: continue
            
            price = df['Close'].ffill().iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            score = (price > ma20) + (price > ma50) + (ma20 > ma50)
            if score == 3: bullish.append(symbol)
            elif score == 0: bearish.append(symbol)
        except Exception as e:
            logger.debug(f"[{symbol}] 每日复盘分析跳过: {e}")

    report = f"🎯 **今日全市场高优股票扫描总结**\n\n**📈 绝对多头排列 (强势)**\n> {', '.join(bullish) if bullish else '无'}\n\n**📉 绝对空头排列 (弱势)**\n> {', '.join(bearish) if bearish else '无'}"
    send_alert("📝 全市场优选股复盘", report)

# ================= 6. 终极回测引击模块 =================
def run_backtest_engine() -> None:
    """自动化评估历史推荐表现，并推送周报"""
    logger.info(">>> 启动终极历史绩效回测引擎...")
    if not os.path.exists(Config.LOG_FILE):
        logger.warning(f"⚠️ 找不到历史日志文件 {Config.LOG_FILE}，跳过回测。请先运行常规扫描积累数据。")
        return

    logs = []
    with open(Config.LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass

    trades = []
    for log in logs:
        log_date = log.get('date')
        for pick in log.get('top_picks', []):
            trades.append({
                'date': log_date,
                'symbol': pick['symbol'],
                'score': pick['score']
            })

    if not trades:
        logger.warning("历史记录中没有有效的选股数据。")
        return

    unique_symbols = list(set([t['symbol'] for t in trades]))
    earliest_date = min([t['date'] for t in trades])
    start_date = (datetime.strptime(earliest_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')

    logger.info(f"⏳ 正在向 Yahoo Finance 批量拉取 {len(unique_symbols)} 只曾上榜股票进行真实回测...")
    try:
        df_download = yf.download(unique_symbols, start=start_date, progress=False)
        df_close = df_download['Close'] if 'Close' in df_download else df_download
        if len(unique_symbols) == 1:
            df_close = pd.DataFrame(df_close)
            df_close.columns = unique_symbols
        df_close.index = df_close.index.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"❌ 拉取历史数据失败: {e}")
        return

    stats = {'T+1': [], 'T+3': [], 'T+5': []}
    logger.info("⚙️ 正在逐笔匹配交易日并计算 T+N 真实收益率...")
    for trade in trades:
        sym = trade['symbol']
        rec_date = trade['date']
        if sym not in df_close.columns:
            continue

        valid_dates = df_close.index[df_close.index >= rec_date]
        if len(valid_dates) == 0:
            continue

        entry_date = valid_dates[0]
        entry_idx = df_close.index.get_loc(entry_date)
        entry_price = df_close.at[entry_date, sym]

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        for t_days in [1, 3, 5]:
            exit_idx = entry_idx + t_days
            if exit_idx < len(df_close):
                exit_price = df_close.iloc[exit_idx][sym]
                if not pd.isna(exit_price):
                    ret = (exit_price - entry_price) / entry_price
                    stats[f'T+{t_days}'].append(ret)

    results = {}
    for period, returns in stats.items():
        if returns:
            win_count = sum(1 for r in returns if r > 0)
            results[period] = {
                'win_rate': win_count / len(returns),
                'avg_ret': sum(returns) / len(returns),
                'total_trades': len(returns)
            }

    if not results:
        logger.info("💤 暂无足够交易闭环数据进行评估。")
        return

    try:
        with open(Config.STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logger.error(f"写入回测统计 JSON 失败: {e}")

    alert_lines = []
    md_lines = [
        "# 📈 自动化量化监控战报 (Auto-Backtest Report)",
        f"**最后更新时间:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "| 持仓周期 | 胜率 (Win Rate) | 平均收益 (Avg Return) | 样本交易笔数 |",
        "| :---: | :---: | :---: | :---: |"
    ]

    for period in ['T+1', 'T+3', 'T+5']:
        if period in results:
            data = results[period]
            wr_str = f"{data['win_rate']*100:.1f}%"
            ar_str = f"{data['avg_ret']*100:+.2f}%"
            trds = data['total_trades']
            md_lines.append(f"| **{period}** | {wr_str} | {ar_str} | {trds} |")
            alert_lines.append(f"> ⏱️ **{period} 持仓**: 胜率 **{wr_str}** | 均收益 **{ar_str}** (样本: {trds}笔)")
        else:
            md_lines.append(f"| **{period}** | N/A | N/A | 0 |")

    try:
        with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
    except Exception as e:
        logger.error(f"写入 Markdown 失败: {e}")

    alert_text = "\n\n".join(alert_lines) + "\n\n*(注：数据已同步至 GitHub 代码库，回测包含所有往期历史表现，未剔除滑点摩擦)*"
    send_alert("📅 终极策略历史回测周报", alert_text)
    logger.info("====== ✅ 回测评估与周报推送完成 ======")

# ================= 7. 统一入口 =================
if __name__ == "__main__":
    validate_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    logger.info(f"====== 引擎启动 | 模式: [{mode.upper()}] ======")
    
    if mode == "sentinel":
        run_volatility_sentinel()
    elif mode == "matrix":
        run_tech_matrix()
    elif mode == "daily":
        run_daily_screener()
    elif mode == "backtest":
        run_backtest_engine()
    elif mode == "test":
        regime, desc, qqq_df = get_market_regime()
        vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
        test_tickers = get_filtered_watchlist(max_stocks=5)
        send_alert(
            "✅ Pro 量化引擎部署成功", 
            f"已集成 [JIT新闻] + [漏斗过滤] + [回测闭环]！\n{vix_desc}\n大盘: **{desc}**\n当前测试名单: {', '.join(test_tickers)}"
        )
    else:
        logger.error(f"未知的模式: {mode}")
