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
from typing import List, Tuple
from datetime import datetime, timezone

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
        'XLY': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM', 'IDXX', 'MRNA', 'ALGN', 'BMRN', 'GEHC'], # 原代码这里XLV和XLY分类有混淆，这里保留你提供的原样
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA', 'TTWO', 'WBD', 'SIRI', 'CHTR'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM', 'IDXX', 'MRNA', 'ALGN', 'BMRN', 'GEHC'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'MNST', 'WBA']
    }

    # 新增：板块拥挤度过滤强度（0.6~0.9 之间，数值越小惩罚越狠）
    CROWDING_PENALTY: float = 0.75
    # 触发拥挤过滤的最低股票数量（≥2 即认为拥挤）
    CROWDING_MIN_STOCKS: int = 2

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

# [新增优化] 全局新闻缓存，彻底避免短时间内重复拉取导致限流
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
    """轻量级实时新闻上下文拉取器（带30分钟防限流缓存）"""
    current_time = time.time()
    
    # [新增优化] 检查内存级缓存
    if symbol in _NEWS_CACHE:
        cached_time, cached_news = _NEWS_CACHE[symbol]
        if current_time - cached_time < 1800: # 30分钟内复用
            return cached_news

    try:
        news_data = yf.Ticker(symbol).news
        if news_data and len(news_data) > 0:
            latest_news = news_data[0]
            title = latest_news.get('title', '')
            publisher = latest_news.get('publisher', '')
            if title:
                result = f"📰 **[最新动态]** {title} ({publisher})"
                _NEWS_CACHE[symbol] = (current_time, result)
                return result
    except Exception as e:
        logger.debug(f"[{symbol}] 新闻拉取失败: {e}")
        
    # 获取失败也写入缓存空值，防止循环请求
    _NEWS_CACHE[symbol] = (current_time, "")
    return ""

def get_nasdaq_100() -> List[str]:
    logger.info(">>> 正在联网获取最新 纳斯达克 100 成分股名单...")
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; QuantBot/2.0)'}
        response = requests.get(url, headers=headers, timeout=20)
        
        tables = pd.read_html(response.text, match='Ticker|Symbol')
        
        for table in tables:
            if ('Ticker' in table.columns or 'Symbol' in table.columns) and len(table) > 90:
                col_name = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                tickers = [str(t).replace('.', '-').strip() for t in table[col_name].dropna() if len(str(t).strip()) <= 6]
                if len(tickers) > 90:
                    logger.info(f"✅ 成功获取 {len(tickers)} 只纳斯达克 100 股票！")
                    return tickers
                    
        logger.error("❌ 页面中未找到超过90行的有效成分股表格。")
        return Config.CORE_WATCHLIST
    except Exception as e:
        logger.error(f"❌ 获取名单解析失败，使用备用核心名单: {e}")
        return Config.CORE_WATCHLIST

def escape_md_v2(text: str) -> str:
    """[新增优化] 针对 Telegram MarkdownV2 格式，过滤容易导致解析失败的非排版字符"""
    # 避开了排版常用的 * _ > [ ]
    escape_chars = r"()-+={}.!|"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_alert(title: str, content: str) -> None:
    """多渠道广播中心 (Webhook + Telegram)"""
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    # 1. 钉钉/飞书/企微 Webhook 推送
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
                
    # 2. Telegram Bot 推送
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        tg_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        raw_text = f"🤖 *量化监控系统*\n\n*{title}*\n\n{content}\n\n⏱️ _{formatted_time}_"
        # 保护性转义，防止部分数值和符号（如 -, ., !）引发 Telegram 解析错误
        safe_text = escape_md_v2(raw_text)
        
        tg_payload = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": safe_text,
            "parse_mode": "MarkdownV2",  # [新增优化] 升级到更安全的 V2 模式
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(tg_url, json=tg_payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"✅ 成功推送至 Telegram")
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
    if 'TR' in df:
        df['ATR_20'] = df['TR'].rolling(window=20).mean()
    else:
        df['ATR_20'] = df['ATR'].rolling(window=20).mean()
        
    df['KC_Upper'] = df['EMA_20'] + 1.5 * df['ATR_20']
    df['KC_Lower'] = df['EMA_20'] - 1.5 * df['ATR_20']
    
    df['BB_MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA20'] + 2 * df['BB_STD20']
    df['BB_Lower'] = df['BB_MA20'] - 2 * df['BB_STD20']

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
        
    # --- [新增优化] 价格跳空确认 ---
    gap_today = abs((df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2])
    if gap_today > 0.06:
        risk_score += 0.35
        
    df['Event_Risk'] = min(1.0, risk_score)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['Close'] = df['Close'].ffill()
    df['Volume'] = df['Volume'].ffill()
    
    _add_rsi(df)
    _add_macd_and_emas(df)
    _add_atr_and_supertrend(df)
    _add_channels(df)
    _add_ichimoku(df)
    _add_event_risk(df) 
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
            
            hour_change = (curr_price - open_price) / open_price * 100
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

    # === 市场健康自适应权重引擎 ===
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
    else:  # range 震荡市
        sector_strength_capped = max(-0.2, min(0.2, float(avg_sector_strength)))
        health_score = 0.0 if sector_strength_capped < 0.01 else sector_strength_capped * 2

    health_score = max(-1.0, min(1.0, float(health_score)))

    weight_multiplier = 1.0 + health_score * 0.8
    min_score_dynamic = max(Config.MIN_SCORE_THRESHOLD, 8 + round(health_score * -6))
    
    logger.info(f"市场健康度: {health_score:.2f} | 权重倍增器: {weight_multiplier:.2f} | 动态门槛: {min_score_dynamic}")
            
    target_list = get_nasdaq_100()
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
            
            # --- 事件风险防雷网 (The Event Risk Filter) ---
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

            # --- 相对强度 (RS) 包含板块轮动比较 ---
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

            # --- 严格共振引擎 (Adaptive Weights) ---
            
            # 1. 自适应 RSI 超卖策略
            if curr['RSI'] < dynamic_rsi_threshold and prev['RSI'] >= dynamic_rsi_threshold:
                weight = 13 if regime == 'bear' else 8 
                if curr['Close'] > curr['EMA_50'] and is_strong_trend:
                    signals.append(f"🟢 **[强势回踩超卖]** 趋势+自适应买点 (RSI:{curr['RSI']:.1f})")
                    score += int(weight * weight_multiplier)
                    strong_signals_count += 1
                else:
                    signals.append(f"⚠️ **[弱势超卖]** 逆势防范接飞刀")
                    score += int(weight * 0.4)
            
            # 2. RSI 底背离
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
            
            # 3. MACD 策略
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                weight = 10 if regime == 'bull' else 6 
                if curr['MACD'] < 0:
                    signals.append("🔸 **[零下金叉]** 弱反弹预期")
                    score += int(weight * 0.6)
                else:
                    signals.append("🔥 **[零上金叉]** 主升浪高概率启动")
                    score += int(weight * weight_multiplier)
                    strong_signals_count += 1

            # 4. TTM Squeeze 原汁原味破位点火版
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

            # 5. 均线突破
            if prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
                weight = 9 if regime == 'bull' else 5
                if is_strong_trend and is_volume_confirmed:
                    signals.append(f"📈 **[SuperTrend + 有效突破]** 携趋势与巨量站上50日均线")
                    score += int((weight + 2) * weight_multiplier)
                    strong_signals_count += 1
                else:
                    signals.append(f"📉 **[疲软突破]** 站上50日均线 (缺乏量能或大级别趋势配合)")
                    score -= 3 

            # --- 全局趋势阀门 (ICHIMOKU Master Trend Filter) ---
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

            # --- 事件风险惩罚 (财报/突发异动防雷网应用) ---
            event_risk = curr.get('Event_Risk', 0.0)
            if event_risk > 0.85:
                score = int(score * 0.2)  
                signals.append("🚨 **[极高事件风险]** 发生极端量价突变，技术面随时失效！")
            elif event_risk > 0.5:
                score = int(score * (1 - event_risk))
                signals.append("⚠️ **[事件风险预警]** 近期量价异常(疑似事件窗口)，防守降权")

            # === 全局共振噪音过滤器 ===
            if score > 0 and strong_signals_count < 1 and not is_volume_confirmed:
                score = int(score * 0.3) 

            if is_volume_confirmed and score >= 5:
                signals.append("🌊 **[量价共振]** 成交量显著放大(>1.5x)，确认资金介入")
                score += 3
                
            if strong_signals_count >= 2:
                signals.append("🎯 **[严密共振确认]** 多维度技术面与趋势共振达成！")
                score += 5

            # --- 周线共振 (高门槛懒加载) ---
            if signals and score >= 15:
                weekly_trend = get_weekly_trend(symbol)
                if weekly_trend == "bullish":
                    signals.append("🌟 **[周线共振]** 长期趋势多头，胜率倍增")
                    score += 5 
                elif weekly_trend == "bearish":
                    signals.append("⚠️ **[周线逆风]** 长期趋势空头，严防诱多陷阱")
                    score -= 4 

            # --- 推送阈值验证与即时新闻加载 (JIT News Context) ---
            if signals:
                if score < min_score_dynamic:
                    continue

                news_context = get_latest_news(symbol)
                if news_context:
                    signals.append(news_context)

                if len(signals) > 8:
                    signals = signals[:8] + [f"*(...还有 {len(signals)-8} 个强趋势印证)*"]
                    
                turnover = curr['Close'] * curr['Volume']
                turnover_str = f"${turnover/1e9:.2f}B" if turnover >= 1e9 else f"${turnover/1e6:.2f}M"
                    
                report_text = f"**{symbol}** (${curr['Close']:.2f} | 额: {turnover_str} | 🌟动态评分: **{score}**)\n> " + "\n> ".join(signals)
                
                # === 新增：板块拥挤度过滤（Crowding Filter）===
                # 只对最终要推送的股票进行后处理，这里动态获取 sector 以防未定义
                reports.append({
                    "symbol": symbol,
                    "score": score,
                    "text": report_text,
                    "sector": Config.get_sector_etf(symbol), 
                    "raw_score": score
                })
                
        except Exception as e:
            logger.debug(f"[{symbol}] 分析发生静默错误: {e}")

    # ====================== 板块拥挤度后处理（核心升级）======================
    if reports:
        from collections import defaultdict
        sector_groups = defaultdict(list)
        for r in reports:
            sector_groups[r["sector"]].append(r)
            
        # 对每个板块进行拥挤惩罚
        for sector, stocks in sector_groups.items():
            if len(stocks) < Config.CROWDING_MIN_STOCKS:
                continue  # 板块内只有 1 只，不惩罚
                
            # 找出本板块内得分最高的领涨股
            stocks.sort(key=lambda x: x["raw_score"], reverse=True)
            leader_score = stocks[0]["raw_score"]
            
            # 对非领涨股统一施加惩罚
            for stock in stocks[1:]:
                stock["score"] = int(stock["raw_score"] * Config.CROWDING_PENALTY)
                
                # 更新显示文本，标注拥挤降权 (使用精准替换防止破坏原生排版)
                old_score_str = f"🌟动态评分: **{stock['raw_score']}**"
                new_score_str = f"🌟动态评分: **{stock['score']}**（板块拥挤降权）"
                stock["text"] = stock["text"].replace(old_score_str, new_score_str)

        # 重新按最终得分排序
        reports.sort(key=lambda x: x["score"], reverse=True)
        top_reports_text = [r["text"] for r in reports[:15]]
        
        final_report = f"*{vix_desc}*\n*{regime_desc}*\n\n" + "\n\n".join(top_reports_text)
        if len(reports) > 15:
            final_report += f"\n\n*(已过滤低质信号 + 板块拥挤降权，为您优选展示最高分 Top 15)*"
        else:
            final_report += f"\n\n*(动态门槛: {min_score_dynamic} 分 | 权重因子: {weight_multiplier:.2f} | 已应用板块拥挤过滤)*"
            
        send_alert("📊 纳指 100 优选异动池", final_report)
    else:
        logger.info(f"所有个股未通过 Ichimoku 趋势云过滤，或得分低于动态门槛 ({min_score_dynamic}分)。")

def run_daily_screener() -> None:
    logger.info(">>> 启动纳指 100 全量扫盘报告...")
    target_list = get_nasdaq_100()
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

    report = f"🎯 **今日纳斯达克 100 扫描总结**\n\n**📈 绝对多头排列 (强势)**\n> {', '.join(bullish) if bullish else '无'}\n\n**📉 绝对空头排列 (弱势)**\n> {', '.join(bearish) if bearish else '无'}"
    send_alert("📝 纳指 100 全景复盘", report)

# ================= 6. 入口模块 =================
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
    elif mode == "test":
        regime, desc, qqq_df = get_market_regime()
        vix, vix_desc = get_vix_level(qqq_df_for_shadow=qqq_df)
        test_tickers = get_nasdaq_100()[:5]
        send_alert(
            "✅ Pro 量化引擎部署成功", 
            f"已集成 [JIT 实时新闻解析] 与 [Telegram 多渠道推送]！\n{vix_desc}\n大盘: **{desc}**\n当前测试名单: {', '.join(test_tickers)}"
        )
    else:
        logger.error(f"未知的模式: {mode}")
