import yfinance as yf
import requests
import os
import sys
import pandas as pd
import time
import random
import logging
from typing import List, Optional
from datetime import datetime, timezone

# ================= 1. 日志与配置管理 (Config & Logging) =================
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantBot")

class Config:
    """全局配置类"""
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    DINGTALK_KEYWORD: str = "AI"
    CORE_WATCHLIST: List[str] = ["NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]

# ================= 2. 数据与网络工具模块 (Data & Network) =================
def safe_get_history(symbol: str, period: str = "6mo", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    """带重试和随机延迟的安全网络请求，防止被限流"""
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.3, 0.8))
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"[{symbol}] 获取数据最终失败: {e}")
                return pd.DataFrame()
            time.sleep(2)
    return pd.DataFrame()

def get_nasdaq_100() -> List[str]:
    """实时从维基百科抓取最新的纳斯达克100成分股"""
    logger.info(">>> 正在联网获取最新 纳斯达克 100 成分股名单...")
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=15)
        
        tables = pd.read_html(response.text, match='Ticker')
        df = tables[0]
        tickers = df['Ticker'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # 修复雅虎财经格式
        
        logger.info(f"✅ 成功获取 {len(tickers)} 只纳斯达克 100 股票！")
        return tickers
    except Exception as e:
        logger.error(f"❌ 获取名单失败，使用备用核心名单。错误: {e}")
        return Config.CORE_WATCHLIST

def send_dingtalk(title: str, content: str) -> None:
    """发送格式化的钉钉 Markdown 消息"""
    if not Config.WEBHOOK_URL: 
        logger.error("WEBHOOK_URL 环境变量未设置，跳过消息推送")
        return

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": f"【{Config.DINGTALK_KEYWORD}盯盘】{title}",
            "text": f"### 🤖 【{Config.DINGTALK_KEYWORD}量化监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        }
    }
    try:
        requests.post(Config.WEBHOOK_URL, json=payload, timeout=10)
        logger.info(f"成功推送消息: {title}")
    except Exception as e:
        logger.error(f"推送请求发生异常: {e}")

# ================= 3. 核心量化指标模块 (Indicators) =================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有的技术指标"""
    # Wilder 平滑版 RSI (14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # EMA 均线
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # 成交量与 ATR
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # 布林带 (20, 2)
    df['BB_MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Width'] = ((df['BB_MA20'] + 2 * df['BB_STD20']) - (df['BB_MA20'] - 2 * df['BB_STD20'])) / df['BB_MA20']
    
    return df

# ================= 4. 业务策略模块 (Strategies) =================
def run_volatility_sentinel() -> None:
    logger.info(">>> 启动高频异动哨兵模式...")
    now_utc = datetime.now(timezone.utc)
    if not (13 <= now_utc.hour <= 21):
        logger.info(f"💤 当前时间 (UTC {now_utc.hour}:{now_utc.minute}) 处于非主要交易时段，跳过高频扫描。")
        return

    alerts = []
    for symbol in Config.CORE_WATCHLIST:
        try:
            df = safe_get_history(symbol, period="2d", interval="1h")
            if len(df) < 2: continue
            
            curr_price = df['Close'].iloc[-1]
            open_price = df['Open'].iloc[-1] 
            
            # 短线急涨急跌 (小时线)
            hour_change = (curr_price - open_price) / open_price * 100
            if abs(hour_change) > 3:
                alerts.append(f"> **{symbol}** 短线异动 {'🚀' if hour_change > 0 else '🩸'} \n> 1小时内波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            # 日线跳空
            df_daily = safe_get_history(symbol, period="5d", interval="1d")
            if df_daily is not None and len(df_daily) >= 2:
                gap_daily = (df_daily['Open'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2] * 100
                if abs(gap_daily) > 4:
                    alerts.append(f"> **{symbol}** 日线跳空 {'💥' if gap_daily > 0 else '⚠️'}\n> 真实缺口: **{gap_daily:+.2f}%**")
                    
        except Exception as e:
            logger.warning(f"处理 {symbol} 哨兵模式时跳过: {e}")

    if alerts:
        send_dingtalk("⚡ 极端异动警告", "\n\n".join(alerts))
    else:
        logger.info("未发现极端异动。")

def run_tech_matrix() -> None:
    logger.info(">>> 启动复合技术指标诊断...")
    target_list = get_nasdaq_100()
    total_stocks = len(target_list)
    reports = []
    
    for idx, symbol in enumerate(target_list):
        logger.info(f"[进度 {idx+1}/{total_stocks}] 正在分析 {symbol}...")
        try:
            df = safe_get_history(symbol, period="6mo", interval="1d")
            if len(df) < 100: continue

            # 流动性过滤
            if df['Volume'].tail(20).mean() < 1000000:
                continue

            df = calculate_indicators(df)
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            signals = []
            score = 0
            
            # --- RSI 动态阈值 ---
            atr_pct = curr['ATR'] / curr['Close']
            dynamic_rsi_threshold = max(20, min(40, 30 - (atr_pct - 0.03) * 100))
            
            if curr['RSI'] < dynamic_rsi_threshold and prev['RSI'] >= dynamic_rsi_threshold:
                if curr['Close'] > curr['EMA_50']:
                    signals.append(f"🟢 **[趋势回踩超卖]** 顺势买点 (阈值:{dynamic_rsi_threshold:.1f}, RSI:{curr['RSI']:.1f})")
                    score += 10
                else:
                    signals.append(f"⚠️ **[弱势超卖]** 防范接飞刀 (阈值:{dynamic_rsi_threshold:.1f}, RSI:{curr['RSI']:.1f})")
                    score += 3
            
            # --- MACD ---
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                if curr['MACD'] < 0:
                    signals.append("🔸 **[零下金叉]** 弱反弹预期，注意见好就收")
                    score += 4
                else:
                    signals.append("🔥 **[零上金叉]** 强势主升浪确认")
                    score += 8

            # --- 布林带 ---
            avg_bb_width = df['BB_Width'].rolling(100).mean().iloc[-1]
            if pd.notna(avg_bb_width) and curr['BB_Width'] < avg_bb_width * 0.5:
                signals.append("📦 **[布林带挤压]** 波动率降至冰点，即将变盘")
                score += 6

            # --- 均线突破 ---
            if prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
                signals.append(f"🚀 **[均线突破]** 强势站上50日均线 (${curr['EMA_50']:.2f})")
                score += 7

            if signals:
                turnover = curr['Close'] * curr['Volume']
                turnover_str = f"${turnover/1e9:.2f}B" if turnover >= 1e9 else f"${turnover/1e6:.2f}M"
                    
                report_text = f"**{symbol}** (${curr['Close']:.2f} | 额: {turnover_str} | 🌟得分: **{score}**)\n> " + "\n> ".join(signals)
                reports.append({"symbol": symbol, "score": score, "text": report_text})
                
        except Exception as e:
            logger.error(f"[{symbol}] 分析发生错误: {e}")

    logger.info(f"扫描完成！共提取 {len(reports)} 个有效标的。")

    if reports:
        reports.sort(key=lambda x: x['score'], reverse=True)
        top_reports_text = [r['text'] for r in reports[:15]]
        
        final_report = "\n\n".join(top_reports_text)
        if len(reports) > 15:
            final_report += f"\n\n*(已为您过滤并展示得分最高的 Top 15 机会)*"
        send_dingtalk("📊 纳指 100 优选异动池", final_report)
    else:
        logger.info("未触发核心指标策略。")

def run_daily_screener() -> None:
    logger.info(">>> 启动纳指 100 全量扫盘报告...")
    target_list = get_nasdaq_100()
    bullish, bearish = [], []
    total_stocks = len(target_list)
    
    for idx, symbol in enumerate(target_list):
        if idx % 10 == 0:
            logger.info(f"[进度 {idx}/{total_stocks}] 正在评估趋势...")
        try:
            df = safe_get_history(symbol, period="6mo", interval="1d")
            if len(df) < 50: continue
            
            price = df['Close'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            score = (price > ma20) + (price > ma50) + (ma20 > ma50)
            if score == 3: bullish.append(symbol)
            elif score == 0: bearish.append(symbol)
        except Exception:
            pass

    report = f"🎯 **今日纳斯达克 100 扫描总结**\n\n**📈 绝对多头排列 (强势)**\n> {', '.join(bullish) if bullish else '无'}\n\n**📉 绝对空头排列 (弱势)**\n> {', '.join(bearish) if bearish else '无'}"
    send_dingtalk("📝 纳指 100 全景复盘", report)

# ================= 5. 入口模块 (Entrypoint) =================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    logger.info(f"====== 引擎启动 | 模式: [{mode.upper()}] ======")
    
    if mode == "sentinel":
        run_volatility_sentinel()
    elif mode == "matrix":
        run_tech_matrix()
    elif mode == "daily":
        run_daily_screener()
    elif mode == "test":
        test_tickers = get_nasdaq_100()[:5]
        send_dingtalk("✅ Pro 量化引擎部署成功", f"企业级规范架构已启用！\n当前测试获取的前五个代码为: {', '.join(test_tickers)}")
    else:
        logger.error(f"未知的模式: {mode}")
