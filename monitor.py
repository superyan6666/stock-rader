import yfinance as yf
import requests
import os
import sys
import pandas as pd
import time
import random
import logging
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
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    DINGTALK_KEYWORD: str = "AI"
    CORE_WATCHLIST: List[str] = ["NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]
    BLACKLIST: List[str] = [] # 黑名单：遇到即将发财报或退市风险的股票可填入此处跳过扫描
    INDEX_ETF: str = "QQQ" # 纳指100 ETF
    VIX_INDEX: str = "^VIX" # 恐慌指数

def validate_config():
    if not Config.WEBHOOK_URL:
        logger.error("❌ WEBHOOK_URL 未配置，系统无法推送消息。请检查 GitHub Secrets 环境。")
        sys.exit(1)
    logger.info("✅ 环境变量与配置校验通过")

# ================= 2. 数据与网络工具模块 =================
def safe_get_history(symbol: str, period: str = "6mo", interval: str = "1d", retries: int = 4, auto_adjust: bool = True) -> pd.DataFrame:
    """带指数退避和动态随机延迟的安全网络请求"""
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(1.5, 3.0) if "1d" in interval else random.uniform(0.8, 1.5)
            time.sleep(sleep_sec)
            
            # 改进：允许关闭 auto_adjust，防止指数和 ETF 报价失真
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=auto_adjust)
            if not df.empty:
                return df
                
        except Exception as e:
            logger.warning(f"[{symbol}] 第 {attempt+1} 次获取数据失败: {e}")
            if attempt == retries - 1:
                logger.error(f"[{symbol}] 最终获取失败，已放弃")
                return pd.DataFrame()
            time.sleep(3 + attempt * 2)
            
    return pd.DataFrame()

def get_nasdaq_100() -> List[str]:
    logger.info(">>> 正在联网获取最新 纳斯达克 100 成分股名单...")
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=15)
        
        # 改进：增加正则容错，防止维基百科表头在 Ticker 和 Symbol 之间变动
        tables = pd.read_html(response.text, match='Ticker|Symbol')
        df = tables[0]
        col_name = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        tickers = df[col_name].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        logger.info(f"✅ 成功获取 {len(tickers)} 只纳斯达克 100 股票！")
        return tickers
    except Exception as e:
        logger.error(f"❌ 获取名单失败，使用备用核心名单: {e}")
        return Config.CORE_WATCHLIST

def send_dingtalk(title: str, content: str) -> None:
    if not Config.WEBHOOK_URL: return

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": f"【{Config.DINGTALK_KEYWORD}盯盘】{title}",
            "text": f"### 🤖 【{Config.DINGTALK_KEYWORD}量化监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        }
    }
    
    urls = [url.strip() for url in Config.WEBHOOK_URL.split(',') if url.strip()]
    for url in urls:
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"推送请求发生异常: {e}")

# ================= 3. 大盘风控感知系统 (Market & Risk Regime) =================
def get_vix_level() -> Tuple[float, str]:
    """获取 VIX 恐慌指数，用于全局风控"""
    logger.info(">>> 正在获取 VIX 恐慌指数...")
    df = safe_get_history(Config.VIX_INDEX, period="1mo", interval="1d", retries=2, auto_adjust=False)
    if len(df) < 2:
        # 改进：数据缺失时，默认 VIX=20 (防守模式)，避免误判为平静
        return 20.0, "⚠️ VIX 数据暂缺，启用保守防守模式 (默认 VIX=20)"
    
    vix = df['Close'].iloc[-1]
    if vix > 25:
        return vix, f"⚠️ 极度恐慌 (VIX: {vix:.2f} > 25)，系统已自动启动【防接飞刀】全局降权机制！"
    elif vix < 15:
        return vix, f"✅ 市场平静 (VIX: {vix:.2f} < 15)，技术信号可靠性较高。"
    else:
        return vix, f"⚖️ 正常波动 (VIX: {vix:.2f})"

def get_market_regime() -> Tuple[str, str, pd.DataFrame]:
    """判断大盘环境，并返回 QQQ 数据用于后续相对强度(RS)计算"""
    logger.info(f">>> 正在分析大盘环境 ({Config.INDEX_ETF})...")
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", auto_adjust=False)
    
    if len(df) < 200:
        return "range", "大盘数据不足，默认震荡市", df
        
    current_close = df['Close'].iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (current_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if current_close > ma200 and trend_20d > 0.02:
        return "bull", "🐂 牛市主升阶段 (在200日均线之上且趋势向上)", df
    elif current_close < ma200 and trend_20d < -0.02:
        return "bear", "🐻 熊市回调阶段 (在200日均线之下且趋势向下)", df
    else:
        return "range", "⚖️ 震荡整理阶段 (缺乏明确的单边趋势)", df

def get_weekly_trend(symbol: str) -> str:
    """懒加载获取周线，做多时间框架过滤"""
    # 改进：修复 50w MA 需要足够数据的 Bug，将 period 改为 "2y" (约104根周线)
    df_week = safe_get_history(symbol, period="2y", interval="1wk", retries=2)
    if len(df_week) < 55: return "neutral"
        
    price = df_week['Close'].iloc[-1]
    ma50w = df_week['Close'].rolling(50).mean().iloc[-1]
    if pd.isna(ma50w): return "neutral"
        
    if price > ma50w: return "bullish"
    elif price < ma50w * 0.95: return "bearish"
    return "neutral"

# ================= 4. 核心量化指标模块 =================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # 动态阈值需要用到 14 日 ATR
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Keltner Channel (肯特纳通道) 辅助计算 - 采用 20 日均线与 20 日 ATR
    df['ATR_20'] = tr.rolling(window=20).mean()
    df['KC_Upper'] = df['EMA_20'] + 1.5 * df['ATR_20']
    df['KC_Lower'] = df['EMA_20'] - 1.5 * df['ATR_20']
    
    # 布林带计算
    df['BB_MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA20'] + 2 * df['BB_STD20']
    df['BB_Lower'] = df['BB_MA20'] - 2 * df['BB_STD20']
    
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
            
            curr_price = df['Close'].iloc[-1]
            open_price = df['Open'].iloc[-1] 
            
            hour_change = (curr_price - open_price) / open_price * 100
            if abs(hour_change) > 3:
                alerts.append(f"> **{symbol}** 短线异动 {'🚀' if hour_change > 0 else '🩸'} \n> 1小时内波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            df_daily = safe_get_history(symbol, period="5d", interval="1d")
            if df_daily is not None and len(df_daily) >= 2:
                gap_daily_pct = (df_daily['Open'].iloc[-1] - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2] * 100
                if abs(gap_daily_pct) > 4:
                    alerts.append(f"> **{symbol}** 日线跳空 {'💥' if gap_daily_pct > 0 else '⚠️'}\n> 真实缺口: **{gap_daily_pct:+.2f}%**")
                    
        except Exception as e:
            # 改进：记录异常但静默运行，便于排查
            logger.debug(f"[{symbol}] 哨兵模式分析跳过: {e}")

    if alerts:
        send_dingtalk("⚡ 极端异动警告", "\n\n".join(alerts))

def run_tech_matrix() -> None:
    logger.info(">>> 启动复合技术指标诊断...")
    
    # 1. 获取全局风控与大盘数据
    vix, vix_desc = get_vix_level()
    regime, regime_desc, qqq_df = get_market_regime()
    logger.info(vix_desc)
    logger.info(f"当前大盘状态: {regime_desc}")
    
    # 计算 QQQ 20日收益率 (用于 RS 相对强度分析)
    qqq_ret_20d = 0.0
    if not qqq_df.empty and len(qqq_df) > 20:
        qqq_ret_20d = (qqq_df['Close'].iloc[-1] / qqq_df['Close'].iloc[-20]) - 1

    target_list = get_nasdaq_100()
    total_stocks = len(target_list)
    reports = []
    
    for idx, symbol in enumerate(target_list):
        if symbol in Config.BLACKLIST:
            logger.info(f"⚠️ [{symbol}] 在黑名单中，已跳过扫描。")
            continue
            
        logger.info(f"[进度 {idx+1}/{total_stocks}] 正在分析 {symbol}...")
        try:
            df = safe_get_history(symbol, period="6mo", interval="1d")
            if len(df) < 100: continue
            if df['Volume'].tail(20).mean() < 1000000: continue

            df = calculate_indicators(df)
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            signals = []
            score = 0
            
            atr_pct = curr['ATR'] / curr['Close']
            if atr_pct > 0.10: continue # 过滤极端波动 (财报/暴雷)

            dynamic_rsi_threshold = max(20, min(40, 30 - (atr_pct - 0.03) * 100))
            
            # --- 相对强度 (RS) 测算 ---
            stock_ret_20d = (curr['Close'] / df['Close'].iloc[-20]) - 1
            rs = (1 + stock_ret_20d) / (1 + qqq_ret_20d)
            if rs > 1.05:
                signals.append(f"⚡ **[相对强势]** 跑赢大盘 {rs-1:.1%}")
                score += 5
            elif rs < 0.95:
                signals.append(f"🐢 **[相对弱势]** 跑输大盘 {(1-rs):.1%}")
                score -= 3 # 弱势股扣分

            # --- RSI 超卖策略 ---
            if curr['RSI'] < dynamic_rsi_threshold and prev['RSI'] >= dynamic_rsi_threshold:
                weight = 13 if regime == 'bear' else 8 
                if curr['Close'] > curr['EMA_50']:
                    signals.append(f"🟢 **[趋势回踩超卖]** 顺势买点 (RSI:{curr['RSI']:.1f})")
                    score += weight
                else:
                    signals.append(f"⚠️ **[弱势超卖]** 防范接飞刀 (RSI:{curr['RSI']:.1f})")
                    score += int(weight * 0.4)
            
            # --- RSI 底背离 ---
            price_low_5 = df['Close'].iloc[-5:].min()
            idx_low_5 = df['Close'].iloc[-5:].idxmin()
            rsi_at_low_5 = df['RSI'].loc[idx_low_5]
            price_low_prev = df['Close'].iloc[-15:-5].min()
            idx_low_prev = df['Close'].iloc[-15:-5].idxmin()
            rsi_at_low_prev = df['RSI'].loc[idx_low_prev]
            
            if price_low_5 < price_low_prev and rsi_at_low_5 > rsi_at_low_prev and curr['RSI'] < 40:
                weight = 12 if regime == 'bear' else 8
                signals.append("🔍 **[RSI底背离]** 下跌动能衰竭")
                score += weight
            
            # --- MACD 策略 ---
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                weight = 10 if regime == 'bull' else 6 
                if curr['MACD'] < 0:
                    signals.append("🔸 **[零下金叉]** 弱反弹预期")
                    score += int(weight * 0.6)
                else:
                    signals.append("🔥 **[零上金叉]** 强势主升浪确认")
                    score += weight

            # --- TTM Squeeze 策略 (布林带进入肯特纳通道) ---
            # 改进：修复数学比较逻辑，直接比较绝对宽度，更准确地捕获波动率压缩
            bb_abs_width = curr['BB_Upper'] - curr['BB_Lower']
            kc_abs_width = curr['KC_Upper'] - curr['KC_Lower']
            is_sqz_on = bb_abs_width < kc_abs_width
            
            if is_sqz_on:
                weight = 12 if regime == 'range' else 6
                signals.append("📦 **[通道极度挤压]** 波动率降至冰点，即将剧烈爆发")
                score += weight

            # --- 均线突破 ---
            if prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
                weight = 9 if regime == 'bull' else 5
                signals.append(f"🚀 **[均线突破]** 强势站上50日均线")
                score += weight

            # --- 多时间框架共振 ---
            if signals:
                weekly_trend = get_weekly_trend(symbol)
                if weekly_trend == "bullish":
                    signals.append("📈 **[周线共振]** 长期趋势多头，增强日线有效性")
                    score += 5 
                elif weekly_trend == "bearish":
                    signals.append("📉 **[周线逆风]** 长期趋势空头，警惕短线陷阱风险")
                    score -= 4 

            if signals:
                # 全局恐慌风控：VIX > 25 时，所有得分打 7 折，提高上榜门槛
                if vix > 25:
                    score = int(score * 0.7)

                if len(signals) > 5:
                    signals = signals[:5] + [f"*(...还有 {len(signals)-5} 个辅助信号)*"]
                    
                turnover = curr['Close'] * curr['Volume']
                turnover_str = f"${turnover/1e9:.2f}B" if turnover >= 1e9 else f"${turnover/1e6:.2f}M"
                    
                report_text = f"**{symbol}** (${curr['Close']:.2f} | 额: {turnover_str} | 🌟综合分: **{score}**)\n> " + "\n> ".join(signals)
                reports.append({"symbol": symbol, "score": score, "text": report_text})
                
        except Exception as e:
            logger.error(f"[{symbol}] 分析发生错误: {e}")

    if reports:
        # 只取有效得分的股票（得分需大于0）
        reports = [r for r in reports if r['score'] > 0]
        reports.sort(key=lambda x: x['score'], reverse=True)
        top_reports_text = [r['text'] for r in reports[:15]]
        
        final_report = f"*{vix_desc}*\n*{regime_desc}*\n\n" + "\n\n".join(top_reports_text)
        if len(reports) > 15:
            final_report += f"\n\n*(已为您过滤并展示得分最高的 Top 15 机会)*"
            
        send_dingtalk("📊 纳指 100 优选异动池", final_report)
    else:
        logger.info("未触发核心指标策略，或因 VIX 极高导致门槛未达。")

def run_daily_screener() -> None:
    logger.info(">>> 启动纳指 100 全量扫盘报告...")
    target_list = get_nasdaq_100()
    bullish, bearish = [], []
    total_stocks = len(target_list)
    
    for idx, symbol in enumerate(target_list):
        if symbol in Config.BLACKLIST:
            continue
            
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
        vix, vix_desc = get_vix_level()
        regime, desc, _ = get_market_regime()
        test_tickers = get_nasdaq_100()[:5]
        send_dingtalk(
            "✅ Pro 量化引擎部署成功", 
            f"已集成精修版算法（2年周线修复、VIX防守增强、TTM通道修正）！\n{vix_desc}\n大盘: **{desc}**\n当前测试名单: {', '.join(test_tickers)}"
        )
    else:
        logger.error(f"未知的模式: {mode}")
