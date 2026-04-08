import yfinance as yf
import requests
import os
import sys
import pandas as pd
from datetime import datetime, timezone

# ================= 配置区 =================
# Webhook URL (从 GitHub Secrets 获取)
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# 【重要】钉钉安全关键词
DINGTALK_KEYWORD = "AI"

# 1. 核心盯盘清单：用于高频异动哨兵，防止请求过多被封
CORE_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]

# ================= 动态获取股票池 =================
def get_nasdaq_100():
    """实时从维基百科抓取最新的纳斯达克100成分股"""
    print(">>> 正在联网获取最新 纳斯达克 100 成分股名单...")
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        tables = pd.read_html(response.text, match='Ticker')
        df = tables[0]
        tickers = df['Ticker'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # 修复雅虎财经格式
        
        print(f"✅ 成功获取 {len(tickers)} 只纳斯达克 100 股票！")
        return tickers
    except Exception as e:
        print(f"❌ 获取名单失败，使用备用核心名单。错误: {e}")
        return CORE_WATCHLIST

# ================= 通用消息推送模块 =================
def send_dingtalk(title, content):
    """发送格式化的钉钉 Markdown 消息"""
    if not WEBHOOK_URL: 
        print("❌ 错误：WEBHOOK_URL 环境变量未设置")
        return

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": f"【{DINGTALK_KEYWORD}盯盘】{title}",
            "text": f"### 🤖 【{DINGTALK_KEYWORD}量化监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        }
    }
    try:
        requests.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"推送请求发生异常: {e}")

# ================= 核心策略引擎 (共用计算模块) =================
def calculate_indicators(df):
    """计算所有的技术指标"""
    # 计算 RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # 防止除零错误 (加上极小值 1e-10)
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 计算 MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 计算均线
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # 计算 20 日平均成交量
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # 计算 ATR (14) 平均真实波幅，用于评估波动率
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df

# ================= 模式 1: 高频异动哨兵 (仅扫描核心自选) =================
def run_volatility_sentinel():
    print(">>> 启动高频异动哨兵模式...")
    alerts = []
    for symbol in CORE_WATCHLIST:
        try:
            df = yf.Ticker(symbol).history(period="2d", interval="1h")
            if len(df) < 2: continue
            
            curr_price = df['Close'].iloc[-1]
            open_price = df['Open'].iloc[-1] 
            prev_close = df['Close'].iloc[-2]
            
            # 短线急涨急跌
            hour_change = (curr_price - open_price) / open_price * 100
            if abs(hour_change) > 3:
                alerts.append(f"> **{symbol}** 短线异动 {'🚀' if hour_change > 0 else '🩸'} \n> 1小时内波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            # 跳空缺口
            gap_change = (open_price - prev_close) / prev_close * 100
            if abs(gap_change) > 4:
                alerts.append(f"> **{symbol}** 跳空预警 {'💥' if gap_change > 0 else '⚠️'}\n> 缺口: **{gap_change:+.2f}%**")
        except Exception as e:
            pass 

    if alerts:
        send_dingtalk("⚡ 极端异动警告", "\n\n".join(alerts))
    else:
        print("未发现极端异动。")

# ================= 模式 2: 技术指标矩阵 (扫描纳指100) =================
def run_tech_matrix():
    print(">>> 启动复合技术指标诊断...")
    target_list = get_nasdaq_100()
    reports = []
    
    for symbol in target_list:
        try:
            ticker = yf.Ticker(symbol)
            # 数据窗口拉长为 6mo 确保均线准确
            df = ticker.history(period="6mo", interval="1d")
            if len(df) < 55: continue

            df = calculate_indicators(df)
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # 严格的流动性过滤 (排除冷门股)
            if curr['Vol_MA20'] < 1000000:
                continue

            signals = []
            
            # 改进 1: 加入波动率评估的动态 RSI 阈值
            atr_pct = curr['ATR'] / curr['Close']
            # 根据真实波动幅度调整阈值，限制在 20 到 40 之间
            dynamic_rsi_threshold = max(20, min(40, 30 - (atr_pct - 0.03) * 100))

            # RSI 趋势与动态阈值过滤
            if curr['RSI'] < dynamic_rsi_threshold and prev['RSI'] >= dynamic_rsi_threshold:
                if curr['Close'] > curr['EMA_50']:
                    signals.append(f"🟢 **[趋势回踩超卖]** 顺势买点 (阈值:{dynamic_rsi_threshold:.1f}, RSI:{curr['RSI']:.1f})")
                else:
                    signals.append(f"⚠️ **[弱势超卖]** 防范接飞刀 (阈值:{dynamic_rsi_threshold:.1f}, RSI:{curr['RSI']:.1f})")
            
            # 改进 2: MACD 金叉的零轴环境区分
            if prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
                if curr['MACD'] < 0:
                    signals.append("🔸 **[零下金叉]** 弱反弹预期，注意见好就收")
                else:
                    signals.append("🔥 **[零上金叉]** 强势主升浪确认")

            # 改进 3: 放量倍数分级评估
            vol_ratio = curr['Volume'] / curr['Vol_MA20']
            if vol_ratio > 3:
                signals.append(f"🌋 **[极端巨量]** 成交量激增 {vol_ratio:.1f} 倍！")
            elif vol_ratio > 1.8:
                signals.append(f"🌊 **[温和放量]** 成交量放大 {vol_ratio:.1f} 倍")

            # 趋势突破
            if prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
                signals.append(f"🚀 **[均线突破]** 强势站上50日均线 (${curr['EMA_50']:.2f})")

            if signals:
                reports.append(f"**{symbol}** (${curr['Close']:.2f})\n> " + "\n> ".join(signals))
                
        except Exception as e:
            pass

    if reports:
        final_report = "\n\n".join(reports[:15])
        if len(reports) > 15:
            final_report += f"\n\n*(还有 {len(reports)-15} 只股票触发预警，已省略)*"
        send_dingtalk("📊 纳指 100 策略异动池", final_report)
    else:
        print("未触发核心指标策略。")

# ================= 模式 3: 每日全量复盘报告 (扫描纳指100) =================
def run_daily_screener():
    print(">>> 启动纳指 100 全量扫盘报告...")
    target_list = get_nasdaq_100()
    bullish, bearish = [], []
    
    for symbol in target_list:
        try:
            df = yf.Ticker(symbol).history(period="6mo", interval="1d")
            if len(df) < 50: continue
            
            price = df['Close'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            score = (price > ma20) + (price > ma50) + (ma20 > ma50)
            
            if score == 3: bullish.append(symbol)
            elif score == 0: bearish.append(symbol)
        except Exception as e:
            pass

    report = f"🎯 **今日纳斯达克 100 扫描总结**\n\n**📈 绝对多头排列 (强势)**\n> {', '.join(bullish) if bullish else '无'}\n\n**📉 绝对空头排列 (弱势)**\n> {', '.join(bearish) if bearish else '无'}"
    send_dingtalk("📝 纳指 100 全景复盘", report)

# ================= 主控制逻辑 =================
if __name__ == "__main__":
    # 接收 GitHub Actions 传递的执行模式，默认执行 sentinel
    mode = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    print(f"正在以模式运行: [{mode}]")
    
    if mode == "sentinel":
        run_volatility_sentinel()
    elif mode == "matrix":
        run_tech_matrix()
    elif mode == "daily":
        run_daily_screener()
    elif mode == "test":
        test_tickers = get_nasdaq_100()[:5]
        send_dingtalk("✅ Pro 引擎部署成功", f"包含防接飞刀机制与自动纳指100抓取！\n当前测试获取的前五个代码为: {', '.join(test_tickers)}")
    else:
        print(f"未知的模式: {mode}")
