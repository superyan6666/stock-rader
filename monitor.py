import yfinance as yf
import requests
import os
import sys
import pandas as pd
from datetime import datetime, timezone

# ================= 配置区 =================
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# 【重要】钉钉安全关键词
DINGTALK_KEYWORD = "AI"

# 1. 核心盯盘清单：你可以把最关心的几只单独放这里（用于高频异动哨兵）
CORE_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT"]

# ================= 动态获取股票池 =================
def get_nasdaq_100():
    """实时从维基百科抓取最新的纳斯达克100成分股"""
    print(">>> 正在联网获取最新 纳斯达克 100 成分股名单...")
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        # 伪装成浏览器请求，防止被维基百科拦截
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        # 使用 pandas 自动解析网页中包含 'Ticker' 列的表格
        tables = pd.read_html(response.text, match='Ticker')
        df = tables[0]
        
        # 提取股票代码并转换为列表
        tickers = df['Ticker'].tolist()
        
        # 修复雅虎财经的代码格式 (比如维基上是 BRK.B，雅虎需要 BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"✅ 成功获取 {len(tickers)} 只纳斯达克 100 股票！")
        return tickers
    except Exception as e:
        print(f"❌ 获取名单失败，将使用备用名单。错误: {e}")
        # 如果抓取失败，返回几个保底的科技巨头
        return ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]

# ================= 通用消息推送模块 =================
def send_dingtalk(title, content):
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

# ================= 模式 1: 高频异动哨兵 (仅扫描核心自选) =================
def run_volatility_sentinel():
    print(">>> 启动高频异动哨兵模式...")
    alerts = []
    # 高频扫描为了速度和防止被封IP，仅扫描 CORE_WATCHLIST
    for symbol in CORE_WATCHLIST:
        try:
            df = yf.Ticker(symbol).history(period="2d", interval="1h")
            if len(df) < 2: continue
            
            curr_price = df['Close'].iloc[-1]
            open_price = df['Open'].iloc[-1] 
            prev_close = df['Close'].iloc[-2]
            
            hour_change = (curr_price - open_price) / open_price * 100
            if abs(hour_change) > 3:
                alerts.append(f"> **{symbol}** 短线异动 {'🚀' if hour_change > 0 else '🩸'} \n> 1小时内波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            gap_change = (open_price - prev_close) / prev_close * 100
            if abs(gap_change) > 4:
                alerts.append(f"> **{symbol}** 跳空预警 {'💥' if gap_change > 0 else '⚠️'}\n> 缺口: **{gap_change:+.2f}%**")
        except Exception as e:
            pass 

    if alerts:
        send_dingtalk("⚡ 极端异动警告", "\n\n".join(alerts))

# ================= 模式 2: 技术指标矩阵 =================
def run_tech_matrix():
    print(">>> 启动复合技术指标诊断...")
    
    # 动态获取纳斯达克 100 列表！
    target_list = get_nasdaq_100()
    reports = []
    
    for symbol in target_list:
        try:
            df = yf.Ticker(symbol).history(period="1mo", interval="1d") 
            if len(df) < 20: continue

            curr = df.iloc[-1]
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/loss))

            signals = []
            if df['RSI'].iloc[-1] < 30 and df['RSI'].iloc[-2] >= 30:
                signals.append("🟢 **RSI超卖** (寻底反弹机会)")
            elif df['RSI'].iloc[-1] > 70 and df['RSI'].iloc[-2] <= 70:
                signals.append("🔴 **RSI超买** (短期回调风险)")
            
            if curr['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 2:
                signals.append("🔥 **巨量突击** (资金高度活跃)")

            if signals:
                reports.append(f"**{symbol}** (${curr['Close']:.2f})\n> " + "\n> ".join(signals))
        except Exception as e:
            pass

    if reports:
        # 如果触发的股票太多，截取前 15 个防止钉钉消息过长报错
        final_report = "\n\n".join(reports[:15])
        if len(reports) > 15:
            final_report += f"\n\n*(还有 {len(reports)-15} 只股票触发预警，已省略)*"
        send_dingtalk("📊 纳指 100 策略异动池", final_report)

# ================= 模式 3: 每日全量复盘报告 =================
def run_daily_screener():
    print(">>> 启动纳指 100 全量扫盘报告...")
    
    # 动态获取纳斯达克 100 列表！
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    print(f"正在以模式运行: [{mode}]")
    
    if mode == "sentinel":
        run_volatility_sentinel()
    elif mode == "matrix":
        run_tech_matrix()
    elif mode == "daily":
        run_daily_screener()
    elif mode == "test":
        test_tickers = get_nasdaq_100()[:5] # 测试抓取前5个
        send_dingtalk("✅ 纳指动态引擎部署成功", f"成功接入维基百科实时爬虫！\n当前纳指100前五个代码为: {', '.join(test_tickers)}")
    else:
        print(f"未知的模式: {mode}")
