import yfinance as yf
import requests
import os
import sys
from datetime import datetime, timezone

# ================= 配置区 =================
# Webhook URL (从 GitHub Secrets 获取)
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# 【重要】钉钉安全关键词
DINGTALK_KEYWORD = "AI"

# 1. 核心盯盘清单 (高频异动和技术指标)
CORE_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT"]

# 2. 全量扫盘清单 (每日复盘)
EXTENDED_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "COIN", "MSTR"]

# ================= 通用消息推送模块 =================
def send_dingtalk(title, content):
    if not WEBHOOK_URL: 
        print("❌ 错误：WEBHOOK_URL 环境变量未设置")
        return

    # 钉钉 Markdown 格式，严格将关键词嵌入标题和正文中
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": f"【{DINGTALK_KEYWORD}盯盘】{title}",
            "text": f"### 🤖 【{DINGTALK_KEYWORD}量化监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        }
    }
    
    try:
        resp = requests.post(WEBHOOK_URL, json=payload)
        print(f"---- 钉钉服务器真实响应 ----")
        print(f"状态码: {resp.status_code}")
        print(f"响应内容: {resp.text}")
        
        # 错误分析
        resp_json = resp.json()
        if resp_json.get("errcode") != 0:
            print(f"🚨 警告: 消息被钉钉拦截！请再次确认安全设置中的关键词是否完全包含: {DINGTALK_KEYWORD}")
        else:
            print("✅ 消息推送成功！")
            
    except Exception as e:
        print(f"推送请求发生异常: {e}")

# ================= 模式 1: 高频异动哨兵 =================
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
            
            hour_change = (curr_price - open_price) / open_price * 100
            if abs(hour_change) > 3:
                emoji = "🚀" if hour_change > 0 else "🩸"
                alerts.append(f"> **{symbol}** 短线异动 {emoji} \n> 1小时内波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            gap_change = (open_price - prev_close) / prev_close * 100
            if abs(gap_change) > 4:
                emoji = "💥" if gap_change > 0 else "⚠️"
                alerts.append(f"> **{symbol}** 跳空预警 {emoji}\n> 缺口: **{gap_change:+.2f}%**")
        except Exception as e:
            pass 

    if alerts:
        send_dingtalk("⚡ 极端异动警告", "\n\n".join(alerts))
    else:
        print("未发现极端异动。")

# ================= 模式 2: 复合技术指标矩阵 =================
def run_tech_matrix():
    print(">>> 启动复合技术指标诊断...")
    reports = []
    for symbol in CORE_WATCHLIST:
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
                reports.append(f"**{symbol}** (日线级别) - 现价: ${curr['Close']:.2f}\n> " + "\n> ".join(signals))
        except Exception as e:
            pass

    if reports:
        send_dingtalk("📊 策略信号触发", "\n\n".join(reports))
    else:
        print("技术指标未触发核心信号。")

# ================= 模式 3: 每日全量复盘报告 =================
def run_daily_screener():
    print(">>> 启动每日全量扫盘报告...")
    bullish, bearish, neutral = [], [], []
    
    for symbol in EXTENDED_WATCHLIST:
        try:
            df = yf.Ticker(symbol).history(period="6mo", interval="1d")
            if len(df) < 50: continue
            
            price = df['Close'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            score = (price > ma20) + (price > ma50) + (ma20 > ma50)
            info = f"{symbol} (${price:.2f})"
            
            if score == 3: bullish.append(info)
            elif score == 0: bearish.append(info)
            else: neutral.append(info)
        except Exception as e:
            pass

    report = f"**📈 多头排列 (强势)**\n> {', '.join(bullish) if bullish else '无'}\n\n**📉 空头排列 (弱势)**\n> {', '.join(bearish) if bearish else '无'}\n\n**➖ 震荡整理 (中性)**\n> {', '.join(neutral) if neutral else '无'}"
    send_dingtalk("📝 每日市场全景扫描", report)

# ================= 主控制逻辑 =================
if __name__ == "__main__":
    # 接收来自 GitHub Actions 传递的参数
    mode = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    print(f"正在以模式运行: [{mode}]")
    
    if mode == "sentinel":
        run_volatility_sentinel()
    elif mode == "matrix":
        run_tech_matrix()
    elif mode == "daily":
        run_daily_screener()
    elif mode == "test":
        print("执行连接测试...")
        send_dingtalk("✅ 终极引擎部署成功", "如果您收到此消息，说明您的 GitHub Actions 三合一路由与钉钉机器人通道已完美打通！您可以开始享受全自动化盯盘了。")
    else:
        print(f"未知的模式: {mode}")
