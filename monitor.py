import yfinance as yf
import requests
import os
from datetime import datetime

# 从 GitHub Secrets 中读取敏感信息（Webhook 地址）
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')
# 监控列表，可以根据需要修改
WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "0700.HK"]

def check_strategy(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # 获取最近 5 天的数据计算技术指标
        df = ticker.history(period="5d", interval="1h")
        if len(df) < 20: return None

        current_price = df['Close'].iloc[-1]
        # 简单模拟 Tickeron 的 RSI 策略
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + gain/loss)).iloc[-1]

        # 触发条件：RSI < 30 (超卖)
        if rsi < 30:
            return f"⚠️ {symbol} RSI 低于 30 ({rsi:.1f})，处于超卖区域，可能反弹。现价: ${current_price:.2f}"
        
        # 触发条件：单日大跌超过 5%
        change = (df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100
        if change < -5:
            return f"🚨 {symbol} 今日剧烈波动，跌幅 {change:.1f}%。现价: ${current_price:.2f}"

        return None
    except Exception as e:
        print(f"Error checking {symbol}: {e}")
        return None

def send_notification(content):
    if not WEBHOOK_URL: 
        print("Error: WEBHOOK_URL not set")
        return
    
    payload = {
        "msg_type": "text",
        "content": {"text": f"【AI 盯盘助手】\n{content}\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
    }
    requests.post(WEBHOOK_URL, json=payload)

if __name__ == "__main__":
    alerts = []
    for s in WATCHLIST:
        res = check_strategy(s)
        if res: alerts.append(res)
    
    if alerts:
        send_notification("\n\n".join(alerts))
    else:
        print("No signals triggered.")
