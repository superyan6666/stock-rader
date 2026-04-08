import yfinance as yf
import requests
import os
import sys
from datetime import datetime, timezone

# ================= 配置区 =================
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# 【必须包含】钉钉安全关键词（你刚才说你的关键词是 AI）
DINGTALK_KEYWORD = "AI"

# 1. 核心盯盘清单 (用于高频异动和技术指标)
CORE_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT"]

# 2. 全量扫盘清单 (用于每日复盘，可以放很多)
EXTENDED_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "COIN", "MSTR", "0700.HK"]

# ================= 通用函数 =================
def send_dingtalk(title, content):
    """发送格式化的钉钉 Markdown 消息"""
    if not WEBHOOK_URL: return

    # 钉钉的 Markdown 格式要求比较严格，必须在 text 里包含关键词
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": f"【{DINGTALK_KEYWORD} 监控】{title}",
            "text": f"### 🤖 【{DINGTALK_KEYWORD} 量化监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        }
    }
    try:
        requests.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"推送失败: {e}")

# ================= 模式 1: 高频异动哨兵 =================
def run_volatility_sentinel():
    print(">>> 启动高频异动哨兵模式...")
    alerts = []
    for symbol in CORE_WATCHLIST:
        try:
            df = yf.Ticker(symbol).history(period="2d", interval="1h")
            if len(df) < 2: continue
            
            curr_price = df['Close'].iloc[-1]
            open_price = df['Open'].iloc[-1] # 当前小时开盘价
            prev_close = df['Close'].iloc[-2]
            
            # 1. 短线暴跌/暴涨 (1小时内波动 > 3%)
            hour_change = (curr_price - open_price) / open_price * 100
            if abs(hour_change) > 3:
                emoji = "🚀" if hour_change > 0 else "🩸"
                alerts.append(f"> **{symbol}** 短线异动 {emoji} \n> 1小时内剧烈波动: **{hour_change:+.2f}%** (现价: ${curr_price:.2f})")
                
            # 2. 跳空高开/低开 (与上一小时收盘价相比 > 4%)
            gap_change = (open_price - prev_close) / prev_close * 100
            if abs(gap_change) > 4:
                emoji = "💥" if gap_change > 0 else "⚠️"
                alerts.append(f"> **{symbol}** 跳空预警 {emoji}\n> 出现跳空缺口: **{gap_change:+.2f}%**")
                
        except Exception as e:
            pass # 高频模式静默处理错误

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
            df = yf.Ticker(symbol).history(period="1mo", interval="1d") # 用日线级别更稳
            if len(df) < 20: continue

            curr = df.iloc[-1]
            prev = df.iloc[-2]
            price = curr['Close']
            signals = []

            # 简单的 RSI 计算
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/loss))
            curr_rsi = df['RSI'].iloc[-1]
            prev_rsi = df['RSI'].iloc[-2]

            # 策略检测
            if curr_rsi < 30 and prev_rsi >= 30:
                signals.append("🟢 **RSI超卖** (寻底反弹机会)")
            elif curr_rsi > 70 and prev_rsi <= 70:
                signals.append("🔴 **RSI超买** (短期回调风险)")
            
            # 成交量异动
            vol_ma = df['Volume'].rolling(window=20).mean().iloc[-1]
            if curr['Volume'] > vol_ma * 2:
                signals.append("🔥 **巨量突击** (资金高度活跃)")

            if signals:
                reports.append(f"**{symbol}** (日线级别) - 现价: ${price:.2f}\n> " + "\n> ".join(signals))

        except Exception as e:
            pass

    if reports:
        send_dingtalk("📊 策略信号触发", "\n\n".join(reports))
    else:
        print("技术指标未触发核心信号。")

# ================= 模式 3: 每日全量复盘报告 =================
def run_daily_screener():
    print(">>> 启动每日全量扫盘报告...")
    bullish = []
    bearish = []
    neutral = []
    
    for symbol in EXTENDED_WATCHLIST:
        try:
            df = yf.Ticker(symbol).history(period="6mo", interval="1d")
            if len(df) < 50: continue
            
            price = df['Close'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            # 简单趋势打分模型
            score = 0
            if price > ma20: score += 1
            if price > ma50: score += 1
            if ma20 > ma50: score += 1
            
            info = f"**{symbol}**: ${price:.2f}"
            if score == 3:
                bullish.append(info)
            elif score == 0:
                bearish.append(info)
            else:
                neutral.append(info)
        except Exception as e:
            pass

    # 组装研报
    report_lines = [
        "**📈 多头排列 (强势)**", 
        " > " + (", ".join(bullish) if bullish else "无"),
        "\n**📉 空头排列 (弱势)**", 
        " > " + (", ".join(bearish) if bearish else "无"),
        "\n**➖ 震荡整理 (中性)**", 
        " > " + (", ".join(neutral) if neutral else "无")
    ]
    
    send_dingtalk("📝 每日市场全景扫描", "\n".join(report_lines))

# ================= 主控制逻辑 =================
if __name__ == "__main__":
    # 通过命令行参数决定运行哪个模式
    # 如果没有参数，默认运行异动哨兵
    mode = sys.argv[1] if len(sys.argv) > 1 else "sentinel"
    
    if mode == "sentinel":
        run_volatility_sentinel()
    elif mode == "matrix":
        run_tech_matrix()
    elif mode == "daily":
        run_daily_screener()
    elif mode == "test":
        # 强制发送一条测试消息，用于验证 Markdown 和关键词配置
        send_dingtalk("✅ 系统部署成功", f"您的 {DINGTALK_KEYWORD} 监控引擎已完美启动！\n\n> 包含三大模式：\n> 1. 高频异动哨兵\n> 2. 技术指标矩阵\n> 3. 每日全景扫盘")
    else:
        print(f"未知的模式: {mode}")
