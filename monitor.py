import yfinance as yf
import requests
import os
import pandas as pd
from datetime import datetime

# ================= 配置区 =================
# Webhook URL (从 GitHub Secrets 获取)
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# 监控清单 (可无限添加，支持美股和港股如 0700.HK, 加密货币如 BTC-USD)
WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "COIN", "MSTR", "BTC-USD"]

# 策略触发开关 (可以随时关闭不需要的策略)
ENABLE_RSI = True
ENABLE_MACD = True
ENABLE_VOL_SPIKE = True
ENABLE_EMA_BREAKOUT = True

# ================= 核心策略引擎 =================
def calculate_indicators(df):
    """计算所有的技术指标"""
    # 计算 RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
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
    
    return df

def check_strategies(symbol):
    """为单只股票检查所有策略，返回触发的警告列表"""
    alerts = []
    try:
        ticker = yf.Ticker(symbol)
        # 获取最近的数据（以1小时间隔为例，你可以改为 '1d' 看日线级别）
        df = ticker.history(period="1mo", interval="1h")
        if len(df) < 30: 
            return alerts

        df = calculate_indicators(df)
        
        # 获取最新和上一周期的值
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price = curr['Close']

        # 策略 1: RSI 极度超卖 (反弹预警)
        if ENABLE_RSI and curr['RSI'] < 30 and prev['RSI'] >= 30:
            alerts.append(f"🟢 [RSI超卖反弹] RSI跌破30，当前 {curr['RSI']:.1f}")

        # 策略 2: MACD 黄金交叉 (趋势转多预警)
        if ENABLE_MACD and prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']:
            alerts.append(f"🔥 [MACD金叉] MACD上穿零轴信号线，多头启动可能")

        # 策略 3: 异常巨量突击 (资金异动预警)
        if ENABLE_VOL_SPIKE and curr['Volume'] > (curr['Vol_MA20'] * 3):
            vol_multiplier = curr['Volume'] / curr['Vol_MA20']
            alerts.append(f"⚠️ [巨量异动] 成交量是平时均量的 {vol_multiplier:.1f} 倍！")

        # 策略 4: 突破重要均线 (右侧交易信号)
        if ENABLE_EMA_BREAKOUT and prev['Close'] < prev['EMA_50'] and curr['Close'] > curr['EMA_50']:
            alerts.append(f"🚀 [趋势突破] 强势突破 50 周期指数均线 (${curr['EMA_50']:.2f})")

        # 如果有任何策略被触发，组装这只股票的报告
        if alerts:
            report = f"**{symbol}** (现价: ${price:.2f})\n"
            for a in alerts:
                report += f"  {a}\n"
            return report
            
    except Exception as e:
        print(f"分析 {symbol} 失败: {e}")
        
    return None

# ================= 消息推送模块 =================
def send_notification(content):
    if not WEBHOOK_URL:
        print("错误：WEBHOOK_URL 环境变量未设置")
        return
        
    # 构建飞书/钉钉支持的 Markdown/纯文本格式
    payload = {
        "msg_type": "text",
        "content": {
            "text": f"🤖 【AI 量化监控报告】\n\n{content}\n\n⏱️ 扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        }
    }
    
    try:
        resp = requests.post(WEBHOOK_URL, json=payload)
        print(f"推送状态码: {resp.status_code}")
        print(f"推送响应: {resp.text}")
    except Exception as e:
        print(f"推送失败: {e}")

# ================= 主程序入口 =================
if __name__ == "__main__":
    print(f"开始执行全量资产扫描...")
    final_report = []
    
    for symbol in WATCHLIST:
        print(f"正在扫描标的: {symbol}...")
        result = check_strategies(symbol)
        if result:
            final_report.append(result)
            
    if final_report:
        # 如果有多只股票触发信号，用分隔符隔开
        full_message = "\n---\n".join(final_report)
        send_notification(full_message)
    else:
        print("市场平静，没有任何标的触发预警策略。")
        
    # 为了方便你测试，如果是你第一次运行，我加了一行测试防呆代码：
    # （测试成功后可以把下面两行删掉）
    print("\n[注] 如果你一直收不到信号，可以取消下一行的注释强制发一条")
    # send_notification("✅ Pro 引擎部署成功！正在 24/7 守护。")
