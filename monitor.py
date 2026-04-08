import yfinance as yf
import requests
import os
import json
from datetime import datetime

# 从 GitHub Secrets 中读取
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')
WATCHLIST = ["NVDA", "AAPL"] # 缩短列表加快测试

def send_notification(content):
    if not WEBHOOK_URL: 
        print("错误: 环境变量 WEBHOOK_URL 为空")
        return
    
    print(f"准备向 Webhook 发送消息... URL前缀: {WEBHOOK_URL[:30]}...")
    
    # 飞书的 payload 格式
    payload = {
        "msg_type": "text",
        "content": {"text": f"【AI 盯盘助手】\n{content}\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
    }
    
    # 如果你是钉钉，请用下面这行替换上面的 payload（注意取消注释）
    # payload = {"msgtype": "text", "text": {"content": f"【AI 盯盘助手】\n{content}"}}

    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        print(f"---- Webhook 响应信息 ----")
        print(f"HTTP 状态码: {response.status_code}")
        print(f"服务器真实返回: {response.text}")
        print(f"--------------------------")
        
        # 分析常见错误
        if response.status_code == 200:
            res_data = response.json()
            if res_data.get("code") != 0 and res_data.get("errcode") != 0:
                print("🚨 警告: 请求已送达，但被机器人拒绝！请检查【安全设置】中的【自定义关键词】是否包含 'AI' 或 '盯盘助手'。")
            else:
                print("✅ 发送成功！请检查手机。")
        else:
            print("🚨 警告: Webhook 地址可能有误或格式不兼容。")
            
    except Exception as e:
        print(f"发送请求时发生网络错误: {e}")

if __name__ == "__main__":
    print("--- 强制测试模式启动 ---")
    
    # 无论股市如何，先强制发一条测试消息
    test_msg = "这是一条强制测试消息。如果您收到，说明 GitHub 到手机的推送通道已完全畅通！"
    send_notification(test_msg)
