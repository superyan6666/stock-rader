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
    
    # 钉钉专属的 payload 格式
    payload = {
        "msgtype": "text",
        "text": {
            "content": f"【AI 盯盘助手】\n{content}\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        print(f"---- Webhook 响应信息 ----")
        print(f"HTTP 状态码: {response.status_code}")
        print(f"服务器真实返回: {response.text}")
        print(f"--------------------------")
        
        # 分析钉钉的常见错误
        if response.status_code == 200:
            res_data = response.json()
            if res_data.get("errcode") == 0:
                print("✅ 发送成功！请检查手机钉钉。")
            elif res_data.get("errcode") == 310000:
                print("🚨 警告: 关键词不匹配！请确保钉钉机器人的安全设置【自定义关键词】包含 '盯盘助手'。")
            elif res_data.get("errcode") == 404:
                print("🚨 警告: 找不到机器人！你复制的 Webhook URL 地址不完整或格式有误。")
            else:
                print(f"🚨 其他错误: {res_data.get('errmsg')}")
        else:
            print("🚨 警告: 网络请求失败。")
            
    except Exception as e:
        print(f"发送请求时发生网络错误: {e}")

if __name__ == "__main__":
    print("--- 强制测试模式启动 ---")
    
    # 无论股市如何，先强制发一条测试消息
    test_msg = "这是一条强制测试消息。如果您收到，说明 GitHub 到手机的推送通道已完全畅通！"
    send_notification(test_msg)
