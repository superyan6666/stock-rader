import yfinance as yf
import requests
import os
import pandas as pd
from datetime import datetime

# ================= 配置区 =================
# Webhook URL (从 GitHub Secrets 获取)
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# 【重要】：这里填入你在钉钉机器人设置里填写的“自定义关键词”！
# 比如你在钉钉里设置了包含“报警”才发送，这里就必须有“报警”二字。
# 默认我加上了 "提醒" 二字，请确保钉钉的关键词里也有它，或者你把它改成钉钉里设置的词。
DINGTALK_KEYWORD = "AI" 

WATCHLIST = ["NVDA", "AAPL"]

# ================= 消息推送模块 (钉钉专用) =================
def send_dingtalk_notification(content):
    if not WEBHOOK_URL:
        print("❌ 错误：WEBHOOK_URL 环境变量未设置！请检查 GitHub Secrets。")
        return
        
    print(f"尝试向钉钉发送消息... URL截断: {WEBHOOK_URL[:40]}...")
    
    # 钉钉专属的 Payload 格式
    # 注意：text 内容中必须包含你的钉钉安全关键词！
    payload = {
        "msgtype": "text",
        "text": {
            "content": f"【AI 监控 {DINGTALK_KEYWORD}】\n\n{content}\n\n⏱️ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        }
    }
    
    try:
        resp = requests.post(WEBHOOK_URL, json=payload)
        print(f"---- 钉钉服务器真实响应 ----")
        print(f"状态码: {resp.status_code}")
        print(f"响应内容: {resp.text}")
        print(f"--------------------------")
        
        # 钉钉错误码分析
        if resp.status_code == 200:
            res_data = resp.json()
            if res_data.get("errcode") == 310000:
                print("🚨 致命错误 (310000): 安全设置校验失败！")
                print("   原因：你发送的 content 里，没有包含钉钉机器人设置的【自定义关键词】。")
                print(f"   解决：请检查钉钉群的机器人设置，看自定义关键词是什么，并修改代码中的 DINGTALK_KEYWORD。")
            elif res_data.get("errcode") == 0:
                print("✅ 钉钉返回 errcode: 0。消息发送成功！请查看钉钉群。")
            else:
                print(f"⚠️ 其他错误: {res_data}")
        else:
            print(f"❌ 网络请求失败，状态码不是 200。")
            
    except Exception as e:
        print(f"推送请求发生异常: {e}")

# ================= 主程序入口 =================
if __name__ == "__main__":
    print(f"开始执行钉钉专属通道测试...")
    
    # 【强制测试模式】
    # 为了排查，我们忽略股票逻辑，直接强行发一条消息
    test_message = "这是一条来自 GitHub Actions 的强制测试消息。\n如果您在钉钉看到这条消息，说明打通了！\n之后您可以把这行测试代码注释掉，启用真正的股票监控逻辑。"
    
    send_dingtalk_notification(test_message)
    
    print("测试执行完毕。请查看上方的【钉钉服务器真实响应】日志。")
