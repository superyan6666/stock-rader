#!/bin/bash
# 🚀 QuantBot 3.0 Execution Gateway Daemon
# 严格的进程生命周期管理，确保网关成为“不死鸟”

GATEWAY_SCRIPT="quant_engine.py gateway"
LOG_FILE="gateway_stdout.log"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🛡️ 启动 QuantBot 实盘执行网关守护进程..."

# 无限死循环保活机制
while true; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🟢 正在拉起网关进程: $GATEWAY_SCRIPT"
    
    # 启动 Python 进程，并将标准输出与错误流重定向至日志
    python3 $GATEWAY_SCRIPT >> $LOG_FILE 2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🛑 网关进程正常退出 (可能收到了终止信号)。守护结束。"
        break
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 💥 致命告警: 网关进程异常崩溃 (Exit Code: $EXIT_CODE)！"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ♻️ 触发熔断保护，休眠 5 秒后执行涅槃重启..."
        # 预留给外部告警系统的 Hook (例如触发 Telegram 报警脚本)
        sleep 5
    fi
done
