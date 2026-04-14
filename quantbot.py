# 存储路径: .github/workflows/main.yml
name: Tickeron Pro AI Bot

run-name: 🤖 QuantBot 运行状态 - ${{ github.event.inputs.run_mode || '自动调度 (Schedule)' }}

on:
  schedule:
    - cron: '*/30 * * * 1-5'
    - cron: '15 * * * 1-5'
    - cron: '0 23 * * 5'
  workflow_dispatch:
    inputs:
      run_mode:
        description: '选择要手动执行的策略模式'
        required: true
        default: 'test'
        type: choice
        options:
          - test
          - sentinel
          - matrix
          - backtest

concurrency:
  group: bot-running-group
  cancel-in-progress: true

jobs:
  run-bot:
    runs-on: ubuntu-latest
    permissions:
      contents: write     
    timeout-minutes: 15
      
    steps:
      - name: 检出代码
        uses: actions/checkout@v4

      - name: 设置 Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: 缓存 pip 依赖包
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-quantbot
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: 安装量化组件
        run: |
          pip install -r requirements.txt

      - name: 🔍 代码质量与语法检查
        run: |
          flake8 quantbot.py --count --select=E9,F63,F7,F82 --show-source --statistics

      - name: 智能路由与执行
        env:
          WEBHOOK_URL: ${{ secrets.WEBHOOK_URL }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            MODE="${{ github.event.inputs.run_mode }}"
          elif [[ "${{ github.event_name }}" == "schedule" ]]; then
            UTC_H=$(date -u +%-H)
            UTC_M=$(date -u +%-M)
            UTC_D=$(date -u +%-u)
            if [[ "$UTC_D" == "5" && "$UTC_H" == "23" ]]; then
              MODE="backtest"
            elif [[ "$UTC_M" -ge 10 && "$UTC_M" -le 29 ]]; then
              MODE="matrix"
            else
              MODE="sentinel"
            fi
          else
            echo "❌ 拒绝执行：未知触发事件" && exit 1
          fi
          
          echo "RUN_MODE=$MODE" >> $GITHUB_ENV
          echo "🤖 准备以 [$MODE] 模式运行..."
          python quantbot.py $MODE

      - name: 🚨 工作流崩溃告警
        if: failure()
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
              --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
              --data-urlencode "text=🚨 *QuantBot 紧急告警*%0A工作流在执行中发生崩溃，请检查日志！" \
              --data-urlencode "parse_mode=Markdown"
          fi

      - name: 自动提交回测日志与战报
        # [修复逻辑] 只有在 matrix 或 backtest 模式下运行
        if: env.RUN_MODE == 'matrix' || env.RUN_MODE == 'backtest'
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "📝 自动记录选股日志与战绩"
          # [核心修复] 使用通配符或者更宽松的匹配，避免因为某个文件不存在而报错
          # 这样 Git 会自动找到这三个文件中“已存在且被修改”的文件进行提交
          file_pattern: "backtest_log.jsonl strategy_stats.json backtest_report.md"
          # 允许在文件不存在时跳过而不报错
          disable_globbing: false
