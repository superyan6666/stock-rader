# 存储路径: .github/workflows/main.yml
name: Tickeron Pro AI Bot

run-name: 🤖 QuantBot 运行状态 - ${{ github.event.inputs.run_mode || '自动调度 (Schedule)' }}

on:
  schedule:
    - cron: '*/30 * * * 1-5'       # sentinel 每半小时
    - cron: '15 * * * 1-5'         # matrix 每小时15分
    - cron: '0 23 * * 5'           # backtest 周五深夜
  workflow_dispatch:
    inputs:
      run_mode:
        description: '选择运行模式'
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

      - name: 缓存 pip 依赖
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-quantbot
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: 安装组件
        run: |
          pip install -r requirements.txt

      - name: 🔍 代码质量检查
        run: |
          # 确保是在检查 Python 文件，而不是 Workflow 本身
          flake8 quantbot.py --count --select=E9,F63,F7,F82 --show-source --statistics

      - name: 执行引擎
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
          fi
          
          echo "RUN_MODE=$MODE" >> $GITHUB_ENV
          python quantbot.py $MODE

      - name: 🚨 崩溃告警
        if: failure()
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
              --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
              --data-urlencode "text=🚨 *QuantBot 严重崩溃*！请立即检查日志。" \
              --data-urlencode "parse_mode=Markdown"
          fi

      - name: 自动提交数据
        if: env.RUN_MODE == 'matrix' || env.RUN_MODE == 'backtest'
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "📝 自动同步量化数据记录"
          file_pattern: "backtest_log.jsonl strategy_stats.json backtest_report.md"
          disable_globbing: false
