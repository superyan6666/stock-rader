# 存储路径: .github/workflows/main.yml
name: Tickeron Pro AI Bot

on:
  schedule:
    - cron: '*/30 * * * 1-5'
    - cron: '15 * * * 1-5'
    - cron: '0 22 * * 1-5'
    - cron: '0 23 * * 5'  # 每周五 23:00 UTC 执行终极回测
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
          - daily
          - backtest      # 一键触发回测并推送到手机

# [新增] 并发控制：防止多个实例同时运行导致 Git 提交冲突
concurrency:
  group: bot-running-group
  cancel-in-progress: false

jobs:
  run-bot:
    runs-on: ubuntu-latest
    permissions:
      contents: write     
    # [新增] 超时保护：设定物理熔断时间，防止由于网络卡死耗尽 GitHub 免费额度
    timeout-minutes: 15
    
    # [新增] 全局时区对齐美东时间
    env:
      TZ: 'America/New_York'
      
    steps:
      - name: 检出代码
        uses: actions/checkout@v4

      - name: 设置 Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      # [新增] Pip 依赖缓存机制：将几十秒的下载时间压缩至几秒，极大节省运行额度
      - name: 缓存 pip 依赖包
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-quantbot
          restore-keys: |
            ${{ runner.os }}-pip-

      # [新增] 核心组件版本锁定：防止第三方库破坏性更新导致的深夜崩溃
      - name: 安装量化组件 (严格版本锁定)
        run: |
          pip install yfinance==0.2.36 pandas==2.1.0 requests==2.31.0 lxml==4.9.3

      - name: 智能路由与执行
        env:
          WEBHOOK_URL: ${{ secrets.WEBHOOK_URL }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          MODE="sentinel"
          
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            MODE="${{ github.event.inputs.run_mode }}"
          elif [[ "${{ github.event_name }}" == "schedule" ]]; then
            if [[ "${{ github.event.schedule }}" == "0 22 * * 1-5" ]]; then
              MODE="daily"
            elif [[ "${{ github.event.schedule }}" == "15 * * * 1-5" ]]; then
              MODE="matrix"
            elif [[ "${{ github.event.schedule }}" == "0 23 * * 5" ]]; then
              MODE="backtest"
            else
              MODE="sentinel"
            fi
          fi
          
          # 【极简路由】所有模式均由唯一的 quantbot.py 处理
          echo "🤖 准备以 [$MODE] 模式运行 AI 引擎..."
          python quantbot.py $MODE

      - name: 自动提交回测日志与战报
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "📝 自动记录今日回测日志与战绩"
          file_pattern: "backtest_log.jsonl backtest_report.md strategy_stats.json"
