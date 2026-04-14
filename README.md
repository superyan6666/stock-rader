🤖 QuantBot - GitHub Actions 量化盯盘机器人

这是一个专为 GitHub Actions 设计的零成本、免维护量化多因子盯盘机器人。无需购买云服务器，依托 GitHub 官方容器，7x24 小时监控全球市场异动，并将投资机会实时推送给您。

✨ 核心特性 (Features)

🆓 零成本托管：基于 GitHub Actions 定时触发（Cron），告别云服务器续费烦恼。

🧠 智能“四合一”极简架构：

⚡ sentinel (高频异动哨兵)：每半小时动态扫描全市场近期成交最活跃的 Top 40 龙头股，实时捕捉盘中暴涨/暴跌与极端跳空异动。

🎯 matrix (多因子信号矩阵)：监控 MACD 金叉、RSI 背离、均线突破、布林挤压与 TTM Squeeze。

📝 daily (全景复盘)：每日收盘后扫描，梳理多空排列阵型（已并入 Matrix 统一路由）。

📅 backtest (自动阅卷闭环)：每周五自动比对历史推送记录与真实 K 线，生成 T+3 胜率战报。

🌐 动态全市场漏斗：自动从 Wikipedia 抓取 S&P 500、S&P 400 和 Nasdaq 100，并按成交活跃度动态筛选 Top 120。

🛡️ 大盘与板块风控：自动感知 VIX 恐慌指数与大盘趋势，触发“防雷网”和“板块拥挤度”自适应降权。

📢 多渠道消息推送：支持 钉钉/飞书/企业微信 Webhook，以及 Telegram Bot (支持 MarkdownV2)。

🏭 企业级 DevOps 部署：内置防封禁分块下载 (Chunking)、依赖缓存加速、并发死锁熔断以及严格的版本锁定。

🚀 快速开始 (Quick Start)

只需 3 步，即可在您的 GitHub 账户下免费运行此机器人。

1. Fork 本仓库

点击页面右上角的 Fork 按钮，将本项目复制到您自己的 GitHub 账号下。

2. 配置环境变量 (Secrets)

进入您 Fork 后的仓库，点击 Settings -> Secrets and variables -> Actions -> New repository secret，添加以下通知渠道配置（任选其一或全选）：

Secret 名称

描述

是否必填

WEBHOOK_URL

钉钉/飞书/企业微信的群机器人 Webhook 地址（支持英文逗号分隔配置多个）

选填

TELEGRAM_BOT_TOKEN

Telegram 机器人的 Token (通过 @BotFather 获取)

选填

TELEGRAM_CHAT_ID

Telegram 接收消息的 Chat ID

选填

注意：WEBHOOK_URL 和 TELEGRAM 相关的配置必须至少填写一种，否则程序将报错退出。

3. 启用 GitHub Actions

点击仓库顶部的 Actions 标签页。

如果看到 "I understand my workflows, go ahead and enable them"，点击确认启用。

在左侧工作流列表中选择 Tickeron Pro AI Bot。

点击 Run workflow，您可以手动选择 test 模式进行推送测试。

🕒 自动化运行时间表 (Schedule)

机器人完全依靠 GitHub Actions 的 Cron 触发自动运行，默认时间表如下（均为 UTC 时间，工作日运行）：

高频哨兵 (Sentinel): 每半小时执行一次。

多因子矩阵 (Matrix): 每小时的第 15 分钟执行一次。

每日复盘 (Daily): 每天 22:00 UTC 执行。

历史回测 (Backtest): 每周五 23:00 UTC 执行自动数据统计。

💡 调度逻辑由 main.yml 结合实际 UTC 时间戳严格接管，杜绝了并发拥堵与调度漂移。

📁 核心文件结构

quantbot.py: The Single Core (统一核心)。包含数据漏斗、指标计算、风控策略、推送与回测闭环的所有逻辑（不到 500 行）。

requirements.txt: 严格锁定的 Python 环境依赖库。

.github/workflows/main.yml: GitHub Actions 自动化运维部署文件。

backtest_log.jsonl: 自动生成并提交的策略选股历史日志文件（无需手动创建）。

strategy_stats.json: 回测引擎生成的历史胜率库（自动挂载至推送消息头部）。

⚠️ 免责声明 (Disclaimer)

本项目仅供编程学习与量化策略研究使用。所有技术指标与自动化产生的信号不构成任何投资建议。市场有风险，投资需谨慎！使用者需对自身的投资行为负责。
