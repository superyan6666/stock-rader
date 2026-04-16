🤖 QuantBot - 华尔街级全自动量化盯盘与风控机器 (Ultimate Edition)

这是一个专为 GitHub Actions 设计的零成本、免维护、带机器学习的量化多因子盯盘机器人。依托 GitHub 官方容器，7x24 小时监控全球市场异动，自带“凯利公式仓位管理”、“宏观信用避险”与“动态追踪止损”。

✨ 核心特性 (Features)

🆓 零成本永不停机：基于 GitHub Actions 定时触发（Cron），无服务器维护成本。

🧠 机器学习 AI 评分融合：自带 Scikit-Learn Logistic Regression 模型。通过分析历史交易日志，AI 会自动为您预测股票 T+3 的赚钱概率。

⚔️ RRG 行业动量轮动：智能感知标的属于“领涨”还是“拖累”板块，顺风扯满帆，逆风不翻船。

⚖️ 凯利公式真实仓位测算 (Kelly Criterion)：根据 AI 预测胜率与 ATR 动态盈亏比，精准计算每一笔交易的最优买入仓位占比，坚决拦截亏损博弈。

🩸 宏观信用风控与顶背离防御：

跨市场实时拉取 高收益垃圾债(HYG) 与 避险国债(IEF) 利差。系统性股灾前资金抽逃，系统会自动触发全仓熔断警告。

顶背离探测 (Bearish Divergence) 坚决防御短期诱多拉升。

🛡️ 极致地狱级回测引擎 (Backtest)：

移动追踪止损 (Trailing Stop) 模拟：随着股价走高，止损线自动上移保护利润，彻底终结“纸上富贵”。

跳空开盘强平模拟：真实模拟遇到重大利空开盘跌破止损线的“滑点踩踏”场景，只有经过地狱测试存活下来的因子，才是真金！

🚀 快速开始 (Quick Start)

只需 3 步，即可在您的 GitHub 账户下免费运行此机器人。

1. Fork 本仓库

点击页面右上角的 Fork 按钮，将本项目复制到您自己的 GitHub 账号下。

2. 配置环境变量 (Secrets)

进入您 Fork 后的仓库，点击 Settings -> Secrets and variables -> Actions -> New repository secret，添加以下通知配置：

Secret 名称

描述

是否必填

WEBHOOK_URL

钉钉/飞书/企业微信 Webhook 地址（支持逗号分隔配置多个）

选填

TELEGRAM_BOT_TOKEN

Telegram 机器人的 Token (通过 @BotFather 获取)

选填

TELEGRAM_CHAT_ID

Telegram 接收消息的 Chat ID

选填

3. 启用与运行

点击仓库顶部的 Actions 标签页，确认启用工作流。

在左侧列表中选择 Tickeron Pro AI Bot。

点击 Run workflow，您可以手动选择 matrix 模式进行一次全市场深度扫描。

🕒 自动化运行时间表

基于最高算力效率配置（UTC 时间，仅工作日执行）：

Sentinel (高频哨兵): 美东盘中 4 次精准扫描（过滤杂音）。

Matrix (矩阵评分): 每天仅在美东尾盘 16:15 运行，锁定日线最高确定性。

Backtest (回测与AI重训): 每周五 23:00 自动执行并重新训练逻辑回归模型。

⚠️ 免责声明 (Disclaimer)

本项目仅供编程学习与量化策略研究使用。所有技术指标与自动化产生的信号不构成任何投资建议。
