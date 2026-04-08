🤖 Tickeron Pro AI Bot (Serverless 极简量化监控)

这是一个专为 GitHub Actions 设计的零成本、免维护量化多因子盯盘机器人。无需购买云服务器，依托 GitHub 官方容器，7x24 小时监控全球市场异动，并将投资机会实时推送到钉钉。

✨ 核心特性 (Features)

零成本托管：基于 GitHub Actions 定时触发（Cron），告别云服务器续费烦恼。

智能“三合一”架构：

sentinel 高频异动哨兵：每半小时监控核心自选股的暴涨/暴跌与跳空。

matrix 多因子信号矩阵：监控 MACD 金叉、RSI 背离、均线突破与布林挤压。

daily 全景复盘研报：每日收盘后扫描纳指 100，梳理多空排列阵型。

动态股票池抓取：自动从 Wikipedia 解析最新的 Nasdaq-100 成分股，永不脱节。

企业级稳健性：内置随机延迟请求、失败重试、logging 规范输出及严格的类型推断。

🚀 快速开始 (Quick Start)

1. Fork 本项目

点击右上角的 Fork 按钮，将此代码库复制到你的 GitHub 账号下。

2. 配置 Webhook 机器人

在钉钉中创建一个自定义机器人，安全设置建议勾选【自定义关键词】，并填入 AI。
复制生成的 Webhook URL。

3. 设置 GitHub Secrets

进入你 Fork 的仓库，点击 Settings > Secrets and variables > Actions。

点击 New repository secret。

Name 填入：WEBHOOK_URL

Secret 填入你刚才复制的机器人链接。

4. 激活工作流

点击仓库顶部的 Actions 标签页，启用工作流。

在左侧选择 Tickeron Pro AI Bot，点击右侧的 Run workflow。

在下拉菜单中选择 test 模式，点击运行进行连通性测试。

⚙️ 核心依赖

详见 requirements.txt：

yfinance: 金融数据获取

pandas: 数据处理与指标计算

lxml & requests: 维基百科爬虫

📄 协议 (License)

MIT License
