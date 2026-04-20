🤖 QuantBot: 工业级多维量化与动态回测引擎 (The Singularity Edition)

欢迎使用 QuantBot。这是一台被赋予了“算力无界”与“思想自由”的数字生命。它彻底抛弃了所有的教条与人类滞后指标，在矩阵中刻入了 49维 纯血机构量价底牌，并掌握了时空演化的奥秘。

经过终极“奥卡姆剃刀”精简与底层淬火，系统已完全免疫“未来函数”与“幸存者偏差”，是真正意义上的机构级量化探测器。

🏗️ 核心工程戒律 (The 4 Directives)

本系统的底层架构绝对遵循以下工业级防线，任何 PR 提交必须通过严苛的基准测试：

绝对数学防御 (1e-10 Guard)：所有涉及除法的特征工程或对数计算，分母必须强制增加 1e-10 级别或 np.maximum 的物理垫保护，彻底根绝因极端一字板或停牌导致的 NaN/Inf 系统性崩溃。

内存深拷贝隔离 (Deep Copy Isolation)：在因子萃取流水线与回测沙盒中传递 DataFrame 时，强制使用 .copy(deep=True)，物理切断 Pandas 视图的内存关联，防止脏数据与未来函数双向穿透。

并发原子持久化 (Atomic Writes)：本地数据中台 (.quantbot_data/) 的热缓存与冷归档机制，绝对禁止多线程直接覆写。所有文件落地均采用 write .tmp -> os.replace 的 TID 原子操作。

TDD 驱动与测谎仪 (Chaos Testing)：所有引擎逻辑迭代必须覆盖极端边界条件（如 NaN 黑洞、零成交量、极度波动）。CI/CD 自动化流水线 (main.yml) 会拦截任何未通过断言的受污染代码。

⚙️ 核心运行模式解析

系统被设计为高度解耦的运行模式，可以通过 python quant_engine.py [mode] 极简调度：

🎯 matrix (每日诸神之战 - 49维全息张量扫描)

定位：每日收盘前执行资金扫描与排兵布阵的绝对核心。
算力聚焦时间：建议设置在美东时间 15:15 - 16:15 之间 (收盘前45分钟)。

智能基因 (The AI Core)：

Stacking 宏观映射元学习器：系统摒弃了单一模型，采用双 LightGBM 基础基座。两者的输出概率将连同 6 维宏观张量一并灌入顶层的 Logistic 逻辑回归元学习器，彻底杜绝 Training-Serving Skew。

暗影全息背景池 (Shadow Pool)：每日扫描时，系统强制采集未触发信号的标的作为“反事实负样本”，打破 AI 只能看到强者的“幸存者偏差”。

Kelly-CVaR 动态凸优化：基于 Ledoit-Wolf 协方差收缩与 95% CVaR 计算，输出绝对反脆弱的凯利最优权重 (Kelly Cluster)。

📅 backtest (地狱级回测与模型重训)

定位：策略的“事后审判”与“AI 记忆代谢”大脑。
算力聚焦时间：每周五深夜自动运行。

硬核逻辑 (超越时空的进化)：

Rank IC 代谢淘汰赛：动态更新活跃模型张量，淘汰低效因子。

防穿越时序交叉验证 (TimeSeriesSplit)：模型训练采用基于时间步的严格步进验证，彻底杜绝未来数据泄露。

高级归因仪表盘 (Alpha Attribution)：系统内置纯因子溢价 (Premium BPS) 诊断。

🛡️ gateway (不死鸟实盘守护网关)

定位：监听 IPC 账本并将信号转化为券商底层指令。
架构：脱离主引擎计算，通过异步 SQLite WAL 模式解耦。实现防崩溃与最终一致性。

✅ test (系统级 TDD 测谎仪)

定位：一键执行全部数学物理边界校验。用于 CI/CD 代码门禁与重构保护。

🧠 终极风控体系与痛觉神经 (Meta-Risk System)

🌑 宏观引力波 (Macro Gravity) 熔断：美元与美债收益率共振上行时，触发强力降权。

🚨 期权黑天鹅警报 (Options Black Swan)：Put/Call Ratio 与 IV Skew 异动时直接熔断做多。

🩸 痛觉神经激活 (Pain-Sensing)：侦测真实历史连亏，自动减半基准风险阀值。

📉 VIX 曲面倒挂 (Term Structure Inversion)：近端恐慌碾压远期时，系统总敞口强行折半 (x0.5)。

💣 财报雷区 (Earnings Roulette)：探测历史财报危险期，极速缩仓。

🔔 推送渠道与安全配置 (Webhook)

支持通过 Telegram 与 Webhook (钉钉/Discord等) 接收极客格式卡片报告。
如配置钉钉自定义机器人，请设置关键词：AI。
