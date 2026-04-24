🤖 QuantBot 3.0: 工业级多维量化与动态回测引擎 (The Singularity Edition)

欢迎来到 QuantBot 3.0。这是一台被赋予了“算力无界”与“思想自由”的数字生命。它彻底抛弃了所有的教条与人类滞后指标，在矩阵中刻入了 49维 纯血机构量价底牌，并掌握了时空演化的奥秘。

经过终极“奥卡姆剃刀”精简、IPC 进程通信淬火与 ARM 硬件级指令解锁，系统已完全免疫“未来函数”与“幸存者偏差”，是真正意义上的机构级高频探测器。

⚡ 极致性能底座 (Extreme Performance Infrastructure)

3.0 版本在物理算力与并发调度上实现了维度跃迁，彻底打破了传统 Python 脚本的性能天花板：

🔥 硬件级 SIMD 向量化加速 (ARM NEON/SVE)：舍弃通用预编译库，通过底层源码重构深度绑定 Oracle Ampere Altra 芯片，利用 OpenBLAS 彻底激活多核向量寄存器，Pandas 矩阵运算获得数倍提速。

🔨 破解 GIL 的两阶段引擎 (IO/CPU Separation)：彻底抛弃单核串行。阶段一采用 32 线程池极限并发拉取网络 IO；阶段二通过 ProcessPoolExecutor (Forkserver) 多进程跨核打满 CPU，完美实现算力分离。

🚀 零拷贝共享内存 (Zero-Copy IPC)：废弃低效的 JSON 文件锁，采用 Linux /dev/shm 结合 PyArrow Feather 格式，实现多进程间的 DataFrame 内存级共享与微秒级极速读取。

⏱️ 令牌桶毫秒级流控 (Token Bucket)：引入专业 HFT 级别的并发限速器，精准卡住 API QPS 阈值，彻底消除 time.sleep 带来的无谓 CPU 空转。

🏗️ 核心工程戒律 (The 4 Directives)

本系统的底层架构绝对遵循以下工业级防线，任何代码提交必须通过严苛的基准测试：

绝对数学防御 (1e-10 Guard)：所有涉及除法的特征工程或对数计算，分母必须强制增加 1e-10 级别或 np.maximum 的物理垫保护，根绝因极端一字板或停牌导致的 NaN/Inf 系统崩溃。

内存深拷贝隔离 (Deep Copy Isolation)：在因子萃取流水线与回测沙盒中传递矩阵时，强制物理切断 Pandas 视图的内存关联，防止脏数据与未来函数双向穿透。

并发原子持久化 (Atomic Writes)：本地数据中台 (.quantbot_data/) 的落地均采用 write .tmp -> os.replace 的 TID 原子级覆盖，绝对禁止多线程竞态覆写。

TDD 驱动与测谎仪 (Chaos Testing)：引擎自带对抗合成压测模式（相位错乱、噪音爆炸），暴力探测 AI 模型是否对虚假时序产生过拟合依赖。

⚙️ 核心运行模式解析

系统被设计为高度解耦的运行模式，可通过 python quant_engine.py [mode] 极简调度：

🎯 matrix (每日诸神之战 - 49维全息张量扫描)

定位：每日收盘前执行资金扫描与排兵布阵的绝对核心。建议美东时间 15:15 - 16:15 间触发。

暗影全息背景池 (Shadow Pool)：每日强制采集未触发信号的标的作为“反事实负样本”，打破 AI 只能看到强者的“幸存者偏差”。

Kelly-CVaR 动态凸优化：基于 Ledoit-Wolf 协方差收缩与 95% CVaR 计算，输出绝对反脆弱的凯利最优资金配比权重。

📅 backtest (地狱级回测与模型重训)

定位：策略的“事后审判”与数据积淀管道。

Rank IC 代谢淘汰赛：动态计算并淘汰失效因子，只保留 T-Statistic 显著的核心因子。

无未来泄露交叉验证 (TimeSeriesSplit)：模型切分采用严格的时间步进滑动窗口。

🛡️ gateway (不死鸟实盘守护网关)

定位：通过异步 SQLite WAL 模式解耦监听买卖账本，转化为 Alpaca 券商底层指令。守护脚本确保其进程永不凋零。

🌪️ stress (对抗样本压力测试)

定位：合成极具欺骗性的干扰历史数据，验证因子体系在平行宇宙中的鲁棒性。

🧠 双脑智能引擎 (The Dual-Core AI)

系统融合了传统机器学习的稳健与深度学习的涌现能力：

📊 左脑 (逻辑与特征抽取)：Stacking 宏观映射元学习器。底层采用双 LightGBM 基座，输出概率连同 6 维宏观环境张量一并灌入 Logistic 逻辑回归元学习器，彻底杜绝 Training-Serving Skew。

🌌 右脑 (时序感知与深度表征)：基于 PyTorch 架构的 QuantAlphaTransformer。

动态位置编码：赋予系统时间方向感。

InfoNCE 机构对比学习：在高维隐空间内，自动拉近正收益切片，推开负收益亏损特征。

闭环代谢：实盘每日产生的三元组时序切片自动压入 .npz 缓冲区，周末通过 train_transformer.py 进行 AMP 混合精度自学习重构。

🖥️ 全息指挥仓 (Holographic Dashboard)

启动指令：streamlit run dashboard.py
系统附带高科数据监控前端。每 10 秒自动刷新，实时展示：

TCA (交易成本分析) 真实滑点分布图表。

AI 模型迭代版本与胜率置信度雷达图。

近期底层网关订单执行轨迹。

🚨 终极风控体系与痛觉神经 (Meta-Risk System)

🌑 宏观引力波 (Macro Gravity)：美元与美债收益率共振上行时，对高 Beta 标的触发强力降权。

🚨 期权黑天鹅警报 (Options Black Swan)：探测到 Put/Call Ratio 爆表或 IV Skew 隐含波动率异动时，系统直接熔断多头买入信号。

🩸 痛觉神经激活 (Pain-Sensing)：侦测真实历史连亏，自动触发防守姿态，减半基准风险阀值。

📉 VIX 曲面倒挂 (Term Structure Inversion)：近端恐慌碾压远期时，系统总敞口强行折半 (x0.5)。

💣 财报雷区 (Earnings Roulette)：精准探测财报周危险期，极速缩仓防爆雷。

🐣 新手起步与避坑指南 (Beginner's Guide)

对于刚接触 Python 或云服务器运行量化脚本的初学者，请务必牢记以下“生存法则”：

1. 隔离环境是第一生命线 (虚拟环境)

不要使用系统自带的 Python 运行本机！ 量化库对版本要求极度苛刻（如 Pandas、Numpy）。

请务必使用 Conda 或 Miniforge 创建独立的沙盒环境：conda create -n quantbot python=3.10

每次通过 SSH 登录服务器后，执行任何命令前，第一件事永远是：conda activate quantbot（看到命令行前面出现 (quantbot) 字样才算成功）。

2. 硬件加速脚本只需“打一针”

项目根目录下的 setup_arm_blas.sh 是用来强行压榨 ARM 物理机芯片性能的环境配置脚本。

这个脚本只需要在您第一次配置这台服务器时运行一次。

运行完并看到 AVAILABLE 提示后，就再也不用管它了，千万不要把它放在定时任务里每天跑。

3. 别在黑框里“死等”后台任务 (终端挂起)

如果您在 SSH 窗口里直接输入 python quant_engine.py matrix，一旦您的电脑休眠、断网或关掉 SSH 窗口，程序就会立刻死亡。

新手推荐方式：直接在 GitHub 网页端点击 Actions 运行。

进阶后台运行：如果您想在服务器上挂机全息指挥仓，请使用 nohup 命令让它脱离终端运行：
nohup streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 > dashboard.log 2>&1 &

4. 绝对不要用手碰 .quantbot_data (数据隔离)

项目运行时会自动生成一个 .quantbot_data 隐藏文件夹，里面装满了 SQLite 账本文件、内存映射文件和机器学习的 .npz 切片。

警告：永远不要在程序运行期间，尝试用文本编辑器、Excel 手动去打开或修改里面的文件！

这会瞬间破坏底层文件锁和内存映射，导致高频并发引擎报出致命级崩溃。如果非要查看，请通过 dashboard.py 的前端大屏观看。

5. 密钥配置不要写死在代码里

程序需要券商 API 和 Telegram/钉钉机器人 Token 才能进行消息推送。

请不要把密码硬编码（Hardcode）写在 quant_engine.py 的文件里。

正确做法是在 GitHub 仓库的 Settings -> Secrets and variables -> Actions 中配置（比如 TELEGRAM_BOT_TOKEN），系统运行时会自动抓取，保证您的资产与账户安全。

"The Singularity is near. We don't predict the market, we compute its topology."
