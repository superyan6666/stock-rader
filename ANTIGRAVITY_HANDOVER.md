 SYSTEM PROTOCOL: QuantBot 3.0 (Singularity Edition) Handover Context

To Model: Antigravity (or any succeeding AI Coding Agent)
Subject: Context Initialization & Architecture Rules for QuantBot 3.0
Project State: Production-Ready, ARM-Optimized, Fully Decoupled (IO/CPU).

1. 核心架构拓扑 (Architecture Topology)

当前系统由 10 个完美对齐的文件组成，不要随意更改文件职责划分：

quant_engine.py: [绝对主脑] 包含多进程/多线程并发引擎、特征计算(49维)、数据拉取路由(Alpaca -> yfinance)、因子评分矩阵与执行调度。

quant_transformer.py: [右脑结构] PyTorch 编写的 InfoNCE 架构 Transformer，处理时序截面并提取 16 维 Alpha 向量。

train_transformer.py: [代谢中枢] 每周末读取实盘沉淀的 .npz 缓冲区，执行 AMP 混合精度重训。

dashboard.py: [指挥仓] Streamlit 前端，强依赖 .quantbot_data/ 目录下的统计文件与 TCA 账本。

gateway_daemon.sh: [守护网关] 绑定 NUMA/Jemalloc 节点的执行器。

main.yml: [调度流水线] GitHub Actions CI/CD，定义了环境变量、依赖和 crontab 调度。

setup_arm_blas.sh: [物理加速] Oracle ARM 机器的 OpenBLAS/NumPy 底层编译脚本。

requirements.txt: 依赖锁定。

.gitignore: 缓存保护。

README.md: 极其详细的系统白皮书。

2. 绝对工程戒律 (CRITICAL DIRECTIVES - DO NOT VIOLATE)

在进行任何后续代码修改时，必须严格遵守以下红线，否则会导致生产环境崩溃：

🚫 Directive 1: 严禁破坏 GIL 绕过机制 (IPC & Concurrency)

网络拉取必须留在 ThreadPoolExecutor (IO Worker)。

特征计算必须留在 ProcessPoolExecutor (CPU Worker)，绝对禁止在 CPU Worker 中回传完整的 DataFrame。只能回传标量、字典或 numpy array。

主进程与子进程共享数据必须走 SharedDFCache (基于 /dev/shm + PyArrow Feather 的零拷贝)。

🚫 Directive 2: 并发 API 限速器隔离 (TokenBucket)

多进程 (forkserver) 会重新加载模块。禁止跨进程直接依赖 global _API_LIMITER。

必须维持现有的 _worker_pool_initializer 机制，为每个子进程分配独立的平分限额的 TokenBucket。

🚫 Directive 3: Pandas 内存污染防御 (In-place Mutation)

在 calculate_indicators 等入口，必须保留 df = df.copy()。由于底层回测传递的是 xs() 视图，任何类似 df['Col'] = df['Col'].ffill() 的隐式原地修改都会污染全局内存池并触发警告。

🚫 Directive 4: 极端数值安全 (Math Safety)

涉及任何除法计算 (如计算收益率、比率)，分母必须带上 + 1e-10 保护。

特征输出字典在打包前，必须经过 np.nan_to_num(..., nan=0.0, posinf=20.0, neginf=-20.0) 处理。

🚫 Directive 5: 多进程随机种子污染 (Random Seed State)

禁止使用全局 np.random.seed()。必须使用局部的 np.random.default_rng(seed=...)，以免在 CPU Worker 池中导致所有标的生成相同的扰动特征。

3. 当前数据流路由 (Data Routing State)

主通道: Alpaca Data v2 API (_fetch_from_alpaca)。

降级通道 (Fallback): 如果 Alpaca 报 422 (如无权访问某些指数 ^VIX) 或超时，系统将打印 [降级路由] 日志并平滑降级至 yfinance。

数据黑洞: 所有系统产生的文件 (JSONL, SQLite, MD, NPZ, PTH) 现已强制归拢于 .quantbot_data/ 目录。严禁在根目录乱写文件，否则 GitHub Artifacts 收集将报错。

4. Antigravity 接管后建议的潜在演进方向 (Future Roadmap)

作为后续接管的 AI，您可以协助开发者在以下方向平滑演进：

微观结构数据 (Tick-Level Data): 在 _io_fetch_worker 中引入 Alpaca 的 Level 2 Order Book (L2) 抓取逻辑，并新增微观结构因子。

异步并发极限提升 (Asyncio): 目前 IO 阶段使用的是 ThreadPool。可演进为纯 asyncio + aiohttp，将 32 线程并发上限解除至 1000+ 协程并发。

右脑 Transformer 拓扑优化: 探索增加 Time2Vec 时序嵌入，或结合 GNN (图神经网络) 处理行业板块的横向协方差联动。

[End of Handover Context]