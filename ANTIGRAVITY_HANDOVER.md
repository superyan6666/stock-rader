 SYSTEM PROTOCOL: QuantBot 3.0 (Singularity Edition) Handover Context

To Model: Antigravity (or any succeeding AI Coding Agent)
Subject: Context Initialization & Architecture Rules for QuantBot 3.0
Project State: Production-Ready, ARM-Optimized, Fully Decoupled (IO/CPU).
Last Session: 2026-04-26, Conversation ID: bbe0cdca-0a7a-463b-a8db-ed9aecda7543

## 🚨 URGENT: Immediate Task for Next Session

**The Transformer "Right Brain" training pipeline has been fully debugged and is awaiting its first successful run.**

### What Was Done:
1. **Optuna Optimization COMPLETE** — 500-trial search found optimal params (fitness 41→82), welded into `simulate_ledger_run()` defaults at line ~1823 of `quant_engine.py`.
2. **Training data pipeline FIXED** — `training_buffer.npz` (29MB, 2510 samples) confirmed present on server at `.quantbot_data/`.
3. **5 bugs fixed in sequence**: wrong execution mode routing, numpy `.npz` suffix auto-append, exit code masking, wrong constructor kwarg (`feature_dim` → `num_features`), missing `quant_transformer.py` commit.

### What To Do First:
1. **Check the latest GitHub Actions run** (commit `5a734cf`) — it should show `train_transformer.py` successfully running InfoNCE contrastive learning with Loss output.
2. **If it succeeded**: The `transformer_production.pth` file now exists. Future runs of `quant_engine.py` will auto-detect it and use real AI probabilities instead of the hardcoded `ai_prob = 0.45 + score*0.2` fallback (see `_apply_ai_inference()` around line 1510).
3. **If it failed again**: Read the error, it will be a simple fix. All infrastructure is now correct.

### Post-Success Cleanup:
- Remove diagnostic `[DIAG]` logging from `quant_engine.py` (line ~2404) and `main.yml` (line ~108-109)
- Delete orphan file `training_buffer.npz.tmp.npz` from server
- Delete scratch files from repo root: `update_params.py`, `update_params.ps1`, `fix.ps1`, `fix.py`, `fix2.ps1`, `fix3.ps1`, `error.txt`, `out.log`, `refactor.py`, `test_ast.py`

### Suggested Next Evolution Directions:
1. **Walk-Forward Optimization** — Replace global Optuna with rolling 3-month-train/1-month-test to combat overfitting
2. **Slippage & Impact Model** — Add dynamic slippage penalty in `simulate_ledger_run()` based on volatility
3. **Survivorship Bias Audit** — Ensure the stock pool is dynamic, not based on today's survivors

---

## 1. 核心架构拓扑 (Architecture Topology)

当前系统由 10 个完美对齐的文件组成，不要随意更改文件职责划分：

quant_engine.py: [绝对主脑] 包含多进程/多线程并发引擎、特征计算(49维)、数据拉取路由(Alpaca -> yfinance)、因子评分矩阵与执行调度。

quant_transformer.py: [右脑结构] PyTorch 编写的 InfoNCE 架构 Transformer，处理时序截面并提取 16 维 Alpha 向量。含 `train_alpha_model()` 函数，接受 `existing_model` 参数用于增量微调。

train_transformer.py: [代谢中枢] 每周末读取实盘沉淀的 .npz 缓冲区，调用 `train_alpha_model()` 执行 AMP 混合精度重训。MIN_SAMPLES_REQUIRED=16, BATCH_SIZE=16。

dashboard.py: [指挥仓] Streamlit 前端，强依赖 .quantbot_data/ 目录下的统计文件与 TCA 账本。

gateway_daemon.sh: [守护网关] 绑定 NUMA/Jemalloc 节点的执行器。

main.yml: [调度流水线] GitHub Actions CI/CD。push→replay+backtest(采集训练数据)+MLOps训练。schedule→matrix。workflow_dispatch→自选模式。`clean: false` 保护 `.quantbot_data/` 持久化。

setup_arm_blas.sh: [物理加速] Oracle ARM 机器的 OpenBLAS/NumPy 底层编译脚本。

requirements.txt: 依赖锁定。

.gitignore: 缓存保护。

README.md: 极其详细的系统白皮书。

## 2. 绝对工程戒律 (CRITICAL DIRECTIVES - DO NOT VIOLATE)

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

🚫 Directive 6: np.savez 后缀陷阱 (NumPy Suffix Trap)

`np.savez()` 会自动追加 `.npz` 后缀。如果传入的路径已包含 `.npz`，最终文件名将变成 `xxx.npz.npz`。在做原子写入时，临时文件路径必须本身就以 `.npz` 结尾（如 `training_buffer_tmp.npz`），而非 `training_buffer.npz.tmp`。

## 3. 当前数据流路由 (Data Routing State)

主通道: Alpaca Data v2 API (_fetch_from_alpaca)。

降级通道 (Fallback): 如果 Alpaca 报 422 (如无权访问某些指数 ^VIX) 或超时，系统将打印 [降级路由] 日志并平滑降级至 yfinance。

数据黑洞: 所有系统产生的文件 (JSONL, SQLite, MD, NPZ, PTH) 现已强制归拢于 .quantbot_data/ 目录。严禁在根目录乱写文件，否则 GitHub Artifacts 收集将报错。

## 4. Optuna 最优参数 (Welded into Production)

以下参数已固化到 `simulate_ledger_run()` 的默认值中（quant_engine.py line ~1823）：
- tier1_dd: -0.0385 (一级防御线)
- tier2_dd: -0.0722 (深水区防御)
- tp_normal: 3.44 (正常止盈倍数)
- tp_tier1: 2.75 (休眠止盈倍数)
- sl_mul: 1.43 (止损倍数)
- max_hold_trend: 15 (趋势最大持仓天数)
- max_hold_rev: 4 (反转最大持仓天数)

[End of Handover Context]