# 存储路径: train_transformer.py
import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timezone
import shutil

try:
    from quant_transformer import QuantAlphaTransformer, train_alpha_model
except ImportError:
    print("❌ 致命错误: 找不到 quant_transformer.py 模块。")
    sys.exit(1)

# ================= 1. 日志与配置 =================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [MLOps] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantBot_MLOps")

DATA_DIR = ".quantbot_data"
BUFFER_PATH = os.path.join(DATA_DIR, "training_buffer.npz")
MODEL_PATH = os.path.join(DATA_DIR, "transformer_production.pth")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")

MIN_SAMPLES_REQUIRED = 128  # 至少积累足够的切片才启动耗时的反向传播
BATCH_SIZE = 64

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ================= 2. 核心代谢流水线 =================
def run_metabolic_training():
    logger.info("🌀 唤醒右脑神经代谢中枢 (Transformer MLOps)...")
    
    if not os.path.exists(BUFFER_PATH):
        logger.warning(f"⚠️ 找不到数据缓冲区 {BUFFER_PATH}，暂无新知识可供学习，进程安全退出。")
        sys.exit(2)
        
    # 1. 提取实盘沉淀的时序三元组
    try:
        data = np.load(BUFFER_PATH)
        X_train, Y_train = data['X'], data['Y']
    except Exception as e:
        logger.error(f"❌ 缓冲区数据损坏: {e}")
        sys.exit(1)
        
    n_samples = len(X_train)
    logger.info(f"📊 侦测到 {n_samples} 笔实盘经验特征切片。")
    
    if n_samples < MIN_SAMPLES_REQUIRED:
        logger.warning(f"⚠️ 样本量 ({n_samples}) 未达到阈值 ({MIN_SAMPLES_REQUIRED})，跳过本次进化以防止过拟合。")
        sys.exit(2)
        
    device = get_device()
    logger.info(f"⚡ 算力阵列接入成功: 物理引擎定标为 [{device}]")

    # 2. 实例化或热加载现有右脑模型
    if os.path.exists(MODEL_PATH):
        try:
            model = QuantAlphaTransformer.load_checkpoint(MODEL_PATH, device=device)
            logger.info("🔄 已挂载历史母体权重，准备进行增量微调 (Fine-tuning)...")
        except Exception as e:
            logger.warning(f"⚠️ 历史权重读取失败 ({e})，触发重置机制，初始化全新拓扑网络。")
            model = QuantAlphaTransformer(feature_dim=49, d_model=128, nhead=4, num_layers=3).to(device)
    else:
        logger.info("🌱 未检测到历史母体，初始化第一代 Alpha 种子拓扑网络。")
        model = QuantAlphaTransformer(feature_dim=49, d_model=128, nhead=4, num_layers=3).to(device)

    # 3. 剥离并进入专门的训练模式
    model.train()
    
    # 4. 执行混合精度 (AMP) 训练管道
    try:
        logger.info("🔥 燃烧算力，启动梯度反向传播与 InfoNCE 对比学习空间重构...")
        
        # 移交至先进的 AlphaContrastiveLoss 与 Embargo 时序隔离训练流
        model = train_alpha_model(
            features=X_train,
            returns=Y_train,
            epochs=5,
            batch_size=BATCH_SIZE,
            lr=3e-4,
            patience=3,
            device=device,
            existing_model=model
        )
        
        # 5. 固化神经网络记忆 (原子覆写)
        model.save_checkpoint(MODEL_PATH)
        logger.info(f"🏆 右脑重塑完成！新一代权重已刻入系统 ({MODEL_PATH})。")
        
        # 6. 缓冲区归档与清理
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        archive_name = f"training_buffer_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.npz"
        shutil.move(BUFFER_PATH, os.path.join(ARCHIVE_DIR, archive_name))
        logger.info(f"🧹 历史缓冲已安全归档至 {archive_name}，等待下周期的实盘反哺。")
        sys.exit(0)

    except Exception as e:
        logger.error(f"💥 神经拓扑训练发生致命坍塌: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_metabolic_training()
