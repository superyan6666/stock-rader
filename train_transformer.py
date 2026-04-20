# 存储路径: train_transformer.py
import numpy as np
import os
import logging
import torch
import warnings
import time
import shutil
import argparse
import sys  # 🚀 引入 sys 模块控制进程级 Exit Code
from datetime import datetime
from torch.utils.data import DataLoader
from quant_transformer import QuantAlphaTransformer, train_alpha_model, QuantContrastiveDataset, AlphaContrastiveLoss

try:
    from quantbot import send_alert
except ImportError:
    def send_alert(title: str, content: str):
        print(f"\n🔔 [本地模拟推送] {title}\n{content}\n")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Trainer] %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")

DATA_DIR = ".quantbot_data"
BUFFER_PATH = os.path.join(DATA_DIR, "training_buffer.npz")
MODEL_PATH = os.path.join(DATA_DIR, "transformer_production.pth")

def evaluate_model_loss(model: QuantAlphaTransformer, X_val: np.ndarray, Y_val: np.ndarray, device: torch.device) -> float:
    """🚀 核心测谎仪：对模型在最新的验证集上执行对比学习 Loss 结算"""
    model.eval()
    criterion = AlphaContrastiveLoss(temperature=0.1).to(device)
    dataset = QuantContrastiveDataset(X_val, Y_val)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    val_loss = 0.0
    with torch.no_grad():
        for anchor, pos, neg in dataloader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type=='cuda')):
                loss = criterion(model(anchor), model(pos), model(neg))
            val_loss += loss.item()
            
    return val_loss / len(dataloader) if len(dataloader) > 0 else float('inf')

def run_offline_training(args: argparse.Namespace):
    logger.info("🚀 启动独立深度学习代谢神经中枢 (Offline Training Node)...")
    
    if not os.path.exists(BUFFER_PATH):
        logger.error(f"❌ 找不到训练数据缓冲区: {BUFFER_PATH}。")
        sys.exit(1)  # 致命错误退出

    # 缓冲避让时间拉长至 30 秒，确保大量 I/O 写入绝对完成
    mtime = os.path.getmtime(BUFFER_PATH)
    if time.time() - mtime < 60.0:
        logger.warning("⏳ 探测到缓冲区在一分钟内被修改，休眠 30 秒以确保主引擎 I/O 句柄完全释放...")
        time.sleep(30.0)

    try:
        data = np.load(BUFFER_PATH)
        X_arr, Y_arr = data['X'], data['Y']
        
        finite_mask = np.isfinite(X_arr).all(axis=(1, 2)) & np.isfinite(Y_arr)
        valid_count = np.sum(finite_mask)
        X_arr, Y_arr = X_arr[finite_mask], Y_arr[finite_mask]
        logger.info(f"🧹 数据清洗完毕: 保留 {valid_count} 个有效样本 (切除 {len(finite_mask) - valid_count} 个毒药样本)")
        
        if len(X_arr) < 64:
            logger.warning("⚠️ 清洗后样本量过少，强制跳过代谢。")
            sys.exit(2)  # 🚀 Exit Code 2: 样本不足
            
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
        new_model = train_alpha_model(
            features=X_arr, returns=Y_arr, 
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, device=device
        )
        
        split_idx = int(len(X_arr) * 0.8)
        X_val, Y_val = X_arr[split_idx:], Y_arr[split_idx:]
        should_save_new_model = True
        
        report_lines = [
            f"**⚙️ MLOps 训练执行超参:**\n- 批次大小: `{args.batch_size}` | 学习率: `{args.lr}`\n- 清洗后有效切片: `{valid_count}` 笔",
            f"**⚡ 算力阵列:** `{device}` (混合精度 AMP 引擎)",
        ]
        
        if os.path.exists(MODEL_PATH) and len(X_val) > 10:
            try:
                old_model = QuantAlphaTransformer.load_checkpoint(MODEL_PATH, device=device)
                old_loss = evaluate_model_loss(old_model, X_val, Y_val, device)
                new_loss = evaluate_model_loss(new_model, X_val, Y_val, device)
                
                report_lines.append(f"\n### ⚔️ 冠军卫冕战 (Validation Loss)\n- 🏆 线上旧模型 (Champion): **{old_loss:.4f}**\n- 🤺 挑战者模型 (Challenger): **{new_loss:.4f}**")
                
                if new_loss < old_loss * 0.99:
                    report_lines.append("\n**🎉 判决: 挑战成功！新隐空间拓扑已获准接管实盘推流。**")
                    
                    # 在覆盖之前，物理备份旧模型
                    backup_path = MODEL_PATH + f".backup_{datetime.now().strftime('%Y%m%d')}"
                    shutil.copy2(MODEL_PATH, backup_path)
                    logger.info(f"💾 [防线] 已将旧模型冠军安全备份至: {backup_path}")
                    
                else:
                    report_lines.append("\n**🛡️ 判决: 挑战失败！候选权重被视为过拟合毒药，已被系统物理销毁。**")
                    should_save_new_model = False
            except Exception as e:
                report_lines.append(f"\n**⚠️ 冠军卫冕战异常 ({e})，强制免测放行新模型。**")
        else:
            report_lines.append("\n**🆕 初次部署或验证集不足，免测放行新模型上线。**")

        if should_save_new_model:
            new_model.save_checkpoint(MODEL_PATH)
            logger.info(f"✅ 全新隐空间权重已下发至: {MODEL_PATH}")
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(DATA_DIR, f"archive_buffer_{timestamp_str}.npz")
        shutil.copy2(BUFFER_PATH, archive_path)
        
        if os.path.exists(archive_path):
            os.remove(BUFFER_PATH)
            
        send_alert("🌌 深度学习右脑代谢战报 (MLOps)", "\n".join(report_lines))
        
        if not should_save_new_model:
            sys.exit(3)  # 🚀 Exit Code 3: 挑战失败拦截
            
    except SystemExit as se:
        raise se  # 允许合法的 sys.exit 冒泡到操作系统
    except Exception as e:
        send_alert("🚨 深度右脑神经代谢崩溃", f"❌ 异常: {e}")
        logger.error(f"❌ 训练管线崩溃: {e}")
        sys.exit(1)  # 🚀 致命错误退出
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantBot 3.0 - 深度神经代谢中枢")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    
    run_offline_training(args)
    sys.exit(0) # 🚀 一切顺利，成功进化
