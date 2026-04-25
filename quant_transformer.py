# 存储路径: quant_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import logging
import os
import warnings
import copy

# 忽略潜在的 PyTorch 硬件加速警告
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Transformer] %(levelname)s - %(message)s')
logger = logging.getLogger("QuantTransformer")

# ================= 1. 核心数学组件 =================

class Time2Vec(nn.Module):
    """Time2Vec: 让模型具备捕捉市场周期性(Periodicity)与线性趋势的能力"""
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        # 线性分量：捕捉长期趋势
        self.wb = nn.Parameter(torch.randn(input_dim, 1))
        self.bb = nn.Parameter(torch.randn(1))
        # 周期分量：通过正弦函数捕捉季节性与循环
        self.wa = nn.Parameter(torch.randn(input_dim, out_dim - 1))
        self.ba = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq, Features)
        bias = torch.matmul(x, self.wb) + self.bb
        wgts = torch.sin(torch.matmul(x, self.wa) + self.ba)
        return torch.cat([bias, wgts], dim=-1) # 融合线性与周期分量

class PositionalEncoding(nn.Module):
    """位置编码器：让 Transformer 具备时间方向感。"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 120):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / (d_model + 1e-10)))
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        pe[:, 1::2] = cos_term[:, :pe[:, 1::2].shape[1]]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AlphaContrastiveLoss(nn.Module):
    """机构级对比学习损失函数 (InfoNCE 变体)。"""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) 
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

# ================= 2. 深度学习右脑拓扑 =================

class QuantAlphaTransformer(nn.Module):
    """高维时序特征蒸馏器 (The Right Brain)。"""
    def __init__(self, num_features: int = 49, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.2, alpha_dim: int = 16):
        super().__init__()
        self.hp = {
            'num_features': num_features, 'd_model': d_model, 'nhead': nhead,
            'num_layers': num_layers, 'dropout': dropout, 'alpha_dim': alpha_dim
        }
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(num_features)
        
        # 🚀 [架构进化] 引入 Time2Vec 替代原始线性投影
        t2v_dim = 16 # 分配 16 维用于捕捉周期性特征
        self.t2v = Time2Vec(num_features, t2v_dim)
        self.feature_projection = nn.Linear(num_features + t2v_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=120)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.time_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.time_query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_uniform_(self.time_query)
        self.query_norm = nn.LayerNorm(d_model)
        
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, alpha_dim),
            nn.Tanh()
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        assert src.size(1) == 60, f"输入时序长度必须为 60，当前: {src.size(1)}"
        assert src.size(2) == self.hp['num_features'], f"特征维度异常，期待 {self.hp['num_features']}，实际: {src.size(2)}"
        
        src = self.input_norm(src)
        
        # 融合原始特征与 Time2Vec 周期嵌入
        t2v_embed = self.t2v(src)
        combined = torch.cat([src, t2v_embed], dim=-1)
        
        x = self.feature_projection(combined) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        
        query = self.time_query.expand(memory.size(0), -1, -1)
        query = self.query_norm(query) 
        pooled_memory, _ = self.time_attn(query, memory, memory)
        pooled_memory = pooled_memory.squeeze(1)
        
        alpha_vector = self.alpha_head(pooled_memory)
        return alpha_vector

    @torch.inference_mode()
    def extract_alpha(self, features: np.ndarray) -> np.ndarray:
        self.eval() 
        device = next(self.parameters()).device
        # 移除 numpy 的冗余 copy()，结合 inference_mode 极速直读内存
        tensor = torch.from_numpy(features).float().to(device)
        if tensor.dim() == 2: tensor = tensor.unsqueeze(0)
        
        # 强制内存连续化布局，配合 ARM CPU L1/L2 Cache 预取机制
        tensor = tensor.contiguous()
        output = self.forward(tensor)
        return output.cpu().numpy()

    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-3) -> torch.optim.Optimizer:
        import torch.optim as optim
        return optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temp_path = f"{path}.{os.getpid()}.tmp"
        checkpoint = {'hyperparameters': self.hp, 'state_dict': self.state_dict()}
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
            logger.info(f"💾 模型已安全保存至: {path}")
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            logger.error(f"❌ 模型保存失败: {e}")

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None) -> 'QuantAlphaTransformer':
        if not os.path.exists(path): raise FileNotFoundError(f"找不到权重文件: {path}")
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint.get('hyperparameters', {}))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        # 🚀 [ARM 终极推理加速] PyTorch 2.0+ Inductor 后端算子融合编译
        import sys
        if hasattr(torch, 'compile') and sys.platform != "win32":
            try:
                # 动态生成 C++ 算子并融合，针对 ARM NEON 自动进行底层 SIMD 调优
                model = torch.compile(model, backend="inductor", mode="reduce-overhead")
                logger.info("⚡ 已激活 torch.compile (Inductor) 算子融合与 ARM 硬件级编译！")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile 加速启动失败，已平滑降级为普通推理: {e}")
                
        logger.info(f"🔄 模型成功热加载。计算设备: {device}")
        return model

# ================= 3. 全栈式数据与训练管线 =================

class QuantContrastiveDataset(Dataset):
    def __init__(self, raw_features: np.ndarray, returns: np.ndarray, threshold: float = 0.02):
        super().__init__()
        self.features = torch.from_numpy(raw_features.copy()).float()
        self.returns = returns.copy()
        self.threshold = threshold 
        
        self.pos_idx = np.where(self.returns > threshold)[0]
        self.neg_idx = np.where(self.returns < -threshold)[0]
        self.anchor_idx = np.arange(len(self.returns))
        
        if len(self.pos_idx) == 0: self.pos_idx = self.anchor_idx
        if len(self.neg_idx) == 0: self.neg_idx = self.anchor_idx

    def __len__(self): return len(self.anchor_idx)

    def __getitem__(self, idx):
        a_id = self.anchor_idx[idx]
        anchor = self.features[a_id]
        
        noise = torch.randn_like(anchor) * 0.02  
        positive = anchor + noise
        
        anchor_ret = self.returns[a_id]
        if anchor_ret > self.threshold: n_id = np.random.choice(self.neg_idx)
        elif anchor_ret < -self.threshold: n_id = np.random.choice(self.pos_idx)
        else: n_id = np.random.choice(np.concatenate([self.pos_idx, self.neg_idx]))
            
        negative = self.features[n_id]
        return anchor, positive, negative

def train_alpha_model(features: np.ndarray, returns: np.ndarray, 
                      epochs: int = 50, batch_size: int = 64, lr: float = 3e-4, 
                      patience: int = 5, device: torch.device = None) -> QuantAlphaTransformer:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    
    # 🚀 [抗过拟合进化] Purged & Embargoed 时序切分
    # 逻辑：验证集必须与训练集保持至少 60 天的物理隔离，防止滑动窗口导致的信息泄露
    lookback = 60
    total_len = len(features)
    train_end = int(total_len * 0.75) # 提高训练集比重，但由于 Embargo，实际训练样本会减少
    val_start = train_end + lookback  # 强制隔离带 (Embargo)
    
    if val_start >= total_len - 10:
        logger.warning("⚠️ 样本量不足以建立封锁区，将回退至常规时序切分。")
        train_end = int(total_len * 0.8)
        val_start = train_end
        
    train_ds = QuantContrastiveDataset(features[:train_end], returns[:train_end])
    val_ds = QuantContrastiveDataset(features[val_start:], returns[val_start:])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    logger.info(f"📊 训练流状态: 训练集[{train_end}] | 封锁隔离带[{lookback}] | 验证集[{total_len - val_start}]")
    
    model = QuantAlphaTransformer().to(device)
    criterion = AlphaContrastiveLoss(temperature=0.1).to(device)
    optimizer = model.get_optimizer(lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    logger.info(f"🚀 启动混合精度全自动训练流 | 设备: {device.type} | AMP: {use_amp} | 验证集比重: 20%")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(anchor), model(pos), model(neg))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, pos, neg in val_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(anchor), model(pos), model(neg))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {curr_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.warning(f"🛑 触发早停机制 (Early Stopping)！验证集连续 {patience} 轮未优化。")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"✨ 训练管线安全终结，已回滚至最佳验证态权重 (Val Loss: {best_val_loss:.4f})")
    
    return model

if __name__ == "__main__":
    logger.info("🧪 启动混合精度全栈训练闭环压测...")
    
    N = 2000
    mock_features = np.random.randn(N, 60, 49)
    mock_returns = np.random.laplace(0, 0.03, N)
    
    try:
        trained_model = train_alpha_model(
            features=mock_features,
            returns=mock_returns,
            epochs=10, 
            batch_size=128,
            patience=3 
        )
        
        test_path = ".quantbot_data/transformer_production.pth"
        trained_model.save_checkpoint(test_path)
        if os.path.exists(test_path): os.remove(test_path)
        
        logger.info("✅ 全管线验收通过！")
    except Exception as e:
        logger.error(f"❌ 训练管线崩溃：{e}")
        raise e
