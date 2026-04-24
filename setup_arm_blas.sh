#!/bin/bash
# 🚀 QuantBot 3.0 ARM 物理机 BLAS 向量化加速配置脚本
# 适用环境: Oracle Cloud ARM (Ampere Altra) + Ubuntu
# 目的: 解锁 NumPy 在 ARM 架构下的 NEON/SVE 硬件级 SIMD 加速

echo "[$(date +'%T')] 🛡️ 开始配置 ARM 硬件级向量化加速引擎..."

if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未检测到 python3，请先激活 conda/miniforge 环境。"
    exit 1
fi

echo "[$(date +'%T')] 📦 正在注入系统级 pkg-config 与 libopenblas-dev..."
sudo apt-get update
sudo apt-get install -y pkg-config cmake libopenblas-dev build-essential gfortran

echo "[$(date +'%T')] 🗑️ 正在卸载预编译的通用版 NumPy 并清理缓存..."
pip uninstall -y numpy pandas scikit-learn
pip cache purge

echo "[$(date +'%T')] ⚙️ 正在通过源码暴力重构 NumPy (耗时约 2-5 分钟，请耐心等待)..."
export CFLAGS="-O3 -mcpu=native"

pip install numpy==1.26.0 --no-binary numpy
pip install pandas==2.1.0 scikit-learn>=1.3.0 --no-binary pandas,scikit-learn

echo "[$(date +'%T')] ✅ 编译收尾，正在执行硬件加速特性链接诊断..."
python3 -c "
import numpy as np
print('='*40)
print('NumPy Config:')
np.show_config()
print('='*40)
"

echo "🎉 优化部署完成！"
