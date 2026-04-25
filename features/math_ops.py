import numpy as np
import pandas as pd
from typing import Tuple

def safe_div(num, den, cap=20.0):
    if pd.isna(num) or pd.isna(den) or den == 0: return 0.0
    return max(min(num / den, cap), -cap)

def _robust_fft_ensemble(close_prices: np.ndarray, base_length=120, ensemble_count=7) -> float:
    if len(close_prices) < base_length + (ensemble_count // 2) * 5: return 0.0
    votes = []
    for offset in range(ensemble_count):
        win_len = base_length + (offset - ensemble_count//2) * 5
        segment = close_prices[-win_len:]
        if np.isnan(segment).any() or np.all(segment == segment[0]): continue
        fft_res = np.fft.fft(segment - np.mean(segment))
        freqs = np.fft.fftfreq(win_len)
        pos_mask = (freqs > 0.01) & (freqs < 0.2)
        if not np.any(pos_mask): continue
        pos_idx = np.where(pos_mask)[0]
        peak_idx = pos_idx[np.argmax(np.abs(fft_res[pos_idx]))]
        phase = np.angle(fft_res[peak_idx])
        t = win_len - 1
        dom_freq = freqs[peak_idx]
        current_phase = (2 * np.pi * dom_freq * t + phase) % (2 * np.pi)
        if 0 < current_phase < np.pi: votes.append(1.0)
        else: votes.append(-1.0)
    if not votes: return 0.0
    return float(sum(votes) / len(votes))

def _robust_hurst(close_prices: np.ndarray, min_window=30, n_bootstrap=100) -> Tuple[float, float, bool]:
    safe_prices = np.maximum(close_prices, 1e-10)
    log_ret = np.diff(np.log(safe_prices[-121:])) if len(safe_prices) > 120 else np.diff(np.log(safe_prices))
    if len(log_ret) <= min_window: return 0.5, 0.0, False
    
    # ✅ 修复: 使用局部 Generator 替代全局 seed，零污染多进程的全局随机状态
    rng = np.random.default_rng(seed=int(np.sum(np.abs(close_prices[-10:])) * 1e6) % (2**31))
    
    hurst_samples = []
    for _ in range(n_bootstrap):
        sub_len = int(rng.integers(min_window, len(log_ret)))
        start = int(rng.integers(0, len(log_ret) - sub_len))
        sub_ret = log_ret[start:start+sub_len]
        S = np.std(sub_ret)
        if S == 0: continue
        R = np.max(np.cumsum(sub_ret - np.mean(sub_ret))) - np.min(np.cumsum(sub_ret - np.mean(sub_ret)))
        if R <= 0: continue
        hurst_samples.append(np.log(R/S) / np.log(len(sub_ret)))
        
    if len(hurst_samples) < 20: return 0.5, 0.0, False
    h_median, h_iqr = float(np.median(hurst_samples)), float(np.percentile(hurst_samples, 75) - np.percentile(hurst_samples, 25))
    is_reliable = bool((h_iqr < 0.15) and (abs(h_median - 0.5) > 0.1))
    return h_median, h_iqr, is_reliable
