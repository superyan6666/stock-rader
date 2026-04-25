import numpy as np
import pandas as pd
from features.math_ops import safe_div, _robust_fft_ensemble, _robust_hurst

def test_safe_div():
    assert safe_div(10, 2) == 5.0
    assert safe_div(10, 0) == 0.0
    assert safe_div(100, 1, cap=50.0) == 50.0
    assert safe_div(-100, 1, cap=50.0) == -50.0
    assert safe_div(np.nan, 2) == 0.0

def test_robust_fft_ensemble():
    # Provide synthetic sine wave data
    t = np.linspace(0, 10, 200)
    prices = 100 + 10 * np.sin(2 * np.pi * 0.05 * t)
    res = _robust_fft_ensemble(prices, base_length=120, ensemble_count=3)
    assert isinstance(res, float)
    assert -1.0 <= res <= 1.0
    
    # short data
    assert _robust_fft_ensemble(np.array([1, 2, 3])) == 0.0

def test_robust_hurst():
    np.random.seed(42)
    # Generate geometric brownian motion
    returns = np.random.normal(0, 0.01, 200)
    prices = 100 * np.exp(np.cumsum(returns))
    
    h_med, h_iqr, reliable = _robust_hurst(prices)
    assert isinstance(h_med, float)
    assert isinstance(h_iqr, float)
    assert isinstance(reliable, bool)
