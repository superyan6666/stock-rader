import pandas as pd
import numpy as np
from typing import Any
from scipy.stats import gaussian_kde
from features.math_ops import _robust_fft_ensemble, _robust_hurst
from data.models import StockData, ComplexFeatures
from utils.logger import logger

KDE_AVAILABLE = True

def _extract_complex_features(stock: StockData, ctx: Any) -> ComplexFeatures:
    weekly_bullish, weekly_macd_res, fvg_lower, fvg_upper = False, 0.0, 0.0, 0.0
    aligned_w = stock.df_w
    if not aligned_w.empty and len(aligned_w) >= 40:
        if aligned_w['Close'].iloc[-1] > aligned_w['Close'].rolling(40).mean().iloc[-1]:
            ema10, ema30 = aligned_w['Close'].ewm(span=10, adjust=False).mean(), aligned_w['Close'].ewm(span=30, adjust=False).mean()
            weekly_bullish = (aligned_w['Close'].iloc[-1] > ema10.iloc[-1]) and (ema10.iloc[-1] > ema30.iloc[-1])
        macd_w = aligned_w['Close'].ewm(span=12, adjust=False).mean() - aligned_w['Close'].ewm(span=26, adjust=False).mean()
        hist_w = macd_w - macd_w.ewm(span=9, adjust=False).mean()
        if len(hist_w) >= 2 and hist_w.iloc[-1] > 0 and hist_w.iloc[-1] > hist_w.iloc[-2]: weekly_macd_res = 1.0
            
    n = len(stock.df)
    if n >= 22:
        lows, highs = stock.df['Low'].values, stock.df['High'].values
        valid_idx = np.where(lows[n-20:n-1] > highs[n-22:n-3])[0]
        if len(valid_idx) > 0:
            last_i = valid_idx[-1] + n - 20
            fvg_lower, fvg_upper = highs[last_i-2], lows[last_i]
    
    kde_breakout_score = 0.0
    if KDE_AVAILABLE and len(stock.df) >= 60:
        try:
            prices, volumes = stock.df['Close'].iloc[-60:].values, stock.df['Volume'].iloc[-60:].values
            if np.std(prices) > 1e-5:
                kde = gaussian_kde(prices, weights=volumes, bw_method='silverman')
                den_curr = kde.evaluate(prices[-1])[0]
                densities = kde.evaluate(np.linspace(prices.min(), prices.max(), 200))
                den_50 = np.percentile(densities, 50)
                if den_curr <= den_50: kde_breakout_score = min(1.0, 0.5 + 0.5 * (den_50 - den_curr) / (den_50 + 1e-10))
        except Exception as e: logger.debug(f"Exception suppressed: {e}")
        
    fft_ensemble_score = _robust_fft_ensemble(stock.df['Close'].values, base_length=120, ensemble_count=7)
    hurst_med, hurst_iqr, hurst_reliable = _robust_hurst(stock.df['Close'].values)
    
    monthly_inst_flow = 0.0
    if not stock.df_m.empty and len(stock.df_m) >= 3:
        m_flow = (stock.df_m['Close'] - stock.df_m['Open']) / (stock.df_m['High'] - stock.df_m['Low'] + 1e-10) * stock.df_m['Volume']
        if m_flow.iloc[-1] > 0 and m_flow.iloc[-2] > 0 and m_flow.iloc[-3] > 0: monthly_inst_flow = 1.0

    rsi_60m_bounce = 0.0
    if not stock.df_60m.empty and len(stock.df_60m) >= 15:
        delta = stock.df_60m['Close'].diff()
        rs = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean() / (-delta.where(delta < 0, 0).ewm(span=14, adjust=False).mean() + 1e-10)
        rsi_60m = 100 - (100 / (1 + rs))
        if len(rsi_60m) >= 2 and rsi_60m.iloc[-1] > rsi_60m.iloc[-2] and rsi_60m.iloc[-2] < 40: rsi_60m_bounce = 1.0
            
    stock_ret = stock.df['Close'].pct_change().fillna(0)
    beta_60d, spy_realized_vol = 1.0, 15.0 
    tlt_corr, dxy_corr = 0.0, 0.0
    rs_20, pure_alpha = 0.0, 0.0

    # 🚀 精准提取，统一兼容完整的 MarketContext 与序列化后的 MarketContextLite
    is_lite = hasattr(ctx, 'spy_close')
    qqq_c = ctx.qqq_close if is_lite else (ctx.qqq_df['Close'] if not ctx.qqq_df.empty else pd.Series(dtype=float))
    spy_c = ctx.spy_close if is_lite else (ctx.macro_data.get('spy', pd.DataFrame())['Close'] if 'spy' in ctx.macro_data and not ctx.macro_data.get('spy', pd.DataFrame()).empty else pd.Series(dtype=float))
    tlt_c = ctx.tlt_close if is_lite else (ctx.macro_data.get('tlt', pd.DataFrame())['Close'] if 'tlt' in ctx.macro_data and not ctx.macro_data.get('tlt', pd.DataFrame()).empty else pd.Series(dtype=float))
    dxy_c = ctx.dxy_close if is_lite else (ctx.macro_data.get('dxy', pd.DataFrame())['Close'] if 'dxy' in ctx.macro_data and not ctx.macro_data.get('dxy', pd.DataFrame()).empty else pd.Series(dtype=float))

    if not qqq_c.empty and len(stock.df) >= 60:
        m_df = pd.DataFrame({'stock': stock.df['Close'], 'qqq': qqq_c}).dropna()
        if len(m_df) >= 60:
            qqq_ret_20 = max(m_df['qqq'].iloc[-1] / m_df['qqq'].iloc[-20], 0.5)
            rs_20 = float((m_df['stock'].iloc[-1] / m_df['stock'].iloc[-20]) / qqq_ret_20)
            ret_s, ret_q = m_df['stock'].pct_change().dropna(), m_df['qqq'].pct_change().dropna()
            cov_m = np.cov(ret_s.iloc[-60:], ret_q.iloc[-60:])
            beta_l = cov_m[0,1] / (cov_m[1,1] + 1e-10) if cov_m[1,1] > 0 else 1.0
            pure_alpha = float((ret_s.iloc[-5:].mean() - beta_l * ret_q.iloc[-5:].mean()) * 252)

    if not spy_c.empty and len(stock_ret) >= 60:
        spy_ret = spy_c.pct_change().reindex(stock_ret.index).fillna(0)
        cov_mat = np.cov(stock_ret.iloc[-60:], spy_ret.iloc[-60:])
        if cov_mat[1,1] > 0: beta_60d = float(cov_mat[0, 1] / cov_mat[1, 1])
        spy_realized_vol = float(spy_ret.iloc[-20:].std() * np.sqrt(252) * 100)

    if not tlt_c.empty and len(stock_ret) >= 90:
        tlt_ret = tlt_c.pct_change().reindex(stock_ret.index).fillna(0)
        tlt_corr = float(stock_ret.iloc[-90:].corr(tlt_ret.iloc[-90:]))

    if not dxy_c.empty and len(stock_ret) >= 20:
        dxy_ret = dxy_c.pct_change().reindex(stock_ret.index).fillna(0)
        dxy_corr = float(stock_ret.iloc[-20:].corr(dxy_ret.iloc[-20:]))
        
    return ComplexFeatures(
        weekly_bullish=weekly_bullish, fvg_lower=fvg_lower, fvg_upper=fvg_upper, 
        kde_breakout_score=kde_breakout_score, fft_ensemble_score=fft_ensemble_score, 
        hurst_med=hurst_med, hurst_iqr=hurst_iqr, hurst_reliable=hurst_reliable, 
        monthly_inst_flow=monthly_inst_flow, weekly_macd_res=weekly_macd_res, 
        rsi_60m_bounce=rsi_60m_bounce, beta_60d=float(beta_60d), tlt_corr=float(tlt_corr if pd.notna(tlt_corr) else 0), 
        dxy_corr=float(dxy_corr if pd.notna(dxy_corr) else 0), vrp=float((ctx.vix_current - spy_realized_vol) / max(ctx.vix_current, 1.0)), 
        rs_20=float(rs_20), pure_alpha=float(pure_alpha)
    )
