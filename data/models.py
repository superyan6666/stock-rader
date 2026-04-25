import pandas as pd
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MarketContext:
    regime: str; regime_desc: str; w_mul: float; xai_weights: dict; vix_current: float
    vix_desc: str; vix_scalar: float; max_risk: float; macro_gravity: bool
    is_credit_risk_high: bool; vix_inv: bool; qqq_df: pd.DataFrame; macro_data: dict
    total_market_exposure: float; health_score: float; pain_warning: str
    credit_spread_mom: float = 0.0; vix_term_structure: float = 1.0; market_pcr: float = 1.0
    dynamic_min_score: float = 8.0; global_wsb_data: dict = field(default_factory=dict)
    meta_weights: dict = field(default_factory=dict); transformer_model: 'Any' = None  

@dataclass
class MarketContextLite:
    """可安全跨进程序列化的纯值上下文（传递轻量级 Series，完美闭环宏观指标计算）"""
    regime: str; w_mul: float; xai_weights: dict; vix_current: float
    macro_gravity: bool; health_score: float; dynamic_min_score: float
    qqq_close: pd.Series; spy_close: pd.Series; tlt_close: pd.Series; dxy_close: pd.Series

def _build_ctx_lite(ctx: MarketContext) -> MarketContextLite:
    """从完整 ctx 蒸馏出可序列化的轻量版本"""
    def _safe_close(df): return df['Close'] if not df.empty else pd.Series(dtype=float)
    return MarketContextLite(
        regime=ctx.regime, w_mul=ctx.w_mul, xai_weights=ctx.xai_weights,
        vix_current=ctx.vix_current, macro_gravity=ctx.macro_gravity,
        health_score=ctx.health_score, dynamic_min_score=ctx.dynamic_min_score,
        qqq_close=_safe_close(ctx.qqq_df),
        spy_close=_safe_close(ctx.macro_data.get('spy', pd.DataFrame())),
        tlt_close=_safe_close(ctx.macro_data.get('tlt', pd.DataFrame())),
        dxy_close=_safe_close(ctx.macro_data.get('dxy', pd.DataFrame()))
    )

@dataclass
class StockData:
    sym: str; df: pd.DataFrame; df_w: pd.DataFrame; df_m: pd.DataFrame; df_60m: pd.DataFrame
    curr: pd.Series; prev: pd.Series; is_vol: bool; swing_high_10: float

@dataclass
class AltData:
    pcr: float; iv_skew: float; short_change: float; short_float: float
    insider_net_buy: float; analyst_mom: float; nlp_score: float; wsb_accel: float

@dataclass
class ComplexFeatures:
    weekly_bullish: bool; fvg_lower: float; fvg_upper: float; kde_breakout_score: float
    fft_ensemble_score: float; hurst_med: float; hurst_iqr: float; hurst_reliable: bool
    monthly_inst_flow: float; weekly_macd_res: float; rsi_60m_bounce: float
    beta_60d: float; tlt_corr: float; dxy_corr: float; vrp: float; rs_20: float; pure_alpha: float 
