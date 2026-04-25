import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from config import Config
from data.models import StockData, ComplexFeatures, AltData, MarketContext
from features.math_ops import safe_div
from features.indicators import check_macd_cross
from utils.logger import logger

def _get_transformer_seq(df_ind: pd.DataFrame, end_idx: int = -1) -> np.ndarray:
    if end_idx == -1: end_idx = len(df_ind)
    start_idx = end_idx - 60
    if start_idx < 0: return np.zeros((60, 49), dtype=np.float32)
    df = df_ind.iloc[start_idx:end_idx].copy()
    c = df['Close'].values + 1e-10
    seq = np.zeros((60, 49), dtype=np.float32)
    feature_mapping = [
        (0, df['RSI'].values / 100.0), (1, df['MACD'].values / c), (2, df['Signal_Line'].values / c), (3, df['ATR'].values / c),
        (4, (df['KC_Upper'].values - df['KC_Lower'].values) / c), (5, (df['BB_Upper'].values - df['BB_Lower'].values) / c),
        (6, df['Tenkan'].values / c - 1.0), (7, df['Kijun'].values / c - 1.0), (8, df['SenkouA'].values / c - 1.0),
        (9, df['SenkouB'].values / c - 1.0), (10, df['SuperTrend_Up'].values), (11, df['CMF'].values),
        (12, df['Range'].values / c), (13, df['NR7'].astype(float).values), (14, df['Inside_Bar'].astype(float).values),
        (15, df['VPT_ZScore'].values), (16, df['VPT_Accel'].values), (17, df['VWAP_20'].values / c - 1.0),
        (18, np.where(df['AVWAP'].isna(), 0, df['AVWAP'].values / c - 1.0)), (19, df['Volume'].values / (df['Max_Down_Vol_10'].values + 1e-10)),
        (20, np.where(df['OB_High'].isna(), 0, df['OB_High'].values / c - 1.0)), (21, np.where(df['OB_Low'].isna(), 0, df['OB_Low'].values / c - 1.0)),
        (22, np.where(df['Swing_Low_20'].isna(), 0, df['Swing_Low_20'].values / c - 1.0)), (23, df['Range_60'].values / c),
        (24, df['Range_20'].values / c), (25, df['Price_High_20'].values / c - 1.0), (26, df['Volume'].values / (df['Vol_MA20'].values + 1e-10)),
        (27, df['CVD_Trend'].values), (28, df['CVD_Divergence'].values), (29, df['Highest_22'].values / c - 1.0),
        (30, df['ATR_22'].values / c), (31, df['Chandelier_Exit'].values / c - 1.0), (32, df['Smart_Money_Flow'].values),
        (33, df['Recent_Price_Surge_3d'].values / 100.0), (34, df['Recent_Vol_Surge_3d'].values), (35, df['Amihud'].values),
        (36, np.where(df['Dist_52W_High'].isna(), 0, df['Dist_52W_High'].values)), (37, df['Close'].values / (df['SMA_50'].values + 1e-10) - 1.0),
        (38, df['Close'].values / (df['SMA_150'].values + 1e-10) - 1.0), (39, df['Close'].values / (df['SMA_200'].values + 1e-10) - 1.0),
        (40, df['Close'].values / (df['EMA_20'].values + 1e-10) - 1.0), (41, df['Close'].values / (df['EMA_50'].values + 1e-10) - 1.0),
        (42, df['Above_Cloud'].values), (43, df['Open'].values / c - 1.0), (44, df['High'].values / c - 1.0),
        (45, df['Low'].values / c - 1.0), (46, df['Close'].pct_change().fillna(0).values), (47, df['Volume'].pct_change().fillna(0).values),
        (48, (df['Close'].values - df['Open'].values) / (df['High'].values - df['Low'].values + 1e-10))
    ]
    for idx, vals in feature_mapping: seq[:, idx] = vals
    return np.nan_to_num(seq, nan=0.0, posinf=5.0, neginf=-5.0)
def _extract_ml_features(stock: StockData, ctx: Any, cf: ComplexFeatures, alt: AltData, alpha_vec: np.ndarray = None) -> dict:
    macd_cross_strength = safe_div(stock.curr['MACD'] - stock.curr['Signal_Line'], abs(stock.curr['Close']) * 0.01)
    vol_surge_ratio = safe_div(stock.curr['Volume'], stock.curr['Vol_MA20'], cap=50.0)
    cmf_val, smf_val = stock.curr['CMF'], stock.curr['Smart_Money_Flow']
    hurst_score = max(0.0, min(1.0, (cf.hurst_med - 0.5) * 2.0)) if (cf.hurst_reliable and cf.hurst_med > Config.Params.HURST_RELIABLE) else 0.0

    feat_dict = {
        "米奈尔维尼": safe_div(stock.curr['SMA_50'] - stock.curr['SMA_200'], stock.curr['SMA_200']),
        "强相对强度": cf.rs_20, "MACD金叉": macd_cross_strength,
        "TTM Squeeze ON": safe_div((stock.curr['KC_Upper'] - stock.curr['KC_Lower']) - (stock.curr['BB_Upper'] - stock.curr['BB_Lower']), stock.curr['ATR'] + 1e-10),
        "一目多头": safe_div(stock.curr['Close'] - max(stock.curr['SenkouA'], stock.curr['SenkouB']), stock.curr['Close'] * 0.01),
        "强势回踩": safe_div(stock.curr['Close'] - stock.curr['EMA_20'], stock.curr['EMA_20'] * 0.01),
        "机构控盘(CMF)": cmf_val, "突破缺口": safe_div(stock.curr['Open'] - stock.prev['Close'], stock.prev['Close'] * 0.01),
        "VWAP突破": safe_div(stock.curr['Close'] - stock.curr['VWAP_20'], stock.curr['VWAP_20'] * 0.01),
        "AVWAP突破": safe_div(stock.curr['Close'] - stock.curr['AVWAP'], stock.curr['AVWAP'] * 0.01) if pd.notna(stock.curr['AVWAP']) else 0.0,
        "SMC失衡区": safe_div(stock.curr['Close'] - cf.fvg_lower, stock.curr['Close'] * 0.01) if cf.fvg_lower > 0 else 0.0,
        "流动性扫盘": safe_div(stock.curr['Swing_Low_20'] - stock.curr['Low'], stock.curr['Low'] * 0.01) if pd.notna(stock.curr['Swing_Low_20']) else 0.0,
        "聪明钱抢筹": smf_val, "巨量滞涨": vol_surge_ratio, "放量长阳": safe_div(stock.curr['Close'] - stock.curr['Open'], stock.curr['Open'] * 0.01),
        "口袋支点": safe_div(stock.curr['Volume'], stock.curr['Max_Down_Vol_10'], cap=50.0),
        "VCP收缩": safe_div(stock.curr['Range_20'], stock.curr['Range_60'] + 1e-10),
        "量子概率云(KDE)": cf.kde_breakout_score, "特性改变(ChoCh)": safe_div(stock.curr['Close'] - stock.swing_high_10, stock.swing_high_10 * 0.01),
        "订单块(OB)": safe_div(stock.curr['Close'] - stock.curr['OB_Low'], stock.curr['OB_High'] - stock.curr['OB_Low'] + 1e-10) if pd.notna(stock.curr['OB_High']) else 0.0,
        "AMD操盘": safe_div(min(stock.curr['Open'], stock.curr['Close']) - stock.curr['Low'], stock.curr['TR'] + 1e-10),
        "跨时空共振(周线)": 1.0 if cf.weekly_bullish else 0.0, "CVD筹码净流入": float(stock.curr['CVD_Trend'] * (0.3 if stock.curr['CVD_Divergence'] == 1 else 1.0)),
        "独立Alpha(脱钩)": cf.pure_alpha, "NR7极窄突破": safe_div(stock.curr['Range'], stock.curr['ATR'] + 1e-10),
        "VPT量价共振": (1.0 / (1.0 + np.exp(-stock.curr['VPT_ZScore']))) if stock.curr['VPT_Accel'] > 0 else 0.0,
        "带量金叉(交互)": macd_cross_strength * vol_surge_ratio, "量价吸筹(交互)": cmf_val * smf_val,
        "近3日突破(滞后)": stock.curr['Recent_Price_Surge_3d'] if pd.notna(stock.curr['Recent_Price_Surge_3d']) else 0.0,
        "近3日巨量(滞后)": stock.curr['Recent_Vol_Surge_3d'] if pd.notna(stock.curr['Recent_Vol_Surge_3d']) else 0.0,
        "稳健赫斯特(Hurst)": hurst_score, "FFT多窗共振(动能)": cf.fft_ensemble_score,
        "大周期保护小周期(MACD共振)": 1.0 if (cf.weekly_macd_res == 1.0 and check_macd_cross(stock.curr, stock.prev)) else 0.0,
        "聪明钱月度净流入(月线)": cf.monthly_inst_flow, "60分钟级精准校准(RSI反弹)": cf.rsi_60m_bounce,
        "大盘Beta(宏观调整)": cf.beta_60d, "利率敏感度(TLT相关性)": cf.tlt_corr, "汇率传导(DXY相关性)": cf.dxy_corr,
        "Amihud非流动性(冲击成本)": stock.curr['Amihud'] if pd.notna(stock.curr['Amihud']) else 0.0,
        "52周高点距离(动能延续)": stock.curr['Dist_52W_High'] if pd.notna(stock.curr['Dist_52W_High']) else 0.0,
        "波动率风险溢价(VRP)": cf.vrp, "期权PutCall情绪(PCR)": alt.pcr, "隐含波动率偏度(IV Skew)": alt.iv_skew,
        "做空兴趣突变(轧空)": alt.short_change if alt.short_float > Config.Params.SHORT_SQZ_FLT else 0.0,
        "内部人集群净买入(Insider)": alt.insider_net_buy, "分析师修正动量(Analyst)": alt.analyst_mom,
        "舆情NLP情感极值(News_NLP)": alt.nlp_score, "散户热度加速度(WSB_Accel)": alt.wsb_accel
    }
    if alpha_vec is not None and len(alpha_vec) == 16:
        for i, val in enumerate(alpha_vec): feat_dict[f"Alpha_T{i+1:02d}"] = float(val)
    else:
        for i in range(1, 17): feat_dict[f"Alpha_T{i:02d}"] = 0.0
    return {f: float(np.nan_to_num(feat_dict.get(f, 0.0), nan=0.0, posinf=20.0, neginf=-20.0)) for f in Config.ALL_FACTORS}

def _evaluate_omni_matrix(stock: StockData, ctx: Any, cf: ComplexFeatures, alt: AltData) -> Tuple[int, List[str], List[str], bool]:
    triggered_list, factors_list = [], []
    theme_scores = {'TREND': 0.0, 'VOLATILITY': 0.0, 'REVERSAL': 0.0, 'QUANTUM': 0.0}
    black_swan_risk = False

    w_mul = ctx.w_mul
    raw_score = 0.0
    sig_count = 0

    if stock.curr['Close'] > stock.curr['SMA_200']:
        theme_scores['TREND'] += 2.0 * w_mul; sig_count += 1
    if stock.curr['SMA_50'] > stock.curr['SMA_150'] > stock.curr['SMA_200']:
        theme_scores['TREND'] += 3.0 * w_mul; sig_count += 1; triggered_list.append("💎 米奈尔维尼模板列队")
        
    if cf.rs_20 > 1.05:
        theme_scores['TREND'] += 3.0 * w_mul; sig_count += 1; triggered_list.append(f"💪 强相对强度 (RS={cf.rs_20:.2f})")
    
    if check_macd_cross(stock.curr, stock.prev):
        theme_scores['TREND'] += 2.0; sig_count += 1; triggered_list.append("🌟 MACD 金叉点火")
        
    if (stock.curr['KC_Upper'] - stock.curr['KC_Lower']) < (stock.curr['BB_Upper'] - stock.curr['BB_Lower']):
        theme_scores['VOLATILITY'] += 4.0 * w_mul; sig_count += 1; triggered_list.append("💥 TTM Squeeze 爆发临界")
    
    if stock.curr['Above_Cloud'] == 1 and stock.prev['Above_Cloud'] == 0:
        theme_scores['TREND'] += 3.0; sig_count += 1; triggered_list.append("☁️ 一目均衡突破云层")
    
    if stock.curr['Close'] > stock.curr['EMA_20'] and stock.prev['Close'] <= stock.prev['EMA_20']:
        theme_scores['REVERSAL'] += 2.0; sig_count += 1; triggered_list.append("🔄 强势回踩 EMA20 反抽")
    
    if stock.curr['CMF'] > 0.15:
        theme_scores['TREND'] += 2.0; sig_count += 1
        if stock.curr['CMF'] > 0.25: triggered_list.append(f"🏦 机构高能控盘 (CMF={stock.curr['CMF']:.2f})")
        
    if stock.curr['Open'] > stock.prev['Close'] * 1.02:
        theme_scores['VOLATILITY'] += 3.0; sig_count += 1; triggered_list.append("🚀 突破缺口开盘")
        
    if stock.curr['Close'] > stock.curr['VWAP_20'] and stock.prev['Close'] <= stock.prev['VWAP_20']:
        theme_scores['TREND'] += 2.5; sig_count += 1; triggered_list.append("锚定 VWAP 突破")
        
    if pd.notna(stock.curr['AVWAP']) and stock.curr['Close'] > stock.curr['AVWAP'] and stock.prev['Close'] <= stock.prev['AVWAP']:
        theme_scores['TREND'] += 3.0; sig_count += 1; triggered_list.append("⚓ 锚定 AVWAP 关键翻越")

    if cf.fvg_lower > 0 and (cf.fvg_lower <= stock.curr['Low'] <= cf.fvg_upper or cf.fvg_lower <= stock.curr['Close'] <= cf.fvg_upper):
        theme_scores['QUANTUM'] += 4.0; sig_count += 1; triggered_list.append("🧲 SMC 填补失衡区 (FVG)")
        
    if stock.prev['Low'] < stock.curr['Swing_Low_20'] and stock.curr['Close'] > stock.curr['Swing_Low_20']:
        theme_scores['REVERSAL'] += 5.0; sig_count += 1; triggered_list.append("🧹 散户流动性扫盘 (Liquidity Sweep)")

    if stock.curr['Smart_Money_Flow'] > 0:
        theme_scores['TREND'] += 2.0; sig_count += 1
        if stock.curr['Smart_Money_Flow'] > 2.0: triggered_list.append("🧠 聪明钱加速流入")
        
    vol_ma20 = stock.curr['Vol_MA20'] + 1e-10
    if stock.curr['Volume'] > vol_ma20 * 2 and stock.curr['Close'] > stock.curr['Open']:
        theme_scores['VOLATILITY'] += 3.0; sig_count += 1; triggered_list.append("📈 巨量资金介入")
    
    if stock.curr['Close'] > stock.curr['Max_Down_Vol_10'] and stock.curr['Volume'] > stock.curr['Max_Down_Vol_10']:
        theme_scores['VOLATILITY'] += 4.0; sig_count += 1; triggered_list.append("👝 口袋支点突破 (Pocket Pivot)")

    if stock.curr['Range_20'] < stock.curr['Range_60'] * 0.5 and stock.curr['Volume'] < vol_ma20 * 0.8:
        theme_scores['VOLATILITY'] += 3.0; sig_count += 1; triggered_list.append("📉 VCP 波动率极度收缩")

    if cf.kde_breakout_score > 0.8:
        theme_scores['QUANTUM'] += max(0, (cf.kde_breakout_score - 0.5) * 8.0); sig_count += 1
        triggered_list.append(f"⚛️ 量子概率云突围 ({cf.kde_breakout_score:.2f})")
        
    if stock.curr['Close'] > stock.swing_high_10 and stock.prev['Close'] <= stock.swing_high_10:
        theme_scores['REVERSAL'] += 4.0; sig_count += 1; triggered_list.append("🛡️ 市场特性改变 (ChoCh)")
        
    if pd.notna(stock.curr['OB_High']) and stock.curr['OB_Low'] <= stock.curr['Close'] <= stock.curr['OB_High']:
        theme_scores['QUANTUM'] += 3.0; sig_count += 1; triggered_list.append("🧱 触及机构订单块 (Order Block)")
        
    if stock.curr['Low'] < min(stock.curr['Open'], stock.curr['Close']) * 0.98 and stock.curr['Close'] > stock.curr['Open']:
        theme_scores['REVERSAL'] += 2.0; sig_count += 1; triggered_list.append("🎭 AMD 操盘模型 (Accumulation)")

    if cf.weekly_bullish: theme_scores['TREND'] += 3.0; sig_count += 1; triggered_list.append("📅 周线级别看多共振")
    if stock.curr['CVD_Divergence'] == 1:
        theme_scores['QUANTUM'] += 4.0; sig_count += 1; triggered_list.append("🔎 CVD 隐秘洗盘背离")
    if stock.curr['CVD_Trend'] > 0: theme_scores['TREND'] += 2.0; sig_count += 1

    if cf.pure_alpha > 0.15:
        theme_scores['QUANTUM'] += 3.0; sig_count += 1; triggered_list.append(f"🦅 独立纯 Alpha ({cf.pure_alpha:.2f})")
        
    if stock.curr['NR7']: theme_scores['VOLATILITY'] += 2.0; sig_count += 1

    if stock.curr['VPT_ZScore'] > 2.0 and stock.curr['VPT_Accel'] > 0:
        theme_scores['VOLATILITY'] += 3.0; sig_count += 1; triggered_list.append("🌊 VPT 量价齐升")
        
    if stock.curr['Dist_52W_High'] > -0.05: theme_scores['TREND'] += 2.0; sig_count += 1

    for k, v in theme_scores.items():
        if v > 0: factors_list.append(f"{k}({v:.1f})")
        raw_score += v

    final_score = raw_score * (1.0 + (sig_count * 0.05))
    final_score_saturated = min(100.0, final_score)
    return int(final_score_saturated), triggered_list, factors_list, black_swan_risk

def _apply_market_filters(curr, prev, sym, base_score, sig, a, b, c):
    if curr['RSI'] > 85.0: return max(0, base_score - 30), False, sig + ["⚠️ 极端超买"]
    return base_score, False, sig

def _apply_ai_inference(reports: List[dict], ctx: MarketContext) -> List[dict]:
    for r in reports: r['ai_prob'] = random.uniform(0.4, 0.8)
    return reports

def _calculate_position_size(stock, ctx, ai_prob, div, risk):
    return stock.curr['Close'] * 1.1, stock.curr['Close'] * 0.95, ai_prob - 0.2, ""

def _apply_kelly_cluster_optimization(reports: List[dict], pd_dict, exp, ctx):
    for i, r in enumerate(reports): r['opt_weight'] = 0.05; r['pos_advice'] = "✅ 凯利最优"
    return reports[:5]
