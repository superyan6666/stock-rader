import unittest
import pandas as pd
import numpy as np
import warnings
import time  # 🚀 新增：用于性能基准测试
import os    # 🚀 新增：用于控制环境变量测试开关

# 忽略测试中的 pandas 警告
warnings.filterwarnings('ignore')

# 🚀 导入底层组件，用于搭建特征提取沙盒
from quantbot import (
    calculate_indicators, 
    _extract_complex_features, 
    _evaluate_omni_matrix,
    MarketContext, StockData, AltData, Config
)

class TestQuantBotIndicators(unittest.TestCase):
    
    def setUp(self):
        """
        初始化测试基底：人为构造一个包含 150 根 K 线的标准稳健上涨趋势。
        """
        np.random.seed(42)
        periods = 150
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        base_trend = np.linspace(100, 250, periods)
        noise = np.random.normal(0, 1.5, periods)
        
        close_px = base_trend + noise
        open_px = close_px - np.random.uniform(0.5, 2.0, periods)
        high_px = close_px + np.random.uniform(0.5, 3.0, periods)
        low_px = open_px - np.random.uniform(0.5, 3.0, periods)
        volume = np.random.randint(1000000, 5000000, periods)
        
        self.mock_df = pd.DataFrame({
            'Open': open_px,
            'High': high_px,
            'Low': low_px,
            'Close': close_px,
            'Volume': volume
        }, index=dates)

    # ---------------- 辅助沙盒引擎 ----------------
    def _get_mock_context(self) -> MarketContext:
        """构建一个极简的中性宏观虚拟环境，防止测试时空指针"""
        return MarketContext(
            regime="bull", regime_desc="", w_mul=1.0, xai_weights={},
            vix_current=15.0, vix_desc="", vix_scalar=1.0, max_risk=0.015,
            macro_gravity=False, is_credit_risk_high=False, vix_inv=False,
            qqq_df=pd.DataFrame(), macro_data={}, total_market_exposure=1.0,
            health_score=1.0, pain_warning="", credit_spread_mom=0.0,
            vix_term_structure=1.0, market_pcr=1.0, dynamic_min_score=8.0
        )

    def _eval_factors_for_df(self, df: pd.DataFrame) -> list:
        """驱动完整的数据清洗与因子萃取管线"""
        # 🚀 测试隔离：显式使用 deep=True 深拷贝，切断 Pandas 内存视图的底层关联，防止原数据被污染
        df_ind = calculate_indicators(df.copy(deep=True))
        ctx = self._get_mock_context()
        sym = "TEST"
        curr = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        is_vol = curr['Volume'] > curr['Vol_MA20'] * 1.5
        swing_high_10 = df_ind['High'].iloc[-11:-1].max() if len(df_ind) >= 11 else curr['High']
        
        # 组装股票数据，子级别(周/月)K线留空不影响日内因子测试
        stock = StockData(sym, df_ind, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), curr, prev, is_vol, swing_high_10)
        cf = _extract_complex_features(stock, ctx)
        alt = AltData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # 提取被激发的因子集群
        _, _, factors, _ = _evaluate_omni_matrix(stock, ctx, cf, alt)
        return factors

    # ---------------- 常规动能逻辑测试 ----------------
    def test_indicator_shapes_and_nans(self):
        """验证指标计算后的矩阵形态与空值处理是否健壮"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        self.assertEqual(len(df_res), len(self.mock_df), "输出的 DataFrame 长度发生截断缩水！")
        
        core_cols = ['RSI', 'MACD', 'ATR', 'CMF', 'VPT_ZScore']
        for col in core_cols:
            self.assertIn(col, df_res.columns, f"关键指标列 {col} 缺失！")

    # 🚀 ---------------- 新增：参数化矩阵边界测谎仪 ---------------- 🚀
    def test_indicator_value_bounds_parameterized(self):
        """
        参数化测试引擎：利用 subTest 统一遍历验证各类技术指标的物理极值与数学边界，
        消灭重复代码，并确保任一指标越界不影响其他指标的排查。
        """
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        
        # 阵列格式: (指标列名, 理论物理下限, 理论物理上限) 
        # None 代表该方向不设绝对物理极限
        bounds_matrix = [
            ('RSI', 0.0, 100.0),              # 相对强弱指标死界
            ('CMF', -1.0, 1.0),               # 蔡金资金流量死界
            ('ATR', 0.0, None),               # 真实波幅必定非负
            ('Volume', 0.0, None),            # 成交量必定非负
            ('SuperTrend_Up', 0.0, 1.0),      # 趋势状态只能是布尔/二进制
            ('Smart_Money_Flow', -1.0, 1.0),  # 聪明钱净流强度归一化界限
            ('CVD_Trend', -1.0, 1.0)          # CVD 趋势向量限定为 -1, 0, 1
        ]
        
        for col, min_val, max_val in bounds_matrix:
            with self.subTest(indicator=col):
                self.assertIn(col, df_res.columns, f"严重异常：特征矩阵中丢失了指标列 {col}！")
                
                # 剔除预热期产生的合理 NaN 值后进行刚性测试
                valid_vals = df_res[col].dropna()
                self.assertTrue(len(valid_vals) > 0, f"致命异常：{col} 列的计算结果全部为 NaN 毒药！")
                
                if min_val is not None:
                    # 加入 1e-5 微小宽容度，防止浮点数精度误差导致误杀
                    self.assertTrue((valid_vals >= min_val - 1e-5).all(), 
                                    f"数学推导越界：{col} 公式存在下限物理极值 {min_val}，检测到实际值击穿下限，说明底层的绝对值保护或归一化逻辑被破坏！")
                if max_val is not None:
                    self.assertTrue((valid_vals <= max_val + 1e-5).all(), 
                                    f"数学推导越界：{col} 公式存在上限物理极值 {max_val}，检测到实际值击穿上限，说明底层除数溢出或振幅截断逻辑失效！")
            
    def test_rsi_logic_on_uptrend(self):
        """物理测谎：验证持续上涨趋势下的 RSI 表现"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_rsi = df_res['RSI'].iloc[-30:]
        self.assertTrue((tail_rsi > 50).all(), 
                        "数学推导失败：在严格构造的单调上涨序列中，价格变动 Delta > 0 占据绝对主导，"
                        "使得 Avg_Gain 必然大于 Avg_Loss，从而 RS > 1。根据公式 RSI = 100 - 100/(1+RS)，"
                        "RSI 必然严格大于 50 的多空分水岭。实际却跌破 50，指数平滑逻辑存在严重漂移。")
        
    def test_macd_logic_on_uptrend(self):
        """物理测谎：验证持续上涨趋势下的 MACD 多头排列"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_df = df_res.iloc[-30:]
        self.assertTrue((tail_df['MACD'] > 0).all(), 
                        "数学推导失败：在持续单边上涨趋势中，近期价格权重大于远期价格，"
                        "因此短期 EMA(12) 必然严格大于长期 EMA(26)。MACD = EMA(12) - EMA(26) 必须恒为正数。"
                        "当前 MACD 跌落水下，证明底层的 EMA 衰减因子 (Alpha) 计算发生了倒置！")
        
    def test_macd_cross_detection(self):
        """精准形态测谎：验证 MACD 金叉 (Golden Cross) 跃迁的精确捕获能力"""
        periods = 60
        close = np.linspace(100, 70, periods - 2).tolist()  
        close.extend([75.0, 110.0]) 
        
        df = pd.DataFrame({
            'Close': close, 'Open': [c * 0.99 for c in close], 'High': [c * 1.05 for c in close],
            'Low': [c * 0.95 for c in close], 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df_res = calculate_indicators(df.copy(deep=True))
        prev, curr = df_res.iloc[-2], df_res.iloc[-1]
        
        self.assertTrue(prev['MACD'] < prev['Signal_Line'],
                        f"数学推导失败(T-1)：前 58 天为单边阴跌，MACD 必须小于 9 日平滑 Signal_Line 形成死叉预备态。"
                        f"实际 MACD:{prev['MACD']:.3f}, Signal:{prev['Signal_Line']:.3f}")
        
        self.assertTrue(curr['MACD'] > curr['Signal_Line'],
                        f"数学推导失败(T)：T时刻强行灌入 46% 的巨幅阳线反包，这必定拉升 EMA(12) 从而使 MACD 瞬间上穿滞后的 Signal_Line。"
                        f"若未发生金叉，说明移动平均对极端波动的迟滞惩罚过高或公式有误！"
                        f"实际 MACD:{curr['MACD']:.3f}, Signal:{curr['Signal_Line']:.3f}")

    def test_rsi_bounds_extreme(self):
        """极值测谎：验证 RSI 在连续涨停与连续跌停下的理论边界 [0, 100]"""
        close_up = 100 * (1.05 ** np.arange(30))
        df_up = pd.DataFrame({
            'Close': close_up, 'Open': close_up * 0.98, 'High': close_up * 1.02, 
            'Low': close_up * 0.97, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=30))
        res_up = calculate_indicators(df_up.copy(deep=True))
        self.assertTrue(res_up['RSI'].iloc[-1] > 98.0, 
                        "数学推导失败：连续 30 日涨停导致 Avg_Loss 趋近于 0，相对强度 RS 趋近于无穷大 (+∞)。"
                        "代入极限公式 RSI = 100 - (100 / (1 + ∞)) 必然逼近 100.0 物理极值。未达极值说明平滑窗口阻滞过大！")
        
        close_dn = 100 * (0.95 ** np.arange(30))
        df_dn = pd.DataFrame({
            'Close': close_dn, 'Open': close_dn * 1.02, 'High': close_dn * 1.03, 
            'Low': close_dn * 0.98, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=30))
        res_dn = calculate_indicators(df_dn.copy(deep=True))
        self.assertTrue(res_dn['RSI'].iloc[-1] < 2.0, 
                        "数学推导失败：连续 30 日跌停导致 Avg_Gain 趋近于 0，相对强度 RS 趋近于 0。"
                        "代入极限公式 RSI = 100 - (100 / (1 + 0)) 必然逼近 0.0 物理极值。")

    def test_vwap_and_avwap_logic(self):
        """精准锚点测谎：验证 VWAP 与 AVWAP (锚定成交量加权平均价) 的计算合理性与物理边界"""
        periods = 100
        base_px = np.linspace(10, 20, periods)
        df = pd.DataFrame({
            'Open': base_px * 0.98, 'High': base_px * 1.05, 'Low': base_px * 0.95,
            'Close': base_px, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        anchor_idx = 50
        df.iloc[anchor_idx, df.columns.get_loc('Volume')] = 50000000 
        tp_anchor = (df.iloc[anchor_idx]['High'] + df.iloc[anchor_idx]['Low'] + df.iloc[anchor_idx]['Close']) / 3.0
        
        df_res = calculate_indicators(df.copy(deep=True))
        vwap_tail = df_res['VWAP_20'].dropna()
        
        self.assertTrue(len(vwap_tail) > 0, "数学推理异常：存在至少 20 天以上的完整 K 线，VWAP_20 截面不应为空。")
        self.assertTrue(df_res['AVWAP'].iloc[:anchor_idx].isna().all(), 
                        "数学推导失败：AVWAP 的锚点在第 50 天(天量日)，在时序上，第 50 天之前的 AVWAP 应当由于缺乏锚点基底而必定为 NaN。发生未来函数穿透！")
        self.assertAlmostEqual(df_res['AVWAP'].iloc[anchor_idx], tp_anchor, places=4, 
                               msg=f"数学推导失败：锚点首日的 AVWAP 分子分母抵消，其值必定等于当天的 Typical Price = (H+L+C)/3 = {tp_anchor:.4f}。")
        
        avwap_val = df_res['AVWAP'].iloc[anchor_idx]
        self.assertTrue(df.iloc[anchor_idx]['Low'] <= avwap_val <= df.iloc[anchor_idx]['High'], 
                        "物理因果律失效：AVWAP 是基于当天高低收的加权平均值，在数学上永远不可能突破当天 K 线的 [Low, High] 绝对实体物理包络线！")

    def test_chandelier_exit_logic(self):
        """核心风控测谎：验证吊灯止损线 (Chandelier Exit) 的跟随特性与物理边界"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_df = df_res.iloc[-50:]
        
        self.assertIn('Chandelier_Exit', df_res.columns)
        self.assertTrue((tail_df['Chandelier_Exit'] < tail_df['Highest_22']).all(), 
                        "数学推导失败：Chandelier Exit 公式为 Highest_22 - 2.5 * ATR。因为 ATR 必定非负，所以止损线必定严格低于或等于 22 日最高价。")
        self.assertTrue((tail_df['Chandelier_Exit'] < tail_df['Close']).all(), 
                        "防线崩塌推导：这是一个波动极小的单调上涨趋势，2.5倍 ATR 的安全垫必须稳稳托底在最新收盘价下方。若击穿收盘价，说明 ATR 乘数被写倒置或窗口取错！")
        self.assertTrue(tail_df['Chandelier_Exit'].iloc[-1] > tail_df['Chandelier_Exit'].iloc[0], 
                        "棘轮推导失败 (Ratchet Effect)：长达 50 天的价格重心上移必定带动 Highest_22 上移，止损线必须体现出跟随利润向上抬升的棘轮特性！")

    # 🚀 ---------------- 极端博弈形态对抗测谎 ---------------- 🚀

    def test_factor_breakaway_gap(self):
        """极端博弈测谎 1：验证 突破缺口 (Breakaway Gap)"""
        periods = 30
        close = np.linspace(100, 105, periods)
        df = pd.DataFrame({
            'Close': close, 'Open': close - 0.5, 'High': close + 0.5, 
            'Low': close - 1.0, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df.iloc[-1, df.columns.get_loc('Open')] = 108.5
        df.iloc[-1, df.columns.get_loc('Low')] = 108.0
        df.iloc[-1, df.columns.get_loc('Close')] = 109.0
        df.iloc[-1, df.columns.get_loc('High')] = 110.0
        
        factors = self._eval_factors_for_df(df)
        self.assertIn("突破缺口", factors, 
                      "形态推导失败：T日 Open(108.5) > T-1日 Close(105)，跳空幅度达 3.3%，"
                      "远超 >1.5% 的因子硬性起步阈值且超越 ATR 波动下限，但未能触发【突破缺口】动能因子！")

    def test_factor_amd_and_spring(self):
        """极端博弈测谎 2：验证 威科夫弹簧(Spring) 与 AMD诱空洗盘模型"""
        periods = 30
        close = np.linspace(100, 90, periods)  
        df = pd.DataFrame({
            'Close': close, 'Open': close, 'High': close + 1, 
            'Low': close - 1, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df.iloc[-1, df.columns.get_loc('Open')] = 91.0
        df.iloc[-1, df.columns.get_loc('Low')] = 85.0
        df.iloc[-1, df.columns.get_loc('Close')] = 95.0
        df.iloc[-1, df.columns.get_loc('High')] = 95.5
        
        factors = self._eval_factors_for_df(df)
        self.assertIn("威科夫弹簧(Spring)", factors, 
                      "形态推导失败：T日最低价 85.0 刺穿了前期震荡区下限 89.0，随后收盘价 95.0 强力拉回 10 日均线之上，满足猎杀散户止损盘的经典【威科夫弹簧】！")
        self.assertIn("AMD操盘", factors, 
                      "形态推导失败：下影线 (Open 91 - Low 85) = 6.0，占全天振幅 (High 95.5 - Low 85) 的 57%，"
                      "严格大于 >30% 的理论阈值，且上影线极短。这是经典的机构【AMD操盘模型】(Accumulation, Manipulation, Distribution) 却被漏判！")

    def test_factor_volume_surge(self):
        """极端博弈测谎 3：验证 巨量滞涨 (机构挂单吸筹) 与 口袋支点 (破局放量)"""
        periods = 40
        close = np.linspace(100, 80, periods)
        df_stag = pd.DataFrame({
            'Close': close, 'Open': close, 'High': close + 1, 
            'Low': close - 1, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df_stag.iloc[-1, df_stag.columns.get_loc('Volume')] = 5000000 
        df_stag.iloc[-1, df_stag.columns.get_loc('Open')] = 79.0
        df_stag.iloc[-1, df_stag.columns.get_loc('Close')] = 79.1
        df_stag.iloc[-1, df_stag.columns.get_loc('High')] = 79.5
        df_stag.iloc[-1, df_stag.columns.get_loc('Low')] = 78.5
        
        factors_stag = self._eval_factors_for_df(df_stag)
        self.assertIn("巨量滞涨", factors_stag, 
                      "形态推导失败：T日成交量 5M 达到近期 MA20_Vol(1M) 的 5 倍，但实体振幅 (Close - Open = 0.1) 极小，"
                      "完美契合【底部冰山吸筹】特征却未触发巨量滞涨因子！")

        periods = 60
        close = np.linspace(100, 100, periods) 
        df_pp = pd.DataFrame({
            'Close': close, 'Open': close, 'High': close + 1, 
            'Low': close - 1, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df_pp.iloc[-3, df_pp.columns.get_loc('Close')] = 98.0
        df_pp.iloc[-3, df_pp.columns.get_loc('Open')] = 100.0
        df_pp.iloc[-3, df_pp.columns.get_loc('Volume')] = 2000000  
        
        df_pp.iloc[-1, df_pp.columns.get_loc('Close')] = 102.0
        df_pp.iloc[-1, df_pp.columns.get_loc('Volume')] = 3000000 
        
        factors_pp = self._eval_factors_for_df(df_pp)
        self.assertIn("口袋支点", factors_pp, 
                      "形态推导失败：T日成交量(3M) 严格大于过去 10 天内的绝对最大阴量(T-3日的 2M)，"
                      "且收盘价强力站上 EMA(50) 水位，完全满足【口袋支点】破局条件却被遗漏！")

    # 🚀 ---------------- 极限防线：脏数据与异常输入隔离测谎 ---------------- 🚀
    
    def test_nan_and_zero_handling(self):
        """混沌工程测谎：验证指标引擎对缺失数据(NaN)、零成交量、以及一字板(High==Low)的绝对抗崩溃能力"""
        periods = 250
        dates = pd.date_range('2023-01-01', periods=periods)
        df = pd.DataFrame({
            'Open': 100.0, 'High': 105.0, 'Low': 95.0, 'Close': 100.0, 'Volume': 1000000.0
        }, index=dates)
        
        # 1. 注入连续的 NaN 黑洞 (模拟停牌或 API 数据熔断)
        df.iloc[10:15, :] = np.nan
        
        # 2. 注入“一字涨跌停” (High == Low == Close == Open)，测试物理边界压缩带来的除零风险
        df.iloc[50:55, df.columns.get_loc('Open')] = 120.0
        df.iloc[50:55, df.columns.get_loc('High')] = 120.0
        df.iloc[50:55, df.columns.get_loc('Low')] = 120.0
        df.iloc[50:55, df.columns.get_loc('Close')] = 120.0
        
        # 3. 注入极度枯竭的“零成交量”，定向打击 VWAP / VPT 等量价指标的分母
        df.iloc[80:85, df.columns.get_loc('Volume')] = 0.0
        
        # 执行指标洗炼引擎 (如果没有稳健的 +1e-10 保护，这里会抛出 ZeroDivisionError 或引发 DataFrame 链式崩溃)
        df_res = calculate_indicators(df.copy(deep=True))
        
        # 验证 A: 数据未缩水截断
        self.assertEqual(len(df_res), periods, "含有 NaN 的输入导致了数据管道被粗暴截断！")
        
        # 验证 B: 前置 NaN 保留机制 (SMA_50 要求拥有 50 个观测值，前 49 个无论如何计算都必须为空)
        self.assertTrue(df_res['SMA_50'].iloc[:49].isna().all(), "时序推导失败：SMA_50 的初期缓冲窗口被错误填充，发生了不合逻辑的未来函数渗透！")
        self.assertFalse(pd.isna(df_res['SMA_50'].iloc[50]), "时序推导失败：SMA_50 在度过了首批 50 天的盲区后依然未能正常输出有效数值！")
        self.assertTrue(df_res['SMA_200'].iloc[:199].isna().all(), "时序推导失败：SMA_200 的初期缓冲窗口必须严格保留 199 个 NaN 占位符！")
        
        # 验证 C: 一字停板除零防御 (CMF 公式: (C-L)-(H-C) / (H-L) * V)
        cmf_flat = df_res['CMF'].iloc[52]
        self.assertTrue(np.isfinite(cmf_flat), "极值崩溃推导：由于一字板 (High - Low == 0) 作为分母时未加 1e-10 的微量保护，导致 CMF 指标爆出 NaN/Inf！")
        
        # 验证 D: 零成交量除零防御 (VWAP 公式分母包含 sum(Volume))
        vwap_zero_vol = df_res['VWAP_20'].iloc[82]
        self.assertTrue(np.isfinite(vwap_zero_vol), "极值崩溃推导：由于全天成交量 Volume == 0，导致 VWAP_20 等动能指标在除法时爆出 NaN/Inf！")
        
        # 验证 E: 全局抗毒性 (除了正常的 NaN 缓冲带，不允许有任何 Inf 无穷大污染矩阵)
        numeric_df = df_res.select_dtypes(include=[np.number])
        has_inf = np.isinf(numeric_df).values.any()
        self.assertFalse(has_inf, "矩阵崩塌推导：系统未能在全量指标中防御极限输入，矩阵内残留了会毒死机器学习特征器的 Inf (无穷大) 恶性噪音。")

                if max_val is not None:
                    self.assertTrue((valid_vals <= max_val + 1e-5).all(), 
                                    f"物理越界：{col} 存在高于绝对上限 {max_val} 的反常数值！")
            
    def test_rsi_logic_on_uptrend(self):
        """物理测谎：验证持续上涨趋势下的 RSI 表现"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_rsi = df_res['RSI'].iloc[-30:]
        self.assertTrue((tail_rsi > 50).all(), "在绝对上涨趋势中，RSI 跌破 50，逻辑存在漂移。")
        
    def test_macd_logic_on_uptrend(self):
        """物理测谎：验证持续上涨趋势下的 MACD 多头排列"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_df = df_res.iloc[-30:]
        self.assertTrue((tail_df['MACD'] > 0).all(), "绝对上涨趋势中 MACD 跌落水下，指数平滑逻辑错误！")
        
    def test_macd_cross_detection(self):
        """精准形态测谎：验证 MACD 金叉 (Golden Cross) 跃迁的精确捕获能力"""
        periods = 60
        close = np.linspace(100, 70, periods - 2).tolist()  
        close.extend([75.0, 110.0]) 
        
        df = pd.DataFrame({
            'Close': close, 'Open': [c * 0.99 for c in close], 'High': [c * 1.05 for c in close],
            'Low': [c * 0.95 for c in close], 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df_res = calculate_indicators(df.copy(deep=True))
        prev, curr = df_res.iloc[-2], df_res.iloc[-1]
        
        self.assertTrue(prev['MACD'] < prev['Signal_Line'])
        self.assertTrue(curr['MACD'] > curr['Signal_Line'])

    def test_rsi_bounds_extreme(self):
        """极值测谎：验证 RSI 在连续涨停与连续跌停下的理论边界 [0, 100]"""
        close_up = 100 * (1.05 ** np.arange(30))
        df_up = pd.DataFrame({
            'Close': close_up, 'Open': close_up * 0.98, 'High': close_up * 1.02, 
            'Low': close_up * 0.97, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=30))
        res_up = calculate_indicators(df_up.copy(deep=True))
        self.assertTrue(res_up['RSI'].iloc[-1] > 98.0, "连续极端大涨时，RSI 未逼近 100 极值！")
        
        close_dn = 100 * (0.95 ** np.arange(30))
        df_dn = pd.DataFrame({
            'Close': close_dn, 'Open': close_dn * 1.02, 'High': close_dn * 1.03, 
            'Low': close_dn * 0.98, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=30))
        res_dn = calculate_indicators(df_dn.copy(deep=True))
        self.assertTrue(res_dn['RSI'].iloc[-1] < 2.0, "连续极端大跌时，RSI 未逼近 0 极值！")

    def test_vwap_and_avwap_logic(self):
        """精准锚点测谎：验证 VWAP 与 AVWAP (锚定成交量加权平均价) 的计算合理性与物理边界"""
        periods = 100
        base_px = np.linspace(10, 20, periods)
        df = pd.DataFrame({
            'Open': base_px * 0.98, 'High': base_px * 1.05, 'Low': base_px * 0.95,
            'Close': base_px, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        anchor_idx = 50
        df.iloc[anchor_idx, df.columns.get_loc('Volume')] = 50000000 
        tp_anchor = (df.iloc[anchor_idx]['High'] + df.iloc[anchor_idx]['Low'] + df.iloc[anchor_idx]['Close']) / 3.0
        
        df_res = calculate_indicators(df.copy(deep=True))
        vwap_tail = df_res['VWAP_20'].dropna()
        
        self.assertTrue(len(vwap_tail) > 0, "VWAP_20 截面计算全部丢失！")
        self.assertTrue(df_res['AVWAP'].iloc[:anchor_idx].isna().all(), "AVWAP 发生未来函数穿透！")
        self.assertAlmostEqual(df_res['AVWAP'].iloc[anchor_idx], tp_anchor, places=4, msg="AVWAP 锚定首日失真！")
        
        avwap_val = df_res['AVWAP'].iloc[anchor_idx]
        self.assertTrue(df.iloc[anchor_idx]['Low'] <= avwap_val <= df.iloc[anchor_idx]['High'], "AVWAP 突破物理包络线！")

    def test_chandelier_exit_logic(self):
        """核心风控测谎：验证吊灯止损线 (Chandelier Exit) 的跟随特性与物理边界"""
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_df = df_res.iloc[-50:]
        
        self.assertIn('Chandelier_Exit', df_res.columns)
        self.assertTrue((tail_df['Chandelier_Exit'] < tail_df['Highest_22']).all(), "吊灯止损越过绝对最高价！")
        self.assertTrue((tail_df['Chandelier_Exit'] < tail_df['Close']).all(), "上涨主升浪中止损线击穿收盘价！")
        self.assertTrue(tail_df['Chandelier_Exit'].iloc[-1] > tail_df['Chandelier_Exit'].iloc[0], "吊灯止损未能随趋势向上棘轮抬升！")

    # 🚀 ---------------- 极端博弈形态对抗测谎 ---------------- 🚀

    def test_factor_breakaway_gap(self):
        """极端博弈测谎 1：验证 突破缺口 (Breakaway Gap)"""
        periods = 30
        close = np.linspace(100, 105, periods)
        df = pd.DataFrame({
            'Close': close, 'Open': close - 0.5, 'High': close + 0.5, 
            'Low': close - 1.0, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df.iloc[-1, df.columns.get_loc('Open')] = 108.5
        df.iloc[-1, df.columns.get_loc('Low')] = 108.0
        df.iloc[-1, df.columns.get_loc('Close')] = 109.0
        df.iloc[-1, df.columns.get_loc('High')] = 110.0
        
        factors = self._eval_factors_for_df(df)
        self.assertIn("突破缺口", factors, "明显跳空高开，未能触发【突破缺口】逻辑！")

    def test_factor_amd_and_spring(self):
        """极端博弈测谎 2：验证 威科夫弹簧(Spring) 与 AMD诱空洗盘模型"""
        periods = 30
        close = np.linspace(100, 90, periods)  
        df = pd.DataFrame({
            'Close': close, 'Open': close, 'High': close + 1, 
            'Low': close - 1, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df.iloc[-1, df.columns.get_loc('Open')] = 91.0
        df.iloc[-1, df.columns.get_loc('Low')] = 85.0
        df.iloc[-1, df.columns.get_loc('Close')] = 95.0
        df.iloc[-1, df.columns.get_loc('High')] = 95.5
        
        factors = self._eval_factors_for_df(df)
        self.assertIn("威科夫弹簧(Spring)", factors, "跌破前低拉回，未触发【威科夫弹簧】！")
        self.assertIn("AMD操盘", factors, "超长下影线，未触发【AMD操盘模型】！")

    def test_factor_volume_surge(self):
        """极端博弈测谎 3：验证 巨量滞涨 (机构挂单吸筹) 与 口袋支点 (破局放量)"""
        periods = 40
        close = np.linspace(100, 80, periods)
        df_stag = pd.DataFrame({
            'Close': close, 'Open': close, 'High': close + 1, 
            'Low': close - 1, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df_stag.iloc[-1, df_stag.columns.get_loc('Volume')] = 5000000 
        df_stag.iloc[-1, df_stag.columns.get_loc('Open')] = 79.0
        df_stag.iloc[-1, df_stag.columns.get_loc('Close')] = 79.1
        df_stag.iloc[-1, df_stag.columns.get_loc('High')] = 79.5
        df_stag.iloc[-1, df_stag.columns.get_loc('Low')] = 78.5
        
        factors_stag = self._eval_factors_for_df(df_stag)
        self.assertIn("巨量滞涨", factors_stag, "低位放出5倍均量且价格实体极窄，未触发【巨量滞涨】！")

        periods = 60
        close = np.linspace(100, 100, periods) 
        df_pp = pd.DataFrame({
            'Close': close, 'Open': close, 'High': close + 1, 
            'Low': close - 1, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=periods))
        
        df_pp.iloc[-3, df_pp.columns.get_loc('Close')] = 98.0
        df_pp.iloc[-3, df_pp.columns.get_loc('Open')] = 100.0
        df_pp.iloc[-3, df_pp.columns.get_loc('Volume')] = 2000000  
        
        df_pp.iloc[-1, df_pp.columns.get_loc('Close')] = 102.0
        df_pp.iloc[-1, df_pp.columns.get_loc('Volume')] = 3000000 
        
        factors_pp = self._eval_factors_for_df(df_pp)
        self.assertIn("口袋支点", factors_pp, "放量突破 EMA_50 且成交量吞噬阴量，未触发【口袋支点】！")

    # 🚀 ---------------- 极限防线：脏数据与异常输入隔离测谎 ---------------- 🚀
    
    def test_nan_and_zero_handling(self):
        """混沌工程测谎：验证指标引擎对缺失数据(NaN)、零成交量、以及一字板(High==Low)的绝对抗崩溃能力"""
        periods = 250
        dates = pd.date_range('2023-01-01', periods=periods)
        df = pd.DataFrame({
            'Open': 100.0, 'High': 105.0, 'Low': 95.0, 'Close': 100.0, 'Volume': 1000000.0
        }, index=dates)
        
        # 1. 注入连续的 NaN 黑洞 (模拟停牌或 API 数据熔断)
        df.iloc[10:15, :] = np.nan
        
        # 2. 注入“一字涨跌停” (High == Low == Close == Open)，测试物理边界压缩带来的除零风险
        df.iloc[50:55, df.columns.get_loc('Open')] = 120.0
        df.iloc[50:55, df.columns.get_loc('High')] = 120.0
        df.iloc[50:55, df.columns.get_loc('Low')] = 120.0
        df.iloc[50:55, df.columns.get_loc('Close')] = 120.0
        
        # 3. 注入极度枯竭的“零成交量”，定向打击 VWAP / VPT 等量价指标的分母
        df.iloc[80:85, df.columns.get_loc('Volume')] = 0.0
        
        # 执行指标洗炼引擎 (如果没有稳健的 +1e-10 保护，这里会抛出 ZeroDivisionError 或引发 DataFrame 链式崩溃)
        df_res = calculate_indicators(df.copy(deep=True))
        
        # 验证 A: 数据未缩水截断
        self.assertEqual(len(df_res), periods, "含有 NaN 的输入导致了数据管道被粗暴截断！")
        
        # 验证 B: 前置 NaN 保留机制 (SMA_50 要求拥有 50 个观测值，前 49 个无论如何计算都必须为空)
        self.assertTrue(df_res['SMA_50'].iloc[:49].isna().all(), "SMA_50 的初期缓冲窗口被错误填充，发生了未来函数渗透！")
        self.assertFalse(pd.isna(df_res['SMA_50'].iloc[50]), "SMA_50 在度过安全期后未能正常输出有效数值！")
        self.assertTrue(df_res['SMA_200'].iloc[:199].isna().all(), "SMA_200 的初期缓冲窗口未被正确识别为 NaN！")
        
        # 验证 C: 一字停板除零防御 (CMF 公式: (C-L)-(H-C) / (H-L) * V)
        cmf_flat = df_res['CMF'].iloc[52]
        self.assertTrue(np.isfinite(cmf_flat), "一字涨跌停的高低点闭合，导致 CMF (蔡金资金流) 计算出 NaN/Inf 毒药！")
        
        # 验证 D: 零成交量除零防御 (VWAP 公式分母包含 sum(Volume))
        vwap_zero_vol = df_res['VWAP_20'].iloc[82]
        self.assertTrue(np.isfinite(vwap_zero_vol), "零成交量导致 VWAP_20 (成交量加权平均价) 出现 NaN/Inf 毒药！")
        
        # 验证 E: 全局抗毒性 (除了正常的 NaN 缓冲带，不允许有任何 Inf 无穷大污染矩阵)
        numeric_df = df_res.select_dtypes(include=[np.number])
        has_inf = np.isinf(numeric_df).values.any()
        self.assertFalse(has_inf, "指标矩阵中检测到了 Inf 极大值污染！说明底层的除法操作存在漏网之鱼。")

    # 🚀 ---------------- 大数据量级时间复杂度压测 ---------------- 🚀
    
    @unittest.skipUnless(os.getenv("RUN_PERF") == "1", "跳过性能压测。运行需配置环境变量: RUN_PERF=1")
    def test_performance_large_dataset(self):
        """
        高频性能压测：向引擎灌入 5000 根 K 线 (约20年历史)，
        验证 Numpy/Pandas 向量化是否被破坏，排查是否存在 O(N²) 慢循环导致性能崩塌。
        """
        periods = 5000  # 约 20 年的交易日
        dates = pd.date_range(start='2000-01-01', periods=periods, freq='D')
        
        # 构造超长连续伪随机价格漫步序列
        base_px = np.linspace(10, 100, periods)
        df_large = pd.DataFrame({
            'Open': base_px * 0.98,
            'High': base_px * 1.05,
            'Low': base_px * 0.95,
            'Close': base_px,
            'Volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)

        # 启动高精度物理计时器
        start_time = time.perf_counter()
        
        # 驱动核心引擎
        df_res = calculate_indicators(df_large)
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        
        self.assertEqual(len(df_res), periods, "输出的特征矩阵长度发生了灾难性截断！")
        
        # 性能断言界限：
        # 对于 5000 行数据的纯向量化滚动运算，现代 CPU 的处理时间通常在 50 ~ 250 ms 之间。
        # 这里为了兼容 GitHub Actions 服务器的性能波动，将安全阈值放宽到 1500 毫秒。
        # 如果运算耗时超过 1.5 秒，极大概率意味着算法中混入了 O(N^2) 的迭代毒瘤（如误用了 iterrows）。
        print(f"\n[性能探针] 5000 根 K 线全维特征萃取耗时: {elapsed_ms:.2f} ms")
        self.assertTrue(elapsed_ms < 1500, f"💥 性能崩塌！处理耗时 {elapsed_ms:.2f} ms，疑似存在非向量化的 O(N²) 慢循环陷阱！")

if __name__ == '__main__':
    unittest.main(verbosity=2)
