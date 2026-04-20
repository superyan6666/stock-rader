# 存储路径: test_quantbot.py
import unittest
import pandas as pd
import numpy as np
import warnings
import time
import os

warnings.filterwarnings('ignore')

from quantbot import (
    calculate_indicators, 
    _extract_complex_features, 
    _evaluate_omni_matrix,
    MarketContext, StockData, AltData, Config
)

class TestQuantBotIndicators(unittest.TestCase):
    
    def setUp(self):
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

    def _get_mock_context(self) -> MarketContext:
        return MarketContext(
            regime="bull", regime_desc="", w_mul=1.0, xai_weights={},
            vix_current=15.0, vix_desc="", vix_scalar=1.0, max_risk=0.015,
            macro_gravity=False, is_credit_risk_high=False, vix_inv=False,
            qqq_df=pd.DataFrame(), macro_data={}, total_market_exposure=1.0,
            health_score=1.0, pain_warning="", credit_spread_mom=0.0,
            vix_term_structure=1.0, market_pcr=1.0, dynamic_min_score=8.0
        )

    def _eval_factors_for_df(self, df: pd.DataFrame) -> list:
        df_ind = calculate_indicators(df.copy(deep=True))
        ctx = self._get_mock_context()
        sym = "TEST"
        curr = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        is_vol = curr['Volume'] > curr['Vol_MA20'] * 1.5
        swing_high_10 = df_ind['High'].iloc[-11:-1].max() if len(df_ind) >= 11 else curr['High']
        
        stock = StockData(sym, df_ind, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), curr, prev, is_vol, swing_high_10)
        cf = _extract_complex_features(stock, ctx)
        alt = AltData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        _, _, factors, _ = _evaluate_omni_matrix(stock, ctx, cf, alt)
        return factors

    def test_indicator_shapes_and_nans(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        self.assertEqual(len(df_res), len(self.mock_df), "输出的 DataFrame 长度发生截断缩水！")
        
        core_cols = ['RSI', 'MACD', 'ATR', 'CMF', 'VPT_ZScore']
        for col in core_cols:
            self.assertIn(col, df_res.columns, f"关键指标列 {col} 缺失！")

    def test_indicator_value_bounds_parameterized(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        
        bounds_matrix = [
            ('RSI', 0.0, 100.0),
            ('CMF', -1.0, 1.0),
            ('ATR', 0.0, None),
            ('Volume', 0.0, None),
            ('SuperTrend_Up', 0.0, 1.0),
            ('Smart_Money_Flow', -1.0, 1.0),
            ('CVD_Trend', -1.0, 1.0)
        ]
        
        for col, min_val, max_val in bounds_matrix:
            with self.subTest(indicator=col):
                self.assertIn(col, df_res.columns, f"严重异常：特征矩阵中丢失了指标列 {col}！")
                
                valid_vals = df_res[col].dropna()
                self.assertTrue(len(valid_vals) > 0, f"致命异常：{col} 列的计算结果全部为 NaN 毒药！")
                
                if min_val is not None:
                    self.assertTrue((valid_vals >= min_val - 1e-5).all(), 
                                    f"数学推导越界：{col} 公式存在下限物理极值 {min_val}，检测到实际值击穿下限！")
                if max_val is not None:
                    self.assertTrue((valid_vals <= max_val + 1e-5).all(), 
                                    f"数学推导越界：{col} 公式存在上限物理极值 {max_val}，检测到实际值击穿上限！")
            
    def test_rsi_logic_on_uptrend(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_rsi = df_res['RSI'].iloc[-30:]
        self.assertTrue((tail_rsi > 50).all(), 
                        "数学推导失败：在严格构造的单调上涨序列中，RSI 必然严格大于 50 的多空分水岭。实际却跌破 50！")
        
    def test_macd_logic_on_uptrend(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_df = df_res.iloc[-30:]
        self.assertTrue((tail_df['MACD'] > 0).all(), 
                        "数学推导失败：在持续单边上涨趋势中，MACD 必须恒为正数。当前 MACD 跌落水下！")
        
    def test_macd_cross_detection(self):
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
                        f"数学推导失败(T-1)：前 58 天为单边阴跌，MACD 必须小于 Signal_Line 形成死叉预备态。")
        self.assertTrue(curr['MACD'] > curr['Signal_Line'],
                        f"数学推导失败(T)：T时刻强行灌入 46% 的巨幅阳线反包，这必定使 MACD 瞬间上穿 Signal_Line。")

    def test_rsi_bounds_extreme(self):
        close_up = 100 * (1.05 ** np.arange(30))
        df_up = pd.DataFrame({
            'Close': close_up, 'Open': close_up * 0.98, 'High': close_up * 1.02, 
            'Low': close_up * 0.97, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=30))
        res_up = calculate_indicators(df_up.copy(deep=True))
        self.assertTrue(res_up['RSI'].iloc[-1] > 98.0, 
                        "数学推导失败：连续 30 日涨停导致 RSI 必然逼近 100.0 物理极值。")
        
        close_dn = 100 * (0.95 ** np.arange(30))
        df_dn = pd.DataFrame({
            'Close': close_dn, 'Open': close_dn * 1.02, 'High': close_dn * 1.03, 
            'Low': close_dn * 0.98, 'Volume': 1000000
        }, index=pd.date_range('2023-01-01', periods=30))
        res_dn = calculate_indicators(df_dn.copy(deep=True))
        self.assertTrue(res_dn['RSI'].iloc[-1] < 2.0, 
                        "数学推导失败：连续 30 日跌停导致 RSI 必然逼近 0.0 物理极值。")

    def test_vwap_and_avwap_logic(self):
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
        
        self.assertTrue(len(vwap_tail) > 0, "数学推理异常：存在完整 K 线，VWAP_20 截面不应为空。")
        self.assertTrue(df_res['AVWAP'].iloc[:anchor_idx].isna().all(), 
                        "数学推导失败：天量锚点前 AVWAP 应当必定为 NaN。发生了未来函数穿透！")
        self.assertAlmostEqual(df_res['AVWAP'].iloc[anchor_idx], tp_anchor, places=4, 
                               msg=f"数学推导失败：锚点首日的 AVWAP 必定等于当天的 Typical Price = {tp_anchor:.4f}。")
        
        avwap_val = df_res['AVWAP'].iloc[anchor_idx]
        self.assertTrue(df.iloc[anchor_idx]['Low'] <= avwap_val <= df.iloc[anchor_idx]['High'], 
                        "物理因果律失效：AVWAP 在数学上永远不可能突破当天 K 线的 [Low, High] 绝对实体物理包络线！")

    def test_chandelier_exit_logic(self):
        df_res = calculate_indicators(self.mock_df.copy(deep=True))
        tail_df = df_res.iloc[-50:]
        
        self.assertIn('Chandelier_Exit', df_res.columns)
        self.assertTrue((tail_df['Chandelier_Exit'] < tail_df['Highest_22']).all(), 
                        "数学推导失败：因为 ATR 必定非负，所以止损线必定严格低于或等于 22 日最高价。")
        self.assertTrue((tail_df['Chandelier_Exit'] < tail_df['Close']).all(), 
                        "防线崩塌推导：这是一个波动极小的单调上涨趋势，安全垫必须稳稳托底在最新收盘价下方。")
        self.assertTrue(tail_df['Chandelier_Exit'].iloc[-1] > tail_df['Chandelier_Exit'].iloc[0], 
                        "棘轮推导失败 (Ratchet Effect)：止损线必须体现出跟随利润向上抬升的棘轮特性！")

    def test_factor_breakaway_gap(self):
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
        self.assertIn("突破缺口", factors, "形态推导失败：跳空幅度达 3.3% 却未触发【突破缺口】动能因子！")

    def test_factor_amd_and_spring(self):
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
        self.assertIn("威科夫弹簧(Spring)", factors, "形态推导失败：未触发【威科夫弹簧】！")
        self.assertIn("AMD操盘", factors, "形态推导失败：机构诱空拉升的【AMD操盘模型】被漏判！")

    def test_factor_volume_surge(self):
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
        self.assertIn("巨量滞涨", factors_stag, "形态推导失败：完美契合【底部冰山吸筹】特征却未触发巨量滞涨因子！")

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
        self.assertIn("口袋支点", factors_pp, "形态推导失败：完全满足【口袋支点】破局条件却被遗漏！")
    
    def test_nan_and_zero_handling(self):
        periods = 250
        dates = pd.date_range('2023-01-01', periods=periods)
        df = pd.DataFrame({
            'Open': 100.0, 'High': 105.0, 'Low': 95.0, 'Close': 100.0, 'Volume': 1000000.0
        }, index=dates)
        
        df.iloc[10:15, :] = np.nan
        
        df.iloc[50:55, df.columns.get_loc('Open')] = 120.0
        df.iloc[50:55, df.columns.get_loc('High')] = 120.0
        df.iloc[50:55, df.columns.get_loc('Low')] = 120.0
        df.iloc[50:55, df.columns.get_loc('Close')] = 120.0
        
        df.iloc[80:85, df.columns.get_loc('Volume')] = 0.0
        
        df_res = calculate_indicators(df.copy(deep=True))
        
        self.assertEqual(len(df_res), periods, "含有 NaN 的输入导致了数据管道被粗暴截断！")
        self.assertTrue(df_res['SMA_50'].iloc[:49].isna().all(), "时序推导失败：SMA_50 发生了不合逻辑的未来函数渗透！")
        self.assertFalse(pd.isna(df_res['SMA_50'].iloc[50]), "时序推导失败：SMA_50 盲区后未能正常输出！")
        self.assertTrue(df_res['SMA_200'].iloc[:199].isna().all(), "时序推导失败：SMA_200 缓冲不符！")
        
        cmf_flat = df_res['CMF'].iloc[52]
        self.assertTrue(np.isfinite(cmf_flat), "极值崩溃推导：未加 1e-10 保护导致 CMF 爆出 NaN/Inf！")
        
        vwap_zero_vol = df_res['VWAP_20'].iloc[82]
        self.assertTrue(np.isfinite(vwap_zero_vol), "极值崩溃推导：VWAP_20 爆出 NaN/Inf！")
        
        numeric_df = df_res.select_dtypes(include=[np.number])
        has_inf = np.isinf(numeric_df).values.any()
        self.assertFalse(has_inf, "矩阵崩塌推导：全量指标中残留了 Inf (无穷大) 恶性噪音。")
    
    @unittest.skipUnless(os.getenv("RUN_PERF") == "1", "跳过性能压测。运行需配置环境变量: RUN_PERF=1")
    def test_performance_large_dataset(self):
        periods = 5000  
        dates = pd.date_range(start='2000-01-01', periods=periods, freq='D')
        
        base_px = np.linspace(10, 100, periods)
        df_large = pd.DataFrame({
            'Open': base_px * 0.98,
            'High': base_px * 1.05,
            'Low': base_px * 0.95,
            'Close': base_px,
            'Volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)

        start_time = time.perf_counter()
        df_res = calculate_indicators(df_large)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        
        self.assertEqual(len(df_res), periods, "输出的特征矩阵长度发生了灾难性截断！")
        print(f"\n[性能探针] 5000 根 K 线全维特征萃取耗时: {elapsed_ms:.2f} ms")
        self.assertTrue(elapsed_ms < 1500, f"💥 性能崩塌！处理耗时 {elapsed_ms:.2f} ms，疑似存在非向量化的 O(N²) 慢循环陷阱！")

if __name__ == '__main__':
    unittest.main(verbosity=2)
