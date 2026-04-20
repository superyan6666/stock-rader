import unittest
import pandas as pd
import numpy as np
import os
import time

# 🚀 导入待测试的核心抽象层
from execution_gateway import OrderLedger, ExecutionEngine, MockAlpacaGateway, TCA_LOG_PATH

TEST_DB_PATH = ".quantbot_data/test_order_state.db"
TEST_TCA_PATH = ".quantbot_data/test_tca_history.jsonl"

class TestExecutionGateway(unittest.TestCase):
    
    def setUp(self):
        """重置纯净的测试账本沙盒"""
        if os.path.exists(TEST_DB_PATH): os.remove(TEST_DB_PATH)
        if os.path.exists(TEST_DB_PATH + "-wal"): os.remove(TEST_DB_PATH + "-wal")
        if os.path.exists(TEST_DB_PATH + "-shm"): os.remove(TEST_DB_PATH + "-shm")
        if os.path.exists(TEST_TCA_PATH): os.remove(TEST_TCA_PATH)
        
        self.ledger = OrderLedger(TEST_DB_PATH)
        self.broker = MockAlpacaGateway()
        self.engine = ExecutionEngine(self.broker, TEST_DB_PATH)
        
        # 临时替换引擎内的 TCA 路径，防止污染实盘日志
        global TCA_LOG_PATH
        self._orig_tca = TCA_LOG_PATH
        import execution_gateway
        execution_gateway.TCA_LOG_PATH = TEST_TCA_PATH

    def tearDown(self):
        import execution_gateway
        execution_gateway.TCA_LOG_PATH = self._orig_tca

    # ---------------- 状态机跃迁与最终一致性测谎 ----------------
    def test_fsm_state_transitions(self):
        """测谎 1: 验证 PENDING -> OPEN -> FILLED 的黄金流程路径"""
        oid = self.ledger.insert_mock_order("NVDA", "BUY", 10.0, 500.0)
        
        # 第 1 步：引擎扫描队列，将 PENDING 发送至券商
        self.engine._process_queue()
        df = self.ledger.fetch_orders_by_status(['OPEN'])
        
        with self.subTest(msg="数学推导: 订单下达后，状态机必须将其从 PENDING 移除并变更为 OPEN，同时记录 broker_oid。"):
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['status'], 'OPEN')
            self.assertIsNotNone(df.iloc[0]['broker_oid'])
            
        # 第 2 步：轮询状态，Mock券商会在 Fetch 时自动将其模拟为 FILLED
        self.engine._sync_open_orders()
        df_filled = self.ledger.fetch_orders_by_status(['FILLED'])
        
        with self.subTest(msg="数学推导: 模拟券商撮合成交后，本地账本的状态机必须准确跟随跃迁为 FILLED。"):
            self.assertEqual(len(df_filled), 1)
            self.assertEqual(df_filled.iloc[0]['filled_qty'], 10.0)

    def test_crash_recovery_consistency(self):
        """测谎 2: 验证网关崩溃后的“最终一致性”恢复能力"""
        # 手动向数据库直接塞入一个脏数据：状态为 OPEN，且附带 broker_oid (模拟发单后瞬间网关断电崩溃)
        with self.ledger._get_conn() as conn:
            conn.execute('''
                INSERT INTO orders (client_oid, symbol, side, qty, order_type, arrival_price, status, broker_oid)
                VALUES ('CRASH_TEST_1', 'AMD', 'BUY', 5.0, 'MKT', 100.0, 'OPEN', 'MOCK_BROKER_ID_1')
            ''')
        
        # 在 Mock 券商中预埋好这笔订单的终态 (FILLED)
        self.broker._mock_exchange['MOCK_BROKER_ID_1'] = {
            "status": "FILLED", "filled_qty": 5.0, "avg_fill_price": 101.5, "qty": 5.0, "price": 100.0
        }
        
        # 触发重启恢复序列
        self.engine._recover_state()
        
        df = self.ledger.fetch_orders_by_status(['FILLED'])
        with self.subTest(msg="崩溃恢复推导: 网关重启必须能读取 SQLite 中遗留的 OPEN 状态单，并主动向券商拉取最新状态完成最终一致性修复！"):
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['client_oid'], 'CRASH_TEST_1')
            self.assertEqual(df.iloc[0]['avg_fill_price'], 101.5)

    # ---------------- 致命风险防爆墙测谎 ----------------
    def test_fat_finger_kill_switch(self):
        """测谎 3: 验证胖手指防线，防止错下天量订单"""
        # 插入一笔价值 1,000,000 美元的疯狂订单 (远超 50000 上限)
        oid = self.ledger.insert_mock_order("AAPL", "BUY", 10000.0, 100.0)
        
        self.engine._process_queue()
        
        df = self.ledger.fetch_orders_by_status(['REJECTED'])
        with self.subTest(msg="物理极限推导: 当订单金额 (Qty * ArrivalPrice) > FAT_FINGER_MAX_USD 时，引擎必须在不请求券商的情况下，直接将状态熔断至 REJECTED！"):
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['client_oid'], oid)

    def test_tca_slippage_math_guard(self):
        """测谎 4: 验证 TCA 滑点运算中的 `1e-10` 毒药防御"""
        # 模拟一笔到达价格为 0 的异常订单 (如停牌首日恢复造成的缺失数据)
        oid = self.ledger.insert_mock_order("GME", "BUY", 10.0, 0.0)
        self.engine._process_queue()
        
        # 同步成交，触发 TCA 写入
        self.engine._sync_open_orders()
        
        with self.subTest(msg="数学防爆墙推导: 滑点公式 (Exec - Arrival) / Arrival。若 Arrival=0 必须受到 np.maximum(x, 1e-10) 保护，不能抛出 ZeroDivisionError 并导致 TCA 持久化崩溃。"):
            self.assertTrue(os.path.exists(TEST_TCA_PATH), "TCA 日志文件未生成，说明原子写入函数崩溃了！")
            with open(TEST_TCA_PATH, 'r') as f:
                logs = [json.loads(line) for line in f]
                self.assertEqual(len(logs), 1)
                # 因为底层使用了 1e-10，(100 - 0) / 1e-10 会产生巨大的正数值，但系统必须存活！
                self.assertTrue(logs[0]['slippage_bps'] > 1000000.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
