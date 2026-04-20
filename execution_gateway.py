import sqlite3
import json
import time
import logging
import threading
import os
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ================= 1. 全局配置与铁律防线 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Gateway] %(levelname)s - %(message)s')
logger = logging.getLogger("Gateway")

DB_PATH = ".quantbot_data/order_state.db"
TCA_LOG_PATH = ".quantbot_data/tca_history.jsonl"

# 致命风险防爆墙参数
MAX_RETRY_COUNT = 3
FAT_FINGER_MAX_USD = 50000.0  # 胖手指防御：单笔订单绝对金额上限

# ================= 2. SQLite 高频状态账本 (IPC + 最终一致性) =================
class OrderLedger:
    """
    基于 SQLite WAL 模式的订单账本。
    既作为 quantbot.py 与网关通信的 IPC 队列，又作为网关崩溃恢复的 ACID 持久化基座。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        # 开启 WAL 模式，支持并发读写，避免 Database Locked 异常
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute('pragma journal_mode=wal')
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    client_oid TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    arrival_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    filled_qty REAL DEFAULT 0.0,
                    avg_fill_price REAL DEFAULT 0.0,
                    retry_count INTEGER DEFAULT 0,
                    broker_oid TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 为状态查询建立索引，极速拉取 PENDING 订单
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON orders(status)')

    def fetch_orders_by_status(self, statuses: List[str]) -> pd.DataFrame:
        """🚀 铁律 1: 获取订单并强制转为深拷贝 DataFrame，实现内存隔离"""
        with self._get_conn() as conn:
            placeholders = ','.join(['?'] * len(statuses))
            query = f"SELECT * FROM orders WHERE status IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=statuses)
        return df.copy(deep=True)

    def update_order_status(self, client_oid: str, status: str, filled_qty: float = 0.0, avg_fill_price: float = 0.0, broker_oid: str = None):
        with self._get_conn() as conn:
            conn.execute('''
                UPDATE orders 
                SET status = ?, filled_qty = ?, avg_fill_price = ?, broker_oid = COALESCE(?, broker_oid), updated_at = CURRENT_TIMESTAMP
                WHERE client_oid = ?
            ''', (status, filled_qty, avg_fill_price, broker_oid, client_oid))
            
    def increment_retry(self, client_oid: str):
        with self._get_conn() as conn:
            conn.execute('UPDATE orders SET retry_count = retry_count + 1 WHERE client_oid = ?', (client_oid,))

    def insert_mock_order(self, symbol: str, side: str, qty: float, arrival_price: float) -> str:
        """仅供 quantbot.py 测试注入指令使用"""
        client_oid = f"QB_{uuid.uuid4().hex[:8]}"
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO orders (client_oid, symbol, side, qty, order_type, arrival_price, status)
                VALUES (?, ?, ?, ?, 'MKT', ?, 'PENDING_SUBMIT')
            ''', (client_oid, symbol, side, qty, arrival_price))
        return client_oid

# ================= 3. 券商代理抽象层 (Broker Abstraction Layer) =================
@dataclass
class BrokerOrder:
    broker_oid: str
    status: str
    filled_qty: float
    avg_fill_price: float

class BaseBrokerGateway(ABC):
    @abstractmethod
    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None) -> BrokerOrder: pass
    
    @abstractmethod
    def fetch_order(self, broker_oid: str) -> BrokerOrder: pass
    
    @abstractmethod
    def cancel_order(self, broker_oid: str) -> bool: pass

class MockAlpacaGateway(BaseBrokerGateway):
    """用于测试与实盘仿真的 Mock 网关，模拟真实的订单延迟与撮合状态"""
    def __init__(self):
        self._mock_exchange = {}

    def submit_order(self, symbol: str, side: str, qty: float, order_type: str, limit_price: float = None) -> BrokerOrder:
        broker_oid = f"ALPACA_{uuid.uuid4().hex[:12]}"
        # 模拟撮合引擎接收订单
        self._mock_exchange[broker_oid] = {
            "status": "OPEN", "filled_qty": 0.0, "avg_fill_price": 0.0, "qty": qty, "price": limit_price or 100.0
        }
        return BrokerOrder(broker_oid, "OPEN", 0.0, 0.0)

    def fetch_order(self, broker_oid: str) -> BrokerOrder:
        if broker_oid not in self._mock_exchange:
            raise ValueError("Order not found")
        order = self._mock_exchange[broker_oid]
        
        # 模拟随着时间推移，订单变为 FILLED (供测试状态流转)
        if order["status"] == "OPEN":
            order["status"] = "FILLED"
            order["filled_qty"] = order["qty"]
            # 加入微小的滑点噪声
            order["avg_fill_price"] = order["price"] * np.random.uniform(0.999, 1.001)
            
        return BrokerOrder(broker_oid, order["status"], order["filled_qty"], order["avg_fill_price"])

    def cancel_order(self, broker_oid: str) -> bool:
        if broker_oid in self._mock_exchange:
            self._mock_exchange[broker_oid]["status"] = "CANCELED"
            return True
        return False

# ================= 4. 异步有限状态机与执行核心 (Async Order FSM) =================
class ExecutionEngine:
    def __init__(self, broker: BaseBrokerGateway, db_path: str = DB_PATH):
        self.broker = broker
        self.ledger = OrderLedger(db_path)
        self.is_running = False

    def _write_tca_log_atomic(self, order_row: pd.Series):
        """🚀 铁律 3: TCA 并发原子持久化。防止写花日志"""
        tca_record = {
            "client_oid": order_row['client_oid'],
            "symbol": order_row['symbol'],
            "side": order_row['side'],
            "qty": float(order_row['filled_qty']),
            "arrival_price": float(order_row['arrival_price']),
            "execution_price": float(order_row['avg_fill_price']),
            "timestamp": datetime_to_str()
        }
        
        # 🚀 铁律 2: 极端滑点保护运算，加入 1e-10 防 Inf 毒药
        arr_px = np.maximum(tca_record["arrival_price"], 1e-10)
        slippage_bps = (tca_record["execution_price"] - arr_px) / arr_px * 10000.0
        if tca_record["side"] == "SELL":
            slippage_bps = -slippage_bps  # 卖出时价格越低滑点越大
            
        tca_record["slippage_bps"] = float(slippage_bps)

        # 原子写入操作
        tmp_path = f"{TCA_LOG_PATH}.{threading.get_ident()}.tmp"
        try:
            # 读取历史
            history = []
            if os.path.exists(TCA_LOG_PATH):
                with open(TCA_LOG_PATH, 'r') as f:
                    history = [json.loads(line) for line in f]
            
            history.append(tca_record)
            
            # 写入临时文件并原子替换
            with open(tmp_path, 'w') as f:
                for h in history:
                    f.write(json.dumps(h) + '\n')
            os.replace(tmp_path, TCA_LOG_PATH)
            logger.info(f"📊 [TCA] 订单 {tca_record['client_oid']} 归因完成。滑点: {slippage_bps:.2f} bps")
        except Exception as e:
            logger.error(f"TCA 日志原子写入崩溃: {e}")
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def _recover_state(self):
        """🚀 最终一致性恢复：启动时接管崩溃前遗留的在途订单"""
        logger.info("🔄 启动灾难恢复序列：扫描遗留状态单...")
        df_pending = self.ledger.fetch_orders_by_status(['SUBMITTED', 'OPEN', 'PARTIALLY_FILLED'])
        if df_pending.empty:
            logger.info("✅ 账本干净，无在途遗留敞口。")
            return

        for _, row in df_pending.iterrows():
            logger.warning(f"⚠️ 发现遗留订单 {row['client_oid']} (券商ID: {row['broker_oid']})，正在与券商对齐状态...")
            if pd.notna(row['broker_oid']):
                self._sync_single_order(row)
            else:
                # 极端罕见情况：写了 SUBMITTED 但没拿到 broker_oid，直接视为 REJECTED 人工复核
                self.ledger.update_order_status(row['client_oid'], 'REJECTED')
                logger.error(f"❌ 订单 {row['client_oid']} 幽灵态，已强制 REJECTED。")

    def _fat_finger_check(self, row: pd.Series) -> bool:
        """🚀 致命风险防爆墙 1: 胖手指拦截"""
        est_val = row['qty'] * row['arrival_price']
        if est_val > FAT_FINGER_MAX_USD:
            logger.critical(f"🛑 [胖手指拦截] 订单 {row['client_oid']} 动用资金 ${est_val:.2f} 超出硬红线 ${FAT_FINGER_MAX_USD}！")
            return False
        return True

    def _process_queue(self):
        """读取量化引擎通过 SQLite 下达的新指令"""
        df_new = self.ledger.fetch_orders_by_status(['PENDING_SUBMIT'])
        for _, row in df_new.iterrows():
            oid = row['client_oid']
            
            if not self._fat_finger_check(row):
                self.ledger.update_order_status(oid, 'REJECTED')
                continue
                
            if row['retry_count'] >= MAX_RETRY_COUNT:
                logger.error(f"☠️ 订单 {oid} 重试耗尽，打入死信队列 (REJECTED)。")
                self.ledger.update_order_status(oid, 'REJECTED')
                continue

            try:
                # FSM: PENDING_SUBMIT -> SUBMITTED (向券商发送)
                logger.info(f"📤 正在向券商提交订单 {oid}: {row['side']} {row['qty']} {row['symbol']}")
                limit_px = float(row['limit_price']) if pd.notna(row['limit_price']) else None
                b_order = self.broker.submit_order(row['symbol'], row['side'], row['qty'], row['order_type'], limit_px)
                
                self.ledger.update_order_status(oid, b_order.status, b_order.filled_qty, b_order.avg_fill_price, b_order.broker_oid)
                
            except Exception as e:
                logger.error(f"提交订单 {oid} 时遭遇网络异常: {e}")
                self.ledger.increment_retry(oid)

    def _sync_single_order(self, row: pd.Series):
        """同步单个订单状态，处理 FSM 终态结算与 TCA 记录"""
        oid = row['client_oid']
        b_oid = row['broker_oid']
        
        try:
            b_order = self.broker.fetch_order(b_oid)
            
            # FSM 状态跃迁
            if b_order.status != row['status'] or b_order.filled_qty != row['filled_qty']:
                logger.info(f"🔄 订单 {oid} 状态跃迁: {row['status']} -> {b_order.status} (已成交 {b_order.filled_qty})")
                self.ledger.update_order_status(oid, b_order.status, b_order.filled_qty, b_order.avg_fill_price)
                
                # 触及终态，执行 TCA 归因
                if b_order.status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    if b_order.filled_qty > 0:
                        # 更新行数据传入 TCA，确保深拷贝内存隔离
                        row_copy = row.copy(deep=True)
                        row_copy['filled_qty'] = b_order.filled_qty
                        row_copy['avg_fill_price'] = b_order.avg_fill_price
                        self._write_tca_log_atomic(row_copy)
                        
        except Exception as e:
            logger.error(f"同步订单 {oid} (BrokerID: {b_oid}) 失败: {e}")

    def _sync_open_orders(self):
        """轮询同步在途订单状态"""
        df_open = self.ledger.fetch_orders_by_status(['SUBMITTED', 'OPEN', 'PARTIALLY_FILLED'])
        for _, row in df_open.iterrows():
            if pd.notna(row['broker_oid']):
                self._sync_single_order(row)

    def run(self):
        """核心事件循环"""
        self.is_running = True
        logger.info("🚀 QuantBot Execution Gateway 3.0 上线！底层 FSM 开始监听...")
        
        # 启动前必须先恢复历史崩溃遗留状态
        self._recover_state()
        
        try:
            while self.is_running:
                self._process_queue()
                self._sync_open_orders()
                time.sleep(1.0)  # 严格控制轮询频率，防止 API 限流
        except KeyboardInterrupt:
            logger.info("🛑 收到停机指令，网关安全下线。")
            self.is_running = False

def datetime_to_str():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

if __name__ == "__main__":
    # 单机测试入口
    mock_broker = MockAlpacaGateway()
    gateway = ExecutionEngine(mock_broker)
    
    # 模拟 QuantBot 发送两条指令测试
    # db = OrderLedger(DB_PATH)
    # db.insert_mock_order("AAPL", "BUY", 10.0, 150.00)
    # db.insert_mock_order("TSLA", "SELL", 5.0, 900.00)  # 可能会触发胖手指拦截或正常过审
    
    gateway.run()
