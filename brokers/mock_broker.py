import uuid
import random
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from .base_gateway import BaseBrokerGateway, Order, Position, BrokerOrderStatus

logger = logging.getLogger("MockBroker")

class MockBrokerGateway(BaseBrokerGateway):
    """
    模拟券商网关 (Mock Implementation)
    内置极简的撮合引擎，支持随机滑点与延迟成交模拟。
    专门用于网关代码逻辑的单元测试与沙盒演习。
    """
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self._orders = {}
        self._cash = 100000.0
        self._positions = {}

    def connect(self) -> bool:
        self.is_connected = True
        logger.info("🟢 Mock Broker 虚拟专线已连接，进入纸面交易模式。")
        return True

    def disconnect(self):
        self.is_connected = False
        logger.info("🔴 Mock Broker 虚拟专线已断开。")

    def get_account_summary(self) -> Dict[str, float]:
        return {
            "total_cash": self._cash, 
            "buying_power": self._cash * 2, 
            "net_liquidation": self._cash + sum(qty * 100.0 for qty in self._positions.values())
        }

    def get_positions(self) -> List[Position]:
        return [Position(sym, qty, 100.0, qty*100.0, 0.0) for sym, qty in self._positions.items() if qty != 0]

    def place_order(self, order: Order) -> Optional[str]:
        broker_oid = f"MOCK_{uuid.uuid4().hex[:10]}"
        self._orders[broker_oid] = {
            "order": order,
            "status": "OPEN",
            "filled_qty": 0.0,
            "avg_price": 0.0,
            "created_ts": datetime.now(timezone.utc).timestamp()
        }
        logger.info(f"📤 [Mock 交易所] 接收委托: {order.side} {order.qty}股 {order.symbol} -> {broker_oid}")
        return broker_oid
        
    def fetch_order(self, broker_oid: str) -> BrokerOrderStatus:
        if broker_oid not in self._orders:
            raise ValueError(f"订单 {broker_oid} 在 Mock 交易所中不存在")
            
        mock_ord = self._orders[broker_oid]
        
        # 🤖 模拟撮合核心逻辑：订单存活超过 0.5 秒后自动转为 FILLED，并附带双边滑点
        if mock_ord["status"] == "OPEN":
            time_alive = datetime.now(timezone.utc).timestamp() - mock_ord["created_ts"]
            if time_alive > 0.5:
                mock_ord["status"] = "FILLED"
                mock_ord["filled_qty"] = mock_ord["order"].qty
                
                # 模拟基准价格与正态分布滑点 (均值 1.5 bps)
                base_px = mock_ord["order"].limit_price if mock_ord["order"].limit_price else 100.0
                slip_pct = random.gauss(1.5, 1.0) / 10000.0
                
                if mock_ord["order"].side == "BUY":
                    mock_ord["avg_price"] = base_px * (1 + slip_pct)
                    self._cash -= mock_ord["avg_price"] * mock_ord["filled_qty"]
                    self._positions[mock_ord["order"].symbol] = self._positions.get(mock_ord["order"].symbol, 0) + mock_ord["filled_qty"]
                else:
                    mock_ord["avg_price"] = base_px * (1 - slip_pct)
                    self._cash += mock_ord["avg_price"] * mock_ord["filled_qty"]
                    self._positions[mock_ord["order"].symbol] = self._positions.get(mock_ord["order"].symbol, 0) - mock_ord["filled_qty"]
                    
        return BrokerOrderStatus(
            broker_oid=broker_oid,
            status=mock_ord["status"],
            filled_qty=mock_ord["filled_qty"],
            avg_fill_price=mock_ord["avg_price"]
        )

    def cancel_order(self, broker_oid: str) -> bool:
        if broker_oid in self._orders and self._orders[broker_oid]["status"] == "OPEN":
            self._orders[broker_oid]["status"] = "CANCELED"
            logger.info(f"🛑 [Mock 交易所] 撤单成功: {broker_oid}")
            return True
        return False
        
    def get_latest_price(self, symbol: str) -> float:
        # 极简模拟：返回一个波动的随机价格
        return round(random.uniform(50.0, 300.0), 2)
