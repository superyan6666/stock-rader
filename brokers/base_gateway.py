from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

# --- 标准化数据结构 ---
@dataclass
class Order:
    """标准化订单结构"""
    symbol: str
    qty: int
    side: str  # "BUY" or "SELL"
    order_type: str = "MARKET" # "MARKET", "LIMIT"
    limit_price: Optional[float] = None
    client_oid: str = ""  # 策略端生成的唯一ID
    created_at: datetime = datetime.now()

@dataclass
class Execution:
    """标准化成交回报结构"""
    client_oid: str
    broker_oid: str  # 券商端的订单ID
    symbol: str
    side: str
    exec_qty: int
    exec_price: float
    exec_time: datetime
    commission: float = 0.0

@dataclass
class Position:
    """标准化持仓结构"""
    symbol: str
    qty: int
    avg_price: float
    market_value: float
    unrealized_pnl: float

# --- 抽象基类 ---
class BaseBrokerGateway(ABC):
    """
    券商网关抽象基类 (The World Interface)
    所有具体的券商实现 (如 IBKR, TDA, Alpaca) 都必须继承此类并实现以下方法。
    这确保了策略引擎与底层券商的解耦。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """建立与券商API的连接"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    def get_account_summary(self) -> Dict[str, float]:
        """
        获取账户资金摘要
        返回范例: {"total_cash": 10000.0, "buying_power": 20000.0, "net_liquidation": 10000.0}
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取当前所有持仓"""
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Optional[str]:
        """
        发送订单
        返回: 成功返回 broker_oid (券商订单号), 失败返回 None
        """
        pass

    @abstractmethod
    def cancel_order(self, broker_oid: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """获取单个标的的最新市场价格 (用于计算持仓市值)"""
        pass
