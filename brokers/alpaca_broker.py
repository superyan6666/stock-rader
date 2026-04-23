import uuid
import requests
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from .base_gateway import BaseBrokerGateway, Order, Position, BrokerOrderStatus

logger = logging.getLogger("AlpacaBroker")

class AlpacaBrokerGateway(BaseBrokerGateway):
    """
    🚀 真实的 Alpaca REST API 交易网关 (独立模块)
    实现与 Alpaca 纸面/实盘环境的网络层对接。
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get('API_KEY', '')
        self.api_secret = config.get('API_SECRET', '')
        self.base_url = config.get('BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }

    def connect(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if resp.status_code == 200:
                self.is_connected = True
                logger.info("🟢 Alpaca 真实/纸面专线连接成功！")
                return True
            logger.error(f"🔴 Alpaca 连接失败: {resp.text}")
            return False
        except Exception as e:
            logger.error(f"🔴 Alpaca 连接异常: {e}")
            return False

    def disconnect(self):
        self.is_connected = False
        logger.info("🔴 Alpaca 专线已断开。")

    def _map_status(self, alpaca_status: str) -> str:
        mapping = {
            "new": "OPEN", "accepted": "OPEN", "partially_filled": "PARTIALLY_FILLED",
            "filled": "FILLED", "done_for_day": "FILLED", "canceled": "CANCELED",
            "expired": "EXPIRED", "replaced": "OPEN", "pending_cancel": "OPEN",
            "pending_replace": "OPEN", "rejected": "REJECTED", "suspended": "REJECTED"
        }
        return mapping.get(alpaca_status, "OPEN")

    def get_account_summary(self) -> Dict[str, float]:
        resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "total_cash": float(data.get("cash", 0.0)),
                "buying_power": float(data.get("buying_power", 0.0)),
                "net_liquidation": float(data.get("portfolio_value", 0.0))
            }
        return {"total_cash": 0.0, "buying_power": 0.0, "net_liquidation": 0.0}

    def get_positions(self) -> List[Position]:
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
        positions = []
        if resp.status_code == 200:
            for p in resp.json():
                positions.append(Position(
                    symbol=p["symbol"],
                    qty=float(p["qty"]),
                    avg_price=float(p["avg_entry_price"]),
                    market_value=float(p["market_value"]),
                    unrealized_pnl=float(p["unrealized_pl"])
                ))
        return positions

    def place_order(self, order: Order) -> Optional[str]:
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.lower(),
            "type": "market" if order.order_type == "MARKET" else "limit",
            "time_in_force": "day",
            "client_order_id": order.client_oid
        }
        if order.limit_price and payload["type"] == "limit":
            payload["limit_price"] = str(round(order.limit_price, 2))

        resp = requests.post(url, json=payload, headers=self.headers, timeout=10)
        if resp.status_code in [200, 201]:
            data = resp.json()
            logger.info(f"📤 [Alpaca] 委托成功: {order.side} {order.qty}股 {order.symbol} -> {data['id']}")
            return data["id"]
        logger.error(f"❌ [Alpaca] 委托拒绝: {resp.text}")
        return None

    def fetch_order(self, broker_oid: str) -> BrokerOrderStatus:
        url = f"{self.base_url}/v2/orders/{broker_oid}"
        resp = requests.get(url, headers=self.headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return BrokerOrderStatus(
                broker_oid=data["id"],
                status=self._map_status(data["status"]),
                filled_qty=float(data["filled_qty"]),
                avg_fill_price=float(data["filled_avg_price"] or 0.0)
            )
        raise RuntimeError(f"Alpaca Fetch Error: {resp.text}")

    def cancel_order(self, broker_oid: str) -> bool:
        url = f"{self.base_url}/v2/orders/{broker_oid}"
        resp = requests.delete(url, headers=self.headers, timeout=10)
        if resp.status_code in [200, 204]:
            logger.info(f"🛑 [Alpaca] 撤单成功: {broker_oid}")
            return True
        return False

    def get_latest_price(self, symbol: str) -> float:
        # 实盘环境一般通过 WebSocket/yfinance 统一下发，此处占位
        return 0.0
