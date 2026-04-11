import yfinance as yf
import requests
import os
import sys
import pandas as pd
import numpy as np
import time
import random
import logging
import re
import json
from typing import List, Tuple
from datetime import datetime, timezone

# ================= 1. 日志与配置管理 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantBot")

class Config:
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    
    DINGTALK_KEYWORD: str = "AI"
    CORE_WATCHLIST: List[str] = ["NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]
    INDEX_ETF: str = "QQQ" 
    VIX_INDEX: str = "^VIX" 
    MIN_SCORE_THRESHOLD: int = 8 
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'TXN', 'INTU', 'AMAT', 'MU', 'LRCX', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'NXPI', 'MRVL', 'MCHP', 'FTNT', 'CRWD'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD', 'ROST', 'EBAY', 'TSCO', 'PDD', 'DASH', 'CPRT', 'PCAR'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA', 'TTWO', 'WBD', 'SIRI', 'CHTR'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM', 'IDXX', 'MRNA', 'ALGN', 'BMRN', 'GEHC'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'MNST', 'WBA']
    }

    CROWDING_PENALTY: float = 0.75
    CROWDING_MIN_STOCKS: int = 2
    CROWDING_EXCLUDE_SECTORS: List[str] = ["QQQ"]

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF

def load_strategy_performance_tag() -> str:
    """[协同功能] 加载来自 backtesterbot.py 生成的胜率数据"""
    try:
        if os.path.exists("strategy_stats.json"):
            with open("strategy_stats.json", "r", encoding="utf-8") as f:
                stats = json.load(f)
                t3 = stats.get("T+3")
                if t3:
                    return f"📈 **策略历史表现 (T+3)**: 胜率 **{t3['win_rate']:.1%}** | 均收益 **{t3['avg_ret']:+.2%}**\n\n"
    except Exception: pass
    return ""

def escape_md_v2(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_alert(title: str, content: str) -> None:
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    # 1. Webhook
    if Config.WEBHOOK_URL:
        payload = {"msgtype": "markdown", "markdown": {"title": f"【量化】{title}", "text": f"### 🤖 AI量化监控\n#### {title}\n\n{content}\n\n---\n*⏱️ {formatted_time}*"}}
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, timeout=10)
            except Exception: pass
                
    # 2. Telegram (MarkdownV2)
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        tg_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        safe_t, safe_c, safe_tm = escape_md_v2(title), escape_md_v2(content), escape_md_v2(formatted_time)
        tg_text = f"🤖 *量化监控系统*\n\n*{safe_t}*\n\n{safe_c}\n\n⏱️ _{safe_tm}_"
        try: requests.post(tg_url, json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": tg_text, "parse_mode": "MarkdownV2"}, timeout=10)
        except Exception: pass

# ... (此处省略中间重复的指标计算逻辑，保持与 quantbot.py 一致) ...

def run_tech_matrix() -> None:
    # 核心逻辑：扫描、打分、写入 backtest_log.jsonl
    # 并在推送前调用 load_strategy_performance_tag()
    logger.info(">>> 启动复合技术指标诊断 [monitor.py]...")
    # ... 实现代码逻辑 ...
    pass # 生产环境中替换为完整 matrix 逻辑

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if mode == "matrix": run_tech_matrix()
    else: logger.info(f"模式 {mode} 已启动。")
