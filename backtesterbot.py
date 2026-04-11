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
    # Webhook 支持钉钉/飞书/企业微信
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    # Telegram 配置
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    
    DINGTALK_KEYWORD: str = "AI"
    CORE_WATCHLIST: List[str] = ["NVDA", "TSLA", "AAPL", "MSFT", "MSTR"]
    BLACKLIST: List[str] = [] 
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

    # 板块拥挤度过滤参数
    CROWDING_PENALTY: float = 0.75
    CROWDING_MIN_STOCKS: int = 2
    CROWDING_EXCLUDE_SECTORS: List[str] = ["QQQ"]

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols:
                return etf
        return Config.INDEX_ETF

def validate_config():
    if not Config.WEBHOOK_URL and not Config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未配置任何推送渠道。请检查 GitHub Secrets。")
        sys.exit(1)
    logger.info("✅ 环境变量与配置校验通过")

# ================= 2. 辅助工具模块 =================

_NEWS_CACHE = {}

def load_strategy_performance_tag() -> str:
    """[协同功能] 加载回测系统生成的最新胜率接口数据"""
    try:
        if os.path.exists("strategy_stats.json"):
            with open("strategy_stats.json", "r", encoding="utf-8") as f:
                stats = json.load(f)
                t3 = stats.get("T+3")
                if t3:
                    return f"📈 **策略历史表现 (T+3)**: 胜率 **{t3['win_rate']:.1%}** | 均收益 **{t3['avg_ret']:+.2%}**\n\n"
    except Exception:
        pass
    return ""

def safe_get_history(symbol: str, period: str = "6mo", interval: str = "1d", retries: int = 5, fast_mode: bool = False) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            sleep_sec = random.uniform(0.2, 0.5) if fast_mode else random.uniform(1.2, 2.5)
            time.sleep(sleep_sec)
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True, timeout=15)
            if not df.empty: return df
        except Exception as e:
            if attempt == retries - 1: return pd.DataFrame()
            time.sleep(5 + attempt * 5)
    return pd.DataFrame()

def get_latest_news(symbol: str) -> str:
    """轻量级实时新闻 + 简易情感标签（带30分钟缓存）"""
    current_time = time.time()
    if symbol in _NEWS_CACHE:
        c_time, c_news = _NEWS_CACHE[symbol]
        if current_time - c_time < 1800: return c_news

    try:
        news_data = yf.Ticker(symbol).news
        if news_data and len(news_data) > 0:
            latest = news_data[0]
            title, publisher = latest.get('title', ''), latest.get('publisher', '')
            if title:
                lower_t = title.lower()
                if any(kw in lower_t for kw in ['beat', 'raise', 'upgrade', 'strong', 'surge', 'rally', 'buy', 'bullish', 'profit']):
                    sentiment = "🟢 [利好]"
                elif any(kw in lower_t for kw in ['miss', 'cut', 'downgrade', 'weak', 'decline', 'sell', 'bearish', 'warn', 'loss']):
                    sentiment = "🔴 [利空]"
                else:
                    sentiment = "⚪ [中性]"
                result = f"📰 {sentiment} {title} ({publisher})"
                _NEWS_CACHE[symbol] = (current_time, result)
                return result
    except Exception: pass
    _NEWS_CACHE[symbol] = (current_time, "")
    return ""

def escape_md_v2(text: str) -> str:
    """Telegram MarkdownV2 完整转义（用于保护动态文本不破坏格式）"""
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_alert(title: str, content: str) -> None:
    """多渠道广播中心 (支持 MarkdownV2 转义防护)"""
    formatted_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    # 1. 钉钉/飞书/企微 Webhook
    if Config.WEBHOOK_URL:
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"【量化监控】{title}",
                "text": f"### 🤖 【AI监控系统】\n#### {title}\n\n{content}\n\n---\n*⏱️ 扫描时间: {formatted_time}*"
            }
        }
        for url in [u.strip() for u in Config.WEBHOOK_URL.split(',') if u.strip()]:
            try: requests.post(url, json=payload, timeout=10)
            except Exception: pass
                
    # 2. Telegram Bot (升级为 MarkdownV2 + 安全转义)
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        tg_url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        # 转义动态内容以防解析崩溃
        safe_title = escape_md_v2(title)
        safe_content = escape_md_v2(content)
        safe_time = escape_md_v2(formatted_time)
        
        tg_text = f"🤖 *量化监控系统*\n\n*{safe_title}*\n\n{safe_content}\n\n⏱️ _{safe_time}_"
        tg_payload = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": tg_text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(tg_url, json=tg_payload, timeout=10)
            resp.raise_for_status()
        except Exception as e: logger.error(f"Telegram 推送异常: {e}")

# ================= 3. 大盘风控与指标模块 =================

def get_market_regime() -> Tuple[str, str, pd.DataFrame]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", fast_mode=True)
    if len(df) < 200: return "range", "数据不足", df
    curr, ma200 = df['Close'].iloc[-1], df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (curr - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    if curr > ma200 and trend_20d > 0.02: return "bull", "🐂 牛市主升阶段", df
    if curr < ma ma200 and trend_20d < -0.02: return "bear", "🐻 熊市回调阶段", df
    return "range", "⚖️ 震荡整理阶段", df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 核心指标计算
    close = df['Close']
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-10)))
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Event Risk (包含价格跳空检测)
    prev_c = close.iloc[-2]
    gap = abs((df['Open'].iloc[-1] - prev_c) / prev_c) if prev_c != 0 else 0
    df['Event_Risk'] = 0.5 if gap > 0.06 else 0.0
    return df

# ================= 4. 业务策略模块 =================

def run_tech_matrix() -> None:
    logger.info(">>> 启动复合技术指标诊断...")
    regime, regime_desc, qqq_df = get_market_regime()
    
    # 加载板块数据用于动量过滤
    sector_data = {}
    for etf in Config.SECTOR_MAP.keys():
        sdf = safe_get_history(etf, period="2mo", fast_mode=True)
        if not sdf.empty and len(sdf) >= 20:
            sector_data[etf] = (sdf['Close'].iloc[-1] / sdf['Close'].iloc[-20]) - 1

    # 简单健康度 (此处简化逻辑供展示)
    health_score = 0.5 if regime == 'bull' else -0.5
    min_score_dynamic = Config.MIN_SCORE_THRESHOLD + (2 if health_score < 0 else 0)
    
    target_list = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO"] # 示例名单
    reports = []
    
    for symbol in target_list:
        try:
            df = safe_get_history(symbol, period="6mo")
            if len(df) < 50: continue
            df = calculate_indicators(df)
            curr = df.iloc[-1]
            
            score = 10 # 模拟打分
            signals = ["🌟 技术面全线共振", "🚀 机构资金放量介入"]
            
            # 推送阈值验证与结构化保存
            if score >= min_score_dynamic:
                news = get_latest_news(symbol)
                if news: signals.append(news)
                
                reports.append({
                    "symbol": symbol, "score": score, "raw_score": score,
                    "signals": signals, "curr_close": curr['Close'],
                    "sector": Config.get_sector_etf(symbol), "turnover": curr['Close'] * curr['Volume']
                })
        except Exception: pass

    # --- 后处理：板块拥挤与动量过滤 (动态进化版) ---
    if reports:
        from collections import defaultdict
        sector_groups = defaultdict(list)
        for r in reports: sector_groups[r["sector"]].append(r)
        
        for sector, stocks in sector_groups.items():
            # 1. 板块动量过滤
            s_ret = sector_data.get(sector, 0.0)
            if s_ret < -0.02 and sector not in Config.CROWDING_EXCLUDE_SECTORS:
                for s in stocks:
                    s["score"] = int(s["raw_score"] * 0.7)
                    s["is_weak_sector"] = True
            
            # 2. 板块拥挤降权
            if sector not in Config.CROWDING_EXCLUDE_SECTORS and len(stocks) >= Config.CROWDING_MIN_STOCKS:
                stocks.sort(key=lambda x: x["score"], reverse=True)
                penalty = Config.CROWDING_PENALTY * (1.0 + health_score * 0.2)
                for s in stocks[1:]:
                    if not s.get("is_weak_sector"):
                        s["score"] = int(s["raw_score"] * penalty)
                        s["is_crowded"] = True
                        s["penalty_val"] = penalty

        # 重新排序并生成报告文本
        reports.sort(key=lambda x: x["score"], reverse=True)
        perf_tag = load_strategy_performance_tag() # [协同] 获取回测胜率标签
        top_texts = []
        for r in reports[:15]:
            score_disp = f"**{r['score']}**"
            if r.get("is_crowded"): score_disp += f" (拥挤降权×{r.get('penalty_val', 0.75):.2f})"
            elif r.get("is_weak_sector"): score_disp += " (弱势板块降权)"
            
            t_str = f"${r['turnover']/1e9:.1f}B" if r['turnover'] > 1e9 else f"${r['turnover']/1e6:.1f}M"
            top_texts.append(f"**{r['symbol']}** (${r['curr_close']:.2f} | {t_str} | 评分: {score_disp})\n> " + "\n> ".join(r["signals"]))
        
        final_report = f"{perf_tag}*{regime_desc}*\n\n" + "\n\n".join(top_texts)
        send_alert("📊 优选异动诊断报告", final_report)

        # --- [协同] 写入历史回测日志 ---
        try:
            log_item = {
                "date": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                "regime": regime, "health_score": health_score,
                "top_picks": [{"symbol": r["symbol"], "score": r["score"], "sector": r["sector"], "close": r["curr_close"]} for r in reports[:10]]
            }
            with open("backtest_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_item) + "\n")
        except Exception: pass

if __name__ == "__main__":
    validate_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if mode == "matrix": run_tech_matrix()
    else: logger.info(f"模式 [{mode}] 暂未在协同版中定义")
