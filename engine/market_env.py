import pandas as pd
import asyncio
import aiohttp
from typing import Tuple, List
from config import Config
from data.models import MarketContext
from data.async_fetcher import safe_get_history_async

def safe_get_history(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 3, fast_mode: bool = False) -> pd.DataFrame:
    async def _run():
        async with aiohttp.ClientSession() as session:
            return await safe_get_history_async(session, symbol, period, interval, retries, fast_mode)
    try:
        loop = asyncio.get_running_loop()
        return asyncio.run_coroutine_threadsafe(_run(), loop).result()
    except RuntimeError:
        return asyncio.run(_run())

def get_vix_level(qqq_df: pd.DataFrame = None) -> Tuple[float, str]:
    df = safe_get_history(Config.VIX_INDEX, period="5d", interval="1d", fast_mode=True)
    vix = df['Close'].ffill().iloc[-1] if not df.empty else 18.0
    if vix > 30: return vix, f"🚨 极其恐慌 (VIX: {vix:.2f})"
    if vix > 25: return vix, f"⚠️ 市场恐慌 (VIX: {vix:.2f})"
    if vix < 15: return vix, f"✅ 市场平静 (VIX: {vix:.2f})"
    return vix, f"⚖️ 正常波动 (VIX: {vix:.2f})"

def get_market_regime(active_pool: List[str] = None) -> Tuple[str, str, pd.DataFrame, bool, bool]:
    df = safe_get_history(Config.INDEX_ETF, period="1y", interval="1d", fast_mode=True)
    if len(df) < 200: return "range", "默认震荡", df, False, False
    c_close, ma200 = df['Close'].ffill().iloc[-1], df['Close'].rolling(200).mean().iloc[-1]
    trend_20d = (c_close - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    if c_close > ma200:
        return ("bull", "🐂 牛市主升", df, False, False) if trend_20d > 0.02 else ("range", "⚖️ 牛市震荡", df, False, False)
    else:
        return ("bear", "🐻 熊市回调", df, False, False) if trend_20d < -0.02 else ("rebound", "🦅 超跌反弹", df, False, False)

def _build_market_context() -> MarketContext:
    qqq_df = safe_get_history(Config.INDEX_ETF, "1y", "1d", fast_mode=True)
    spy_df = safe_get_history("SPY", "1y", "1d", fast_mode=True)
    tlt_df = safe_get_history("TLT", "1y", "1d", fast_mode=True)
    dxy_df = safe_get_history("DX-Y.NYB", "1y", "1d", fast_mode=True)
    
    vix, vix_desc = get_vix_level(qqq_df)
    regime, regime_desc, _, _, _ = get_market_regime()
    
    return MarketContext(
        regime=regime, regime_desc=regime_desc, w_mul=1.0, xai_weights={}, vix_current=vix, 
        vix_desc=vix_desc, vix_scalar=1.0, max_risk=0.015, macro_gravity=False, 
        is_credit_risk_high=False, vix_inv=False, qqq_df=qqq_df, 
        macro_data={'spy': spy_df, 'tlt': tlt_df, 'dxy': dxy_df}, 
        total_market_exposure=1.0, health_score=1.0, pain_warning="", dynamic_min_score=8.0
    )
