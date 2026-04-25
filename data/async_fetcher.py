import aiohttp
import asyncio
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
import yfinance as yf
from utils.logger import logger
from data.cache_manager import _SHARED_CACHE, _API_LIMITER

async def _fetch_from_alpaca_async(session: aiohttp.ClientSession, symbol: str, period: str, interval: str) -> pd.DataFrame:
    api_key = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    if not api_key or not api_secret:
        return pd.DataFrame()
        
    now = datetime.now(timezone.utc)
    if period.endswith('d'): days = int(period[:-1])
    elif period.endswith('mo'): days = int(period[:-2]) * 30
    elif period.endswith('y'): days = int(period[:-1]) * 365
    else: days = 365
    
    start_time = (now - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    if interval == '1d': timeframe = '1Day'
    elif interval in ['60m', '1h']: timeframe = '1Hour'
    elif interval == '1wk': timeframe = '1Week'
    elif interval == '1mo': timeframe = '1Month'
    else: timeframe = '1Day'

    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {"symbols": symbol, "timeframe": timeframe, "start": start_time, "end": end_time, "limit": 10000, "adjustment": "all"}
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret, "Accept": "application/json"}

    try:
        async with session.get(url, headers=headers, params=params, timeout=10) as resp:
            if resp.status == 200:
                data = (await resp.json()).get('bars', {}).get(symbol, [])
                if not data: return pd.DataFrame()
                df = pd.DataFrame(data)
                df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            elif resp.status == 422:
                pass
            else:
                logger.debug(f"Alpaca API error: {resp.status} - {await resp.text()}")
    except Exception as e:
        logger.debug(f"Alpaca fetch failed: {e}")
    return pd.DataFrame()

async def safe_get_history_async(session: aiohttp.ClientSession, symbol: str, period: str = "1y", interval: str = "1d", retries: int = 3, fast_mode: bool = False) -> pd.DataFrame:
    safe_sym = symbol.replace('^', '_caret_').replace('/', '_slash_')
    cache_key = f"{safe_sym}_{interval}_{period}"
    
    cached_df = _SHARED_CACHE.get(cache_key)
    if cached_df is not None and not cached_df.empty:
        return cached_df.copy()
        
    if period != "1y":
        long_key = f"{safe_sym}_{interval}_1y"
        long_cached = _SHARED_CACHE.get(long_key)
        if long_cached is not None and not long_cached.empty:
            if period == "5d":
                return long_cached.tail(5).copy()
            elif period == "1mo":
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=30)
                return long_cached[long_cached.index >= cutoff].copy()
            elif period == "3mo":
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=90)
                return long_cached[long_cached.index >= cutoff].copy()
            elif period == "6mo":
                cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=183)
                return long_cached[long_cached.index >= cutoff].copy()
        
    for attempt in range(retries):
        wait_time = _API_LIMITER.acquire()
        if wait_time > 0: await asyncio.sleep(wait_time)
        
        try:
            df = await _fetch_from_alpaca_async(session, symbol, period, interval)
            if df is None or df.empty:
                logger.debug(f"Fallback to yfinance for {symbol}")
                def yf_fetch():
                    tk = yf.Ticker(symbol)
                    res = tk.history(period=period, interval=interval, prepost=False, auto_adjust=True)
                    if not res.empty and res.index.tz is None:
                        res.index = res.index.tz_localize('UTC')
                    elif not res.empty and str(res.index.tz) != 'UTC':
                        res.index = res.index.tz_convert('UTC')
                    return res
                df = await asyncio.to_thread(yf_fetch)

            if df is not None and not df.empty:
                df = df[~df.index.duplicated(keep='last')]
                df = df[df['Volume'] > 0]
                df.sort_index(inplace=True)
                _SHARED_CACHE.set(cache_key, df)
                return df.copy()
        except Exception as e:
            logger.debug(f"Fetch {symbol} failed attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1: await asyncio.sleep(2 ** attempt)
                
    return pd.DataFrame()
