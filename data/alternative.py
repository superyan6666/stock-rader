import os
import json
import threading
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Dict
import yfinance as yf
from utils.logger import logger
from config import Config, _GLOBAL_HEADERS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_GLOBAL_SESSION = requests.Session()
_retry_strategy = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_GLOBAL_SESSION.mount("https://", HTTPAdapter(max_retries=_retry_strategy, pool_connections=20, pool_maxsize=20))
_GLOBAL_SESSION.headers.update(_GLOBAL_HEADERS)

_DAILY_EXT_CACHE = {}; _DAILY_EXT_LOCK = threading.Lock()
_ALT_DATA_CACHE = {}; _ALT_DATA_LOCK = threading.Lock()

def _init_ext_cache():
    global _DAILY_EXT_CACHE
    with _DAILY_EXT_LOCK:
        if not _DAILY_EXT_CACHE:
            if os.path.exists(Config.EXT_CACHE_FILE):
                try:
                    with open(Config.EXT_CACHE_FILE, 'r', encoding='utf-8') as f: _DAILY_EXT_CACHE = json.load(f)
                except Exception as e: logger.debug(f"Exception suppressed: {e}")
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if _DAILY_EXT_CACHE.get("date") != today: _DAILY_EXT_CACHE = {"date": today, "sentiment": {}, "alt": {}}

def _save_ext_cache():
    with _DAILY_EXT_LOCK:
        try:
            temp_ext = f"{Config.EXT_CACHE_FILE}.{threading.get_ident()}.tmp"
            with open(temp_ext, 'w', encoding='utf-8') as f: json.dump(_DAILY_EXT_CACHE, f)
            os.replace(temp_ext, Config.EXT_CACHE_FILE)
        except Exception as e: logger.debug(f"Exception suppressed: {e}")

def fetch_global_wsb_data() -> Dict[str, float]:
    with _ALT_DATA_LOCK:
        if "WSB_ACCEL_GLOBAL" in _ALT_DATA_CACHE: return _ALT_DATA_CACHE["WSB_ACCEL_GLOBAL"]
    cache_file = Config.WSB_CACHE_FILE
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    history = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f: history = json.load(f)
        except Exception as e: logger.debug(f"Exception suppressed: {e}")
    current_data = {}
    try:
        resp = _GLOBAL_SESSION.get("https://tradestie.com/api/v1/apps/reddit", timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                tk = item.get('ticker')
                if tk and item.get('sentiment') == 'Bullish': current_data[tk] = item.get('no_of_comments', 0)
    except Exception as e:
        logger.debug(f"WSB API 熔断保护触发: {e}")
    if current_data: history[today_str] = current_data
    sorted_dates = sorted(history.keys())
    if len(sorted_dates) > 5:
        for d in sorted_dates[:-5]: del history[d]
    try:
        temp_wsb = f"{cache_file}.{threading.get_ident()}.tmp"
        with open(temp_wsb, "w", encoding="utf-8") as f: json.dump(history, f)
        os.replace(temp_wsb, cache_file)
    except Exception as e: logger.debug(f"Exception suppressed: {e}")
    wsb_accel_dict = {}
    dates = sorted(history.keys())
    if len(dates) >= 3:
        d0, d1, d2 = dates[-1], dates[-2], dates[-3]
        for tk in history[d0].keys():
            v0, v1, v2 = history[d0].get(tk, 0), history[d1].get(tk, 0), history[d2].get(tk, 0)
            wsb_accel_dict[tk] = float((v0 - v1) - (v1 - v2))
    else:
        for tk in current_data.keys(): wsb_accel_dict[tk] = 0.0
    with _ALT_DATA_LOCK: _ALT_DATA_CACHE["WSB_ACCEL_GLOBAL"] = wsb_accel_dict
    return wsb_accel_dict

def safe_get_sentiment_data(symbol: str) -> Tuple[float, float, float, float]:
    _init_ext_cache()
    with _DAILY_EXT_LOCK:
        if symbol in _DAILY_EXT_CACHE["sentiment"]: return tuple(_DAILY_EXT_CACHE["sentiment"][symbol])
    pcr, iv_skew, short_change, short_float = 0.0, 0.0, 0.0, 0.0
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        curr_short, prev_short = info.get('sharesShort', 0), info.get('sharesShortPriorMonth', 0)
        short_float = info.get('shortPercentOfFloat', 0)
        if curr_short and prev_short and prev_short > 0: short_change = (curr_short - prev_short) / prev_short
        exps = tk.options
        if exps:
            opt = tk.option_chain(exps[0]) 
            c_vol = opt.calls['volume'].sum() if 'volume' in opt.calls else 0
            p_vol = opt.puts['volume'].sum() if 'volume' in opt.puts else 0
            if c_vol > 0: pcr = p_vol / c_vol
            c_iv = opt.calls['impliedVolatility'].median() if 'impliedVolatility' in opt.calls else 0
            p_iv = opt.puts['impliedVolatility'].median() if 'impliedVolatility' in opt.puts else 0
            iv_skew = p_iv - c_iv
    except Exception as e: logger.debug(f"Exception suppressed: {e}")
    with _DAILY_EXT_LOCK: _DAILY_EXT_CACHE["sentiment"][symbol] = (pcr, iv_skew, short_change, short_float)
    _save_ext_cache()
    return pcr, iv_skew, short_change, short_float

def safe_get_alt_data(symbol: str) -> Tuple[float, float, float, str]:
    _init_ext_cache()
    with _DAILY_EXT_LOCK:
        if symbol in _DAILY_EXT_CACHE["alt"]: return tuple(_DAILY_EXT_CACHE["alt"][symbol])
    insider_net_buy, analyst_mom, nlp_score, news_summary = 0.0, 0.0, 0.0, ""
    try:
        tk = yf.Ticker(symbol)
        try:
            insiders = tk.insider_transactions
            if insiders is not None and not insiders.empty and 'Shares' in insiders.columns:
                recent = insiders.head(20)
                buys = recent[recent['Shares'] > 0]['Shares'].sum()
                sells = recent[recent['Shares'] < 0]['Shares'].abs().sum()
                if (buys + sells) > 0: insider_net_buy = (buys - sells) / (buys + sells)
        except Exception as e: logger.debug(f"Exception suppressed: {e}")
        try:
            upgrades = tk.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                upgrades.index = pd.to_datetime(upgrades.index, utc=True)
                recent_1y = upgrades[upgrades.index >= pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=365)].copy()
                if not recent_1y.empty and 'Action' in recent_1y.columns:
                    action_map = {'up': 1, 'down': -1, 'main': 0, 'init': 0.5, 'reit': 0}
                    recent_1y['score'] = recent_1y['Action'].map(action_map).fillna(0)
                    monthly_scores = recent_1y['score'].resample('30D').sum()
                    if len(monthly_scores) >= 3:
                        analyst_mom = (monthly_scores.iloc[-1] - monthly_scores.iloc[:-1].mean()) / (monthly_scores.iloc[:-1].std() + 1e-5)
                    else: analyst_mom = recent_1y['score'].sum() * 0.1 
        except Exception as e: logger.debug(f"Exception suppressed: {e}")
    except Exception as e: logger.debug(f"Exception suppressed: {e}")
    with _DAILY_EXT_LOCK: _DAILY_EXT_CACHE["alt"][symbol] = (insider_net_buy, analyst_mom, nlp_score, news_summary)
    _save_ext_cache()
    return insider_net_buy, analyst_mom, nlp_score, news_summary

def check_earnings_risk(symbol: str) -> bool:
    try:
        tk = yf.Ticker(symbol)
        now_date = datetime.now(timezone.utc).date()
        try:
            cal = tk.calendar
            if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
                ed = cal['Earnings Date']
                if ed and hasattr(ed[0], 'date') and 0 <= (ed[0].date() - now_date).days <= 5: return True
        except Exception as e: logger.debug(f"Exception suppressed: {e}")
        try:
            ed_df = tk.earnings_dates
            if ed_df is not None and not ed_df.empty:
                for d in ed_df.index:
                    if hasattr(d, 'date') and d.date() >= now_date and 0 <= (d.date() - now_date).days <= 5: return True
        except Exception as e: logger.debug(f"Exception suppressed: {e}")
    except Exception as e: logger.debug(f"Exception suppressed: {e}")
    return False
