import os
import json

_GLOBAL_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json'
}

class Config:
    WEBHOOK_URL: str = os.environ.get('WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN: str = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '')
    DINGTALK_KEYWORD: str = "AI"
    INDEX_ETF: str = "QQQ" 
    VIX_INDEX: str = "^VIX" 
    
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'QCOM', 'AMD', 'INTC', 'CRM', 'ADBE'],
        'XLY': ['AMZN', 'TSLA', 'BKNG', 'SBUX', 'MAR', 'MELI', 'LULU', 'HD', 'ROST'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'CMCSA', 'TMUS', 'EA', 'TTWO'],
        'XLV': ['AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'BIIB', 'ILMN', 'DXCM'],
        'XLP': ['PEP', 'COST', 'MDLZ', 'KDP', 'KHC', 'MNST', 'WBA']
    }
    
    LOG_PREFIX: str = "backtest_log_"
    DATA_DIR: str = ".quantbot_data"
    STATS_FILE: str = os.path.join(DATA_DIR, "strategy_stats.json")
    REPORT_FILE: str = os.path.join(DATA_DIR, "backtest_report.md")
    EXT_CACHE_FILE: str = os.path.join(DATA_DIR, "daily_ext_cache.json")
    ORDER_DB_PATH: str = os.path.join(DATA_DIR, "order_state.db") 
    TCA_LOG_PATH: str = os.path.join(DATA_DIR, "tca_history.jsonl")
    
    CACHE_FILE: str = "tickers_cache.json"
    ALERT_CACHE_FILE: str = "alert_history.json"
    MODEL_FILE: str = "scoring_model.pkl"
    WSB_CACHE_FILE: str = "wsb_history.json"  
    CUSTOM_CONFIG_FILE: str = "quantbot_config.json" 
    MODEL_VERSION: str = "3.0" 

    CORE_WATCHLIST: list = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "QCOM", "JPM", "V", "MA", "UNH", "JNJ", "XOM", "PG", "HD", "COST", "ABBV", "CRM", "NFLX", "ADBE", "NOW", "UBER", "INTU", "IBM", "ISRG", "SYK", "SPY", "QQQ", "IWM", "DIA"]
    CROWDING_EXCLUDE_SECTORS: list = [INDEX_ETF] 

    CORE_FACTORS = ["米奈尔维尼", "强相对强度", "MACD金叉", "TTM Squeeze ON", "一目多头", "强势回踩", "机构控盘(CMF)", "突破缺口", "VWAP突破", "AVWAP突破", "SMC失衡区", "流动性扫盘", "聪明钱抢筹", "巨量滞涨", "放量长阳", "口袋支点", "VCP收缩", "特性改变(ChoCh)", "订单块(OB)", "AMD操盘", "跨时空共振(周线)"]
    ADVANCED_MATH_FACTORS = ["量子概率云(KDE)", "稳健赫斯特(Hurst)", "FFT多窗共振(动能)", "CVD筹码净流入", "独立Alpha(脱钩)", "NR7极窄突破", "VPT量价共振", "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)", "Amihud非流动性(冲击成本)", "波动率风险溢价(VRP)"]
    INTERACTION_FACTORS = ["带量金叉(交互)", "量价吸筹(交互)", "近3日突破(滞后)", "近3日巨量(滞后)", "大周期保护小周期(MACD共振)", "60分钟级精准校准(RSI反弹)", "52周高点距离(动能延续)"]
    ALT_DATA_FACTORS = ["聪明钱月度净流入(月线)", "期权PutCall情绪(PCR)", "隐含波动率偏度(IV Skew)", "做空兴趣突变(轧空)", "内部人集群净买入(Insider)", "分析师修正动量(Analyst)", "舆情NLP情感极值(News_NLP)", "散户热度加速度(WSB_Accel)"]
    TRANSFORMER_FACTORS = [f"Alpha_T{i:02d}" for i in range(1, 17)]
    ALL_FACTORS = CORE_FACTORS + ADVANCED_MATH_FACTORS + INTERACTION_FACTORS + ALT_DATA_FACTORS + TRANSFORMER_FACTORS
    GROUP_A_FACTORS = CORE_FACTORS + ["聪明钱月度净流入(月线)", "大盘Beta(宏观调整)", "利率敏感度(TLT相关性)", "汇率传导(DXY相关性)", "52周高点距离(动能延续)", "内部人集群净买入(Insider)", "分析师修正动量(Analyst)"] + [f for f in INTERACTION_FACTORS if "共振" in f or "滞后" in f]
    GROUP_B_FACTORS = ADVANCED_MATH_FACTORS + ALT_DATA_FACTORS + [f for f in CORE_FACTORS if "扫盘" in f or "失衡" in f or "抢筹" in f] + TRANSFORMER_FACTORS

    class Params:
        MAX_WORKERS = 8
        MIN_SCORE_THRESHOLD = 8
        BASE_MAX_RISK = 0.015       
        CROWDING_PENALTY = 0.75     
        CROWDING_MIN_STOCKS = 2     
        PORTFOLIO_VALUE = 100000.0  
        SLIPPAGE = 0.003            
        COMMISSION = 0.0005         
        MIN_T_STAT = 1.0            
        PCR_BEAR = 1.5           
        PCR_BULL = 0.5           
        IV_SKEW_BEAR = 0.08      
        IV_SKEW_BULL = -0.05     
        SHORT_SQZ_CHG = 0.10     
        SHORT_SQZ_FLT = 0.05     
        VRP_EXTREME = 0.25       
        WSB_ACCEL = 20.0         
        ANALYST_UP = 1.5         
        ANALYST_DN = -1.5        
        NLP_BULL = 0.5           
        NLP_BEAR = -0.5          
        INSIDER_BUY = 0.5        
        AMIHUD_ILLIQ = 0.5       
        DIST_52W = -0.05         
        KDE_BREAKOUT = 0.5       
        HURST_RELIABLE = 0.65    
        FFT_RESONANCE = 0.71     
        VCP_BULL = 0.5           
        VCP_BEAR = 0.4           
        MACD_WATERLINE = 0.0     

        @classmethod
        def load_overrides(cls):
            if os.path.exists(Config.CUSTOM_CONFIG_FILE):
                try:
                    with open(Config.CUSTOM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                        custom_cfg = json.load(f)
                        for k, v in custom_cfg.items():
                            if hasattr(cls, k):
                                orig_val = getattr(cls, k)
                                if isinstance(orig_val, bool): setattr(cls, k, str(v).lower() in ['true', '1', 'yes', 't'])
                                elif isinstance(orig_val, (int, float, str)):
                                    try: setattr(cls, k, type(orig_val)(v))
                                    except: pass
                                else: setattr(cls, k, v)
                except Exception: pass

    @classmethod
    def get_current_log_file(cls) -> str:
        from datetime import datetime, timezone
        return f"{cls.LOG_PREFIX}{datetime.now(timezone.utc).strftime('%Y_%m')}.jsonl"

    @staticmethod
    def get_sector_etf(symbol: str) -> str:
        for etf, symbols in Config.SECTOR_MAP.items():
            if symbol in symbols: return etf
        return Config.INDEX_ETF
