import pandas as pd
import numpy as np

def check_macd_cross(curr: pd.Series, prev: pd.Series) -> bool:
    return prev['MACD'] < prev['Signal_Line'] and curr['MACD'] > curr['Signal_Line']

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # ✅ 修复：显式深拷贝隔离，切断对上游 xs() 视图的原地修改，防止底层数据污染
    df = df.copy()
    
    df = df.sort_index()
    df['Close'], df['Volume'] = df['Close'].ffill(), df['Volume'].ffill()
    df['Open'], df['High'], df['Low'] = df['Open'].ffill(), df['High'].ffill(), df['Low'].ffill()
    
    df['SMA_50'], df['SMA_150'], df['SMA_200'] = df['Close'].rolling(50).mean(), df['Close'].rolling(150).mean(), df['Close'].rolling(200).mean()
    df['EMA_20'], df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean(), df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    rs = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean() / (-delta.where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean() + 1e-10)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    atr20 = df['TR'].rolling(20).mean()
    
    df['KC_Upper'], df['KC_Lower'] = df['EMA_20'] + 1.5 * atr20, df['EMA_20'] - 1.5 * atr20
    bb_ma, bb_std = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['BB_Upper'], df['BB_Lower'] = bb_ma + 2 * bb_std, bb_ma - 2 * bb_std
    
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Above_Cloud'] = (df['Close'] > df[['SenkouA', 'SenkouB']].max(axis=1)).astype(int)
    df['SuperTrend_Up'] = (df['Close'] > df['EMA_20']).astype(int) 
    
    hl_diff = np.maximum(df['High'].values - df['Low'].values, 1e-10)
    dollar_vol = df['Close'].values * df['Volume'].values
    clv_num = (df['Close'].values - df['Low'].values) - (df['High'].values - df['Close'].values)
    df['CMF'] = pd.Series((clv_num / hl_diff * dollar_vol)).rolling(20).sum() / (pd.Series(dollar_vol).rolling(20).sum() + 1e-10)
    df['CMF'] = df['CMF'].clip(lower=-1.0, upper=1.0).fillna(0.0)

    df['Range'] = df['High'] - df['Low']
    df['NR7'] = (df['Range'] <= df['Range'].rolling(7).min())
    df['Inside_Bar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))

    close_shift = np.roll(df['Close'].values, 1); close_shift[0] = np.nan
    vpt_base = np.where((close_shift == 0) | np.isnan(close_shift), 0.0, (df['Close'].values - close_shift) / close_shift)
    df['VPT_Cum'] = (vpt_base * df['Volume'].values).cumsum()
    vpt_ma50, vpt_std50 = df['VPT_Cum'].rolling(50).mean(), df['VPT_Cum'].rolling(50).std()
    df['VPT_ZScore'] = (df['VPT_Cum'] - vpt_ma50) / np.where((np.isnan(vpt_std50)) | (vpt_std50 == 0), 1e-6, vpt_std50)
    df['VPT_Accel'] = np.gradient(np.nan_to_num(df['VPT_ZScore']))

    df['VWAP_20'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)
    df['AVWAP'] = df['VWAP_20'] 
    df['Max_Down_Vol_10'] = df['Volume'].where(df['Close'] < df['Close'].shift(), 0).shift(1).rolling(10).max()
    
    surge = (df['Close'] > df['Close'].shift(1) * 1.04) & (df['Volume'] > df['Volume'].rolling(20).mean())
    df['OB_High'] = df['High'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['OB_Low'] = df['Low'].shift(1).where(surge, np.nan).ffill(limit=20)
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Range_60'] = df['High'].rolling(60).max() - df['Low'].rolling(60).min()
    df['Range_20'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Price_High_20'] = df['High'].rolling(20).max()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    df['CVD_Smooth'] = ((df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)).cumsum()).ewm(span=5).mean()
    df['CVD_Trend'] = np.where(df['CVD_Smooth'].rolling(10).mean() > df['CVD_Smooth'].rolling(30).mean(), 1.0, -1.0)
    df['CVD_Divergence'] = ((df['Close'] >= df['Price_High_20'] * 0.99) & (df['CVD_Smooth'] < df['CVD_Smooth'].rolling(20).max() * 0.95)).astype(int)
    
    df['Highest_22'] = df['High'].rolling(22).max()
    df['ATR_22'] = df['TR'].rolling(22).mean()
    df['Chandelier_Exit'] = df['Highest_22'] - 2.5 * df['ATR_22']
    
    df['Smart_Money_Flow'] = pd.Series(clv_num / hl_diff, index=df.index).rolling(10).mean()

    df['Recent_Price_Surge_3d'] = (df['Close'] / df['Open'] - 1).rolling(3).max().shift(1) * 100
    df['Recent_Vol_Surge_3d'] = (df['Volume'] / (df['Vol_MA20']+1)).rolling(3).max().shift(1)
    df['Amihud'] = (df['Close'].pct_change().abs() / (df['Close'] * df['Volume'] + 1e-10)).rolling(20).mean() * 1e6
    high_52w = df['High'].rolling(252, min_periods=20).max()
    df['Dist_52W_High'] = (df['Close'] - high_52w) / (high_52w + 1e-10)

    return df
