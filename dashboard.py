import streamlit as st
import pandas as pd
import json
import os
import random
from datetime import datetime, timedelta
import altair as alt

# --- 🎯 页面全局配置 ---
st.set_page_config(
    page_title="Tickeron Pro AI | 全息指挥仓",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 📁 数据加载与模拟注入管道 ---
DATA_DIR = ".quantbot_data"
STATS_FILE = os.path.join(DATA_DIR, "strategy_stats.json")
TCA_FILE = os.path.join(DATA_DIR, "tca_history.jsonl")

def inject_mock_data_if_empty():
    """如果侦测到无实盘数据，自动注入 50 笔逼真的模拟数据以点亮大屏"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(STATS_FILE):
        mock_stats = {
            "model_version": "v1.2_Transformer_Alpha",
            "overall": {
                "T+3": {"win_rate": 0.685, "profit_factor": 1.82, "ai_win_rate": 0.741}
            },
            "last_updated": datetime.now().isoformat()
        }
        with open(STATS_FILE, "w") as f:
            json.dump(mock_stats, f)
            
    if not os.path.exists(TCA_FILE):
        symbols = ['NVDA', 'TSLA', 'AMZN', 'AAPL', 'MSFT', 'AMD', 'META']
        with open(TCA_FILE, "w") as f:
            for i in range(50):
                # 生成过去 10 天内的随机时间
                rand_time = datetime.now() - timedelta(days=random.randint(0, 10), hours=random.randint(0, 23))
                record = {
                    "timestamp": rand_time.isoformat(),
                    "client_oid": f"MOCK_{rand_time.strftime('%m%d%H%M')}_{i}",
                    "symbol": random.choice(symbols),
                    "side": random.choice(["BUY", "SELL"]),
                    "qty": random.randint(10, 100) * 10,
                    "arrival_price": round(random.uniform(100, 500), 2),
                    "execution_price": round(random.uniform(100, 500), 2),
                    "slippage_bps": round(random.gauss(1.5, 2.0), 1), # 模拟正态分布滑点
                    "ai_prob": round(random.uniform(0.55, 0.95), 2)   # 模拟高置信度
                }
                f.write(json.dumps(record) + "\n")

@st.cache_data(ttl=10) # 缩短缓存时间，提升刷新灵敏度
def load_data():
    inject_mock_data_if_empty() # 执行注入检查
    
    stats_data = {}
    with open(STATS_FILE, "r") as f:
        stats_data = json.load(f)

    tca_records = []
    with open(TCA_FILE, "r") as f:
        for line in f:
            tca_records.append(json.loads(line.strip()))
            
    tca_df = pd.DataFrame(tca_records)
    return stats_data, tca_df

# --- 🖥️ 核心渲染逻辑 ---
def main():
    st.title("🤖 Quant Engine 全息指挥仓")
    st.markdown("实时监控底座：甲骨文 4核 24G ARM 物理矩阵 | 算力中枢：GitHub Actions")
    st.divider()

    stats_data, tca_df = load_data()

    # --- 模块一：核心代谢指标 (Metrics) ---
    st.subheader("💡 引擎代谢指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="当前右脑版本 (Version)", value=stats_data.get("model_version", "v0_baseline"))
    with col2:
        t3_stats = stats_data.get("overall", {}).get("T+3", {})
        st.metric(label="T+3 原始胜率 (Win Rate)", value=f"{t3_stats.get('win_rate', 0.0):.2%}")
    with col3:
        st.metric(label="T+3 盈亏比 (Profit Factor)", value=f"{t3_stats.get('profit_factor', 0.0):.2f}")
    with col4:
        st.metric(label="AI 元学习胜率过滤", value=f"{t3_stats.get('ai_win_rate', 0.0):.2%}" if 'ai_win_rate' in t3_stats else "N/A")

    st.divider()

    # --- 模块二：近期交易截面 (TCA Tracker) ---
    st.subheader("📊 近期交易截面追踪 (TCA 滑点监控)")
    
    if not tca_df.empty:
        tca_df['timestamp'] = pd.to_datetime(tca_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_cols = [c for c in ['timestamp', 'client_oid', 'symbol', 'side', 'qty', 'arrival_price', 'execution_price', 'slippage_bps', 'ai_prob'] if c in tca_df.columns]
        
        st.dataframe(
            tca_df[display_cols].sort_values(by='timestamp', ascending=False).head(20), 
            use_container_width=True,
            hide_index=True
        )
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 实盘执行滑点分布 (Slippage BPS)")
            chart1 = alt.Chart(tca_df.dropna(subset=['slippage_bps'])).mark_bar().encode(
                x=alt.X('slippage_bps:Q', bin=alt.Bin(maxbins=20), title='滑点基点 (BPS)'),
                y=alt.Y('count()', title='笔数'),
                color=alt.Color('side:N', scale=alt.Scale(domain=['BUY', 'SELL'], range=['#00C853', '#D50000']))
            ).properties(height=300)
            st.altair_chart(chart1, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### AI 置信度胜率分布 (Confidence)")
            chart2 = alt.Chart(tca_df.dropna(subset=['ai_prob'])).mark_area(opacity=0.6).encode(
                x=alt.X('ai_prob:Q', bin=alt.Bin(maxbins=20), title='AI 预判胜率'),
                y=alt.Y('count()', title='信号数量', stack=None),
                color=alt.value('#2962FF')
            ).properties(height=300)
            st.altair_chart(chart2, use_container_width=True)

if __name__ == "__main__":
    main()
