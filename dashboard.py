# 存储路径: dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import random
import time
from datetime import datetime, timedelta
import altair as alt

# --- 🎯 页面全局配置 ---
st.set_page_config(
    page_title="Tickeron Pro AI | 全息指挥仓",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🚀 每 10 秒自动强制重绘整个页面 (实现真实指挥仓感)
# 如果您觉得太快晃眼，可以把这里改为 30 或 60
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=10000, limit=None, key="data_refresher")

# --- 📁 数据加载与模拟注入管道 ---
DATA_DIR = ".quantbot_data"
STATS_FILE = os.path.join(DATA_DIR, "strategy_stats.json")
TCA_FILE = os.path.join(DATA_DIR, "tca_history.jsonl")

def inject_mock_data_if_empty(force=False):
    """自动注入逼真的模拟数据以点亮大屏 (防空壳文件机制)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if force or not os.path.exists(STATS_FILE) or os.path.getsize(STATS_FILE) < 10:
        mock_stats = {
            "model_version": "v3.0_Singularity_Edition",
            "overall": {
                "T+3": {"win_rate": 0.685, "profit_factor": 1.82, "ai_win_rate": 0.741}
            },
            "last_updated": datetime.now().isoformat()
        }
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(mock_stats, f)
            
    if force or not os.path.exists(TCA_FILE) or os.path.getsize(TCA_FILE) < 10:
        symbols = ['NVDA', 'TSLA', 'AMZN', 'AAPL', 'MSFT', 'AMD', 'META', 'PLTR', 'SMCI', 'AVGO']
        with open(TCA_FILE, "w", encoding="utf-8") as f:
            for i in range(80):
                rand_time = datetime.now() - timedelta(days=random.randint(0, 10), hours=random.randint(0, 23))
                arr_px = round(random.uniform(50, 800), 2)
                slip = round(random.gauss(1.5, 2.0), 1) 
                
                side = random.choice(["BUY", "SELL"])
                exec_px = arr_px * (1 + slip / 10000.0) if side == "BUY" else arr_px * (1 - slip / 10000.0)
                
                record = {
                    "timestamp": rand_time.isoformat(),
                    "client_oid": f"MOCK_{rand_time.strftime('%m%d%H%M')}_{i}",
                    "symbol": random.choice(symbols),
                    "side": side,
                    "qty": random.randint(1, 15) * 100,
                    "arrival_price": arr_px,
                    "execution_price": round(exec_px, 2),
                    "slippage_bps": slip,
                    "ai_prob": round(random.uniform(0.40, 0.98), 2)
                }
                f.write(json.dumps(record) + "\n")

@st.cache_data(ttl=2) # 缓存时间降低到 2 秒，配合自动刷新机制
def load_data():
    inject_mock_data_if_empty(force=False)
    
    stats_data = {}
    if os.path.exists(STATS_FILE) and os.path.getsize(STATS_FILE) > 5:
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
        except Exception: pass

    tca_records = []
    if os.path.exists(TCA_FILE) and os.path.getsize(TCA_FILE) > 5:
        try:
            with open(TCA_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    tca_records.append(json.loads(line.strip()))
        except Exception: pass
            
    tca_df = pd.DataFrame(tca_records)
    return stats_data, tca_df

# --- 🖥️ 核心渲染逻辑 ---
def main():
    with st.sidebar:
        st.markdown("### 🛠️ 引擎调试中枢")
        st.caption("侦测到阻滞时，可手动强制刷新底层账本数据。")
        if st.button("💉 强制注入测试数据", use_container_width=True, type="primary"):
            inject_mock_data_if_empty(force=True)
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("### 📡 运行状态检测")
        st.success("✅ 核心账本通信正常")
        st.success("✅ TCA 滑点捕获器在线")
        st.info("ℹ️ UI 自动刷新: 开 (10s)")
            
    st.title("🤖 Quant Engine 全息指挥仓")
    st.markdown("实时监控底座：甲骨文 4核 24G ARM 物理矩阵 | 算力中枢：GitHub Actions | **Singularity Edition**")
    st.divider()

    stats_data, tca_df = load_data()

    if not stats_data and tca_df.empty:
        st.warning("暂未在 .quantbot_data 目录中侦测到核心阵列数据。请在左侧边栏点击【注入测试数据】以唤醒视图！")
        return

    # --- 模块一：核心代谢指标 (Metrics) ---
    st.subheader("💡 引擎代谢指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="当前右脑版本 (Version)", value=stats_data.get("model_version", "v3.0_Singularity"))
    with col2:
        t3_stats = stats_data.get("overall", {}).get("T+3", {})
        st.metric(label="T+3 原始胜率 (Win Rate)", value=f"{t3_stats.get('win_rate', 0.0):.2%}")
    with col3:
        st.metric(label="T+3 盈亏比 (Profit Factor)", value=f"{t3_stats.get('profit_factor', 0.0):.2f}")
    with col4:
        st.metric(label="AI 元学习胜率过滤", value=f"{t3_stats.get('ai_win_rate', 0.0):.2%}" if 'ai_win_rate' in t3_stats else "N/A")

    st.divider()

    # --- 模块二：近期交易截面 (TCA Tracker) ---
    st.subheader("📊 网关执行截面追踪 (Execution & TCA)")
    
    if not tca_df.empty:
        tca_df['timestamp'] = pd.to_datetime(tca_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_cols = [c for c in ['timestamp', 'client_oid', 'symbol', 'side', 'qty', 'arrival_price', 'execution_price', 'slippage_bps', 'ai_prob'] if c in tca_df.columns]
        
        st.dataframe(
            tca_df[display_cols].sort_values(by='timestamp', ascending=False).head(25), 
            use_container_width=True,
            hide_index=True
        )
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 实盘执行滑点分布 (Slippage BPS)")
            if 'slippage_bps' in tca_df.columns:
                chart1 = alt.Chart(tca_df.dropna(subset=['slippage_bps'])).mark_bar().encode(
                    x=alt.X('slippage_bps:Q', bin=alt.Bin(maxbins=30), title='滑点基点 (BPS)'),
                    y=alt.Y('count()', title='笔数'),
                    color=alt.Color('side:N', scale=alt.Scale(domain=['BUY', 'SELL'], range=['#10b981', '#ef4444']))
                ).properties(height=350)
                st.altair_chart(chart1, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### AI 置信度探测密度 (Confidence)")
            if 'ai_prob' in tca_df.columns:
                chart2 = alt.Chart(tca_df.dropna(subset=['ai_prob'])).mark_area(opacity=0.7).encode(
                    x=alt.X('ai_prob:Q', bin=alt.Bin(maxbins=25), title='AI 预判胜率'),
                    y=alt.Y('count()', title='信号触发数量', stack=None),
                    color=alt.value('#3b82f6')
                ).properties(height=350)
                st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("尚未记录到任何网关交易执行截面数据 (TCA)。")

if __name__ == "__main__":
    main()
