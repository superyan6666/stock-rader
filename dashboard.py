import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import altair as alt

# --- 🎯 页面全局配置 ---
st.set_page_config(
    page_title="Tickeron Pro AI | 全息指挥仓",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 📁 数据加载管道 ---
DATA_DIR = ".quantbot_data"
STATS_FILE = os.path.join(DATA_DIR, "strategy_stats.json")
TCA_FILE = os.path.join(DATA_DIR, "tca_history.jsonl")

@st.cache_data(ttl=60) # 缓存 60 秒，避免频繁读取硬盘
def load_data():
    stats_data = {}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                stats_data = json.load(f)
        except Exception as e:
            st.error(f"无法读取状态文件: {e}")

    tca_records = []
    if os.path.exists(TCA_FILE):
        try:
            with open(TCA_FILE, "r") as f:
                for line in f:
                    tca_records.append(json.loads(line.strip()))
        except Exception as e:
            st.error(f"无法读取交易历史记录: {e}")
            
    tca_df = pd.DataFrame(tca_records) if tca_records else pd.DataFrame()
    return stats_data, tca_df

# --- 🖥️ 核心渲染逻辑 ---
def main():
    st.title("🤖 Quant Engine 全息指挥仓")
    st.markdown("实时监控底座：甲骨文 4核 24G ARM 物理矩阵 | 算力中枢：GitHub Actions")
    st.divider()

    stats_data, tca_df = load_data()

    if not stats_data and tca_df.empty:
        st.warning("暂未在 .quantbot_data 目录中侦测到核心阵列数据，请先执行一次 Matrix 扫盘或等待周末 Backtest 完成。")
        return

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
        if 'timestamp' in tca_df.columns:
            tca_df['timestamp'] = pd.to_datetime(tca_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
        display_cols = [c for c in ['timestamp', 'client_oid', 'symbol', 'side', 'qty', 'arrival_price', 'execution_price', 'slippage_bps'] if c in tca_df.columns]
        
        st.dataframe(
            tca_df[display_cols].sort_index(ascending=False).head(20), 
            use_container_width=True,
            hide_index=True
        )
        
        if 'slippage_bps' in tca_df.columns:
            st.markdown("#### 实盘执行滑点分布 (Slippage BPS)")
            chart = alt.Chart(tca_df.dropna(subset=['slippage_bps'])).mark_bar().encode(
                x=alt.X('slippage_bps:Q', bin=alt.Bin(maxbins=30), title='滑点基点 (BPS)'),
                y=alt.Y('count()', title='笔数'),
                color=alt.Color('side:N', scale=alt.Scale(domain=['BUY', 'SELL'], range=['#00C853', '#D50000']))
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("尚未记录到任何网关交易执行截面数据 (TCA)。")

if __name__ == "__main__":
    main()
