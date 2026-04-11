import json
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from collections import defaultdict

# ================= 1. 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Backtester")

# ================= 2. 核心回测类 =================
class LightweightBacktester:
    def __init__(self, log_file="backtest_log.jsonl"):
        self.log_file = log_file
        self.logs = []
        self.tickers = set()
        self.market_data = None
        self.results = []

    def load_logs(self):
        """读取结构化 JSONL 日志文件"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    self.logs.append(record)
                    for pick in record.get('top_picks', []):
                        self.tickers.add(pick['symbol'])
            
            logger.info(f"✅ 成功加载回测日志: 共 {len(self.logs)} 个交易日，涉及 {len(self.tickers)} 只唯一标的。")
            return True
        except FileNotFoundError:
            logger.error(f"❌ 未找到日志文件 {self.log_file}，请先让主程序运行一段时间积累数据。")
            return False
        except Exception as e:
            logger.error(f"❌ 读取日志失败: {e}")
            return False

    def fetch_market_data(self):
        """一次性批量拉取所有相关标的的历史行情（极速、防限流）"""
        if not self.tickers:
            return False
            
        logger.info(">>> 正在批量拉取历史收盘价以供回测，请稍候...")
        # 批量下载，只取 'Close' 收盘价，避免冗余数据
        df = yf.download(list(self.tickers), period="1y", interval="1d", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            self.market_data = df['Close']
        else:
            # 如果只有一个 ticker，yf.download 返回的不是 MultiIndex
            self.market_data = pd.DataFrame(df['Close'], columns=list(self.tickers))
            
        # 将索引转换为纯字符串日期（如 '2024-05-10'），方便与日志比对
        self.market_data.index = self.market_data.index.strftime('%Y-%m-%d')
        logger.info("✅ 历史数据拉取完成！")
        return True

    def run_simulation(self, horizons=[1, 3, 5]):
        """运行模拟：计算 T+1, T+3, T+5 的真实收益率"""
        logger.info(">>> 正在进行时间序列拟合与收益计算...")
        
        dates_available = list(self.market_data.index)
        
        for record in self.logs:
            rec_date = record['date']
            regime = record.get('regime', 'unknown')
            health = record.get('health_score', 0.0)
            
            # 找到记录日期在数据表中的位置
            if rec_date not in dates_available:
                # 可能是非交易日（如周末测试），往后顺延找到最近的交易日
                future_dates = [d for d in dates_available if d > rec_date]
                if not future_dates: continue
                base_idx = dates_available.index(future_dates[0])
            else:
                base_idx = dates_available.index(rec_date)
            
            for pick in record.get('top_picks', []):
                symbol = pick['symbol']
                if symbol not in self.market_data.columns: continue
                
                # 获取 T+0 (推荐当日) 的基准价格
                base_price = self.market_data[symbol].iloc[base_idx]
                if pd.isna(base_price): continue
                
                trade_result = {
                    "date": rec_date,
                    "symbol": symbol,
                    "score": pick['score'],
                    "regime": regime,
                    "is_crowded": pick.get('crowded', False),
                    "is_weak_sector": pick.get('weak_sector', False),
                    "returns": {}
                }
                
                # 计算各个持有期的收益
                for t in horizons:
                    target_idx = base_idx + t
                    if target_idx < len(dates_available):
                        future_price = self.market_data[symbol].iloc[target_idx]
                        if not pd.isna(future_price):
                            pct_change = (future_price - base_price) / base_price
                            trade_result["returns"][f"T+{t}"] = pct_change
                            
                if trade_result["returns"]:
                    self.results.append(trade_result)

    def generate_report(self):
        """生成并打印高度可视化的回测简报"""
        if not self.results:
            logger.warning("没有足够的数据生成回测报告。")
            return
            
        df_res = pd.DataFrame(self.results)
        
        # 展开 returns 字典为独立的列
        ret_df = pd.json_normalize(df_res['returns'])
        df_res = pd.concat([df_res.drop(columns=['returns']), ret_df], axis=1)
        
        print("\n" + "="*50)
        print(" 🎯 QuantBot 历史信号回测简报 (T+1/3/5)")
        print("="*50)
        
        print(f"总信号数: {len(df_res)}")
        print(f"涉及标的: {df_res['symbol'].nunique()} 只")
        
        # 1. 总体胜率与平均收益
        print("\n📊 [总体表现]")
        for col in [c for c in df_res.columns if c.startswith('T+')]:
            valid_data = df_res[col].dropna()
            if len(valid_data) == 0: continue
            win_rate = (valid_data > 0).mean()
            avg_ret = valid_data.mean()
            print(f"  {col} -> 胜率: {win_rate:.1%} | 平均收益: {avg_ret:+.2%}")
            
        # 2. 按大盘环境拆解 (验证牛熊市是否不同)
        print("\n🌍 [大盘状态 (Regime) 拆解 - 以 T+5 为例]")
        if 'T+5' in df_res.columns:
            for regime in df_res['regime'].unique():
                sub_df = df_res[df_res['regime'] == regime]['T+5'].dropna()
                if len(sub_df) > 0:
                    win_rate = (sub_df > 0).mean()
                    avg_ret = sub_df.mean()
                    print(f"  {regime.upper():<7} -> 信号数: {len(sub_df):<3} | 胜率: {win_rate:.1%} | 均收益: {avg_ret:+.2%}")

        # 3. 验证板块拥挤/弱势过滤器的价值
        print("\n🛡️ [过滤器有效性验证 - 以 T+3 为例]")
        if 'T+3' in df_res.columns:
            # 正常高分标的 vs 被系统判定为“板块拥挤/弱势”而降权的标的
            normal_df = df_res[~df_res['is_crowded'] & ~df_res['is_weak_sector']]['T+3'].dropna()
            penalized_df = df_res[df_res['is_crowded'] | df_res['is_weak_sector']]['T+3'].dropna()
            
            if len(normal_df) > 0:
                print(f"  ✅ 正常优选龙头 -> 胜率: {(normal_df > 0).mean():.1%} | 均收益: {normal_df.mean():+.2%}")
            if len(penalized_df) > 0:
                print(f"  ⚠️ 被降权跟风股 -> 胜率: {(penalized_df > 0).mean():.1%} | 均收益: {penalized_df.mean():+.2%}")

        print("="*50 + "\n")

# ================= 3. 执行入口 =================
if __name__ == "__main__":
    backtester = LightweightBacktester()
    if backtester.load_logs():
        if backtester.fetch_market_data():
            backtester.run_simulation(horizons=[1, 3, 5])
            backtester.generate_report()
