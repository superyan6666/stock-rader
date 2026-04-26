import re
import os

with open('quant_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add optuna import
if 'import optuna' not in content:
    content = content.replace('import json', 'import json\nimport optuna')

# 2. Extract ledger loop
start_marker = "    # === 🚀 真实的逐日盯市事件驱动引擎 (Event-Driven Ledger) ==="
start_idx = content.find(start_marker)

end_marker = "    with open(Config.STATS_FILE, 'w', encoding='utf-8') as f: json.dump("
end_idx = content.find(end_marker)

ledger_code = content[start_idx:end_idx]

ledger_code = ledger_code.replace("    portfolio_metrics = {}", """    portfolio_metrics = {}
    params = params or {}
    tier2_dd = params.get('tier2_dd', -0.08)
    tier1_dd = params.get('tier1_dd', -0.05)
    tp_normal = params.get('tp_normal', 3.0)
    tp_tier1 = params.get('tp_tier1', 2.0)
    sl_mul = params.get('sl_mul', 1.2)
    max_hold_trend = params.get('max_hold_trend', 7)
    max_hold_rev = params.get('max_hold_rev', 3)
    pos_weight_normal = params.get('pos_weight_normal', 0.10)
    pos_weight_tier1 = params.get('pos_weight_tier1', 0.05)
    max_entries_normal = params.get('max_entries_normal', 5)
    max_entries_tier1 = params.get('max_entries_tier1', 2)""")

ledger_code = ledger_code.replace("if positions and (equity - dd_peak) / dd_peak < -0.08:", "if positions and (equity - dd_peak) / dd_peak < tier2_dd:")
ledger_code = ledger_code.replace("max_daily_entries = 5", "max_daily_entries = max_entries_normal")
ledger_code = ledger_code.replace("pos_weight = 0.10", "pos_weight = pos_weight_normal")
ledger_code = ledger_code.replace("if current_dd < -0.05:  # Tier 1: 谨慎模式", "if current_dd < tier1_dd:  # Tier 1: 谨慎模式")
ledger_code = ledger_code.replace("max_daily_entries = 2", "max_daily_entries = max_entries_tier1")
ledger_code = ledger_code.replace("pos_weight = 0.05", "pos_weight = pos_weight_tier1")
ledger_code = ledger_code.replace("tp_mul = 2.0 if pos_weight < 0.10 else 3.0", "tp_mul = tp_tier1 if pos_weight < pos_weight_normal else tp_normal")
ledger_code = ledger_code.replace("sl = entry_cost * (1 - 1.2 * atr_pct)", "sl = entry_cost * (1 - sl_mul * atr_pct)")
ledger_code = ledger_code.replace("max_hold = 3 if any(k in factor_str for k in reversal_kw) else 7", "max_hold = max_hold_rev if any(k in factor_str for k in reversal_kw) else max_hold_trend")
ledger_code = ledger_code.replace('print(f"\\n[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\\n{mc_report}\\n")', 'if not is_optuna:\n                print(f"\\n[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\\n{mc_report}\\n")')
ledger_code = ledger_code.replace('with open(os.path.join(Config.DATA_DIR, "fitness_score.txt"), "w", encoding=\'utf-8\') as f:\n                f.write(f"[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\\n{mc_report}")', 'if not is_optuna:\n                with open(os.path.join(Config.DATA_DIR, "fitness_score.txt"), "w", encoding=\'utf-8\') as f:\n                    f.write(f"[AGENT_FITNESS_SCORE]: {fitness_score:.4f}\\n{mc_report}")\n            return fitness_score, portfolio_metrics')

lines = ledger_code.split('\n')
dedented_lines = [line[4:] if line.startswith('    ') else line for line in lines]
ledger_func = "def simulate_ledger_run(valid_trades, all_dates, h_arr, l_arr, c_arr, o_arr, params=None, is_optuna=False):\n" + "\n".join("    " + line for line in dedented_lines) + "\n    return -9999.0, {}"

new_ledger_call = """    # === 🚀 真实的逐日盯市事件驱动引擎 (Event-Driven Ledger) ===
    all_dates = sorted(list(df_c.index))
    fitness_score, portfolio_metrics = simulate_ledger_run(valid_trades, all_dates, h_arr, l_arr, c_arr, o_arr, params=None, is_optuna=False)
"""

content = content[:start_idx] + new_ledger_call + content[end_idx:]

func_start = content.find("def run_backtest_engine():")
content = content[:func_start] + ledger_func + "\n\n" + content[func_start:]

optuna_code = """

def run_optuna_search():
    import pickle
    import glob
    logger.info("🚀 启动 Optuna 超参搜索！")
    
    cache_file = os.path.join(Config.DATA_DIR, "backtest_valid_trades_cache.pkl")
    if not os.path.exists(cache_file):
        logger.error("❌ 找不到 backtest_valid_trades_cache.pkl！请先运行一遍 python quant_engine.py backtest 收集信号数据。")
        return
        
    logger.info("📦 加载回测数据缓存...")
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
        
    valid_trades = cache_data['valid_trades']
    all_dates = cache_data['all_dates']
    h_arr = cache_data['h_arr']
    l_arr = cache_data['l_arr']
    c_arr = cache_data['c_arr']
    o_arr = cache_data['o_arr']
    
    def objective(trial):
        params = {
            'tier1_dd': trial.suggest_float('tier1_dd', -0.07, -0.03),
            'tier2_dd': trial.suggest_float('tier2_dd', -0.12, -0.06),
            'tp_normal': trial.suggest_float('tp_normal', 2.0, 5.0),
            'tp_tier1': trial.suggest_float('tp_tier1', 1.5, 3.0),
            'sl_mul': trial.suggest_float('sl_mul', 1.0, 2.0),
            'max_hold_trend': trial.suggest_int('max_hold_trend', 5, 15),
            'max_hold_rev': trial.suggest_int('max_hold_rev', 2, 5),
        }
        if params['tier2_dd'] >= params['tier1_dd']:
            raise optuna.TrialPruned()
            
        fitness, _ = simulate_ledger_run(valid_trades, all_dates, h_arr, l_arr, c_arr, o_arr, params=params, is_optuna=True)
        return fitness

    study = optuna.create_study(direction='maximize', study_name='quant_engine_hyperparams')
    logger.info("🔍 开始 Optuna Trial 搜索...")
    study.optimize(objective, n_trials=500, n_jobs=4)
    
    logger.info("🏆 Optuna 搜索完成！")
    logger.info(f"最高 Fitness Score: {study.best_value:.4f}")
    logger.info("最优参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
        
    with open(os.path.join(Config.DATA_DIR, "optuna_best_params.json"), "w", encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=4)
"""

if "def run_optuna_search():" not in content:
    content = content.replace('def run_historical_replay():', optuna_code + '\ndef run_historical_replay():')
    
content = content.replace('elif m == "stress": run_synthetic_stress_test()', 'elif m == "stress": run_synthetic_stress_test()\n        elif m == "optuna": run_optuna_search()')

cache_logic = """    # === 🚀 真实的逐日盯市事件驱动引擎 (Event-Driven Ledger) ===
    all_dates = sorted(list(df_c.index))
    
    import pickle
    cache_file = os.path.join(Config.DATA_DIR, "backtest_valid_trades_cache.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump({
            'valid_trades': valid_trades,
            'all_dates': all_dates,
            'h_arr': h_arr,
            'l_arr': l_arr,
            'c_arr': c_arr,
            'o_arr': o_arr
        }, f)
"""
content = content.replace("    # === 🚀 真实的逐日盯市事件驱动引擎 (Event-Driven Ledger) ===\n    all_dates = sorted(list(df_c.index))", cache_logic)


with open('quant_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
