import re

with open('quant_engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find run_backtest_engine start
rbe_start = -1
for i, line in enumerate(lines):
    if line.startswith('def run_backtest_engine():'):
        rbe_start = i
        break

# Find simulate_ledger_run start
slr_start = -1
for i, line in enumerate(lines):
    if line.startswith('def simulate_ledger_run('):
        slr_start = i
        break

# Find the end of simulate_ledger_run body
# The body ends at line 2347 where it writes to fitness_score.txt
# The next line 2348 starts with "    with open(Config.STATS_FILE"
stats_file_idx = -1
for i in range(slr_start, len(lines)):
    if '    with open(Config.STATS_FILE' in lines[i]:
        stats_file_idx = i
        break

# Extracted simulate_ledger_run
slr_lines = lines[slr_start:stats_file_idx]
# Make sure it returns fitness_score, portfolio_metrics at the end
if 'return fitness_score, portfolio_metrics' not in ''.join(slr_lines[-5:]):
    slr_lines.append('    return fitness_score, portfolio_metrics\n')

# Reconstruct run_backtest_engine
# It should contain everything from rbe_start up to slr_start (excluding the run_backtest_engine_inner call)
rbe_first_part = lines[rbe_start:slr_start]
# Remove the last line if it's the run_backtest_engine_inner call
for i in range(len(rbe_first_part)-1, -1, -1):
    if 'run_backtest_engine_inner' in rbe_first_part[i]:
        rbe_first_part.pop(i)
        break

cache_code = """    all_dates = sorted(list(df_c.index))
    import pickle
    cache_file = os.path.join(Config.DATA_DIR, "backtest_valid_trades_cache.pkl")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump({
                'valid_trades': valid_trades,
                'all_dates': all_dates,
                'h_arr': h_arr,
                'l_arr': l_arr,
                'c_arr': c_arr,
                'o_arr': o_arr
            }, f)
    except Exception as e:
        pass
        
    fitness_score, portfolio_metrics = simulate_ledger_run(valid_trades, all_dates, h_arr, l_arr, c_arr, o_arr, params=None, is_optuna=False)
"""

rbe_second_part = lines[stats_file_idx:]
# We need to find where run_backtest_engine ends. It ends when a new function starts with 0 spaces.
next_func_idx = len(rbe_second_part)
for i, line in enumerate(rbe_second_part):
    if line.startswith('def '):
        next_func_idx = i
        break

rbe_reporting = rbe_second_part[:next_func_idx]
rest_of_file = rbe_second_part[next_func_idx:]

# Now reconstruct the whole file
new_lines = lines[:rbe_start] + slr_lines + ['\n'] + rbe_first_part + [cache_code] + rbe_reporting + rest_of_file

with open('quant_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Surgery successful.")
