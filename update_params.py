import os
import re

with open('quant_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("tier2_dd = params.get('tier2_dd', -0.08)", "tier2_dd = params.get('tier2_dd', -0.0722)")
content = content.replace("tier1_dd = params.get('tier1_dd', -0.05)", "tier1_dd = params.get('tier1_dd', -0.0385)")
content = content.replace("tp_normal = params.get('tp_normal', 3.0)", "tp_normal = params.get('tp_normal', 3.44)")
content = content.replace("tp_tier1 = params.get('tp_tier1', 2.0)", "tp_tier1 = params.get('tp_tier1', 2.75)")
content = content.replace("sl_mul = params.get('sl_mul', 1.2)", "sl_mul = params.get('sl_mul', 1.43)")
content = content.replace("max_hold_trend = params.get('max_hold_trend', 7)", "max_hold_trend = params.get('max_hold_trend', 15)")
content = content.replace("max_hold_rev = params.get('max_hold_rev', 3)", "max_hold_rev = params.get('max_hold_rev', 4)")

with open('quant_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully!")
