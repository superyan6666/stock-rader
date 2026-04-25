import sys
content = open('quant_engine.py', 'r', encoding='utf-8').read()
content = content.replace('except Exception: pass', 'except Exception as e: logger.debug(f"Exception suppressed: {e}")')
open('quant_engine.py', 'w', encoding='utf-8').write(content)
print('Replaced exceptions')
