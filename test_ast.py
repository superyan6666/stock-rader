import ast
import traceback

try:
    with open('quant_engine.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("OK")
except Exception as e:
    with open('err.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
    print("Error written to err.txt")
