import sys

def main():
    with open('quant_engine.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if line.startswith('def _extract_complex_features'):
            start_idx = i
        if start_idx != -1 and line.startswith('def _io_fetch_worker'):
            # The previous lines are empty or comments.
            # _build_market_context ends a few lines before _io_fetch_worker
            # Let's search backwards from here
            for j in range(i-1, start_idx, -1):
                if '# ================= 🚀 两阶段并行引擎工作函数' in lines[j]:
                    end_idx = j
                    break
            break

    if start_idx != -1 and end_idx != -1:
        print(f"Deleting from {start_idx} to {end_idx}")
        del lines[start_idx:end_idx]
        with open('quant_engine.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Done!")
    else:
        print(f"Could not find bounds! Start: {start_idx}, End: {end_idx}")

if __name__ == '__main__':
    main()
