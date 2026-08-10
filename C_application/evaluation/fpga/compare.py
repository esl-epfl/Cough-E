import sys
import math

def compare_logs(file1, file2, tolerance=1e-2):
    with open(file1, 'r') as f:
        lines1 = [l.strip() for l in f if l.strip() and "CLK CYCLE" not in l and "POS2FLOAT" not in l]
    with open(file2, 'r') as f:
        lines2 = [l.strip() for l in f if l.strip() and "CLK CYCLE" not in l and "POS2FLOAT" not in l]

    if len(lines1) != len(lines2):
        print(f"Structure mismatch: {file1} has {len(lines1)} lines, {file2} has {len(lines2)} lines.")
        return

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatches = 0
    total_nums = 0
    diffs = []

    for idx, (l1, l2) in enumerate(zip(lines1, lines2)):
        tokens1 = l1.split()
        tokens2 = l2.split()

        if len(tokens1) != len(tokens2):
            print(f"Line {idx} has different token counts.")
            continue

        for t1, t2 in zip(tokens1, tokens2):
            v1_str = t1.split(':')[-1]
            v2_str = t2.split(':')[-1]

            try:
                v1 = float(v1_str)
                v2 = float(v2_str)
            except ValueError:
                if t1 != t2:
                    print(f"Line {idx} string mismatch: '{t1}' vs '{t2}'")
                continue

            total_nums += 1
            diff = abs(v1 - v2)
            diffs.append(diff)
            if diff > max_abs_diff:
                max_abs_diff = diff
            if v1 != 0:
                rel = diff / abs(v1)
                if rel > max_rel_diff:
                    max_rel_diff = rel
            if diff > tolerance:
                mismatches += 1
                print(f"Line {idx} drift exceeds tolerance: {t1} vs {t2} (diff: {diff:.6f})")

    mean_drift = 0.0
    std_drift = 0.0
    if total_nums > 0:
        mean_drift = sum(diffs) / total_nums
        variance = sum((d - mean_drift) ** 2 for d in diffs) / total_nums
        std_drift = math.sqrt(variance)

    print("\n--- Summary ---")
    print(f"Total elements compared: {total_nums}")
    print(f"Max Absolute Drift: {max_abs_diff:.6f}")
    print(f"Max Relative Drift: {max_rel_diff:.6f}")
    print(f"Mean Absolute Drift: {mean_drift:.6f}")
    print(f"Std Dev Absolute Drift: {std_drift:.6f}")
    print(f"Elements exceeding tolerance ({tolerance}): {mismatches}")

if __name__ == '__main__':
    compare_logs('desktop.log', 'fpga.log')
