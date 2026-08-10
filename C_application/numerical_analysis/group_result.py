#!/usr/bin/env python3
"""
Group per-feature numerical accuracy results (output of compare_numerical_accuracy.py)
into per-family summaries: float vs upos32nq, and float vs upos16q, independently.

A "family" = feature name with trailing numeric index stripped, e.g.
MFCC_MEAN_0..63 -> MFCC_MEAN.

Produces two files:
  <output>.csv  - plain CSV
  <output>.tex  - LaTeX tabular, ready to \input{} into a report
"""
import argparse
import csv
import re
from collections import defaultdict


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def family_of(feature_label):
    name = feature_label.split("] ", 1)[1] if "] " in feature_label else feature_label
    name = re.sub(r"_\d+$", "", name)
    kind = "AUDIO" if feature_label.startswith("AUDIO") else "IMU"
    return f"{kind}_{name}"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_rows(input_path):
    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[family_of(r["feature"])].append(r)

    out_rows = []
    for fam, grp in groups.items():
        n = len(grp)
        avg_magnitude = mean(to_float(r["avg_float_value"]) for r in grp)
        # mre32 = mean(to_float(r["MRE_upos32nq"]) for r in grp if r["MRE_upos32nq"] != "N/A")
        mae32 = mean(to_float(r["MAE_upos32nq"]) for r in grp)
        nmae32 = mae32 / avg_magnitude
        # mre16 = mean(to_float(r["MRE_upos16q"]) for r in grp if r["MRE_upos16q"] != "N/A")
        mae16 = mean(to_float(r["MAE_upos16q"]) for r in grp)
        nmae16 = mae16 / avg_magnitude
        out_rows.append({
            "family": fam,
            "n_features": n,
            "avg_magnitude": avg_magnitude,
            "NMAE_upos32nq": nmae32,
            "MAE_upos32nq": mae32,
            "NMAE_upos16q": nmae16,
            "MAE_upos16q": mae16,
        })

    out_rows.sort(key=lambda r: r["family"])
    return out_rows


def fmt(v, decimals=10):
    return "N/A" if v is None else f"{v:.{decimals}f}"


def write_csv(rows, path):
    fieldnames = ["family", "n_features", "avg_magnitude",
                  "NMAE_upos32nq", "MAE_upos32nq", "NMAE_upos16q", "MAE_upos16q"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in rows:
            w.writerow([r["family"], r["n_features"], fmt(r["avg_magnitude"]),
                        fmt(r["NMAE_upos32nq"]), fmt(r["MAE_upos32nq"]),
                        fmt(r["NMAE_upos16q"]), fmt(r["MAE_upos16q"])])


def escape_tex(s):
    return s.replace("_", r"\_")


def write_tex(rows, path):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Feature Family} & \\textbf{N} & \\textbf{Magnitude} & \\textbf{NMAE (posit32)} & \\textbf{MAE (posit32)} & \\textbf{NMAE (posit16+q)} & \\textbf{MAE (posit16+q)} \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(f"\\texttt{{{escape_tex(r['family'])}}} & {r['n_features']} & "
                    f"{fmt(r['avg_magnitude'])} & "
                    f"{fmt(r['NMAE_upos32nq'])} & {fmt(r['MAE_upos32nq'])} & "
                    f"{fmt(r['NMAE_upos16q'])} & {fmt(r['MAE_upos16q'])} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="numerical_accuracy.csv")
    ap.add_argument("--output", default="numerical_accuracy_by_family",
                    help="output basename (produces <output>.csv and <output>.tex)")
    args = ap.parse_args()

    rows = build_rows(args.input)
    write_csv(rows, args.output + ".csv")
    write_tex(rows, args.output + ".tex")
    print(f"Wrote {len(rows)} families to {args.output}.csv and {args.output}.tex")


if __name__ == "__main__":
    main()
