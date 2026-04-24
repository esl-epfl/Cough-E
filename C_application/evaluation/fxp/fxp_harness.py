#!/usr/bin/env python3
"""Unified FxP validation and error-metric harness.

This is the single Python owner for FxP error testing. The C harness files in
this directory are narrow numeric probes; all CLI modes, build/run decisions,
parsing, aggregation, baselines, and evaluator handoff live here.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
C_APP_DIR = THIS_DIR.parents[1]
EVAL_DIR = C_APP_DIR / "evaluation"
KISS_DIR = C_APP_DIR / "kiss_fftr"
STAGE_BIN = THIS_DIR / "fxp_stage_harness"
BASELINE_DEFAULT = THIS_DIR / "baselines" / "fxp_stage_baseline.json"
KISSFFT_SIGNALS = ["impulse", "tone_bin7", "dual_5_37", "chirp", "noise"]


@dataclass
class KissfftCaseMetrics:
    nfft: int
    signal: str
    bins: int
    alpha: float
    rmse_re: float
    rmse_im: float
    rmse_mag: float
    rel_rmse_mag_pct: float
    max_abs_re: float
    max_abs_im: float
    max_abs_mag: float
    rmse_mag_gc: float
    rel_rmse_mag_gc_pct: float
    max_abs_mag_gc: float


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def build_stage(twiddle: int) -> None:
    lock_path = THIS_DIR / ".fxp_stage_harness.build.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(["make", "-B", "-C", str(THIS_DIR), "fxp_stage_harness", f"FFT_MODE=-DFIXED_POINT={twiddle}"])


def backend_stage_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    mapping = [
        ("audio_features_backend", "--audio-features-backend"),
        ("audio_model_backend", "--audio-model-backend"),
        ("imu_features_backend", "--imu-features-backend"),
        ("imu_model_backend", "--imu-model-backend"),
    ]
    for attr, flag in mapping:
        value = getattr(args, attr, None)
        if value:
            out += [flag, value]
    return out


def run_stage(args: argparse.Namespace, sweep: bool = False, trace_limit: int = 0) -> list[str]:
    build_stage(args.twiddle)
    cmd = [str(STAGE_BIN), "--max-windows", str(args.max_windows)]
    if sweep:
        cmd.append("--sweep")
    if trace_limit:
        cmd += ["--trace-limit", str(trace_limit)]
    cmd += backend_stage_args(args)
    result = run(cmd)
    return result.stdout.splitlines()


def print_filtered(lines: list[str], prefixes: tuple[str, ...], contains: str | None = None) -> None:
    for line in lines:
        if not line.startswith(prefixes):
            continue
        if contains and contains not in line:
            continue
        print(line)


def ensure_kissfft_twiddles() -> None:
    generator = KISS_DIR / "gen_twiddles_fixed.py"
    if generator.exists():
        run([sys.executable, str(generator), "--formats", "q15", "q31"])


def parse_kissfft_bins(stdout: str, expected_bins: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for line in stdout.splitlines():
        if not line.startswith("KISSFFT_BIN,"):
            continue
        row = parse_kv_line(line)
        out.append((float(row["re"]), float(row["im"])))
    if len(out) != expected_bins:
        raise RuntimeError(f"Expected {expected_bins} KissFFT bins, got {len(out)}")
    return out


def collect_kissfft_bins(twiddle: int, nffts: list[int], signals: list[str]) -> dict[tuple[int, str], list[tuple[float, float]]]:
    build_stage(twiddle)
    bins: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for nfft in nffts:
        for signal in signals:
            result = run([
                str(STAGE_BIN),
                "--kissfft-bins",
                "--nfft",
                str(nfft),
                "--signal",
                signal,
            ])
            bins[(nfft, signal)] = parse_kissfft_bins(result.stdout, nfft // 2 + 1)
    return bins


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def compute_kissfft_case(nfft: int,
                         signal: str,
                         q15: list[tuple[float, float]],
                         q31: list[tuple[float, float]]) -> KissfftCaseMetrics:
    q15_re = [x[0] for x in q15]
    q15_im = [x[1] for x in q15]
    q31_re = [x[0] for x in q31]
    q31_im = [x[1] for x in q31]

    numerator = sum(a * b + c * d for a, c, b, d in zip(q15_re, q15_im, q31_re, q31_im))
    denominator = sum(a * a + c * c for a, c in zip(q15_re, q15_im))
    alpha = numerator / denominator if denominator > 0.0 else 1.0

    delta_re = [a - b for a, b in zip(q15_re, q31_re)]
    delta_im = [a - b for a, b in zip(q15_im, q31_im)]
    mag15 = [math.hypot(a, b) for a, b in zip(q15_re, q15_im)]
    mag31 = [math.hypot(a, b) for a, b in zip(q31_re, q31_im)]
    delta_mag = [a - b for a, b in zip(mag15, mag31)]

    mag15_gc = [math.hypot(alpha * a, alpha * b) for a, b in zip(q15_re, q15_im)]
    delta_mag_gc = [a - b for a, b in zip(mag15_gc, mag31)]

    rmse_mag = rms(delta_mag)
    rmse_mag_gc = rms(delta_mag_gc)
    ref_rms_mag = rms(mag31)

    return KissfftCaseMetrics(
        nfft=nfft,
        signal=signal,
        bins=len(q31),
        alpha=alpha,
        rmse_re=rms(delta_re),
        rmse_im=rms(delta_im),
        rmse_mag=rmse_mag,
        rel_rmse_mag_pct=100.0 * rmse_mag / ref_rms_mag if ref_rms_mag > 0.0 else 0.0,
        max_abs_re=max((abs(v) for v in delta_re), default=0.0),
        max_abs_im=max((abs(v) for v in delta_im), default=0.0),
        max_abs_mag=max((abs(v) for v in delta_mag), default=0.0),
        rmse_mag_gc=rmse_mag_gc,
        rel_rmse_mag_gc_pct=100.0 * rmse_mag_gc / ref_rms_mag if ref_rms_mag > 0.0 else 0.0,
        max_abs_mag_gc=max((abs(v) for v in delta_mag_gc), default=0.0),
    )


def aggregate_kissfft(rows: list[KissfftCaseMetrics], nfft: int | None = None) -> dict[str, float]:
    subset = [row for row in rows if nfft is None or row.nfft == nfft]
    total_bins = sum(row.bins for row in subset)
    if total_bins == 0:
        return {"bins": 0.0}

    def weighted(getter) -> float:
        return sum(getter(row) * row.bins for row in subset) / total_bins

    return {
        "bins": float(total_bins),
        "rmse_re": weighted(lambda row: row.rmse_re),
        "rmse_im": weighted(lambda row: row.rmse_im),
        "rmse_mag": weighted(lambda row: row.rmse_mag),
        "rel_rmse_mag_pct": weighted(lambda row: row.rel_rmse_mag_pct),
        "max_abs_mag": max(row.max_abs_mag for row in subset),
        "rmse_mag_gc": weighted(lambda row: row.rmse_mag_gc),
        "rel_rmse_mag_gc_pct": weighted(lambda row: row.rel_rmse_mag_gc_pct),
        "max_abs_mag_gc": max(row.max_abs_mag_gc for row in subset),
    }


def print_kissfft_rows(rows: list[KissfftCaseMetrics], nffts: list[int]) -> None:
    for row in rows:
        print(
            "KISSFFT_CASE,"
            f"nfft={row.nfft},signal={row.signal},bins={row.bins},"
            f"alpha_gain_fit={row.alpha:.10g},"
            f"rmse_mag={row.rmse_mag:.10g},rel_rmse_mag_pct={row.rel_rmse_mag_pct:.9g},"
            f"max_abs_mag={row.max_abs_mag:.10g},"
            f"rmse_mag_gain_corrected={row.rmse_mag_gc:.10g},"
            f"rel_rmse_mag_gain_corrected_pct={row.rel_rmse_mag_gc_pct:.9g},"
            f"max_abs_mag_gain_corrected={row.max_abs_mag_gc:.10g}"
        )

    for nfft in [*nffts, None]:
        agg = aggregate_kissfft(rows, nfft=nfft)
        scope = f"nfft_{nfft}" if nfft is not None else "overall"
        print(
            "KISSFFT_SUMMARY,"
            f"scope={scope},bins={int(agg['bins'])},"
            f"rmse_mag={agg['rmse_mag']:.10g},rel_rmse_mag_pct={agg['rel_rmse_mag_pct']:.9g},"
            f"max_abs_mag={agg['max_abs_mag']:.10g},"
            f"rmse_mag_gain_corrected={agg['rmse_mag_gc']:.10g},"
            f"rel_rmse_mag_gain_corrected_pct={agg['rel_rmse_mag_gc_pct']:.9g},"
            f"max_abs_mag_gain_corrected={agg['max_abs_mag_gc']:.10g}"
        )


def write_kissfft_csv(output_dir: Path, rows: list[KissfftCaseMetrics], nffts: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_csv = output_dir / "twiddle_precision_cases.csv"
    summary_csv = output_dir / "twiddle_precision_summary.csv"

    with case_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "nfft",
            "signal",
            "bins",
            "alpha_gain_fit",
            "rmse_re",
            "rmse_im",
            "rmse_mag",
            "rel_rmse_mag_pct",
            "max_abs_re",
            "max_abs_im",
            "max_abs_mag",
            "rmse_mag_gain_corrected",
            "rel_rmse_mag_gain_corrected_pct",
            "max_abs_mag_gain_corrected",
        ])
        for row in rows:
            writer.writerow([
                row.nfft,
                row.signal,
                row.bins,
                f"{row.alpha:.10f}",
                f"{row.rmse_re:.10e}",
                f"{row.rmse_im:.10e}",
                f"{row.rmse_mag:.10e}",
                f"{row.rel_rmse_mag_pct:.6f}",
                f"{row.max_abs_re:.10e}",
                f"{row.max_abs_im:.10e}",
                f"{row.max_abs_mag:.10e}",
                f"{row.rmse_mag_gc:.10e}",
                f"{row.rel_rmse_mag_gc_pct:.6f}",
                f"{row.max_abs_mag_gc:.10e}",
            ])

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope",
            "bins",
            "rmse_re",
            "rmse_im",
            "rmse_mag",
            "rel_rmse_mag_pct",
            "max_abs_mag",
            "rmse_mag_gain_corrected",
            "rel_rmse_mag_gain_corrected_pct",
            "max_abs_mag_gain_corrected",
        ])
        for nfft in [*nffts, None]:
            agg = aggregate_kissfft(rows, nfft=nfft)
            writer.writerow([
                f"nfft_{nfft}" if nfft is not None else "overall",
                int(agg["bins"]),
                f"{agg['rmse_re']:.10e}",
                f"{agg['rmse_im']:.10e}",
                f"{agg['rmse_mag']:.10e}",
                f"{agg['rel_rmse_mag_pct']:.6f}",
                f"{agg['max_abs_mag']:.10e}",
                f"{agg['rmse_mag_gc']:.10e}",
                f"{agg['rel_rmse_mag_gc_pct']:.6f}",
                f"{agg['max_abs_mag_gc']:.10e}",
            ])

    print(f"KISSFFT_CSV,cases={case_csv},summary={summary_csv}")


def parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = line.split(",")
    out["_kind"] = parts[0]
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def stage_rows(lines: list[str]) -> list[dict[str, str]]:
    return [parse_kv_line(line) for line in lines if line.startswith("FXP_STAGE,")]


def row_id(row: dict[str, str]) -> str:
    return "|".join(row.get(k, "") for k in ("mode", "block", "stage", "qformat"))


def cmd_single_kernel(args: argparse.Namespace) -> int:
    lines = run_stage(args)
    for line in lines:
        if not line.startswith("FXP_STAGE,") or "mode=single-kernel" not in line:
            continue
        row = parse_kv_line(line)
        if args.block and row.get("block") != args.block:
            continue
        if args.kernel and args.kernel not in row.get("kernel", row.get("stage", "")):
            continue
        print(line)
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    lines = run_stage(args)
    for line in lines:
        if not line.startswith("FXP_STAGE,"):
            continue
        row = parse_kv_line(line)
        if row.get("mode") not in ("conversion", "block", "intermediate", "end_to_end", "hybrid", "hybrid-bridge"):
            continue
        if args.block and row.get("block") != args.block:
            continue
        print(line)
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    lines = run_stage(args, trace_limit=args.trace_limit)
    print_filtered(lines, ("FXP_TRACE", "FXP_STAGE"))
    return 0


def cmd_q_sensitivity(args: argparse.Namespace) -> int:
    lines = run_stage(args, sweep=True)
    print_filtered(lines, ("FXP_QSWEEP,",))
    return 0


def cmd_stage_all(args: argparse.Namespace) -> int:
    lines = run_stage(args, sweep=args.sweep, trace_limit=args.trace_limit)
    print_filtered(lines, ("FXP_STAGE,", "FXP_QSWEEP,", "FXP_TRACE,", "FXP_HYBRID_CONFIG,"))
    return 0


def cmd_descriptors(args: argparse.Namespace) -> int:
    build_stage(args.twiddle)
    result = run([str(STAGE_BIN), "--list-descriptors"])
    print(result.stdout, end="")
    return 0


def cmd_hybrid(args: argparse.Namespace) -> int:
    lines = run_stage(args, trace_limit=args.trace_limit)
    for line in lines:
        if line.startswith("FXP_HYBRID_CONFIG,") or line.startswith("FXP_TRACE,"):
            print(line)
            continue
        if not line.startswith("FXP_STAGE,"):
            continue
        row = parse_kv_line(line)
        if row.get("mode") in ("hybrid", "hybrid-bridge", "conversion", "intermediate", "block", "end_to_end"):
            print(line)
    return 0


def cmd_kissfft(args: argparse.Namespace) -> int:
    invalid = sorted(set(args.signals) - set(KISSFFT_SIGNALS))
    if invalid:
        print(f"Unknown KissFFT signal(s): {', '.join(invalid)}")
        print(f"Allowed signals: {', '.join(KISSFFT_SIGNALS)}")
        return 2

    ensure_kissfft_twiddles()
    q15 = collect_kissfft_bins(16, args.nffts, args.signals)
    q31 = collect_kissfft_bins(32, args.nffts, args.signals)

    rows = [
        compute_kissfft_case(nfft, signal, q15[(nfft, signal)], q31[(nfft, signal)])
        for nfft in args.nffts
        for signal in args.signals
    ]

    print_kissfft_rows(rows, args.nffts)
    if args.write_csv or args.output_dir:
        write_kissfft_csv(args.output_dir or THIS_DIR, rows, args.nffts)
    return 0


def cmd_e2e(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(EVAL_DIR / "evaluate.py"),
        "--compare",
        "--twiddle",
        str(args.twiddle),
        "--no-save",
    ]
    if args.skip_transform:
        cmd.append("--skip-transform")
    if args.subjects:
        cmd += ["--subjects", *args.subjects]
    if args.sounds:
        cmd += ["--sounds", *args.sounds]
    if args.noises:
        cmd += ["--noises", *args.noises]
    result = run(cmd, cwd=C_APP_DIR, check=False)
    print(result.stdout, end="")
    return result.returncode

def cmd_regression(args: argparse.Namespace) -> int:
    lines = run_stage(args, sweep=True)
    rows = stage_rows(lines)

    if args.write_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        data = {row_id(row): row for row in rows}
        args.baseline.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"Wrote baseline: {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"No baseline found at {args.baseline}. Current stage rows:")
        print_filtered(lines, ("FXP_STAGE,", "FXP_QSWEEP,"))
        return 0

    baseline = json.loads(args.baseline.read_text())
    failures: list[str] = []
    current = {row_id(row): row for row in rows}

    for key, old in baseline.items():
        new = current.get(key)
        if not new:
            failures.append(f"missing current row: {key}")
            continue
        for metric in ("rel_rmse_pct", "wape_pct", "max_abs_pct"):
            if metric not in old or metric not in new:
                continue
            old_v = float(old[metric])
            new_v = float(new[metric])
            allowed = max(args.abs_tolerance_pct, old_v * (1.0 + args.rel_tolerance))
            if new_v > allowed:
                failures.append(f"{key} {metric}: {new_v:.6g} > {allowed:.6g} (baseline {old_v:.6g})")

    if failures:
        print("Regression failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"Regression passed against {args.baseline}", flush=True)
    return 0


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def audit_slice(path: Path, start: str | None = None, end: str | None = None) -> str:
    text = path.read_text()
    if start:
        idx = text.find(start)
        if idx >= 0:
            text = text[idx:]
    if end:
        idx = text.find(end)
        if idx >= 0:
            text = text[:idx]
    return strip_c_comments(text)


def cmd_audit(args: argparse.Namespace) -> int:
    banned = re.compile(r"\b(float|double|sqrtf|logf|expf|powf|sinf|cosf|floorf|ceilf|roundf|kissfft_bridge_spectrum_to_float)\b")
    checks = [
        (C_APP_DIR / "Src" / "feature_extraction_fxp.c", "#if defined(FXP_MODE) && defined(FIXED_POINT)", None),
        (C_APP_DIR / "FxP" / "audio" / "audio_pipeline_fxp.c", "#if defined(FXP_MODE) && defined(FIXED_POINT)", None),
        (C_APP_DIR / "FxP" / "imu" / "imu_pipeline.c", "#ifdef FXP_MODE", "/* -------------------------------------------------------------------------- */\n/*  Dispatch"),
        (C_APP_DIR / "Src" / "audio_model.c", "#ifdef FXP_MODE", "#else"),
        (C_APP_DIR / "Src" / "imu_model.c", "#ifdef FXP_MODE", "#else"),
        (C_APP_DIR / "Src" / "postprocessing.c", "#ifdef FXP_MODE", "#else"),
    ]

    allowed = {
        "feature_extraction_fxp.c": (
            "audio_features_fxp_q16(const int8_t *features_selector, const float *sig",
            "imu_features_fxp_q16(const int8_t *features_selector, const float sig",
        )
    }

    failures: list[str] = []
    for path, start, end in checks:
        text = audit_slice(path, start, end)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not banned.search(line):
                continue
            if "float-only" in line:
                continue
            if any(token in line for token in allowed.get(path.name, ())):
                continue
            failures.append(f"{path.relative_to(C_APP_DIR)}:{lineno}: {line.strip()}")

    kiss_text = audit_slice(C_APP_DIR / "kiss_fftr" / "kiss_fft.c", "void kf_factor", "kiss_fft_cfg kiss_fft_alloc")
    fixed_block = re.search(r"#ifdef FIXED_POINT(.*?)#else", kiss_text, flags=re.S)
    if fixed_block and banned.search(fixed_block.group(1)):
        failures.append("kiss_fftr/kiss_fft.c fixed-point factorization contains banned float token")

    if failures:
        print("FxP float fallback audit failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("FxP float fallback audit passed for runtime fixed-point slices.")
    print("Allowed float boundary: source .h data to fixed carriers only.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FxP percentage-oriented validation harness")
    parser.add_argument("--twiddle", type=int, choices=[16, 32], default=16)
    parser.add_argument("--max-windows", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="Static audit for float fallbacks in fixed runtime slices")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("descriptors", help="List current block/kernel/stage descriptors")
    p.set_defaults(func=cmd_descriptors)

    p = sub.add_parser("single-kernel", help="Per selected kernel percentage error rows")
    p.add_argument("--block", choices=["audio", "imu"])
    p.add_argument("--kernel", help="Substring filter on kernel/stage name")
    p.set_defaults(func=cmd_single_kernel)

    p = sub.add_parser("block", help="Conversion and processing-block percentage error rows")
    p.add_argument("--block", choices=["audio", "imu", "kissfft", "postprocessing"])
    p.set_defaults(func=cmd_block)

    p = sub.add_parser("hybrid", help="Run runtime hybrid float/FxP feature/model combinations")
    p.add_argument("--audio-features-backend", choices=["float", "fxp"], default="fxp")
    p.add_argument("--audio-model-backend", choices=["float", "fxp"], default="fxp")
    p.add_argument("--imu-features-backend", choices=["float", "fxp"], default="fxp")
    p.add_argument("--imu-model-backend", choices=["float", "fxp"], default="fxp")
    p.add_argument("--trace-limit", type=int, default=0)
    p.set_defaults(func=cmd_hybrid)

    p = sub.add_parser("trace", help="Stage-by-stage trace rows for early windows")
    p.add_argument("--trace-limit", type=int, default=1)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("q-sensitivity", help="Q-format sensitivity sweeps at marked locations")
    p.set_defaults(func=cmd_q_sensitivity)

    p = sub.add_parser("stage-all", help="All stage rows, optional trace and Q sweep")
    p.add_argument("--trace-limit", type=int, default=0)
    p.add_argument("--sweep", action="store_true")
    p.set_defaults(func=cmd_stage_all)

    p = sub.add_parser("kissfft", help="Q15-vs-Q31 KissFFT precision study")
    p.add_argument("--nffts", nargs="+", type=int, default=[900, 2048, 6400])
    p.add_argument("--signals", nargs="+", default=KISSFFT_SIGNALS)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--write-csv", action="store_true")
    p.set_defaults(func=cmd_kissfft)

    p = sub.add_parser("end-to-end", help="Run the production evaluator in float-vs-FxP compare mode")
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--sounds", nargs="+")
    p.add_argument("--noises", nargs="+")
    p.add_argument("--skip-transform", action="store_true", default=False)
    p.set_defaults(func=cmd_e2e)

    p = sub.add_parser("regression", help="Compare stage metrics against an accepted baseline")
    p.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT)
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--rel-tolerance", type=float, default=0.05)
    p.add_argument("--abs-tolerance-pct", type=float, default=0.05)
    p.set_defaults(func=cmd_regression)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
