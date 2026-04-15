#!/usr/bin/env python3
"""
Compare KissFFT twiddle precision modes: Q1.15 vs Q1.31.

This script:
  1) regenerates fixed-point twiddle headers,
  2) compiles a small KissFFT harness for FIXED_POINT=16 and FIXED_POINT=32,
  3) runs deterministic test signals for NFFT in {900, 2048, 6400},
  4) reports deviation metrics (raw and gain-corrected).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class CaseMetrics:
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _kiss_dir() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def _ensure_twiddle_headers(kiss_dir: Path) -> None:
    gen = kiss_dir / "gen_twiddles_fixed.py"
    proc = _run(["python3", str(gen), "--formats", "q15", "q31"])
    if proc.returncode != 0:
        raise RuntimeError(f"Twiddle generation failed:\n{proc.stderr}")


def _build_harness(kiss_dir: Path, out_bin: Path, fixed_point: int) -> None:
    harness = kiss_dir.parent / "test" / "kissfft_twiddle_harness.c"
    cmd = [
        "gcc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        f"-DFIXED_POINT={fixed_point}",
        "-I",
        str(kiss_dir),
        "-o",
        str(out_bin),
        str(harness),
        str(kiss_dir / "kiss_fft.c"),
        str(kiss_dir / "kiss_fftr.c"),
        "-lm",
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Build failed ({fixed_point=}):\n{proc.stderr}")


def _write_signal(path: Path, x: Iterable[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in x:
            f.write(f"{v:.10f}\n")


def _parse_bins(stdout: str, expected_bins: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for line in stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 4 or parts[0] != "BIN":
            continue
        re = float(parts[2])
        im = float(parts[3])
        out.append((re, im))
    if len(out) != expected_bins:
        raise RuntimeError(f"Expected {expected_bins} bins, got {len(out)}")
    return out


def _run_harness(bin_path: Path, nfft: int, signal_path: Path) -> List[Tuple[float, float]]:
    proc = _run([str(bin_path), str(nfft), str(signal_path)])
    if proc.returncode != 0:
        raise RuntimeError(f"Harness run failed:\n{proc.stderr}")
    return _parse_bins(proc.stdout, nfft // 2 + 1)


def _make_signals(nfft: int) -> Dict[str, List[float]]:
    t = [float(i) for i in range(nfft)]
    amp = 0.85
    rng = random.Random(12345 + nfft)

    impulse = [0.0] * nfft
    impulse[0] = amp

    tone = [amp * math.sin(2.0 * math.pi * 7.0 * i / nfft) for i in t]
    dual = [
        0.6 * amp * math.sin(2.0 * math.pi * 5.0 * i / nfft)
        + 0.4 * amp * math.sin(2.0 * math.pi * 37.0 * i / nfft + 0.3)
        for i in t
    ]
    chirp = [
        amp
        * math.sin(
            2.0
            * math.pi
            * (3.0 + (120.0 - 3.0) * (i / nfft))
            * i
            / nfft
        )
        for i in t
    ]
    noise = [amp * (2.0 * rng.random() - 1.0) for _ in t]

    return {
        "impulse": impulse,
        "tone_bin7": tone,
        "dual_5_37": dual,
        "chirp": chirp,
        "noise": noise,
    }


def _rms(xs: Iterable[float]) -> float:
    vals = list(xs)
    if not vals:
        return 0.0
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def _compute_case_metrics(nfft: int, signal: str, q15: List[Tuple[float, float]], q31: List[Tuple[float, float]]) -> CaseMetrics:
    bins = len(q31)
    q15_re = [x[0] for x in q15]
    q15_im = [x[1] for x in q15]
    q31_re = [x[0] for x in q31]
    q31_im = [x[1] for x in q31]

    # Least-squares scalar gain fit: q31 ~= alpha * q15
    num = sum(a * b + c * d for a, c, b, d in zip(q15_re, q15_im, q31_re, q31_im))
    den = sum(a * a + c * c for a, c in zip(q15_re, q15_im))
    alpha = (num / den) if den > 0.0 else 1.0

    d_re = [a - b for a, b in zip(q15_re, q31_re)]
    d_im = [a - b for a, b in zip(q15_im, q31_im)]
    m15 = [math.hypot(a, b) for a, b in zip(q15_re, q15_im)]
    m31 = [math.hypot(a, b) for a, b in zip(q31_re, q31_im)]
    d_mag = [a - b for a, b in zip(m15, m31)]

    q15_re_gc = [alpha * v for v in q15_re]
    q15_im_gc = [alpha * v for v in q15_im]
    m15_gc = [math.hypot(a, b) for a, b in zip(q15_re_gc, q15_im_gc)]
    d_mag_gc = [a - b for a, b in zip(m15_gc, m31)]

    rmse_re = _rms(d_re)
    rmse_im = _rms(d_im)
    rmse_mag = _rms(d_mag)
    rmse_mag_gc = _rms(d_mag_gc)

    ref_rms_mag = _rms(m31)
    rel_rmse_mag_pct = 100.0 * (rmse_mag / ref_rms_mag) if ref_rms_mag > 0.0 else 0.0
    rel_rmse_mag_gc_pct = 100.0 * (rmse_mag_gc / ref_rms_mag) if ref_rms_mag > 0.0 else 0.0

    return CaseMetrics(
        nfft=nfft,
        signal=signal,
        bins=bins,
        alpha=alpha,
        rmse_re=rmse_re,
        rmse_im=rmse_im,
        rmse_mag=rmse_mag,
        rel_rmse_mag_pct=rel_rmse_mag_pct,
        max_abs_re=max(abs(v) for v in d_re) if d_re else 0.0,
        max_abs_im=max(abs(v) for v in d_im) if d_im else 0.0,
        max_abs_mag=max(abs(v) for v in d_mag) if d_mag else 0.0,
        rmse_mag_gc=rmse_mag_gc,
        rel_rmse_mag_gc_pct=rel_rmse_mag_gc_pct,
        max_abs_mag_gc=max(abs(v) for v in d_mag_gc) if d_mag_gc else 0.0,
    )


def _write_case_csv(path: Path, rows: List[CaseMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.nfft,
                    r.signal,
                    r.bins,
                    f"{r.alpha:.10f}",
                    f"{r.rmse_re:.10e}",
                    f"{r.rmse_im:.10e}",
                    f"{r.rmse_mag:.10e}",
                    f"{r.rel_rmse_mag_pct:.6f}",
                    f"{r.max_abs_re:.10e}",
                    f"{r.max_abs_im:.10e}",
                    f"{r.max_abs_mag:.10e}",
                    f"{r.rmse_mag_gc:.10e}",
                    f"{r.rel_rmse_mag_gc_pct:.6f}",
                    f"{r.max_abs_mag_gc:.10e}",
                ]
            )


def _aggregate(rows: List[CaseMetrics], nfft: int | None = None) -> Dict[str, float]:
    subset = [r for r in rows if nfft is None or r.nfft == nfft]
    n = sum(r.bins for r in subset)
    if n == 0:
        return {"bins": 0}

    # Weighted by bins per case.
    w = [r.bins for r in subset]
    sum_w = float(sum(w))

    def wavg(getter) -> float:
        return sum(getter(r) * r.bins for r in subset) / sum_w

    return {
        "bins": n,
        "rmse_re": wavg(lambda r: r.rmse_re),
        "rmse_im": wavg(lambda r: r.rmse_im),
        "rmse_mag": wavg(lambda r: r.rmse_mag),
        "rel_rmse_mag_pct": wavg(lambda r: r.rel_rmse_mag_pct),
        "max_abs_mag": max(r.max_abs_mag for r in subset),
        "rmse_mag_gc": wavg(lambda r: r.rmse_mag_gc),
        "rel_rmse_mag_gc_pct": wavg(lambda r: r.rel_rmse_mag_gc_pct),
        "max_abs_mag_gc": max(r.max_abs_mag_gc for r in subset),
    }


def _write_summary_csv(path: Path, rows: List[CaseMetrics], nffts: List[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )
        for nfft in nffts:
            a = _aggregate(rows, nfft=nfft)
            writer.writerow(
                [
                    f"nfft_{nfft}",
                    a["bins"],
                    f"{a['rmse_re']:.10e}",
                    f"{a['rmse_im']:.10e}",
                    f"{a['rmse_mag']:.10e}",
                    f"{a['rel_rmse_mag_pct']:.6f}",
                    f"{a['max_abs_mag']:.10e}",
                    f"{a['rmse_mag_gc']:.10e}",
                    f"{a['rel_rmse_mag_gc_pct']:.6f}",
                    f"{a['max_abs_mag_gc']:.10e}",
                ]
            )
        all_a = _aggregate(rows, nfft=None)
        writer.writerow(
            [
                "overall",
                all_a["bins"],
                f"{all_a['rmse_re']:.10e}",
                f"{all_a['rmse_im']:.10e}",
                f"{all_a['rmse_mag']:.10e}",
                f"{all_a['rel_rmse_mag_pct']:.6f}",
                f"{all_a['max_abs_mag']:.10e}",
                f"{all_a['rmse_mag_gc']:.10e}",
                f"{all_a['rel_rmse_mag_gc_pct']:.6f}",
                f"{all_a['max_abs_mag_gc']:.10e}",
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Study Q15 vs Q31 KissFFT twiddle precision")
    parser.add_argument(
        "--nffts",
        nargs="+",
        type=int,
        default=[900, 2048, 6400],
        help="FFT sizes to test (default: 900 2048 6400)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_kiss_dir(),
        help="Directory for CSV outputs (default: kiss_fftr directory)",
    )
    args = parser.parse_args()

    kiss_dir = _kiss_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_twiddle_headers(kiss_dir)

    rows: List[CaseMetrics] = []
    with tempfile.TemporaryDirectory(prefix="twiddle_study_") as td:
        tmp = Path(td)
        bin_q15 = tmp / "kissfft_q15"
        bin_q31 = tmp / "kissfft_q31"

        _build_harness(kiss_dir, bin_q15, fixed_point=16)
        _build_harness(kiss_dir, bin_q31, fixed_point=32)

        for nfft in args.nffts:
            signals = _make_signals(nfft)
            for signal_name, x in signals.items():
                sig_path = tmp / f"sig_{nfft}_{signal_name}.txt"
                _write_signal(sig_path, x)
                out_q15 = _run_harness(bin_q15, nfft, sig_path)
                out_q31 = _run_harness(bin_q31, nfft, sig_path)
                rows.append(_compute_case_metrics(nfft, signal_name, out_q15, out_q31))

    case_csv = args.output_dir / "twiddle_precision_cases.csv"
    summary_csv = args.output_dir / "twiddle_precision_summary.csv"
    _write_case_csv(case_csv, rows)
    _write_summary_csv(summary_csv, rows, args.nffts)

    print("Twiddle precision study complete (Q15 vs Q31)")
    print(f"  Case results   : {case_csv}")
    print(f"  Summary results: {summary_csv}")

    overall = _aggregate(rows)
    print(
        "  Overall: "
        f"RMSE_mag={overall['rmse_mag']:.6e}, "
        f"RelRMSE_mag={overall['rel_rmse_mag_pct']:.4f}%, "
        f"RMSE_mag(gain-corrected)={overall['rmse_mag_gc']:.6e}, "
        f"RelRMSE_mag(gain-corrected)={overall['rel_rmse_mag_gc_pct']:.4f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
