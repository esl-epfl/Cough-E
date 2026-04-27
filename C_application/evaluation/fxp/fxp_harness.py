#!/usr/bin/env python3
"""Run FxP-vs-float error metrics for the fixed-point pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
STAGE_BIN = THIS_DIR / "fxp_stage_harness"
TWIDDLE_BITS = 32


def run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout


def parse_kv(line: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for part in line.split(",")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            row[key] = value
    return row


def print_error(row: dict[str, str]) -> None:
    print(
        "FXP_ERROR,"
        f"block={row.get('block', '')},"
        f"kernel={row.get('kernel', row.get('stage', ''))},"
        f"stage={row.get('stage', '')},"
        f"qformat={row.get('qformat', '')},"
        f"n={row.get('n', '0')},"
        f"rmse_pct={row.get('rel_rmse_pct', '0')},"
        f"max_abs_pct={row.get('max_abs_pct', '0')}"
    )


def main() -> int:
    run([
        "make",
        "-B",
        "-C",
        str(THIS_DIR),
        "fxp_stage_harness",
        f"FFT_MODE=-DFIXED_POINT={TWIDDLE_BITS}",
    ])

    for line in run([str(STAGE_BIN)]).splitlines():
        if not line.startswith("FXP_STAGE,"):
            continue
        row = parse_kv(line)
        if row.get("mode") == "single-kernel":
            print_error(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
