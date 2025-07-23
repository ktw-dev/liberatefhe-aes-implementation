#!/usr/bin/env python3
"""compress_coeff.py
Compress a sparse complex coefficient matrix (e.g. 16×16 FFT/polynomial
coefficients) into a compact JSON representation that stores only the
non-zero entries while preserving their indices and (optionally)
recording detected Hermitian symmetry.

Usage:
  python3 compress_coeff.py fft_coeffs.npy            # writes fft_coeffs.json
  python3 compress_coeff.py fft_coeffs.npy -o out.json --tol 1e-9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def detect_hermitian(arr: np.ndarray, tol: float) -> bool:
    """Check F(-k,-l) = conj(F(k,l)) for all indices."""
    nrows, ncols = arr.shape
    for i in range(nrows):
        for j in range(ncols):
            if not np.allclose(arr[i, j], np.conj(arr[-i % nrows, -j % ncols]), atol=tol):
                return False
    return True


def compress(arr: np.ndarray, tol: float) -> dict[str, Any]:
    mask = np.abs(arr) > tol
    entries: list[list[float | int]] = []
    for i, j in zip(*np.nonzero(mask)):
        c = arr[i, j]
        entries.append([int(i), int(j), float(c.real), float(c.imag)])

    data: dict[str, Any] = {
        "shape": list(arr.shape),
        "tol": tol,
        "entries": entries,
    }
    if detect_hermitian(arr, tol):
        data["symmetry"] = "hermitian"
    return data


def decompress(data: dict[str, Any]) -> np.ndarray:
    shape = tuple(data["shape"])
    tol = data.get("tol", 0.0)
    arr = np.zeros(shape, dtype=np.complex128)
    for row, col, real, imag in data["entries"]:
        arr[row, col] = complex(real, imag)
    if data.get("symmetry") == "hermitian":
        nrows, ncols = shape
        for i in range(nrows):
            for j in range(ncols):
                if np.abs(arr[i, j]) > tol and np.abs(arr[-i % nrows, -j % ncols]) <= tol:
                    arr[-i % nrows, -j % ncols] = np.conj(arr[i, j])
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress coefficient matrix to JSON")
    parser.add_argument("npy", type=Path, help="Input .npy file path")
    parser.add_argument("-o", "--out", type=Path, help="Output JSON path")
    parser.add_argument("--tol", type=float, default=1e-12, help="Tolerance for zero")
    args = parser.parse_args()

    if not args.npy.exists():
        parser.error(f"File not found: {args.npy}")
    arr = np.load(args.npy)
    if arr.ndim != 2:
        parser.error("Expected 2-D array")

    data = compress(arr, args.tol)
    out_path = args.out if args.out else args.npy.with_suffix(".json")
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    kept = len(data["entries"])
    print(f"Compressed → {out_path} (kept {kept}/{arr.size} entries, tol={args.tol})")


if __name__ == "__main__":
    main() 