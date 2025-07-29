#!/usr/bin/env python3
"""anal_coeff.py
Quick analysis utility for inspecting the 16×16 complex coefficient
matrix stored in ``fft_coeffs.npy``.

Usage::

    python3 anal_coeff.py [path/to/fft_coeffs.npy]

Outputs structural information (shape, dtype), sparsity, largest
coefficients, and verifies Hermitian symmetry that is typical for FFT
coefficients of real-valued inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def analyze(coeffs: np.ndarray, *, top_k: int = 10, tol: float = 1e-12) -> None:
    """Print various statistics and symmetry checks for *coeffs*."""

    print("— Basic information —")
    print(f"Shape: {coeffs.shape}, dtype: {coeffs.dtype}")

    # Sparsity.
    nonzero_mask = np.abs(coeffs) > tol
    nz = np.count_nonzero(nonzero_mask)
    total = coeffs.size
    print(f"Non-zero elements: {nz}/{total} ({nz / total * 100:.1f}%)")

    # Largest coefficients by magnitude.
    flat_idx = np.argsort(np.abs(coeffs), axis=None)[::-1]
    print(f"\n— Top {top_k} coefficients by magnitude —")
    for rank in range(min(top_k, flat_idx.size)):
        i, j = np.unravel_index(flat_idx[rank], coeffs.shape)
        val = coeffs[i, j]
        print(f"{rank + 1:2d}: |{val:.6g}| at (row {i}, col {j}) → {val}")

    # Hermitian symmetry check: F(-k) = conj(F(k))
    nrows, ncols = coeffs.shape
    hermitian_ok = True
    for i in range(nrows):
        for j in range(ncols):
            if not np.allclose(coeffs[i, j], np.conj(coeffs[-i % nrows, -j % ncols]), atol=tol):
                hermitian_ok = False
                break
        if not hermitian_ok:
            break
    print("\nHermitian symmetry about origin:", "Yes" if hermitian_ok else "No")

    # Row/column energy (ℓ¹ norm of magnitudes) gives a heat-map style summary.
    row_energy = np.sum(np.abs(coeffs), axis=1)
    col_energy = np.sum(np.abs(coeffs), axis=0)
    print("\n— Row energy (sum |coeff|) —")
    print(row_energy)
    print("\n— Column energy (sum |coeff|) —")
    print(col_energy)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fft_coeffs.npy")

    if not path.exists():
        sys.exit(f"File not found: {path}")

    coeffs = np.load(path)
    if coeffs.ndim != 2:
        sys.exit("Expected a 2-D array of coefficients")

    analyze(coeffs)


if __name__ == "__main__":
    main() 