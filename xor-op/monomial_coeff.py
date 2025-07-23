"""Generate monomial coefficients C[p,q] such that
    P(x,y) = Σ C[p,q] x^p y^q  satisfies  P(ζ^a, ζ^b) = ζ^(a xor b)
for all a,b in 0..15,  where ζ = exp(-2πi/16).

We exploit the fact that this is exactly a 2-D inverse DFT of the table
    f[a,b] = ζ^(a xor b).

Hence  C = ifft2(f).
The script writes the sparse coefficients to `xor_mono_coeffs.json` (fft_coeffs
compatible schema) and the full matrix to `xor_mono_coeffs.npy`.
Run:
    python xor-op/monomial_coeff.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# 1. compute coefficients via 2-D inverse FFT
# -----------------------------------------------------------------------------


def compute_xor_mono_coeffs_zeta(M: int = 16) -> np.ndarray:
    """Return complex matrix C[p,q] (shape M×M)."""
    zeta = np.exp(-2j * np.pi / M)
    # f[a,b] = ζ^(a xor b)
    f = np.fromfunction(lambda a, b: zeta ** (a.astype(int) ^ b.astype(int)), (M, M), dtype=int)
    C = np.fft.ifft2(f)
    return C


# -----------------------------------------------------------------------------
# 2. verification helper
# -----------------------------------------------------------------------------


def verify(C: np.ndarray, trials: int = 50, tol: float = 1e-8, M: int = 16) -> bool:
    zeta = np.exp(-2j * np.pi / M)
    for _ in range(trials):
        a, b = np.random.randint(0, M, size=2)
        x = zeta ** a
        y = zeta ** b
        # Evaluate polynomial at (x,y)
        val = (C * np.outer(x ** np.arange(M), y ** np.arange(M))).sum()
        if abs(val - zeta ** (a ^ b)) > tol:
            print(f"[FAIL] a={a}, b={b}, got {val}, expected {zeta ** (a ^ b)}")
            return False
    print(f"[PASS] verification succeeded for {trials} random points")
    return True


# -----------------------------------------------------------------------------
# 3. JSON compression helper
# -----------------------------------------------------------------------------


def compress_coefficients(C: np.ndarray, tol: float = 1e-12) -> List[Tuple[int, int, float, float]]:
    M, N = C.shape
    entries: List[Tuple[int, int, float, float]] = []
    for p in range(M):
        for q in range(N):
            if abs(C[p, q]) > tol:
                entries.append((p, q, float(C[p, q].real), float(C[p, q].imag)))
    return entries


# -----------------------------------------------------------------------------
# 4. main entry
# -----------------------------------------------------------------------------


def main() -> None:
    M = 16
    print("[INFO] Computing coefficients via 2-D ifft…")
    C = compute_xor_mono_coeffs_zeta(M)

    verify(C)

    entries = compress_coefficients(C)
    print(f"[INFO] Non-zero coeffs: {len(entries)}/{M*M} ({len(entries)/(M*M):.2%})")

    obj = {
        "shape": [M, M],
        "tol": 1e-12,
        "entries": entries,
        "note": "monomial coefficients for XOR in zeta domain via ifft2"
    }

    (THIS_DIR / "xor_mono_coeffs.json").write_text(json.dumps(obj, indent=2))
    np.save(THIS_DIR / "xor_mono_coeffs.npy", C)
    print("[INFO] Saved xor_mono_coeffs.json / .npy")

    # print small sample
    print("[INFO] Sample coefficients (p,q<4):")
    for p in range(4):
        for q in range(4):
            val = C[p, q]
            print(f"  c[{p},{q}] = {val.real:+.6f}{val.imag:+.6f}j")


if __name__ == "__main__":
    main()
