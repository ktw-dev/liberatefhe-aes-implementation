#!/usr/bin/env python3
"""sbox_coefficient.py

Generate 16×16 monomial coefficient matrices C_hi[p,q] and C_lo[p,q] such that

    P_hi(ζ^a, ζ^b) = hi(S_BOX[a<<4 | b])        (integer 0–15)
    P_lo(ζ^a, ζ^b) = lo(S_BOX[a<<4 | b])        (integer 0–15)

for all 4-bit inputs a,b ∈ {0,…,15}, where ζ = exp(−2πi/16).

The matrices are obtained via 2-D inverse DFT (NumPy ``ifft2``).  The resulting
complex coefficients are saved to ``coeffs/sbox_coeffs.json`` using the same
schema as existing coefficient tables in this repository:

```
{
  "sbox_upper_mv_coeffs_real": [[...16 floats...], ...],
  "sbox_upper_mv_coeffs_imag": [[...16 floats...], ...],
  "sbox_lower_mv_coeffs_real": [[...16 floats...], ...],
  "sbox_lower_mv_coeffs_imag": [[...16 floats...], ...],
  "shape": [16, 16],
  "tol": 1e-12,
  "note": "monomial coefficients for AES S-Box upper/lower nibbles via ifft2"
}
```

A quick Monte-Carlo verification (100 random inputs) checks that the generated
polynomials evaluate to the correct nibble values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Constants and paths
# ----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
COEFF_DIR = THIS_DIR / "coeffs"
COEFF_DIR.mkdir(exist_ok=True)
OUT_PATH = COEFF_DIR / "sbox_coeffs.json"

ZETA: complex = np.exp(-2j * np.pi / 16)
TOL = 1e-12

# ----------------------------------------------------------------------------
# Import reference S-Box (ground truth)
# ----------------------------------------------------------------------------

from aes_128_numpy import S_BOX  # noqa: E402  (late import for clarity)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def _build_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Return two (16,16) arrays of integer nibble values.

    f_hi[a,b] == upper-nibble of S_BOX[a<<4 | b]
    f_lo[a,b] == lower-nibble of S_BOX[a<<4 | b]
    """
    f_hi = np.empty((16, 16), dtype=np.float64)
    f_lo = np.empty((16, 16), dtype=np.float64)

    for a in range(16):
        for b in range(16):
            x = (a << 4) | b
            y = int(S_BOX[x])
            f_hi[a, b] = (y >> 4) & 0xF
            f_lo[a, b] = y & 0xF
    return f_hi, f_lo


def _ifft2_coeff(table: np.ndarray) -> np.ndarray:
    """Compute 2-D inverse FFT producing coefficient matrix."""
    return np.fft.ifft2(table)


def _save_coefficients(C_hi: np.ndarray, C_lo: np.ndarray) -> None:
    """Serialize coefficient matrices into JSON file."""
    obj = {
        "sbox_upper_mv_coeffs_real": C_hi.real.tolist(),
        "sbox_upper_mv_coeffs_imag": C_hi.imag.tolist(),
        "sbox_lower_mv_coeffs_real": C_lo.real.tolist(),
        "sbox_lower_mv_coeffs_imag": C_lo.imag.tolist(),
        "shape": [16, 16],
        "tol": TOL,
        "note": "monomial coefficients for AES S-Box upper/lower nibbles via ifft2",
    }
    OUT_PATH.write_text(json.dumps(obj, indent=2))
    print(f"[INFO] Saved coefficient file → {OUT_PATH.relative_to(THIS_DIR)}")


def _evaluate_polynomial(C: np.ndarray, a: int, b: int) -> complex:
    """Evaluate polynomial with coefficient matrix *C* at (ζ^a, ζ^b)."""
    # Pre-compute powers of ζ^a and ζ^b for efficiency
    x_powers = ZETA ** (a * np.arange(16))
    y_powers = ZETA ** (b * np.arange(16))
    return np.sum(C * np.outer(x_powers, y_powers))


def _verify(C_hi: np.ndarray, C_lo: np.ndarray, trials: int = 100) -> bool:
    """Monte-Carlo verification of the generated coefficients."""
    rng = np.random.default_rng()
    for _ in range(trials):
        a = int(rng.integers(0, 16))
        b = int(rng.integers(0, 16))

        # Ground-truth values
        x = (a << 4) | b
        y = int(S_BOX[x])
        y_hi = (y >> 4) & 0xF
        y_lo = y & 0xF

        # Polynomial evaluations
        p_hi = _evaluate_polynomial(C_hi, a, b)
        p_lo = _evaluate_polynomial(C_lo, a, b)

        if not (abs(p_hi - y_hi) < 1e-6 and abs(p_lo - y_lo) < 1e-6):
            print(
                f"[FAIL] a={a}, b={b}: expected (hi,lo)=({y_hi},{y_lo}), "
                f"got ({p_hi:.6f}, {p_lo:.6f})"
            )
            return False
    print(f"[PASS] Verification succeeded for {trials} random points.")
    return True


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------


def main() -> None:
    print("[INFO] Building S-Box nibble tables…")
    f_hi, f_lo = _build_tables()

    print("[INFO] Computing 2-D inverse FFTs…")
    C_hi = _ifft2_coeff(f_hi)
    C_lo = _ifft2_coeff(f_lo)

    print("[INFO] Running verification…")
    if not _verify(C_hi, C_lo):
        raise SystemExit("❌  Verification failed – coefficients incorrect.")

    print("[INFO] Saving coefficient JSON file…")
    _save_coefficients(C_hi, C_lo)
    print("✅  Done.")


if __name__ == "__main__":
    main()
