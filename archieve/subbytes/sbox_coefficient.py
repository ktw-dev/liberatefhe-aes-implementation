#!/usr/bin/env python3
"""sbox_coefficient.py

Generate 16×16 monomial coefficient matrices `C_hi[p,q]`, `C_lo[p,q]` such that

    P_hi(ζ^a, ζ^b) = ζ^{hi(S_BOX[a<<4|b])}
    P_lo(ζ^a, ζ^b) = ζ^{lo(S_BOX[a<<4|b])}

for all a,b∈{0,…,15}, where ζ = exp(−2πi/16).

The coefficients are obtained via 2-D inverse DFT (`np.fft.ifft2`).  The result is
serialized to `coeffs/sbox_coeffs.json` with the same schema used elsewhere in
this repository (real/imag components split).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
COEFF_DIR = ROOT / "coeffs"
COEFF_DIR.mkdir(exist_ok=True)
OUT_PATH = COEFF_DIR / "sbox_coeffs.json"

ZETA: complex = np.exp(-2j * np.pi / 16)
TOL = 1e-12

# -----------------------------------------------------------------------------
# Reference S-Box
# -----------------------------------------------------------------------------
from aes_128_numpy import S_BOX  # pylint: disable=wrong-import-position

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _build_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Return two (16,16) arrays with ζ^hi and ζ^lo values."""
    f_hi = np.empty((16, 16), dtype=np.complex128)
    f_lo = np.empty((16, 16), dtype=np.complex128)

    for a in range(16):
        for b in range(16):
            x = (a << 4) | b
            y = int(S_BOX[x])
            hi = (y >> 4) & 0xF
            lo = y & 0xF
            f_hi[a, b] = ZETA ** hi
            f_lo[a, b] = ZETA ** lo
    return f_hi, f_lo


def _ifft2_coeff(table: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(table)


def _compress_coeffs(C: np.ndarray):
    """Return list[[p,q,re,im]] dropping ~0 coeffs."""
    entries = []
    for p in range(16):
        for q in range(16):
            c = C[p, q]
            if abs(c) > TOL:
                entries.append([p, q, float(c.real), float(c.imag)])
    return entries

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------

def _zeta_to_int(zeta_arr: np.ndarray) -> np.ndarray:
    """Inverse of `transform_to_zeta` assuming unit-magnitude complex numbers.

    Values are mapped back to integers 0‥15 by measuring their phase.
    """
    angles = np.angle(zeta_arr)  # range (-π, π]
    k      = (-angles * 16) / (2 * np.pi)
    k      = np.mod(np.rint(k), 16).astype(np.uint8)
    return k

def _verify(C_hi: np.ndarray, C_lo: np.ndarray, trials: int = 256) -> bool:
    rng = np.random.default_rng()
    for _ in range(trials):
        a = int(rng.integers(0, 16))
        b = int(rng.integers(0, 16))
        x = ZETA ** a
        y = ZETA ** b
        # evaluate polynomials
        val_hi = (C_hi * np.outer(x ** np.arange(16), y ** np.arange(16))).sum()
        val_lo = (C_lo * np.outer(x ** np.arange(16), y ** np.arange(16))).sum()
        hi = _zeta_to_int(val_hi)
        lo = _zeta_to_int(val_lo)
        exp_byte = S_BOX[(a << 4) | b]
        if hi != ((exp_byte >> 4) & 0xF) or lo != (exp_byte & 0xF):
            print(f"[FAIL] a={a} b={b}: got {(hi,lo)}, expected {(exp_byte>>4)&0xF, exp_byte&0xF}")
            return False
    print(f"[PASS] Verification succeeded for {trials} random points.")
    return True

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("[INFO] Building ζ-domain tables …")
    f_hi, f_lo = _build_tables()

    print("[INFO] Computing 2-D inverse FFTs …")
    C_hi = _ifft2_coeff(f_hi)
    C_lo = _ifft2_coeff(f_lo)

    print("[INFO] Verification …")
    if not _verify(C_hi, C_lo):
        raise SystemExit("❌ Verification failed – coefficients wrong")

    print("[INFO] Compress & save JSON …")
    obj = {
        "sbox_upper_mv_coeffs_real": C_hi.real.tolist(),
        "sbox_upper_mv_coeffs_imag": C_hi.imag.tolist(),
        "sbox_lower_mv_coeffs_real": C_lo.real.tolist(),
        "sbox_lower_mv_coeffs_imag": C_lo.imag.tolist(),
        "shape": [16, 16],
        "tol": TOL,
        "note": "monomial coeffs for AES S-Box (ζ-domain) via ifft2",
    }
    OUT_PATH.write_text(json.dumps(obj, indent=2))
    print(f"✅  Coefficient file written → {OUT_PATH}")


if __name__ == "__main__":
    main()
