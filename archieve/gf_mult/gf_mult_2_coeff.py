"""gf_mult_2_coeff.py

Generate 16×16 monomial coefficients (C[p,q]) such that
    P_hi(ζ^{a}, ζ^{b}) = ζ^{ hi( MUL2(a<<4|b) ) }
    P_lo(ζ^{a}, ζ^{b}) = ζ^{ lo( MUL2(a<<4|b) ) }
where ζ = exp(-2πi/16).

Outputs two JSON files:
  gf_mult2_hi_coeffs.json
  gf_mult2_lo_coeffs.json

Each JSON schema:
{
  "shape": [16,16],
  "tol"  : 1e-12,
  "entries" : [[p,q,re,im], ...],
  "note": "monomial coeffs for GF×2 upper nibble" (or lower)
}
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# locate aes_MixColumns to import GF_MULT_2 table
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aes_MixColumns import GF_MULT_2  # type: ignore

THIS_DIR = Path(__file__).resolve().parent

ZETA = np.exp(-2j * np.pi / 16)
TOL = 1e-12


def _build_tables():
    """Return two 16×16 numpy arrays (f_hi, f_lo) of ζ^value."""
    f_hi = np.empty((16, 16), dtype=np.complex128)
    f_lo = np.empty((16, 16), dtype=np.complex128)

    for a in range(16):
        for b in range(16):
            x = (a << 4) | b
            y = int(GF_MULT_2[x])
            y_hi = (y >> 4) & 0xF
            y_lo = y & 0xF
            f_hi[a, b] = ZETA ** y_hi
            f_lo[a, b] = ZETA ** y_lo
    return f_hi, f_lo


def _ifft2_coeff(table: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(table)


def _compress_coeffs(C: np.ndarray) -> list[list[float]]:
    entries: list[list[float]] = []
    for p in range(16):
        for q in range(16):
            if abs(C[p, q]) > TOL:
                entries.append([p, q, float(C[p, q].real), float(C[p, q].imag)])
    return entries


def main():
    f_hi, f_lo = _build_tables()
    C_hi = _ifft2_coeff(f_hi)
    C_lo = _ifft2_coeff(f_lo)

    for name, C in [("hi", C_hi), ("lo", C_lo)]:
        entries = _compress_coeffs(C)
        obj = {
            "shape": [16, 16],
            "tol": TOL,
            "entries": entries,
            "note": f"monomial coefficients for GF×2 {name} nibble via ifft2"
        }
        out_path = THIS_DIR / f"gf_mult2_{name}_coeffs.json"
        out_path.write_text(json.dumps(obj, indent=2))
        print(f"[INFO] Saved {out_path.name}  (non-zero={len(entries)}/{16*16})")


if __name__ == "__main__":
    main()
