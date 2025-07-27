# -----------------------------------------------------------------------------
# Homomorphic int→ζ transformation utilities (do NOT modify existing functions)
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
from engine_context import CKKS_EngineContext
from aes_xor import build_power_basis  # reuse existing helper

__all__ = [
    "int_to_zeta",
]                         # shape (32768,), dtype=complex128

def int_to_zeta(arr: np.ndarray) -> np.ndarray:
    result = np.exp(-2j * np.pi * (arr % 16) / 16)
    return result
    
def zeta_to_int(zeta_arr: np.ndarray) -> np.ndarray:
    """Inverse of `transform_to_zeta` assuming unit-magnitude complex numbers.

    Values are mapped back to integers 0‥15 by measuring their phase.
    """
    angles = np.angle(zeta_arr)  # range (-π, π]
    k      = (-angles * 16) / (2 * np.pi)
    k      = np.mod(np.rint(k), 16).astype(np.uint8)
    return k



# ----------------- coefficient helper ----------------------------------------
_ZETA = np.exp(-2j * np.pi / 16)

# Cache plain coefficients per engine id
_PT_I2Z_CACHE: Dict[int, List[Any]] = {}


def _compute_int2zeta_coeffs() -> np.ndarray:
    """Return array a[0..15] s.t. g(k)=ζ^k with g(t)=Σ a_p t^p."""
    k_vals = np.arange(16)
    f = _ZETA ** k_vals
    # Vandermonde solve
    A = np.vander(k_vals, N=16, increasing=True)
    coeff = np.linalg.solve(A, f)
    return coeff.astype(complex)


_I2Z_COEFF = _compute_int2zeta_coeffs()
# Determine effective maximum degree required (ignore coefficients ~0)
_EPS_COEFF = 1e-12
_ACTIVE_IDX = [idx for idx, c in enumerate(_I2Z_COEFF) if abs(c) > _EPS_COEFF]
_DEG_I2Z = max(_ACTIVE_IDX)

# (coefficients print removed)


def _pre_encode_i2z(engine) -> List[Any]:
    """Encode int→ζ coefficients as CKKS plaintexts (length 16)."""
    slot = engine.slot_count
    pt_list: List[Any] = []
    for a in _I2Z_COEFF:
        vec = np.full(slot, a, dtype=np.complex128)
        pt_list.append(engine.encode(vec))
    return pt_list


# ----------------- power basis & sum helper ----------------------------------

def _build_power_basis(ct: Any, degree: int, ctx: CKKS_EngineContext):
    engine = ctx.get_engine()
    rlk = ctx.get_relinearization_key()
    pub = ctx.get_public_key()
    slot = engine.slot_count
    ones = engine.encrypt(np.ones(slot, dtype=np.complex128), pub)
    basis = [ones, ct]
    for d in range(2, degree + 1):
        basis.append(engine.multiply(basis[d - 1], ct, rlk))
    return basis  # len = degree+1


def _sum_terms_tree(terms: List[Any], ctx: CKKS_EngineContext):
    engine = ctx.get_engine()
    if not terms:
        return engine.encrypt(np.zeros(engine.slot_count, dtype=np.complex128), ctx.get_public_key())
    while len(terms) > 1:
        new_terms = []
        for i in range(0, len(terms), 2):
            if i + 1 < len(terms):
                new_terms.append(engine.add(terms[i], terms[i + 1]))
            else:
                new_terms.append(terms[i])
        terms = new_terms
    return terms[0]


# ----------------- public API -------------------------------------------------

def int_cipher_to_zeta_cipher(ct_int: Any, ctx: CKKS_EngineContext):
    """Convert ciphertext encrypting integers 0..15 to ciphertext encrypting ζ^k.

    Parameters
    ----------
    ct_int : Ciphertext (CKKS)
        Ciphertext whose slots contain integers 0–15 (encoded as CKKS reals).
    ctx : CKKS_EngineContext
        Context providing engine & keys.

    Returns
    -------
    Ciphertext encrypting ζ^{k} for each slot value k.
    """
    engine = ctx.get_engine()
    eid = id(engine)
    if eid not in _PT_I2Z_CACHE:
        _PT_I2Z_CACHE[eid] = _pre_encode_i2z(engine)
    pt_coeffs = _PT_I2Z_CACHE[eid]

    rlk = ctx.get_relinearization_key()
    conj_key = ctx.get_conjugation_key()

    basis_dict = build_power_basis(engine, ct_int, rlk, conj_key)  # exponents 0..15

    basis = [basis_dict[p] for p in range(_DEG_I2Z + 1)]

    # Evaluate polynomial Σ a_p t^p
    terms = [engine.multiply(basis[p], pt_coeffs[p]) for p in range(_DEG_I2Z + 1)]
    return _sum_terms_tree(terms, ctx)

def __main__():
    arr = np.random.randint(0, 16, size=16*2048, dtype=np.uint8)
    print(f"arr: {arr[0]}")
    print(f"arr shape: {arr.shape}")
    print(f"arr dtype: {arr.dtype}")
    result = int_to_zeta(arr)
    print(f"int_to_zeta: {result[0]}")
    print(f"int_to_zeta shape: {result.shape}")
    print(f"int_to_zeta dtype: {result.dtype}")
    result = zeta_to_int(result)
    print(f"zeta_to_int: {result[0]}")
    print(f"zeta_to_int shape: {result.shape}")
    print(f"zeta_to_int dtype: {result.dtype}")

if __name__ == "__main__":
    __main__()