import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple
from functools import lru_cache
from dataclasses import dataclass
import numpy as np
import desilofhe
from engine_context import CKKS_EngineContext

# -----------------------------------------------------------------------------
# Import helper from project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from desilofhe import Engine  # pylint: disable=import-error

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
DEGREE      = 15    # Maximum exponent needed for XOR polynomial
COEFFS_JSON = Path(__file__).resolve().parent / "coeffs" / "xor_mono_coeffs.json"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ones_cipher(engine_context: CKKS_EngineContext, template_ct):
    """Return ciphertext with all slots = 1 matching scale/level of template_ct."""
    # Create zero ciphertext with same scale/level via scalar multiply
    zero_ct = engine_context.ckks_multiply(template_ct, 0.0)  # keeps params, slots all 0

    try:
        # Preferred path if library supports add_plain
        ones_ct = engine_context.ckks_add(zero_ct, 1.0)
    except AttributeError:
        # Fallback: encode plaintext ones then add as ciphertext-plaintext
        ones_pt = engine_context.get_engine().encode(np.ones(engine_context.get_slot_count()))
        ones_ct = engine_context.ckks_add(zero_ct, ones_pt)
    return ones_ct


def build_power_basis(engine_context: CKKS_EngineContext, ct):
    """Return dict exp→ct for exponents 0‥15 using power_basis + conjugates.

    Steps:
    1. `engine.make_power_basis(ct, 8, relin_key)` → ct^1..ct^8.
    2. Conjugate the first 7 powers to obtain ct^-1..ct^-7 ≡ ct^15..ct^9.
    3. Add exponent 0 as an encryption of the all-ones vector.
    """
    # Positive powers 1..8
    pos_basis = engine_context.ckks_power_basis(ct, 8)  # list length 8

    basis: Dict[int, object] = {}
    basis[0] = ones_cipher(engine_context, ct)

    for idx, c in enumerate(pos_basis, start=1):
        basis[idx] = c  # exponents 1..8

    # Negative powers: ct^-k  (k=1..7) → ct^(16-k)
    for k in range(1, 8):
        conj_ct = engine_context.ckks_conjugate(pos_basis[k - 1])  # ct^-k
        basis[16 - k] = conj_ct  # exponents 15..9

    return basis

@lru_cache(maxsize=1)
def _load_coeffs(path=COEFFS_JSON):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {(int(i), int(j)): complex(r, im)
            for i, j, r, im in data["entries"] if r or im}

@lru_cache(maxsize=1)

def _coeff_plaintexts(slot_count: int):
    """Return dict[(i,j)] -> plaintext (encoded complex coeff) for given slot count."""
    coeffs = _load_coeffs()
    # We create a temporary Engine instance only to get encode? Actually we cannot.
    # We will receive engine later; workaround: this function will be wrapped.
    return coeffs

# We'll implement engine-specific cache below
_coeff_pt_cache: dict[int, Dict[Tuple[int,int], object]] = {}


def _get_coeff_plaintexts(engine: Engine):
    """Get or build plaintexts of coefficients encoded for the given engine."""
    slot_cnt = engine.slot_count
    if slot_cnt in _coeff_pt_cache:
        return _coeff_pt_cache[slot_cnt]
    coeffs = _load_coeffs()
    pt_dict: Dict[Tuple[int,int], object] = {}
    for key, coeff in coeffs.items():
        vec = np.full(slot_cnt, coeff, dtype=np.complex128)
        pt = engine.encode(vec)
        pt_dict[key] = pt
    _coeff_pt_cache[slot_cnt] = pt_dict
    return pt_dict

# -----------------------------------------------------------------------------
# XOR operation
# -----------------------------------------------------------------------------

def _xor_operation(engine_context, enc_alpha, enc_beta):
    engine = engine_context.engine
    relin_key = engine_context.relinearization_key
    conjugate_key = engine_context.conjugation_key
    
    # 1. Build power bases
    base_x = build_power_basis(engine_context, enc_alpha)
    base_y = build_power_basis(engine_context, enc_beta)

    # 2. Pre-encoded polynomial coefficients
    coeff_pts = _get_coeff_plaintexts(engine)
    
    # 3. Evaluate polynomial securely
    cipher_res = engine_context.ckks_multiply(enc_alpha, 0.0)

    for (i, j), coeff_pt in coeff_pts.items():
        term = engine_context.ckks_multiply(base_x[i], base_y[j])
        term_res = engine_context.ckks_multiply(term, coeff_pt)
        cipher_res = engine_context.ckks_add(cipher_res, term_res)
           
    return cipher_res




# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------
import time
import math

def transform_to_zeta(arr: np.ndarray) -> np.ndarray:
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

if __name__ == "__main__":
    engine_context = CKKS_EngineContext(signature=1, use_bootstrap=True, mode="parallel", thread_count=16, device_id=0)
    engine = engine_context.engine
    public_key = engine_context.public_key
    secret_key = engine_context.secret_key
    relinearization_key = engine_context.relinearization_key
    conjugation_key = engine_context.conjugation_key
    bootstrap_key = engine_context.bootstrap_key
    
    # 1. Encrypt inputs
    np.random.seed(42)
    alpha_int = np.random.randint(0, 16, size=32768, dtype=np.uint8)
    beta_int  = np.random.randint(0, 16, size=32768, dtype=np.uint8)
    expected_int = np.bitwise_xor(alpha_int, beta_int)

    # Map to zeta domain
    alpha = transform_to_zeta(alpha_int)
    beta  = transform_to_zeta(beta_int)
    
    enc_alpha = engine.encrypt(alpha, public_key, level=5)
    enc_beta = engine.encrypt(beta, public_key, level=5)
    
    # 2. Evaluate XOR operation
    start_time = time.time()
    cipher_res = _xor_operation(engine_context, enc_alpha, enc_beta)
    
    bootstrap_ct = engine.bootstrap(cipher_res, relinearization_key, conjugation_key, bootstrap_key)
    end_time = time.time()
    print(f"XOR time taken: {end_time - start_time} seconds")

    start_time = time.time()
    # 3. Decrypt result
    decoded_zeta = engine.decrypt(cipher_res, secret_key)
    unit_dec = decoded_zeta / np.abs(decoded_zeta)
    decoded_int = zeta_to_int(unit_dec)
    
    print(np.all(decoded_int == expected_int))