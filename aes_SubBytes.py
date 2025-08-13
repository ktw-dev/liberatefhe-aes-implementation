import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np

from aes_xor import build_power_basis  # reuse power-basis helper
from engine_context import CKKS_EngineContext

_THIS_DIR = Path(__file__).resolve().parent
_COEFF_PATH = _THIS_DIR / "coeffs" / "sbox_coeffs.json"

# -----------------------------------------------------------------------------
# 1. Load sparse coefficient dictionaries
# -----------------------------------------------------------------------------

with _COEFF_PATH.open("r", encoding="utf-8") as f:
    _data = json.load(f)

_TOL: float = _data.get("tol", 1e-12)

_COEFF_HI: Dict[Tuple[int, int], complex] = {}
_COEFF_LO: Dict[Tuple[int, int], complex] = {}

arr_hi = np.array(_data["sbox_upper_mv_coeffs_real"]) + 1j * np.array(_data["sbox_upper_mv_coeffs_imag"])
arr_lo = np.array(_data["sbox_lower_mv_coeffs_real"]) + 1j * np.array(_data["sbox_lower_mv_coeffs_imag"])

for p in range(16):
    for q in range(16):
        c_hi = arr_hi[p, q]
        c_lo = arr_lo[p, q]
        if abs(c_hi) > _TOL:    
            _COEFF_HI[(p, q)] = complex(c_hi)
        if abs(c_lo) > _TOL:
            _COEFF_LO[(p, q)] = complex(c_lo)

# -----------------------------------------------------------------------------
# 2. Plain-text cache per engine (encoded coefficients)
# -----------------------------------------------------------------------------

_PT_CACHE: Dict[Tuple[int, str], Dict[Tuple[int, int], Any]] = {}


def _get_coeff_plaintexts(engine, which: str):
    """Return dict[(p,q)] → encoded-coeff plaintexts for given engine & "hi"/"lo"."""
    key = (id(engine), which)
    if key in _PT_CACHE:
        return _PT_CACHE[key]

    coeffs = _COEFF_HI if which == "hi" else _COEFF_LO
    slot_cnt = engine.slot_count
    pt_dict: Dict[Tuple[int, int], Any] = {}
    for (p, q), c in coeffs.items():
        vec = np.full(slot_cnt, c, dtype=np.complex128)
        pt_dict[(p, q)] = engine.encode(vec)
    _PT_CACHE[key] = pt_dict
    return pt_dict

# -----------------------------------------------------------------------------
# 3. Polynomial evaluation helper
# -----------------------------------------------------------------------------


def _poly_eval(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any, which: str):
    engine = engine_context.get_engine()
    relin_key = engine_context.get_relinearization_key()
    conj_key = engine_context.get_conjugation_key()

    basis_x = _build_power_basis(engine_context, ct_hi, relin_key, conj_key)
    basis_y = _build_power_basis(engine_context, ct_lo, relin_key, conj_key)

    coeff_pt = _get_coeff_plaintexts(engine, which)

    # start with zero ciphertext (scale/level match)
    cipher_res = engine.multiply(ct_hi, 0.0)

    for (p, q), pt in coeff_pt.items():
        term = engine.multiply(basis_x[p], basis_y[q], relin_key)
        term = engine.multiply(term, pt)
        cipher_res = engine.add(cipher_res, term)

    return cipher_res

def _ones_cipher(engine, engine_context):
    """Encrypt vector of all-ones directly (avoids scale mismatch)."""
    ones_vec = np.ones(engine.slot_count, dtype=np.complex128)
    return engine.encrypt(ones_vec, engine_context.get_public_key())

def _build_power_basis(engine_context: CKKS_EngineContext, ct, relin_key, conj_key):
    """Return dict exp→ct for exponents 0‥15 using power_basis + conjugates.

    Steps:
    1. `engine.make_power_basis(ct, 8, relin_key)` → ct^1..ct^8.
    2. Conjugate the first 7 powers to obtain ct^-1..ct^-7 ≡ ct^15..ct^9.
    3. Add exponent 0 as an encryption of the all-ones vector.
    """
    engine = engine_context.get_engine()
    
    # Positive powers 1..8
    pos_basis = engine.make_power_basis(ct, 8, relin_key)  # list length 8

    basis: Dict[int, object] = {}
    basis[0] = _ones_cipher(engine, engine_context)

    for idx, c in enumerate(pos_basis, start=1):
        basis[idx] = c  # exponents 1..8

    # Negative powers: ct^-k  (k=1..7) → ct^(16-k)
    for k in range(1, 8):
        conj_ct = engine.conjugate(pos_basis[k - 1], conj_key)  # ct^-k
        basis[16 - k] = conj_ct  # exponents 15..9

    return basis

def _sum_terms_tree(engine_context: CKKS_EngineContext, terms: list[Any]) -> Any:
    engine = engine_context.get_engine()
    if not terms: return engine.encrypt(np.zeros(engine.slot_count), engine_context.get_public_key())
    while len(terms) > 1:
        new_terms = [engine.add(terms[i], terms[i + 1]) if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
        terms = new_terms
    return terms[0]


def _sbox_poly(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
    """Homomorphic evaluation of AES S-Box on 4-bit nibbles (upper/lower)."""
    engine = engine_context.get_engine()
    relin_key = engine_context.get_relinearization_key()
    conj_key = engine_context.get_conjugation_key()

    hi_ct = _poly_eval(engine_context, ct_hi, ct_lo, "hi")
    lo_ct = _poly_eval(engine_context, ct_hi, ct_lo, "lo")

    # Optionally refresh scale/level via bootstrap (kept from original code)
    hi_ct = engine.bootstrap(hi_ct, relin_key, conj_key, engine_context.get_bootstrap_key())
    lo_ct = engine.bootstrap(lo_ct, relin_key, conj_key, engine_context.get_bootstrap_key())
    
    hi_ct = engine.intt(hi_ct)
    lo_ct = engine.intt(lo_ct)

    return hi_ct, lo_ct

# API with context first
def sub_bytes(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any):
    """Public API wrapper matching previous signature."""
    return _sbox_poly(engine_context, ct_hi, ct_lo)



if __name__ == "__main__":
    from aes_transform_zeta import int_to_zeta, zeta_to_int
    from aes_split_to_nibble import split_to_nibbles
    import time
    
    engine_context = CKKS_EngineContext(signature=1, use_bootstrap=True, mode="parallel", thread_count=16, device_id=0)
    engine = engine_context.engine
    public_key = engine_context.public_key
    secret_key = engine_context.secret_key
    relinearization_key = engine_context.relinearization_key
    conjugation_key = engine_context.conjugation_key
    bootstrap_key = engine_context.bootstrap_key
    
    print("engine init")
    
    # 1. Encrypt inputs
    np.random.seed(42)
    int_array = np.random.randint(0, 255, size=32768, dtype=np.uint8)

    alpha_int, beta_int = split_to_nibbles(int_array)
    
    # Map to zeta domain
    alpha = int_to_zeta(alpha_int)
    beta  = int_to_zeta(beta_int)
    
    enc_alpha = engine.encrypt(alpha, public_key, level = 10)
    enc_beta = engine.encrypt(beta, public_key, level = 10)
    
    # 2. Evaluate SubBytes operation
    start_time = time.time()
    print("sub_bytes.level: ", enc_alpha.level)
    sub_bytes_hi, sub_bytes_lo = sub_bytes(engine_context, enc_alpha, enc_beta)
    end_time = time.time()
    print(f"SubBytes time taken: {end_time - start_time} seconds")
    print("sub_bytes_hi.level: ", sub_bytes_hi.level)

    start_time = time.time()
    # 3. Decrypt result
    decoded_zeta_hi = engine.decrypt(sub_bytes_hi, secret_key)
    decoded_int_hi = zeta_to_int(decoded_zeta_hi)
    decoded_zeta_lo = engine.decrypt(sub_bytes_lo, secret_key)
    decoded_int_lo = zeta_to_int(decoded_zeta_lo)

    print(decoded_int_hi)
    print(decoded_int_lo)

    # 4. Validate against NumPy AES-128 S-Box implementation
    from aes_128_numpy import S_BOX  # NumPy reference S-Box table

    # Expected SubBytes output using reference table
    expected_bytes = S_BOX[int_array]

    # Combine decrypted high/low nibbles from FHE evaluation
    output_bytes = (decoded_int_hi.astype(np.uint8) << 4) | decoded_int_lo.astype(np.uint8)

    # Compare
    mismatches = np.sum(expected_bytes != output_bytes)
    if mismatches == 0:
        print("\u2705  SubBytes output matches NumPy AES S-Box for all samples!")
    else:
        print(f"\u274C  SubBytes mismatch in {mismatches} out of {int_array.size} samples.")
        # Show first few mismatching indices for debugging
        mismatch_idx = np.where(expected_bytes != output_bytes)[0][:10]
        for idx in mismatch_idx:
            in_byte = int(int_array[idx])
            exp_byte = int(expected_bytes[idx])
            out_byte = int(output_bytes[idx])
            print(f"  idx {idx}: input 0x{in_byte:02X} -> expected 0x{exp_byte:02X}, got 0x{out_byte:02X}")
        raise AssertionError("SubBytes result does not match reference implementation.")
    

if __name__ == "__main__":
    from aes_transform_zeta import int_to_zeta, zeta_to_int
    from aes_split_to_nibble import split_to_nibbles
    import time
    
    engine_context = CKKS_EngineContext(signature=1, use_bootstrap=True, mode="parallel", thread_count=16, device_id=0)
    engine = engine_context.engine
    public_key = engine_context.public_key
    secret_key = engine_context.secret_key
    relinearization_key = engine_context.relinearization_key
    conjugation_key = engine_context.conjugation_key
    bootstrap_key = engine_context.bootstrap_key
    
    print("engine init")
    
    # 1. Encrypt inputs
    np.random.seed(42)
    int_array = np.random.randint(0, 255, size=32768, dtype=np.uint8)

    alpha_int, beta_int = split_to_nibbles(int_array)
    
    # Map to zeta domain
    alpha = int_to_zeta(alpha_int)
    beta  = int_to_zeta(beta_int)
    
    enc_alpha = engine.encrypt(alpha, public_key, level = 10)
    enc_beta = engine.encrypt(beta, public_key, level = 10)
    
    # 2. Evaluate SubBytes operation
    start_time = time.time()
    print("sub_bytes.level: ", enc_alpha.level)
    sub_bytes_hi, sub_bytes_lo = sub_bytes(engine_context, enc_alpha, enc_beta)
    end_time = time.time()
    print(f"SubBytes time taken: {end_time - start_time} seconds")
    print("sub_bytes_hi.level: ", sub_bytes_hi.level)

    start_time = time.time()
    # 3. Decrypt result
    decoded_zeta_hi = engine.decrypt(sub_bytes_hi, secret_key)
    decoded_int_hi = zeta_to_int(decoded_zeta_hi)
    decoded_zeta_lo = engine.decrypt(sub_bytes_lo, secret_key)
    decoded_int_lo = zeta_to_int(decoded_zeta_lo)

    print(decoded_int_hi)
    print(decoded_int_lo)

    # 4. Validate against NumPy AES-128 S-Box implementation
    from aes_128_numpy import S_BOX  # NumPy reference S-Box table

    # Expected SubBytes output using reference table
    expected_bytes = S_BOX[int_array]

    # Combine decrypted high/low nibbles from FHE evaluation
    output_bytes = (decoded_int_hi.astype(np.uint8) << 4) | decoded_int_lo.astype(np.uint8)

    # Compare
    mismatches = np.sum(expected_bytes != output_bytes)
    if mismatches == 0:
        print("\u2705  SubBytes output matches NumPy AES S-Box for all samples!")
    else:
        print(f"\u274C  SubBytes mismatch in {mismatches} out of {int_array.size} samples.")
        # Show first few mismatching indices for debugging
        mismatch_idx = np.where(expected_bytes != output_bytes)[0][:10]
        for idx in mismatch_idx:
            in_byte = int(int_array[idx])
            exp_byte = int(expected_bytes[idx])
            out_byte = int(output_bytes[idx])
            print(f"  idx {idx}: input 0x{in_byte:02X} -> expected 0x{exp_byte:02X}, got 0x{out_byte:02X}")
        raise AssertionError("SubBytes result does not match reference implementation.")