import json
import os
import numpy as np
from typing import Any, List, Tuple, Dict
from engine_context import CKKS_EngineContext
from aes_transform_zeta import int_cipher_to_zeta_cipher

_BASE = os.path.dirname(__file__)
_COEFF_PATH = os.path.join(_BASE, "coeffs", "sbox_coeffs.json")
with open(_COEFF_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)
C_hi: np.ndarray = (np.array(_data["sbox_upper_mv_coeffs_real"]) + 1j * np.array(_data["sbox_upper_mv_coeffs_imag"]))
C_lo: np.ndarray = (np.array(_data["sbox_lower_mv_coeffs_real"]) + 1j * np.array(_data["sbox_lower_mv_coeffs_imag"]))
_DEG = C_hi.shape[0] - 1
_EPS = 1e-12
_PT_COEFFS_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

def _pre_encode_coeffs(engine: Any) -> Tuple[np.ndarray, np.ndarray]:
    print("\nINFO: Pre-encoding S-Box coefficients for the first time...")
    pt_c_hi, pt_c_lo = np.empty_like(C_hi, dtype=object), np.empty_like(C_lo, dtype=object)
    slot_count = 16 * 2048
    for i in range(_DEG + 1):
        for j in range(_DEG + 1):
            if abs(C_hi[i, j]) >= _EPS: pt_c_hi[i, j] = engine.encode(np.full(slot_count, C_hi[i, j]))
            if abs(C_lo[i, j]) >= _EPS: pt_c_lo[i, j] = engine.encode(np.full(slot_count, C_lo[i, j]))
    print("INFO: Coefficient pre-encoding complete.")
    return pt_c_hi, pt_c_lo

def _build_optimized_power_basis(engine_context: CKKS_EngineContext, ct: Any) -> Dict[int, Any]:
    engine, rlk, conj_key = engine_context.get_engine(), engine_context.get_relinearization_key(), engine_context.get_conjugation_key()
    basis: Dict[int, Any] = {}
    pos_basis = engine.make_power_basis(ct, 8, rlk)
    for i, c in enumerate(pos_basis, start=1): basis[i] = c
    for i in range(1, 8): basis[16 - i] = engine.conjugate(pos_basis[i - 1], conj_key)
    basis[0] = engine.encrypt(np.ones(engine.slot_count, dtype=np.complex128), engine_context.get_public_key())
    return basis

def _sum_terms_tree(engine_context: CKKS_EngineContext, terms: List[Any]) -> Any:
    engine = engine_context.get_engine()
    if not terms: return engine.encrypt(np.zeros(engine.slot_count), engine_context.get_public_key())
    while len(terms) > 1:
        new_terms = [engine.add(terms[i], terms[i + 1]) if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
        terms = new_terms
    return terms[0]


def sbox_poly(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
    engine = engine_context.get_engine()
    rlk = engine_context.get_relinearization_key()
    conj_key = engine_context.get_conjugation_key()
    engine_id = id(engine)
    if engine_id not in _PT_COEFFS_CACHE:
        _PT_COEFFS_CACHE[engine_id] = _pre_encode_coeffs(engine)
    pt_c_hi, pt_c_lo = _PT_COEFFS_CACHE[engine_id]

    hi_basis = _build_optimized_power_basis(engine_context, ct_hi)
    lo_basis = _build_optimized_power_basis(engine_context, ct_lo)

    k = 4
    num_chunks = (_DEG + 1) // k
    chunk_results_hi, chunk_results_lo = [], []

    for m in range(num_chunks):
        baby_step_terms_hi, baby_step_terms_lo = [], []
        for i in range(k):
            inner_terms_hi, inner_terms_lo = [], []
            for j in range(_DEG + 1):
                idx_y = i + m * k
                ct_hi_pow = hi_basis[j]
                if abs(C_hi[j, idx_y]) >= _EPS:
                    inner_terms_hi.append(engine.multiply(ct_hi_pow, pt_c_hi[j, idx_y]))
                if abs(C_lo[j, idx_y]) >= _EPS:
                    inner_terms_lo.append(engine.multiply(ct_hi_pow, pt_c_lo[j, idx_y]))
            
            inner_sum_hi = _sum_terms_tree(engine_context, inner_terms_hi)
            inner_sum_lo = _sum_terms_tree(engine_context, inner_terms_lo)
            
            ct_lo_pow = lo_basis[i]
            baby_step_terms_hi.append(engine.multiply(inner_sum_hi, ct_lo_pow, rlk))
            baby_step_terms_lo.append(engine.multiply(inner_sum_lo, ct_lo_pow, rlk))

        chunk_results_hi.append(_sum_terms_tree(engine_context, baby_step_terms_hi))
        chunk_results_lo.append(_sum_terms_tree(engine_context, baby_step_terms_lo))

    final_result_hi = chunk_results_hi[0]
    final_result_lo = chunk_results_lo[0]
    for m in range(1, num_chunks):
        ct_lo_pow_k = lo_basis[m * k]
        term_hi = engine.multiply(chunk_results_hi[m], ct_lo_pow_k, rlk)
        final_result_hi = engine.add(final_result_hi, term_hi)
        term_lo = engine.multiply(chunk_results_lo[m], ct_lo_pow_k, rlk)
        final_result_lo = engine.add(final_result_lo, term_lo)
    
    final_result_hi = engine.bootstrap(final_result_hi, rlk, conj_key, engine_context.get_bootstrap_key())
    final_result_lo = engine.bootstrap(final_result_lo, rlk, conj_key, engine_context.get_bootstrap_key())
    
    # 지금까지의 결과는 정수 형태이므로, 이를 ζ 형태로 변환한다. 
    final_result_hi = int_cipher_to_zeta_cipher(engine_context, final_result_hi)
    final_result_lo = int_cipher_to_zeta_cipher(engine_context, final_result_lo)
       
    return final_result_hi, final_result_lo

# API with context first
def sub_bytes(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any):
    return sbox_poly(engine_context, ct_hi, ct_lo)



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
    decoded_zeta = engine.decrypt(sub_bytes_hi, secret_key)
    decoded_int_hi = zeta_to_int(decoded_zeta)
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