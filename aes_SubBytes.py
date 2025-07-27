import json
import os
import numpy as np
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass
from engine_context import CKKS_EngineContext

@dataclass
class NibblePack:
    hi: Any
    lo: Any
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
    slot_count = engine.slot_count
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


def sbox_poly(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any) -> NibblePack:
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
            
            inner_sum_hi = _sum_terms_tree(inner_terms_hi, engine_context)
            inner_sum_lo = _sum_terms_tree(inner_terms_lo, engine_context)
            
            ct_lo_pow = lo_basis[i]
            baby_step_terms_hi.append(engine.multiply(inner_sum_hi, ct_lo_pow, rlk))
            baby_step_terms_lo.append(engine.multiply(inner_sum_lo, ct_lo_pow, rlk))

        chunk_results_hi.append(_sum_terms_tree(baby_step_terms_hi, engine_context))
        chunk_results_lo.append(_sum_terms_tree(baby_step_terms_lo, engine_context))

    final_result_hi = chunk_results_hi[0]
    final_result_lo = chunk_results_lo[0]
    for m in range(1, num_chunks):
        ct_lo_pow_k = lo_basis[m * k]
        term_hi = engine.multiply(chunk_results_hi[m], ct_lo_pow_k, rlk)
        final_result_hi = engine.add(final_result_hi, term_hi)
        term_lo = engine.multiply(chunk_results_lo[m], ct_lo_pow_k, rlk)
        final_result_lo = engine.add(final_result_lo, term_lo)
    
    return NibblePack(hi=final_result_hi, lo=final_result_lo)

# API with context first
def sub_bytes(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any) -> NibblePack:
    return sbox_poly(engine_context, ct_hi, ct_lo)