"""
aes_inv_SubBytes.py
───────────────────────
• 역할 : 암호문(ζ‑입력)을 AES Inverse S‑Box로 변환하여 제타(ζ) 암호문을 출력.
• 구조 : aes_SubBytes.py와 동일한 구조로 리팩토링됨.
         - 희소(sparse) 계수 딕셔너리 사용
         - 재사용 가능한 _poly_eval 헬퍼 함수
         - 연산 후 부트스트래핑 적용
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List
import numpy as np

from engine_context import CKKS_EngineContext

# ----------------------------------------------------------------------
# 1. 계수 로드 및 희소(Sparse) 딕셔너리 생성
# ----------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_COEFF_PATH = _THIS_DIR / "coeffs" / "inverse_sbox_coeffs.json"

with _COEFF_PATH.open("r", encoding="utf-8") as f:
    _data = json.load(f)

_TOL: float = _data.get("tol", 1e-12)
_COEFF_HI: Dict[Tuple[int, int], complex] = {}
_COEFF_LO: Dict[Tuple[int, int], complex] = {}

# JSON의 2D 배열을 순회하며 0이 아닌 값만 딕셔너리에 저장
arr_hi = np.array(_data["inverse_sbox_upper_mv_coeffs_real"]) + 1j * np.array(_data["inverse_sbox_upper_mv_coeffs_imag"])
arr_lo = np.array(_data["inverse_sbox_lower_mv_coeffs_real"]) + 1j * np.array(_data["inverse_sbox_lower_mv_coeffs_imag"])

for p in range(16):
    for q in range(16):
        c_hi = arr_hi[p, q]
        c_lo = arr_lo[p, q]
        if abs(c_hi) > _TOL: 
            _COEFF_HI[(p, q)] = complex(c_hi)
        if abs(c_lo) > _TOL: 
            _COEFF_LO[(p, q)] = complex(c_lo)

# ----------------------------------------------------------------------
# 2. Plaintext 계수 캐시
# ----------------------------------------------------------------------
_PT_CACHE: Dict[Tuple[int, str], Dict[Tuple[int, int], Any]] = {}

def _get_coeff_plaintexts(engine, which: str):
    """엔진별, hi/lo별로 인코딩된 계수 평문을 캐시에서 가져오거나 생성합니다."""
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

# ----------------------------------------------------------------------
# 3. 다항식 평가 및 헬퍼 함수
# ----------------------------------------------------------------------
def _ones_cipher(engine, engine_context: CKKS_EngineContext):
    """스케일 불일치 문제를 피하기 위해 1 벡터를 직접 암호화합니다."""
    ones_vec = np.ones(engine.slot_count, dtype=np.complex128)
    return engine.encrypt(ones_vec, engine_context.get_public_key())

def _build_power_basis(engine_context: CKKS_EngineContext, ct, rlk, conj_key):
    """0~15차수 암호문 딕셔너리를 생성합니다."""
    engine = engine_context.get_engine()
    pos_basis = engine.make_power_basis(ct, 8, rlk)
    basis: Dict[int, object] = {_: None for _ in range(16)}
    basis[0] = _ones_cipher(engine, engine_context)
    for idx, c in enumerate(pos_basis, start=1): basis[idx] = c
    for k in range(1, 8): basis[16 - k] = engine.conjugate(pos_basis[k - 1], conj_key)
    return basis

def _poly_eval(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any, which: str):
    """주어진 암호문들에 대해 hi 또는 lo 다항식을 평가합니다."""
    engine = engine_context.get_engine()
    rlk = engine_context.get_relinearization_key()
    conj_key = engine_context.get_conjugation_key()
    
    basis_x = _build_power_basis(engine_context, ct_hi, rlk, conj_key)
    basis_y = _build_power_basis(engine_context, ct_lo, rlk, conj_key)
    
    coeff_pt = _get_coeff_plaintexts(engine, which)

    cipher_res = engine.multiply(ct_hi, 0.0) # 결과 암호문 초기화

    for (p, q), pt in coeff_pt.items():
        term = engine.multiply(basis_x[p], basis_y[q], rlk)
        term = engine.multiply(term, pt)
        cipher_res = engine.add(cipher_res, term)

    return cipher_res

def inv_sbox_poly(ctx: CKKS_EngineContext, ct_hi_zeta: Any, ct_lo_zeta: Any) -> Tuple[Any, Any]:
    """동형 Inverse S-Box의 고수준 오케스트레이션 함수."""
    engine = ctx.get_engine()
    rlk, conj_key = ctx.get_relinearization_key(), ctx.get_conjugation_key()
    
    res_hi = _poly_eval(ctx, ct_hi_zeta, ct_lo_zeta, "hi")
    res_lo = _poly_eval(ctx, ct_hi_zeta, ct_lo_zeta, "lo")

    # 암호화 파이프라인의 안정성을 위해 부트스트래핑을 동일하게 적용
    res_hi = engine.bootstrap(res_hi, rlk, conj_key, ctx.get_bootstrap_key())
    res_lo = engine.bootstrap(res_lo, rlk, conj_key, ctx.get_bootstrap_key())

    return res_hi, res_lo

# ----------------------------------------------------------------------
# 4. 외부 공개 API
# ----------------------------------------------------------------------
def inv_sub_bytes(ctx: CKKS_EngineContext, ct_hi_zeta: Any, ct_lo_zeta: Any):
    """Public API: aes_SubBytes와 동일한 시그니처를 가집니다."""
    return inv_sbox_poly(ctx, ct_hi_zeta, ct_lo_zeta)



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
    sub_bytes_hi, sub_bytes_lo = inv_sub_bytes(engine_context, enc_alpha, enc_beta)
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
    from aes_128_numpy import INV_S_BOX  # NumPy reference S-Box table

    # Expected SubBytes output using reference table
    expected_bytes = INV_S_BOX[int_array]

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
    
