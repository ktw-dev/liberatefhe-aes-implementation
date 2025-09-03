"""aes-MixColumns.py

MixColumns operation for AES-128 ECB mode with FHE compatibility.

The MixColumns operation treats each column of the AES state as a polynomial
and multiplies it by a fixed polynomial in GF(2^8). This provides diffusion
in the AES cipher.

Each column is transformed by multiplying with the matrix:
[02 03 01 01]
[01 02 03 01]  
[01 01 02 03]
[03 01 01 02]

Multiplication in GF(2^8) is implemented using lookup tables and XOR operations.
"""
from __future__ import annotations

import numpy as np
from engine_context import CKKS_EngineContext
from typing import Any
from aes_gf_mult import gf_mult_2, gf_mult_3
from aes_xor import _xor_operation

__all__ = [
    "mix_columns",
]

def mix_columns(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any):
    """두 개의 블록을 받아 MixColumns 연산을 수행하고 결과를 반환   
    Args:
        engine_context: CKKS_EngineContext
        ct_hi: 첫 번째 블록
        ct_lo: 두 번째 블록
        
    Returns:
        ct_hi_out: 첫 번째 블록의 결과
        ct_lo_out: 두 번째 블록의 결과
        
    ------------------------------------------------------------
    Functions / Operation counts
    ------------------------------------------------------------
    • rotate_batch : 2 calls (each produces 3 rotations -> 6 rotated ciphertexts)
    • gf_mul_2     : 1 call
    • gf_mul_3     : 1 call
    • bootstrap    : 6 calls (4 pre-XOR, 2 post-XOR)
    • xor          : 6 calls (3 for high-nibble, 3 for low-nibble)
    
    level 소모량
    
    """
    engine = engine_context.get_engine()
    
    fixed_rotation_key_neg_4_2048 = engine_context.get_fixed_rotation_key(12 * 2048)
    fixed_rotation_key_neg_8_2048 = engine_context.get_fixed_rotation_key(8 * 2048)
    fixed_rotation_key_neg_12_2048 = engine_context.get_fixed_rotation_key(4 * 2048)
    
    list_of_fixed_rotation_keys = [fixed_rotation_key_neg_4_2048, fixed_rotation_key_neg_8_2048, fixed_rotation_key_neg_12_2048]
    
    # ----------------------------------------------------------
    # -------------- 1. rotate 연산 ----------------------------
    # ----------------------------------------------------------
    # ct_hi 암호문 3개 생성    
    # 각각 -4, 8, 4 만큼 회전한 암호문 3개 생성
    ct_hi_rot_list = [engine.rotate(ct_hi, fixed_rotation_key_neg_4_2048), engine.rotate(ct_hi, fixed_rotation_key_neg_8_2048), engine.rotate(ct_hi, fixed_rotation_key_neg_12_2048)]
    
    # ct_lo 암호문 3개 생성    
    # 각각 -4, 8, 4 만큼 회전한 암호문 3개 생성
    ct_lo_rot_list = [engine.rotate(ct_lo, fixed_rotation_key_neg_4_2048), engine.rotate(ct_lo, fixed_rotation_key_neg_8_2048), engine.rotate(ct_lo, fixed_rotation_key_neg_12_2048)]
    
    # ----------------------------------------------------------
    # -------------- 2. variable naming convention --------------
    # ----------------------------------------------------------
    # gf_mul_2 연산을 오리지널 암호문에 대해 수행: level 5 소모 
    one_ct_hi, one_ct_lo = gf_mult_2(engine_context, ct_hi, ct_lo)    
    
    # gf_mul_3 연산을 -4 * 2048 만큼 왼쪽으로 회전한 암호문에 대해 수행: level 5 소모
    two_ct_hi, two_ct_lo = gf_mult_3(engine_context, ct_hi_rot_list[0], ct_lo_rot_list[0])
    
    three_ct_hi = ct_hi_rot_list[1] # -8 * 2048 만큼 왼쪽으로 회전한 암호문
    three_ct_lo = ct_lo_rot_list[1] # -8 * 2048 만큼 왼쪽으로 회전한 암호문
    
    four_ct_hi = ct_hi_rot_list[2] # -12 * 2048 만큼 왼쪽으로 회전한 암호문
    four_ct_lo = ct_lo_rot_list[2] # -12 * 2048 만큼 왼쪽으로 회전한 암호문
    

    # -------------------------------------------------------
    # -------------------- 3. XOR 연산 -----------------------
    # -------------------------------------------------------
    # 각 xor마다 level 5 감소
    
    # high nibble
    mixed_ct_hi = _xor_operation(engine_context, one_ct_hi, two_ct_hi) # level 5 감소
    
    # # bootstrap 연산 수행
    # mixed_ct_hi = engine.bootstrap(mixed_ct_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key()) # level 10 복귀
    
    mixed_ct_hi = _xor_operation(engine_context, mixed_ct_hi, three_ct_hi) # level 5 감소    
    mixed_ct_hi = _xor_operation(engine_context, mixed_ct_hi, four_ct_hi) # level 5 감소
        
    # low nibble
    mixed_ct_lo = _xor_operation(engine_context, one_ct_lo, two_ct_lo) # level 5 감소
    
    # Bootstrap 연산 수행
    mixed_ct_lo = engine.bootstrap(mixed_ct_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key()) # level 10 복귀
    
    mixed_ct_lo = _xor_operation(engine_context, mixed_ct_lo, three_ct_lo) # level 5 감소
    mixed_ct_lo = _xor_operation(engine_context, mixed_ct_lo, four_ct_lo) # level 5 감소
    
    # 전체 bootstrap 연산 수행 후 반환
    mixed_ct_hi = engine.bootstrap(mixed_ct_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key()) # level 10 복귀
    mixed_ct_lo = engine.bootstrap(mixed_ct_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key()) # level 10 복귀
    
    mixed_ct_hi = engine.intt(mixed_ct_hi)
    mixed_ct_lo = engine.intt(mixed_ct_lo)

    return mixed_ct_hi, mixed_ct_lo


from aes_transform_zeta import int_to_zeta, zeta_to_int
import time
from aes_split_to_nibble import split_to_nibbles

if __name__ == "__main__":
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
    
    print(int_array.shape)
    print(int_array[0], int_array[1 * 2048], int_array[2 * 2048], int_array[3 * 2048], int_array[4 * 2048], int_array[5 * 2048], int_array[6 * 2048], int_array[7 * 2048], int_array[8 * 2048], int_array[9 * 2048], int_array[10 * 2048], int_array[11 * 2048], int_array[12 * 2048], int_array[13 * 2048], int_array[14 * 2048], int_array[15 * 2048])
    
    print(int_array[1], int_array[1 * 2048 + 1], int_array[2 * 2048 + 1], int_array[3 * 2048 + 1], int_array[4 * 2048 + 1], int_array[5 * 2048 + 1], int_array[6 * 2048 + 1], int_array[7 * 2048 + 1], int_array[8 * 2048 + 1], int_array[9 * 2048 + 1], int_array[10 * 2048 + 1], int_array[11 * 2048 + 1], int_array[12 * 2048 + 1], int_array[13 * 2048 + 1], int_array[14 * 2048 + 1], int_array[15 * 2048 + 1])

    int_2d_array = int_array.reshape(16, 2048).T.reshape(-1, 4, 4)    
    print(int_2d_array.shape)
    print(int_2d_array[0, :, :])
    print(int_2d_array[1, :, :])

    
    def mix_columns_numpy(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.uint8)
        st = a.copy()
        def xtime(x):
            return (((x << 1) & 0xFF) ^ (((x >> 7) & 1) * 0x1B)).astype(np.uint8)
        s0, s1, s2, s3 = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
        tmp = s0 ^ s1 ^ s2 ^ s3   
        t0 = xtime(s0 ^ s1) ^ tmp ^ s0
        t1 = xtime(s1 ^ s2) ^ tmp ^ s1
        t2 = xtime(s2 ^ s3) ^ tmp ^ s2
        t3 = xtime(s3 ^ s0) ^ tmp ^ s3

        mixed = np.stack([t0, t1, t2, t3], axis=1).astype(np.uint8)
        return mixed       

    numpy_result = mix_columns_numpy(int_2d_array)
    
    
    # hi/lo nibble로 분할
    alpha_int, beta_int = split_to_nibbles(int_array)

    # Map to zeta domain
    alpha = int_to_zeta(alpha_int)
    beta  = int_to_zeta(beta_int)
    
    enc_alpha = engine.encrypt(alpha, public_key, level = 10)
    enc_beta = engine.encrypt(beta, public_key, level = 10)
    
    print("mix_columns")
    # 2. Evaluate MixColumns operation
    start_time = time.time()
    mixed_ct_hi, mixed_ct_lo = mix_columns(engine_context, enc_alpha, enc_beta)
    end_time = time.time()
    print(f"MixColumns time taken: {end_time - start_time} seconds")

    start_time = time.time()
    # 3. Decrypt result
    decoded_zeta = engine.decrypt(mixed_ct_hi, secret_key)
    decoded_int_hi = zeta_to_int(decoded_zeta)
    decoded_zeta_lo = engine.decrypt(mixed_ct_lo, secret_key)
    decoded_int_lo = zeta_to_int(decoded_zeta_lo)
    
    print(decoded_int_hi)
    print(decoded_int_lo)
    
    print(numpy_result[0, :, :])

    decoded_int = decoded_int_hi << 4 | decoded_int_lo

    # helper: return indices of k-th AES state inside flat vector (column-major layout)
    def state_indices(k: int) -> np.ndarray:
        base = k
        # positions are (col*4 + row) * 2048 + k
        offs = np.array([
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9, 10, 11,
            12, 13, 14, 15
        ]) * 2048
        return base + offs

    idx0 = state_indices(0)  # first state
    decoded_first = decoded_int[idx0]

    print("decoded first state:", decoded_first)
    print("numpy first state  :", numpy_result[0].reshape(-1))
    print("match mask         :", numpy_result[0].reshape(-1) == decoded_first)
    
    
    idx0 = state_indices(1)  # first state
    decoded_first = decoded_int[idx0]

    print("decoded first state:", decoded_first)
    print("numpy first state  :", numpy_result[1].reshape(-1))
    print("match mask         :", numpy_result[1].reshape(-1) == decoded_first)
    
    
    idx0 = state_indices(2)  # first state
    decoded_first = decoded_int[idx0]

    print("decoded first state:", decoded_first)
    print("numpy first state  :", numpy_result[2].reshape(-1))
    print("match mask         :", numpy_result[2].reshape(-1) == decoded_first)