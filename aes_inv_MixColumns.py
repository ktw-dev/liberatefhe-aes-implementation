"""aes-MixColumns.py

MixColumns operation for AES-128 ECB mode with FHE compatibility.

The MixColumns operation treats each column of the AES state as a polynomial
and multiplies it by a fixed polynomial in GF(2^8). This provides diffusion
in the AES cipher.

Each column is transformed by multiplying with the matrix:
[14 11 13 09]
[09 14 11 13]  
[13 09 14 11]
[11 13 09 14]

Multiplication in GF(2^8) is implemented using lookup tables and XOR operations.
"""
from __future__ import annotations

import numpy as np
from engine_context import CKKS_EngineContext
from typing import Any
from aes_gf_mult import gf_mult_9, gf_mult_11, gf_mult_13, gf_mult_14
from aes_xor import _xor_operation

__all__ = [
    "inv_mix_columns",
]

def inv_mix_columns(engine_context: CKKS_EngineContext, ct_hi: Any, ct_lo: Any):
    """두 개의 블록을 받아 inverse MixColumns 연산을 수행하고 결과를 반환   
    Args:
        engine_context: CKKS_EngineContext
        ct_hi: 첫 번째 블록
        ct_lo: 두 번째 블록
        
    Returns:
        ct_hi_out: 첫 번째 블록의 결과
        ct_lo_out: 두 번째 블록의 결과
        
    ------------------------------------------------------------
    functions:
        이 연산은 두 개의 암호문(hi_nibble of ct, low_nibble of ct)을 받아 inverse MixColumns 연산을 수행하고 반환한다.
        
        rotate_batch를 사용하여 각각 -4 * 2048, 8 * 2048, 4 * 2048 만큼 회전한 암호문을 3개 생성한다.
        original, -4, 8, 4 순서로 각각 one_ct, two_ct, three_ct, four_ct 라고 명명한다.
        
        one_ct는 gf_mul_2 연산을 수행하고, two_ct는 gf_mul_3 연산을 수행한다.
        three_ct와 four_ct는 별도의 연산을 수행하지 않는다.
        
        이후 모든 암호문들에 대해 XOR 연산을 수행하고, 결과를 반환하면 연산이 완료된다.
        
    """
    engine = engine_context.get_engine()
    
    fixed_rotation_key_neg_4_2048 = engine_context.get_fixed_rotation_key(-4 * 2048)
    fixed_rotation_key_neg_8_2048 = engine_context.get_fixed_rotation_key(-8 * 2048)
    fixed_rotation_key_neg_12_2048 = engine_context.get_fixed_rotation_key(-12 * 2048)
    
    list_of_fixed_rotation_keys = [fixed_rotation_key_neg_4_2048, fixed_rotation_key_neg_8_2048, fixed_rotation_key_neg_12_2048]
    
    # ----------------------------------------------------------
    # -------------- 1. rotate 연산 ----------------------------
    # ----------------------------------------------------------
    # ct_hi 암호문 3개 생성    
    # 각각 4, 8, 12 만큼 회전한 암호문 3개 생성
    ct_hi_rot_list = engine.rotate_batch(ct_hi, list_of_fixed_rotation_keys)
    
    # ct_lo 암호문 3개 생성    
    # 각각 4, 8, 12 만큼 회전한 암호문 3개 생성
    ct_lo_rot_list = engine.rotate_batch(ct_lo, list_of_fixed_rotation_keys)
    
    # ----------------------------------------------------------
    # -------------- 2. variable naming convention --------------
    # ----------------------------------------------------------
    # gf_mul_2 연산을 오리지널 암호문에 대해 수행: level 5 소모 
    one_ct_hi, one_ct_lo = gf_mult_14(engine_context, ct_hi, ct_lo)    
    
    # gf_mul_3 연산을 4 * 2048 만큼 왼쪽으로 회전한 암호문에 대해 수행: level 10 소모
    two_ct_hi, two_ct_lo = gf_mult_11(engine_context, ct_hi_rot_list[0], ct_lo_rot_list[0])
    
    three_ct_hi, three_ct_lo = gf_mult_13(engine_context, ct_hi_rot_list[1], ct_lo_rot_list[1]) # -8 * 2048 만큼 왼쪽으로 회전한 암호문
    
    
    four_ct_hi, four_ct_lo = gf_mult_9(engine_context, ct_hi_rot_list[2], ct_lo_rot_list[2]) # -12 * 2048 만큼 왼쪽으로 회전한 암호문
    
    
    # -------------------------------------------------------
    # ------------------ 3. Bootstrap 연산 -------------------
    # -------------------------------------------------------
    one_ct_hi_bootstrap = engine.bootstrap(one_ct_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    one_ct_lo_bootstrap = engine.bootstrap(one_ct_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    two_ct_hi_bootstrap = engine.bootstrap(two_ct_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    two_ct_lo_bootstrap = engine.bootstrap(two_ct_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    # DEBUG
    print(f"one_ct_hi_bootstrap.level: {one_ct_hi_bootstrap.level}")
    print(f"two_ct_hi_bootstrap.level: {two_ct_hi_bootstrap.level}")
    print(f"three_ct_hi.level: {three_ct_hi.level}")
    print(f"four_ct_hi.level: {four_ct_hi.level}")
    
    # -------------------------------------------------------
    # -------------------- 3. XOR 연산 -----------------------
    # -------------------------------------------------------
    # 각 xor마다 level 5 감소
    
    # high nibble
    mixed_ct_hi = _xor_operation(engine_context, one_ct_hi_bootstrap, two_ct_hi_bootstrap)
    mixed_ct_hi = _xor_operation(engine_context, mixed_ct_hi, three_ct_hi)
    
    # Bootstrap 연산 수행
    mixed_ct_hi = engine.bootstrap(mixed_ct_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    mixed_ct_hi = _xor_operation(engine_context, mixed_ct_hi, four_ct_hi)
        
        
    # low nibble
    mixed_ct_lo = _xor_operation(engine_context, one_ct_lo_bootstrap, two_ct_lo_bootstrap)
    mixed_ct_lo = _xor_operation(engine_context, mixed_ct_lo, three_ct_lo)
    
    # Bootstrap 연산 수행
    mixed_ct_lo = engine.bootstrap(mixed_ct_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    mixed_ct_lo = _xor_operation(engine_context, mixed_ct_lo, four_ct_lo)
    
    # DEBUG
    print(f"mixed_ct_hi.level: {mixed_ct_hi.level}")
    print(f"mixed_ct_lo.level: {mixed_ct_lo.level}")
    
    return mixed_ct_hi, mixed_ct_lo


from aes_transform_zeta import int_to_zeta, zeta_to_int
import time
from aes_split_to_nibble import split_to_nibbles

if __name__ == "__main__":
    engine_context = CKKS_EngineContext(signature=1, use_bootstrap=True, mode="parallel", thread_count=8, device_id=0)
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

    
    def inv_mix_columns_numpy(arr: np.ndarray) -> np.ndarray:
        """Vectorised inverse MixColumns (NumPy).

        Parameters
        ----------
        arr : np.ndarray
            Array with shape (N, 4, 4) and dtype uint8. Any leading dimension
            *N* represents the number of AES states.
        Returns
        -------
        np.ndarray
            Array of same shape/type with inverse MixColumns applied.
        """

        a = np.asarray(arr, dtype=np.uint8)
        if a.shape[-2:] != (4, 4):
            raise ValueError("input last two dims must be (4,4)")

        st = a.copy()
        from aes_gf_mult import gf_mult_lookup
        
        # For each column apply inverse MixColumns formula using lookup tables
        for col in range(4):
            s0 = st[:, 0, col]
            s1 = st[:, 1, col]
            s2 = st[:, 2, col]
            s3 = st[:, 3, col]

            st[:, 0, col] = (
                gf_mult_lookup(s0, 14) ^ gf_mult_lookup(s1, 11) ^ gf_mult_lookup(s2, 13) ^ gf_mult_lookup(s3, 9)
            )
            st[:, 1, col] = (
                gf_mult_lookup(s0, 9) ^ gf_mult_lookup(s1, 14) ^ gf_mult_lookup(s2, 11) ^ gf_mult_lookup(s3, 13)
            )
            st[:, 2, col] = (
                gf_mult_lookup(s0, 13) ^ gf_mult_lookup(s1, 9) ^ gf_mult_lookup(s2, 14) ^ gf_mult_lookup(s3, 11)
            )
            st[:, 3, col] = (
                gf_mult_lookup(s0, 11) ^ gf_mult_lookup(s1, 13) ^ gf_mult_lookup(s2, 9) ^ gf_mult_lookup(s3, 14)
            )

        return st
        
    numpy_result = inv_mix_columns_numpy(int_2d_array)
    
    
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
    mixed_ct_hi, mixed_ct_lo = inv_mix_columns(engine_context, enc_alpha, enc_beta)
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