"""
aes-key-scheduling explanation

FHE Compatibility:
- No np.reshape() - maintains 1D array structure
- No direct indexing - uses boolean masking
- Preserves SIMD batching for multiple keys simultaneously

Key Scheduling Context:
- AES-128: 11 round keys (128-bit original + 10 derived keys)
- Key expansion uses RotWord on every 4th word
- Combined with SubWord (S-box) and round constant XOR

RotWord Operation - Step by Step:
1. Input: 4-byte word from AES key (e.g., [a0, a1, a2, a3])
2. Circular left shift by 1 byte: [a1, a2, a3, a0]
3. Concatenate rotated bytes and remained bytes

SubWord Operation - Step by Step:
1. Input: 4-byte word from AES key (e.g., [a0, a1, a2, a3])
2. Apply S-box substitution to each byte: [S(a0), S(a1), S(a2), S(a3), S(a4), S(a5), S(a6), S(a7), S(a8), S(a9), S(a10), S(a11), S(a12), S(a13), S(a14), S(a15)]
3. Concatenate substituted bytes

RconXOR Operation - Step by Step:

XOR operation - Step by Step:
"""

from __future__ import annotations
from aes_transform_zeta import int_to_zeta, zeta_to_int
import time
import numpy as np
from engine_context import CKKS_EngineContext
from aes_SubBytes import sub_bytes
from aes_xor import _xor_operation

# -----------------------------------------------------------------------------
# Dynamic import helpers (copied from aes-main-process) ------------------------
# -----------------------------------------------------------------------------

AES_RCON = np.array([
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
], dtype=np.uint8)

aes_rcon_hi = np.array([
    0x0, 0x0, 0x0, 0x0, 0x1, 0x2, 0x4, 0x8, 0x1, 0x3
], dtype=np.uint8)
aes_rcon_lo = np.array([
    0x1, 0x2, 0x4, 0x8, 0x0, 0x0, 0x0, 0x0, 0xb, 0x6
], dtype=np.uint8)

# Round constants for AES key expansion (Rcon)
AES_RCON = np.array([
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
], dtype=np.uint8)

__all__ = [
    "rot_word",
    "sub_word", 
    "rcon_xor",
    "aes_key_schedule",
    "generate_round_keys_flat"
]

# -----------------------------------------------------------------------------
# _rot_word ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _rot_word(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    """Apply RotWord operation to the first 4 word (bytes 0-3) of flat key array.
    
    SIMD batching된 키를 입력으로 받아, 첫 행의 4 bytes를 왼쪽으로 1 byte circular shift하는 연산을 취한 후 반환한다.
    
    Parameters
    ----------
    engine_context : CKKS_EngineContext
    enc_key_hi : Ciphertext
    enc_key_lo : Ciphertext
        
    Returns
    -------
    rotated_hi_bytes : Ciphertext
    rotated_lo_bytes : Ciphertext

    -------
    example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 13]
    """
    # load engine and keys
    engine = engine_context.get_engine()
    
    t_minus_1_word_hi = engine.clone(enc_key_hi)
    t_minus_1_word_lo = engine.clone(enc_key_lo)
    
    # ------------------------------Rotating------------------------------
    rotated_word_hi = engine.rotate(t_minus_1_word_hi, engine_context.get_fixed_rotation_key(4 * 2048))
    rotated_word_lo = engine.rotate(t_minus_1_word_lo, engine_context.get_fixed_rotation_key(4 * 2048))
    
    # ------------------------------Masking------------------------------
    key_mask_0_0 = np.concatenate([np.ones(1 * 2048), np.zeros(15 * 2048)])
    key_mask_0_123 = np.concatenate([np.zeros(1 * 2048), np.ones(3 * 2048), np.zeros(12 * 2048)])
    
    key_mask_hi_0_0 = engine.multiply(rotated_word_hi, key_mask_0_0)
    key_mask_hi_0_123 = engine.multiply(rotated_word_hi, key_mask_0_123)
    
    key_mask_lo_0_0 = engine.multiply(rotated_word_lo, key_mask_0_0)
    key_mask_lo_0_123 = engine.multiply(rotated_word_lo, key_mask_0_123)
    
    # ------------------------------Rotating------------------------------
    rot_key_0_3_hi = engine.rotate(key_mask_hi_0_0, engine_context.get_fixed_rotation_key(3 * 2048))
    rot_key_0_3_lo = engine.rotate(key_mask_lo_0_0, engine_context.get_fixed_rotation_key(3 * 2048))
    
    rot_key_123_012_hi = engine.rotate(key_mask_hi_0_123, engine_context.get_fixed_rotation_key(-1 * 2048))
    rot_key_123_012_lo = engine.rotate(key_mask_lo_0_123, engine_context.get_fixed_rotation_key(-1 * 2048))
    
    # ------------------------------Concatenating------------------------------
    rot_key_0_hi = engine.multiply(rot_key_0_3_hi, rot_key_123_012_hi, engine_context.get_relinearization_key())
    rot_key_0_lo = engine.multiply(rot_key_0_3_lo, rot_key_123_012_lo, engine_context.get_relinearization_key())
    
    # ------------------------------Bootstrap------------------------------
    rot_key_0_hi = engine.bootstrap(rot_key_0_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    rot_key_0_lo = engine.bootstrap(rot_key_0_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    return rot_key_0_hi, rot_key_0_lo

# -----------------------------------------------------------------------------
# _sub_word ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _sub_word(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    engine = engine_context.get_engine()
    
    masks = np.concatenate([np.ones(4 * 2048), np.zeros(12 * 2048)])
    
    sub_bytes_hi, sub_bytes_lo = sub_bytes(engine_context, enc_key_hi, enc_key_lo)
    
    # ------------------------------Masking------------------------------
    sub_bytes_hi = engine.multiply(sub_bytes_hi, masks)
    sub_bytes_lo = engine.multiply(sub_bytes_lo, masks)
    
    sub_bytes_hi = engine.intt(sub_bytes_hi)
    sub_bytes_lo = engine.intt(sub_bytes_lo)
    
    # ------------------------------Bootstrap------------------------------
    sub_bytes_hi = engine.bootstrap(sub_bytes_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    sub_bytes_lo = engine.bootstrap(sub_bytes_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    return sub_bytes_hi, sub_bytes_lo

# -----------------------------------------------------------------------------
# _rcon_xor ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _rcon_xor(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo, round_num: int):
    """Apply Rcon XOR to the first byte of the key.
    
    Parameters
    ----------
    engine_context : CKKS_EngineContext
    enc_key_hi : Ciphertext
    enc_key_lo : Ciphertext  
    round_num : int
        Round number (0-9) for selecting Rcon value
        
    Returns
    -------
    result_hi : Ciphertext
    result_lo : Ciphertext
    """
    engine = engine_context.get_engine()
    
    rcon_hi = AES_RCON[round_num]
    rcon_lo = AES_RCON[round_num]
    
    rcon_xor_hi = _xor_operation(engine_context, enc_key_hi, rcon_hi)
    rcon_xor_lo = _xor_operation(engine_context, enc_key_lo, rcon_lo)
    
    rcon_xor_hi = engine.bootstrap(rcon_xor_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    rcon_xor_lo = engine.bootstrap(rcon_xor_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    rcon_xor_hi = engine.intt(rcon_xor_hi)
    rcon_xor_lo = engine.intt(rcon_xor_lo)
    
    return rcon_xor_hi, rcon_xor_lo

# -----------------------------------------------------------------------------
# _xor -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _xor(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo, xor_key_hi, xor_key_lo):
    engine = engine_context.get_engine()
    
    xor_hi, xor_lo = _xor_operation(engine_context, enc_key_hi, xor_key_hi), _xor_operation(engine_context, enc_key_lo, xor_key_lo)
    
    xor_hi = engine.bootstrap(xor_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    xor_lo = engine.bootstrap(xor_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    xor_hi = engine.intt(xor_hi)
    xor_lo = engine.intt(xor_lo)
    
    return xor_hi, xor_lo

# -----------------------------------------------------------------------------
# key_scheduling --------------------------------------------------------------
# -----------------------------------------------------------------------------

def key_scheduling(engine_context, enc_key_hi_list, enc_key_lo_list):
    """
    AES key scheduling algorithm
    
    입력으로 들어오는 키를 original key로 하여, 11개의 round key를 생성하는 알고리즘이다.
    
    round key는 4개의 워드, 16 bytes로 구성된다.
    
    original key, 4 round key, 8 round key, 11 round key를 생성할 때는 rot_word, sub_word, rcon_xor, xor 연산을 사용한다.
    	
    - W[0] ~ W[3]: 입력 키 그대로 사용
	- W[4] = RotWord(SubWord(W[3])) ⊕ Rcon[1] ⊕ W[0]
	- W[5] = W[4] ⊕ W[1]
	- W[6] = W[5] ⊕ W[2]
	- W[7] = W[6] ⊕ W[3]
	- W[8] = RotWord(SubWord(W[7])) ⊕ Rcon[2] ⊕ W[4]
	- W[9] = W[8] ⊕ W[5]
	- W[10] = W[9] ⊕ W[6]
	- W[11] = W[10] ⊕ W[7]
	- … 반복해서 …
	- W[43] = W[42] ⊕ W[39]
    
    Parameters
    ----------
    engine_context : CKKS_EngineContext
    enc_key_hi : Ciphertext
    enc_key_lo : Ciphertext
    
    Dependencies Analysis:
    W[4] = W[0] ⊕ f(W[3]) ⊕ Rcon[1]   ← W[3]에 의존
    W[5] = W[1] ⊕ W[4]                ← W[4]에 의존  
    W[6] = W[2] ⊕ W[5]                ← W[5]에 의존
    W[7] = W[3] ⊕ W[6]                ← W[6]에 의존
    
    W[0]  W[1]  W[2]  W[3]  ← 초기 키 (병렬 가능)
    │     │     │     │
    │     │     │   ┌─┴─┐
    │     │     │   │f()│   ← SubWord(RotWord) + Rcon
    │     │     │   └─┬─┘
    └─────┼─────┼─────┴──→ W[4]
          │     │           │
          └─────┼───────────┴──→ W[5]
                │                │  
                └────────────────┴──→ W[6]


    Returns
    -------
    round_keys_hi : list[Ciphertext]
    round_keys_lo : list[Ciphertext]

    특이사항
    ------
    - 키 확장 과정은 모든 워드가 이전 워드에 의존하기 때문에 병렬이 불가능하다.
    - 따라서 모든 키를 처리시 하나의 워드 씩 분리하여 처리한다. 즉 44개의 암호문을 생성하고 처리하게 될 것이다.
    - 이 과정에서 xor 연산 시 레벨이 5씩 감소하기 때문에 xor 처리하기 전 레벨이 5 미만이라면 부트스트랩을 통해 레벨을 10으로 만들어준다.
    """
    word_hi = enc_key_hi_list.copy()
    word_lo = enc_key_lo_list.copy()
    
        # key scheduling round 
    for i in range(4, 44):
        if i % 4 == 0:
            # 4의 배수 - 1의 워드를 rot_word 연산 후 sub_word 연산 후 rcon_xor 연산 후 4의 배수 - 4 번째 워드와 xor 연산
            start_time = time.time()
            rot_word_hi, rot_word_lo = _rot_word(engine_context, word_hi[i-1], word_lo[i-1])
            sub_word_hi, sub_word_lo = _sub_word(engine_context, rot_word_hi, rot_word_lo)
            rcon_xor_hi, rcon_xor_lo = _rcon_xor(engine_context, sub_word_hi, sub_word_lo, i//4)
            xor_hi, xor_lo = _xor(engine_context, rcon_xor_hi, rcon_xor_lo, word_hi[i-4], word_lo[i-4])
            word_hi.append(xor_hi)
            word_lo.append(xor_lo)
            end_time = time.time()
            print(f"Key scheduling round {i} time: {end_time - start_time} seconds")
        else:
            start_time = time.time()
            # 4의 배수 x - 4의 배수 -1 번째 워드와 4의 배수 - 3 번째 워드를 xor 연산
            word_hi_i_minus_1 = engine.rotate(word_hi[i-1], engine_context.get_fixed_rotation_key(4 * 2048))
            word_lo_i_minus_1 = engine.rotate(word_lo[i-1], engine_context.get_fixed_rotation_key(4 * 2048))
            word_hi_i_minus_3 = word_hi[i-4]
            word_lo_i_minus_3 = word_lo[i-4]
            
            xor_hi, xor_lo = _xor(engine_context, word_hi_i_minus_1, word_lo_i_minus_1, word_hi_i_minus_3, word_lo_i_minus_3)
            word_hi.append(xor_hi)
            word_lo.append(xor_lo)
            end_time = time.time()
            print(f"Key scheduling round {i} time: {end_time - start_time} seconds")
    return word_hi, word_lo
    
# -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    from aes_main_process import engine_initiation, key_initiation_fixed
    from aes_transform_zeta import zeta_to_int
    delta = [1 * 2048, 2 * 2048, 3 * 2048, 4 * 2048, 5 * 2048, 6 * 2048, 7 * 2048, 8 * 2048, 9 * 2048, 10 * 2048, 11 * 2048, 12 * 2048, 13 * 2048, 14 * 2048, 15 * 2048]
    engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0, fixed_rotation=True, delta_list=delta) 
    print(engine_context.get_slot_count())
    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()
    
    key_zeta_hi, key_zeta_lo = key_initiation_fixed()
    
    key_zeta_hi = engine.encrypt(key_zeta_hi, public_key, level=10)
    key_zeta_lo = engine.encrypt(key_zeta_lo, public_key, level=10)
    
    enc_key_hi_list = []
    enc_key_lo_list = []
    
    mask_row_0 = np.concatenate((np.ones(4 * 2048), np.zeros(12 * 2048)))
    mask_row_1 = np.concatenate((np.zeros(4 * 2048), np.ones(4 * 2048), np.zeros(8 * 2048)))
    mask_row_2 = np.concatenate((np.zeros(8 * 2048), np.ones(4 * 2048), np.zeros(4 * 2048)))
    mask_row_3 = np.concatenate((np.zeros(12 * 2048), np.ones(4 * 2048))) 
    
    row_hi_0 = engine.multiply(key_zeta_hi, mask_row_0)
    row_hi_1 = engine.multiply(key_zeta_hi, mask_row_1)
    row_hi_2 = engine.multiply(key_zeta_hi, mask_row_2)
    row_hi_3 = engine.multiply(key_zeta_hi, mask_row_3)
    
    row_lo_0 = engine.multiply(key_zeta_lo, mask_row_0)
    row_lo_1 = engine.multiply(key_zeta_lo, mask_row_1)
    row_lo_2 = engine.multiply(key_zeta_lo, mask_row_2)
    row_lo_3 = engine.multiply(key_zeta_lo, mask_row_3)

    enc_key_hi_list.append(row_hi_0)
    enc_key_hi_list.append(row_hi_1)
    enc_key_hi_list.append(row_hi_2)
    enc_key_hi_list.append(row_hi_3)
    
    enc_key_lo_list.append(row_lo_3)
    enc_key_lo_list.append(row_lo_0)
    enc_key_lo_list.append(row_lo_1)
    enc_key_lo_list.append(row_lo_2)
    
    print("Key scheduling start")
    start_time = time.time()
    key_hi_list, key_lo_list = key_scheduling(engine_context, enc_key_hi_list, enc_key_lo_list)
    end_time = time.time()
    print(f"Key scheduling time: {end_time - start_time} seconds")
    
    print()
    
    
    
    
    
    