from __future__ import annotations

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

import numpy as np
from typing import Tuple
import importlib.util
import pathlib
from engine_context import CKKS_EngineContext
from aes_SubBytes import sub_bytes

# -----------------------------------------------------------------------------
# Dynamic import helpers (copied from aes-main-process) ------------------------
# -----------------------------------------------------------------------------

_THIS_DIR = pathlib.Path(__file__).resolve().parent

AES_RCON = np.array([
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
], dtype=np.uint8)

aes_rcon_hi = np.array([
    0x0, 0x0, 0x0, 0x0, 0x1, 0x2, 0x4, 0x8, 0x1, 0x3
], dtype=np.uint8)
aes_rcon_lo = np.array([
    0x1, 0x2, 0x4, 0x8, 0x0, 0x0, 0x0, 0x0, 0xb, 0x6
], dtype=np.uint8)

def _load_module(fname: str, alias: str):
    """Load a Python file in the current directory as a module with *alias*."""
    path = _THIS_DIR / fname
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {fname}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

# Dynamically load `aes-xor.py` as module alias `aes_xor`
_aes_xor_mod = _load_module("aes_xor.py", "aes_xor")
_xor_operation = _aes_xor_mod._xor_operation


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
    public_key = engine_context.get_public_key()
    
    # ------------------------------Rotating------------------------------
    rotated_hi_bytes = engine.rotate(enc_key_hi, engine_context.get_fixed_rotation_key(4 * 2048))
    rotated_lo_bytes = engine.rotate(enc_key_lo, engine_context.get_fixed_rotation_key(4 * 2048))

    return rotated_hi_bytes, rotated_lo_bytes
    


def _sub_word(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    sub_bytes_hi, sub_bytes_lo = sub_bytes(engine_context, enc_key_hi, enc_key_lo)
    return sub_bytes_hi, sub_bytes_lo

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
    public_key = engine_context.get_public_key()
    
    max_blocks = 2048
    
    # Create Rcon mask - only first byte position
    first_byte_mask = np.concatenate([np.ones(max_blocks), np.zeros(15 * max_blocks)])
    
    # Get Rcon value and transform to zeta
    rcon_value = AES_RCON[round_num]
    rcon_hi = (rcon_value >> 4) & 0x0F
    rcon_lo = rcon_value & 0x0F
    
    # Transform to zeta representation
    rcon_zeta_hi = transform_to_zeta(np.full(16 * max_blocks, rcon_hi))
    rcon_zeta_lo = transform_to_zeta(np.full(16 * max_blocks, rcon_lo))
    
    # Apply mask to Rcon
    rcon_zeta_hi_masked = rcon_zeta_hi * first_byte_mask
    rcon_zeta_lo_masked = rcon_zeta_lo * first_byte_mask
    
    # Encode as plaintext
    rcon_plain_hi = engine.encode(rcon_zeta_hi_masked)
    rcon_plain_lo = engine.encode(rcon_zeta_lo_masked)
    
    # XOR with key using plaintext-ciphertext addition
    result_hi = _xor_operation(engine_context, enc_key_hi, engine.encrypt(rcon_plain_hi, public_key))
    result_lo = _xor_operation(engine_context, enc_key_lo, engine.encrypt(rcon_plain_lo, public_key))
    
    return result_hi, result_lo

def _xor(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo, xor_key_hi, xor_key_lo):
    return _xor_operation(engine_context, enc_key_hi, xor_key_hi), _xor_operation(engine_context, enc_key_lo, xor_key_lo)

def key_scheduling(engine_context, enc_key_hi, enc_key_lo):
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
    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()
    
    max_blocks = 2048
    
    # Initialize round keys list with original key as first round key
    word_hi = []
    word_lo = []
    
    # ------------------------------Masking------------------------------
    word_0_mask = np.concatenate((np.ones(4 * max_blocks), np.zeros(12 * max_blocks)))
    word_1_mask = np.concatenate((np.zeros(4 * max_blocks), np.ones(4 * max_blocks), np.zeros(8 * max_blocks)))
    word_2_mask = np.concatenate((np.zeros(8 * max_blocks), np.ones(4 * max_blocks), np.zeros(4 * max_blocks)))
    word_3_mask = np.concatenate((np.zeros(12 * max_blocks), np.ones(4 * max_blocks)))
    
    word_0_mask_plain = engine.encode(word_0_mask)
    word_1_mask_plain = engine.encode(word_1_mask)
    word_2_mask_plain = engine.encode(word_2_mask)
    word_3_mask_plain = engine.encode(word_3_mask)
    
    # ------------------------------Masking hi bytes------------------------------
    word_0_enc_hi = engine.multiply(enc_key_hi, word_0_mask_plain)
    word_1_enc_hi = engine.multiply(enc_key_hi, word_1_mask_plain)
    word_2_enc_hi = engine.multiply(enc_key_hi, word_2_mask_plain)
    word_3_enc_hi = engine.multiply(enc_key_hi, word_3_mask_plain)
    
    # ------------------------------Masking lo bytes------------------------------
    word_0_enc_lo = engine.multiply(enc_key_lo, word_0_mask_plain)
    word_1_enc_lo = engine.multiply(enc_key_lo, word_1_mask_plain)
    word_2_enc_lo = engine.multiply(enc_key_lo, word_2_mask_plain)
    word_3_enc_lo = engine.multiply(enc_key_lo, word_3_mask_plain)
    
    # ------------------------------append to word_hi and word_lo------------------------------
    word_hi.append(word_0_enc_hi)
    word_lo.append(word_0_enc_lo)
    word_hi.append(word_1_enc_hi)
    word_lo.append(word_1_enc_lo)
    word_hi.append(word_2_enc_hi)
    word_lo.append(word_2_enc_lo)
    word_hi.append(word_3_enc_hi)
    word_lo.append(word_3_enc_lo)
    
        # key scheduling round 
    for i in range(4, 44):
        if i % 4 == 0:
            # 4의 배수 - 1의 워드를 rot_word 연산 후 sub_word 연산 후 rcon_xor 연산 후 4의 배수 - 4 번째 워드와 xor 연산
            rot_word_hi, rot_word_lo = _rot_word(engine_context, word_hi[i-1], word_lo[i-1])
            sub_word_hi, sub_word_lo = _sub_word(engine_context, rot_word_hi, rot_word_lo)
            rcon_xor_hi, rcon_xor_lo = _rcon_xor(engine_context, sub_word_hi, sub_word_lo, i//4)
            xor_hi, xor_lo = _xor(engine_context, rcon_xor_hi, rcon_xor_lo, word_hi[i-4], word_lo[i-4])
            word_hi.append(xor_hi)
            word_lo.append(xor_lo)
        else:
            # 4의 배수 x - 4의 배수 -1 번째 워드와 4의 배수 - 3 번째 워드를 xor 연산
            xor_hi, xor_lo = _xor(engine_context, word_hi[i-1], word_lo[i-1], word_hi[i-4], word_lo[i-4])
            word_hi.append(xor_hi)
            word_lo.append(xor_lo)
            
    return word_hi, word_lo
    
    
    
if __name__ == "__main__":
    from aes_main_process import engine_initiation, key_initiation
    from aes_transform_zeta import zeta_to_int
    delta = [1 * 2048, 2 * 2048, 3 * 2048, 4 * 2048, 5 * 2048, 6 * 2048, 7 * 2048, 8 * 2048, 9 * 2048, 10 * 2048, 11 * 2048, 12 * 2048, 13 * 2048, 14 * 2048, 15 * 2048]
    engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0, fixed_rotation=True, delta_list=delta) 
    print(engine_context.get_slot_count())
    engine = engine_context.get_engine()
    
    _, _, key_upper, key_lower, key_zeta_upper, key_zeta_lower = key_initiation()
    
    max_blocks = 2048
    
    enc_key_hi = engine.encrypt(key_zeta_upper, engine_context.get_public_key())
    enc_key_lo = engine.encrypt(key_zeta_lower, engine_context.get_public_key())
    print(enc_key_hi)
    print(enc_key_lo)
    
    # ------------------------------Masking------------------------------
    word_0_mask = np.concatenate((np.ones(4 * max_blocks), np.zeros(12 * max_blocks)))
    word_1_mask = np.concatenate((np.zeros(4 * max_blocks), np.ones(4 * max_blocks), np.zeros(8 * max_blocks)))
    word_2_mask = np.concatenate((np.zeros(8 * max_blocks), np.ones(4 * max_blocks), np.zeros(4 * max_blocks)))
    word_3_mask = np.concatenate((np.zeros(12 * max_blocks), np.ones(4 * max_blocks)))
    
    word_0_mask_plain = engine.encode(word_0_mask)
    word_1_mask_plain = engine.encode(word_1_mask)
    word_2_mask_plain = engine.encode(word_2_mask)
    word_3_mask_plain = engine.encode(word_3_mask)
    
    # ------------------------------Masking hi bytes------------------------------
    word_0_enc_hi = engine.multiply(enc_key_hi, word_0_mask_plain)
    word_1_enc_hi = engine.multiply(enc_key_hi, word_1_mask_plain)
    word_2_enc_hi = engine.multiply(enc_key_hi, word_2_mask_plain)
    word_3_enc_hi = engine.multiply(enc_key_hi, word_3_mask_plain)
    
    # ------------------------------Masking lo bytes------------------------------
    word_0_enc_lo = engine.multiply(enc_key_lo, word_0_mask_plain)
    word_1_enc_lo = engine.multiply(enc_key_lo, word_1_mask_plain)
    word_2_enc_lo = engine.multiply(enc_key_lo, word_2_mask_plain)
    word_3_enc_lo = engine.multiply(enc_key_lo, word_3_mask_plain)
    
    word_0_dec_hi = engine.decrypt(word_0_enc_hi, engine_context.get_secret_key())
    word_0_dec_lo = engine.decrypt(word_0_enc_lo, engine_context.get_secret_key())
    word_1_dec_hi = engine.decrypt(word_1_enc_hi, engine_context.get_secret_key())
    word_1_dec_lo = engine.decrypt(word_1_enc_lo, engine_context.get_secret_key())
    word_2_dec_hi = engine.decrypt(word_2_enc_hi, engine_context.get_secret_key())
    word_2_dec_lo = engine.decrypt(word_2_enc_lo, engine_context.get_secret_key())
    word_3_dec_hi = engine.decrypt(word_3_enc_hi, engine_context.get_secret_key())
    word_3_dec_lo = engine.decrypt(word_3_enc_lo, engine_context.get_secret_key())
    
    word_0_int_hi = zeta_to_int(word_0_dec_hi)
    word_0_int_lo = zeta_to_int(word_0_dec_lo)
    word_1_int_hi = zeta_to_int(word_1_dec_hi)
    word_1_int_lo = zeta_to_int(word_1_dec_lo)
    word_2_int_hi = zeta_to_int(word_2_dec_hi)
    word_2_int_lo = zeta_to_int(word_2_dec_lo)
    word_3_int_hi = zeta_to_int(word_3_dec_hi)
    word_3_int_lo = zeta_to_int(word_3_dec_lo)
    
    print(word_0_int_hi)
    print(word_0_int_lo)
    print(word_1_int_hi)
    print(word_1_int_lo)
    print(word_2_int_hi)
    print(word_2_int_lo)
    print(word_3_int_hi)
    print(word_3_int_lo)
    
    
    
    