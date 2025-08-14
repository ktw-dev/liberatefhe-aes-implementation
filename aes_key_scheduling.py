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

# verification
from aes_ground_truth import WI_HEX_I4_TO_I43

# -----------------------------------------------------------------------------
# Table -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# AES_RCON = np.array([
#     0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
# ], dtype=np.uint8)

aes_rcon_hi = np.array([
    0x0, 0x0, 0x0, 0x0, 0x1, 0x2, 0x4, 0x8, 0x1, 0x3
], dtype=np.uint8)
aes_rcon_lo = np.array([
    0x1, 0x2, 0x4, 0x8, 0x0, 0x0, 0x0, 0x0, 0xb, 0x6
], dtype=np.uint8)

masking_container = {
    "row_0": np.concatenate([np.ones(4 * 2048), np.zeros(12 * 2048)]),
    "row_1": np.concatenate([np.zeros(4 * 2048), np.ones(4 * 2048), np.zeros(8 * 2048)]),
    "row_2": np.concatenate([np.zeros(8 * 2048), np.ones(4 * 2048), np.zeros(4 * 2048)]),
    "row_3": np.concatenate([np.zeros(12 * 2048), np.ones(4 * 2048)]),
    "row_0_0": np.concatenate([np.ones(1 * 2048), np.zeros(15 * 2048)]),
    "row_0_123": np.concatenate([np.zeros(1 * 2048), np.ones(3 * 2048), np.zeros(12 * 2048)])
}

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
    """Apply RotWord operation to the specific 4 word (bytes 0-3) of flat key array.
    
    SIMD batching된 키를 입력으로 받아, 특정 4 bytes를 왼쪽으로 1 byte circular shift하는 연산을 취한 후 반환한다.
    
    Parameters
    ----------
    engine_context : CKKS_EngineContext
    enc_key_hi : Ciphertext
    enc_key_lo : Ciphertext
        
    Returns
    -------
    rotated_hi_bytes : Ciphertext
    rotated_lo_bytes : Ciphertext
    
    function:
        1. 입력으로 받은 키 nibble에 대해 오른쪽으로 4 * 2048 만큼 회전시킨다. 이 과정을 통해 인덱스 12 * 2048 부터 16 * 2048 - 1 까지의 원소들이 인덱스 0 부터 4 * 2048 - 1 까지의 원소들로 이동한다.
        2. 회전된 키 nibble에 대해 마스킹을 적용한다. 이 과정을 통해 0번;index 0 to 1 * 2048 - 1 과 123번;index 1 * 2048 to 4 * 2048 - 1 의 원소들을 분리할 수 있다.
        3. 분리된 키 0번에 대해 오른쪽으로 3 * 2048 만큼 회전시키고 123번 키에 대해 왼쪽으로 1 * 2048 만큼 회전시킨다. 그 결과 circular left shift 연산이 작동한다.
        4. 마지막으로 두 키를 재조합하여 반환한다. 이때 engine.multiply() 연산을 사용한다. 그 결과 rotation word가 완성된다
    """
    # load engine and keys
    engine = engine_context.get_engine()
    
    t_minus_1_word_hi = engine.clone(enc_key_hi)
    t_minus_1_word_lo = engine.clone(enc_key_lo)
    
    print(_extract_bytes_hex(engine_context, t_minus_1_word_hi, t_minus_1_word_lo))

    # ------------------------------Rotating------------------------------
    rotated_word_hi = engine.rotate(t_minus_1_word_hi, engine_context.get_fixed_rotation_key(4 * 2048))
    rotated_word_lo = engine.rotate(t_minus_1_word_lo, engine_context.get_fixed_rotation_key(4 * 2048))
    
    # ------------------------------Masking------------------------------
    rotated_word_hi_0 = engine.multiply(rotated_word_hi, masking_container["row_0_0"])
    rotated_word_lo_0 = engine.multiply(rotated_word_lo, masking_container["row_0_0"])
    
    rotated_word_hi_123 = engine.multiply(rotated_word_hi, masking_container["row_0_123"])
    rotated_word_lo_123 = engine.multiply(rotated_word_lo, masking_container["row_0_123"])
    
    # ------------------------------Intt------------------------------
    rotated_word_hi_0 = engine.intt(rotated_word_hi_0)
    rotated_word_lo_0 = engine.intt(rotated_word_lo_0)
    
    rotated_word_hi_123 = engine.intt(rotated_word_hi_123)
    rotated_word_lo_123 = engine.intt(rotated_word_lo_123)

    # ------------------------------Rotating------------------------------
    rotated_word_hi_0_to_3 = engine.rotate(rotated_word_hi_0, engine_context.get_fixed_rotation_key(3 * 2048))
    rotated_word_lo_0_to_3 = engine.rotate(rotated_word_lo_0, engine_context.get_fixed_rotation_key(3 * 2048))
    
    rotated_word_hi_123_to_012 = engine.rotate(rotated_word_hi_123, engine_context.get_fixed_rotation_key(-1 * 2048))
    rotated_word_lo_123_to_012 = engine.rotate(rotated_word_lo_123, engine_context.get_fixed_rotation_key(-1 * 2048))
    
    # ------------------------------Concatenating------------------------------
    rotated_word_hi = engine.multiply(rotated_word_hi_0_to_3, rotated_word_hi_123_to_012, engine_context.get_relinearization_key())
    rotated_word_lo = engine.multiply(rotated_word_lo_0_to_3, rotated_word_lo_123_to_012, engine_context.get_relinearization_key())
    
    # ------------------------------Bootstrap------------------------------
    rotated_word_hi = engine.bootstrap(rotated_word_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    rotated_word_lo = engine.bootstrap(rotated_word_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    # ------------------------------Intt------------------------------
    rotated_word_hi = engine.intt(rotated_word_hi)
    rotated_word_lo = engine.intt(rotated_word_lo)
    
    return rotated_word_hi, rotated_word_lo

# -----------------------------------------------------------------------------
# _sub_word -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _sub_word(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    """Apply SubWord operation to the specific 4 word (bytes 0-3) of flat key array.
    
    Parameters
    ----------
    engine_context : CKKS_EngineContext
    enc_key_hi : Ciphertext
    enc_key_lo : Ciphertext
    
    Returns
    -------
    sub_word_hi : Ciphertext
    sub_word_lo : Ciphertext
    
    function:
        1. 입력으로 받은 키 nibble에 대해 sub bytes 연산을 취한다.
        2. 마스킹을 적용하여 0123번 키만 남기고 나머지는 0으로 만든다.
    """ 
    engine = engine_context.get_engine()
    
    # ------------------------------Sub Bytes------------------------------
    sub_bytes_hi, sub_bytes_lo = sub_bytes(engine_context, enc_key_hi, enc_key_lo)
    
    # ------------------------------Concatenating------------------------------
    sub_bytes_hi = engine.multiply(sub_bytes_hi, masking_container["row_0"])
    sub_bytes_lo = engine.multiply(sub_bytes_lo, masking_container["row_0"])
    
    sub_bytes_hi = engine.intt(sub_bytes_hi)
    sub_bytes_lo = engine.intt(sub_bytes_lo)
    
    # ------------------------------Bootstrap------------------------------
    sub_bytes_hi = engine.bootstrap(sub_bytes_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    sub_bytes_lo = engine.bootstrap(sub_bytes_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    # ------------------------------Intt------------------------------
    sub_bytes_hi = engine.intt(sub_bytes_hi)
    sub_bytes_lo = engine.intt(sub_bytes_lo)
    
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
    
    function:
        1. 입력으로 받은 키 nibble에 대해 rcon xor 연산을 취한다.
    """
    engine = engine_context.get_engine()
    current_level = enc_key_hi.level
    
    rcon_index = (round_num // 4) - 1
    
    # ------------------------------Rcon Indexing------------------------------
    rcon_hi = np.repeat(aes_rcon_hi[rcon_index], 2048)
    rcon_lo = np.repeat(aes_rcon_lo[rcon_index], 2048)
    
    rcon_hi = int_to_zeta(rcon_hi)
    rcon_lo = int_to_zeta(rcon_lo)
    
    rcon_hi_encrypted = engine.encrypt(rcon_hi, engine_context.get_public_key(), current_level)
    rcon_lo_encrypted = engine.encrypt(rcon_lo, engine_context.get_public_key(), current_level)
    
    # ------------------------------XOR------------------------------
    rcon_xor_hi = _xor_operation(engine_context, enc_key_hi, rcon_hi_encrypted)
    rcon_xor_lo = _xor_operation(engine_context, enc_key_lo, rcon_lo_encrypted)
    
    # ------------------------------Bootstrap------------------------------
    rcon_xor_hi = engine.bootstrap(rcon_xor_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    rcon_xor_lo = engine.bootstrap(rcon_xor_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    # ------------------------------Intt------------------------------
    rcon_xor_hi = engine.intt(rcon_xor_hi)
    rcon_xor_lo = engine.intt(rcon_xor_lo)
    
    return rcon_xor_hi, rcon_xor_lo

# -----------------------------------------------------------------------------
# _xor -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _xor(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo, xor_key_hi, xor_key_lo):
    engine = engine_context.get_engine()
    
    # ------------------------------XOR------------------------------
    xor_hi = _xor_operation(engine_context, enc_key_hi, xor_key_hi)
    xor_lo = _xor_operation(engine_context, enc_key_lo, xor_key_lo)
    
    # ------------------------------Bootstrap------------------------------
    xor_hi = engine.bootstrap(xor_hi, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    xor_lo = engine.bootstrap(xor_lo, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    # ------------------------------Intt------------------------------
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
            print(_extract_bytes_hex(engine_context, rot_word_hi, rot_word_lo))
            sub_word_hi, sub_word_lo = _sub_word(engine_context, rot_word_hi, rot_word_lo)
            print(_extract_bytes_hex(engine_context, sub_word_hi, sub_word_lo))
            rcon_xor_hi, rcon_xor_lo = _rcon_xor(engine_context, sub_word_hi, sub_word_lo, i)
            print(_extract_bytes_hex(engine_context, rcon_xor_hi, rcon_xor_lo))
            xor_hi, xor_lo = _xor(engine_context, rcon_xor_hi, rcon_xor_lo, word_hi[i-4], word_lo[i-4])
            word_hi.append(xor_hi)
            word_lo.append(xor_lo)

            # ---- Verification against ground truth ----------------------
            gt_hex = WI_HEX_I4_TO_I43[i - 4]
            got_hex = _extract_word_hex(engine_context, xor_hi, xor_lo)
            print(f"W[{i}] match: {got_hex == gt_hex} (got={got_hex}, gt={gt_hex})")
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

            # ---- Verification ------------------------------------------
            gt_hex = WI_HEX_I4_TO_I43[i - 4]
            got_hex = _extract_word_hex(engine_context, xor_hi, xor_lo)
            print(f"W[{i}] match: {got_hex == gt_hex} (got={got_hex}, gt={gt_hex})")
            end_time = time.time()
            print(f"Key scheduling round {i} time: {end_time - start_time} seconds")
    return word_hi, word_lo


# -----------------------------------------------------------------------------
# Verification helpers ---------------------------------------------------------
# -----------------------------------------------------------------------------
def _extract_word_hex(engine_context: CKKS_EngineContext, ct_hi, ct_lo) -> str:
    """Decrypt hi/lo nibble ciphertexts and return the first 4 bytes as hex.

    Assumes each byte occupies one 2048-slot block, repeated within the block.
    Takes indices 2048 * i for i = 0..3 (the leading 4 blocks) and formats as
    an 8-hex-digit lowercase string (big-endian byte order).
    """
    engine = engine_context.get_engine()
    sk = engine_context.get_secret_key()

    dec_hi = engine.decrypt(ct_hi, sk)
    dec_lo = engine.decrypt(ct_lo, sk)

    # Map zeta to integers 0..15, and zero-out tiny numerical noise
    nib_hi = zeta_to_int(dec_hi).astype(np.uint8)
    nib_lo = zeta_to_int(dec_lo).astype(np.uint8)

    # Indices for the first element of each 2048-slot block
    indices = np.arange(0, 16 * 2048, 2048)

    # Extract one representative nibble per block
    hi_blocks = nib_hi[indices].astype(np.uint16)
    lo_blocks = nib_lo[indices].astype(np.uint16)

    # Combine nibbles into bytes
    bytes_arr = ((hi_blocks << 4) | lo_blocks).astype(np.uint8)

    # Take the first 4 bytes (row_0 window)
    word_bytes = bytes_arr[0:4]
    hex_str = "".join(f"{b:02x}" for b in word_bytes.tolist())

    return hex_str


def _extract_bytes_hex(engine_context: CKKS_EngineContext, ct_hi, ct_lo):
    """Decrypt hi/lo ciphertexts and return 16 bytes as hex strings array.

    Selects indices 2048 * i for i = 0..15 and combines hi/lo nibbles to bytes.
    Returns a Python list of 16 two-digit lowercase hex strings.
    """
    engine = engine_context.get_engine()
    sk = engine_context.get_secret_key()

    dec_hi = engine.decrypt(ct_hi, sk)
    dec_lo = engine.decrypt(ct_lo, sk)

    nib_hi = zeta_to_int(dec_hi).astype(np.uint8)
    nib_lo = zeta_to_int(dec_lo).astype(np.uint8)

    indices = np.arange(0, 16 * 2048, 2048)
    hi_blocks = nib_hi[indices].astype(np.uint16)
    lo_blocks = nib_lo[indices].astype(np.uint16)
    bytes_arr = ((hi_blocks << 4) | lo_blocks).astype(np.uint8)

    return [f"{b:02x}" for b in bytes_arr.tolist()]


# -----------------------------------------------------------------------------
def key_initiation_fixed():
    byte_array = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    int_array = np.frombuffer(byte_array, dtype=np.uint8)
    int_array = int_array.reshape(4,4).T
    int_array = int_array.flatten()
    int_array = np.repeat(int_array, 2048) # 각 원소가 2048번씩 반복되어 나타난다
    
    int_array_hi = ((int_array >> 4) & 0x0F).astype(np.uint8)
    int_array_lo = (int_array & 0x0F).astype(np.uint8)
    
    zeta_array_hi = int_to_zeta(int_array_hi)
    zeta_array_lo = int_to_zeta(int_array_lo)
        
    return zeta_array_hi, zeta_array_lo


if __name__ == "__main__":
    from aes_main_process import engine_initiation
    from aes_transform_zeta import zeta_to_int
    delta = [1 * 2048, 2 * 2048, 3 * 2048, 4 * 2048, 5 * 2048, 6 * 2048, 7 * 2048, 8 * 2048, 9 * 2048, 10 * 2048, 11 * 2048, 12 * 2048, 13 * 2048, 14 * 2048, 15 * 2048]
    engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0, fixed_rotation=True, delta_list=delta) 
    print("slot_count: ", engine_context.get_slot_count())
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
    
    
    
    
    
    