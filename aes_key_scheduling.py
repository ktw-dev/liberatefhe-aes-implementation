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
    example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] -> [2, 3, 4, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    """
    # load engine and keys
    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()
    secret_key = engine_context.get_secret_key()
    relin_key = engine_context.get_relinearization_key()
    fixed_rotation_key_neg_2048 = engine_context.get_fixed_rotation_key(-2048)
    fixed_rotation_key_3_2048 = engine_context.get_fixed_rotation_key(3 * 2048)
    
    # ------------------------------load masks------------------------------
    max_blocks = 2048
    first_bytes_mask = np.concatenate([np.ones(max_blocks), np.zeros(15 * max_blocks)])
    secondtofourth_bytes_mask = np.concatenate([np.zeros(max_blocks), np.ones(3 * max_blocks), np.zeros(12 * max_blocks)])
    remained_bytes_mask = np.concatenate([np.zeros(4 * max_blocks), np.ones(12 * max_blocks)])
    
    # ------------------------------transform mask to plaintext------------------------------
    first_bytes_mask_plain = engine.encode(first_bytes_mask, public_key)
    secondtofourth_bytes_mask_plain = engine.encode(secondtofourth_bytes_mask, public_key)
    remained_bytes_mask_plain = engine.encode(remained_bytes_mask, public_key)
    
    # ------------------------------encrypt masks------------------------------
    first_bytes_mask_enc = engine.encrypt(first_bytes_mask_plain, public_key)
    secondtofourth_bytes_mask_enc = engine.encrypt(secondtofourth_bytes_mask_plain, public_key)
    remained_bytes_mask_enc = engine.encrypt(remained_bytes_mask_plain, public_key)
    
    # ------------------------------Masking hi bytes------------------------------
    first_bytes_enc_hi = engine.multiply(enc_key_hi, first_bytes_mask_enc)
    secondtofourth_bytes_enc_hi = engine.multiply(enc_key_hi, secondtofourth_bytes_mask_enc)
    remained_bytes_enc_hi = engine.multiply(enc_key_hi, remained_bytes_mask_enc)
    
    # ------------------------------Masking lo bytes------------------------------
    first_bytes_enc_lo = engine.multiply(enc_key_lo, first_bytes_mask_enc)
    secondtofourth_bytes_enc_lo = engine.multiply(enc_key_lo, secondtofourth_bytes_mask_enc)
    remained_bytes_enc_lo = engine.multiply(enc_key_lo, remained_bytes_mask_enc)
    
    # ------------------------------Apply RotWord to hi bytes------------------------------
    # Move secondtofourth_bytes to positions 0,1,2 and first_bytes to position 3
    new_positions_012 = engine.rotate(secondtofourth_bytes_enc_hi, fixed_rotation_key_neg_2048)
    new_position_3 = engine.fixed_rotate(first_bytes_enc_hi, fixed_rotation_key_3_2048)

    rotated_hi_bytes = engine.add(new_positions_012, new_position_3)
    rotated_hi_bytes = engine.add(rotated_hi_bytes, remained_bytes_enc_hi)
    
    # ------------------------------Apply RotWord to lo bytes------------------------------
    new_positions_012 = engine.rotate(secondtofourth_bytes_enc_lo, fixed_rotation_key_neg_2048)
    new_position_3 = engine.fixed_rotate(first_bytes_enc_lo, fixed_rotation_key_3_2048)

    rotated_lo_bytes = engine.add(new_positions_012, new_position_3)
    rotated_lo_bytes = engine.add(rotated_lo_bytes, remained_bytes_enc_lo)

    return rotated_hi_bytes, rotated_lo_bytes


def _sub_word(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    nibble_pack = sub_bytes(engine_context, enc_key_hi, enc_key_lo)
    
    return 

def _rcon_xor(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    pass

def _xor(engine_context: CKKS_EngineContext, enc_key_hi, enc_key_lo):
    return _xor_operation(engine_context, enc_key_hi, enc_key_lo)

def key_scheduling(engine_context, enc_key_hi, enc_key_lo):
    pass