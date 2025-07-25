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
_aes_xor_mod = _load_module("aes-xor.py", "aes_xor")
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


def rot_word(engine_context, enc_key_hi_list, enc_key_lo_list):
    """Apply RotWord operation to the first 4 word (bytes 0-3) of flat key array.
    
    Performs circular left shift by 1 byte on the first 4-byte word.
    
    Parameters
    ----------
    engine_context : FHEContext
    enc_key_hi_list : list[engine.Ciphertext]
        List of ciphertexts where each ciphertext encrypts 2048 elements parallelly
    enc_key_lo_list : list[engine.Ciphertext]
        List of ciphertexts where each ciphertext encrypts 2048 elements parallelly
        
    Returns
    -------
    rotated_hi_list : list[engine.Ciphertext]
        List of ciphertexts where each ciphertext encrypts 2048 elements parallelly
    rotated_lo_list : list[engine.Ciphertext]
        List of ciphertexts where each ciphertext encrypts 2048 elements parallelly

    -------
    example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] -> [2, 3, 4, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    """

    # 1. Slicing first 4 bytes of two lists
    enc_key_hi_list_first_4 = [enc_key_hi_list[:4]]
    enc_key_hi_list_remained = [enc_key_hi_list[4:]]
    enc_key_lo_list_first_4 = [enc_key_lo_list[:4]]
    enc_key_lo_list_remained = [enc_key_lo_list[4:]]
    
    # 2. Rotate first 4 bytes
    enc_key_hi_list_first_4_rotated = np.roll(enc_key_hi_list_first_4, -1)
    enc_key_lo_list_first_4_rotated = np.roll(enc_key_lo_list_first_4, -1)
    
    # 4. Concatenate rotated bytes and remained bytes
    enc_key_hi_list_concatenated = np.concatenate(enc_key_hi_list_first_4_rotated, enc_key_hi_list_remained)
    enc_key_lo_list_concatenated = np.concatenate(enc_key_lo_list_first_4_rotated, enc_key_lo_list_remained)
    
    return enc_key_hi_list_concatenated, enc_key_lo_list_concatenated

def sub_word(engine_context, enc_key_hi_list, enc_key_lo_list):
    pass

def rcon_xor(engine_context, enc_key_hi_list, enc_key_lo_list):
    pass

def xor(engine_context, enc_key_hi_list, enc_key_lo_list):
    return _xor_operation(engine_context, enc_key_hi_list, enc_key_lo_list)

def key_scheduling(engine_context, enc_key_hi_list, enc_key_lo_list):
    pass