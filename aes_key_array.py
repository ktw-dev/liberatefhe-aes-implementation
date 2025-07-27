"""aes-key-array.py

Utility for constructing a *flat* key array that matches the plaintext layout
used in the AES 2048-block SIMD batching scheme (ECB mode).

Layout recap (length = 16 × max_blocks = 32 768):
    [ k0, k0, … 2048×,   # byte 0 of the key repeated
      k1, k1, … 2048×,   # byte 1 of the key repeated
      …
      k15, k15, … 2048× ]

This allows a single np.bitwise_xor() with the plaintext flat array to apply
AddRoundKey for all blocks simultaneously.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

__all__ = [
    "key_to_flat_array",
    "split_to_nibbles",
]


def key_to_flat_array(key: Sequence[int] | np.ndarray, max_blocks: int = 2048) -> np.ndarray:
    """Return a flattened key array compatible with *aes-block-array* layout.

    Parameters
    ----------
    key : sequence of int (length 16) or ndarray of uint8
        AES-128 secret key bytes (0-255 each).
    max_blocks : int, optional
        Column width – must match plaintext packing (default 2048).

    Returns
    -------
    flat_key : np.ndarray, shape (16 * max_blocks,), dtype=np.uint8
        Key bytes replicated row-wise then flattened.
    """
    key_arr = np.asarray(key, dtype=np.uint8)
    if key_arr.size != 16:
        raise ValueError("key must contain exactly 16 bytes for AES-128")

    # Create (16, max_blocks) matrix: each row = single key byte repeated
    key_matrix = np.repeat(key_arr.reshape(16, 1), max_blocks, axis=1)

    return key_matrix.reshape(-1)

# -----------------------------------------------------------------------------
# Nibble splitting -------------------------------------------------------------
# -----------------------------------------------------------------------------


def split_to_nibbles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised split of uint8 array into upper/lower 4-bit nibbles.

    Parameters
    ----------
    arr : np.ndarray of uint8

    Returns
    -------
    hi : np.ndarray uint8
        Upper nibbles (values 0-15)
    lo : np.ndarray uint8
        Lower nibbles (values 0-15)
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    hi = np.right_shift(arr, 4, dtype=np.uint8)
    lo = np.bitwise_and(arr, 0x0F, dtype=np.uint8)
    return hi, lo


if __name__ == "__main__":
    # Quick self-test / demo
    rng = np.random.default_rng()
    default_key = rng.integers(0, 256, size=16, dtype=np.uint8)
    flat_key = key_to_flat_array(default_key)
    hi, lo = split_to_nibbles(flat_key)

    print("Original key      :", [f"{b:02X}" for b in default_key])
    print("Flat key length   :", flat_key.size)
    print("First 32 bytes    :", [f"{b:02X}" for b in flat_key[:32]])
    print("Byte-8 segment    :", [f"{b:02X}" for b in flat_key[8*2048 : 8*2048 + 8]])
    print("Upper nibbles (0-31):", [f"{b:X}" for b in hi[:32]])
    print("Lower nibbles (0-31):", [f"{b:X}" for b in lo[:32]])
