"""aes-addroundkey.py

AddRoundKey operation for AES-128 ECB mode with FHE compatibility.

The AddRoundKey operation performs XOR between the state and round key.
In our batched implementation, this is a simple element-wise XOR between
the flat state array and flat round key array.

Layout compatibility:
- State array: (16 * max_blocks,) uint8 - from aes-block-array.py
- Round key array: (16 * max_blocks,) uint8 - from aes-key-array.py
"""
from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = [
    "add_round_key",
]


def add_round_key(state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
    """Apply AddRoundKey operation: XOR state with round key.
    
    Parameters
    ----------
    state : np.ndarray, shape (16 * max_blocks,), dtype uint8
        Flat state array from blocks_to_flat_array()
    round_key : np.ndarray, shape (16 * max_blocks,), dtype uint8
        Flat round key array from key_to_flat_array()
    
    Returns
    -------
    new_state : np.ndarray, shape (16 * max_blocks,), dtype uint8
        State after XOR with round key
    """
    if state.shape != round_key.shape:
        raise ValueError(f"State and round key must have same shape, got {state.shape} vs {round_key.shape}")
    
    if state.dtype != np.uint8 or round_key.dtype != np.uint8:
        raise ValueError("Both state and round key must be uint8 arrays")
    
    # Element-wise XOR - vectorized operation
    return np.bitwise_xor(state, round_key, dtype=np.uint8)


if __name__ == "__main__":
    # Test with sample data
    import pathlib
    import importlib.util
    
    # Load helper modules
    def _load_module(fname: str, alias: str):
        path = pathlib.Path(__file__).parent / fname
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    aes_block_array = _load_module("aes_block_array.py", "aes_block_array")
    aes_key_array = _load_module("aes_key_array.py", "aes_key_array")
    
    # Generate test data
    rng = np.random.default_rng()
    
    # Create test blocks
    test_blocks = rng.integers(0, 256, size=(4, 16), dtype=np.uint8)
    state_flat = aes_block_array.blocks_to_flat_array(test_blocks)
    
    # Create test key
    test_key = rng.integers(0, 256, size=16, dtype=np.uint8)
    key_flat = aes_key_array.key_to_flat_array(test_key)
    
    # Apply AddRoundKey
    result = add_round_key(state_flat, key_flat)
    
    print("Test AddRoundKey operation:")
    print(f"State shape: {state_flat.shape}")
    print(f"Key shape: {key_flat.shape}")
    print(f"Result shape: {result.shape}")
    print(f"First 16 bytes state: {[f'{b:02X}' for b in state_flat[:16]]}")
    print(f"First 16 bytes key:   {[f'{b:02X}' for b in key_flat[:16]]}")
    print(f"First 16 bytes result:{[f'{b:02X}' for b in result[:16]]}")
    
    # Verify XOR property: result XOR key = original state
    verify = add_round_key(result, key_flat)
    assert np.array_equal(verify, state_flat), "AddRoundKey verification failed!"
    print("✓ Verification passed: (state XOR key) XOR key = state")