"""aes-ShiftRows.py

ShiftRows operation for AES-128 ECB mode with FHE compatibility.

The ShiftRows operation performs cyclic left shifts on the rows of the AES state:
- Row 0: no shift
- Row 1: left shift by 1 position  
- Row 2: left shift by 2 positions
- Row 3: left shift by 3 positions

For our flat array layout (16 * max_blocks), we need to apply these shifts
within each 4x4 block while maintaining the batching structure.

Layout reminder:
- Flat array: [b0_all_blocks, b1_all_blocks, ..., b15_all_blocks]
- Each b_i segment has max_blocks elements
- Within each block: bytes 0,1,2,3 = row 0; bytes 4,5,6,7 = row 1; etc.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "shift_rows",
]


def shift_rows(state: np.ndarray, max_blocks: int = 2048) -> np.ndarray:
    """Apply ShiftRows transformation to the state.
    
    Uses masking and np.roll to implement circular shifts without indexing/slicing,
    following the rot_word implementation pattern.
    
    Parameters
    ----------
    state : np.ndarray, shape (16 * max_blocks,), dtype uint8
        Flat state array from blocks_to_flat_array()
    max_blocks : int, optional
        Maximum number of blocks in the layout (default 2048)
    
    Returns
    -------
    new_state : np.ndarray, shape (16 * max_blocks,), dtype uint8
        State after ShiftRows transformation
    """
    if state.dtype != np.uint8:
        state = state.astype(np.uint8, copy=False)
    
    if state.size != 16 * max_blocks:
        raise ValueError(f"State must have size {16 * max_blocks}, got {state.size}")
    
    # Create masks for each row of the 4x4 matrix
    # Row 0: bytes 0,1,2,3 (positions 0*max_blocks to 4*max_blocks)
    row0_mask = np.concatenate([
        np.ones(max_blocks),    # byte 0
        np.ones(max_blocks),    # byte 1  
        np.ones(max_blocks),    # byte 2
        np.ones(max_blocks),    # byte 3
        np.zeros(12 * max_blocks)  # bytes 4-15
    ])
    
    # Row 1: bytes 4,5,6,7 (positions 4*max_blocks to 8*max_blocks)
    row1_mask = np.concatenate([
        np.zeros(4 * max_blocks),   # bytes 0-3
        np.ones(max_blocks),        # byte 4
        np.ones(max_blocks),        # byte 5
        np.ones(max_blocks),        # byte 6
        np.ones(max_blocks),        # byte 7
        np.zeros(8 * max_blocks)    # bytes 8-15
    ])
    
    # Row 2: bytes 8,9,10,11 (positions 8*max_blocks to 12*max_blocks)
    row2_mask = np.concatenate([
        np.zeros(8 * max_blocks),   # bytes 0-7
        np.ones(max_blocks),        # byte 8
        np.ones(max_blocks),        # byte 9
        np.ones(max_blocks),        # byte 10
        np.ones(max_blocks),        # byte 11
        np.zeros(4 * max_blocks)    # bytes 12-15
    ])
    
    # Row 3: bytes 12,13,14,15 (positions 12*max_blocks to 16*max_blocks)
    row3_mask = np.concatenate([
        np.zeros(12 * max_blocks),  # bytes 0-11
        np.ones(max_blocks),        # byte 12
        np.ones(max_blocks),        # byte 13
        np.ones(max_blocks),        # byte 14
        np.ones(max_blocks)         # byte 15
    ])

    masked_state0 = state * row0_mask
    masked_state1 = state * row1_mask
    masked_state2 = state * row2_mask
    masked_state3 = state * row3_mask

    # Apply shifts using np.roll (following rot_word pattern):
    # Row 0: no shift (keep as is)
    # Row 1: left shift by 1 -> shift array left by 1*max_blocks positions
    # Row 2: left shift by 2 -> shift array left by 2*max_blocks positions  
    # Row 3: left shift by 3 -> shift array left by 3*max_blocks positions
    
    shifted_row0 = masked_state0

    shifted_row1_567 = np.roll(masked_state1, -1 * max_blocks)  # Left shift by 1
    shifted_row1_4 = np.roll(masked_state1, 3 * max_blocks)  # Right shift by 3
    shifted_row1 = (shifted_row1_567 * row1_mask) + (shifted_row1_4 * row1_mask)

    shifted_row2_1011 = np.roll(masked_state2, -2 * max_blocks)
    shifted_row2_89 = np.roll(masked_state2, 2 * max_blocks)
    shifted_row2 = (shifted_row2_1011 * row2_mask) + (shifted_row2_89 * row2_mask)

    shifted_row3_15 = np.roll(masked_state3, -3 * max_blocks) # Left shift by 3
    shifted_row3_121314 = np.roll(masked_state3, 1 * max_blocks) # Right shift by 1
    shifted_row3 = (shifted_row3_121314  * row3_mask) + (shifted_row3_15 * row3_mask)
    
    # Combine all shifted rows
    result = shifted_row0 + shifted_row1 + shifted_row2 + shifted_row3
    
    return result.astype(np.uint8)


if __name__ == "__main__":
    # Test ShiftRows operation with direct (16*2048,) input
    max_blocks = 2048
    
    # Create test input: (16*2048,) array where each byte is repeated 2048 times
    # Pattern: [0,0,...,0, 1,1,...,1, 2,2,...,2, ..., 15,15,...,15]
    base_bytes = np.arange(16, dtype=np.uint8)
    state = np.repeat(base_bytes, max_blocks)  # Shape: (16*2048,)
    
    print("Input state shape:", state.shape)
    print("Original block layout (showing first value of each byte position):")
    print("Row 0:", [f"{state[i*max_blocks]:02X}" for i in range(0, 4)])    # bytes 0,1,2,3
    print("Row 1:", [f"{state[i*max_blocks]:02X}" for i in range(4, 8)])    # bytes 4,5,6,7
    print("Row 2:", [f"{state[i*max_blocks]:02X}" for i in range(8, 12)])   # bytes 8,9,10,11
    print("Row 3:", [f"{state[i*max_blocks]:02X}" for i in range(12, 16)])  # bytes 12,13,14,15
    
    # Apply ShiftRows
    result = shift_rows(state, max_blocks)
    
    print("\nOutput shape:", result.shape)
    print("After ShiftRows (showing first value of each byte position):")
    print("Row 0:", [f"{result[i*max_blocks]:02X}" for i in range(0, 4)])    # Should be [00, 01, 02, 03]
    print("Row 1:", [f"{result[i*max_blocks]:02X}" for i in range(4, 8)])    # Should be [05, 06, 07, 04]
    print("Row 2:", [f"{result[i*max_blocks]:02X}" for i in range(8, 12)])   # Should be [0A, 0B, 08, 09]
    print("Row 3:", [f"{result[i*max_blocks]:02X}" for i in range(12, 16)])  # Should be [0F, 0C, 0D, 0E]
    
    # Verify expected pattern
    expected_row0 = [0x00, 0x01, 0x02, 0x03]  # No shift
    expected_row1 = [0x05, 0x06, 0x07, 0x04]  # Left shift by 1
    expected_row2 = [0x0A, 0x0B, 0x08, 0x09]  # Left shift by 2
    expected_row3 = [0x0F, 0x0C, 0x0D, 0x0E]  # Left shift by 3
    
    actual_row0 = [int(result[i*max_blocks]) for i in range(0, 4)]
    actual_row1 = [int(result[i*max_blocks]) for i in range(4, 8)]
    actual_row2 = [int(result[i*max_blocks]) for i in range(8, 12)]
    actual_row3 = [int(result[i*max_blocks]) for i in range(12, 16)]
    
    assert actual_row0 == expected_row0, f"Row 0 failed: got {actual_row0}, expected {expected_row0}"
    assert actual_row1 == expected_row1, f"Row 1 shift failed: got {actual_row1}, expected {expected_row1}"
    assert actual_row2 == expected_row2, f"Row 2 shift failed: got {actual_row2}, expected {expected_row2}"
    assert actual_row3 == expected_row3, f"Row 3 shift failed: got {actual_row3}, expected {expected_row3}"
    
    print("✓ ShiftRows verification passed")
    
    # Additional test: verify all blocks have same pattern
    print("\nVerifying consistency across all blocks...")
    for block_idx in [0, 1, 100, 1000, 2047]:  # Test some representative blocks
        block_row1 = [int(result[i*max_blocks + block_idx]) for i in range(4, 8)]
        assert block_row1 == expected_row1, f"Block {block_idx} row 1 inconsistent: {block_row1}"
    print("✓ All blocks consistent")