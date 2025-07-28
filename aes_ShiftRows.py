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

def shift_rows(engine_context: CKKS_EngineContext, ct_hi, ct_lo):
    """
    ShiftRows operation for AES-128 ECB mode with FHE compatibility.

    The ShiftRows operation performs cyclic left shifts on the rows of the AES state:
    - Row 0: no shift
    - Row 1: left shift by 1 position  
    - Row 2: left shift by 2 positions
    - Row 3: left shift by 3 positions
    
    모든 연산은 masking_plaintext 연산과 rotate 연산을 사용한다.
    """
    engine = engine_context.engine
    
    # -----------------------------------------------------------------------------
    # engine_context.get_fixed_rotation_key
    # -----------------------------------------------------------------------------
    fixed_rotation_key_list = [engine_context.get_fixed_rotation_key(i * 2048) for i in range(-3, 4) if i != 0]
    # -3 -2 -1 1 2 3
    
    # -----------------------------------------------------------------------------
    # masking_plaintext 생성
    # -----------------------------------------------------------------------------
    row_0_mask = np.concatenate((np.ones(4 * 2048), np.zeros(12 * 2048)))
    row_1_0_mask = np.concatenate((np.zeros(4 * 2048), np.ones(4 * 2048), np.zeros(8 * 2048)))
    row_2_01_mask = np.concatenate((np.zeros(8 * 2048), np.ones(4 * 2048), np.zeros(4 * 2048)))
    row_3_012_mask = np.concatenate((np.zeros(12 * 2048), np.ones(4 * 2048)))
    
    row_1_123_mask = np.concatenate((np.zeros(5 * 2048), np.ones(3 * 2048), np.zeros(8 * 2048)))
    row_2_23_mask = np.concatenate((np.zeros(10 * 2048), np.ones(2 * 2048), np.zeros(4 * 2048)))
    row_3_3_mask = np.concatenate((np.zeros(15 * 2048), np.ones(1 * 2048)))
    
    
    # -----------------------------------------------------------------------------
    # encode masking_plaintext
    # -----------------------------------------------------------------------------
    row_0_mask_plaintext = engine.encode(row_0_mask)
    row_1_0_mask_plaintext = engine.encode(row_1_0_mask)
    row_2_01_mask_plaintext = engine.encode(row_2_01_mask)
    row_3_012_mask_plaintext = engine.encode(row_3_012_mask)
    
    row_1_123_mask_plaintext = engine.encode(row_1_123_mask)
    row_2_23_mask_plaintext = engine.encode(row_2_23_mask)
    row_3_3_mask_plaintext = engine.encode(row_3_3_mask)
    
    # -----------------------------------------------------------------------------
    # masking operation
    # -----------------------------------------------------------------------------
    masked_row_0 = engine.multiply(ct_hi, row_0_mask_plaintext)
    
    masked_row_1_0 = engine.multiply(ct_hi, row_1_0_mask_plaintext)
    masked_row_1_123 = engine.multiply(ct_hi, row_1_123_mask_plaintext)
    
    masked_row_2_01 = engine.multiply(ct_hi, row_2_01_mask_plaintext)
    masked_row_2_23 = engine.multiply(ct_hi, row_2_23_mask_plaintext)
    
    masked_row_3_012 = engine.multiply(ct_hi, row_3_012_mask_plaintext)
    masked_row_3_3 = engine.multiply(ct_hi, row_3_3_mask_plaintext)
    
    # -----------------------------------------------------------------------------
    # rotate operation
    # -----------------------------------------------------------------------------

    return None

def inverse_shift_rows(engine_context: CKKS_EngineContext, ct_hi, ct_lo):
    """
    Inverse ShiftRows operation for AES-128 ECB mode with FHE compatibility.
    """
    pass