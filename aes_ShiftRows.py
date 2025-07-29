`"""aes-ShiftRows.py

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
# -----------------------------------------------------------------------------
# ShiftRows helpers
# -----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np

__all__ = [
    "shift_rows",
]


# -----------------------------------------------------------------------------
# Mask plaintext cache
# -----------------------------------------------------------------------------

def _get_shift_rows_masks(engine_context: "CKKS_EngineContext"):
    """Return plaintext masks for (inverse) ShiftRows.

    The expensive numpy-concatenate & engine.encode operations are executed **once
    per CKKS_EngineContext** and the resulting Plaintext objects are cached on
    the context instance (attribute ``_shift_rows_masks``). Subsequent calls
    simply reuse the cached values.
    """

    # Fast-path: masks already cached on this context --------------------------------
    if hasattr(engine_context, "_shift_rows_masks"):
        return engine_context._shift_rows_masks  # type: ignore[attr-defined]

    engine = engine_context.engine
    max_blocks = 2048  # number of AES blocks packed in one ciphertext

    # ---------------------------- build boolean masks ------------------------------
    row_0_mask = np.concatenate((np.ones(4 * max_blocks), np.zeros(12 * max_blocks)))

    row_1_0_mask = np.concatenate(
        (np.zeros(4 * max_blocks), np.ones(4 * max_blocks), np.zeros(8 * max_blocks))
    )
    row_2_01_mask = np.concatenate(
        (np.zeros(8 * max_blocks), np.ones(4 * max_blocks), np.zeros(4 * max_blocks))
    )
    row_3_012_mask = np.concatenate((np.zeros(12 * max_blocks), np.ones(4 * max_blocks)))

    row_1_123_mask = np.concatenate(
        (np.zeros(5 * max_blocks), np.ones(3 * max_blocks), np.zeros(8 * max_blocks))
    )
    row_2_23_mask = np.concatenate(
        (np.zeros(10 * max_blocks), np.ones(2 * max_blocks), np.zeros(4 * max_blocks))
    )
    row_3_3_mask = np.concatenate((np.zeros(15 * max_blocks), np.ones(1 * max_blocks)))

    # ---------------------------- encode and cache ----------------------------------
    masks = {
        "row_0": engine.encode(row_0_mask),
        "row_1_0": engine.encode(row_1_0_mask),
        "row_2_01": engine.encode(row_2_01_mask),
        "row_3_012": engine.encode(row_3_012_mask),
        "row_1_123": engine.encode(row_1_123_mask),
        "row_2_23": engine.encode(row_2_23_mask),
        "row_3_3": engine.encode(row_3_3_mask),
    }

    # Persist on context for future reuse
    engine_context._shift_rows_masks = masks  # type: ignore[attr-defined]
    return masks


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
    # Load / cache plaintext masks -------------------------------------------------
    # -----------------------------------------------------------------------------
    _masks = _get_shift_rows_masks(engine_context)

    row_0_mask_plaintext = _masks["row_0"]
    row_1_0_mask_plaintext = _masks["row_1_0"]
    row_2_01_mask_plaintext = _masks["row_2_01"]
    row_3_012_mask_plaintext = _masks["row_3_012"]

    row_1_123_mask_plaintext = _masks["row_1_123"]
    row_2_23_mask_plaintext = _masks["row_2_23"]
    row_3_3_mask_plaintext = _masks["row_3_3"]
    
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
    # fixed_rotation_key_list 내용물은 -3 -2 -1 1 2 3 이렇게 저장됨.
    # mask_row_1에 대해 0은 3 로 한번, 123은 -1로 한 번 회전
    rotated_row_1_0 = engine.rotate(masked_row_1_0, fixed_rotation_key_list[5])
    rotated_row_1_123 = engine.rotate(masked_row_1_123, fixed_rotation_key_list[2])
    
    # mask_row_2에 대해 01은 2로 한번, 23은 -2로 한 번 회전
    rotated_row_2_01 = engine.rotate(masked_row_2_01, fixed_rotation_key_list[4])
    rotated_row_2_23 = engine.rotate(masked_row_2_23, fixed_rotation_key_list[1])
    
    # mask_row_3에 대해 012는 1로 한번, 3은 -3로 한 번 회전
    rotated_row_3_012 = engine.rotate(masked_row_3_012, fixed_rotation_key_list[3])
    rotated_row_3_3 = engine.rotate(masked_row_3_3, fixed_rotation_key_list[0])
    
    # concatenate all the rotated rows
    rotated_rows_0 = engine.add(masked_row_0, rotated_row_1_0)
    rotated_rows_1 = engine.add(rotated_rows_0, rotated_row_1_123)
    rotated_rows_2 = engine.add(rotated_rows_1, rotated_row_2_01)
    rotated_rows_3 = engine.add(rotated_rows_2, rotated_row_2_23)
    rotated_rows_4 = engine.add(rotated_rows_3, rotated_row_3_012)
    rotated_rows = engine.add(rotated_rows_4, rotated_row_3_3)

    return rotated_rows

def inverse_shift_rows(engine_context: CKKS_EngineContext, ct_hi, ct_lo):
    """
    Inverse ShiftRows operation for AES-128 ECB mode with FHE compatibility.
    """
    pass