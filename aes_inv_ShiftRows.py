"""aes_inv_ShiftRows.py

Inverse ShiftRows operation for AES-128 ECB mode with FHE compatibility.

The Inverse ShiftRows operation performs cyclic right shifts on the rows of the AES state:
- Row 0: no shift
- Row 1: right shift by 1 position  
- Row 2: right shift by 2 positions
- Row 3: right shift by 3 positions

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
from engine_context import CKKS_EngineContext

__all__ = [
    "inv_shift_rows",
]


# -----------------------------------------------------------------------------
# Mask plaintext cache
# -----------------------------------------------------------------------------

def _get_inv_shift_rows_masks(engine_context: "CKKS_EngineContext"):
    """Return plaintext masks for (inverse) ShiftRows.

    The expensive numpy-concatenate & engine.encode operations are executed **once
    per CKKS_EngineContext** and the resulting Plaintext objects are cached on
    the context instance (attribute ``_shift_rows_masks``). Subsequent calls
    simply reuse the cached values.
    """

    # Fast-path: masks already cached on this context --------------------------------
    if hasattr(engine_context, "_inv_shift_rows_masks"):
        return engine_context._inv_shift_rows_masks  # type: ignore[attr-defined]

    engine = engine_context.engine
    max_blocks = 2048  # number of AES blocks packed in one ciphertext

    # ---------------------------- build boolean masks ------------------------------
    row_0_mask = np.concatenate((np.ones(4 * max_blocks), np.zeros(12 * max_blocks)))

    # row 1 is split into 3+1 bytes -> segments 4,5,6 and 7
    row_1_012_mask = np.concatenate(
        (np.zeros(4 * max_blocks), np.ones(3 * max_blocks), np.zeros(9 * max_blocks))
    )
    row_1_3_mask = np.concatenate(
        (np.zeros(7 * max_blocks), np.ones(1 * max_blocks), np.zeros(8 * max_blocks))
    )
    # row 2 is split into 2+2 bytes -> segments 8,9 and 10,11
    row_2_01_mask = np.concatenate(
        (np.zeros(8 * max_blocks), np.ones(2 * max_blocks), np.zeros(6 * max_blocks))
    )
    row_2_23_mask = np.concatenate(
        (np.zeros(10 * max_blocks), np.ones(2 * max_blocks), np.zeros(4 * max_blocks))
    )
    # row 3 is split into 1+3 bytes -> segments 12 and 13,14,15
    row_3_0_mask = np.concatenate(
        (np.zeros(12 * max_blocks), np.ones(1 * max_blocks), np.zeros(3 * max_blocks))
    )
    row_3_123_mask = np.concatenate(
        (np.zeros(13 * max_blocks), np.ones(3 * max_blocks))
    )


    # ---------------------------- encode and cache ----------------------------------
    masks = {
        "row_0": engine.encode(row_0_mask),
        # ---------------- Inverse ShiftRows masks -----------------
        # naming convention: rows_<rowIndex>_<segment description>
        #   rows_0               : entire row 0 (no rotation)
        #   rows_1_012 / rows_1_3: row-1 split before inverse rotation (bytes 4-6 vs 7)
        #   rows_2_01  / rows_2_23: row-2 split (bytes 8,9 vs 10,11)
        #   rows_3_0   / rows_3_123: row-3 split (byte 12 vs 13-15)
        "rows_0": engine.encode(row_0_mask),
        "rows_1_012": engine.encode(row_1_012_mask),
        "rows_1_3": engine.encode(row_1_3_mask),
        "rows_2_01": engine.encode(row_2_01_mask),  # same as forward
        "rows_2_23": engine.encode(row_2_23_mask),  # same as forward
        "rows_3_0": engine.encode(row_3_0_mask),
        "rows_3_123": engine.encode(row_3_123_mask),
    }

    # Persist on context for future reuse
    engine_context._inv_shift_rows_masks = masks
    return masks


def inv_shift_rows(engine_context: CKKS_EngineContext, ct_hi, ct_lo):
    """
    ShiftRows operation for AES-128 ECB mode with FHE compatibility.

    The ShiftRows operation performs cyclic right shifts on the rows of the AES state:
    - Row 0: no shift
    - Row 1: right shift by 1 position  
    - Row 2: right shift by 2 positions
    - Row 3: right shift by 3 positions
    
    모든 연산은 masking_plaintext 연산과 rotate 연산을 사용한다.
    """
    engine = engine_context.engine
    
    # -----------------------------------------------------------------------------
    # engine_context.get_fixed_rotation_key
    # -----------------------------------------------------------------------------
    fixed_rotation_key_list = [engine_context.get_fixed_rotation_key(i * 2048) for i in range(-3, 4) if i != 0]
    # -3 -2 -1 1 2 3
    print(fixed_rotation_key_list)
    
    # -----------------------------------------------------------------------------
    # Load / cache plaintext masks -------------------------------------------------
    # -----------------------------------------------------------------------------
    _masks = _get_inv_shift_rows_masks(engine_context)

    row_0_mask_plaintext = _masks["row_0"]
    row_1_012_mask_plaintext = _masks["rows_1_012"]
    row_1_3_mask_plaintext = _masks["rows_1_3"]
    row_2_01_mask_plaintext = _masks["rows_2_01"]
    row_2_23_mask_plaintext = _masks["rows_2_23"]
    row_3_0_mask_plaintext = _masks["rows_3_0"]
    row_3_123_mask_plaintext = _masks["rows_3_123"]
    
    # -----------------------------------------------------------------------------
    # masking operation of High nibble
    # -----------------------------------------------------------------------------
    masked_row_hi_0 = engine.multiply(ct_hi, row_0_mask_plaintext)
    
    masked_row_hi_1_012 = engine.multiply(ct_hi, row_1_012_mask_plaintext)
    masked_row_hi_1_3 = engine.multiply(ct_hi, row_1_3_mask_plaintext)
    
    masked_row_hi_2_01 = engine.multiply(ct_hi, row_2_01_mask_plaintext)
    masked_row_hi_2_23 = engine.multiply(ct_hi, row_2_23_mask_plaintext)
    
    masked_row_hi_3_0 = engine.multiply(ct_hi, row_3_0_mask_plaintext)
    masked_row_hi_3_123 = engine.multiply(ct_hi, row_3_123_mask_plaintext)
    
    # -----------------------------------------------------------------------------
    # masking operation of Low nibble
    # -----------------------------------------------------------------------------
    masked_row_lo_0 = engine.multiply(ct_lo, row_0_mask_plaintext)
    
    masked_row_lo_1_012 = engine.multiply(ct_lo, row_1_012_mask_plaintext)
    masked_row_lo_1_3 = engine.multiply(ct_lo, row_1_3_mask_plaintext)
    
    masked_row_lo_2_01 = engine.multiply(ct_lo, row_2_01_mask_plaintext)
    masked_row_lo_2_23 = engine.multiply(ct_lo, row_2_23_mask_plaintext)
    
    masked_row_lo_3_0 = engine.multiply(ct_lo, row_3_0_mask_plaintext)
    masked_row_lo_3_123 = engine.multiply(ct_lo, row_3_123_mask_plaintext)
    
    # -----------------------------------------------------------------------------
    # rotate operation of High nibble
    # -----------------------------------------------------------------------------
    # fixed_rotation_key_list 내용물은 -3 -2 -1 1 2 3 이렇게 저장됨.
    # mask_row_1에 대해 0은 3 로 한번, 123은 -1로 한 번 회전
    rotated_row_hi_1_012 = engine.rotate(masked_row_hi_1_012, fixed_rotation_key_list[5])
    rotated_row_hi_1_3 = engine.rotate(masked_row_hi_1_3, fixed_rotation_key_list[2])
    
    # mask_row_2에 대해 01은 2로 한번, 23은 -2로 한 번 회전
    rotated_row_hi_2_01 = engine.rotate(masked_row_hi_2_01, fixed_rotation_key_list[4])
    rotated_row_hi_2_23 = engine.rotate(masked_row_hi_2_23, fixed_rotation_key_list[1])
    
    # mask_row_3에 대해 012는 1로 한번, 3은 -3로 한 번 회전
    rotated_row_hi_3_0 = engine.rotate(masked_row_hi_3_0, fixed_rotation_key_list[3])
    rotated_row_hi_3_123 = engine.rotate(masked_row_hi_3_123, fixed_rotation_key_list[0])
    
    # concatenate all the rotated rows
    rotated_rows_hi_0 = engine.add(masked_row_hi_0, rotated_row_hi_1_0)
    rotated_rows_hi_1 = engine.add(rotated_rows_hi_0, rotated_row_hi_1_012)
    rotated_rows_hi_2 = engine.add(rotated_rows_hi_1, rotated_row_hi_2_01)
    rotated_rows_hi_3 = engine.add(rotated_rows_hi_2, rotated_row_hi_2_23)
    rotated_rows_hi_4 = engine.add(rotated_rows_hi_3, rotated_row_hi_3_0)
    rotated_rows_hi = engine.add(rotated_rows_hi_4, rotated_row_hi_3_123)
    
    # -----------------------------------------------------------------------------
    # rotate operation of Low nibble
    # -----------------------------------------------------------------------------
    # fixed_rotation_key_list 내용물은 -3 -2 -1 1 2 3 이렇게 저장됨.
    # mask_row_1에 대해 012는 1로 한번, 3은 -3로 한 번 회전
    rotated_row_lo_1_012 = engine.rotate(masked_row_lo_1_012, fixed_rotation_key_list[3])
    rotated_row_lo_1_3 = engine.rotate(masked_row_lo_1_3, fixed_rotation_key_list[0])
    
    # mask_row_2에 대해 01은 2로 한번, 23은 -2로 한 번 회전
    rotated_row_lo_2_01 = engine.rotate(masked_row_lo_2_01, fixed_rotation_key_list[4])
    rotated_row_lo_2_23 = engine.rotate(masked_row_lo_2_23, fixed_rotation_key_list[1])
    
    # mask_row_3에 대해 123는 -1로 한번, 0은 3로 한 번 회전
    rotated_row_lo_3_0 = engine.rotate(masked_row_lo_3_0, fixed_rotation_key_list[5])
    rotated_row_lo_3_123 = engine.rotate(masked_row_lo_3_123, fixed_rotation_key_list[2])
    
    # concatenate all the rotated rows
    rotated_rows_lo_0 = engine.add(masked_row_lo_0, rotated_row_lo_1_012)
    rotated_rows_lo_1 = engine.add(rotated_rows_lo_0, rotated_row_lo_1_3)
    rotated_rows_lo_2 = engine.add(rotated_rows_lo_1, rotated_row_lo_2_01)
    rotated_rows_lo_3 = engine.add(rotated_rows_lo_2, rotated_row_lo_2_23)
    rotated_rows_lo_4 = engine.add(rotated_rows_lo_3, rotated_row_lo_3_0)
    rotated_rows_lo = engine.add(rotated_rows_lo_4, rotated_row_lo_3_123)
    


    return rotated_rows_hi, rotated_rows_lo


if __name__ == "__main__":
    from aes_transform_zeta import int_to_zeta, zeta_to_int
    from aes_split_to_nibble import split_to_nibbles
    import numpy as np
    import time

    # CKKS 엔진 초기화
    engine_context = CKKS_EngineContext(signature=1, use_bootstrap=True, mode="parallel", thread_count=16, device_id=0)
    engine = engine_context.engine
    public_key = engine_context.public_key
    secret_key = engine_context.secret_key

    print("engine init")

    slots_per_block = 2048
    num_blocks = 16

    # 0~15 값을 각각 2048번 반복해서 붙이기
    int_array = np.repeat(np.arange(num_blocks, dtype=np.uint8), slots_per_block)

    print(int_array.shape)   # (32768,)
    print(int_array[:6])
    print(int_array[2040:2055])  # 경계 확인: 앞은 0, 뒤는 1로 바뀜

    # hi/lo nibble로 분할
    alpha_int, beta_int = split_to_nibbles(int_array)

    # zeta domain 매핑
    alpha = int_to_zeta(alpha_int)
    beta  = int_to_zeta(beta_int)

    # 암호화
    enc_alpha = engine.encrypt(alpha, public_key, level=10)
    enc_beta  = engine.encrypt(beta, public_key, level=10)

    # 2. ShiftRows 실행
    print("inv_ShiftRows 실행")
    print(f"before shiftrows.level: hi={enc_alpha.level}, lo={enc_beta.level}")
    start_time = time.time()
    shifted_hi_ct, shifted_lo_ct = inv_shift_rows(engine_context, enc_alpha, enc_beta)
    end_time = time.time()
    print(f"inv_ShiftRows time taken: {end_time - start_time} seconds")
    print(f"after shiftrows.level: hi={shifted_hi_ct.level}, lo={shifted_lo_ct.level}")

    # 3. 복호화 (hi/lo 둘 다)
    decoded_zeta_hi = engine.decrypt(shifted_hi_ct, secret_key)
    decoded_int_hi = zeta_to_int(decoded_zeta_hi)

    decoded_zeta_lo = engine.decrypt(shifted_lo_ct, secret_key)
    decoded_int_lo = zeta_to_int(decoded_zeta_lo)

    # 최종 바이트 결합
    decoded_bytes = ((decoded_int_hi.astype(np.uint8) << 4) |
                     decoded_int_lo.astype(np.uint8))

    # 5. 비교 (row-major 블록 매핑 기반)
    slots_per_block = 2048
    rows = [
        [0, 1, 2, 3],     # Row 0
        [4, 5, 6, 7],     # Row 1
        [8, 9, 10, 11],   # Row 2
        [12, 13, 14, 15]  # Row 3
    ]

    # ShiftRows 회전
    rows[1] = np.roll(rows[1], 1)  # Row1 right shift by 1
    rows[2] = np.roll(rows[2], 2)  # Row2 right shift by 2
    rows[3] = np.roll(rows[3], 3)  # Row3 right shift by 3

    # 최종 블록 순서
    expected_block_order = [b for row in rows for b in row]

    # 블록 단위 비교
    mismatches = 0
    for block_idx, expected_val in enumerate(expected_block_order):
        start = block_idx * slots_per_block
        end = start + slots_per_block
        block_values = decoded_bytes[start:end]
        if not np.all(block_values == expected_val):
            mismatches += 1
            print(f"❌ Block {block_idx}: expected {expected_val}, got unique values {np.unique(block_values)}")

    if mismatches == 0:
        print("✅ inv_ShiftRows output matches block mapping for all blocks!")
    else:
        raise AssertionError(f"inv_ShiftRows block mismatch in {mismatches} out of {len(expected_block_order)} blocks.")