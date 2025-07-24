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

def homomorphic_shift_rows(ct_state: List[Any]) -> List[Any]:
    """
    암호화된 상태(state)에 대해 동형 ShiftRows 연산을 수행합니다.
    컬럼-메이저 SIMD 구조에서는 FHE 연산 없이 암호문 리스트를 재배열합니다.

    Args:
        ct_state (List[Ciphertext]): AES 상태를 나타내는 16개 암호문의 리스트.

    Returns:
        List[Ciphertext]: ShiftRows가 적용된 후의 16개 암호문 리스트.
    """
    if len(ct_state) != 16:
        raise ValueError("입력 암호문 리스트는 반드시 16개의 요소를 가져야 합니다.")

    shifted_ct_state = [None] * 16
    
    # AES 상태 행렬의 (row, col)과 1차원 리스트 인덱스(idx) 매핑
    # idx = col * 4 + row
    # state[row, col]의 값은 state[row, (col - row) % 4]로 이동 (왼쪽 순환 이동)
    for row in range(4):
        for col in range(4):
            original_idx = col * 4 + row
            new_col = (col - row) % 4
            new_idx = new_col * 4 + row
            
            # 재배열: new_idx 위치에는 original_idx에 있던 암호문이 위치
            shifted_ct_state[new_idx] = ct_state[original_idx]
            
    return shifted_ct_state