import numpy as np
from desilofhe import Engine

"""
    XOR process

    Input:
        - engine_context: Engine context
        - alpha_hi: alpha_hi
        - alpha_lo: alpha_lo
        - beta_hi: beta_hi
        - beta_lo: beta_lo
        
    Output:
        - xor_result: XOR result
        - xor_result_hi: XOR result (high)
        - xor_result_lo: XOR result (low)
        
    Process:
        - XOR the alpha and beta
        
    Function:
        - xor_process: XOR process
        - _compute_power_basis: compute power basis of Input. 
        - _compute_conjugate_power_basis
        - _compute_coefficient
"""

def xor_process(engine_context, alpha_hi, alpha_lo, beta_hi, beta_lo):
    """
    두 2비트 암호문에 대해 전체 및 상·하위 비트 XOR 결과를 계산.

    Parameters
    ----------
    engine_context : FHEContext
        암호 연산을 수행할 엔진 컨텍스트 객체.
    alpha_hi : Ciphertext
        첫 번째 피연산자의 상위 비트 암호문.
    alpha_lo : Ciphertext
        첫 번째 피연산자의 하위 비트 암호문.
    beta_hi : Ciphertext
        두 번째 피연산자의 상위 비트 암호문.
    beta_lo : Ciphertext
        두 번째 피연산자의 하위 비트 암호문.

    Returns
    -------
    xor_result : Ciphertext
        복원된 2비트 전체 XOR 결과.
    xor_result_hi : Ciphertext
        결과의 상위 비트 암호문.
    xor_result_lo : Ciphertext
        결과의 하위 비트 암호문.

    Notes
    -----
    내부적으로 `_compute_power_basis` 와
    `_compute_conjugate_power_basis` 를 사용하여
    1, x, x², x³ 등의 전치(전력) 기반을 계산합니다.
    """





