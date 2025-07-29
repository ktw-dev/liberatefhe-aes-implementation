#!/usr/bin/env python3
"""
inverse_sbox_coeffs.py (zeta → zeta 모델 최종본)
──────────────────────────
AES Inverse S-Box를 '제타 입력 → 제타 출력' 모델의 다변수 다항식 계수로 변환한다.
"""
from pathlib import Path
import json
import numpy as np
from aes_128_numpy import INV_S_BOX

# ----------------------------------------------------------------------
# 경로 및 상수
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
COEFF_DIR = ROOT / "coeffs"
COEFF_DIR.mkdir(exist_ok=True)
OUT_PATH = COEFF_DIR / "inverse_sbox_coeffs.json"

ZETA: complex = np.exp(-2j * np.pi / 16)

# ----------------------------------------------------------------------
# 계수 생성
# ----------------------------------------------------------------------
def build_target_tables() -> tuple[np.ndarray, np.ndarray]:
    """INV_S_BOX 출력 니블을 ζ-인코딩하여 (16,16) 목표 테이블을 생성한다."""
    f_hi = np.empty((16, 16), dtype=np.complex128)
    f_lo = np.empty((16, 16), dtype=np.complex128)

    for a in range(16):  # 입력 상위 니블 (a)
        for b in range(16):  # 입력 하위 니블 (b)
            # 입력값 (a,b)에 대한 S-Box의 결과
            sbox_in = (a << 4) | b
            sbox_out = int(INV_S_BOX[sbox_in])
            
            # 출력 니블을 다시 제타(zeta) 값으로 변환
            out_hi = (sbox_out >> 4) & 0xF
            out_lo = sbox_out & 0xF
            f_hi[a, b] = ZETA ** out_hi
            f_lo[a, b] = ZETA ** out_lo
            
    return f_hi, f_lo

def main() -> None:
    print("[INFO] Building ζ-domain target tables for Inverse S-Box…")
    f_hi_target, f_lo_target = build_target_tables()

    print("[INFO] Computing 2-D inverse FFTs to find coefficients…")
    C_hi = np.fft.ifft2(f_hi_target)
    C_lo = np.fft.ifft2(f_lo_target)

    print("[INFO] Saving coefficients to JSON…")
    obj = {
        "shape": [16, 16],
        "sbox_upper_mv_coeffs_real": C_hi.real.tolist(),
        "sbox_upper_mv_coeffs_imag": C_hi.imag.tolist(),
        "sbox_lower_mv_coeffs_real": C_lo.real.tolist(),
        "sbox_lower_mv_coeffs_imag": C_lo.imag.tolist(),
        "note": "Monomial coefficients for AES Inverse S-Box (ζ-input to ζ-output model) via ifft2",
    }
    OUT_PATH.write_text(json.dumps(obj, indent=2))
    print(f"✅ Coefficients for Inverse S-Box (zeta→zeta model) written → {OUT_PATH}")

if __name__ == "__main__":
    main()