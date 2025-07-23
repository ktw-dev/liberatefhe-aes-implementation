import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Import helper from project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from aes_transform_zeta import transform_to_zeta  # pylint: disable=import-error
from desilofhe import Engine  # pylint: disable=import-error

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
SLOT_COUNT = 32768  # 2^15
DEGREE      = 15    # Maximum exponent needed for XOR polynomial
COEFFS_JSON = Path(__file__).with_name("fft_coeffs.json")

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def zeta_to_int(zeta_arr: np.ndarray) -> np.ndarray:
    """Inverse of `transform_to_zeta` assuming unit-magnitude complex numbers.

    Values are mapped back to integers 0‥15 by measuring their phase.
    """
    angles = np.angle(zeta_arr)  # range (-π, π]
    k      = (-angles * 16) / (2 * np.pi)
    k      = np.mod(np.rint(k), 16).astype(np.uint8)
    return k


def build_power_basis(engine: Engine, ct, relin_key, public_key):
    """Return list of ciphertexts ct^k for k = 0..DEGREE."""
    basis = [engine.encrypt(np.ones(SLOT_COUNT), public_key)]  # k = 0
    basis.append(ct)                                           # k = 1
    for k in range(2, DEGREE + 1):
        new_ct = engine.multiply(basis[-1], ct, relin_key)
        basis.append(new_ct)
    return basis


# -----------------------------------------------------------------------------
# 1. Prepare inputs
# -----------------------------------------------------------------------------
np.random.seed(42)
alpha_int = np.random.randint(0, 16, size=SLOT_COUNT, dtype=np.uint8)
beta_int  = np.random.randint(0, 16, size=SLOT_COUNT, dtype=np.uint8)
expected_int = np.bitwise_xor(alpha_int, beta_int)

# Map to zeta domain
alpha = transform_to_zeta(alpha_int)
beta  = transform_to_zeta(beta_int)

# -----------------------------------------------------------------------------
# 2. Initialise engine & keys
# -----------------------------------------------------------------------------
engine = Engine(log_coeff_count=16, special_prime_count=1, mode="cpu")
secret_key    = engine.create_secret_key()
public_key    = engine.create_public_key(secret_key)
relin_key     = engine.create_relinearization_key(secret_key)
conjugate_key = engine.create_conjugation_key(secret_key)  # might be unused directly

# -----------------------------------------------------------------------------
# 3. Encrypt inputs
# -----------------------------------------------------------------------------
enc_alpha = engine.encrypt(alpha, public_key)
enc_beta  = engine.encrypt(beta,  public_key)

# -----------------------------------------------------------------------------
# 4. Build power bases
# -----------------------------------------------------------------------------
print("[INFO] Building power bases …")
base_x = build_power_basis(engine, enc_alpha, relin_key, public_key)
base_y = build_power_basis(engine, enc_beta,  relin_key, public_key)

# -----------------------------------------------------------------------------
# 5. Load polynomial coefficients (sparse)
# -----------------------------------------------------------------------------
print("[INFO] Loading polynomial coefficients …")
with open(COEFFS_JSON, "r", encoding="utf-8") as f:
    coeff_data = json.load(f)

# Dict[(i, j)] -> complex
coeffs: Dict[Tuple[int, int], complex] = {}
for entry in coeff_data:
    i = int(entry["x_exp"])
    j = int(entry["y_exp"])
    c = complex(entry["real"], entry["imag"])
    if c != 0:
        coeffs[(i, j)] = c

print(f"[INFO] Non-zero coefficients loaded: {len(coeffs)} (≈{len(coeffs)/(DEGREE+1)**2:.2%})")

# -----------------------------------------------------------------------------
# 6. Evaluate polynomial securely
# -----------------------------------------------------------------------------
print("[INFO] Evaluating XOR polynomial …")
zero_ct = engine.encrypt(np.zeros(SLOT_COUNT), public_key)
cipher_res = zero_ct

for (i, j), coeff in coeffs.items():
    term = engine.multiply(base_x[i], base_y[j], relin_key)

    # Real component (scalar multiply)
    real_part = coeff.real
    real_ct   = engine.multiply(term, real_part)

    # Imag component (plaintext vector multiply)
    imag_part = coeff.imag
    if imag_part != 0:
        imag_vector = np.full(SLOT_COUNT, imag_part * 1j, dtype=complex)
        imag_pt     = engine.encode(imag_vector)
        imag_ct     = engine.multiply(term, imag_pt)
        term_total  = engine.add(real_ct, imag_ct)
    else:
        term_total  = real_ct

    cipher_res = engine.add(cipher_res, term_total)

# -----------------------------------------------------------------------------
# 7. Decrypt & verify
# -----------------------------------------------------------------------------
print("[INFO] Decrypting …")
decoded_zeta = engine.decrypt(cipher_res, secret_key)

decoded_int = zeta_to_int(decoded_zeta)

try:
    np.testing.assert_array_equal(decoded_int, expected_int)
    print("[PASS] XOR homomorphic evaluation matches plaintext XOR.")
except AssertionError as err:
    mismatch = np.count_nonzero(decoded_int != expected_int)
    print(f"[FAIL] XOR result mismatch in {mismatch}/{SLOT_COUNT} slots.")
    raise err 