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

def transform_to_zeta(arr: np.ndarray) -> np.ndarray:
    result = np.exp(-2j * np.pi * (arr % 16) / 16)
    return result

def zeta_to_int(zeta_arr: np.ndarray) -> np.ndarray:
    """Inverse of `transform_to_zeta` assuming unit-magnitude complex numbers.

    Values are mapped back to integers 0‥15 by measuring their phase.
    """
    angles = np.angle(zeta_arr)  # range (-π, π]
    k      = (-angles * 16) / (2 * np.pi)
    k      = np.mod(np.rint(k), 16).astype(np.uint8)
    return k


def ones_cipher(engine: Engine, template_ct):
    """Return ciphertext with all slots = 1 matching scale/level of template_ct."""
    # Create zero ciphertext with same scale/level via scalar multiply
    zero_ct = engine.multiply(template_ct, 0.0)  # keeps params, slots all 0

    try:
        # Preferred path if library supports add_plain
        ones_ct = engine.add_plain(zero_ct, 1.0)
    except AttributeError:
        # Fallback: encode plaintext ones then add as ciphertext-plaintext
        ones_pt = engine.encode(np.ones(SLOT_COUNT))
        ones_ct = engine.add(zero_ct, ones_pt)
    return ones_ct


def build_power_basis(engine: Engine, ct, relin_key, conj_key, public_key):
    """Return dict exp→ct for exponents 0‥15 using power_basis + conjugates.

    Steps:
    1. `engine.make_power_basis(ct, 8, relin_key)` → ct^1..ct^8.
    2. Conjugate the first 7 powers to obtain ct^-1..ct^-7 ≡ ct^15..ct^9.
    3. Add exponent 0 as an encryption of the all-ones vector.
    """
    # Positive powers 1..8
    pos_basis = engine.make_power_basis(ct, 8, relin_key)  # list length 8

    basis: Dict[int, object] = {}
    basis[0] = ones_cipher(engine, ct)

    for idx, c in enumerate(pos_basis, start=1):
        basis[idx] = c  # exponents 1..8

    # Negative powers: ct^-k  (k=1..7) → ct^(16-k)
    for k in range(1, 8):
        conj_ct = engine.conjugate(pos_basis[k - 1], conj_key)  # ct^-k
        basis[16 - k] = conj_ct  # exponents 15..9

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
print("[INFO] Initialising engine …")
engine = Engine(log_coeff_count=16, special_prime_count=1, mode="cpu")
secret_key    = engine.create_secret_key()
public_key    = engine.create_public_key(secret_key)
relin_key     = engine.create_relinearization_key(secret_key)
conjugate_key = engine.create_conjugation_key(secret_key)  # might be unused directly

# -----------------------------------------------------------------------------
# 3. Encrypt inputs
# -----------------------------------------------------------------------------
print("[INFO] Encrypting inputs …")
enc_alpha = engine.encrypt(alpha, public_key)
enc_beta  = engine.encrypt(beta,  public_key)

# -----------------------------------------------------------------------------
# 4. Build power bases
# -----------------------------------------------------------------------------
print("[INFO] Building power bases …")
base_x = build_power_basis(engine, enc_alpha, relin_key, conjugate_key, public_key)
base_y = build_power_basis(engine, enc_beta,  relin_key, conjugate_key, public_key)

print("[DEBUG] sorted(base_x.keys()) =", sorted(base_x.keys()))
print("[DEBUG] sorted(base_y.keys()) =", sorted(base_y.keys()))

# -----------------------------------------------------------------------------
# 5. Load polynomial coefficients (sparse)
# -----------------------------------------------------------------------------
print("[INFO] Loading polynomial coefficients …")
with open(COEFFS_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# entries are [[i, j, real, imag], ...]
entries = data.get("entries", [])

# Dict[(i, j)] -> complex
coeffs: Dict[Tuple[int, int], complex] = {}
for entry in entries:
    if len(entry) != 4:
        continue
    i, j, real_val, imag_val = entry
    c = complex(real_val, imag_val)
    if abs(c) > 0:
        coeffs[(int(i), int(j))] = c

print(f"[INFO] Non-zero coefficients loaded: {len(coeffs)} (≈{len(coeffs)/(DEGREE+1)**2:.2%})")

# # enc_alpha / basis_x[1] 복호해 보기
# print("alpha[0:5] =", alpha_int[:5])
# print("dec alpha  =", zeta_to_int(engine.decrypt(enc_alpha, secret_key)[:5]))

# for k in [1, 2, 3]:
#     print(f"⟨x^{k}⟩[0] =", engine.decrypt(base_x[k], secret_key)[0])
#     print(f"⟨y^{k}⟩[0] =", engine.decrypt(base_y[k], secret_key)[0])
    
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

# Subtract constant coefficient so result lies on unit circle for angle decoding
if (const_val := coeffs.get((0, 0))) is not None:
    ones_ct = base_x[0]  # encryption of ones
    const_ct = engine.multiply(ones_ct, const_val.real)  # purely real
    cipher_res = engine.subtract(cipher_res, const_ct)

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