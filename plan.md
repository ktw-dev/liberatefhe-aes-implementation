# XOR Verification Experiment Plan

This document describes the high-level steps we will follow to build **`test.py`** that verifies the custom XOR circuit with the compressed polynomial coefficients stored in **`xor-op/`**.

---

## 1. Prepare the Environment

1.1  Install the `desilofhe` library (if not already available).

1.2  Confirm that the files below exist inside `xor-op/` and contain the compressed FFT coefficient data that encodes the XOR polynomial:

* `fft_coeffs.json` – list of objects with keys `x_exp`, `y_exp`, `real`, `imag` (or similar)
* `fft_coeffs.npy` – optional NumPy binary of same data (used only if helpful)

---

## 2. Load & Parse Coefficients

2.1  Read **`xor-op/fft_coeffs.json`**.

2.2  Convert each entry to: `(i, j) → complex(real, imag)` where `i` = exponent of `x`, `j` = exponent of `y`.

2.3  Build a Python dict `coeffs[(i, j)] = complex_value` **containing only non-zero coefficients** to keep the evaluation loop tight (≈33 % sparsity).

---

## 3. Generate / Load Plaintext Inputs

3.1  Create two NumPy arrays `alpha` and `beta` of shape `(32768,)` with values in `[0, 15]` (4-bit numbers).

* Option A: load from existing test vectors (if provided later).
* Option B: generate random vectors with `np.random.randint(0, 16, size=32768)`.

3.2  Compute the **expected** XOR result in plaintext for later comparison: `expected_int = np.bitwise_xor(alpha_int, beta_int)`.

3.3  Convert the integer arrays into their 16-th primitive-root (zeta) representation before encryption:

```python
from aes_transform_zeta import transform_to_zeta  # or copied helper
alpha = transform_to_zeta(alpha_int)
beta  = transform_to_zeta(beta_int)
```

`alpha` and `beta` are now complex-valued NumPy arrays (shape 32,768).

---

## 4. Initialise Desilo Engine & Keys

4.1

```python
from desilofhe import Engine
engine = Engine(log_coeff_count=16, special_prime_count=1, mode="cpu")
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relin_key = engine.create_relinearization_key(secret_key)
conjugate_key = engine.create_conjugation_key(secret_key)
```

---

## 5. Encrypt Inputs

5.1  Encrypt `alpha` and `beta` (already in zeta domain) with `engine.encrypt(data, public_key)` → ciphertexts `enc_alpha`, `enc_beta`.

---

## 6. Build Power Bases (up to degree 15)

6.1  Use `engine.make_power_basis(ciphertext, 8)` to get powers `x¹ … x⁸` (for each operand as needed).

6.2  Use `engine.conjugate(power_basis)` to extend to powers 9-15.

6.3  Store results in a dict `basis_x[k]`, `basis_y[k]` where `k` ranges 0-15.

---

## 7. Evaluate XOR Polynomial (with complex coefficients)

```python
# helper to build a plaintext vector filled with a constant

const_vec = lambda c: [c] * 32768  # shape matches slot count

cipher_res = engine.encrypt([0]*32768, public_key)  # start with zero
for (i, j), coeff in coeffs.items():  # iterate ONLY over non-zero terms

# 1) compute x^i * y^j  (both ciphertexts)

term = engine.multiply(basis_x[i], basis_y[j], relin_key)

# 2) split complex coefficient into real & imaginary parts

real_part = coeff.real
imag_part = coeff.imag

# 3-a) real part – scalar multiply (ciphertext * constant)

real_ct = engine.multiply(term, real_part)  # constant multiplication

# 3-b) imaginary part – plaintext vector multiply (ciphertext * plaintext)

if abs(imag_part) > 0:
    # create a complex vector (imag_part * i) across all slots
    imag_vector = np.full(32768, imag_part * 1j, dtype=complex)
    imag_pt = engine.encode(imag_vector)
imag_ct = engine.multiply(term, imag_pt)       # cm-multiplication
term_total = engine.add(real_ct, imag_ct)
else:
term_total = real_ct

# 4) accumulate into result

cipher_res = engine.add(cipher_res, term_total)
```

Key points:

1. **Scalar vs. Plaintext multiply** – `engine.multiply(ciphertext, scalar)` performs constant multiplication, while `engine.multiply(ciphertext, plaintext)` carries out ciphertext-plaintext (cm) multiplication.
2. Building the plaintext vector with `engine.encode` ensures the imaginary coefficient is correctly embedded across all slots.
3. Repeat the same logic for every non-zero `(i, j)` coefficient.

---

## 8. Decrypt & Verify

8.1  Decrypt: `decoded_zeta = engine.decrypt(cipher_res, secret_key)`.

8.2  Convert back to integers using the inverse transform:

```python
def zeta_to_transform(zeta_arr: np.ndarray) -> np.ndarray:
    # Map complex values back to 0–15 by measuring angle.
    angles = np.angle(zeta_arr)        # range (-π, π]
    # Because we used exp(-2πi k / 16), k = (-angle * 16) / (2π)
    k = (-angles * 16) / (2*np.pi)
    k = np.mod(np.rint(k), 16).astype(np.uint8)  # round & wrap
    return k

decoded_int = zeta_to_transform(decoded_zeta)
```

8.3  Compare to expected integers:

```python
np.testing.assert_array_equal(decoded_int, expected_int)
```

8.4  Print **PASS** if assertion holds; otherwise print diagnostics.

---

## 9. Performance / Correctness Extras (Optional)

* Time each major step (encryption, basis construction, evaluation).
* Optionally vectorise term accumulation to minimise ciphertext operations (e.g., Horner-style).
* Explore unrolling loops vs. list-comprehension for clarity.

---

## 10. File Structure to Produce

```
liberatefhe-aes-implementation/
├── test.py        # implements steps 2-8
└── xor-op/
    ├── fft_coeffs.json
    └── ...
```

* `test.py` will be independent of `aes-xor.py` as requested.

---

## 11. Next Steps

* Get user confirmation on this plan.
* Implement `test.py` following the outlined steps.
* Run the script and share results.

