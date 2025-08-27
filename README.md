## Project Architecture Analysis

This analysis is based on the overall file structure and the central orchestration script, `aes_main_process.py`.

### Main Modules and Their Responsibilities

The project follows a highly modular design, with each component of the AES algorithm isolated in its own file. This promotes clarity and focused testing.

- **`aes_main_process.py`**: The central nervous system of the project. It orchestrates the entire FHE-AES encryption and decryption process, from setting up the encryption engine to sequencing the AES rounds.

- **`engine_context.py`**: Manages the FHE (CKKS) engine. It is responsible for creating and holding the encryption keys (public, secret), managing engine parameters (e.g., `mode`, `use_bootstrap`), and providing the core `engine` object used for all homomorphic operations.

- **Data Preparation Modules**:
  - **`aes_block_array.py`**: Converts standard 16-byte AES blocks into a flattened, 1-D array suitable for batched processing in the FHE scheme.
  - **`aes_split_to_nibble.py`**: Splits each byte in the flattened array into its upper and lower 4-bit nibbles. This is a crucial step for making the non-linear S-Box operation manageable in FHE.
  - **`aes_transform_zeta.py`**: Transforms the integer nibbles into a complex number representation (ζ^n), which is the format required for encoding data into polynomials for the CKKS scheme.
  - **`aes_key_array.py`**: Performs a similar flattening operation for the AES key as `aes_block_array.py` does for the data.

- **AES Operation Modules**:
  - Each core AES step is implemented in its own file: `aes_SubBytes.py`, `aes_ShiftRows.py`, `aes_MixColumns.py`, `aes_key_scheduling.py`, and `aes_xor.py` (for AddRoundKey). There are corresponding inverse operation files as well (`aes_inv_*`).
  - These modules contain the logic to perform the AES operations on data that is already encrypted.

- **Coefficient Files (`coeffs/*.json`)**:
  - These are critical data files, not code modules. They store pre-computed coefficients for the polynomial functions that approximate the AES operations, most notably the S-Box (`sbox_coeffs.json`) and the GF(2^8) multiplications in MixColumns (`gf_mult*_coeffs.json`).

### Data Flow and Dependencies

The data flows through the system in a sequential pipeline orchestrated by `aes_main_process.py`:

1.  **Engine Setup**: `engine_initiation` is called to create the `CKKS_EngineContext`.
2.  **Plaintext Preparation**: Raw data (and the AES key) is processed through `blocks_to_flat_array` -> `split_to_nibbles` -> `int_to_zeta` to convert it from bytes to zeta-encoded complex numbers.
3.  **Homomorphic Encryption**: The prepared data and key arrays are encrypted using the `engine` from the context, turning them into ciphertexts.
4.  **Key Scheduling**: The encrypted key is passed to `key_scheduling` to generate all the necessary round keys in their encrypted form.
5.  **AES Rounds (Encryption)**: The main script iterates through 10 rounds:
    - **Round 1-9**: `SubBytes` -> `ShiftRows` -> `MixColumns` -> `AddRoundKey`.
    - **Round 10**: `SubBytes` -> `ShiftRows` -> `AddRoundKey`.
    Each step takes the encrypted data from the previous step and applies the corresponding homomorphic operation.
6.  **AES Rounds (Decryption)**: A similar process is followed in reverse using the inverse operation modules.
7.  **Final Decryption**: The final ciphertext is decrypted using the `engine` and the secret key to produce the result.

### Potential Architectural Issues and Considerations

1.  **Repetitive Code**: The main execution block in `aes_main_process.py` contains manually coded sequences for each of the 10 encryption and decryption rounds. This could be refactored into loops to improve readability and reduce code duplication. For instance, rounds 1-9 of encryption are identical and are a prime candidate for a loop.

2.  **Dynamic Imports**: The use of `importlib` to load modules with hyphens in their filenames (e.g., `aes-xor-archive.py`) is non-standard. Renaming the files to use underscores (e.g., `aes_xor_archive.py`) would allow for conventional `import` statements, making the code more idiomatic.

3.  **Hardcoded Parameters**: FHE engine parameters (like `delta_list`, `thread_count`) are hardcoded in the `if __name__ == "__main__"` block. For greater flexibility and reusability, these could be externalized into a configuration file (e.g., YAML, JSON) or passed as command-line arguments.

4.  **Lack of Higher-Level Abstraction**: The main script is very low-level, manually calling every single AES step for every round. Introducing a class, such as `FHE_AES_Cipher`, could encapsulate the entire round-based logic. The main script would then simply need to instantiate this class and call a high-level method like `cipher.encrypt(encrypted_data)` or `cipher.decrypt(encrypted_data)`, making the main execution flow much cleaner.

## Homomorphic XOR Operation

The `aes_xor.py` script implements a homomorphic XOR operation on two encrypted 4-bit nibbles. This is a fundamental component of the `AddRoundKey` step in the AES algorithm.

### Execution Analysis

The core challenge is performing a bitwise XOR operation on encrypted data. Since the CKKS FHE scheme operates on complex numbers, a direct bitwise operation is not possible. The strategy is as follows:

1.  **Zeta Transformation**: The 4-bit integers (nibbles, 0-15) are transformed into 16th roots of unity (ζ^k), where ζ = e^(-2πi/16). This maps the integers to unique points on the complex unit circle.
2.  **Polynomial Evaluation**: The XOR operation in this zeta domain is equivalent to evaluating a specific bivariate polynomial, `P(x, y)`. The script evaluates this polynomial on the two encrypted inputs (`enc_alpha`, `enc_beta`).
3.  **Coefficient-based Computation**: The polynomial `P(x, y)` is expressed as a sum of terms `c_ij * x^i * y^j`. The coefficients `c_ij` are pre-computed and stored in `coeffs/xor_mono_coeffs.json`. The script performs the homomorphic computation by multiplying the encrypted inputs raised to the required powers with these pre-computed coefficients.

### Data Flow

The data flow within the `_xor_operation` function is as follows:

1.  **Inputs**: Two ciphertexts, `enc_alpha` and `enc_beta`, which are the encrypted zeta representations of the two nibbles to be XORed.
2.  **Power Basis Generation**:
    *   The `build_power_basis` function is called for both `enc_alpha` and `enc_beta`.
    *   It computes the powers of the ciphertext from 1 to 8 (`ct^1` to `ct^8`) using `engine.make_power_basis`.
    *   It then computes the powers from 9 to 15 by taking the cryptographic `conjugate` of the first 7 powers. In the CKKS scheme, `conjugate(ct^k)` is equivalent to `ct^-k`, and for a 16th root of unity, `ζ^-k` is the same as `ζ^(16-k)`. This is a highly efficient way to compute all necessary powers.
    *   The result is a dictionary mapping exponents (0-15) to their corresponding ciphertexts.
3.  **Coefficient Loading**:
    *   The `_get_coeff_plaintexts` function loads the pre-computed coefficients from `coeffs/xor_mono_coeffs.json`.
    *   These coefficients are encoded into FHE plaintexts, ready for homomorphic multiplication.
4.  **Polynomial Evaluation**:
    *   The script initializes an empty result ciphertext.
    *   It iterates through the loaded coefficients. For each non-zero coefficient `c_ij`, it retrieves the corresponding powered ciphertexts `base_x[i]` and `base_y[j]`.
    *   It computes the term `(base_x[i] * base_y[j]) * c_ij` using homomorphic multiplications (`engine.multiply`).
    *   This result is added to the total sum using homomorphic addition (`engine.add`).
5.  **Output**: The function returns a single ciphertext that encrypts the zeta representation of the XORed result. This output ciphertext can then be used in subsequent homomorphic operations or be decrypted to reveal the final integer result.

## Noise Reduction (`aes_noise_reduction.py`)

The `aes_noise_reduction.py` script provides a function to manage and reduce the noise accumulated in a ciphertext during homomorphic operations. This is crucial for executing deep computational graphs, such as the multiple rounds in AES, without the noise overwhelming the signal and corrupting the result.

### `noise_reduction(engine_context, state, n=16)`

This is the primary function in the module. It applies a specific noise reduction algorithm to a given ciphertext.

**Parameters:**

-   `engine_context`: The context object holding the FHE engine and associated keys.
-   `state`: The input ciphertext whose noise needs to be reduced.
-   `n`: An integer parameter that controls the noise reduction process. It defaults to `16` and can also be `256`. This parameter influences the number of iterations (`p`) in the core loop.

**Functionality and Data Flow:**

The function implements a noise management technique that appears to be a custom polynomial approximation designed to recenter the ciphertext value and reduce its noise level. The process is as follows:

1.  **Initialization**:
    *   The FHE `engine` is retrieved from the `engine_context`.
    *   The input `state` is copied to a variable `a` to preserve the original ciphertext.
    *   An iteration parameter `p` is set based on `n`:
        *   If `n` is 256, `p` is 8.
        *   If `n` is 16, `p` is 4.

2.  **Core Squaring Loop**:
    *   The function iterates `p` times. In each iteration, it squares the `state` ciphertext using `engine.multiply(state, state, ...)`.
    *   Before each multiplication, it checks if the ciphertext's level is 0. If it is, a `bootstrap` operation is performed to refresh the ciphertext, enabling further computations. This repeated squaring rapidly increases the magnitude of the underlying value, which is a key part of the noise reduction scheme.

3.  **Polynomial Application**:
    *   After the loop, the function computes two main terms that are later added together. This part of the process resembles the evaluation of a polynomial `f(x) = c1*x + c0` where `x` is the result of the squaring loop.
    *   **First Term**:
        1.  The result from the squaring loop (`state`) is multiplied by the original ciphertext (`a`).
        2.  This product is then multiplied by the plaintext constant `(-1/n)`.
    *   **Second Term**:
        1.  The original ciphertext `a` is multiplied by the plaintext constant `(1 + 1/n)`.

4.  **Final Combination and Output**:
    *   The two terms are added together using `engine.add(state, a)`.
    *   The final result is bootstrapped to ensure it is "fresh" and has minimal noise.
    *   An `engine.intt` (Inverse Number Theoretic Transform) is applied, which is likely the final step to get the result back into a standard polynomial representation within the CKKS scheme.
    *   The resulting ciphertext `out` is returned.

Throughout this process, bootstrapping is performed whenever a ciphertext's level drops to 0, which is a standard practice in leveled FHE schemes to allow for arbitrary-depth computations.

### noise_reduction과 doublefree
aes_key_scheduling.py에서 noise_reduction을 사용하기 위해 _row_word, _sub_word, _rcon_xor, _xor의 '내부'에 noise_reduction을 추가했으나 이 과정을 거치고 return되며 double-free가 발생하였다. 이유는 알 수 없으나 함수 내 함수로 배치하는 경우 발생하는 것으로 판단하여 모든 noise_reduction 함수를 밖으로 배치하였다.
