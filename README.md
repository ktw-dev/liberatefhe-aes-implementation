
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
