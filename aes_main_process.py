"""aes-main-process.py

High-level orchestration script for (future) fully homomorphic AES evaluation.
Currently implements only the *data initiation* stage:
    1. Build random AES state blocks (user-selected count ≤ 2048)
    2. Flatten to 1-D array following batching layout (via aes-block-array.py)
    3. Split each byte into upper / lower 4-bit nibbles (via aes-split-to-nibble.py)

Sub-modules have dashes in their filenames, so they are loaded dynamically with
importlib to avoid name conflicts.

Future work (place-holders):
    • key_scheduling(...)
    • sub_bytes(...)
    • shift_rows(...)
    • mix_columns(...)
    • add_round_key(...)
"""
from __future__ import annotations

import pathlib
import importlib.util
import numpy as np
import time
from typing import Tuple, List, Any

# import custom modules
from engine_context import CKKS_EngineContext
from aes_block_array import blocks_to_flat_array
from aes_split_to_nibble import split_to_nibbles
from aes_key_array import key_to_flat_array
from aes_transform_zeta import int_to_zeta, zeta_to_int
from aes_xor import _xor_operation
from aes_key_scheduling import key_scheduling
from key_scheduling_numpy import key_expansion_flat_nibbles

# operation modules
from aes_inv_SubBytes import inv_sub_bytes as _inv_sub_bytes
from aes_inv_ShiftRows import inv_shift_rows as _inv_shift_rows
from aes_inv_MixColumns import inv_mix_columns as _inv_mix_columns
from aes_SubBytes import sub_bytes as _sub_bytes
from aes_ShiftRows import shift_rows as _shift_rows
from aes_MixColumns import mix_columns as _mix_columns

# demo modules
from aes_128_numpy import make_all_simd_round_key_vectors

# -----------------------------------------------------------------------------
# Engine Initiation ------------------------------------------------------------
# -----------------------------------------------------------------------------

def engine_initiation(
                signature: int, 
                *,
                max_level: int = 30, 
                mode: str = 'cpu', 
                use_bootstrap: bool = False, 
                use_multiparty: bool = False, 
                thread_count: int = 0, 
                device_id: int = 0, 
                fixed_rotation: bool = False, 
                delta_list: list[int] = None, 
                log_coeff_count: int = 0, 
                special_prime_count: int = 0) -> CKKS_EngineContext:
    """Create engine and all keys, returning a bundled FHEContext."""
    
    print("create engine\n")
    
    engine_context = CKKS_EngineContext(signature, max_level=max_level, mode=mode, use_bootstrap=use_bootstrap, use_multiparty=use_multiparty, thread_count=thread_count, device_id=device_id, fixed_rotation=fixed_rotation, delta_list=delta_list, log_coeff_count=log_coeff_count, special_prime_count=special_prime_count)
    
    print("engine created\n")
    return engine_context

# -----------------------------------------------------------------------------
# Data initiation --------------------------------------------------------------
# -----------------------------------------------------------------------------

def data_initiation(num_blocks: int, *, rng: np.random.Generator | None = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare initial plaintext data arrays for FHE-AES pipeline.

    Parameters
    ----------
    num_blocks : int
        Number of AES 16-byte state blocks to generate (0 ≤ N ≤ 2048).
    rng : np.random.Generator, optional
        Custom random generator; default uses NumPy default_rng().

    Returns
    -------
    blocks : (N,16) uint8 ndarray
        Raw random blocks (row-major).
    flat   : (16*2048,) uint8 ndarray
        Padded & flattened block array suitable for batching.
    upper  : (16*2048,) uint8 ndarray
        Upper nibbles (hi) of *flat*.
    lower  : (16*2048,) uint8 ndarray
        Lower nibbles (lo) of *flat*.
    zeta_upper : (16*2048,) complex128 ndarray
        ζ^(upper) values.
    zeta_lower : (16*2048,) complex128 ndarray
        ζ^(lower) values.
    """
    if not (1 <= num_blocks <= 2048):
        raise ValueError("num_blocks must be between 1 and 2048 inclusive")

    rng = rng or np.random.default_rng()

    # 1. Generate random data-blocks
    blocks = rng.integers(0, 256, size=(num_blocks, 16), dtype=np.uint8)
    
    print("blocks: ", blocks)

    # 2. Flatten to 1-D array following batching layout
    flat = blocks_to_flat_array(blocks)

    # 3. Split each byte into upper / lower 4-bit nibbles
    upper, lower = split_to_nibbles(flat)

    # 4. ζ-변환 (SIMD-style) – repeatable vectorized op
    zeta_upper = int_to_zeta(upper)
    zeta_lower = int_to_zeta(lower)

    return blocks, flat, upper, lower, zeta_upper, zeta_lower

# -----------------------------------------------------------------------------
# Data initiation demo --------------------------------------------------------------
# -----------------------------------------------------------------------------

def data_initiation_demo() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare initial plaintext data arrays for FHE-AES pipeline.

    Parameters
    ----------
    num_blocks : int
        Number of AES 16-byte state blocks to generate (0 ≤ N ≤ 2048).

    Returns
    -------
    blocks : (N,16) uint8 ndarray
        Raw random blocks (row-major).
    flat   : (16*2048,) uint8 ndarray
        Padded & flattened block array suitable for batching.
    upper  : (16*2048,) uint8 ndarray
        Upper nibbles (hi) of *flat*.
    lower  : (16*2048,) uint8 ndarray
        Lower nibbles (lo) of *flat*.
    zeta_upper : (16*2048,) complex128 ndarray
        ζ^(upper) values.
    zeta_lower : (16*2048,) complex128 ndarray
        ζ^(lower) values.
    """
    # 1. Generate random data-blocks
    blocks = np.array([0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34], dtype=np.uint8)
    
    print("blocks: ", blocks)

    # 2. Flatten to 1-D array following batching layout
    flat = blocks_to_flat_array(blocks)
    
    print("flat: ", flat)

    # 3. Split each byte into upper / lower 4-bit nibbles
    upper, lower = split_to_nibbles(flat)

    # 4. ζ-변환 (SIMD-style) – repeatable vectorized op
    zeta_upper = int_to_zeta(upper)
    zeta_lower = int_to_zeta(lower)

    return zeta_upper, zeta_lower
# -----------------------------------------------------------------------------
# Key initiation --------------------------------------------------------------
# -----------------------------------------------------------------------------


def key_initiation(*, rng: np.random.Generator | None = None, max_blocks: int = 2048
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random AES-128 key and prepare its flat & nibble arrays.

    Returns
    -------
    key_bytes  : (16,) uint8    – secret key
    key_flat   : (16*max_blocks,) uint8 – replicated key array
    key_upper  : (16*max_blocks,) uint8 – upper 4-bit nibbles
    key_lower  : (16*max_blocks,) uint8 – lower 4-bit nibbles
    key_zeta_upper : (16*max_blocks,) complex128 – ζ^upper
    key_zeta_lower : (16*max_blocks,) complex128 – ζ^lower
    """
    rng = rng or np.random.default_rng()
    key = rng.integers(0, 256, size=16, dtype=np.uint8)

    key_flat = key_to_flat_array(key, max_blocks)
    key_upper, key_lower = split_to_nibbles(key_flat)

    key_zeta_upper = int_to_zeta(key_upper)
    key_zeta_lower = int_to_zeta(key_lower)

    return key, key_flat, key_upper, key_lower, key_zeta_upper, key_zeta_lower

# -----------------------------------------------------------------------------
# demo Key Scheduling ---------------------------------------------------------
# -----------------------------------------------------------------------------

def demo_key_scheduling(engine_context: CKKS_EngineContext | None = None) -> Tuple[List[Any], List[Any]]:
    """Generate SIMD-batched round keys, map to ζ-domain per nibble, and encrypt.

    Returns
    -------
    enc_round_keys_upper : list
        List of 11 ciphertexts encrypting ζ^(upper nibble) for each round key.
    enc_round_keys_lower : list
        List of 11 ciphertexts encrypting ζ^(lower nibble) for each round key.
    """
    # Ensure we have an engine context
    if engine_context is None:
        assert False, "engine_context is required"

    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()

    # 1) Build row-major SIMD arrays (11, 32768) as uint8
    r_key_list = make_all_simd_round_key_vectors(
        bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
        2048
    )

    # 2) Split to nibbles
    upper = ((r_key_list >> 4) & 0x0F).astype(np.uint8)  # (11, 32768)
    lower = (r_key_list & 0x0F).astype(np.uint8)         # (11, 32768)

    # 3) Map to ζ-domain
    zeta_upper = int_to_zeta(upper)
    zeta_lower = int_to_zeta(lower)

    # 4) Encrypt per round
    enc_upper: List[Any] = []
    enc_lower: List[Any] = []
    for r in range(11):
        ct_u = engine.encrypt(zeta_upper[r], public_key, level=10)
        ct_l = engine.encrypt(zeta_lower[r], public_key, level=10)
        enc_upper.append(ct_u)
        enc_lower.append(ct_l)

    return enc_upper, enc_lower


# -----------------------------------------------------------------------------
# Key Scheduling --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _key_scheduling(engine_context, enc_key_hi, enc_key_lo):
    
    enc_key_hi_list = []
    enc_key_lo_list = []
    
    mask_row_0 = np.concatenate((np.ones(4 * 2048), np.zeros(12 * 2048)))
    mask_row_1 = np.concatenate((np.zeros(4 * 2048), np.ones(4 * 2048), np.zeros(8 * 2048)))
    mask_row_2 = np.concatenate((np.zeros(8 * 2048), np.ones(4 * 2048), np.zeros(4 * 2048)))
    mask_row_3 = np.concatenate((np.zeros(12 * 2048), np.ones(4 * 2048)))    
    
    row_hi_0 = engine.multiply(enc_key_hi, mask_row_0)
    row_hi_1 = engine.multiply(enc_key_hi, mask_row_1)
    row_hi_2 = engine.multiply(enc_key_hi, mask_row_2)
    row_hi_3 = engine.multiply(enc_key_hi, mask_row_3)
    
    row_lo_0 = engine.multiply(enc_key_lo, mask_row_0)
    row_lo_1 = engine.multiply(enc_key_lo, mask_row_1)
    row_lo_2 = engine.multiply(enc_key_lo, mask_row_2)
    row_lo_3 = engine.multiply(enc_key_lo, mask_row_3)

    enc_key_hi_list.append(row_hi_0)
    enc_key_hi_list.append(row_hi_1)
    enc_key_hi_list.append(row_hi_2)
    enc_key_hi_list.append(row_hi_3)
    
    enc_key_lo_list.append(row_lo_3)
    enc_key_lo_list.append(row_lo_0)
    enc_key_lo_list.append(row_lo_1)
    enc_key_lo_list.append(row_lo_2)

    scheduled_hi_list, scheduled_lo_list = key_scheduling(engine_context, enc_key_hi_list, enc_key_lo_list)
    return scheduled_hi_list, scheduled_lo_list


# -----------------------------------------------------------------------------
# AddRoundKey --------------------------------------------------------------
# -----------------------------------------------------------------------------

def AddRoundKey(engine_context, enc_data, key):
    engine = engine_context.get_engine()
    
    enc_data = _xor_operation(engine_context, enc_data, key)
    enc_data = engine.bootstrap(enc_data, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    return enc_data

# -----------------------------------------------------------------------------
# ShiftRows --------------------------------------------------------------
# -----------------------------------------------------------------------------
def shift_rows(engine_context, enc_data_hi, enc_data_lo):    
    enc_data_hi, enc_data_lo = _shift_rows(engine_context, enc_data_hi, enc_data_lo)
    return enc_data_hi, enc_data_lo

# -----------------------------------------------------------------------------
# SubBytes --------------------------------------------------------------
# -----------------------------------------------------------------------------
def sub_bytes(engine_context, enc_data_hi, enc_data_lo):
    enc_data_hi, enc_data_lo = _sub_bytes(engine_context, enc_data_hi, enc_data_lo)
    return enc_data_hi, enc_data_lo 

# -----------------------------------------------------------------------------
# MixColumns --------------------------------------------------------------
# -----------------------------------------------------------------------------
def mix_columns(engine_context, enc_data_hi, enc_data_lo):
    enc_data_hi, enc_data_lo = _mix_columns(engine_context, enc_data_hi, enc_data_lo)
    return enc_data_hi, enc_data_lo

# -----------------------------------------------------------------------------
# Inverse ShiftRows --------------------------------------------------------------
# -----------------------------------------------------------------------------
def inv_shift_rows(engine_context, enc_data_hi, enc_data_lo):
    enc_data_hi, enc_data_lo = _inv_shift_rows(engine_context, enc_data_hi, enc_data_lo)
    return enc_data_hi, enc_data_lo

# -----------------------------------------------------------------------------
# Inverse SubBytes --------------------------------------------------------------
# -----------------------------------------------------------------------------
def inv_sub_bytes(engine_context, enc_data_hi, enc_data_lo):
    enc_data_hi, enc_data_lo = _inv_sub_bytes(engine_context, enc_data_hi, enc_data_lo)
    return enc_data_hi, enc_data_lo

# -----------------------------------------------------------------------------
# Inverse MixColumns --------------------------------------------------------------
# -----------------------------------------------------------------------------
def inv_mix_columns(engine_context, enc_data_hi, enc_data_lo):
    enc_data_hi, enc_data_lo = _inv_mix_columns(engine_context, enc_data_hi, enc_data_lo)
    return enc_data_hi, enc_data_lo

# -----------------------------------------------------------------------------
# Utility: stage completion ----------------------------------------------------
# -----------------------------------------------------------------------------


def wait_next_stage(stage: str, next_stage: str, delay: float = 1.0) -> None:
    """Print completion banner and sleep *delay* seconds."""
    time.sleep(delay)
    print("\n")
    time.sleep(delay)
    print("--------------------------------")
    time.sleep(delay)
    print(f"{stage} complete!!!", flush= True)
    time.sleep(delay)
    print(f"waiting for the {next_stage} module...", flush=True)
    print("--------------------------------")
    time.sleep(delay)
    print("\n")


def verify_round_output(
    engine_context: CKKS_EngineContext,
    ct_hi,
    ct_lo,
    ground_truth: np.ndarray | list[int],
    *,
    label: str = "",
    mode: str = "demo"
) -> bool:
    """Decrypt *ct_hi/ct_lo*, reconstruct bytes, and compare with *ground_truth*.

    Parameters
    ----------
    engine_context : CKKS_EngineContext
        Context holding engine & secret key.
    ct_hi, ct_lo : Ciphertext
        CKKS ciphertexts holding ζ^(upper nibble) / ζ^(lower nibble) values.
    ground_truth : array-like, length 16
        Expected 16-byte block (row-major order).
    label : str, optional
        Name of the stage for logging.

    Returns
    -------
    bool
        True if all 16 bytes match ground truth, else False.
    """
    if mode == "practice":
        return None
    
    engine = engine_context.get_engine()
    sk = engine_context.get_secret_key()

    gt = np.asarray(ground_truth, dtype=np.uint8).reshape(16)
    
    print("gt: ", gt)

    # Decrypt & convert back to ints 0..15 per nibble
    dec_hi = engine.decrypt(ct_hi, sk)
    dec_lo = engine.decrypt(ct_lo, sk)
    nib_hi = zeta_to_int(dec_hi).astype(np.uint8)
    nib_lo = zeta_to_int(dec_lo).astype(np.uint8)

    # Combine hi/lo nibbles and sample every 2048-slot block (first 16 bytes only)
    slot_stride = 2048
    combined_bytes = ((nib_hi[0::slot_stride] << 4) | nib_lo[0::slot_stride]).astype(np.uint8)
    
    print("combined_bytes length: ", len(combined_bytes))

    print("combined_bytes: ", combined_bytes)

    passed = np.array_equal(combined_bytes, gt)

    report_label = f"[{label}] " if label else ""
    if passed:
        print(f"✓ {report_label}verification passed")
    else:
        print(f"⚠️  {report_label}mismatch – needs review")
        print("  expected:", [f"{b:02x}" for b in gt])
        print("  got     :", [f"{b:02x}" for b in combined_bytes[:16]])
    return passed


# -----------------------------------------------------------------------------
# Demo / manual test -----------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        n_str = input("Enter number of AES blocks (1–2048): ")
        n_blocks = int(n_str)
    except ValueError:
        raise SystemExit("❌  Invalid integer input.")

    # Enforce range 1–2048 inclusive for user input
    if not (1 <= n_blocks <= 2048):
        raise SystemExit("❌  Block count must be between 1 and 2048 inclusive.")

    # --------------------------------------------------------------------
    # Select operating mode (demo vs practice)
    # --------------------------------------------------------------------
    mode_choice = input("Select mode – 'demo' or 'practice': ").strip().lower()
    if mode_choice not in {"demo", "practice"}:
        raise SystemExit("❌  Invalid mode; expected 'demo' or 'practice'.")
    
    # --- Engine initiation stage -----------------------------------------------
    
    delta_list = [1*2048, 2*2048, 3*2048, 4*2048, 8*2048, -1*2048, -2*2048, -3*2048, -4*2048]
    
    engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0, fixed_rotation=True, delta_list=delta_list)
    
    # fixed rotation, delta_list는 필수로 사용함
    
    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()
    secret_key = engine_context.get_secret_key()   

    wait_next_stage("Engine Initiation", "Data initiation")
    
    if mode_choice == "practice":
        # --- Data initiation stage ------------------------------------------------
        blocks, flat, _, _, data_zeta_hi, data_zeta_lo = data_initiation(n_blocks)
    else:
        # --- Data initiation stage ------------------------------------------------
        data_zeta_hi, data_zeta_lo = data_initiation_demo()
        
    wait_next_stage("Data initiation", "key initiation")

    # --- Key initiation stage -------------------------------------------------
    if mode_choice == "practice":
        key_bytes, key_flat, key_upper, key_lower, key_zeta_hi, key_zeta_lo = key_initiation()
    else:
        print("demo mode don't need key initiation. Only demo key scheduling is available.")

    # DEBUG
    # print("Secret key bytes (hex):", [f"{b:02X}" for b in key_bytes])

    # print("ζ(key upper)[0-3]       :", [f"{c:.2f}" for c in key_zeta_hi[:4]])
    # print("ζ(key lower)[0-3]       :", [f"{c:.2f}" for c in key_zeta_lo[:4]])
    
    # --- data/key HE-encryption stage ------------------------------------------------
    
    # 1. 데이터 암호화
    enc_data_hi = engine.encrypt(data_zeta_hi, public_key)
    enc_data_lo = engine.encrypt(data_zeta_lo, public_key)
    
    # 2. 키 암호화
    if mode_choice == "practice":
        enc_key_hi = engine.encrypt(key_zeta_hi, public_key)
        enc_key_lo = engine.encrypt(key_zeta_lo, public_key)
    
    # DEBUG
    # print(enc_data_hi)
    # print(enc_data_lo)
    
    wait_next_stage("data/key HE-encryption", "key Scheduling")

    # --- key scheduling stage ------------------------------------------------
    if mode_choice == "practice":
        enc_key_hi_list, enc_key_lo_list = _key_scheduling(engine_context, key_zeta_hi, key_zeta_lo)    
    else:
        enc_key_hi_list, enc_key_lo_list = demo_key_scheduling(engine_context)

    wait_next_stage("key Scheduling", "encryption stage")
    
    # ========================================================================
    # === Encryption stage ===================================================
    # ========================================================================

    # --- Round 0 --------------------------------------------------------------
    start_time = time.time()
    enc_data_hi_round_1 = AddRoundKey(engine_context, enc_data_hi, enc_key_hi_list[0])
    enc_data_lo_round_1 = AddRoundKey(engine_context, enc_data_lo, enc_key_lo_list[0])
    end_time = time.time()
    print(f"addkey complete!!! Time taken: {(end_time - start_time)} seconds")
    
    verify = verify_round_output(engine_context, enc_data_hi_round_1, enc_data_lo_round_1, ground_truth = [0x2b, 0x28, 0xab, 0x09, 0x7e, 0xae, 0xf7, 0xcf, 0x15, 0xd2, 0x15, 0x4f, 0x16, 0xa6, 0x88, 0x3c], mode=mode_choice)
        
    # --- Round 1 --------------------------------------------------------------
    start_time = time.time()
    sub_s_time = time.time()
    enc_data_hi_round_1, enc_data_lo_round_1 = sub_bytes(engine_context, enc_data_hi_round_1, enc_data_lo_round_1)
    enc_data_hi_round_1 = engine.intt(enc_data_hi_round_1)
    enc_data_lo_round_1 = engine.intt(enc_data_lo_round_1)
    sub_e_time = time.time()
    print(f"sub_bytes complete!!! Time taken: {sub_e_time - sub_s_time} seconds")
    
    verify = verify_round_output(engine_context, enc_data_hi_round_1, enc_data_lo_round_1, ground_truth = [0xd4, 0xe0, 0xb8, 0x1e, 0x27, 0xbf, 0xb4, 0x41, 0x11, 0x98, 0x5d, 0x52, 0xae, 0xf1, 0xe5, 0x30], mode=mode_choice)
    
    shift_s_time = time.time()
    enc_data_hi_round_1, enc_data_lo_round_1 = shift_rows(engine_context, enc_data_hi_round_1, enc_data_lo_round_1)
    enc_data_hi_round_1 = engine.intt(enc_data_hi_round_1)
    enc_data_lo_round_1 = engine.intt(enc_data_lo_round_1)
    shift_e_time = time.time()
    print(f"shift_rows complete!!! Time taken: {shift_e_time - shift_s_time} seconds")
    
    verify = verify_round_output(engine_context, enc_data_hi_round_1, enc_data_lo_round_1, ground_truth = [0xd4, 0xe0, 0xb8, 0x1e, 0xbf, 0xb4, 0x41, 0x27, 0x5d, 0x52, 0x11, 0x98, 0x30, 0xae, 0xf1, 0xe5], mode=mode_choice)
    
    mix_s_time = time.time()
    enc_data_hi_round_1, enc_data_lo_round_1 = mix_columns(engine_context, enc_data_hi_round_1, enc_data_lo_round_1)
    enc_data_hi_round_1 = engine.intt(enc_data_hi_round_1)
    enc_data_lo_round_1 = engine.intt(enc_data_lo_round_1)
    mix_e_time = time.time()
    print(f"mix_columns complete!!! Time taken: {mix_e_time - mix_s_time} seconds")
    
    verify = verify_round_output(engine_context, enc_data_hi_round_1, enc_data_lo_round_1, ground_truth= [0x04, 0xe0, 0x48, 0x28, 0x66, 0xcb, 0xf8, 0x06, 0x81, 0x19, 0xd3, 0x26, 0xe4, 0x9a, 0x7a, 0x4c], mode=mode_choice)
    
    addkey_s_time = time.time()
    enc_data_hi_round_2 = AddRoundKey(engine_context, enc_data_hi_round_1, enc_key_hi_list[1])
    enc_data_lo_round_2 = AddRoundKey(engine_context, enc_data_lo_round_1, enc_key_lo_list[1])
    enc_data_hi_round_2 = engine.intt(enc_data_hi_round_2)
    enc_data_lo_round_2 = engine.intt(enc_data_lo_round_2)
    addkey_e_time = time.time()
    print(f"addkey complete!!! Time taken: {addkey_e_time - addkey_s_time} seconds")    
    stop_time = time.time()
    print(f"round 1 complete!!! Time taken: {(stop_time - start_time)} seconds")
    
    verify = verify_round_output(engine_context, enc_data_hi_round_2, enc_data_lo_round_2, ground_truth= [0xa0, 0x88, 0x23, 0x2a, 0xfa, 0x54, 0xa3, 0x6c, 0xfe, 0x2c, 0x39, 0x76, 0x17, 0xb1, 0x39, 0x05], mode=mode_choice)
    
    # --- Round 2 --------------------------------------------------------------
    r2_time = time.time()
    enc_data_hi_round_2, enc_data_lo_round_2 = sub_bytes(engine_context, enc_data_hi_round_2, enc_data_lo_round_2)
    enc_data_hi_round_2 = engine.intt(enc_data_hi_round_2)
    enc_data_lo_round_2 = engine.intt(enc_data_lo_round_2)
    
    enc_data_hi_round_2, enc_data_lo_round_2 = shift_rows(engine_context, enc_data_hi_round_2, enc_data_lo_round_2)
    enc_data_hi_round_2 = engine.intt(enc_data_hi_round_2)
    enc_data_lo_round_2 = engine.intt(enc_data_lo_round_2)
    
    enc_data_hi_round_2, enc_data_lo_round_2 = mix_columns(engine_context, enc_data_hi_round_2, enc_data_lo_round_2)
    enc_data_hi_round_2 = engine.intt(enc_data_hi_round_2)
    enc_data_lo_round_2 = engine.intt(enc_data_lo_round_2)
    
    enc_data_hi_round_3 = AddRoundKey(engine_context, enc_data_hi_round_2, enc_key_hi_list[2])
    enc_data_lo_round_3 = AddRoundKey(engine_context, enc_data_lo_round_2, enc_key_lo_list[2])
    enc_data_hi_round_3 = engine.intt(enc_data_hi_round_3)
    enc_data_lo_round_3 = engine.intt(enc_data_lo_round_3)
    r2_e_time = time.time()
    print(f"round 2 complete!!! Time taken: {(r2_e_time - r2_time)} seconds")
    
    verify = verify_round_output(engine_context, enc_data_hi_round_3, enc_data_lo_round_3, ground_truth= [0xf2, 0x7a, 0x59, 0x73, 0xc2, 0x96, 0x35, 0x59, 0x95, 0xb9, 0x80, 0xf6, 0xf2, 0x43, 0x7a, 0x7f], mode=mode_choice)
    
    # --- Round 3 --------------------------------------------------------------
    enc_data_hi_round_3, enc_data_lo_round_3 = sub_bytes(engine_context, enc_data_hi_round_3, enc_data_lo_round_3)
    enc_data_hi_round_3 = engine.intt(enc_data_hi_round_3)
    enc_data_lo_round_3 = engine.intt(enc_data_lo_round_3)
    
    enc_data_hi_round_3, enc_data_lo_round_3 = shift_rows(engine_context, enc_data_hi_round_3, enc_data_lo_round_3)
    enc_data_hi_round_3 = engine.intt(enc_data_hi_round_3)
    enc_data_lo_round_3 = engine.intt(enc_data_lo_round_3)
    
    enc_data_hi_round_3, enc_data_lo_round_3 = mix_columns(engine_context, enc_data_hi_round_3, enc_data_lo_round_3)
    enc_data_hi_round_3 = engine.intt(enc_data_hi_round_3)
    enc_data_lo_round_3 = engine.intt(enc_data_lo_round_3)
    
    enc_data_hi_round_4 = AddRoundKey(engine_context, enc_data_hi_round_3, enc_key_hi_list[3])
    enc_data_lo_round_4 = AddRoundKey(engine_context, enc_data_lo_round_3, enc_key_lo_list[3])
    enc_data_hi_round_4 = engine.intt(enc_data_hi_round_4)
    enc_data_lo_round_4 = engine.intt(enc_data_lo_round_4)
    
    # --- Round 4 --------------------------------------------------------------
    enc_data_hi_round_4, enc_data_lo_round_4 = sub_bytes(engine_context, enc_data_hi_round_4, enc_data_lo_round_4)
    enc_data_hi_round_4 = engine.intt(enc_data_hi_round_4)
    enc_data_lo_round_4 = engine.intt(enc_data_lo_round_4)
    
    enc_data_hi_round_4, enc_data_lo_round_4 = shift_rows(engine_context, enc_data_hi_round_4, enc_data_lo_round_4)
    enc_data_hi_round_4 = engine.intt(enc_data_hi_round_4)
    enc_data_lo_round_4 = engine.intt(enc_data_lo_round_4)
    
    enc_data_hi_round_4, enc_data_lo_round_4 = mix_columns(engine_context, enc_data_hi_round_4, enc_data_lo_round_4)
    enc_data_hi_round_4 = engine.intt(enc_data_hi_round_4)
    enc_data_lo_round_4 = engine.intt(enc_data_lo_round_4)
    
    enc_data_hi_round_5 = AddRoundKey(engine_context, enc_data_hi_round_4, enc_key_hi_list[4])
    enc_data_lo_round_5 = AddRoundKey(engine_context, enc_data_lo_round_4, enc_key_lo_list[4])
    enc_data_hi_round_5 = engine.intt(enc_data_hi_round_5)
    enc_data_lo_round_5 = engine.intt(enc_data_lo_round_5)
    
    # --- Round 5 --------------------------------------------------------------
    enc_data_hi_round_5, enc_data_lo_round_5 = sub_bytes(engine_context, enc_data_hi_round_5, enc_data_lo_round_5)
    enc_data_hi_round_5 = engine.intt(enc_data_hi_round_5)
    enc_data_lo_round_5 = engine.intt(enc_data_lo_round_5)
    
    enc_data_hi_round_5, enc_data_lo_round_5 = shift_rows(engine_context, enc_data_hi_round_5, enc_data_lo_round_5)
    enc_data_hi_round_5 = engine.intt(enc_data_hi_round_5)
    enc_data_lo_round_5 = engine.intt(enc_data_lo_round_5)
    
    enc_data_hi_round_5, enc_data_lo_round_5 = mix_columns(engine_context, enc_data_hi_round_5, enc_data_lo_round_5)
    enc_data_hi_round_5 = engine.intt(enc_data_hi_round_5)
    enc_data_lo_round_5 = engine.intt(enc_data_lo_round_5)
    
    enc_data_hi_round_6 = AddRoundKey(engine_context, enc_data_hi_round_5, enc_key_hi_list[5])
    enc_data_lo_round_6 = AddRoundKey(engine_context, enc_data_lo_round_5, enc_key_lo_list[5])
    enc_data_hi_round_6 = engine.intt(enc_data_hi_round_6)
    enc_data_lo_round_6 = engine.intt(enc_data_lo_round_6)
    
    # --- Round 6 --------------------------------------------------------------
    enc_data_hi_round_6, enc_data_lo_round_6 = sub_bytes(engine_context, enc_data_hi_round_6, enc_data_lo_round_6)
    enc_data_hi_round_6 = engine.intt(enc_data_hi_round_6)
    enc_data_lo_round_6 = engine.intt(enc_data_lo_round_6)
    
    enc_data_hi_round_6, enc_data_lo_round_6 = shift_rows(engine_context, enc_data_hi_round_6, enc_data_lo_round_6)
    enc_data_hi_round_6 = engine.intt(enc_data_hi_round_6)
    enc_data_lo_round_6 = engine.intt(enc_data_lo_round_6)
    
    enc_data_hi_round_6, enc_data_lo_round_6 = mix_columns(engine_context, enc_data_hi_round_6, enc_data_lo_round_6)
    enc_data_hi_round_6 = engine.intt(enc_data_hi_round_6)
    enc_data_lo_round_6 = engine.intt(enc_data_lo_round_6)
    
    enc_data_hi_round_7 = AddRoundKey(engine_context, enc_data_hi_round_6, enc_key_hi_list[6])
    enc_data_lo_round_7 = AddRoundKey(engine_context, enc_data_lo_round_6, enc_key_lo_list[6])
    enc_data_hi_round_7 = engine.intt(enc_data_hi_round_7)
    enc_data_lo_round_7 = engine.intt(enc_data_lo_round_7)
    
    # --- Round 7 --------------------------------------------------------------
    enc_data_hi_round_7, enc_data_lo_round_7 = sub_bytes(engine_context, enc_data_hi_round_7, enc_data_lo_round_7)
    enc_data_hi_round_7 = engine.intt(enc_data_hi_round_7)
    enc_data_lo_round_7 = engine.intt(enc_data_lo_round_7)
    
    enc_data_hi_round_7, enc_data_lo_round_7 = shift_rows(engine_context, enc_data_hi_round_7, enc_data_lo_round_7)
    enc_data_hi_round_7 = engine.intt(enc_data_hi_round_7)
    enc_data_lo_round_7 = engine.intt(enc_data_lo_round_7)
    
    enc_data_hi_round_7, enc_data_lo_round_7 = mix_columns(engine_context, enc_data_hi_round_7, enc_data_lo_round_7)
    enc_data_hi_round_7 = engine.intt(enc_data_hi_round_7)
    enc_data_lo_round_7 = engine.intt(enc_data_lo_round_7)
    
    enc_data_hi_round_8 = AddRoundKey(engine_context, enc_data_hi_round_7, enc_key_hi_list[7])
    enc_data_lo_round_8 = AddRoundKey(engine_context, enc_data_lo_round_7, enc_key_lo_list[7])
    enc_data_hi_round_8 = engine.intt(enc_data_hi_round_8)
    enc_data_lo_round_8 = engine.intt(enc_data_lo_round_8)
    
    # --- Round 8 --------------------------------------------------------------
    enc_data_hi_round_8, enc_data_lo_round_8 = sub_bytes(engine_context, enc_data_hi_round_8, enc_data_lo_round_8)
    enc_data_hi_round_8 = engine.intt(enc_data_hi_round_8)
    enc_data_lo_round_8 = engine.intt(enc_data_lo_round_8)
    
    enc_data_hi_round_8, enc_data_lo_round_8 = shift_rows(engine_context, enc_data_hi_round_8, enc_data_lo_round_8)
    enc_data_hi_round_8 = engine.intt(enc_data_hi_round_8)
    enc_data_lo_round_8 = engine.intt(enc_data_lo_round_8)
    
    enc_data_hi_round_8, enc_data_lo_round_8 = mix_columns(engine_context, enc_data_hi_round_8, enc_data_lo_round_8)
    enc_data_hi_round_8 = engine.intt(enc_data_hi_round_8)
    enc_data_lo_round_8 = engine.intt(enc_data_lo_round_8)
    
    enc_data_hi_round_9 = AddRoundKey(engine_context, enc_data_hi_round_8, enc_key_hi_list[8])
    enc_data_lo_round_9 = AddRoundKey(engine_context, enc_data_lo_round_8, enc_key_lo_list[8])
    enc_data_hi_round_9 = engine.intt(enc_data_hi_round_9)
    enc_data_lo_round_9 = engine.intt(enc_data_lo_round_9)
    
    # --- Round 9 --------------------------------------------------------------
    enc_data_hi_round_9, enc_data_lo_round_9 = sub_bytes(engine_context, enc_data_hi_round_9, enc_data_lo_round_9)
    enc_data_hi_round_9 = engine.intt(enc_data_hi_round_9)
    enc_data_lo_round_9 = engine.intt(enc_data_lo_round_9)
    
    enc_data_hi_round_9, enc_data_lo_round_9 = shift_rows(engine_context, enc_data_hi_round_9, enc_data_lo_round_9)
    enc_data_hi_round_9 = engine.intt(enc_data_hi_round_9)
    enc_data_lo_round_9 = engine.intt(enc_data_lo_round_9)
    
    enc_data_hi_round_9, enc_data_lo_round_9 = mix_columns(engine_context, enc_data_hi_round_9, enc_data_lo_round_9)
    enc_data_hi_round_9 = engine.intt(enc_data_hi_round_9)
    enc_data_lo_round_9 = engine.intt(enc_data_lo_round_9)
    
    enc_data_hi_round_10 = AddRoundKey(engine_context, enc_data_hi_round_9, enc_key_hi_list[9])
    enc_data_lo_round_10 = AddRoundKey(engine_context, enc_data_lo_round_9, enc_key_lo_list[9])
    enc_data_hi_round_10 = engine.intt(enc_data_hi_round_10)
    enc_data_lo_round_10 = engine.intt(enc_data_lo_round_10)
    
    # --- Round 10 --------------------------------------------------------------
    enc_data_hi_round_10, enc_data_lo_round_10 = sub_bytes(engine_context, enc_data_hi_round_10, enc_data_lo_round_10)
    enc_data_hi_round_10 = engine.intt(enc_data_hi_round_10)
    enc_data_lo_round_10 = engine.intt(enc_data_lo_round_10)
    
    enc_data_hi_round_10, enc_data_lo_round_10 = shift_rows(engine_context, enc_data_hi_round_10, enc_data_lo_round_10)
    enc_data_hi_round_10 = engine.intt(enc_data_hi_round_10)
    enc_data_lo_round_10 = engine.intt(enc_data_lo_round_10)
    
    enc_data_hi = AddRoundKey(engine_context, enc_data_hi_round_10, enc_key_hi_list[10])
    enc_data_lo = AddRoundKey(engine_context, enc_data_lo_round_10, enc_key_lo_list[10])
    enc_data_hi = engine.intt(enc_data_hi)
    enc_data_lo = engine.intt(enc_data_lo)
    
    wait_next_stage("encryption stage", "decryption stage")

    # --- Decryption stage ----------------------------------------------------
    
    # 암호화된 데이터 사용
    enc_data_hi = enc_data_hi
    enc_data_lo = enc_data_lo
    
    # 기존의 키 리스트 사용(역순)
    key_hi_list = [key_hi_list[::-1]]
    key_lo_list = [key_lo_list[::-1]]
        
    # --- Round 0 --------------------------------------------------------------
    dec_data_hi_round_0 = AddRoundKey(enc_data_hi, key_hi_list[0])
    dec_data_lo_round_0 = AddRoundKey(enc_data_lo, key_lo_list[0])
        
    dec_data_hi_round_0, dec_data_lo_round_0 = inv_shift_rows(engine_context, dec_data_hi_round_0, dec_data_lo_round_0)
    
    dec_data_hi_round_1, dec_data_lo_round_1 = inv_sub_bytes(engine_context, dec_data_hi_round_0, dec_data_lo_round_0)
    
    # --- Round 1 --------------------------------------------------------------
    dec_data_hi_round_1 = AddRoundKey(dec_data_hi_round_1, key_hi_list[1])
    dec_data_lo_round_1 = AddRoundKey(dec_data_lo_round_1, key_lo_list[1])
    
    dec_data_hi_round_1, dec_data_lo_round_1 = inv_mix_columns(engine_context, dec_data_hi_round_1, dec_data_lo_round_1)
    
    dec_data_hi_round_1, dec_data_lo_round_1 = inv_shift_rows(engine_context, dec_data_hi_round_1, dec_data_lo_round_1)
    
    dec_data_hi_round_2, dec_data_lo_round_2 = inv_sub_bytes(engine_context, dec_data_hi_round_1, dec_data_lo_round_1)
    
    # --- Round 2 --------------------------------------------------------------
    dec_data_hi_round_2 = AddRoundKey(dec_data_hi_round_2, key_hi_list[2])
    dec_data_lo_round_2 = AddRoundKey(dec_data_lo_round_2, key_lo_list[2])
    
    dec_data_hi_round_2, dec_data_lo_round_2 = inv_mix_columns(engine_context, dec_data_hi_round_2, dec_data_lo_round_2)
    
    dec_data_hi_round_2, dec_data_lo_round_2 = inv_shift_rows(engine_context, dec_data_hi_round_2, dec_data_lo_round_2)
    
    dec_data_hi_round_3, dec_data_lo_round_3 = inv_sub_bytes(engine_context, dec_data_hi_round_2, dec_data_lo_round_2)
    
    # --- Round 3 --------------------------------------------------------------
    dec_data_hi_round_3 = AddRoundKey(dec_data_hi_round_3, key_hi_list[3])
    dec_data_lo_round_3 = AddRoundKey(dec_data_lo_round_3, key_lo_list[3])
    
    dec_data_hi_round_3, dec_data_lo_round_3 = inv_mix_columns(engine_context, dec_data_hi_round_3, dec_data_lo_round_3)
    
    dec_data_hi_round_3, dec_data_lo_round_3 = inv_shift_rows(engine_context, dec_data_hi_round_3, dec_data_lo_round_3)
    
    dec_data_hi_round_4, dec_data_lo_round_4 = inv_sub_bytes(engine_context, dec_data_hi_round_3, dec_data_lo_round_3)
    
    # --- Round 4 --------------------------------------------------------------
    dec_data_hi_round_4 = AddRoundKey(dec_data_hi_round_4, key_hi_list[4])
    dec_data_lo_round_4 = AddRoundKey(dec_data_lo_round_4, key_lo_list[4])
    
    dec_data_hi_round_4, dec_data_lo_round_4 = inv_mix_columns(engine_context, dec_data_hi_round_4, dec_data_lo_round_4)
    
    dec_data_hi_round_4, dec_data_lo_round_4 = inv_shift_rows(engine_context, dec_data_hi_round_4, dec_data_lo_round_4)
    
    dec_data_hi_round_5, dec_data_lo_round_5 = inv_sub_bytes(engine_context, dec_data_hi_round_4, dec_data_lo_round_4)
    
    # --- Round 5 --------------------------------------------------------------
    dec_data_hi_round_5 = AddRoundKey(dec_data_hi_round_5, key_hi_list[5])
    dec_data_lo_round_5 = AddRoundKey(dec_data_lo_round_5, key_lo_list[5])
    
    dec_data_hi_round_5, dec_data_lo_round_5 = inv_mix_columns(engine_context, dec_data_hi_round_5, dec_data_lo_round_5)
    
    dec_data_hi_round_5, dec_data_lo_round_5 = inv_shift_rows(engine_context, dec_data_hi_round_5, dec_data_lo_round_5)
    
    dec_data_hi_round_6, dec_data_lo_round_6 = inv_sub_bytes(engine_context, dec_data_hi_round_5, dec_data_lo_round_5)
    
    # --- Round 6 --------------------------------------------------------------
    dec_data_hi_round_6 = AddRoundKey(dec_data_hi_round_6, key_hi_list[6])
    dec_data_lo_round_6 = AddRoundKey(dec_data_lo_round_6, key_lo_list[6])
    
    dec_data_hi_round_6, dec_data_lo_round_6 = inv_mix_columns(engine_context, dec_data_hi_round_6, dec_data_lo_round_6)
    
    dec_data_hi_round_6, dec_data_lo_round_6 = inv_shift_rows(engine_context, dec_data_hi_round_6, dec_data_lo_round_6)
    
    dec_data_hi_round_7, dec_data_lo_round_7 = inv_sub_bytes(engine_context, dec_data_hi_round_6, dec_data_lo_round_6)
    
    # --- Round 7 --------------------------------------------------------------
    dec_data_hi_round_7 = AddRoundKey(dec_data_hi_round_7, key_hi_list[7])
    dec_data_lo_round_7 = AddRoundKey(dec_data_lo_round_7, key_lo_list[7])
    
    dec_data_hi_round_7, dec_data_lo_round_7 = inv_mix_columns(engine_context, dec_data_hi_round_7, dec_data_lo_round_7)
    
    dec_data_hi_round_7, dec_data_lo_round_7 = inv_shift_rows(engine_context, dec_data_hi_round_7, dec_data_lo_round_7)
    
    dec_data_hi_round_8, dec_data_lo_round_8 = inv_sub_bytes(engine_context, dec_data_hi_round_7, dec_data_lo_round_7)
    
    # --- Round 8 --------------------------------------------------------------
    dec_data_hi_round_8 = AddRoundKey(dec_data_hi_round_8, key_hi_list[8])
    dec_data_lo_round_8 = AddRoundKey(dec_data_lo_round_8, key_lo_list[8])
    
    dec_data_hi_round_8, dec_data_lo_round_8 = inv_mix_columns(engine_context, dec_data_hi_round_8, dec_data_lo_round_8)
    
    dec_data_hi_round_8, dec_data_lo_round_8 = inv_shift_rows(engine_context, dec_data_hi_round_8, dec_data_lo_round_8)
    
    dec_data_hi_round_9, dec_data_lo_round_9 = inv_sub_bytes(engine_context, dec_data_hi_round_8, dec_data_lo_round_8)
    
    # --- Round 9 --------------------------------------------------------------
    dec_data_hi_round_9 = AddRoundKey(dec_data_hi_round_9, key_hi_list[9])
    dec_data_lo_round_9 = AddRoundKey(dec_data_lo_round_9, key_lo_list[9])
    
    dec_data_hi_round_9, dec_data_lo_round_9 = inv_mix_columns(engine_context, dec_data_hi_round_9, dec_data_lo_round_9)
    
    dec_data_hi_round_9, dec_data_lo_round_9 = inv_shift_rows(engine_context, dec_data_hi_round_9, dec_data_lo_round_9)
    
    dec_data_hi_round_10, dec_data_lo_round_10 = inv_sub_bytes(engine_context, dec_data_hi_round_9, dec_data_lo_round_9)
    
    # --- Round 10 --------------------------------------------------------------
    dec_data_hi_round_10 = AddRoundKey(dec_data_hi_round_10, key_hi_list[10])
    dec_data_lo_round_10 = AddRoundKey(dec_data_lo_round_10, key_lo_list[10])
    
    dec_data_hi = engine.decrypt(dec_data_hi_round_10, engine_context.get_secret_key())
    dec_data_lo = engine.decrypt(dec_data_lo_round_10, engine_context.get_secret_key())
    
    dec_data_hi_int = zeta_to_int(dec_data_hi)
    dec_data_lo_int = zeta_to_int(dec_data_lo)
    
    print(dec_data_hi_int)
    print(dec_data_lo_int)

    
    

    