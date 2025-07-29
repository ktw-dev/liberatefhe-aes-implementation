"""aes-main-process.py

High-level orchestration script for (future) fully homomorphic AES evaluation.
Currently implements only the *data initiation* stage:
    1. Build random AES state blocks (user-selected count ≤ 2048)
    2. Flatten to 1-D array following batching layout (via aes-block-array.py)
    3. Split each byte into upper / lower 4-bit nibbles (via aes-split-to-nibble.py)

Sub-modules have dashes in their filenames, so they are loaded dynamically with
importlib to avoid name conflicts.

Future work (place-holders):
    • key_schedule(...)
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
from typing import Tuple

# import custom modules
from engine_context import CKKS_EngineContext
from aes_block_array import blocks_to_flat_array
from aes_split_to_nibble import split_to_nibbles
from aes_key_array import key_to_flat_array
from aes_transform_zeta import int_to_zeta, zeta_to_int

# -----------------------------------------------------------------------------
# Dynamic import helpers -------------------------------------------------------
# -----------------------------------------------------------------------------

_THIS_DIR = pathlib.Path(__file__).resolve().parent


def _load_module(fname: str, alias: str):
    """Load a Python file in the current directory as a module with *alias*."""
    path = _THIS_DIR / fname
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Cannot load {fname}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

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

    # 2. Flatten to 1-D array following batching layout
    flat = blocks_to_flat_array(blocks)

    # 3. Split each byte into upper / lower 4-bit nibbles
    upper, lower = split_to_nibbles(flat)

    # 4. ζ-변환 (SIMD-style) – repeatable vectorized op
    zeta_upper = int_to_zeta(upper)
    zeta_lower = int_to_zeta(lower)

    return blocks, flat, upper, lower, zeta_upper, zeta_lower

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
# Key Scheduling --------------------------------------------------------------
# -----------------------------------------------------------------------------

def key_scheduling(engine_context, enc_key_hi, enc_key_lo):
    pass





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
    
    # --- Engine initiation stage -----------------------------------------------
    delta_list = [1*2048, 2*2048, 3*2048, 4*2048, 8*2048, -1*2048, -2*2048, -3*2048, -4*2048]
    
    engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0, fixed_rotation=True, delta_list=delta_list)
    # fixed rotation, delta_list는 필수로 사용함
    
    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()
    secret_key = engine_context.get_secret_key()   

    wait_next_stage("Engine Initiation", "Data initiation")
    
    # --- Data initiation stage ------------------------------------------------
    blocks, flat, _, _, data_zeta_hi, data_zeta_lo = data_initiation(n_blocks)

    # DEBUG
    # print("Generated", len(blocks), "block(s)")
    # print("First block bytes (hex):", [f"{b:02X}" for b in blocks[0]] if blocks.size else [])
    # print("Flat array sample (0-15):", [f"{b:02X}" for b in flat[:16]])
    # print("ζ(upper)[0-3]          :", [f"{c:.2f}" for c in data_zeta_hi[:4]])
    # print("ζ(lower)[0-3]          :", [f"{c:.2f}" for c in data_zeta_lo[:4]])

    wait_next_stage("Data initiation", "key initiation")

    # --- Key initiation stage -------------------------------------------------
    key_bytes, key_flat, _, _, key_zeta_hi, key_zeta_lo = key_initiation()

    # DEBUG
    # print("Secret key bytes (hex):", [f"{b:02X}" for b in key_bytes])

    # print("ζ(key upper)[0-3]       :", [f"{c:.2f}" for c in key_zeta_hi[:4]])
    # print("ζ(key lower)[0-3]       :", [f"{c:.2f}" for c in key_zeta_lo[:4]])

    wait_next_stage("Key initiation", "data/key HE-encryption")
    
    # --- data HE-encryption stage ------------------------------------------------
    
    # 1. 데이터 암호화
    enc_data_hi = engine.encrypt(data_zeta_hi, public_key)
    enc_data_lo = engine.encrypt(data_zeta_lo, public_key)
    
    # DEBUG
    # print(enc_data_hi)
    # print(enc_data_lo)

    
    # --- key HE-encryption stage ------------------------------------------------
    
    # 1. 키 암호화
    enc_key_hi = engine.encrypt(key_zeta_hi, public_key)
    enc_key_lo = engine.encrypt(key_zeta_lo, public_key)
    
    # DEBUG
    # print(enc_key_hi)
    # print(enc_key_lo)
    
    wait_next_stage("data/key HE-encryption", "key Scheduling")
    
    # --- key Scheduling stage -------------------------------------------------



    # --- AddRoundKey stage ----------------------------------------------------
    
    
    
    
    # Stage 1: SubBytes