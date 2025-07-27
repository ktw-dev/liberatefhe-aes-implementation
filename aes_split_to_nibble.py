import numpy as np

__all__ = [
    "split_to_nibbles",
]


def split_to_nibbles(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1-D uint8 array of AES bytes into upper & lower 4-bit nibbles.

    Parameters
    ----------
    flat : np.ndarray, shape (N,), dtype uint8
        Input byte array (e.g. length 32 768 from *aes-block-array.py*).

    Returns
    -------
    upper : np.ndarray, shape (N,), dtype uint8
        Upper four bits of every byte (values 0-15).
    lower : np.ndarray, shape (N,), dtype uint8
        Lower four bits of every byte (values 0-15).
    """
    if flat.dtype != np.uint8:
        flat = flat.astype(np.uint8, copy=False)

    # Vectorised bit operations – backed by SIMD in NumPy’s ufunc engine.
    upper = np.right_shift(flat, 4, dtype=np.uint8)   # flat >> 4
    lower = np.bitwise_and(flat, 0x0F, dtype=np.uint8)  # flat & 0x0F

    return upper, lower


if __name__ == "__main__":
    # --- Demo: reuse block construction from aes-block-array.py ---
    import importlib.util
    import pathlib

    blocks_file = pathlib.Path(__file__).with_name("aes_block_array.py")

    spec = importlib.util.spec_from_file_location("aes_block_array", blocks_file)
    aes_block_array = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aes_block_array)  # type: ignore

    # Build random blocks – user chooses how many
    try:
        count_str = input("Enter number of AES blocks (0–2048): ")
        count = int(count_str)
    except ValueError:
        raise SystemExit("❌  Invalid integer input.")

    if not (0 <= count <= 2048):
        raise SystemExit("❌  Block count must be between 0 and 2048.")

    rng = np.random.default_rng()
    blocks = rng.integers(0, 256, size=(count, 16), dtype=np.uint8)
    demo_flat = aes_block_array.blocks_to_flat_array(blocks)

    hi, lo = split_to_nibbles(demo_flat)

    # Pretty-print first few elements in hexadecimal
    print("Original byte       :", [f"{b:02X}" for b in demo_flat[:16]])
    print("Upper nibbles (hi)  :", [f"{b:X}" for b in hi[:16]])
    print("Lower nibbles (lo)  :", [f"{b:X}" for b in lo[:16]])

    # Sanity check: recombine and verify equality
    recombined = (hi << 4) | lo
    assert np.array_equal(recombined, demo_flat), "Nibble split/recombine failed!"
    print("✔ Sanity check passed.")
