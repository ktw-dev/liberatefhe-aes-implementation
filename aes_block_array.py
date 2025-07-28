import numpy as np


def blocks_to_flat_array(blocks, max_blocks: int = 2048) -> np.ndarray:
    """Return a 1-D NumPy array that stores up to *max_blocks* AES blocks (16-byte each)
    in the layout required by the FHE batching routine.

    Layout (length = 16 * max_blocks):
        [ b0_block0, b0_block1, ..., b0_blockN, 0-pad, ...,   # 0-th byte for every block
          b1_block0, b1_block1, ..., b1_blockN, 0-pad, ...,   # 1-st byte for every block
          ...                                                 # ...
          b15_block0, b15_block1, ..., b15_blockN, 0-pad ...] # 15-th byte for every block

    If the number of provided *blocks* is < *max_blocks*, the remaining entries are
    zero-filled.  If it is greater, a ValueError is raised.

    Parameters
    ----------
    blocks : (N, 16) array-like of uint8
        AES state blocks to pack.  N must be <= *max_blocks*.
    max_blocks : int, optional
        Maximum number of blocks that can be packed (default 2048).

    Returns
    -------
    flat : np.ndarray, shape (16 * max_blocks,), dtype=np.uint8
        Packed 1-D array in row-major order.
    """
    blocks = np.asarray(blocks, dtype=np.uint8)

    # Accept a single block passed as a flat 1-D array of length 16
    if blocks.ndim == 1:
        if blocks.size != 16:
            raise ValueError("Single block must have exactly 16 bytes")
        blocks = blocks[np.newaxis, :]

    if blocks.ndim != 2 or blocks.shape[1] != 16:
        raise ValueError("Input must have shape (N, 16) where each row is a 16-byte AES block")

    n_blocks = blocks.shape[0]
    if n_blocks > max_blocks:
        raise ValueError(f"Too many blocks ({n_blocks} > {max_blocks})")

    # Allocate 16 × max_blocks matrix, zero-initialised
    # matrix = np.zeros((16, max_blocks), dtype=np.uint8)
    # # Place the actual data – transpose to align bytes per position
    # matrix[:, :n_blocks] = blocks.T
    matrix = blocks.T

    # Flatten row-wise so each byte position segment is contiguous
    return matrix.reshape(-1)


if __name__ == "__main__":
    # Interactive usage: user specifies how many blocks (≤ 2048)
    try:
        count_str = input("Enter number of AES blocks (0–2048): ")
        count = int(count_str)
    except ValueError:
        raise SystemExit("❌  Invalid integer input.")

    if not (0 <= count <= 2048):
        raise SystemExit("❌  Block count must be between 0 and 2048.")

    # Generate random blocks: each byte uniform in 0x00–0xFF
    
    rng = np.random.default_rng()
    blocks = rng.integers(0, 256, size=(count, 16), dtype=np.uint8)

    print(blocks[:16])
    flat = blocks_to_flat_array(blocks)

    print("Generated flat array (length =", flat.size, ")")
    print("First 16 bytes:", flat[:16]) 
    print("Second 16 bytes:", flat[2048:2064]) 
    
    