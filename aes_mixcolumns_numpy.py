# aes_mixcolumns_numpy.py
"""Pure NumPy reference implementation of AES MixColumns.

Useful for verifying homomorphic MixColumns (aes_MixColumns.py).
Accepts any leading batch dimensions; the trailing dimensions must form
`(..., 4, 4)` where the last axis is the column index and the second-to-last
axis is the row (standard AES state layout).
"""
from __future__ import annotations

import numpy as np

from aes_gf_mult import gf_mult_lookup

__all__ = ["mix_columns_numpy"]

# -----------------------------------------------------------------------------
# Convenience wrapper for project-specific 1-D layout (length N, N % 16 == 0)
# -----------------------------------------------------------------------------


def mix_columns_numpy_flat(arr1d: np.ndarray) -> np.ndarray:
    """Apply MixColumns to a 1-D uint8 array laid out in the project’s order.

    The current codebase stores *2048* AES states (32768 bytes) such that
    bytes belonging to the same position across all states are contiguous.
    That is,

    >>> shaped = arr.reshape(16, -1).T   # (num_states, 16)
    >>> shaped.reshape(-1, 4, 4)         # (num_states, 4, 4)

    This helper performs the same rearrangement, runs MixColumns on every
    state, and restores the original 1-D layout so it can be compared directly
    with ciphertext results.
    """
    a = np.asarray(arr1d, dtype=np.uint8)
    if a.ndim != 1 or a.size % 16 != 0:
        raise ValueError("Input must be 1-D with length multiple of 16 bytes.")

    num_states = a.size // 16
    # Reorder to (num_states, 4, 4) as used elsewhere in tests
    states = a.reshape(16, num_states).T.reshape(-1, 4, 4)

    mixed_states = mix_columns_numpy(states)

    # Restore original 1-D ordering
    mixed_flat = mixed_states.reshape(num_states, 16).T.reshape(-1)
    return mixed_flat.astype(np.uint8)

# expose in __all__
__all__.append("mix_columns_numpy_flat")


def mix_columns_numpy(state: np.ndarray) -> np.ndarray:
    """Apply AES MixColumns to *state*.

    Parameters
    ----------
    state : np.ndarray
        uint8 array whose final two axes are (4, 4). Any leading dimensions are
        treated as independent AES states.

    Returns
    -------
    np.ndarray
        New array of same shape with MixColumns applied.
    """
    s = np.asarray(state, dtype=np.uint8)
    if s.shape[-2:] != (4, 4):
        raise ValueError("Input's last two dimensions must be (4, 4)")

    # Work on a copy to keep input untouched
    st = s.copy().reshape(-1, 4, 4)  # (N, row, col)

    # Process each column (4 columns) vectorised over batch N
    for col in range(4):
        a0 = st[:, 0, col]
        a1 = st[:, 1, col]
        a2 = st[:, 2, col]
        a3 = st[:, 3, col]

        # MixColumns formula
        st[:, 0, col] = (
            gf_mult_lookup(a0, 2)
            ^ gf_mult_lookup(a1, 3)
            ^ a2
            ^ a3
        )
        st[:, 1, col] = (
            a0 ^ gf_mult_lookup(a1, 2) ^ gf_mult_lookup(a2, 3) ^ a3
        )
        st[:, 2, col] = (
            a0 ^ a1 ^ gf_mult_lookup(a2, 2) ^ gf_mult_lookup(a3, 3)
        )
        st[:, 3, col] = (
            gf_mult_lookup(a0, 3) ^ a1 ^ a2 ^ gf_mult_lookup(a3, 2)
        )

    return st.reshape(s.shape)


if __name__ == "__main__":
    np.random.seed(42)
    # Demo with a single AES state (4×4) and with the project-specific 1-D layout
    block = np.random.randint(0, 256, size=(4, 4), dtype=np.uint8)
    flat = block.reshape(16)
    print(block[:16])

    print("input block:\n", block)
    print("after MixColumns:\n", mix_columns_numpy(block))

    print("flat demo:")
    print("after MixColumns flat:", mix_columns_numpy_flat(flat)) 