import numpy as np
from aes_gf_mult import gf_mult_lookup
    
    
    
# -----------------------------------------------------------------------------
# Forward MixColumns – accepts flat or (N,4,4)
# -----------------------------------------------------------------------------


def _reshape_to_state(a: np.ndarray):
    """Return (states, orig_shape, is_flat)."""
    if a.ndim == 1:
        if a.size % 16 != 0:
            raise ValueError("flat input length must be multiple of 16")
        n = a.size // 16
        states = a.reshape(16, n).T.reshape(-1, 4, 4)
        return states, a.shape, True
    if a.shape[-2:] != (4, 4):
        raise ValueError("input last two dims must be (4,4)")
    return a.copy(), a.shape, False


def _restore_shape(states: np.ndarray, orig_shape, was_flat: bool):
    if was_flat:
        n = states.shape[0]
        return states.reshape(n, 16).T.reshape(orig_shape)
    return states.reshape(orig_shape)


def mix_columns_numpy(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.uint8)
        st, orig_shape, was_flat = _reshape_to_state(a)
        def xtime(x):
            return (((x << 1) & 0xFF) ^ (((x >> 7) & 1) * 0x1B)).astype(np.uint8)
        s0, s1, s2, s3 = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
        tmp = s0 ^ s1 ^ s2 ^ s3   
        t0 = xtime(s0 ^ s1) ^ tmp ^ s0
        t1 = xtime(s1 ^ s2) ^ tmp ^ s1
        t2 = xtime(s2 ^ s3) ^ tmp ^ s2
        t3 = xtime(s3 ^ s0) ^ tmp ^ s3

        mixed = np.stack([t0, t1, t2, t3], axis=1).astype(np.uint8)
        return _restore_shape(mixed, orig_shape, was_flat)

def inv_mix_columns_numpy(arr: np.ndarray) -> np.ndarray:
    """Vectorised inverse MixColumns (NumPy).

    Parameters
    ----------
    arr : np.ndarray
        Array with shape (N, 4, 4) and dtype uint8. Any leading dimension
        *N* represents the number of AES states.
    Returns
    -------
    np.ndarray
        Array of same shape/type with inverse MixColumns applied.
    """

    a = np.asarray(arr, dtype=np.uint8)
    st, orig_shape, was_flat = _reshape_to_state(a)
    from aes_gf_mult import gf_mult_lookup
    
    # For each column compute outputs into temporaries to avoid in-place aliasing
    for col in range(4):
        s0 = st[:, 0, col].copy()
        s1 = st[:, 1, col].copy()
        s2 = st[:, 2, col].copy()
        s3 = st[:, 3, col].copy()

        out0 = (
            gf_mult_lookup(s0, 14) ^ gf_mult_lookup(s1, 11) ^ gf_mult_lookup(s2, 13) ^ gf_mult_lookup(s3, 9)
        )
        out1 = (
            gf_mult_lookup(s0, 9) ^ gf_mult_lookup(s1, 14) ^ gf_mult_lookup(s2, 11) ^ gf_mult_lookup(s3, 13)
        )
        out2 = (
            gf_mult_lookup(s0, 13) ^ gf_mult_lookup(s1, 9) ^ gf_mult_lookup(s2, 14) ^ gf_mult_lookup(s3, 11)
        )
        out3 = (
            gf_mult_lookup(s0, 11) ^ gf_mult_lookup(s1, 13) ^ gf_mult_lookup(s2, 9) ^ gf_mult_lookup(s3, 14)
        )

        st[:, 0, col] = out0
        st[:, 1, col] = out1
        st[:, 2, col] = out2
        st[:, 3, col] = out3

    return _restore_shape(st, orig_shape, was_flat)


if __name__ == "__main__":
    int_array = np.random.randint(0, 255, size=32768, dtype=np.uint8)
    
    
    # mixcolumns -> inv_mixcolumns = identity
    mixed = mix_columns_numpy(int_array)
    inv_mixed = inv_mix_columns_numpy(mixed)
    
    print(np.all(int_array == inv_mixed))
    
    
    
    
    
    