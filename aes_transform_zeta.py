import numpy as np

__all__ = [
    "transform_to_zeta",
]                         # shape (32768,), dtype=complex128

def transform_to_zeta(arr: np.ndarray) -> np.ndarray:
    result = np.exp(-2j * np.pi * (arr % 16) / 16)
    return result
    
def __main__():
    arr = np.random.randint(0, 16, size=16*2048, dtype=np.uint8)
    result = transform_to_zeta(arr)
    print(result)
    print(result.shape)
    print(result.dtype)

if __name__ == "__main__":
    __main__()