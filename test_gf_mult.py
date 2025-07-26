import numpy as np
from engine_context import CKKS_EngineContext

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def transform_to_zeta(arr: np.ndarray) -> np.ndarray:
    """Convert integers (mod 16) to 16-th roots of unity ζ^k."""
    return np.exp(-2j * np.pi * (arr % 16) / 16)


def angle_to_int(z: complex) -> int:
    """Map a complex 16-th root of unity back to integer 0‥15."""
    angle = np.angle(z)
    return int(round(-angle * 16 / (2 * np.pi))) % 16


def gf_mult_4bit(a: int, b: int) -> int:
    """GF(2^4) multiplication with irreducible poly x^4 + x + 1 (0x13)."""
    result = 0
    for _ in range(4):
        if b & 1:
            result ^= a
        high_bit = a & 0x8  # MSB
        a = (a << 1) & 0xF  # shift & keep in 4 bits
        if high_bit:
            a ^= 0x3  # (0x13 mod 0x10) == 0x3
        b >>= 1
    return result & 0xF


# -----------------------------------------------------------------------------
# FHE setup (signature 1, parallel mode, 16 threads, device_id 0)
# -----------------------------------------------------------------------------

ctx = CKKS_EngineContext(
    1,  # signature 1
    mode="cpu",  # 'parallel' may not be recognised; use CPU with threads
    thread_count=16,
    device_id=0,
)

engine = ctx.get_engine()

# -----------------------------------------------------------------------------
# Prepare plaintext inputs
# -----------------------------------------------------------------------------

rng = np.random.default_rng(42)
ints_a = rng.integers(0, 16, size=16, dtype=int)
ints_b = rng.integers(0, 16, size=16, dtype=int)

zeta_a = transform_to_zeta(ints_a)
zeta_b = transform_to_zeta(ints_b)

# Encrypt arrays (slot-wise CKKS)
ct_a = engine.encrypt(zeta_a.tolist(), ctx.get_public_key())
ct_b = engine.encrypt(zeta_b.tolist(), ctx.get_public_key())

# Homomorphic multiplication (slot-wise)
ct_c = engine.multiply(ct_a, ct_b)

# Decrypt back
zeta_c = np.array(engine.decrypt(ct_c, ctx.get_secret_key()))

# -----------------------------------------------------------------------------
# Compare against GF(2^4) multiplication
# -----------------------------------------------------------------------------

mismatches = []
for idx, (a, b, z) in enumerate(zip(ints_a, ints_b, zeta_c)):
    gf = gf_mult_4bit(int(a), int(b))
    zeta_int = angle_to_int(z)
    if gf != zeta_int:
        mismatches.append((idx, a, b, gf, zeta_int))

print("=== Homomorphic ζ-multiplication vs GF(2^4) ===")
print(f"Total slots: 16; mismatches: {len(mismatches)}")

if mismatches:
    print("First mismatches (up to 10):")
    for idx, a, b, gf, z in mismatches[:10]:
        print(f"  slot {idx}: {a}×{b} → GF={gf}, ζ-mul={z}")
else:
    print("All slots match (unexpected).")
