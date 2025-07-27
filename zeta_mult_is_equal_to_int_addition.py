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
    mode="parallel",  # 'parallel' may not be recognised; use CPU with threads
    use_bootstrap=True,
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

# 1. Homomorphic multiplication (slot-wise)
# Homomorphic multiplication (slot-wise)
ct_c = engine.multiply(ct_a, ct_b)

# Decrypt back
zeta_c = np.array(engine.decrypt(ct_c, ctx.get_secret_key()))

# 2. Homomorphic addition (slot-wise)
ct_c = engine.add(ct_a, ct_b)

# Decrypt back
zeta_added_c = np.array(engine.decrypt(ct_c, ctx.get_secret_key()))

# 3. Homomorphic Constant Multiplication (slot-wise)
print(f"ct_a.level: {ct_a.level}")
ct_c_one = engine.multiply(ct_a, ct_a, ctx.get_relinearization_key())
print(f"ct_c_one.level: {ct_c_one.level}")
ct_c_two = engine.multiply(ct_c_one, ct_a, ctx.get_relinearization_key())
print(f"ct_c_two.level: {ct_c_two.level}")
ct_c_three = engine.multiply(ct_c_two, ct_a, ctx.get_relinearization_key())
print(f"ct_c_three.level: {ct_c_three.level}")
ct_c_four = engine.multiply(ct_c_three, ct_a, ctx.get_relinearization_key())
print(f"ct_c_four.level: {ct_c_four.level}")

# Decrypt back
zeta_c_one = np.array(engine.decrypt(ct_c_one, ctx.get_secret_key()))
zeta_c_two = np.array(engine.decrypt(ct_c_two, ctx.get_secret_key()))
zeta_c_three = np.array(engine.decrypt(ct_c_three, ctx.get_secret_key()))
zeta_c_four = np.array(engine.decrypt(ct_c_four, ctx.get_secret_key()))

zeta_c_one_dec = np.array(engine.decrypt(ct_c_one, ctx.get_secret_key()))
zeta_c_two_dec = np.array(engine.decrypt(ct_c_two, ctx.get_secret_key()))
zeta_c_three_dec = np.array(engine.decrypt(ct_c_three, ctx.get_secret_key()))
zeta_c_four_dec = np.array(engine.decrypt(ct_c_four, ctx.get_secret_key()))

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
    for idx, a, b, gf, z in mismatches[:16]:
        print(f"  slot {idx}: {a}×{b} → GF={gf}, ζ-mul={z}")
else:
    print("All slots match (unexpected).")
    
print(zeta_c[:16])
zeta_2_int = []
for z in zeta_c[:16]:
    zeta_2_int.append(angle_to_int(z))
    
print(ints_a)
print(ints_b)
print(zeta_2_int)

print("결론: zeta 위상에서의 곱은 정수에서의 16 모듈러 합과 동일하다.")

# 정수의 상수곱을 zeta 위상에서의 곱을 여러번 함으로써 구현하기

print(zeta_c_one_dec[:16])
print(zeta_c_two_dec[:16])
print(zeta_c_three_dec[:16])
print(zeta_c_four_dec[:16])

zeta_c_one_dec_2_int = []
for z in zeta_c_one_dec[:16]:
    zeta_c_one_dec_2_int.append(angle_to_int(z))

zeta_c_two_dec_2_int = []
for z in zeta_c_two_dec[:16]:
    zeta_c_two_dec_2_int.append(angle_to_int(z))

zeta_c_three_dec_2_int = []
for z in zeta_c_three_dec[:16]:
    zeta_c_three_dec_2_int.append(angle_to_int(z))

zeta_c_four_dec_2_int = []
for z in zeta_c_four_dec[:16]:
    zeta_c_four_dec_2_int.append(angle_to_int(z))

print(f"zeta_c_one_dec_2_int: {zeta_c_one_dec_2_int}")
print(f"zeta_c_two_dec_2_int: {zeta_c_two_dec_2_int}")
print(f"zeta_c_three_dec_2_int: {zeta_c_three_dec_2_int}")
print(f"zeta_c_four_dec_2_int: {zeta_c_four_dec_2_int}")

print(f"ints_a * 2: {ints_a * 2 % 16}")
print(f"ints_a * 3: {ints_a * 3 % 16}")
print(f"ints_a * 4: {ints_a * 4 % 16}")

print(f"checking zeta_c_one_dec_2_int: {zeta_c_one_dec_2_int == ints_a * 2 % 16}")
print(f"checking zeta_c_two_dec_2_int: {zeta_c_two_dec_2_int == ints_a * 3 % 16}")
print(f"checking zeta_c_three_dec_2_int: {zeta_c_three_dec_2_int == ints_a * 4 % 16}")
print(f"checking zeta_c_four_dec_2_int: {zeta_c_four_dec_2_int == ints_a * 5 % 16}")

# checking zeta_added_c
print(zeta_added_c[:16])
zeta_added_2_int = []
for z in zeta_added_c[:16]:
    zeta_added_2_int.append(angle_to_int(z))

print(zeta_added_2_int)
print(ints_a + ints_b)

print("결론: zeta 위상에서의 합은 정수 연산과 그 어떤 관계도 없다.")