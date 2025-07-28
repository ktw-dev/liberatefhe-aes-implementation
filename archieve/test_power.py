"""test_power.py – compare two strategies for building ct^p power basis.

1. Full power basis: engine.make_power_basis(ct, 15)
2. Optimised basis  : build_power_basis(ct) from aes_xor (make 1..8 then conjugate)

Outputs wall-clock time for each strategy.
"""
from __future__ import annotations

import time
import numpy as np

from engine_context import CKKS_EngineContext
from aes_xor import build_power_basis


def main():
    ctx = CKKS_EngineContext(1, thread_count=16)
    eng = ctx.get_engine()
    pk = ctx.get_public_key()
    rlk = ctx.get_relinearization_key()
    cjk = ctx.get_conjugation_key()

    # random plaintext vector
    vec = np.random.randint(0, 16, size=eng.slot_count, dtype=np.uint8)
    ct = eng.encrypt(vec, pk)

    # Strategy A: direct make_power_basis up to 15
    t0 = time.perf_counter()
    basis_a = eng.make_power_basis(ct, 15, rlk)
    t1 = time.perf_counter()

    # Strategy B: build_power_basis (1..8 + conjugate)
    t2 = time.perf_counter()
    basis_b = build_power_basis(eng, ct, rlk, cjk)
    t3 = time.perf_counter()

    print("--- Power basis timing (slot_count = {} ) ---".format(eng.slot_count))
    print(f"A) make_power_basis upto 15 : {(t1 - t0):.4f} s  (outputs {len(basis_a)} items)")
    print(f"B) 1..8 + conjugate method  : {(t3 - t2):.4f} s  (outputs {len(basis_b)} items)")


if __name__ == "__main__":
    main() 