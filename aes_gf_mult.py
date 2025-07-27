"""const_mult.py

Utility function to perform slot-wise constant multiplication in the ζ-domain
using desilofhe Engine.  Given a ciphertext that encrypts ζ^a and a small
non-negative integer `const`, the function returns a ciphertext encrypting
ζ^{a*const} by performing `const` homomorphic multiplications.

Note
----
This routine is naive (linear-time) but simple: it multiplies the ciphertext
`ciphertext_base` with itself `const` − 1 times.  For tiny constants (≤15) the depth
and noise growth are well within typical CKKS budgets.  For larger constants
an exponentiation-by-squaring strategy would be preferable.

Parameters
~~~~~~~~~~
engine_context   : CKKS_EngineContext
        Context that owns the Engine and keys.
ciphertext_base : Ciphertext
        Ciphertext encrypting ζ^a (vector).  Must be at the desired input
        scale.
const : int
        Non-negative integer (e.g. 0-15).  The function computes ζ^{a*const}.

Returns
~~~~~~~
Ciphertext at the same scale as `ciphertext_base` (modulo rescaling) that encrypts
ζ^{a*const}.
"""

from __future__ import annotations
from typing import Any
import numpy as np  # Only for type hints; not used inside core logic.
from engine_context import CKKS_EngineContext

__all__ = ["gf_mult"]


def gf_mult(engine_context: CKKS_EngineContext, ciphertext_base: Any, const: int,
):
    """Return ciphertext encrypting ζ^{a*const} given ct(ζ^a) and small constant.

    Parameters
    ----------
    engine_context : CKKS_EngineContext
        Context that owns the Engine.
    ciphertext_base : desilofhe.Ciphertext
        Ciphertext encrypting ζ^a.
    const : int
        Constant multiplier (0–15 typically).
    """
    engine = engine_context.get_engine()
    relin_key = engine_context.get_relinearization_key()

    if const == 1:
        return ciphertext_base
    # Naive repeated multiplication.
    result = ciphertext_base
    for _ in range(const - 1):
        result = engine.multiply(result, ciphertext_base, relin_key)
    return result 