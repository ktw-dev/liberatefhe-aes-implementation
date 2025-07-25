"""engine_context.py – Convenience wrapper around desilofhe.Engine

Provides FHEContext class that instantiates an Engine and eagerly generates
all commonly-used keys (secret, public, relinearisation, conjugation, rotation).

Usage example
-------------
>>> from desilo_context import FHEContext
>>> ctx = FHEContext(max_level=30, thread_count=8)
>>> ct = ctx.encrypt([1,2,3,4])
>>> pt = ctx.decrypt(ct)

The single context instance can be passed to helper functions like
`_xor_operation(ctx, enc_a, enc_b)`.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from desilofhe import Engine

__all__ = ["FHEContext"]


class CKKS_EngineContext:
    """High-level container that owns an Engine and all related keys."""

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self,  
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
             special_prime_count: int = 0) -> None:
        """Create an Engine and generate all default keys.
        
        지원되는 생성자 시그니처
        1. Engine(mode:str='cpu', use_bootstrap:bool=False, use_multiparty: bool = False, thread_count: int = 0, device_id: int = 0)
        2. Engine(max_level: int, mode: str = ‘cpu’, *, use_multiparty: bool = False, thread_count: int = 0, device_id: int = 0)
        3. Engine(log_coeff_count: int, special_prime_count: int, mode: str = ‘cpu’, *, use_multiparty: bool = False, thread_count: int = 0, device_id: int = 0)
        """
        if signature == 1:
            self.engine = Engine(
                mode=mode,
                use_bootstrap=use_bootstrap,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id
            )
        elif signature == 2:
            self.engine = Engine(
                max_level=max_level,
                mode=mode,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id
            )
        elif signature == 3:
            self.engine = Engine(
                log_coeff_count=log_coeff_count,
                special_prime_count=special_prime_count,
                mode=mode,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id
            )
        else:
            raise ValueError(f"Unsupported signature: {signature}")

        self.fixed_rotation_key_list = []
        # ---- Key generation ------------------------------------------------
        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relinearization_key = self.engine.create_relinearization_key(self.secret_key)
        self.conjugation_key = self.engine.create_conjugation_key(self.secret_key)
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)

        if fixed_rotation and delta_list is not None:
            for delta in delta_list:
                self.fixed_rotation_key_list.append(self.engine.create_fixed_rotation_key(self.secret_key, delta))

        # Some applications may not need small bootstrap; keep optional
        self.small_bootstrap_key = self.engine.create_small_bootstrap_key(self.secret_key)
        self.bootstrap_key = self.engine.create_bootstrap_key(self.secret_key)

    # ---------------------------------------------------------------------
    # Representation helpers
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FHEContext(engine=Engine(slot_count={self.engine.slot_count}), "
            f"keys=[sk, pk, rlk, cjk, rot])"
        ) 
        
    # ---------------------------------------------------------------------
    # Getters
    # ---------------------------------------------------------------------
    def get_max_level(self) -> int:
        return self.engine.max_level
    
    def get_slot_count(self) -> int:
        return self.engine.slot_count
    
    def get_mode(self) -> str:
        return self.engine.mode
    
    def get_public_key(self) -> "desilofhe.PublicKey":
        return self.public_key
    
    def get_secret_key(self) -> "desilofhe.SecretKey":
        return self.secret_key
    
    def get_relinearization_key(self) -> "desilofhe.RelinearizationKey":
        return self.relinearization_key
    
    def get_conjugation_key(self) -> "desilofhe.ConjugationKey":
        return self.conjugation_key
    
    def get_rotation_key(self) -> "desilofhe.RotationKey":
        return self.rotation_key
    
    def get_small_bootstrap_key(self) -> "desilofhe.SmallBootstrapKey":
        return self.small_bootstrap_key
    
    def get_bootstrap_key(self) -> "desilofhe.BootstrapKey":
        return self.bootstrap_key
    
    def get_engine(self):
        return self.engine