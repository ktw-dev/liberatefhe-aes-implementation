from engine_context import CKKS_EngineContext
import numpy as np
import time
import math
import importlib.util
import pathlib


_THIS_DIR = pathlib.Path(__file__).parent


def _load_module(fname: str, alias: str):
    """Load a Python file in the current directory as a module with *alias*."""
    path = _THIS_DIR / fname
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Cannot load {fname}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

xor_module = _load_module("aes_xor.py", "xor_operation")
_xor_operation = xor_module._xor_operation

data_module = _load_module("aes_main_process.py", "data_initiation")
_data_initiation = data_module.data_initiation

def transform_to_zeta(arr: np.ndarray) -> np.ndarray:
    result = np.exp(-2j * np.pi * (arr % 16) / 16)
    return result

def zeta_to_int(zeta_arr: np.ndarray) -> np.ndarray:
    """Inverse of `transform_to_zeta` assuming unit-magnitude complex numbers.

    Values are mapped back to integers 0‥15 by measuring their phase.
    """
    angles = np.angle(zeta_arr)  # range (-π, π]
    k      = (-angles * 16) / (2 * np.pi)
    k      = np.mod(np.rint(k), 16).astype(np.uint8)
    return k


engine_context = CKKS_EngineContext(mode="parallel", use_bootstrap=True, thread_count=16, device_id=0)

engine = engine_context.engine
public_key = engine_context.get_public_key()
secret_key = engine_context.get_secret_key()
relinearization_key = engine_context.get_relinearization_key()
conjugation_key = engine_context.get_conjugation_key()
rotation_key = engine_context.get_rotation_key()
small_bootstrap_key = engine_context.get_small_bootstrap_key()
bootstrap_key = engine_context.get_bootstrap_key()

blocks, flat, _, _, zeta_hi_list, zeta_lo_list = _data_initiation(2048)

enc_zeta_hi_list = [engine.encrypt(zeta_hi, engine_context.public_key) for zeta_hi in zeta_hi_list]
enc_zeta_lo_list = [engine.encrypt(zeta_lo, engine_context.public_key) for zeta_lo in zeta_lo_list]

start_time = time.time()

print(f"enc_zeta_hi_list's level: {enc_zeta_hi_list[0].level}")
xor_list = [_xor_operation(engine_context, enc_zeta_hi_list[0], enc_zeta_lo_list[0])]
print(f"xor_list's level: {xor_list[0].level}")

end_time = time.time()
print(f"XOR time taken: {end_time - start_time} seconds")

start_time = time.time()


xor_list[0] = engine_context.engine.bootstrap(xor_list[0], relinearization_key, conjugation_key, bootstrap_key)
print(f"xor_list's level: {xor_list[0].level}")

end_time = time.time()
print(f"bootstrapping time taken: {end_time - start_time} seconds")
