from aes_ground_truth import WI_HEX_I4_TO_I43
from engine_context import CKKS_EngineContext
from aes_main_process import engine_initiation
from aes_transform_zeta import zeta_to_int
from aes_key_scheduling import key_initiation_fixed, _extract_word_hex, _extract_bytes_hex
import numpy as np

delta = [1 * 2048, 2 * 2048, 3 * 2048, 4 * 2048, 5 * 2048, 6 * 2048, 7 * 2048, 8 * 2048, 9 * 2048, 10 * 2048, 11 * 2048, 12 * 2048, 13 * 2048, 14 * 2048, 15 * 2048]

engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0, fixed_rotation=True, delta_list=delta) 

print("slot_count: ", engine_context.get_slot_count())

engine = engine_context.get_engine()
public_key = engine_context.get_public_key()

key_zeta_hi, key_zeta_lo = key_initiation_fixed()

key_zeta_hi = engine.encrypt(key_zeta_hi, public_key, level=10)
key_zeta_lo = engine.encrypt(key_zeta_lo, public_key, level=10)

enc_key_hi_list = []
enc_key_lo_list = []

mask_row_0 = np.concatenate((np.ones(4 * 2048), np.zeros(12 * 2048)))
mask_row_1 = np.concatenate((np.zeros(4 * 2048), np.ones(4 * 2048), np.zeros(8 * 2048)))
mask_row_2 = np.concatenate((np.zeros(8 * 2048), np.ones(4 * 2048), np.zeros(4 * 2048)))
mask_row_3 = np.concatenate((np.zeros(12 * 2048), np.ones(4 * 2048))) 

row_hi_0 = engine.multiply(key_zeta_hi, mask_row_0)
row_hi_1 = engine.multiply(key_zeta_hi, mask_row_1)
row_hi_2 = engine.multiply(key_zeta_hi, mask_row_2)
row_hi_3 = engine.multiply(key_zeta_hi, mask_row_3)

row_lo_0 = engine.multiply(key_zeta_lo, mask_row_0)
row_lo_1 = engine.multiply(key_zeta_lo, mask_row_1)
row_lo_2 = engine.multiply(key_zeta_lo, mask_row_2)
row_lo_3 = engine.multiply(key_zeta_lo, mask_row_3)


print("row_hi_0", _extract_bytes_hex(engine_context, row_hi_0, row_lo_0))
print("row_hi_1", _extract_bytes_hex(engine_context, row_hi_1, row_lo_1))
print("row_hi_2", _extract_bytes_hex(engine_context, row_hi_2, row_lo_2))
print("row_hi_3", _extract_bytes_hex(engine_context, row_hi_3, row_lo_3))

