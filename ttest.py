from engine_context import CKKS_EngineContext
from aes_main_process import engine_initiation
from aes_transform_zeta import zeta_to_int, int_to_zeta
from aes_key_scheduling import key_initiation_fixed, _extract_word_hex, _extract_bytes_hex, _rot_word, _sub_word, _rcon_xor, _xor
import numpy as np

mask1 = np.concatenate([np.zeros(12 * 2048, dtype=np.uint8), np.ones(4 * 2048, dtype=np.uint8)])
mask2 = np.concatenate([np.ones(12 * 2048, dtype=np.uint8), np.zeros(4 * 2048, dtype=np.uint8)])

num_array = np.array([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F])
num_array = np.repeat(num_array, 2048)

engine_context = engine_initiation(signature=1, mode='parallel', use_bootstrap=True, thread_count = 16, device_id = 0)
engine = engine_context.get_engine()

# split to hi and lo
num_array_hi = ((num_array >> 4) & 0x0F).astype(np.uint8)
num_array_lo = (num_array & 0x0F).astype(np.uint8)

num_array_hi = int_to_zeta(num_array_hi)
num_array_lo = int_to_zeta(num_array_lo)

enc_num_array_hi = engine.encrypt(num_array_hi, engine_context.get_public_key())
enc_num_array_lo = engine.encrypt(num_array_lo, engine_context.get_public_key())

enc_masked_num_array_hi_1 = engine.multiply(enc_num_array_hi, mask1)
enc_masked_num_array_lo_1 = engine.multiply(enc_num_array_lo, mask1)

enc_masked_num_array_hi_2 = engine.multiply(enc_num_array_hi, mask2)
enc_masked_num_array_lo_2 = engine.multiply(enc_num_array_lo, mask2)

enc_masked_num_array_hi_2 = engine.rotate(enc_masked_num_array_hi_2, engine_context.get_fixed_rotation_key(4 * 2048))
enc_masked_num_array_lo_2 = engine.rotate(enc_masked_num_array_lo_2, engine_context.get_fixed_rotation_key(4 * 2048))

rotated_num_array_hi_1, rotated_num_array_lo_1 = _rot_word(engine_context,enc_masked_num_array_hi_1, enc_masked_num_array_lo_1)
subbed_num_array_hi_1, subbed_num_array_lo_1 = _sub_word(engine_context, rotated_num_array_hi_1, rotated_num_array_lo_1)

enc_num_array_hi_1 = engine.add(subbed_num_array_hi_1, enc_masked_num_array_hi_2)
enc_num_array_lo_1 = engine.add(subbed_num_array_lo_1, enc_masked_num_array_lo_2)

dec_num_array_hi_1 = engine.decrypt(enc_num_array_hi_1, engine_context.get_secret_key())
dec_num_array_lo_1 = engine.decrypt(enc_num_array_lo_1, engine_context.get_secret_key())

dec_num_array_hi_1 = zeta_to_int(dec_num_array_hi_1)
dec_num_array_lo_1 = zeta_to_int(dec_num_array_lo_1)

dec_num_array_hi_1 = (dec_num_array_hi_1 << 4) & 0xFF
dec_num_array_lo_1 = dec_num_array_lo_1 & 0xFF

dec_num_array = dec_num_array_hi_1 | dec_num_array_lo_1

for i in range(16):
    idx = i * 2048
    print("dec_num_array[idx:idx+4]", [hex(x) for x in dec_num_array[idx:idx+4]])
    print("--------------------------------")