# aes-main-process.py (최종 통합본)
"""
FHE-AES 전체 암호화 파이프라인을 관장하는 고수준 스크립트.
"""
from __future__ import annotations
import pathlib
import importlib.util
import numpy as np
import time
from typing import Tuple, List
from engine_context import CKKS_EngineContext

# --- 동적 임포트 헬퍼 ---
_THIS_DIR = pathlib.Path(__file__).resolve().parent
def _load_module(fname: str, alias: str):
    path = _THIS_DIR / fname
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None: raise ImportError(f"{fname} 파일을 로드할 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- FHE 연산 및 유틸리티 모듈 임포트 ---
from aes_128 import key_expansion, encrypt as ground_truth_encrypt
from aes_SubBytes import sub_bytes, NibblePack
from aes_shift_rows import shift_rows
aes_block_array = _load_module("aes_block_array.py", "aes_block_array")
aes_split_to_nibble = _load_module("aes_split_to_nibble.py", "aes_split_to_nibble")
aes_key_array = _load_module("aes_key_array.py", "aes_key_array")
aes_transform_zeta = _load_module("aes_transform_zeta.py", "aes_transform_zeta")
aes_xor = _load_module("aes_xor.py", "aes_xor")

# --- FHE 엔진 및 데이터/키 초기화 ---
def engine_initiation(signature: int, **kwargs) -> CKKS_EngineContext:
    print("FHE 엔진 및 키 생성을 시작합니다...")
    engine_context = CKKS_EngineContext(signature, **kwargs)
    print("엔진 생성이 완료되었습니다.")
    return engine_context

def data_initiation(num_blocks: int, *, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    blocks = rng.integers(0, 256, size=(num_blocks, 16), dtype=np.uint8)
    flat = aes_block_array.blocks_to_flat_array(blocks)
    upper, lower = aes_split_to_nibble.split_to_nibbles(flat)
    zeta_upper = aes_transform_zeta.int_to_zeta(upper)
    zeta_lower = aes_transform_zeta.int_to_zeta(lower)
    return blocks, zeta_upper, zeta_lower

def key_initiation(*, rng: np.random.Generator | None = None, max_blocks: int = 2048) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    rng = rng or np.random.default_rng()
    key_bytes = rng.integers(0, 256, size=16, dtype=np.uint8)
    all_round_keys_matrix = key_expansion(key_bytes)
    round_key_zeta_packs = []
    for i in range(11):
        round_key_flat_bytes = all_round_keys_matrix[i].T.flatten()
        key_flat = aes_key_array.key_to_flat_array(round_key_flat_bytes, max_blocks)
        key_upper, key_lower = aes_split_to_nibble.split_to_nibbles(key_flat)
        key_zeta_upper = aes_transform_zeta.int_to_zeta(key_upper)
        key_zeta_lower = aes_transform_zeta.int_to_zeta(key_lower)
        round_key_zeta_packs.append((key_zeta_upper, key_zeta_lower))
    return key_bytes, round_key_zeta_packs

# --- FHE 연산 래퍼 함수 ---
def add_round_key(state_pack: NibblePack, key_pack: NibblePack, context: CKKS_EngineContext) -> NibblePack:
    print("    - AddRoundKey (동형 XOR) 수행...")
    result_hi = aes_xor._xor_operation(context, state_pack.hi, key_pack.hi)
    result_lo = aes_xor._xor_operation(context, state_pack.lo, key_pack.lo)
    return NibblePack(hi=result_hi, lo=result_lo)
    
# --- 유틸리티 ---
def wait_next_stage(stage: str, next_stage: str, delay: float = 0.3) -> None:
    print(f"\n{'-'*40}\n✅ {stage} 단계 완료!\n   다음 단계: {next_stage}\n{'-'*40}\n")
    time.sleep(delay)

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    def mix_columns(state_pack: NibblePack, context: CKKS_EngineContext) -> NibblePack:
        print("    - (임시 플레이스홀더) MixColumns 수행...")
        return state_pack

    try:
        n_str = input("처리할 AES 블록 수를 입력하세요 (1–2048): ")
        n_blocks = int(n_str)
    except ValueError: raise SystemExit("❌ 잘못된 정수 입력입니다.")
    if not (1 <= n_blocks <= 2048): raise SystemExit("❌ 블록 수는 1에서 2048 사이여야 합니다.")
    
    engine_context = engine_initiation(signature=2, max_level=30, mode='parallel', thread_count=16)
    engine = engine_context.get_engine()
    public_key = engine_context.get_public_key()

    blocks, data_zeta_hi, data_zeta_lo = data_initiation(n_blocks)
    key_bytes, round_key_zeta_packs = key_initiation()
    wait_next_stage("엔진 및 데이터/키 초기화", "라운드 키 암호화")
    
    enc_data_hi = engine.encrypt(data_zeta_hi, public_key)
    enc_data_lo = engine.encrypt(data_zeta_lo, public_key)
    current_state_pack = NibblePack(hi=enc_data_hi, lo=enc_data_lo)
    
    fhe_round_keys = []
    print(f"총 {len(round_key_zeta_packs)}개의 라운드 키를 암호화합니다...")
    for i, (key_zeta_hi, key_zeta_lo) in enumerate(round_key_zeta_packs):
        enc_key_hi = engine.encrypt(key_zeta_hi, public_key)
        enc_key_lo = engine.encrypt(key_zeta_lo, public_key)
        fhe_round_keys.append(NibblePack(hi=enc_key_hi, lo=enc_key_lo))
    wait_next_stage("라운드 키 암호화", "FHE-AES 암호화 파이프라인")

    print(f"\n{'='*50}\n🚀 FHE-AES 암호화 파이프라인 시작 🚀\n{'='*50}\n")
    
    print("--- Round 0: 초기 AddRoundKey ---")
    current_state_pack = add_round_key(current_state_pack, fhe_round_keys[0], engine_context)
    print("--- Round 0 완료 ---\n")
    
    for i in range(1, 10):
        print(f"--- Round {i} ---")
        sbox_state = sub_bytes(current_state_pack.hi, current_state_pack.lo, engine_context)
        shifted_state = shift_rows(sbox_state, engine_context)
        mixed_state = mix_columns(shifted_state, engine_context)
        current_state_pack = add_round_key(mixed_state, fhe_round_keys[i], engine_context)
        print(f"--- Round {i} 완료 ---\n")
        
    print("--- Round 10 (최종) ---")
    sbox_state = sub_bytes(current_state_pack.hi, current_state_pack.lo, engine_context)
    shifted_state = shift_rows(sbox_state, engine_context)
    final_ciphertext_pack = add_round_key(shifted_state, fhe_round_keys[10], engine_context)
    print("--- Round 10 완료 ---\n")
    
    print(f"\n{'='*50}\n🔍 최종 결과 검증\n{'='*50}\n")
    expected_ciphertext_bytes = ground_truth_encrypt(blocks[0].tobytes(), key_bytes)
    
    secret_key = engine_context.get_secret_key()
    dec_final_hi = engine.decrypt(final_ciphertext_pack.hi, secret_key)
    dec_final_lo = engine.decrypt(final_ciphertext_pack.lo, secret_key)
    result_hi_nibbles = aes_transform_zeta.zeta_to_int(dec_final_hi)
    result_lo_nibbles = aes_transform_zeta.zeta_to_int(dec_final_lo)
    
    result_bytes = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        byte_val = (result_hi_nibbles[i * 2048] << 4) | result_lo_nibbles[i * 2048]
        result_bytes[i] = byte_val

    print(f"입력 평문 (0번 블록)  : {blocks[0].tobytes().hex()}")
    print(f"비밀 키                : {key_bytes.hex()}")
    print(f"{'-'*50}")
    print(f"FHE 최종 암호문       : {result_bytes.tobytes().hex()}")
    print(f"정답 암호문 (평문 AES) : {expected_ciphertext_bytes.hex()}")

    if result_bytes.tobytes() == expected_ciphertext_bytes:
        print("\n✅ 전체 파이프라인 검증 성공!")
    else:
        print("\n❌ 전체 파이프라인 검증 실패! (임시 플레이스홀더로 인해 현재는 실패가 정상입니다)")
    print(f"\n{'='*50}\n🏁 FHE-AES 파이프라인 실행 완료!\n{'='*50}")