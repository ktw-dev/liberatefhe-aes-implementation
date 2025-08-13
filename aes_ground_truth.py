from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


# -----------------------------------------------------------------------------
# Public constants: initial key (AES-128 test vector from FIPS-197)
# -----------------------------------------------------------------------------

# 16 bytes: 2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c
KEY_BYTES: bytes = bytes(
    [
        0x2B, 0x7E, 0x15, 0x16,
        0x28, 0xAE, 0xD2, 0xA6,
        0xAB, 0xF7, 0x15, 0x88,
        0x09, 0xCF, 0x4F, 0x3C,
    ]
)


# -----------------------------------------------------------------------------
# AES S-Box and Rcon tables (standard, fixed)
# -----------------------------------------------------------------------------

SBOX: List[int] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
    0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
    0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
    0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
    0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
    0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
    0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
    0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
    0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16,
]

# Rcon for AES-128 (round 1..10). We store 11 entries with a dummy 0 at index 0.
RCON: List[int] = [
    0x00,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36,
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _rot_word(word: int) -> int:
    """Rotate 32-bit word left by 8 bits."""
    return ((word << 8) & 0xFFFFFFFF) | (word >> 24)


def _sub_word(word: int) -> int:
    """Apply AES S-Box to each byte of a 32-bit word."""
    b0 = SBOX[(word >> 24) & 0xFF]
    b1 = SBOX[(word >> 16) & 0xFF]
    b2 = SBOX[(word >> 8) & 0xFF]
    b3 = SBOX[word & 0xFF]
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


def _rcon_word(round_index: int) -> int:
    """Return Rcon[round_index] as a 32-bit word at the MSByte position."""
    return RCON[round_index] << 24


def _bytes_to_words(key_bytes: bytes) -> List[int]:
    """Split 16 bytes into four big-endian 32-bit words."""
    assert len(key_bytes) == 16
    return [
        (key_bytes[0] << 24)
        | (key_bytes[1] << 16)
        | (key_bytes[2] << 8)
        | key_bytes[3],
        (key_bytes[4] << 24)
        | (key_bytes[5] << 16)
        | (key_bytes[6] << 8)
        | key_bytes[7],
        (key_bytes[8] << 24)
        | (key_bytes[9] << 16)
        | (key_bytes[10] << 8)
        | key_bytes[11],
        (key_bytes[12] << 24)
        | (key_bytes[13] << 16)
        | (key_bytes[14] << 8)
        | key_bytes[15],
    ]


def to_hex(word: int) -> str:
    """Return 8-hex-digit lowercase string for a 32-bit word."""
    return f"{word:08x}"


@dataclass(frozen=True)
class Step:
    """One expansion step for AES-128 when i >= 4.

    Fields store the intermediate results that match tables from FIPS-197.
    All integers are 32-bit words.
    """

    index: int
    temp: int
    rotword: int | None
    subword: int | None
    rcon_word: int | None
    xor_with_rcon: int | None
    w_imnk: int
    w_i: int


def expand_key_128(key_bytes: bytes) -> Tuple[List[int], List[Step]]:
    """Expand 16-byte AES-128 key into 44 words and record per-step details."""
    Nk = 4
    Nb = 4
    Nr = 10

    words: List[int] = _bytes_to_words(key_bytes)
    steps: List[Step] = []

    # i from 4..43
    round_index = 1
    for i in range(Nk, Nb * (Nr + 1)):
        temp = words[i - 1]

        if i % Nk == 0:
            rotated = _rot_word(temp)
            substituted = _sub_word(rotated)
            rconw = _rcon_word(round_index)
            temp2 = substituted ^ rconw
            round_index += 1
        else:
            rotated = None
            substituted = None
            rconw = None
            temp2 = temp

        w_imnk = words[i - Nk]
        w_i = w_imnk ^ temp2

        words.append(w_i)

        steps.append(
            Step(
                index=i,
                temp=temp,
                rotword=rotated,
                subword=substituted,
                rcon_word=rconw,
                xor_with_rcon=temp2 if i % Nk == 0 else None,
                w_imnk=w_imnk,
                w_i=w_i,
            )
        )

    return words, steps


def get_words() -> List[int]:
    """Return the 44 32-bit words for the public test key."""
    return WORDS.copy()


def get_round_keys() -> List[bytes]:
    """Return the 11 round keys (16 bytes each) for the public test key."""
    rk: List[bytes] = []
    for r in range(11):
        start = r * 4
        block = b"".join(WORDS[start + j].to_bytes(4, "big") for j in range(4))
        rk.append(block)
    return rk


def get_steps() -> List[Step]:
    """Return the recorded per-word expansion steps for the public test key."""
    return STEPS.copy()


def as_table() -> List[Dict[str, str]]:
    """Return a human-readable table of the expansion for the public key.

    Each row matches columns often shown in textbooks:
      i, temp, RotWord(temp), SubWord(), Rcon[i/Nk], temp^Rcon, w[i-Nk], w[i]
    Values are hex strings (8 hex digits).
    """
    rows: List[Dict[str, str]] = []
    for st in STEPS:
        rows.append(
            {
                "i": str(st.index),
                "temp": to_hex(st.temp),
                "RotWord(temp)": to_hex(st.rotword) if st.rotword is not None else "",
                "SubWord()": to_hex(st.subword) if st.subword is not None else "",
                "Rcon[i/Nk]": to_hex(st.rcon_word) if st.rcon_word is not None else "",
                "temp^Rcon": to_hex(st.xor_with_rcon) if st.xor_with_rcon is not None else "",
                "w[i-Nk]": to_hex(st.w_imnk),
                "w[i]": to_hex(st.w_i),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Pre-compute ground truth for the provided test key so this module can be
# imported anywhere and the schedule reused without recomputation.
# -----------------------------------------------------------------------------

WORDS, STEPS = expand_key_128(KEY_BYTES)

# 11 round keys (AES-128)
ROUND_KEYS = get_round_keys()


def get_wi_hex_i4_to_i43() -> List[str]:
    """Return hex strings of w[i] for i=4..43 (40 words).

    This corresponds to the expanded words beyond the initial key words.
    """
    return [to_hex(w) for w in WORDS[4:44]]


# Precomputed convenience list for easy import/use
WI_HEX_I4_TO_I43: List[str] = get_wi_hex_i4_to_i43()


__all__ = [
    "KEY_BYTES",
    "SBOX",
    "RCON",
    "Step",
    "WORDS",
    "STEPS",
    "ROUND_KEYS",
    "WI_HEX_I4_TO_I43",
    "to_hex",
    "expand_key_128",
    "get_words",
    "get_steps",
    "get_round_keys",
    "as_table",
    "get_wi_hex_i4_to_i43",
]


# -----------------------------------------------------------------------------
# SIMD helpers for round keys
# -----------------------------------------------------------------------------

def round_key_to_simd_vector(key_block: bytes, block_slots: int = 2048) -> np.ndarray:
    """Expand a 16-byte AES round key into a 1-D SIMD layout of length 2^15.

    The output vector has 16 contiguous blocks, each of length ``block_slots``.
    The j-th block (0-based) is filled with the j-th byte of ``key_block``.
    By default ``block_slots`` is 2048, producing a vector of length 32768.
    """
    assert len(key_block) == 16
    out = np.empty(16 * block_slots, dtype=np.uint8)
    for j in range(16):
        start = j * block_slots
        out[start : start + block_slots].fill(key_block[j])
    return out


def get_round_keys_simd(block_slots: int = 2048) -> List[np.ndarray]:
    """Return all 11 round keys expanded into SIMD vectors (len = 16*block_slots)."""
    return [round_key_to_simd_vector(rk, block_slots) for rk in ROUND_KEYS]


__all__ += [
    "round_key_to_simd_vector",
    "get_round_keys_simd",
]


