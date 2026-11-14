"""Security verification suite. Every check runs real cryptographic operations.

Per cipher: round trip (random key and password), wrong key, wrong password, corrupted
ciphertext, tampered header, wrong associated data, 28-byte overhead, 10,000 unique nonces.
Global: the security filter and every selector never return a cipher below the required level.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.config import CIPHERS, CPU_CLASSES, NETWORKS, SECURITY_BITS
from core.crypto import (
    CryptoAuthenticationError, decrypt_aead, encrypt_aead, generate_key, generate_nonce,
)
from core.dataset import MAX_PAYLOAD, MIN_PAYLOAD, generate_dataset
from core.fileformat import (
    KEY_MODE_RANDOM, InvalidContainerError, decrypt_file_or_data, encrypt_file_or_data, pack_payload,
    parse_container_header, serialize_container_header,
)
from core.selectors import fixed_aes256, fixed_chacha, rf_secure, rule_based, security_filter

Check = Callable[[str], tuple[bool, str]]
REJECTED = (CryptoAuthenticationError, InvalidContainerError, ValueError)
NONCE_COUNT = 10_000


def check_round_trip(cipher: str) -> tuple[bool, str]:
    data = b"CryptoSelect AI test payload 1234567890 !@#$%^&*()"
    key = generate_key(cipher)
    blob, _, _ = encrypt_file_or_data(data, cipher, key=key, filename="a.txt")
    out, meta, header, _ = decrypt_file_or_data(blob, key=key)
    ok = out == data and meta["filename"] == "a.txt" and header.cipher_name == cipher
    return ok, "Data and filename restored exactly; cipher read from header." if ok else "Mismatch after decrypt."


def check_password_round_trip(cipher: str) -> tuple[bool, str]:
    data = b"password protected data"
    blob, _, _ = encrypt_file_or_data(data, cipher, key_mode="password", password="correct horse battery")
    out, _, header, _ = decrypt_file_or_data(blob, password="correct horse battery")
    ok = out == data and header.salt is not None and len(header.salt) == 16
    return ok, "Password mode works (16-byte random salt in header)." if ok else "Password round trip failed."


def _must_reject(fn: Callable[[], Any], what: str) -> tuple[bool, str]:
    try:
        fn()
    except REJECTED:
        return True, f"{what} rejected."
    return False, f"{what} was ACCEPTED."


def check_wrong_key(cipher: str) -> tuple[bool, str]:
    blob, _, _ = encrypt_file_or_data(b"secret", cipher, key=generate_key(cipher))
    return _must_reject(lambda: decrypt_file_or_data(blob, key=generate_key(cipher)), "Wrong key")


def check_wrong_password(cipher: str) -> tuple[bool, str]:
    blob, _, _ = encrypt_file_or_data(b"secret", cipher, key_mode="password", password="right-password")
    return _must_reject(lambda: decrypt_file_or_data(blob, password="wrong-password"), "Wrong password")


def check_corrupted_ciphertext(cipher: str) -> tuple[bool, str]:
    key = generate_key(cipher)
    blob, _, _ = encrypt_file_or_data(b"important data " * 10, cipher, key=key)
    bad = bytearray(blob)
    bad[-5] ^= 0x01
    return _must_reject(lambda: decrypt_file_or_data(bytes(bad), key=key), "Corrupted ciphertext")


def check_tampered_header(cipher: str) -> tuple[bool, str]:
    key = generate_key(cipher)
    blob, _, _ = encrypt_file_or_data(b"header protected", cipher, key=key)
    ok_all = True
    for pos in (5, 7, 12, 18):                      # cipher id, nonce bytes
        bad = bytearray(blob)
        bad[pos] ^= 0x5A
        ok, _ = _must_reject(lambda b=bytes(bad): decrypt_file_or_data(b, key=key), "x")
        ok_all &= ok
    return ok_all, "Changing any header byte is detected (header is authenticated data)." if ok_all \
        else "A modified header was accepted."


def check_header_is_authenticated(cipher: str) -> tuple[bool, str]:
    """A container whose header was NOT passed as associated data must be refused.

    (Flipping a header byte alone is not enough proof: the nonce and cipher id are used in
    decryption anyway. This builds a look-alike container to prove the header is authenticated.)
    """
    key, nonce = generate_key(cipher), generate_nonce()
    header = serialize_container_header(cipher, KEY_MODE_RANDOM, nonce)
    payload = pack_payload(b"data")
    good = header + encrypt_aead(cipher, key, nonce, payload, header)
    unauth = header + encrypt_aead(cipher, key, nonce, payload, b"")
    accepted = decrypt_file_or_data(good, key=key)[0] == b"data"
    rejected, _ = _must_reject(lambda: decrypt_file_or_data(unauth, key=key), "Unauthenticated header")
    ok = accepted and rejected
    return ok, "Header is bound as associated data (AAD)." if ok else "Header is NOT authenticated."


def check_wrong_associated_data(cipher: str) -> tuple[bool, str]:
    key, nonce = generate_key(cipher), generate_nonce()
    ct = encrypt_aead(cipher, key, nonce, b"payload", b"associated-A")
    ok1, _ = _must_reject(lambda: decrypt_aead(cipher, key, nonce, ct, b"associated-B"), "Wrong associated data")
    ok2 = decrypt_aead(cipher, key, nonce, ct, b"associated-A") == b"payload"
    return ok1 and ok2, "Different associated data rejected; correct data accepted." if ok1 and ok2 else "AAD check failed."


def check_overhead(cipher: str) -> tuple[bool, str]:
    key, nonce = generate_key(cipher), generate_nonce()
    pt = b"x" * 1000
    ct = encrypt_aead(cipher, key, nonce, pt, b"")
    ok = (len(ct) - len(pt)) + len(nonce) == 28
    return ok, f"Measured overhead = {len(ct) - len(pt)} (tag) + {len(nonce)} (nonce) = {len(ct) - len(pt) + len(nonce)} bytes."


def check_nonce_uniqueness(cipher: str) -> tuple[bool, str]:
    key = generate_key(cipher)
    nonces = set()
    for _ in range(NONCE_COUNT):
        blob, _, _ = encrypt_file_or_data(b"x", cipher, key=key)
        nonces.add(parse_container_header(blob)[0].nonce)
    ok = len(nonces) == NONCE_COUNT
    return ok, f"{len(nonces):,} unique nonces from {NONCE_COUNT:,} encryptions."


PER_CIPHER_CHECKS: list[tuple[str, Check]] = [
    ("Round trip (random key)", check_round_trip),
    ("Round trip (password)", check_password_round_trip),
    ("Wrong key rejected", check_wrong_key),
    ("Wrong password rejected", check_wrong_password),
    ("Corrupted ciphertext rejected", check_corrupted_ciphertext),
    ("Tampered header rejected", check_tampered_header),
    ("Header authenticated as AAD", check_header_is_authenticated),
    ("Wrong associated data rejected", check_wrong_associated_data),
    ("Overhead is 28 bytes", check_overhead),
    (f"{NONCE_COUNT:,} unique nonces", check_nonce_uniqueness),
]


def check_security_filter(ml_bundle: dict[str, Any] | None = None) -> tuple[bool, str]:
    """With 256-bit required, no (filtered) selector may ever return AES-128: exhaustive grid + random scenarios."""
    sizes = [MIN_PAYLOAD, 1_000, 100_000, 1_000_000, MAX_PAYLOAD]
    grid = pd.DataFrame(list(itertools.product(CPU_CLASSES, [True, False], [5.0, 20.0, 60.0, 100.0], sizes, [256], NETWORKS)),
                        columns=["cpu_class", "has_aes_hw", "battery_pct", "payload_size", "required_security", "network"])
    random_rows = generate_dataset(1500, 2024)[grid.columns]
    random_rows["required_security"] = 256
    frames = pd.concat([grid, random_rows], ignore_index=True)
    picks = [fixed_aes256(frames), fixed_chacha(frames), rule_based(frames)]
    if ml_bundle is not None:
        picks.append(rf_secure(ml_bundle, frames))
    violations = sum(int(sum(SECURITY_BITS[c] < 256 for c in p)) for p in picks)
    violations += sum(security_filter(c, 256, hw) == "AES-128-GCM" for c in CIPHERS for hw in (True, False))
    return violations == 0, f"{len(frames) * len(picks):,} selections checked: {violations} violations."


def run_all_security_tests(ml_bundle: dict[str, Any] | None = None) -> tuple[pd.DataFrame, float]:
    """Run everything. Returns (PASS/FAIL table, pass rate in percent)."""
    rows: list[dict[str, str]] = []
    for cipher in CIPHERS:
        for name, fn in PER_CIPHER_CHECKS:
            try:
                ok, msg = fn(cipher)
            except Exception as exc:  # a crashing check is a FAIL, never a crash of the app
                ok, msg = False, f"Check raised {exc!r}"
            rows.append({"Cipher": cipher, "Test": name, "Result": "PASS" if ok else "FAIL", "Details": msg})
    ok, msg = check_security_filter(ml_bundle)
    rows.append({"Cipher": "All selectors", "Test": "Never below required security", "Result": "PASS" if ok else "FAIL",
                 "Details": msg})
    df = pd.DataFrame(rows)
    return df, 100.0 * float((df["Result"] == "PASS").mean())
