"""Crypto, password mode and file-format tests."""
import pytest

from core.config import CIPHERS
from core.crypto import CryptoAuthenticationError, derive_key_from_password, generate_key
from core.fileformat import (
    HEADER_SIZE_PASSWORD, HEADER_SIZE_RANDOM, InvalidContainerError, decrypt_file_or_data, encrypt_file_or_data,
    pack_payload, parse_container_header, unpack_payload,
)

DATA = b"hello CryptoSelect AI " * 50


@pytest.mark.parametrize("cipher", CIPHERS)
def test_round_trip_and_filename(cipher):
    key = generate_key(cipher)
    blob, _, _ = encrypt_file_or_data(DATA, cipher, key=key, filename="report.pdf", mime_type="application/pdf")
    out, meta, header, _ = decrypt_file_or_data(blob, key=key)
    assert out == DATA and meta["filename"] == "report.pdf" and header.cipher_name == cipher


@pytest.mark.parametrize("cipher", CIPHERS)
def test_password_mode(cipher):
    blob, _, _ = encrypt_file_or_data(DATA, cipher, key_mode="password", password="pw-12345678")
    header, _ = parse_container_header(blob)
    assert header.salt is not None and len(header.header_bytes) == HEADER_SIZE_PASSWORD
    assert decrypt_file_or_data(blob, password="pw-12345678")[0] == DATA
    with pytest.raises(CryptoAuthenticationError):
        decrypt_file_or_data(blob, password="nope")


def test_same_password_different_salt_gives_different_keys():
    s1, s2 = b"a" * 16, b"b" * 16
    assert derive_key_from_password("pw", s1, CIPHERS[1]) != derive_key_from_password("pw", s2, CIPHERS[1])


def test_header_layout_random_mode():
    blob, _, _ = encrypt_file_or_data(b"x", CIPHERS[0], key=generate_key(CIPHERS[0]))
    assert blob[:4] == b"CSAI" and blob[4] == 1 and blob[5] == 0x01 and blob[6] == 0x01
    assert len(parse_container_header(blob)[0].header_bytes) == HEADER_SIZE_RANDOM


def test_pack_unpack_payload():
    raw, meta = unpack_payload(pack_payload(b"abc", "f.txt", "text/plain"))
    assert raw == b"abc" and meta["filename"] == "f.txt"


@pytest.mark.parametrize("bad", [b"", b"CSAI", b"XXXX" + b"\x00" * 60, b"CSAI\x09" + b"\x00" * 60])
def test_invalid_containers_raise_clean_error(bad):
    with pytest.raises(InvalidContainerError):
        parse_container_header(bad)


@pytest.mark.parametrize("cipher", CIPHERS)
def test_any_bit_flip_is_rejected(cipher):
    key = generate_key(cipher)
    blob, _, _ = encrypt_file_or_data(b"payload", cipher, key=key)
    for pos in (5, 8, 15, len(blob) - 1):
        bad = bytearray(blob)
        bad[pos] ^= 1
        with pytest.raises((CryptoAuthenticationError, InvalidContainerError, ValueError)):
            decrypt_file_or_data(bytes(bad), key=key)
