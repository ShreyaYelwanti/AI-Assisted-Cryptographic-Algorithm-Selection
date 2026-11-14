"""Self-describing encrypted container format for CryptoSelect AI.

Binary layout:
--------------------------------------------------------------------------------
Offset (Random) | Offset (Password) | Field        | Size (bytes) | Description
--------------------------------------------------------------------------------
0               | 0                 | Magic        | 4            | b"CSAI"
4               | 4                 | Version      | 1            | 0x01
5               | 5                 | Cipher ID    | 1            | 0x01: AES-128-GCM
                |                   |              |              | 0x02: AES-256-GCM
                |                   |              |              | 0x03: ChaCha20-Poly1305
6               | 6                 | Key Mode     | 1            | 0x01: Random key
                |                   |              |              | 0x02: Password-derived (Scrypt)
-               | 7                 | Salt         | 16 (if pwd)  | Random Scrypt salt
7               | 23                | Nonce        | 12           | Fresh random 12-byte nonce
19              | 35                | Ciphertext   | Variable     | AEAD-encrypted payload + 16-byte tag
--------------------------------------------------------------------------------

The entire header (19 bytes for random, 35 bytes for password) is passed as
Associated Data (AAD) to the AEAD cipher, guaranteeing tamper detection for all metadata.
Inside the ciphertext payload, original filename and mime type are safely encapsulated:
[4 bytes: metadata_len][metadata_json][raw_data]
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Any, Final

from core.crypto import (
    CIPHER_ID_MAP,
    CRYPTO_OVERHEAD_BYTES,
    ID_TO_CIPHER_MAP,
    NONCE_LENGTH_BYTES,
    SALT_LENGTH_BYTES,
    TAG_LENGTH_BYTES,
    CryptoAuthenticationError,
    CryptoError,
    decrypt_aead,
    derive_key_from_password,
    encrypt_aead,
    generate_key,
    generate_nonce,
    generate_salt,
    validate_cipher_name,
)


class InvalidContainerError(CryptoError):
    """Raised when file structure, magic bytes, or version are invalid."""
    pass


MAGIC_BYTES: Final[bytes] = b"CSAI"
CONTAINER_VERSION: Final[int] = 0x01

KEY_MODE_RANDOM: Final[int] = 0x01
KEY_MODE_PASSWORD: Final[int] = 0x02

KEY_MODE_NAME_MAP: Final[dict[int, str]] = {
    KEY_MODE_RANDOM: "Random Key",
    KEY_MODE_PASSWORD: "Password-Derived (Scrypt)",
}

HEADER_SIZE_RANDOM: Final[int] = 4 + 1 + 1 + 1 + NONCE_LENGTH_BYTES  # 19 bytes
HEADER_SIZE_PASSWORD: Final[int] = (
    4 + 1 + 1 + 1 + SALT_LENGTH_BYTES + NONCE_LENGTH_BYTES  # 35 bytes
)


@dataclass(frozen=True)
class ContainerHeader:
    """Parsed header metadata from a CryptoSelect AI encrypted container."""

    magic: bytes
    version: int
    cipher_id: int
    cipher_name: str
    key_mode: int
    key_mode_name: str
    salt: bytes | None
    nonce: bytes
    header_bytes: bytes


def serialize_container_header(
    cipher_name: str,
    key_mode: int,
    nonce: bytes,
    salt: bytes | None = None,
) -> bytes:
    """Construct the raw self-describing header bytes."""
    validate_cipher_name(cipher_name)
    cipher_id = CIPHER_ID_MAP[cipher_name]

    if key_mode not in (KEY_MODE_RANDOM, KEY_MODE_PASSWORD):
        raise ValueError(f"Invalid key mode: {key_mode}")

    if len(nonce) != NONCE_LENGTH_BYTES:
        raise ValueError(f"Nonce must be {NONCE_LENGTH_BYTES} bytes.")

    header_parts = [
        MAGIC_BYTES,
        bytes([CONTAINER_VERSION]),
        bytes([cipher_id]),
        bytes([key_mode]),
    ]

    if key_mode == KEY_MODE_PASSWORD:
        if salt is None or len(salt) != SALT_LENGTH_BYTES:
            raise ValueError(
                f"Password mode requires a {SALT_LENGTH_BYTES}-byte salt."
            )
        header_parts.append(salt)

    header_parts.append(nonce)
    return b"".join(header_parts)


def parse_container_header(data: bytes) -> tuple[ContainerHeader, bytes]:
    """Parse and validate container header from raw file bytes.

    Returns:
        tuple of (ContainerHeader, remaining_ciphertext_bytes)
    """
    if len(data) < HEADER_SIZE_RANDOM:
        raise InvalidContainerError(
            f"File is too short ({len(data)} bytes) to contain a valid CSAI header."
        )

    magic = data[:4]
    if magic != MAGIC_BYTES:
        raise InvalidContainerError(
            f"Invalid file format: expected magic 'CSAI', found {magic!r}"
        )

    version = data[4]
    if version != CONTAINER_VERSION:
        raise InvalidContainerError(
            f"Unsupported container version {version} (expected {CONTAINER_VERSION})."
        )

    cipher_id = data[5]
    if cipher_id not in ID_TO_CIPHER_MAP:
        raise InvalidContainerError(f"Unknown cipher ID: {cipher_id:#04x}")
    cipher_name = ID_TO_CIPHER_MAP[cipher_id]

    key_mode = data[6]
    if key_mode not in KEY_MODE_NAME_MAP:
        raise InvalidContainerError(f"Unknown key mode: {key_mode:#04x}")

    offset = 7
    salt: bytes | None = None

    if key_mode == KEY_MODE_PASSWORD:
        if len(data) < HEADER_SIZE_PASSWORD:
            raise InvalidContainerError(
                f"Password mode container truncated: length {len(data)} < {HEADER_SIZE_PASSWORD}"
            )
        salt = data[offset : offset + SALT_LENGTH_BYTES]
        offset += SALT_LENGTH_BYTES

    nonce = data[offset : offset + NONCE_LENGTH_BYTES]
    offset += NONCE_LENGTH_BYTES

    header_bytes = data[:offset]
    ciphertext = data[offset:]

    if len(ciphertext) < TAG_LENGTH_BYTES:
        raise InvalidContainerError(
            f"File has no ciphertext/tag: remaining length is {len(ciphertext)} bytes."
        )

    header = ContainerHeader(
        magic=magic,
        version=version,
        cipher_id=cipher_id,
        cipher_name=cipher_name,
        key_mode=key_mode,
        key_mode_name=KEY_MODE_NAME_MAP[key_mode],
        salt=salt,
        nonce=nonce,
        header_bytes=header_bytes,
    )
    return header, ciphertext


def pack_payload(
    raw_data: bytes,
    filename: str = "payload.bin",
    mime_type: str = "application/octet-stream",
) -> bytes:
    """Pack raw data and metadata into a single confidential payload block."""
    meta = {
        "filename": filename or "payload.bin",
        "mime_type": mime_type or "application/octet-stream",
        "timestamp": time.time(),
        "size": len(raw_data),
    }
    meta_bytes = json.dumps(meta).encode("utf-8")
    meta_len = len(meta_bytes)
    return struct.pack(">I", meta_len) + meta_bytes + raw_data


def unpack_payload(payload: bytes) -> tuple[bytes, dict[str, Any]]:
    """Unpack raw data and metadata from decrypted payload block."""
    if len(payload) < 4:
        raise InvalidContainerError("Decrypted payload is truncated.")

    meta_len = struct.unpack(">I", payload[:4])[0]
    if len(payload) < 4 + meta_len:
        raise InvalidContainerError("Payload metadata is corrupted or incomplete.")

    meta_bytes = payload[4 : 4 + meta_len]
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as e:
        raise InvalidContainerError(f"Corrupted metadata JSON: {str(e)}") from e

    raw_data = payload[4 + meta_len :]
    return raw_data, meta


def encrypt_file_or_data(
    data: bytes,
    cipher_name: str,
    key_mode: str = "random",  # "random" or "password"
    password: str | None = None,
    key: bytes | None = None,
    filename: str = "message.txt",
    mime_type: str = "text/plain",
) -> tuple[bytes, bytes, float]:
    """High-level encryption pipeline for CryptoSelect AI.

    Args:
        data: Plaintext bytes.
        cipher_name: Approved cipher ('AES-128-GCM', 'AES-256-GCM', 'ChaCha20-Poly1305').
        key_mode: 'random' or 'password'.
        password: User passphrase if key_mode == 'password'.
        key: Pre-existing key bytes if key_mode == 'random' (generated if None).
        filename: Original file name.
        mime_type: MIME type of original data.

    Returns:
        tuple of (container_bytes, active_key_bytes, elapsed_seconds)
    """
    t0 = time.perf_counter()
    validate_cipher_name(cipher_name)

    nonce = generate_nonce()
    salt: bytes | None = None
    mode_byte = KEY_MODE_RANDOM

    if key_mode == "password":
        if not password:
            raise ValueError("Password is required when key_mode is 'password'.")
        mode_byte = KEY_MODE_PASSWORD
        salt = generate_salt()
        active_key = derive_key_from_password(password, salt, cipher_name)
    else:
        mode_byte = KEY_MODE_RANDOM
        active_key = key if key is not None else generate_key(cipher_name)

    # 1. Build self-describing header
    header_bytes = serialize_container_header(
        cipher_name=cipher_name,
        key_mode=mode_byte,
        nonce=nonce,
        salt=salt,
    )

    # 2. Package metadata and plaintext
    payload = pack_payload(data, filename=filename, mime_type=mime_type)

    # 3. AEAD encrypt with header as Associated Data (AAD)
    ciphertext = encrypt_aead(
        cipher_name=cipher_name,
        key=active_key,
        nonce=nonce,
        plaintext=payload,
        associated_data=header_bytes,
    )

    elapsed = time.perf_counter() - t0
    container = header_bytes + ciphertext
    return container, active_key, elapsed


def decrypt_file_or_data(
    container_bytes: bytes,
    password: str | None = None,
    key: bytes | None = None,
) -> tuple[bytes, dict[str, Any], ContainerHeader, float]:
    """High-level decryption pipeline for CryptoSelect AI.

    Reads cipher and key mode directly from header (AI never used for decryption).
    Authenticates header via AEAD Associated Data (AAD).

    Returns:
        tuple of (raw_data, metadata_dict, container_header, elapsed_seconds)

    Raises:
        CryptoAuthenticationError: If wrong key/password or tampered file.
        InvalidContainerError: If container structure is invalid.
    """
    t0 = time.perf_counter()
    header, ciphertext = parse_container_header(container_bytes)

    if header.key_mode == KEY_MODE_PASSWORD:
        if not password:
            raise ValueError(
                "This file was encrypted with a password. Please provide the password."
            )
        if header.salt is None:
            raise InvalidContainerError("Missing salt in password-mode container.")
        active_key = derive_key_from_password(password, header.salt, header.cipher_name)
    else:
        if key is None:
            raise ValueError(
                f"This file was encrypted with a random key. Please provide the "
                f"{header.cipher_name} key."
            )
        active_key = key

    # Decrypt and verify AAD
    decrypted_payload = decrypt_aead(
        cipher_name=header.cipher_name,
        key=active_key,
        nonce=header.nonce,
        ciphertext=ciphertext,
        associated_data=header.header_bytes,
    )

    raw_data, meta = unpack_payload(decrypted_payload)
    elapsed = time.perf_counter() - t0
    return raw_data, meta, header, elapsed
