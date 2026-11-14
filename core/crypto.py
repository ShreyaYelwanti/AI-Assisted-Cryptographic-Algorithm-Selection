"""Cryptographic operations module for CryptoSelect AI.

Follows strict AEAD security requirements:
- Only AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305.
- Uses exclusively the standard 'cryptography' library.
- Fresh 12-byte random nonce generated via os.urandom() for each encryption.
- Key derivation using Scrypt (n=16384, r=8, p=1, salt=16 bytes).
- Full Associated Data (AAD) support to bind header metadata and prevent tampering.
- Clean authentication failure exceptions.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Final, Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class CryptoError(Exception):
    """Base exception for all cryptographic errors in CryptoSelect AI."""
    pass


class CryptoAuthenticationError(CryptoError):
    """Raised when tag verification fails due to wrong key/password or tampered data."""
    pass


# Cipher definitions & IDs
CIPHER_AES_128_GCM: Final[str] = "AES-128-GCM"
CIPHER_AES_256_GCM: Final[str] = "AES-256-GCM"
CIPHER_CHACHA20_POLY1305: Final[str] = "ChaCha20-Poly1305"

CIPHER_ID_MAP: Final[dict[str, int]] = {
    CIPHER_AES_128_GCM: 0x01,
    CIPHER_AES_256_GCM: 0x02,
    CIPHER_CHACHA20_POLY1305: 0x03,
}

ID_TO_CIPHER_MAP: Final[dict[int, str]] = {v: k for k, v in CIPHER_ID_MAP.items()}

KEY_LENGTH_MAP: Final[dict[str, int]] = {
    CIPHER_AES_128_GCM: 16,  # 128 bits
    CIPHER_AES_256_GCM: 32,  # 256 bits
    CIPHER_CHACHA20_POLY1305: 32,  # 256 bits
}

SECURITY_LEVEL_MAP: Final[dict[str, int]] = {
    CIPHER_AES_128_GCM: 128,
    CIPHER_AES_256_GCM: 256,
    CIPHER_CHACHA20_POLY1305: 256,
}

NONCE_LENGTH_BYTES: Final[int] = 12
SALT_LENGTH_BYTES: Final[int] = 16
TAG_LENGTH_BYTES: Final[int] = 16
CRYPTO_OVERHEAD_BYTES: Final[int] = NONCE_LENGTH_BYTES + TAG_LENGTH_BYTES  # 28 bytes


def validate_cipher_name(cipher_name: str) -> None:
    """Validate that the cipher is one of the three approved AEAD ciphers."""
    if cipher_name not in CIPHER_ID_MAP:
        raise ValueError(
            f"Unsupported cipher '{cipher_name}'. "
            f"Allowed ciphers are: {list(CIPHER_ID_MAP.keys())}"
        )


def generate_key(cipher_name: str) -> bytes:
    """Generate a cryptographically secure random key for the specified cipher.

    Args:
        cipher_name: One of 'AES-128-GCM', 'AES-256-GCM', 'ChaCha20-Poly1305'.

    Returns:
        Raw bytes of the generated key (16 or 32 bytes).
    """
    validate_cipher_name(cipher_name)
    key_len = KEY_LENGTH_MAP[cipher_name]
    return os.urandom(key_len)


def generate_nonce() -> bytes:
    """Generate a fresh cryptographically secure 12-byte nonce."""
    return os.urandom(NONCE_LENGTH_BYTES)


def generate_salt() -> bytes:
    """Generate a fresh cryptographically secure 16-byte salt for Scrypt KDF."""
    return os.urandom(SALT_LENGTH_BYTES)


def derive_key_from_password(password: str, salt: bytes, cipher_name: str) -> bytes:
    """Derive a cryptographic key from a user password using Scrypt KDF.

    Uses RFC 7914 compliant parameters:
    - n = 16384 (CPU/memory cost parameter)
    - r = 8 (block size)
    - p = 1 (parallelization parameter)

    Args:
        password: User supplied passphrase.
        salt: 16-byte random salt.
        cipher_name: Target cipher (determines derived key length).

    Returns:
        Derived key bytes (16 or 32 bytes).
    """
    validate_cipher_name(cipher_name)
    if not password:
        raise ValueError("Password cannot be empty.")
    if len(salt) != SALT_LENGTH_BYTES:
        raise ValueError(f"Salt must be exactly {SALT_LENGTH_BYTES} bytes.")

    key_len = KEY_LENGTH_MAP[cipher_name]
    kdf = Scrypt(
        salt=salt,
        length=key_len,
        n=16384,
        r=8,
        p=1,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_aead(
    cipher_name: str,
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    associated_data: bytes = b"",
) -> bytes:
    """Encrypt plaintext using the chosen AEAD cipher and authenticate associated data.

    Args:
        cipher_name: Approved cipher name.
        key: Raw key bytes (length must match cipher).
        nonce: 12-byte unique nonce.
        plaintext: Raw data to encrypt.
        associated_data: Additional authenticated data (e.g. file header).

    Returns:
        Ciphertext with 16-byte AEAD authentication tag appended.

    Raises:
        ValueError: If key or nonce length is invalid.
        CryptoError: On cryptographic operation failure.
    """
    validate_cipher_name(cipher_name)
    expected_key_len = KEY_LENGTH_MAP[cipher_name]

    if len(key) != expected_key_len:
        raise ValueError(
            f"Invalid key length for {cipher_name}: expected {expected_key_len} bytes, "
            f"got {len(key)} bytes."
        )
    if len(nonce) != NONCE_LENGTH_BYTES:
        raise ValueError(
            f"Nonce must be exactly {NONCE_LENGTH_BYTES} bytes, got {len(nonce)} bytes."
        )

    try:
        if cipher_name in (CIPHER_AES_128_GCM, CIPHER_AES_256_GCM):
            aesgcm = AESGCM(key)
            return aesgcm.encrypt(nonce, plaintext, associated_data)
        elif cipher_name == CIPHER_CHACHA20_POLY1305:
            chacha = ChaCha20Poly1305(key)
            return chacha.encrypt(nonce, plaintext, associated_data)
        else:
            raise ValueError(f"Unknown cipher: {cipher_name}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise CryptoError(f"Encryption failed: {str(e)}") from e


def decrypt_aead(
    cipher_name: str,
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    associated_data: bytes = b"",
) -> bytes:
    """Decrypt ciphertext and verify the AEAD authentication tag against associated data.

    Args:
        cipher_name: Approved cipher name.
        key: Raw key bytes.
        nonce: 12-byte nonce.
        ciphertext: Ciphertext with 16-byte tag.
        associated_data: Additional authenticated data to verify.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        CryptoAuthenticationError: If tag verification fails (tampering/wrong key).
        ValueError: On parameter validation issues.
    """
    validate_cipher_name(cipher_name)
    expected_key_len = KEY_LENGTH_MAP[cipher_name]

    if len(key) != expected_key_len:
        raise ValueError(
            f"Invalid key length for {cipher_name}: expected {expected_key_len} bytes, "
            f"got {len(key)} bytes."
        )
    if len(nonce) != NONCE_LENGTH_BYTES:
        raise ValueError(
            f"Nonce must be exactly {NONCE_LENGTH_BYTES} bytes, got {len(nonce)} bytes."
        )

    try:
        if cipher_name in (CIPHER_AES_128_GCM, CIPHER_AES_256_GCM):
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, associated_data)
        elif cipher_name == CIPHER_CHACHA20_POLY1305:
            chacha = ChaCha20Poly1305(key)
            return chacha.decrypt(nonce, ciphertext, associated_data)
        else:
            raise ValueError(f"Unknown cipher: {cipher_name}")
    except InvalidTag as e:
        raise CryptoAuthenticationError(
            "Authentication failed: invalid key/password or corrupted/tampered file."
        ) from e
    except Exception as e:
        if isinstance(e, (CryptoAuthenticationError, ValueError)):
            raise
        raise CryptoAuthenticationError(f"Authentication failed: {str(e)}") from e


def key_to_base64(key: bytes) -> str:
    """Encode key to Base64 string for display and transport."""
    return base64.b64encode(key).decode("utf-8")


def key_from_base64(b64_key: str) -> bytes:
    """Decode key from Base64 string."""
    try:
        return base64.b64decode(b64_key.strip())
    except Exception as e:
        raise ValueError(f"Invalid Base64 key string: {str(e)}") from e


def key_to_hex(key: bytes) -> str:
    """Encode key to Hex string."""
    return key.hex()


def key_from_hex(hex_key: str) -> bytes:
    """Decode key from Hex string."""
    try:
        return bytes.fromhex(hex_key.strip())
    except Exception as e:
        raise ValueError(f"Invalid Hex key string: {str(e)}") from e


# --------------------------------------------------------------------------------------
# Thin helpers around the library objects (used by the benchmark; no custom cryptography)
# --------------------------------------------------------------------------------------
def make_cipher(cipher_name: str) -> AESGCM | ChaCha20Poly1305:
    """Library cipher object with a fresh random key (used for speed measurements)."""
    key = generate_key(cipher_name)
    return ChaCha20Poly1305(key) if cipher_name == CIPHER_CHACHA20_POLY1305 else AESGCM(key)


def raw_encrypt(cipher: AESGCM | ChaCha20Poly1305, data: bytes) -> bytes:
    """nonce (12 B, fresh random) + ciphertext + tag (16 B)."""
    nonce = os.urandom(NONCE_LENGTH_BYTES)
    return nonce + cipher.encrypt(nonce, data, None)


def raw_decrypt(cipher: AESGCM | ChaCha20Poly1305, blob: bytes) -> bytes:
    """Inverse of raw_encrypt; raises cryptography's InvalidTag when authentication fails."""
    return cipher.decrypt(blob[:NONCE_LENGTH_BYTES], blob[NONCE_LENGTH_BYTES:], None)
