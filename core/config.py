"""Shared constants: paths, the three approved ciphers, and user-facing option lists."""
from __future__ import annotations

from pathlib import Path

from core.crypto import (
    CIPHER_AES_128_GCM,
    CIPHER_AES_256_GCM,
    CIPHER_CHACHA20_POLY1305,
    CRYPTO_OVERHEAD_BYTES,
    SECURITY_LEVEL_MAP,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "selector.joblib"

# The only ciphers this project will ever use (order matters: it is the column order everywhere).
CIPHERS: list[str] = [CIPHER_AES_128_GCM, CIPHER_AES_256_GCM, CIPHER_CHACHA20_POLY1305]
SECURITY_BITS: dict[str, int] = SECURITY_LEVEL_MAP
OVERHEAD_BYTES = CRYPTO_OVERHEAD_BYTES  # 12-byte nonce + 16-byte tag = 28

CPU_CLASSES = ["low", "mid", "high"]
NETWORKS = ["slow", "medium", "fast", "gigabit"]
PRIORITIES = ["Balanced", "Fastest", "Energy-saving", "Max security"]

# label, bandwidth in Mbit/s. Network time is MODELLED from bandwidth, not measured.
NETWORK_INFO: dict[str, tuple[str, float]] = {
    "slow": ("Slow (1 Mbit/s)", 1.0),
    "medium": ("Medium (10 Mbit/s)", 10.0),
    "fast": ("Fast (100 Mbit/s)", 100.0),
    "gigabit": ("Gigabit (1000 Mbit/s)", 1000.0),
}

CPU_LABELS = {"low": "Low-power (sensor, old phone)", "mid": "Mid-range (typical phone/laptop)",
              "high": "High-performance (desktop, server)"}

CIPHER_BLURB = {
    CIPHER_AES_128_GCM: "Very fast on devices with AES hardware. 128-bit strength.",
    CIPHER_AES_256_GCM: "Maximum strength (256-bit). Very fast on devices with AES hardware.",
    CIPHER_CHACHA20_POLY1305: "Fast in plain software, so ideal for devices without AES hardware. 256-bit strength.",
}
