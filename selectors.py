"""The selectors compared in the project, and the security filter.

The AI only CHOOSES among the three approved ciphers. A security filter runs before the model
(AES-128 is removed when 256-bit security is required) and the same rule is the final safety net.
Names match the research notebook: Fixed AES-256, Fixed ChaCha20, Rule-based, Random Forest,
RF + security filter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from core.config import CIPHERS, SECURITY_BITS
from core.crypto import CIPHER_AES_128_GCM, CIPHER_AES_256_GCM, CIPHER_CHACHA20_POLY1305
from core.dataset import FEATURES, encode_features

SELECTOR_NAMES = ["Fixed AES-256", "Fixed ChaCha20", "Rule-based", "Random Forest", "RF + security filter"]

# How the priority choices map onto the cost model (battery only matters through the energy weight):
#   Fastest        -> energy weight 0.1   (represented by a battery value in the >= 50% band)
#   Energy-saving  -> energy weight 0.7   (represented by a battery value in the < 20% band)
#   Max security   -> 256-bit security required
#   Balanced       -> the energy weight follows the real battery level
PRIORITY_BATTERY = {"Fastest": 75.0, "Energy-saving": 10.0}


@dataclass(frozen=True)
class Scenario:
    """One situation: the six inputs of the model."""
    cpu_class: str
    has_aes_hw: bool
    battery_pct: float
    payload_size: int
    required_security: int
    network: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{c: getattr(self, c) for c in FEATURES}])


def apply_priority(s: Scenario, priority: str) -> Scenario:
    """Translate the user's priority into the inputs the cost model understands."""
    if priority in PRIORITY_BATTERY:
        return replace(s, battery_pct=PRIORITY_BATTERY[priority])
    if priority == "Max security":
        return replace(s, required_security=256)
    return s


def allowed_ciphers(required_security: int) -> list[str]:
    return [c for c in CIPHERS if SECURITY_BITS[c] >= required_security]


def security_filter(cipher: str, required_security: int, has_aes_hw: bool = True) -> str:
    """Return `cipher` if strong enough, otherwise the nearest allowed replacement."""
    if SECURITY_BITS[cipher] >= required_security:
        return cipher
    return CIPHER_AES_256_GCM if has_aes_hw else CIPHER_CHACHA20_POLY1305


def fixed_aes256(df: pd.DataFrame) -> pd.Series:
    return pd.Series(CIPHER_AES_256_GCM, index=df.index)


def fixed_chacha(df: pd.DataFrame) -> pd.Series:
    return pd.Series(CIPHER_CHACHA20_POLY1305, index=df.index)


def rule_based(df: pd.DataFrame) -> pd.Series:
    """Hand-written from general knowledge, NOT from the labels or the cost formula.

    No AES hardware -> ChaCha20; 256-bit needed -> AES-256; otherwise AES-128.
    """
    out = np.where(~df["has_aes_hw"].to_numpy(bool), CIPHER_CHACHA20_POLY1305,
                   np.where(df["required_security"].to_numpy(int) == 256, CIPHER_AES_256_GCM, CIPHER_AES_128_GCM))
    return pd.Series(out, index=df.index)


def rf_plain(bundle: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Random Forest prediction with no security filter."""
    X = encode_features(df).reindex(columns=bundle["columns"], fill_value=0.0)
    return pd.Series(bundle["model"].predict(X), index=df.index)


def rf_proba(bundle: dict[str, Any], df: pd.DataFrame, filtered: bool = True) -> pd.DataFrame:
    """Class probabilities by cipher. With filtered=True, disallowed ciphers get 0 and the rest are rescaled."""
    model = bundle["model"]
    X = encode_features(df).reindex(columns=bundle["columns"], fill_value=0.0)
    p = pd.DataFrame(model.predict_proba(X), columns=list(model.classes_), index=df.index)
    p = p.reindex(columns=CIPHERS, fill_value=0.0)
    if filtered:
        p.loc[df["required_security"] == 256, CIPHER_AES_128_GCM] = 0.0
        sums = p.sum(axis=1).replace(0, 1.0)
        p = p.div(sums, axis=0)
    return p


def rf_secure(bundle: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Random Forest with the hard security filter on top (this is what the app recommends)."""
    model = bundle["model"]
    X = encode_features(df).reindex(columns=bundle["columns"], fill_value=0.0)
    p = pd.DataFrame(model.predict_proba(X), columns=list(model.classes_), index=df.index)
    p.loc[df["required_security"] == 256, CIPHER_AES_128_GCM] = -1  # same as the notebook
    return p.idxmax(axis=1)


def select_all(df: pd.DataFrame, bundle: dict[str, Any] | None) -> dict[str, pd.Series]:
    """Every selector's choice for every row."""
    out = {"Fixed AES-256": fixed_aes256(df), "Fixed ChaCha20": fixed_chacha(df), "Rule-based": rule_based(df)}
    if bundle is not None:
        out["Random Forest"] = rf_plain(bundle, df)
        out["RF + security filter"] = rf_secure(bundle, df)
    return out
