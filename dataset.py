"""Simulated device scenarios, the cost model that labels them, and the Random Forest selector.

This follows the research notebook exactly, so the website reproduces its numbers.

How a scenario gets its label
-----------------------------
1. Base AES compute time (encrypt + decrypt) per payload size comes from the speed test
   (medians). ChaCha20 time = AES time x an ASSUMED ratio (0.8 at <= 2 KB rising to 1.25
   at >= 64 KB); AES-256 = AES-128 time x an ASSUMED factor (1.15).
2. Device factors scale it: slower CPU, and AES without AES hardware is slower (x4).
3. 3% random (log-normal) noise is added to every cipher's compute time.
4. total time = compute + network delay (size x 8 / bandwidth); energy = compute x CPU watts.
5. cost = w_time * total/total_min + w_energy * energy/energy_min over the ALLOWED ciphers
   (AES-128 is not allowed when 256-bit security is required). Energy weight depends on battery:
   0.7 below 20%, 0.4 below 50%, otherwise 0.1.
6. label = the allowed cipher with the lowest cost (1.0 means "as good as the best possible").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.config import CIPHERS, MODEL_PATH, MODELS_DIR, RESULTS_DIR
from core.crypto import CIPHER_AES_128_GCM, CIPHER_AES_256_GCM, CIPHER_CHACHA20_POLY1305

# ---------------------------------------------------------------------------------------
# Documented assumptions (all of these belong in the paper's method section)
# ---------------------------------------------------------------------------------------
CPU_SLOW = {"low": 8.0, "mid": 3.0, "high": 1.0}        # slowdown vs. the measuring laptop
POWER_W = {"low": 0.5, "mid": 1.5, "high": 4.0}         # CPU power draw (watts)
HW_PROB = {"low": 0.30, "mid": 0.70, "high": 0.95}      # chance a device has AES hardware
CPU_PROBS = [0.30, 0.40, 0.30]                          # low, mid, high
NET_MBPS = {"slow": 1, "medium": 10, "fast": 100, "gigabit": 1000}
MIN_PAYLOAD, MAX_PAYLOAD = 100, 10_485_760
FEATURES = ["cpu_class", "has_aes_hw", "battery_pct", "payload_size", "required_security", "network"]
# Column order produced by pd.get_dummies in the notebook (kept fixed so training and prediction agree).
FEATURE_COLUMNS = ["has_aes_hw", "battery_pct", "payload_size", "required_security",
                   "cpu_class_high", "cpu_class_low", "cpu_class_mid",
                   "network_fast", "network_gigabit", "network_medium", "network_slow"]


@dataclass(frozen=True)
class Assumptions:
    """Simulation assumptions. The sensitivity analysis varies chacha_scale and aes256_factor."""
    aes_sw_penalty: float = 4.0     # AES slowdown when the device has no AES hardware
    aes256_factor: float = 1.15     # AES-256 time vs AES-128 (measured was about 1.04)
    chacha_scale: float = 1.0       # multiplies the ChaCha20/AES ratio below
    noise_sigma: float = 0.03       # log-normal noise on every compute time


def energy_weight(battery_pct: float) -> float:
    """Weight on energy in the cost; the rest goes to time."""
    return 0.7 if battery_pct < 20 else 0.4 if battery_pct < 50 else 0.1


def chacha_ratio(size: float, scale: float = 1.0) -> float:
    """Assumed ChaCha20/AES time ratio: 0.8 up to 2 KB, rising to 1.25 from 64 KB (log-linear between)."""
    return scale * float(np.interp(np.log(size), np.log([2048, 65536]), [0.8, 1.25]))


# ---------------------------------------------------------------------------------------
# Speed profile (measured AES compute time vs payload size)
# ---------------------------------------------------------------------------------------
# Median encrypt/decrypt microseconds (100 runs, 5 warm-ups) from the research notebook.
# Machine: Windows 11, AMD Ryzen (family 25), Python 3.13.5, cryptography 50.0.2.
NOTEBOOK_MEDIANS = pd.DataFrame(
    [("AES-128-GCM", "1KB", 1024, 4.85, 4.40), ("AES-128-GCM", "10KB", 10240, 6.05, 5.60),
     ("AES-128-GCM", "100KB", 102400, 39.50, 39.20), ("AES-128-GCM", "1MB", 1048576, 1292.30, 926.90),
     ("AES-128-GCM", "10MB", 10485760, 8502.45, 8296.30),
     ("AES-256-GCM", "1KB", 1024, 2.40, 2.20), ("AES-256-GCM", "10KB", 10240, 6.00, 5.70),
     ("AES-256-GCM", "100KB", 102400, 41.55, 41.15), ("AES-256-GCM", "1MB", 1048576, 1319.55, 959.75),
     ("AES-256-GCM", "10MB", 10485760, 8270.45, 7829.40),
     ("ChaCha20-Poly1305", "1KB", 1024, 2.10, 2.00), ("ChaCha20-Poly1305", "10KB", 10240, 6.50, 6.40),
     ("ChaCha20-Poly1305", "100KB", 102400, 50.50, 50.50), ("ChaCha20-Poly1305", "1MB", 1048576, 1226.20, 966.20),
     ("ChaCha20-Poly1305", "10MB", 10485760, 8966.80, 8945.10)],
    columns=["cipher", "size_label", "size", "enc_median_us", "dec_median_us"])

NOTEBOOK_SOURCE = "Measurements from the research notebook (100 runs per setting)"


def profile_from_medians(summary: pd.DataFrame, source: str) -> dict[str, Any]:
    """AES compute time (s) per size = mean of AES-128 and AES-256 (encrypt + decrypt medians)."""
    s = summary.copy()
    s["comp_s"] = (s["enc_median_us"] + s["dec_median_us"]) / 1e6
    aes = s[s["cipher"].str.startswith("AES")].groupby("size")["comp_s"].mean().sort_index()
    if len(aes) < 2:
        raise ValueError("Speed test needs at least two payload sizes.")
    return {"sizes": aes.index.to_numpy(float).tolist(), "aes_s": aes.to_numpy(float).tolist(), "source": source}


def notebook_profile() -> dict[str, Any]:
    return profile_from_medians(NOTEBOOK_MEDIANS, NOTEBOOK_SOURCE)


def local_profile(results_dir: Path | str | None = None) -> dict[str, Any] | None:
    """Profile from this computer's latest speed test, or None if it has not been run."""
    path = Path(results_dir or RESULTS_DIR) / "benchmark_summary_v2.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return profile_from_medians(df, "Measured on this computer (latest speed test)")
    except Exception:
        return None


def aes_time(profile: dict[str, Any], size: float) -> float:
    return float(np.exp(np.interp(np.log(size), np.log(profile["sizes"]), np.log(profile["aes_s"]))))


def base_time(profile: dict[str, Any], a: Assumptions, cipher: str, size: float) -> float:
    t = aes_time(profile, size)
    return t * chacha_ratio(size, a.chacha_scale) if cipher == CIPHER_CHACHA20_POLY1305 else t


def allowed_ciphers_for(required_security: int) -> list[str]:
    """AES-128 is not allowed when 256-bit security is required."""
    return [c for c in CIPHERS if not (c == CIPHER_AES_128_GCM and required_security == 256)]


def network_delay_s(size_bytes: float, network: str) -> float:
    """Modelled transfer time: (size + 28 B overhead) x 8 / bandwidth."""
    return (size_bytes + 28) * 8 / (NET_MBPS[network] * 1e6)


def score_scenario(cpu: str, hw: bool, battery: float, size: int, sec: int, net: str,
                   profile: dict[str, Any], a: Assumptions, noise: list[float] | None = None) -> dict[str, Any]:
    """Compute time, total time, energy and cost for every cipher, plus the best allowed cipher."""
    comp: dict[str, float] = {}
    for i, c in enumerate(CIPHERS):
        t = base_time(profile, a, c, size) * CPU_SLOW[cpu]
        if c.startswith("AES"):
            if c == CIPHER_AES_256_GCM:
                t *= a.aes256_factor
            if not hw:
                t *= a.aes_sw_penalty
        comp[c] = t * (noise[i] if noise else 1.0)
    delay = network_delay_s(size, net)
    allowed = allowed_ciphers_for(sec)
    total = {c: comp[c] + delay for c in CIPHERS}
    energy = {c: comp[c] * POWER_W[cpu] for c in CIPHERS}
    w_e = energy_weight(battery)
    tmin = min(total[c] for c in allowed)
    emin = min(energy[c] for c in allowed)
    cost = {c: (1 - w_e) * total[c] / tmin + w_e * energy[c] / emin for c in CIPHERS}
    return {"comp": comp, "total": total, "energy": energy, "cost": cost, "allowed": allowed,
            "w_energy": w_e, "label": min(allowed, key=lambda c: cost[c])}


def generate_dataset(n: int = 4000, seed: int = 42, profile: dict[str, Any] | None = None,
                     assumptions: Assumptions | None = None) -> pd.DataFrame:
    """Reproducible scenarios. Same seed and profile give exactly the notebook's dataset."""
    profile = profile or notebook_profile()
    a = assumptions or Assumptions()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        cpu = rng.choice(["low", "mid", "high"], p=CPU_PROBS)
        hw = bool(rng.random() < HW_PROB[cpu])
        batt = rng.uniform(5, 100)
        size = int(np.exp(rng.uniform(np.log(MIN_PAYLOAD), np.log(MAX_PAYLOAD))))
        sec = int(rng.choice([128, 256]))
        net = str(rng.choice(list(NET_MBPS)))
        noise = [rng.lognormal(0, a.noise_sigma) for _ in CIPHERS]
        r = score_scenario(str(cpu), hw, batt, size, sec, net, profile, a, noise)
        rows.append(dict(cpu_class=cpu, has_aes_hw=hw, battery_pct=round(batt, 1), payload_size=size,
                         required_security=sec, network=net,
                         **{f"cost_{c}": r["cost"][c] for c in CIPHERS},
                         **{f"total_{c}": r["total"][c] for c in CIPHERS},
                         label=r["label"]))
    return pd.DataFrame(rows)


def estimate_scenario(cpu: str, hw: bool, battery: float, size: int, sec: int, net: str,
                      profile: dict[str, Any] | None = None, assumptions: Assumptions | None = None) -> pd.DataFrame:
    """Noise-free per-cipher estimates for ONE situation (used to explain a recommendation)."""
    r = score_scenario(cpu, hw, battery, size, sec, net, profile or notebook_profile(), assumptions or Assumptions())
    return pd.DataFrame({
        "cipher": CIPHERS,
        "compute_ms": [r["comp"][c] * 1000 for c in CIPHERS],
        "total_ms": [r["total"][c] * 1000 for c in CIPHERS],
        "energy": [r["energy"][c] for c in CIPHERS],
        "cost": [r["cost"][c] for c in CIPHERS],
        "allowed": [c in r["allowed"] for c in CIPHERS],
    }).set_index("cipher")


# ---------------------------------------------------------------------------------------
# Features and training
# ---------------------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode CPU class and network exactly like the notebook, in a fixed column order."""
    x = pd.get_dummies(df[FEATURES], columns=["cpu_class", "network"]).astype(float)
    return x.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)


def split_dataset(df: pd.DataFrame, seed: int = 42):
    """80/20 stratified split (the same one the notebook uses)."""
    X, y = encode_features(df), df["label"]
    strat = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=0.2, stratify=strat, random_state=seed)


EXTRA_MODELS: dict[str, tuple[Any, dict[str, list[Any]]]] = {
    "Decision Tree": (lambda s: DecisionTreeClassifier(random_state=s), {"max_depth": [4, 6, 8, 12]}),
    "Logistic Regression": (lambda s: Pipeline([("scale", StandardScaler()),
                                                ("clf", LogisticRegression(max_iter=2000))]),
                            {"clf__C": [0.1, 1.0, 10.0]}),
}


def train_selector(df: pd.DataFrame, seed: int = 42, profile: dict[str, Any] | None = None,
                   assumptions: Assumptions | None = None, compare_models: bool = False) -> dict[str, Any]:
    """Random Forest tuned by 5-fold CV on the TRAINING split only, as in the notebook."""
    X_tr, X_te, y_tr, y_te = split_dataset(df, seed)
    grid = GridSearchCV(RandomForestClassifier(random_state=seed),
                        {"n_estimators": [100, 300], "max_depth": [5, 10, None]},
                        cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    rf = grid.best_estimator_
    rows = []
    if compare_models:
        pred = rf.predict(X_te)
        rows.append({"Model": "Random Forest (used by the app)", "Best settings": str(grid.best_params_),
                     "CV macro-F1 (train)": round(grid.best_score_, 4),
                     "Test accuracy": round(accuracy_score(y_te, pred), 4),
                     "Test macro-F1": round(f1_score(y_te, pred, average="macro"), 4)})
        for name, (make, params) in EXTRA_MODELS.items():
            g = GridSearchCV(make(seed), params, cv=5, scoring="f1_macro", n_jobs=-1).fit(X_tr, y_tr)
            p = g.best_estimator_.predict(X_te)
            rows.append({"Model": name, "Best settings": str(g.best_params_),
                         "CV macro-F1 (train)": round(g.best_score_, 4),
                         "Test accuracy": round(accuracy_score(y_te, p), 4),
                         "Test macro-F1": round(f1_score(y_te, p, average="macro"), 4)})
    profile = profile or notebook_profile()
    assumptions = assumptions or Assumptions()
    return {"model": rf, "columns": list(X_tr.columns), "best_params": grid.best_params_,
            "cv_macro_f1": float(grid.best_score_), "n_train": len(X_tr), "n_test": len(X_te),
            "seed": seed, "n_rows": len(df), "profile": profile, "assumptions": asdict(assumptions),
            "feature_importances": dict(zip(X_tr.columns, rf.feature_importances_)),
            "label_counts": df["label"].value_counts().to_dict(),
            "comparison": pd.DataFrame(rows)}


def save_model(bundle: dict[str, Any], path: Path | str = MODEL_PATH) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return Path(path)


def load_model(path: Path | str = MODEL_PATH) -> dict[str, Any] | None:
    """None if missing or unreadable (for example saved by a different scikit-learn version)."""
    try:
        return joblib.load(path) if Path(path).exists() else None
    except Exception:
        return None
