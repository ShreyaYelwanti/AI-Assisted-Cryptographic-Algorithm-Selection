"""Plain-language explanations built from the cost model's own numbers (no canned claims)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import SECURITY_BITS
from core.crypto import CIPHER_CHACHA20_POLY1305
from core.dataset import Assumptions, estimate_scenario, energy_weight
from core.selectors import SELECTOR_NAMES, Scenario

FACTOR_GROUPS = {
    "Device CPU": ["cpu_class_high", "cpu_class_low", "cpu_class_mid"],
    "AES hardware": ["has_aes_hw"],
    "Battery level": ["battery_pct"],
    "Payload size": ["payload_size"],
    "Required security": ["required_security"],
    "Network speed": ["network_fast", "network_gigabit", "network_medium", "network_slow"],
}


def grouped_importances(bundle: dict[str, Any] | None) -> pd.DataFrame:
    """Feature importances summed per user-facing factor (all CPU columns together, and so on)."""
    imp = (bundle or {}).get("feature_importances") or {}
    if not imp:
        return pd.DataFrame(columns=["Factor", "Importance"])
    rows = [{"Factor": g, "Importance": float(sum(imp.get(c, 0.0) for c in cols))} for g, cols in FACTOR_GROUPS.items()]
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)


def explain_decision(s: Scenario, chosen: str, priority: str, confidence: float | None = None,
                     profile: dict[str, Any] | None = None, assumptions: Assumptions | None = None) -> list[str]:
    """Short sentences explaining WHY `chosen` fits. `s` is the scenario after the priority was applied."""
    est = estimate_scenario(s.cpu_class, s.has_aes_hw, s.battery_pct, s.payload_size, s.required_security,
                            s.network, profile, assumptions)
    allowed = est[est["allowed"]]
    others = allowed.drop(index=chosen, errors="ignore")
    w_e = energy_weight(s.battery_pct)
    out: list[str] = []

    if s.required_security == 256:
        out.append("256-bit security is required, so AES-128-GCM was ruled out before the AI decided.")
    if not s.has_aes_hw:
        if chosen == CIPHER_CHACHA20_POLY1305:
            out.append("The device has no AES hardware, so AES would run slowly in software. "
                       "ChaCha20 is designed to be fast without special hardware.")
        else:
            out.append("The device has no AES hardware, but other factors outweighed that here.")
    else:
        out.append("The device has AES hardware, which makes AES very fast.")
    if priority == "Fastest":
        out.append(f"Priority is Fastest: energy counts for {w_e:.0%} of the score and speed for {1 - w_e:.0%}.")
    elif priority == "Energy-saving":
        out.append(f"Priority is Energy-saving: energy counts for {w_e:.0%} of the score and speed for {1 - w_e:.0%}.")
    elif priority == "Max security":
        out.append("Priority is Max security: only 256-bit ciphers were considered.")
    elif s.battery_pct < 20:
        out.append(f"Battery is low ({s.battery_pct:.0f}%), so energy counts for {w_e:.0%} of the score.")
    else:
        out.append(f"Battery is {s.battery_pct:.0f}%, so energy counts for {w_e:.0%} of the score and speed for {1 - w_e:.0%}.")

    if not others.empty and chosen in est.index:
        runner = others["cost"].idxmin()
        c, r = est.loc[chosen], est.loc[runner]
        out.append(f"Estimated encryption work: {chosen} about {c.compute_ms:.3f} ms vs {runner} about {r.compute_ms:.3f} ms.")
        if c.total_ms > 0 and (c.total_ms - c.compute_ms) / c.total_ms > 0.95:
            out.append("On this network, sending the data takes far longer than encrypting it, "
                       "so the choice has only a small effect on total time.")
    if confidence is not None:
        out.append(f"The AI's confidence in this choice is {confidence:.0f}%.")
    return out


def comparison_table(s: Scenario, choices: dict[str, str]) -> pd.DataFrame:
    """What each selector would pick for this situation."""
    ai = choices.get("RF + security filter")
    rows = []
    for name in SELECTOR_NAMES:
        c = choices.get(name)
        if c is None:
            continue
        rows.append({"Selector": name, "Cipher": c, "Strength": f"{SECURITY_BITS[c]}-bit",
                     "Meets requirement": "Yes" if SECURITY_BITS[c] >= s.required_security else "No",
                     "Same as the AI": "-" if name == "RF + security filter" else ("Yes" if c == ai else "No")})
    return pd.DataFrame(rows)
