"""Selectors must respect the security filter; dataset generation must be reproducible."""
import numpy as np
import pytest

from core.config import CIPHERS, SECURITY_BITS
from core.dataset import FEATURE_COLUMNS, encode_features, generate_dataset, train_selector
from core.selectors import (
    Scenario, apply_priority, fixed_aes256, fixed_chacha, rf_plain, rf_proba, rf_secure, rule_based, security_filter,
    select_all,
)


@pytest.fixture(scope="module")
def trained():
    return train_selector(generate_dataset(1500, 11), 11)


def test_dataset_reproducible_same_seed():
    assert generate_dataset(300, 7).equals(generate_dataset(300, 7))


def test_dataset_differs_with_other_seed():
    assert not generate_dataset(300, 7)["payload_size"].equals(generate_dataset(300, 8)["payload_size"])


def test_labels_never_violate_security():
    df = generate_dataset(1500, 3)
    assert all(SECURITY_BITS[c] >= r for c, r in zip(df["label"], df["required_security"]))


def test_encoding_columns_are_fixed_even_for_one_row():
    one = Scenario("low", False, 50.0, 1000, 128, "slow").to_frame()
    assert list(encode_features(one).columns) == FEATURE_COLUMNS


def test_security_filter_function():
    assert security_filter("AES-128-GCM", 256) != "AES-128-GCM"
    assert security_filter("AES-128-GCM", 128) == "AES-128-GCM"


def test_all_filtered_selectors_respect_256(trained):
    df = generate_dataset(1500, 99)
    df["required_security"] = 256
    for name, picks in select_all(df, trained).items():
        if name == "Random Forest":      # the unfiltered model is allowed to be wrong; the filtered one is not
            continue
        assert all(SECURITY_BITS[c] >= 256 for c in picks), name


def test_probabilities_valid_and_aes128_zero_when_256(trained):
    df = generate_dataset(300, 4)
    df["required_security"] = 256
    p = rf_proba(trained, df)
    assert np.allclose(p.sum(axis=1), 1.0) and list(p.columns) == CIPHERS and (p["AES-128-GCM"] == 0).all()


def test_no_hw_rule_picks_chacha():
    df = Scenario("low", False, 50.0, 1_000_000, 128, "fast").to_frame()
    assert rule_based(df).iloc[0] == "ChaCha20-Poly1305"
    assert fixed_aes256(df).iloc[0] == "AES-256-GCM" and fixed_chacha(df).iloc[0] == "ChaCha20-Poly1305"


def test_priority_mapping():
    s = Scenario("mid", True, 90.0, 5000, 128, "fast")
    assert apply_priority(s, "Energy-saving").battery_pct < 20
    assert apply_priority(s, "Fastest").battery_pct >= 50
    assert apply_priority(s, "Max security").required_security == 256
    assert apply_priority(s, "Balanced") == s


def test_rf_secure_matches_rf_plain_when_128(trained):
    df = generate_dataset(200, 5)
    df["required_security"] = 128
    assert (rf_secure(trained, df) == rf_plain(trained, df)).all()
