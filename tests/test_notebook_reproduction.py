"""The website must reproduce the research notebook's printed numbers (seed 42, 4000 scenarios)."""
import pandas as pd
import pytest

from core.dataset import generate_dataset, train_selector
from core.evaluate import evaluate_all, new_seed_tables


@pytest.fixture(scope="module")
def trained():
    df = generate_dataset(4000, 42)
    return df, train_selector(df, 42)


def test_label_distribution_matches_notebook(trained):
    df, _ = trained
    share = df["label"].value_counts(normalize=True).round(3)
    assert share["ChaCha20-Poly1305"] == 0.648 and share["AES-128-GCM"] == 0.193 and share["AES-256-GCM"] == 0.159


def test_crosstab_matches_notebook(trained):
    df, _ = trained
    ct = pd.crosstab([df["has_aes_hw"], df["required_security"]], df["label"])
    assert ct.loc[(False, 128), "ChaCha20-Poly1305"] == 726 and ct.loc[(False, 256), "ChaCha20-Poly1305"] == 717
    assert ct.loc[(True, 128), "AES-128-GCM"] == 773 and ct.loc[(True, 256), "AES-256-GCM"] == 636


def test_model_close_to_notebook(trained):
    _, b = trained
    assert abs(b["cv_macro_f1"] - 0.9766) < 0.01
    cls = evaluate_all(b)["classification"].set_index("Selector")
    assert abs(cls.loc["Random Forest", "Accuracy"] - 0.9812) < 0.02
    assert abs(cls.loc["Rule-based", "Accuracy"] - 0.7075) < 0.001      # no model involved: exact
    assert abs(cls.loc["Fixed ChaCha20", "Accuracy"] - 0.6475) < 0.001


def test_costs_close_to_notebook(trained):
    _, b = trained
    oc = evaluate_all(b)["outcome"].set_index("Selector")
    assert abs(oc.loc["Random Forest", "Mean cost"] - 1.0001) < 0.005
    assert abs(oc.loc["Rule-based", "Mean cost"] - 1.0358) < 0.001
    assert oc["% security violations"].eq(0).all()


def test_new_seed_has_zero_violations_with_filter(trained):
    _, b = trained
    table, by_net, _ = new_seed_tables(b, 777, 1000)
    assert table.set_index("Selector").loc["RF + security filter", "% security violations"] == 0
    assert list(by_net.index) == ["slow", "medium", "fast", "gigabit"]
