"""Evaluation tables, identical in method to the research notebook.

1. classification metrics + confusion matrix + feature importance on the 20% test split
2. outcome table: mean cost with 95% CI, % optimal, mean time, security violations, Wilcoxon tests
3. fresh scenarios from a different seed (with and without the security filter) and cost by network
4. sensitivity of the conclusions to the ChaCha20 and AES-256 assumptions
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from core.config import CIPHERS, NETWORKS
from core.dataset import Assumptions, encode_features, generate_dataset, split_dataset
from core.selectors import rf_plain, rf_secure, rule_based, select_all

BASELINES = ["Fixed AES-256", "Fixed ChaCha20", "Rule-based"]
CIPHER_COL = {c: i for i, c in enumerate(CIPHERS)}


def _rebuild(bundle: dict[str, Any]) -> pd.DataFrame:
    """Re-create the exact dataset the model was trained on (deterministic)."""
    return generate_dataset(bundle["n_rows"], bundle["seed"], bundle["profile"], Assumptions(**bundle["assumptions"]))


def test_split(bundle: dict[str, Any]) -> pd.DataFrame:
    """The held-out 20% rows, with their cost columns."""
    df = _rebuild(bundle)
    _, X_te, _, _ = split_dataset(df, bundle["seed"])
    return df.loc[X_te.index]


def _cost_of(df: pd.DataFrame, pred: pd.Series) -> np.ndarray:
    mat = df[[f"cost_{c}" for c in CIPHERS]].to_numpy()
    return mat[np.arange(len(df)), pred.map(CIPHER_COL).to_numpy()]


def _total_of(df: pd.DataFrame, pred: pd.Series) -> np.ndarray:
    mat = df[[f"total_{c}" for c in CIPHERS]].to_numpy()
    return mat[np.arange(len(df)), pred.map(CIPHER_COL).to_numpy()]


def _violations(df: pd.DataFrame, pred: pd.Series) -> float:
    return float(((pred == "AES-128-GCM") & (df["required_security"] == 256)).mean() * 100)


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:                      # all differences are zero
        return float("nan")


def classification_table(te: pd.DataFrame, preds: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        pr, rc, f1, _ = precision_recall_fscore_support(te["label"], p, average="macro", zero_division=0)
        rows.append([name, (p == te["label"]).mean(), pr, rc, f1])
    return pd.DataFrame(rows, columns=["Selector", "Accuracy", "Precision", "Recall", "F1"]).round(4)


def confusion(te: pd.DataFrame, pred: pd.Series) -> np.ndarray:
    return confusion_matrix(te["label"], pred, labels=CIPHERS)


def feature_importance(bundle: dict[str, Any]) -> pd.Series:
    return pd.Series(bundle["feature_importances"]).sort_values()


def outcome_table(te: pd.DataFrame, preds: dict[str, pd.Series], reference: str = "Random Forest") -> pd.DataFrame:
    """Mean cost with 95% CI, % optimal, mean time, % violations, Wilcoxon p vs the reference."""
    rows, cost_of = [], {}
    for name, p in preds.items():
        cost = _cost_of(te, p)
        cost_of[name] = cost
        ci = stats.t.interval(0.95, len(cost) - 1, loc=cost.mean(), scale=stats.sem(cost))
        rows.append([name, cost.mean(), ci[0], ci[1], (p == te["label"]).mean() * 100,
                     _total_of(te, p).mean() * 1000, _violations(te, p)])
    out = pd.DataFrame(rows, columns=["Selector", "Mean cost", "CI low", "CI high", "% optimal",
                                      "Mean time (ms)", "% security violations"]).round(4)
    out["Wilcoxon p (RF vs this)"] = out["Selector"].map(
        {n: _wilcoxon(cost_of[reference], cost_of[n]) for n in BASELINES if n in cost_of})
    ref_cost = out.loc[out["Selector"] == reference, "Mean cost"].values[0]
    out["Cost reduction by RF (%)"] = ((out["Mean cost"] - ref_cost) / out["Mean cost"] * 100).round(2)
    return out


def new_seed_tables(bundle: dict[str, Any], seed: int = 777, n: int = 4000
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fresh scenarios from a different seed. Returns (summary table, cost by network, the scenarios)."""
    new_df = generate_dataset(n, seed, bundle["profile"], Assumptions(**bundle["assumptions"]))
    preds = select_all(new_df, bundle)
    rows = []
    for name, p in preds.items():
        cst = _cost_of(new_df, p)
        pv = float("nan")
        if name not in ("Random Forest", "RF + security filter"):
            pv = _wilcoxon(_cost_of(new_df, preds["Random Forest"]), cst)
        rows.append([name, (p == new_df["label"]).mean() * 100,
                     precision_recall_fscore_support(new_df["label"], p, average="macro", zero_division=0)[2],
                     cst.mean(), np.median(cst), _violations(new_df, p), pv])
    table = pd.DataFrame(rows, columns=["Selector", "% optimal", "Macro-F1", "Mean cost", "Median cost",
                                        "% security violations", "Wilcoxon p (RF vs this)"]).round(4)
    by_net = pd.DataFrame({name: pd.Series(_cost_of(new_df, p), index=new_df.index)
                           .groupby(new_df["network"]).mean() for name, p in preds.items()}
                          ).loc[NETWORKS].round(4)
    return table, by_net, new_df


def _sensitivity_once(profile: dict[str, Any], chacha_scale: float, aes256: float,
                      seed: int = 101, n: int = 3000) -> dict[str, float]:
    a = Assumptions(aes256_factor=aes256, chacha_scale=chacha_scale)
    df = generate_dataset(n, seed, profile, a)
    counts = df["label"].value_counts()
    mix = {c: counts.get(c, 0) / len(df) * 100 for c in CIPHERS}
    Xs = encode_features(df)
    strat = df["label"] if (counts.min() >= 5 and len(counts) == 3) else None
    Xtr, Xte, ytr, yte = train_test_split(Xs, df["label"], test_size=0.2, stratify=strat, random_state=42)
    m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(Xtr, ytr)
    t = df.loc[Xte.index]
    cm_ = t[[f"cost_{c}" for c in CIPHERS]].to_numpy()

    def mc(p: Any) -> float:
        return cm_[np.arange(len(t)), pd.Series(p, index=t.index).map(CIPHER_COL).to_numpy()].mean()

    p_rf, p_rule = m.predict(Xte), rule_based(t).to_numpy()
    p_cha = np.array(["ChaCha20-Poly1305"] * len(t))
    y = t["label"].to_numpy()
    return dict(pct_AES128=mix["AES-128-GCM"], pct_AES256=mix["AES-256-GCM"], pct_ChaCha=mix["ChaCha20-Poly1305"],
                rf_optimal=(p_rf == y).mean() * 100, rule_optimal=(p_rule == y).mean() * 100,
                chacha_optimal=(p_cha == y).mean() * 100,
                rf_cost=mc(p_rf), rule_cost=mc(p_rule), chacha_cost=mc(p_cha))


def sensitivity_table(profile: dict[str, Any]) -> pd.DataFrame:
    """9 re-runs: ChaCha20 ratio x {0.8, 1.0, 1.2} and AES-256 factor in {1.05, 1.15, 1.25}."""
    rows = []
    for scale in (0.8, 1.0, 1.2):
        for a256 in (1.05, 1.15, 1.25):
            rows.append(dict(chacha_scale=scale, aes256_factor=a256, **_sensitivity_once(profile, scale, a256)))
    s = pd.DataFrame(rows)
    s["RF gain vs rule (%)"] = (s.rule_cost - s.rf_cost) / s.rule_cost * 100
    s["RF gain vs ChaCha (%)"] = (s.chacha_cost - s.rf_cost) / s.chacha_cost * 100
    s = s.round(3)
    return s[["chacha_scale", "aes256_factor", "pct_AES128", "pct_AES256", "pct_ChaCha", "rf_optimal",
              "rule_optimal", "chacha_optimal", "RF gain vs rule (%)", "RF gain vs ChaCha (%)"]]


def evaluate_all(bundle: dict[str, Any]) -> dict[str, Any]:
    """Everything for cells 11 and 12 of the notebook (the seed-42 held-out split)."""
    te = test_split(bundle)
    preds = {"Fixed AES-256": select_all(te, None)["Fixed AES-256"],
             "Fixed ChaCha20": select_all(te, None)["Fixed ChaCha20"],
             "Rule-based": rule_based(te), "Random Forest": rf_plain(bundle, te)}
    return {"te": te, "classification": classification_table(te, preds), "confusion": confusion(te, preds["Random Forest"]),
            "importance": feature_importance(bundle), "outcome": outcome_table(te, preds)}
