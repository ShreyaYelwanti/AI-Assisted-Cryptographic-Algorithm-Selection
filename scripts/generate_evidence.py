"""Optional, command-line only: regenerate every table and figure for the research report.

    python scripts/generate_evidence.py            # full run (30 / 100 / 30 benchmark runs)
    python scripts/generate_evidence.py --quick    # fast smoke test

File names match the research notebook (fig1_throughput.png, table_outcome.csv, ...).
By default the scenarios are built from the notebook's own measurements, so the numbers match the
notebook. Use --profile local to build them from the speed test this script has just run.
The website never shows or exports any of this; delete this folder if you do not need it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core.benchmark import get_system_specs, run_all, save_results  # noqa: E402
from core.dataset import generate_dataset, notebook_profile, save_model, train_selector  # noqa: E402
from core.evaluate import evaluate_all, new_seed_tables, sensitivity_table  # noqa: E402
from core.figures import fig_confusion, fig_cost_by_network, fig_importance, fig_throughput  # noqa: E402
from core.security_tests import run_all_security_tests  # noqa: E402
from core.dataset import local_profile  # noqa: E402

TRAIN_SEED, NEW_SEED = 42, 777


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--profile", choices=["notebook", "local"], default="notebook")
    ap.add_argument("--no-save-model", action="store_true", help="do not overwrite models/selector.joblib")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, name: str) -> None:
        fig.savefig(out / name, dpi=200)

    print("1/5 Speed tests...")
    bench = run_all(quick=args.quick)
    save_results(bench, out)
    save_fig(fig_throughput(bench["micro_summary"]), "fig1_throughput.png")
    bench["transfer_summary"].to_csv(out / "e2e_summary.csv", index=False)

    print("2/5 Dataset and training...")
    profile = (local_profile(out) if args.profile == "local" else None) or notebook_profile()
    df = generate_dataset(4000, TRAIN_SEED, profile)
    df.to_csv(out / "device_dataset.csv", index=False)
    bundle = train_selector(df, TRAIN_SEED, profile, compare_models=True)
    if not args.no_save_model:
        save_model(bundle)
    bundle["comparison"].to_csv(out / "table_model_comparison.csv", index=False)

    print("3/5 Evaluation...")
    res = evaluate_all(bundle)
    res["classification"].to_csv(out / "table_classification_metrics.csv", index=False)
    res["outcome"].to_csv(out / "table_outcome.csv", index=False)
    res["importance"].sort_values(ascending=False).to_csv(out / "table_feature_importance.csv", header=["importance"])
    save_fig(fig_confusion(res["confusion"]), "fig2_confusion_matrix.png")
    save_fig(fig_importance(res["importance"]), "fig3_feature_importance.png")
    new_table, by_net, _ = new_seed_tables(bundle, NEW_SEED, 4000)
    new_table.to_csv(out / "table_new_seed.csv", index=False)
    by_net.to_csv(out / "table_cost_by_network.csv")
    save_fig(fig_cost_by_network(by_net), "fig4_cost_by_network.png")

    print("4/5 Security tests...")
    sec, rate = run_all_security_tests(bundle)
    sec.to_csv(out / "security_tests.csv", index=False)

    print("5/5 Sensitivity analysis...")
    sens = sensitivity_table(profile)
    sens.to_csv(out / "table_sensitivity.csv", index=False)

    lines = [
        "# Results summary", "",
        f"Scenario speed data: {profile['source']}. Train seed {TRAIN_SEED}; new-scenario seed {NEW_SEED}.",
        f"Best Random Forest settings: {bundle['best_params']}; cross-validated macro-F1 {bundle['cv_macro_f1']:.4f}.", "",
        "## Classification metrics (held-out 20%)", "", md_table(res["classification"]), "",
        "## Outcome (cost, time, security)", "", md_table(res["outcome"]), "",
        "## New scenarios (different seed)", "", md_table(new_table), "",
        "## Mean cost by network", "", md_table(by_net, index=True), "",
        "## Sensitivity to assumptions", "", md_table(sens), "",
        "## Security tests", "", f"Pass rate: {rate:.1f}%", "",
        "## Experimental setup", "", *[f"- {k}: {v}" for k, v in get_system_specs().items()], "",
    ]
    (out / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
