"""Task 1 Step 2: matched short-run beta forks from the original initialization.

For each beta in configs/dpo.yaml:betas, train a fresh LoRA from Qwen2.5-1.5B-Instruct on the
first `short_ablation_examples` rows of the fixed standard training file (same rows, seed,
optimizer and LoRA config for all betas), then evaluate with task1_dpo.evaluate under the common
protocol. Already-finished stages are skipped, so the script can be resumed after a Colab
disconnect. Writes results/task1_dpo/beta_summary.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training


def flatten(summary, train_summary, label, budget):
    p = summary["preference"]["overall"]
    g = summary.get("generation", {})
    row = {
        "condition": label, "budget": budget, "beta": summary["beta"],
        "train_pairs": train_summary.get("n_train_pairs") if train_summary else 0,
        "optimizer_steps": train_summary.get("optimizer_steps") if train_summary else 0,
        "heldout_dpo_loss": p["dpo_loss"], "heldout_pref_acc": p["preference_accuracy"],
        "heldout_margin_mean": p["margin_mean"],
        "implicit_reward_chosen": p["implicit_reward_chosen"],
        "implicit_reward_rejected": p["implicit_reward_rejected"],
    }
    if g:
        row.update({
            "kl_token_mean": g["kl_token_mean_heldout"], "kl_seq_sum_mean": g["kl_seq_sum_mean_heldout"],
            "rm_score_mean": g["reward_heldout"]["mean"], "rm_score_std": g["reward_heldout"]["std"],
            "len_mean": g["length_tokens_heldout"]["mean"], "len_std": g["length_tokens_heldout"]["std"],
            "len_iqr": g["length_tokens_heldout"]["iqr"], "truncated_frac": g["truncated_frac_heldout"],
            "word_limit_compliance": g["word_limit_compliance"],
        })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="*", help="subset of betas to run (default: all)")
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if outputs exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    res = cfg["results_dir"]

    for b in betas:
        name = f"beta_{b:g}"
        adapter = str(Path(cfg["standard_output"]).parent / name)
        if args.force or not (repo_path(adapter) / "adapter_config.json").exists():
            run_training(args.config, run_name=name, output_path=adapter, beta=b, max_examples=n)
        if args.force or not (repo_path(res) / "eval" / name / "summary.json").exists():
            evaluate(args.config, adapter, name, beta=b, n_gen=args.n_gen)

    # Collect every condition that exists (SFT, standard, and all beta forks) into one table.
    rows = []
    for name, label, budget in (
        [("sft", "SFT (no adapter)", "none"), ("standard", "standard (1 epoch)", "full")]
        + [(f"beta_{float(b):g}", f"beta={float(b):g} (short)", f"first {n} rows") for b in cfg["betas"]]
    ):
        sp = repo_path(res) / "eval" / name / "summary.json"
        if not sp.exists():
            continue
        tp = repo_path(res) / "train" / name / "train_summary.json"
        rows.append(flatten(load_json(sp), load_json(tp) if tp.exists() else None, label, budget))
    df = pd.DataFrame(rows)
    out = repo_path(res) / "beta_summary.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
