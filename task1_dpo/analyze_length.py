"""Task 1 Step 3: length-confounding study.

1. Dataset diagnostics (no model needed): length structure of the standard vs length-balanced
   training pairs (is the preferred response systematically longer?).
2. Train one DPO model from the original initialization on the supplied length-balanced subset
   (same config/budget as the standard run: one epoch, beta from configs/dpo.yaml).
3. Evaluate SFT, standard DPO and length-balanced DPO on the supplied length-stratified held-out
   set (per-stratum accuracy/loss/margin), and compare generated length + word-limit compliance on
   the common prompt set used in task1_dpo.evaluate.

Outputs: results/task1_dpo/length_dataset_stats.json, length_strata.csv, length_generation.csv
Resumable: finished training/evaluation stages are skipped unless --force.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import load_tokenizer
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training


def stratum(c, r):
    """Approximation of the course stratification rule, inferred from the supplied files
    (matched: chosen/rejected token ratio in [0.9, 1/0.9]; longer: ratio >= 1.2 and >= 24 tokens)."""
    ratio, diff = c / max(r, 1), c - r
    if 0.9 <= ratio <= 1 / 0.9:
        return "length_matched"
    if ratio >= 1.2 and diff >= 24:
        return "preferred_longer"
    if ratio <= 1 / 1.2 and diff <= -24:
        return "rejected_longer"
    return "intermediate"


def dataset_stats(path, tok):
    rows = read_jsonl(path)
    c_len, r_len = [], []
    for row in rows:
        c, r = preference_responses(row)
        c_len.append(len(tok(c, add_special_tokens=False)["input_ids"]))
        r_len.append(len(tok(r, add_special_tokens=False)["input_ids"]))
    c_len, r_len = np.array(c_len), np.array(r_len)
    diff = c_len - r_len
    strata = pd.Series([stratum(a, b) for a, b in zip(c_len, r_len)]).value_counts(normalize=True)
    return {
        "file": str(path), "n_pairs": int(len(rows)),
        "chosen_tokens_mean": float(c_len.mean()), "rejected_tokens_mean": float(r_len.mean()),
        "chosen_tokens_median": float(np.median(c_len)), "rejected_tokens_median": float(np.median(r_len)),
        "frac_chosen_longer": float((diff > 0).mean()),
        "mean_length_difference": float(diff.mean()), "median_length_difference": float(np.median(diff)),
        "strata_fraction_inferred_rule": {k: float(v) for k, v in strata.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-dataset-stats", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    strat_file = cfg["paths"]["dpo_length_eval"]

    # 1. dataset diagnostics
    if not args.skip_dataset_stats:
        tok = load_tokenizer(cfg["base_model"])
        stats = {k: dataset_stats(cfg["paths"][k], tok)
                 for k in ["dpo_standard_train", "dpo_length_train", "dpo_standard_eval", "dpo_length_eval"]}
        save_json(res / "length_dataset_stats.json", stats)
        for k, v in stats.items():
            print(f"{k:22s} n={v['n_pairs']:5d} chosen-longer={v['frac_chosen_longer']:.3f} "
                  f"mean diff={v['mean_length_difference']:+.1f} tokens")

    # 2. length-balanced training run (same budget as standard)
    lb_adapter = cfg["length_output"]
    if args.force or not (repo_path(lb_adapter) / "adapter_config.json").exists():
        run_training(args.config, run_name="length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=lb_adapter)

    # 3. evaluations
    jobs = [
        # (eval name, adapter, pref file, with generation?)
        ("sft_strat", "none", strat_file, False),
        ("standard_strat", cfg["standard_output"], strat_file, False),
        ("length_balanced_strat", lb_adapter, strat_file, False),
        ("length_balanced", lb_adapter, None, True),  # common generation prompts + standard held-out pairs
    ]
    for name, adapter, pref, gen in jobs:
        if args.force or not (res / "eval" / name / "summary.json").exists():
            evaluate(args.config, adapter, name, pref_path=pref, n_gen=args.n_gen, skip_generation=not gen)

    rows = []
    for name, label in [("sft_strat", "SFT"), ("standard_strat", "standard DPO"), ("length_balanced_strat", "length-balanced DPO")]:
        s = load_json(res / "eval" / name / "summary.json")["preference"]
        for st, v in list(s.get("by_stratum", {}).items()) + [("ALL", s["overall"])]:
            rows.append({"model": label, "stratum": st, **v})
    strata_df = pd.DataFrame(rows)
    strata_df.to_csv(res / "length_strata.csv", index=False)

    gen_rows = []
    for name, label in [("sft", "SFT"), ("standard", "standard DPO"), ("length_balanced", "length-balanced DPO")]:
        p = res / "eval" / name / "summary.json"
        if not p.exists():
            print(f"(missing {p}; run task1_dpo.evaluate for '{name}' first)")
            continue
        g = load_json(p)["generation"]
        gen_rows.append({
            "model": label,
            "heldout_len_mean": g["length_tokens_heldout"]["mean"], "heldout_len_std": g["length_tokens_heldout"]["std"],
            "heldout_len_iqr": g["length_tokens_heldout"]["iqr"], "truncated_frac": g["truncated_frac_heldout"],
            "rm_score_mean": g["reward_heldout"]["mean"], "kl_token_mean": g["kl_token_mean_heldout"],
            "word_limit_compliance": g["word_limit_compliance"],
            "word_limit_words_mean": g["word_limit_words"]["mean"],
        })
    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(res / "length_generation.csv", index=False)
    pd.set_option("display.width", 250)
    print(strata_df[["model", "stratum", "n", "preference_accuracy", "dpo_loss", "margin_mean", "ref_logp_prefers_chosen"]].to_string(index=False))
    print(gen_df.to_string(index=False))


if __name__ == "__main__":
    main()
