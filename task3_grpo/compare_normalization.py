"""Task 3 Step 3: canonical GRPO (1/T_k) vs Dr. GRPO-style (1/max_completion_length) normalization.

Two matched short continuations from the identical supplied midpoint: same seeded prompts, K, reward
model, beta, epsilon, learning rate, generation settings and `fork_updates`; only loss_type differs.
Each is evaluated with the common held-out protocol (task3_grpo.evaluate).

Length-conditioned statistics, from every in-loss completion logged during training
(results/task3_grpo/train/<run>/sequence_log.jsonl):
  * grad_norm: L2 norm of the gradient contributed by that single completion (its exact share of
    the batch loss, see continue_train);
  * token_weight: |A_k| / T_k (grpo) or |A_k| / L_max (dr_grpo), the weight each token receives;
  * per run: Spearman correlation of grad_norm and token_weight with completion length, and their
    means for short vs long completions (split at the median in-loss length of both runs pooled).
Outputs results/task3_grpo/normalization.csv and normalization_length_bins.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate


def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"])

    names = {}
    for lt in ("grpo", "dr_grpo"):
        name = f"fork_{lt}"
        names[lt] = name
        adapter = str(Path(cfg["output"]).parent / name)
        if args.force or not (repo_path(adapter) / "adapter_config.json").exists():
            run_grpo(args.config, output=adapter, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name)
        if args.force or not (res / "eval" / name / "summary.json").exists():
            evaluate(args.config, adapter, name, n_prompts=args.n_prompts)

    seqs = {lt: pd.DataFrame([json.loads(l) for l in (res / "train" / n / "sequence_log.jsonl").read_text().splitlines() if l.strip()])
            for lt, n in names.items()}
    pooled = pd.concat([d[d.in_loss] for d in seqs.values()])
    cut = float(pooled.length.median())

    rows, brows = [], []
    for lt, name in names.items():
        s = load_json(res / "eval" / name / "summary.json")
        ups = [json.loads(l) for l in (res / "train" / name / "update_log.jsonl").read_text().splitlines() if l.strip()]
        ts = load_json(res / "train" / name / "train_summary.json")
        d = seqs[lt][seqs[lt].in_loss]
        rows.append({
            "loss_type": lt, "run": name,
            "heldout_reward": s["reward_raw"]["mean"], "heldout_reward_se": s["reward_raw"]["se"],
            "heldout_kl_token": s["kl_token_mean"], "heldout_entropy": s["entropy_token_mean"],
            "heldout_len_mean": s["length_tokens"]["mean"], "heldout_len_std": s["length_tokens"]["std"],
            "heldout_truncated_frac": s["truncated_frac"],
            "train_reward_mean": float(np.mean([u["reward_mean"] for u in ups])),
            "train_len_mean": float(np.mean([u["response_len"] for u in ups])),
            "train_kl_last": ups[-1]["kl_token_mean"],
            "train_grad_norm_mean": float(np.mean([u["grad_norm"] for u in ups])),
            "generated_tokens": ts["generated_tokens_total"],
            "n_in_loss_completions": int(len(d)),
            "spearman_len_vs_seq_grad_norm": spearman(d.length, d.grad_norm),
            "spearman_len_vs_token_weight": spearman(d.length, d.token_weight),
            "length_split_tokens": cut,
        })
        for b, sub in (("short", d[d.length <= cut]), ("long", d[d.length > cut])):
            brows.append({"loss_type": lt, "length_bin": b, "n": int(len(sub)),
                          "mean_length": float(sub.length.mean()) if len(sub) else float("nan"),
                          "mean_seq_grad_norm": float(sub.grad_norm.mean()) if len(sub) else float("nan"),
                          "mean_token_weight": float(sub.token_weight.mean()) if len(sub) else float("nan"),
                          "mean_abs_advantage": float(sub.advantage.abs().mean()) if len(sub) else float("nan"),
                          "share_of_total_grad_norm": float(sub.grad_norm.sum() / max(d.grad_norm.sum(), 1e-12))})
    df, bd = pd.DataFrame(rows), pd.DataFrame(brows)
    df.to_csv(res / "normalization.csv", index=False)
    bd.to_csv(res / "normalization_length_bins.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print(bd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
