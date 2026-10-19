"""Supplementary multi-epoch DPO: one table, one row per evaluated epoch (plus the SFT baseline).

Reads results/task1_dpo_long/eval/epoch_<k>/ (written by task1_dpo.evaluate) and the training log,
and writes results/task1_dpo_long/epoch_summary.csv. No GPU needed.

Columns
  train_*            mean over that epoch's optimizer steps (training pairs; memorisation shows here)
  heldout_*          the 290 held-out preference pairs (generalisation shows here)
  frac_both_down     share of held-out pairs where BOTH chosen and rejected log-probs fell below the
                     reference (the 'push both down, rejected faster' DPO pattern)
  chosen_logp_delta  mean (policy - reference) log-prob of chosen responses; same for rejected
  kl / reward / len  generation metrics on the held-out prompts (RM score, tokens, 256-token cap)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path


def row_from_eval(name, d):
    s = json.loads((d / "summary.json").read_text())
    p = pd.read_csv(d / "preference.csv")
    o = s["preference"]["overall"]
    r = {"model": name, "heldout_pref_acc": o["preference_accuracy"], "heldout_dpo_loss": o["dpo_loss"],
         "heldout_margin": o["margin_mean"],
         "chosen_logp_delta": float((p.policy_chosen_logp - p.ref_chosen_logp).mean()),
         "rejected_logp_delta": float((p.policy_rejected_logp - p.ref_rejected_logp).mean()),
         "frac_both_down": float(((p.policy_chosen_logp < p.ref_chosen_logp) &
                                  (p.policy_rejected_logp < p.ref_rejected_logp)).mean())}
    g = s.get("generation")
    if g:
        r.update({"kl_token": g["kl_token_mean_heldout"], "reward": g["reward_heldout"]["mean"],
                  "len_tokens": g["length_tokens_heldout"]["mean"], "truncated_frac": g["truncated_frac_heldout"],
                  "word_limit_ok": g["word_limit_compliance"]})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo_long.yaml")
    ap.add_argument("--baseline-results", default="results/task1_dpo", help="where the graded SFT eval lives")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])

    train = {}
    log = res / "train" / "standard" / "train_log.jsonl"
    if log.exists():
        t = pd.DataFrame([json.loads(l) for l in log.read_text().splitlines() if l.strip()])
        for e, g in t.groupby("epoch"):
            train[int(e) + 1] = {"train_loss": g.loss.mean(), "train_acc": g.acc.mean(), "train_margin": g.margin.mean(),
                                 "opt_steps_total": int(g.opt_step.max())}

    rows = []
    sft = repo_path(args.baseline_results) / "eval" / "sft"
    if (sft / "summary.json").exists():
        rows.append({"epoch": 0, **row_from_eval("sft", sft)})
    for d in sorted((res / "eval").glob("epoch_*"), key=lambda d: int(d.name.split("_")[1])):
        if (d / "summary.json").exists():
            k = int(d.name.split("_")[1])
            rows.append({"epoch": k, **train.get(k, {}), **row_from_eval(d.name, d)})
    if not rows:
        raise SystemExit("No evaluations found; run task1_dpo.evaluate on the epoch adapters first.")
    df = pd.DataFrame(rows)
    df.to_csv(res / "epoch_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    print("wrote", res / "epoch_summary.csv")


if __name__ == "__main__":
    main()
