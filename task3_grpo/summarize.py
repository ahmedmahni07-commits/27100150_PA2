"""Collect Task 3 tables + qualitative candidates (CPU only; reads saved results).

Writes results/task3_grpo/standard_trajectory.csv, eval_summary.csv, qualitative_candidates.md.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task2_ppo.summarize import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    log = res / "train" / "standard" / "update_log.jsonl"
    if log.exists():
        ups = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        pd.DataFrame(ups).drop(columns=["prompt_ids"]).to_csv(res / "standard_trajectory.csv", index=False)
    rows = []
    for d in sorted((res / "eval").glob("*/summary.json")):
        s = json.loads(d.read_text())
        rows.append({"name": s["name"], "n_prompts": s["n_prompts"], "reward": s["reward_raw"]["mean"],
                     "reward_se": s["reward_raw"]["se"], "kl_token": s["kl_token_mean"], "entropy": s["entropy_token_mean"],
                     "len_mean": s["length_tokens"]["mean"], "len_std": s["length_tokens"]["std"],
                     "truncated_frac": s["truncated_frac"]})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(res / "eval_summary.csv", index=False)
        print(df.round(4).to_string(index=False))
    md = ["# Task 3 qualitative candidates (you judge quality)\n"]
    md += compare(res, "midpoint", "standard")
    md += compare(res, "fork_grpo", "fork_dr_grpo")
    # Uninformative groups seen during the standard run: all K completions scored identically.
    ro = res / "train" / "standard" / "rollouts.jsonl"
    if ro.exists():
        r = pd.DataFrame(read_jsonl(ro))
        g = r.groupby("update").reward.agg(["std", "mean"]).reset_index()
        md.append("\n## Standard run: groups ranked by within-group reward std (lowest = least informative)\n")
        for _, row in g.sort_values("std").head(3).iterrows():
            sub = r[r["update"] == row["update"]]
            md.append(f"\n### update {int(row['update'])} | group std {row['std']:.3f} | mean reward {row['mean']:.2f}\n")
            for _, c in sub.iterrows():
                md.append(f"- k={c.k} reward {c.reward:+.2f} tokens {c.response_tokens} truncated {c.truncated}: "
                          f"{str(c.response)[:250].strip()} [...]\n")
    (res / "qualitative_candidates.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", res / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
