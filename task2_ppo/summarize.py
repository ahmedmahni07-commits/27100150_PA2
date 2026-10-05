"""Collect Task 2 tables + qualitative candidates (no GPU needed; reads saved results).

Writes:
  results/task2_ppo/standard_trajectory.csv   per-update diagnostics of the standard continuation
  results/task2_ppo/eval_summary.csv          every evaluated condition on the common held-out set
  results/task2_ppo/qualitative_candidates.md side-by-side responses for chosen comparisons
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def clip(t, n=600):
    t = str(t).strip()
    return t if len(t) <= n else t[:n] + " [...]"


def compare(res, a, b, top=6):
    pa, pb = res / "eval" / a / "generations.jsonl", res / "eval" / b / "generations.jsonl"
    if not (pa.exists() and pb.exists()):
        return [f"\n_(skipped {a} vs {b}: missing eval)_\n"]
    m = pd.DataFrame(read_jsonl(pa)).merge(pd.DataFrame(read_jsonl(pb)), on="prompt_id", suffixes=("_a", "_b"))
    m["gain"] = m.reward_effective_b - m.reward_effective_a
    out = [f"\n## {b} vs {a}: largest reward gains (does the response actually improve?)\n"]
    for _, r in m.sort_values("gain", ascending=False).head(top).iterrows():
        out.append(f"\n### {r.prompt_id[:12]} reward {r.reward_effective_a:+.2f} -> {r.reward_effective_b:+.2f} | "
                   f"tokens {r.response_tokens_a} -> {r.response_tokens_b} | truncated {r.truncated_a} -> {r.truncated_b}\n")
        out.append(f"**Prompt:** {clip(r.prompt_a, 350)}\n\n**{a}:** {clip(r.response_a)}\n\n**{b}:** {clip(r.response_b)}\n")
    out.append(f"\n## {b} vs {a}: largest reward drops\n")
    for _, r in m.sort_values("gain").head(top // 2).iterrows():
        out.append(f"\n### {r.prompt_id[:12]} reward {r.reward_effective_a:+.2f} -> {r.reward_effective_b:+.2f} | "
                   f"tokens {r.response_tokens_a} -> {r.response_tokens_b}\n")
        out.append(f"**Prompt:** {clip(r.prompt_a, 350)}\n\n**{a}:** {clip(r.response_a)}\n\n**{b}:** {clip(r.response_b)}\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])

    log = res / "train" / "standard" / "update_log.jsonl"
    if log.exists():
        ups = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        cols = ["update", "reward_raw", "reward_effective", "kl_token_mean", "kl_seq_sum", "entropy", "policy_loss",
                "value_loss", "clip_frac", "affected_frac", "grad_norm_policy", "grad_norm_value", "response_len",
                "truncated_frac", "value_mean", "return_mean", "explained_var_before", "explained_var_after",
                "approx_kl_update", "elapsed_s", "peak_vram_gb"]
        pd.DataFrame([{c: u.get(c) for c in cols} for u in ups]).to_csv(res / "standard_trajectory.csv", index=False)

    rows = []
    for d in sorted((res / "eval").glob("*/summary.json")):
        s = json.loads(d.read_text())
        rows.append({"name": s["name"], "n_prompts": s["n_prompts"], "reward_effective": s["reward_effective"]["mean"],
                     "reward_se": s["reward_effective"]["se"], "reward_raw": s["reward_raw"]["mean"],
                     "kl_token": s["kl_token_mean"], "entropy": s["entropy_token_mean"],
                     "len_mean": s["length_tokens"]["mean"], "len_std": s["length_tokens"]["std"],
                     "len_iqr": s["length_tokens"]["iqr"], "truncated_frac": s["truncated_frac"], "eos_frac": s["eos_frac"]})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(res / "eval_summary.csv", index=False)
        pd.set_option("display.width", 250)
        print(df.round(4).to_string(index=False))

    md = ["# Task 2 qualitative candidates (you judge quality; reward is only the RM's opinion)\n"]
    md += compare(res, "midpoint", "standard")
    md += compare(res, "fork_eps0.2_kl0.2", "fork_eps0.2_kl0")
    (res / "qualitative_candidates.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", res / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
