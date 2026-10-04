"""Collect candidate examples for the Task 1 qualitative evidence (you still pick and judge them).

Writes results/task1_dpo/qualitative_candidates.md with, for the same prompts:
  A. largest reward-model gains of DPO over SFT (check: is the DPO answer actually better, or
     just longer / more confident / truncated?)
  B. word-limit prompts side by side (instruction compliance vs. length bias)
  C. held-out preference pairs with the largest positive DPO margin whose *chosen* response is
     much longer than the rejected one (is the margin explained by length?)
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def load_gen(res, name):
    p = res / "eval" / name / "generations.jsonl"
    return pd.DataFrame(read_jsonl(p)) if p.exists() else None


def clip(t, n=700):
    t = str(t).strip()
    return t if len(t) <= n else t[:n] + " [...]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--model", default="standard", help="eval name to compare against SFT")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    sft, dpo = load_gen(res, "sft"), load_gen(res, args.model)
    if sft is None or dpo is None:
        raise SystemExit("Run task1_dpo.evaluate for 'sft' and the chosen model first.")
    m = sft.merge(dpo, on=["set", "prompt_id"], suffixes=("_sft", "_dpo"))
    m["reward_gain"] = m.reward_dpo - m.reward_sft
    m["len_gain"] = m.response_tokens_dpo - m.response_tokens_sft
    out = [f"# Task 1 qualitative candidates (SFT vs {args.model})\n"]

    out.append("\n## A. Largest reward gains on held-out prompts\n")
    for _, r in m[m.set == "heldout"].sort_values("reward_gain", ascending=False).head(args.top).iterrows():
        out.append(f"\n### {r.prompt_id[:12]}  reward {r.reward_sft:.2f} -> {r.reward_dpo:.2f}  | tokens {r.response_tokens_sft} -> {r.response_tokens_dpo}"
                   f"  | truncated {r.truncated_sft} -> {r.truncated_dpo}\n")
        out.append(f"**Prompt:** {clip(r.prompt_sft, 400)}\n\n**SFT:** {clip(r.response_sft)}\n\n**DPO:** {clip(r.response_dpo)}\n")

    out.append("\n## B. Word-limit prompts\n")
    for _, r in m[m.set == "word_limit"].iterrows():
        out.append(f"\n### {r.prompt_id} (limit {r.word_limit_sft} words) SFT {r.words_sft}w ok={r.word_limit_ok_sft} reward {r.reward_sft:.2f} | "
                   f"DPO {r.words_dpo}w ok={r.word_limit_ok_dpo} reward {r.reward_dpo:.2f}\n")
        out.append(f"**Prompt:** {r.prompt_sft}\n\n**SFT:** {clip(r.response_sft)}\n\n**DPO:** {clip(r.response_dpo)}\n")

    pref_p = res / "eval" / args.model / "preference.csv"
    if pref_p.exists():
        pref = pd.read_csv(pref_p)
        rows = {str(r["prompt_id"]): r for r in read_jsonl(cfg["paths"]["dpo_standard_eval"])}
        pref["len_c"] = [len(rows[str(i)]["chosen"][-1]["content"]) for i in pref.prompt_id]
        pref["len_r"] = [len(rows[str(i)]["rejected"][-1]["content"]) for i in pref.prompt_id]
        pref["char_ratio"] = pref.len_c / pref.len_r.clip(lower=1)
        out.append("\n## C. Held-out pairs: largest DPO margins where chosen is >=1.5x longer (chars)\n")
        for _, r in pref[pref.char_ratio >= 1.5].sort_values("margin", ascending=False).head(args.top // 2).iterrows():
            row = rows[str(r.prompt_id)]
            out.append(f"\n### {str(r.prompt_id)[:12]} margin {r.margin:+.2f}  chars chosen {r.len_c} vs rejected {r.len_r}\n")
            out.append(f"**Prompt:** {clip(row['prompt'], 300)}\n\n**Chosen:** {clip(row['chosen'][-1]['content'], 500)}\n\n"
                       f"**Rejected:** {clip(row['rejected'][-1]['content'], 500)}\n")

    path = res / "qualitative_candidates.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
