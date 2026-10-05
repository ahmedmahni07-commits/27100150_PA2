"""Common held-out evaluation for PPO (and reused by GRPO): one protocol for every condition.

Prompt set: the first `--n-prompts` rows of data/rl_prompt_pool_eval.jsonl (file order) whose
rendered prompt fits max_prompt_length. One sampled response per prompt (configs/base.yaml
decoding, course seed, cap = eval_max_response_length). Metrics:
  reward_raw / reward_effective (raw minus missing_eos_penalty if the response never ended),
  KL = token-averaged log pi - log ref over all sampled tokens (common.metrics convention),
  entropy = mean full-distribution token entropy, length stats, truncation / EOS rates.
Outputs results/<task>/eval/<name>/{summary.json, generations.jsonl}.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.rl import generate, load_prompt_pool, score_responses, token_logprobs_and_entropy

GEN_BATCH = 8


def stats(x):
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "std": float(x.std()), "se": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
            "median": float(med), "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max()), "n": int(len(x))}


def evaluate(config_path, adapter, name, n_prompts=64, max_new_tokens=None, results_dir=None):
    cfg = load_yaml(config_path)
    max_new = int(max_new_tokens or cfg.get("eval_max_response_length", cfg.get("max_completion_length", 768)))
    out = repo_path(results_dir or cfg["results_dir"]) / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    adapter = None if adapter in (None, "", "none", "sft") else adapter

    tok = load_tokenizer(cfg["base_model"])
    pool, dropped = load_prompt_pool(cfg["paths"]["rl_prompt_eval"], tok, cfg["max_prompt_length"])
    rows = pool[: int(n_prompts)]
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    print(f"[eval {name}] adapter={adapter} prompts={len(rows)} cap={max_new}")

    set_seed(int(cfg["seed"]))
    recs, kl_num, ent_num, tok_den = [], 0.0, 0.0, 0.0
    for s in range(0, len(rows), GEN_BATCH):
        chunk = rows[s:s + GEN_BATCH]
        msgs = [prompt_messages(r) for r in chunk]
        g = generate(model, tok, msgs, cfg, max_new, cfg["max_prompt_length"])
        for j, r in enumerate(chunk):  # one sequence at a time keeps the vocab-sized logits small
            sl = slice(j, j + 1)
            m = g["response_mask"][sl].float()
            with torch.no_grad():
                lp, ent = token_logprobs_and_entropy(model, g["sequences"][sl], g["attention_mask"][sl],
                                                     g["prompt_width"], g["response_ids"][sl])
                with reference_mode(model):
                    rlp, _ = token_logprobs_and_entropy(model, g["sequences"][sl], g["attention_mask"][sl],
                                                        g["prompt_width"], g["response_ids"][sl], with_entropy=False)
            d = ((lp - rlp) * m).sum()
            kl_num += float(d)
            ent_num += float((ent * m).sum())
            tok_den += float(m.sum())
            recs.append({
                "prompt_id": str(r["prompt_id"]), "prompt": msgs[j][-1]["content"],
                "response": g["responses"][j], "response_tokens": int(g["response_lengths"][j]),
                "terminated_with_eos": bool(g["terminated_with_eos"][j]), "truncated": bool(g["truncated"][j]),
                "seq_kl_sum": float(d), "seq_kl_mean": float(d / m.sum().clamp_min(1)),
                "seq_entropy_mean": float((ent * m).sum() / m.sum().clamp_min(1)),
                "messages": msgs[j],
            })
        print(f"    generated {min(s + GEN_BATCH, len(rows))}/{len(rows)}")
    clear_gpu(model)
    model = None

    rm, rtok = load_reward_model(cfg)
    for s in range(0, len(recs), GEN_BATCH):
        chunk = recs[s:s + GEN_BATCH]
        raw, eff = score_responses(rm, rtok, [r["messages"] for r in chunk], [r["response"] for r in chunk],
                                   [r["terminated_with_eos"] for r in chunk], cfg.get("missing_eos_penalty", 0.0),
                                   cfg.get("reward_max_length", 1280))
        for r, a, b in zip(chunk, raw.tolist(), eff.tolist()):
            r["reward_raw"], r["reward_effective"] = a, b
    clear_gpu(rm)
    for r in recs:
        r.pop("messages")
    write_jsonl(out / "generations.jsonl", recs)

    summary = {
        "name": name, "adapter": adapter, "n_prompts": len(recs), "max_new_tokens": max_new,
        "prompt_ids": [r["prompt_id"] for r in recs], "n_eval_prompts_dropped_too_long": len(dropped),
        "decoding": cfg["generation"], "seed": cfg["seed"],
        "reward_raw": stats([r["reward_raw"] for r in recs]),
        "reward_effective": stats([r["reward_effective"] for r in recs]),
        "kl_token_mean": kl_num / max(tok_den, 1.0),
        "kl_seq_sum": stats([r["seq_kl_sum"] for r in recs]),
        "entropy_token_mean": ent_num / max(tok_den, 1.0),
        "length_tokens": stats([r["response_tokens"] for r in recs]),
        "truncated_frac": float(np.mean([r["truncated"] for r in recs])),
        "eos_frac": float(np.mean([r["terminated_with_eos"] for r in recs])),
    }
    save_json(out / "summary.json", summary)
    print(f"[eval {name}] reward {summary['reward_effective']['mean']:.3f} (raw {summary['reward_raw']['mean']:.3f}) | "
          f"KL {summary['kl_token_mean']:.4f} | H {summary['entropy_token_mean']:.3f} | "
          f"len {summary['length_tokens']['mean']:.0f} | trunc {summary['truncated_frac']:.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int, default=64)
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.n_prompts)


if __name__ == "__main__":
    main()
