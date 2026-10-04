"""Task 1 evaluation: one common protocol for every DPO condition (and the SFT baseline).

Outputs (under results/task1_dpo/eval/<name>/):
  summary.json        aggregate metrics (preference + generation)
  preference.csv      per held-out pair log-probs, DPO margin, loss
  generations.jsonl   per prompt sampled response, length, reward, KL, word-limit compliance

Protocol (identical across conditions):
  * Preference metrics on a fixed held-out pair file (default: dpo_standard_eval).
      m = [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)]   (summed over response tokens)
      accuracy = mean(m > 0); loss = -log sigmoid(beta * m)
  * Generation metrics on a fixed prompt set = the first `--n-gen` held-out prompts whose rendered
    prompt is <= GEN_PROMPT_MAX tokens (so batch_generate never truncates a prompt), plus all
    word-limit prompts. Sampling uses configs/base.yaml generation settings, the course seed and
    max_generation_tokens from configs/dpo.yaml.
  * KL = common.metrics.sampled_kl over all valid sampled response tokens (token-averaged,
    pooled across prompts). The per-sequence summed KL is also reported.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_fitting_rows, make_collate, model_device, pair_logps, to_device

GEN_PROMPT_MAX = 512
GEN_BATCH = 8
PREF_BATCH = 2


def length_stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "std": float(x.std()), "median": float(med),
            "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max()), "n": int(len(x))}


def generation_prompts(cfg, tokenizer, n_gen):
    """Fixed generation prompt set shared by every Task 1 condition."""
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    out = []
    for r in rows:
        msgs = prompt_messages_from_preference(r)
        n = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if n <= GEN_PROMPT_MAX:
            out.append({"prompt_id": str(r["prompt_id"]), "set": "heldout", "messages": msgs})
        if len(out) >= n_gen:
            break
    for r in read_jsonl(cfg["paths"]["word_limit_prompts"]):
        out.append({"prompt_id": str(r["prompt_id"]), "set": "word_limit", "messages": r["messages"]})
    return out


@torch.no_grad()
def preference_eval(model, tokenizer, cfg, pref_path, beta):
    rows = read_jsonl(pref_path)
    rows, dropped = filter_fitting_rows(rows, tokenizer, int(cfg["max_sequence_length"]))
    loader = DataLoader(rows, batch_size=PREF_BATCH, shuffle=False,
                        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])))
    device = model_device(model)
    model.eval()
    recs, k = [], 0
    for batch in loader:
        b = batch["n_pairs"]
        batch = to_device(batch, device)
        pc, pr, rc, rr = pair_logps(model, batch)
        m = (pc - rc) - (pr - rr)
        loss_vec = -torch.nn.functional.logsigmoid(beta * m)
        for j in range(b):
            row = rows[k]
            k += 1
            recs.append({
                "prompt_id": str(row.get("prompt_id")),
                "length_stratum": row.get("length_stratum", "all"),
                "chosen_tokens": row.get("chosen_tokens"),
                "rejected_tokens": row.get("rejected_tokens"),
                "policy_chosen_logp": float(pc[j]), "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]), "ref_rejected_logp": float(rr[j]),
                "margin": float(m[j]),
                "implicit_reward_chosen": float(beta * (pc[j] - rc[j])),
                "implicit_reward_rejected": float(beta * (pr[j] - rr[j])),
                "dpo_loss": float(loss_vec[j]),
                "correct": float(m[j] > 0),
                "ref_prefers_chosen": float(rc[j] > rr[j]),
            })
    df = pd.DataFrame(recs)

    def agg(d):
        return {"n": int(len(d)), "dpo_loss": float(d.dpo_loss.mean()),
                "preference_accuracy": float(d.correct.mean()),
                "margin_mean": float(d.margin.mean()), "margin_std": float(d.margin.std(ddof=0)),
                "implicit_reward_chosen": float(d.implicit_reward_chosen.mean()),
                "implicit_reward_rejected": float(d.implicit_reward_rejected.mean()),
                "ref_logp_prefers_chosen": float(d.ref_prefers_chosen.mean())}

    out = {"overall": agg(df), "dropped_prompt_too_long": dropped, "pref_file": str(pref_path), "beta": beta}
    if df.length_stratum.nunique() > 1:
        out["by_stratum"] = {s: agg(d) for s, d in df.groupby("length_stratum")}
    return out, df


@torch.no_grad()
def generation_eval(model, tokenizer, cfg, prompts):
    gcfg = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    set_seed(int(cfg["seed"]))
    recs = []
    kl_num, kl_den = 0.0, 0.0
    for s in range(0, len(prompts), GEN_BATCH):
        chunk = prompts[s:s + GEN_BATCH]
        g = batch_generate(model, tokenizer, [p["messages"] for p in chunk],
                           max_prompt_length=GEN_PROMPT_MAX + 8, max_new_tokens=max_new,
                           temperature=float(gcfg["temperature"]), top_p=float(gcfg["top_p"]),
                           do_sample=bool(gcfg["do_sample"]))
        pol_lp, _ = response_token_logprobs(model, g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"])
        with reference_mode(model):
            ref_lp, _ = response_token_logprobs(model, g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"])
        mask = g["response_mask"].float()
        diff = (pol_lp - ref_lp) * mask
        kl_num += float(diff.sum())
        kl_den += float(mask.sum())
        for j, p in enumerate(chunk):
            text = g["responses"][j]
            prompt_text = p["messages"][-1]["content"]
            recs.append({
                "prompt_id": p["prompt_id"], "set": p["set"], "prompt": prompt_text,
                "response": text,
                "response_tokens": int(g["response_lengths"][j]),
                "truncated": bool(g["truncated"][j]),
                "words": word_count(text),
                "word_limit": parse_word_limit(prompt_text) if p["set"] == "word_limit" else None,
                "word_limit_ok": word_limit_compliance(prompt_text, text) if p["set"] == "word_limit" else None,
                "seq_kl_sum": float(diff[j].sum()),
                "seq_kl_mean": float(diff[j].sum() / mask[j].sum().clamp_min(1)),
                "seq_neg_logp_mean": float(-(pol_lp[j] * mask[j]).sum() / mask[j].sum().clamp_min(1)),
            })
        print(f"    generated {min(s + GEN_BATCH, len(prompts))}/{len(prompts)}")
    return recs, kl_num / max(kl_den, 1.0)


@torch.no_grad()
def reward_eval(cfg, prompts, recs):
    rm, rtok = load_reward_model(cfg)
    rtok.truncation_side = "left"  # never cut the response off the end when scoring
    by_id = {(p["set"], p["prompt_id"]): p["messages"] for p in prompts}
    for s in range(0, len(recs), GEN_BATCH):
        chunk = recs[s:s + GEN_BATCH]
        scores = score_reward_pairs(rm, rtok, [by_id[(r["set"], r["prompt_id"])] for r in chunk],
                                    [r["response"] for r in chunk], max_length=1280)
        for r, sc in zip(chunk, scores.tolist()):
            r["reward"] = float(sc)
    clear_gpu(rm)


def evaluate(config_path, adapter, name, beta=None, pref_path=None, n_gen=100, skip_generation=False, out_dir=None):
    cfg = load_yaml(config_path)
    beta = float(cfg["beta"] if beta is None else beta)
    pref_path = pref_path or cfg["paths"]["dpo_standard_eval"]
    out = repo_path(out_dir or f"{cfg['results_dir']}/eval/{name}")
    out.mkdir(parents=True, exist_ok=True)
    adapter = None if adapter in (None, "", "none", "sft") else adapter

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    print(f"[eval {name}] adapter={adapter} beta={beta} pref={pref_path}")

    pref, pref_df = preference_eval(model, tokenizer, cfg, pref_path, beta)
    pref_df.to_csv(out / "preference.csv", index=False)
    summary = {"name": name, "adapter": adapter, "beta": beta, "preference": pref}

    if not skip_generation:
        prompts = generation_prompts(cfg, tokenizer, n_gen)
        recs, kl_tok = generation_eval(model, tokenizer, cfg, prompts)
        clear_gpu(model)
        model = None
        reward_eval(cfg, prompts, recs)
        write_jsonl(out / "generations.jsonl", recs)
        held = [r for r in recs if r["set"] == "heldout"]
        wl = [r for r in recs if r["set"] == "word_limit"]
        summary["generation"] = {
            "n_heldout_prompts": len(held),
            "heldout_prompt_ids": [r["prompt_id"] for r in held],
            "max_new_tokens": int(cfg["max_generation_tokens"]),
            "decoding": cfg["generation"],
            "kl_token_mean_all_prompts": kl_tok,
            "kl_token_mean_heldout": float(np.sum([r["seq_kl_sum"] for r in held]) / max(1, np.sum([r["response_tokens"] for r in held]))),
            "kl_seq_sum_mean_heldout": float(np.mean([r["seq_kl_sum"] for r in held])),
            "reward_heldout": length_stats([r["reward"] for r in held]),
            "length_tokens_heldout": length_stats([r["response_tokens"] for r in held]),
            "truncated_frac_heldout": float(np.mean([r["truncated"] for r in held])),
            "neg_logp_per_token_heldout": float(np.mean([r["seq_neg_logp_mean"] for r in held])),
            "word_limit_compliance": float(np.mean([r["word_limit_ok"] for r in wl])) if wl else None,
            "word_limit_words": length_stats([r["words"] for r in wl]),
            "word_limit_length_tokens": length_stats([r["response_tokens"] for r in wl]),
            "reward_word_limit": length_stats([r["reward"] for r in wl]),
        }
    if model is not None:
        clear_gpu(model)
    save_json(out / "summary.json", summary)
    o = summary["preference"]["overall"]
    print(f"[eval {name}] pref acc {o['preference_accuracy']:.3f} loss {o['dpo_loss']:.4f}", end="")
    if "generation" in summary:
        gsum = summary["generation"]
        print(f" | KL {gsum['kl_token_mean_heldout']:.4f} | reward {gsum['reward_heldout']['mean']:.3f} | "
              f"len {gsum['length_tokens_heldout']['mean']:.1f} | word-limit ok {gsum['word_limit_compliance']}")
    else:
        print()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT baseline")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: config beta)")
    ap.add_argument("--pref-file", help="held-out pair file (default: dpo_standard_eval)")
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.beta, args.pref_file, args.n_gen, args.skip_generation)


if __name__ == "__main__":
    main()
