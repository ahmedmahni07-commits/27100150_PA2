"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (in-domain) or SVAMP (transfer).

Protocol (identical for every policy and both datasets):
  * one greedy response per problem (do_sample=False), cap math_max_new_tokens, fixed file order;
  * exact accuracy = task5_feedback.rlvr.exact_reward (designated `#### <number>` must equal gold);
  * format compliance = the designated final is parseable (extract_designated_final is not None);
  * response length in generated tokens;
  * AI pairwise result vs SFT with the fixed PairwiseAIJudge (A = the policy, B = SFT; the judge
    balances A/B orientation internally): win = 1, tie = 0.5, loss = 0. Responses identical to SFT
    are scored as ties without calling the judge and counted separately;
  * verifier-judge agreement on the same pairs: among pairs where the verifier separates the two
    responses (one correct, one not), the share where the judge prefers the correct one; among
    verifier ties, the judge's tie rate.
Outputs results/task5_feedback/<dataset>/{generations_<policy>.jsonl, pairwise_<policy>_vs_sft.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

GEN_BATCH = 8


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def row_id(r):
    return str(r.get("prompt_id", r.get("source_index")))


def generate_policy(cfg, rows, tok, name, path):
    model = load_frozen_policy(cfg, name)
    recs = []
    t0 = time.perf_counter()
    for s in range(0, len(rows), GEN_BATCH):
        chunk = rows[s:s + GEN_BATCH]
        g = batch_generate(model, tok, [r["messages"] for r in chunk], max_prompt_length=512,
                           max_new_tokens=int(cfg["math_max_new_tokens"]), do_sample=False)
        for r, text, n, trunc in zip(chunk, g["responses"], g["response_lengths"], g["truncated"]):
            pred = extract_designated_final(text)
            recs.append({"id": row_id(r), "gold_final": str(r["gold_final"]), "response": text,
                         "response_tokens": int(n), "truncated": bool(trunc), "pred_final": pred,
                         "format_ok": pred is not None, "correct": exact_reward(text, str(r["gold_final"]))})
        print(f"    [{name}] {min(s + GEN_BATCH, len(rows))}/{len(rows)}")
    clear_gpu(model)
    write_jsonl(path, recs)
    return recs, time.perf_counter() - t0


def pairwise(judge, rows, a_recs, b_recs, path):
    out, t_judge, n_calls = [], 0.0, 0
    for r, a, b in zip(rows, a_recs, b_recs):
        assert a["id"] == b["id"] == row_id(r)
        if a["response"].strip() == b["response"].strip():
            pref, identical = "TIE", True
        else:
            t0 = time.perf_counter()
            pref = judge.compare(r["question"], a["response"], b["response"])
            t_judge += time.perf_counter() - t0
            n_calls += 1
            identical = False
        out.append({"id": a["id"], "pref": pref, "identical": identical, "a_correct": a["correct"], "b_correct": b["correct"],
                    "score": 1.0 if pref == "A" else 0.5 if pref == "TIE" else 0.0})
    write_jsonl(path, out)
    return out, t_judge, n_calls


def agreement(pairs):
    dec = [p for p in pairs if p["a_correct"] != p["b_correct"]]
    same = [p for p in pairs if p["a_correct"] == p["b_correct"]]
    agree = [((p["pref"] == "A") and p["a_correct"] > p["b_correct"]) or ((p["pref"] == "B") and p["b_correct"] > p["a_correct"]) for p in dec]
    contra = [((p["pref"] == "B") and p["a_correct"] > p["b_correct"]) or ((p["pref"] == "A") and p["b_correct"] > p["a_correct"]) for p in dec]
    return {"n_verifier_decisive": len(dec),
            "judge_agrees_with_verifier": float(np.mean(agree)) if dec else float("nan"),
            "judge_contradicts_verifier": float(np.mean(contra)) if dec else float("nan"),
            "judge_tie_when_verifier_decisive": float(np.mean([p["pref"] == "TIE" for p in dec])) if dec else float("nan"),
            "n_verifier_tie": len(same),
            "judge_tie_when_verifier_tie": float(np.mean([p["pref"] == "TIE" for p in same])) if same else float("nan"),
            "judge_tie_when_verifier_tie_excl_identical": float(np.mean([p["pref"] == "TIE" for p in same if not p["identical"]])) if any(not p["identical"] for p in same) else float("nan")}


def stats(x):
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    return {"mean": float(x.mean()), "std": float(x.std()), "iqr": float(q3 - q1), "median": float(np.median(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    gens, gen_secs = {}, {}
    for name in policy_specs(cfg):
        p = out / f"generations_{name}.jsonl"
        if p.exists() and not args.force:
            gens[name] = read_jsonl(p)
            print(f"[{name}] loaded {len(gens[name])} cached generations")
        else:
            gens[name], gen_secs[name] = generate_policy(cfg, rows, tok, name, p)

    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
    pair_res = {}
    for name in ("rlvr", "rlaif"):
        p = out / f"pairwise_{name}_vs_sft.jsonl"
        pairs, tj, nc = pairwise(judge, rows, gens[name], gens["sft"], p)
        pair_res[name] = {"pairs": pairs, "judge_seconds": tj, "judge_calls": nc}

    t0 = time.perf_counter()
    for r in gens["sft"]:
        exact_reward(r["response"], r["gold_final"])
    verifier_us = (time.perf_counter() - t0) / max(len(gens["sft"]), 1) * 1e6

    summary = {"dataset": args.dataset, "file": dataset_path(cfg, args.dataset), "n": len(rows),
               "decoding": "greedy", "max_new_tokens": int(cfg["math_max_new_tokens"]),
               "judge_model": cfg["ai_judge_model"], "verifier_microseconds_per_call": verifier_us, "policies": {}}
    for name, recs in gens.items():
        acc = np.array([r["correct"] for r in recs], dtype=float)
        s = {"exact_accuracy": float(acc.mean()), "exact_accuracy_se": float(acc.std(ddof=1) / np.sqrt(len(acc))),
             "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
             "length_tokens": stats([r["response_tokens"] for r in recs]),
             "truncated_frac": float(np.mean([r["truncated"] for r in recs])),
             "generation_seconds": gen_secs.get(name)}
        if name in pair_res:
            pr = pair_res[name]["pairs"]
            sc = np.array([p["score"] for p in pr])
            s.update({"win_rate_vs_sft": float(sc.mean()), "win_rate_vs_sft_se": float(sc.std(ddof=1) / np.sqrt(len(sc))),
                      "wins": int(sum(p["pref"] == "A" for p in pr)), "losses": int(sum(p["pref"] == "B" for p in pr)),
                      "ties": int(sum(p["pref"] == "TIE" for p in pr)),
                      "ties_identical_to_sft": int(sum(p["identical"] for p in pr)),
                      "judge_calls": pair_res[name]["judge_calls"],
                      "judge_seconds_per_call": pair_res[name]["judge_seconds"] / max(pair_res[name]["judge_calls"], 1),
                      "verifier_judge_agreement": agreement(pr)})
        summary["policies"][name] = s
    save_json(out / "summary.json", summary)
    for name, s in summary["policies"].items():
        extra = (f" | win vs SFT {s['win_rate_vs_sft']:.3f} (W{s['wins']}/T{s['ties']}/L{s['losses']}, identical {s['ties_identical_to_sft']})"
                 if "win_rate_vs_sft" in s else "")
        print(f"[{args.dataset}] {name:5s} acc {s['exact_accuracy']:.3f} | format {s['format_compliance']:.3f} | "
              f"len {s['length_tokens']['mean']:.0f} | trunc {s['truncated_frac']:.2f}{extra}")


if __name__ == "__main__":
    main()
