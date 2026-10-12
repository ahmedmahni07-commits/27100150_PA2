"""Task 5 final comparison: in-domain, transfer and controlled-diagnostic results in one place.

Reads results/task5_feedback/{gsm,transfer}/summary.json and diagnostics/summary.json and writes
results/task5_feedback/feedback_comparison.csv (one row per policy) and feedback_comparison.json
(including the per-mechanism diagnostic summary and the inference-cost figures used to compare
feedback sources). Also writes qualitative_candidates.md with examples for the report.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def clip(t, n=700):
    t = str(t).strip()
    return t if len(t) <= n else t[:n] + " [...]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, tr, dg = load(base / "gsm" / "summary.json"), load(base / "transfer" / "summary.json"), load(base / "diagnostics" / "summary.json")
    if gsm is None:
        raise SystemExit("Run task5_feedback.evaluate_math --dataset gsm first.")

    rows = []
    for name, s in gsm["policies"].items():
        t = tr["policies"].get(name) if tr else None
        row = {"policy": name,
               "gsm_accuracy": s["exact_accuracy"], "gsm_accuracy_se": s["exact_accuracy_se"],
               "gsm_format": s["format_compliance"], "gsm_len_mean": s["length_tokens"]["mean"],
               "gsm_len_std": s["length_tokens"]["std"], "gsm_truncated": s["truncated_frac"],
               "gsm_win_vs_sft": s.get("win_rate_vs_sft"), "gsm_ties": s.get("ties"),
               "gsm_ties_identical": s.get("ties_identical_to_sft"),
               "gsm_verifier_judge_agreement": (s.get("verifier_judge_agreement") or {}).get("judge_agrees_with_verifier")}
        if t:
            row.update({"svamp_accuracy": t["exact_accuracy"], "svamp_accuracy_se": t["exact_accuracy_se"],
                        "svamp_format": t["format_compliance"], "svamp_len_mean": t["length_tokens"]["mean"],
                        "svamp_win_vs_sft": t.get("win_rate_vs_sft"), "svamp_ties": t.get("ties"),
                        "accuracy_drop_gsm_to_svamp": s["exact_accuracy"] - t["exact_accuracy"],
                        "win_rate_drop_gsm_to_svamp": (s.get("win_rate_vs_sft") - t.get("win_rate_vs_sft"))
                        if s.get("win_rate_vs_sft") is not None and t.get("win_rate_vs_sft") is not None else None})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(base / "feedback_comparison.csv", index=False)

    judge_sec = [s.get("judge_seconds_per_call") for d in (gsm, tr) if d for s in d["policies"].values() if s.get("judge_seconds_per_call")]
    cost = {"verifier_microseconds_per_call": gsm["verifier_microseconds_per_call"],
            "judge_seconds_per_pairwise_call": float(sum(judge_sec) / len(judge_sec)) if judge_sec else None,
            "judge_calls_per_group_of_K": "K*(K-1)/2 pairwise calls per prompt group (e.g. 6 for K=4)",
            "judge_model": gsm["judge_model"]}
    save_json(base / "feedback_comparison.json", {"table": rows, "diagnostics": dg, "cost": cost})
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    if dg:
        print("S_reason", dg["S_reason"], "S_outcome", dg["S_outcome"])
    print("cost:", cost)

    md = ["# Task 5 qualitative candidates (verify each yourself)\n"]
    pr = base / "diagnostics" / "pair_results.csv"
    diag_rows = {(str(r["problem_id"]), r["variant_type"]): r for r in read_jsonl(cfg["paths"]["task5_diagnostics"])}
    if pr.exists():
        p = pd.read_csv(pr)
        for ptype, title in [("reasoning", "Reasoning-only corruption (verifier ties by design)"),
                             ("filler", "Persuasive filler (does the judge reward style?)"),
                             ("distractor", "Gold-number distractor with wrong final")]:
            md.append(f"\n## {title}\n")
            sub = p[p.pair == ptype]
            # Show judge mistakes first: those explain verifier-judge disagreement.
            sub = sub.assign(mistake=(sub.judge_order1 != "better").astype(int) + (sub.judge_order2 != "better").astype(int))
            for _, r in sub.sort_values("mistake", ascending=False).head(2).iterrows():
                b, w = diag_rows[(str(r.problem_id), r.better)], diag_rows[(str(r.problem_id), r.worse)]
                md.append(f"\n### problem {r.problem_id}: verifier {r.verifier_pref}, judge {r.judge_order1}/{r.judge_order2}\n"
                          f"**Gold:** {b['gold_final']}\n\n**{r.better}:** {clip(b['response'])}\n\n**{r.worse}:** {clip(w['response'])}\n")
    for ds in ("gsm", "transfer"):
        for name in ("rlvr", "rlaif"):
            pp = base / ds / f"pairwise_{name}_vs_sft.jsonl"
            if not pp.exists():
                continue
            pairs = pd.DataFrame(read_jsonl(pp))
            dis = pairs[((pairs.pref == "A") & (pairs.a_correct < pairs.b_correct)) | ((pairs.pref == "B") & (pairs.a_correct > pairs.b_correct))]
            md.append(f"\n## {ds}: judge contradicts verifier ({name} vs SFT): {len(dis)} cases\n")
            gens = {n: {r["id"]: r for r in read_jsonl(base / ds / f"generations_{n}.jsonl")} for n in (name, "sft")}
            for _, r in dis.head(2).iterrows():
                a, b = gens[name][r.id], gens["sft"][r.id]
                md.append(f"\n### {r.id} gold {a['gold_final']} | judge prefers {'%s' % name if r.pref == 'A' else 'sft'}\n"
                          f"**{name}** (pred {a['pred_final']}, correct {a['correct']}): {clip(a['response'], 500)}\n\n"
                          f"**sft** (pred {b['pred_final']}, correct {b['correct']}): {clip(b['response'], 500)}\n")
    (base / "qualitative_candidates.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", base / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
