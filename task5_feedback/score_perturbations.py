"""Task 5 Step 2: score the Controlled Reward Diagnostic Set with both feedback mechanisms.

20 problems x 5 variants. Controlled pairs, each (diagnostically better, worse):
  reasoning   clean_correct vs corrupt_reasoning_correct_final   (final held correct, reasoning degraded)
  outcome     clean_correct vs good_reasoning_wrong_final        (reasoning ~fixed, final changed)
  filler      clean_correct vs persuasive_filler_correct         (same correct content + persuasive filler;
                                                                  preferring the filler = style bias)
  distractor  clean_correct vs gold_distractor_wrong_final       (gold number mentioned, wrong final)
Mechanisms:
  verifier  exact_reward on each response; prefers whichever scores higher, tie if equal;
  judge     PairwiseAIJudge.compare, asked in BOTH presentation orders (each counts 0.5), so a pure
            position preference cannot masquerade as a preference for content.
Per pair type and mechanism: better / tie / wrong-preference rates, plus judge order-consistency.
S_reason = better-rate on the reasoning pairs; S_outcome = better-rate on outcome-changing pairs
(reported for the outcome pair and pooled with the distractor pair).
Also: verifier reward per variant (checked against expected_exact_reward) and the direct-RLAIF
group reward per variant (PairwiseAIJudge.group_rewards over the 5 variants of each problem),
i.e. the reward each variant would receive during RLAIF training.
Outputs results/task5_feedback/diagnostics/{pair_results.csv, pair_summary.csv, variant_scores.csv, summary.json}.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
PAIRS = [
    ("reasoning", "clean_correct", "corrupt_reasoning_correct_final"),
    ("outcome", "clean_correct", "good_reasoning_wrong_final"),
    ("filler", "clean_correct", "persuasive_filler_correct"),
    ("distractor", "clean_correct", "gold_distractor_wrong_final"),
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-group-rewards", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")

    vrows, prows = [], []
    for pid, v in groups.items():
        q = v["clean_correct"]["question"]
        gold = str(v["clean_correct"]["gold_final"])
        ver = {k: exact_reward(v[k]["response"], gold) for k in VARIANT_ORDER}
        grp = None if args.skip_group_rewards else judge.group_rewards(q, [v[k]["response"] for k in VARIANT_ORDER])
        for i, k in enumerate(VARIANT_ORDER):
            vrows.append({"problem_id": pid, "variant": k, "verifier_reward": ver[k],
                          "expected_exact_reward": v[k].get("expected_exact_reward"),
                          "verifier_matches_expected": (v[k].get("expected_exact_reward") is None) or (float(ver[k]) == float(v[k]["expected_exact_reward"])),
                          "rlaif_group_reward": None if grp is None else grp[i],
                          "response_chars": len(v[k]["response"])})
        for ptype, better, worse in PAIRS:
            vb, vw = ver[better], ver[worse]
            vpref = "better" if vb > vw else "worse" if vw > vb else "tie"
            j1 = judge.compare(q, v[better]["response"], v[worse]["response"])    # better shown as A
            j2 = judge.compare(q, v[worse]["response"], v[better]["response"])    # better shown as B
            o1 = {"A": "better", "B": "worse", "TIE": "tie"}[j1]
            o2 = {"A": "worse", "B": "better", "TIE": "tie"}[j2]
            prows.append({"problem_id": pid, "pair": ptype, "better": better, "worse": worse,
                          "verifier_better": vb, "verifier_worse": vw, "verifier_pref": vpref,
                          "judge_order1": o1, "judge_order2": o2, "judge_consistent": o1 == o2})

    pr, vr = pd.DataFrame(prows), pd.DataFrame(vrows)
    pr.to_csv(out / "pair_results.csv", index=False)
    vr.to_csv(out / "variant_scores.csv", index=False)

    summ = []
    for ptype, g in pr.groupby("pair", sort=False):
        jb = (np.mean(g.judge_order1 == "better") + np.mean(g.judge_order2 == "better")) / 2
        jt = (np.mean(g.judge_order1 == "tie") + np.mean(g.judge_order2 == "tie")) / 2
        jw = (np.mean(g.judge_order1 == "worse") + np.mean(g.judge_order2 == "worse")) / 2
        summ += [
            {"pair": ptype, "mechanism": "verifier", "n": len(g), "better_rate": float(np.mean(g.verifier_pref == "better")),
             "tie_rate": float(np.mean(g.verifier_pref == "tie")), "wrong_rate": float(np.mean(g.verifier_pref == "worse")),
             "order_consistency": 1.0},
            {"pair": ptype, "mechanism": "ai_judge", "n": len(g), "better_rate": float(jb), "tie_rate": float(jt),
             "wrong_rate": float(jw), "order_consistency": float(g.judge_consistent.mean())},
        ]
    sd = pd.DataFrame(summ)
    sd.to_csv(out / "pair_summary.csv", index=False)

    def br(mech, pairs):
        return float(sd[(sd.mechanism == mech) & sd.pair.isin(pairs)].better_rate.mean())

    result = {
        "n_problems": len(groups),
        "S_reason": {"verifier": br("verifier", ["reasoning"]), "ai_judge": br("ai_judge", ["reasoning"])},
        "S_outcome": {"verifier": br("verifier", ["outcome"]), "ai_judge": br("ai_judge", ["outcome"])},
        "S_outcome_incl_distractor": {"verifier": br("verifier", ["outcome", "distractor"]), "ai_judge": br("ai_judge", ["outcome", "distractor"])},
        "verifier_matches_expected_rate": float(vr.verifier_matches_expected.mean()),
        "verifier_mismatches": vr[~vr.verifier_matches_expected][["problem_id", "variant"]].to_dict("records"),
        "mean_verifier_reward_by_variant": vr.groupby("variant").verifier_reward.mean().to_dict(),
        "mean_rlaif_group_reward_by_variant": (None if args.skip_group_rewards else vr.groupby("variant").rlaif_group_reward.mean().to_dict()),
        "judge_order_consistency_overall": float(pr.judge_consistent.mean()),
    }
    save_json(out / "summary.json", result)
    pd.set_option("display.width", 200)
    print(sd.round(3).to_string(index=False))
    print("S_reason", result["S_reason"], "| S_outcome", result["S_outcome"])
    print("verifier matches expected:", result["verifier_matches_expected_rate"])
    if result["mean_rlaif_group_reward_by_variant"]:
        print("RLAIF group reward by variant:", {k: round(v, 3) for k, v in result["mean_rlaif_group_reward_by_variant"].items()})


if __name__ == "__main__":
    main()
