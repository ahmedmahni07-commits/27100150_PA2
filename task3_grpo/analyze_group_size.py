"""Task 3 Step 2: group-size study on the supplied K=8 completion/reward cache (no training).

Equal-generation regrouping: the cache holds 24 prompts x 8 completions = 192 generations. For
K in {2, 4, 8}, each prompt's 8 completions are split into 8/K disjoint groups of K, so every K
uses all 192 generations exactly once (K=2: 96 groups, K=4: 48, K=8: 24). The split is by
generation_index ("fixed" partition); because the order of completions is arbitrary, every metric
is also averaged over N_PERM random within-prompt permutations (seeded) to remove that arbitrariness.

Per K we report:
  informative_rate        share of groups whose reward std > STD_TOL (same tolerance as the helper)
  group_std_mean          mean within-group reward std (population std, as in the advantage)
  adv_var                 variance of the group-relative advantage A over all 192 completions
  centered_var            variance of r - mean_group (signal before std-normalisation)
  nonzero_adv_frac        share of completions with |A| > 1e-6
and the same quantities per prompt-difficulty bin.

Difficulty bins (defined once): a prompt's difficulty is its mean reward over all 8 cached
completions; prompts are split into terciles of that mean: "hard" (lowest third), "medium",
"easy" (highest third). Ties at the cut points go to the lower bin.
Outputs results/task3_grpo/group_size.csv, group_size_by_difficulty.csv, group_size.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

STD_TOL = 1e-6
N_PERM = 200


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, order=None):
    """Split each prompt's first 8 completions into 8/k disjoint groups of size k.

    `order` optionally maps prompt -> permutation of range(8). Returns a list of
    (prompt_id, [rewards]) with every cached completion used exactly once.
    """
    if 8 % k:
        raise ValueError("k must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        idx = list(range(8)) if order is None else list(order[pid])
        rewards = [float(comps[i]["reward"]) for i in idx]
        for s in range(0, 8, k):
            groups.append((pid, rewards[s:s + k]))
    return groups


def group_metrics(groups):
    r = torch.tensor([x for _, g in groups for x in g])
    gid = torch.tensor([i for i, (_, g) in enumerate(groups) for _ in g])
    adv = group_relative_advantages(r, gid, eps=STD_TOL)
    stds = np.array([np.std(g) for _, g in groups])
    centered = np.concatenate([np.array(g) - np.mean(g) for _, g in groups])
    return {
        "n_groups": len(groups),
        "n_generations": int(r.numel()),
        "informative_rate": float(np.mean(stds > STD_TOL)),
        "uninformative_frac": float(np.mean(stds <= STD_TOL)),
        "group_std_mean": float(stds.mean()),
        "adv_var": float(adv.var(unbiased=False)),
        "centered_var": float(centered.var()),
        "nonzero_adv_frac": float((adv.abs() > 1e-6).float().mean()),
    }


def difficulty_bins(by_prompt):
    means = {pid: float(np.mean([c["reward"] for c in comps[:8]])) for pid, comps in by_prompt.items()}
    vals = np.array(sorted(means.values()))
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    bins = {pid: ("hard" if m <= q1 else "medium" if m <= q2 else "easy") for pid, m in means.items()}
    return bins, means, (float(q1), float(q2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    ks = [int(k) for k in cfg["group_sizes"]]
    bins, means, cuts = difficulty_bins(by_prompt)
    rng = np.random.default_rng(int(cfg["seed"]))
    perms = [{pid: rng.permutation(8) for pid in by_prompt} for _ in range(N_PERM)]

    rows, drows = [], []
    for k in ks:
        fixed = group_metrics(regroup_equal_generation_budget(by_prompt, k))
        perm = [group_metrics(regroup_equal_generation_budget(by_prompt, k, p)) for p in perms]
        row = {"K": k, **{f"fixed_{m}": v for m, v in fixed.items()}}
        for m in ["informative_rate", "group_std_mean", "adv_var", "centered_var", "nonzero_adv_frac"]:
            vals = np.array([x[m] for x in perm])
            row[f"{m}"] = float(vals.mean())
            row[f"{m}_sd_over_partitions"] = float(vals.std())
        rows.append(row)
        for b in ["hard", "medium", "easy"]:
            sub = {pid: c for pid, c in by_prompt.items() if bins[pid] == b}
            fixed_b = group_metrics(regroup_equal_generation_budget(sub, k))
            perm_b = [group_metrics(regroup_equal_generation_budget(sub, k, {pid: p[pid] for pid in sub})) for p in perms]
            drow = {"K": k, "difficulty": b, "n_prompts": len(sub),
                    "mean_prompt_reward": float(np.mean([means[pid] for pid in sub])),
                    "n_groups": fixed_b["n_groups"]}
            for m in ["informative_rate", "group_std_mean", "adv_var", "centered_var"]:
                drow[m] = float(np.mean([x[m] for x in perm_b]))
                drow[f"fixed_{m}"] = fixed_b[m]
            drows.append(drow)

    res = repo_path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    df, dd = pd.DataFrame(rows), pd.DataFrame(drows)
    df.to_csv(res / "group_size.csv", index=False)
    dd.to_csv(res / "group_size_by_difficulty.csv", index=False)
    all_r = np.array([c["reward"] for comps in by_prompt.values() for c in comps])
    save_json(res / "group_size.json", {
        "cache": cfg["group_cache"], "n_prompts": len(by_prompt), "generations": int(all_r.size),
        "std_tolerance": STD_TOL, "n_random_partitions": N_PERM, "seed": cfg["seed"],
        "difficulty_rule": "tercile of per-prompt mean reward over its 8 cached completions (hard = lowest)",
        "difficulty_cutpoints": cuts, "prompt_difficulty": {pid: {"mean_reward": means[pid], "bin": bins[pid]} for pid in by_prompt},
        "reward_mean": float(all_r.mean()), "reward_std": float(all_r.std()),
        "exact_duplicate_rewards_within_prompt": int(sum(len(c) - len({x["reward"] for x in c}) for c in by_prompt.values())),
    })
    pd.set_option("display.width", 250)
    print(df[["K", "n_groups" if "n_groups" in df else "fixed_n_groups", "informative_rate", "group_std_mean", "adv_var",
              "centered_var", "nonzero_adv_frac"]].round(4).to_string(index=False))
    print(dd[["K", "difficulty", "n_prompts", "mean_prompt_reward", "informative_rate", "group_std_mean", "centered_var"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
