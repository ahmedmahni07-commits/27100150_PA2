"""Task 2 Step 3: reward-overoptimisation / KL-pressure study.

Matched short forks from the identical PPO midpoint for kl_beta in configs/ppo.yaml:kl_values, at
the configured clip_epsilon, evaluated with the common held-out protocol. Also writes the per-update
trajectories of every fork side by side so you can see which quantity moves first.
Outputs results/task2_ppo/kl_forks.csv and kl_trajectories.csv.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from task2_ppo.forks import fork_row, read_updates, run_fork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--n-prompts", type=int, default=64)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    eps = float(cfg["clip_epsilon"])
    rows, traj = [], []
    for beta in cfg["kl_values"]:
        name = run_fork(args.config, eps, beta, n_prompts=args.n_prompts)
        rows.append(fork_row(cfg, name, clip_epsilon=eps, kl_beta=float(beta)))
        for u in read_updates(cfg, name):
            traj.append({"kl_beta": float(beta), "update": u["update"], "reward_effective": u["reward_effective"],
                         "reward_raw": u["reward_raw"], "kl_token_mean": u["kl_token_mean"], "entropy": u["entropy"],
                         "response_len": u["response_len"], "approx_kl_update": u["approx_kl_update"],
                         "clip_frac": u["clip_frac"], "value_loss": u["value_loss"]})
    res = repo_path(cfg["results_dir"])
    df = pd.DataFrame(rows)
    df.to_csv(res / "kl_forks.csv", index=False)
    pd.DataFrame(traj).to_csv(res / "kl_trajectories.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
