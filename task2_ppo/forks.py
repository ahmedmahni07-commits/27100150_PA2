"""Matched short PPO forks shared by the clipping study and the KL-pressure study.

Every fork starts from the identical supplied midpoint policy + critic, uses the same seeded
prompt sequence, `fork_updates` updates and the same generation cap; only (clip_epsilon, kl_beta)
differ. The (eps=0.20, beta=0.10) fork belongs to both studies and is trained once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate


def fork_name(eps, beta):
    return f"fork_eps{float(eps):g}_kl{float(beta):g}"


def run_fork(config_path, eps, beta, n_prompts=64, force=False):
    cfg = load_yaml(config_path)
    name = fork_name(eps, beta)
    adapter = str(Path(cfg["output"]).parent / name)
    if force or not (repo_path(adapter) / "adapter_config.json").exists():
        run_ppo(config_path, output=adapter, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name)
    if force or not (repo_path(cfg["results_dir"]) / "eval" / name / "summary.json").exists():
        evaluate(config_path, adapter, name, n_prompts=n_prompts)
    return name


def read_updates(cfg, name):
    p = repo_path(cfg["results_dir"]) / "train" / name / "update_log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def stability_stats(updates):
    """Optimisation-stability statistics over a run's updates (definitions in README)."""
    pl = [u["policy_loss"] for u in updates]
    rew = [u["reward_effective"] for u in updates]
    return {
        "mean_clip_frac": float(np.mean([u["clip_frac"] for u in updates])),
        "mean_affected_frac": float(np.mean([u["affected_frac"] for u in updates])),
        "max_ratio": float(np.max([u["ratio_max"] for u in updates])),
        "min_ratio": float(np.min([u["ratio_min"] for u in updates])),
        "max_update_kl": float(np.max([u["approx_kl_update"] for u in updates])),
        "mean_update_kl": float(np.mean([u["approx_kl_update"] for u in updates])),
        "max_grad_norm": float(np.max([u["grad_norm_policy_max"] for u in updates])),
        "policy_loss_std": float(np.std(pl)),
        "train_reward_mean": float(np.mean(rew)),
        "train_kl_last": float(updates[-1]["kl_token_mean"]),
        "train_entropy_mean": float(np.mean([u["entropy"] for u in updates])),
        "value_loss_mean": float(np.mean([u["value_loss"] for u in updates])),
    }


def fork_row(cfg, name, **extra):
    s = load_json(repo_path(cfg["results_dir"]) / "eval" / name / "summary.json")
    row = dict(extra)
    row.update({
        "run": name,
        "heldout_reward": s["reward_effective"]["mean"], "heldout_reward_se": s["reward_effective"]["se"],
        "heldout_reward_raw": s["reward_raw"]["mean"],
        "heldout_kl_token": s["kl_token_mean"], "heldout_entropy": s["entropy_token_mean"],
        "heldout_len_mean": s["length_tokens"]["mean"], "heldout_len_std": s["length_tokens"]["std"],
        "heldout_truncated_frac": s["truncated_frac"],
    })
    tp = repo_path(cfg["results_dir"]) / "train" / name / "update_log.jsonl"
    if tp.exists():
        row.update(stability_stats(read_updates(cfg, name)))
        ts = load_json(repo_path(cfg["results_dir"]) / "train" / name / "train_summary.json")
        row.update({"generated_tokens": ts["generated_tokens_total"], "wall_clock_s": ts["wall_clock_s"]})
    return row
