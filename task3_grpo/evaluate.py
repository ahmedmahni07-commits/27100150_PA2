"""Common held-out evaluation for GRPO conditions.

Uses exactly the PPO protocol (task2_ppo.evaluate): the same first 64 eval-pool prompts, one sampled
response each, the course decoding and seed, and the same 768-token cap as the PPO evaluation, so
PPO and GRPO held-out numbers are directly comparable. GRPO has no missing-EOS penalty, so
reward_effective == reward_raw here; truncation is reported separately.
Outputs results/task3_grpo/eval/<name>/{summary.json, generations.jsonl}.
"""
from __future__ import annotations

import argparse

from task2_ppo.evaluate import evaluate as _evaluate

EVAL_CAP = 768


def evaluate(config_path, adapter, name, n_prompts=64):
    return _evaluate(config_path, adapter, name, n_prompts=n_prompts, max_new_tokens=EVAL_CAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int, default=64)
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.n_prompts)


if __name__ == "__main__":
    main()
