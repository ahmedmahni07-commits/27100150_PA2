"""Task 2 Step 2: clipping study.

Part A (cached batch, no new sampling): rebuild the fixed 32-rollout batch from
cached/ppo_rollout.pt (response text -> tokens, + EOS when the rollout terminated), compute
KL-shaped rewards from the cached old/reference log-probs, GAE from the cached critic values,
and whitened advantages. Then, for each epsilon, start from the supplied midpoint policy and run
`ppo_epochs` full-batch PPO passes on this identical batch, measuring after every pass:
  clip fraction     = share of valid tokens with ratio outside [1-eps, 1+eps]
  affected fraction = share of valid tokens where the clipped branch is the active one in
                      min(rho*A, clip(rho)*A) (ratio > 1+eps with A > 0, or < 1-eps with A < 0)
  unclipped and clipped surrogate values, ratio quantiles.
Pass 0 (before any step) checks that the midpoint reproduces the cached old log-probs.

Part B: matched short forks (task2_ppo.forks) for each epsilon at the configured kl_beta,
evaluated with the common protocol. Writes results/task2_ppo/clipping_cached.json/.csv and
clipping_forks.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import chat_prompt_ids, load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from common.rl import device_of, disable_dropout, token_logprobs_and_entropy
from task2_ppo.forks import fork_name, fork_row, run_fork
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

MICRO = 2


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def build_cached_batch(cfg, tok, rows):
    """Left-pad prompts / right-pad responses so every response starts at the same column."""
    by_id = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            by_id.setdefault(str(r["prompt_id"]), r)
    prompts, resps = [], []
    for r in rows:
        p = chat_prompt_ids(tok, prompt_messages(by_id[str(r["prompt_id"])]))
        ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if r.get("terminated_with_eos"):
            ids = ids + [tok.eos_token_id]
        n = int(r.get("response_tokens", len(ids)))
        if len(ids) != n:
            raise ValueError(f"Re-tokenised response length {len(ids)} != cached {n} for {r['prompt_id']}")
        prompts.append(p)
        resps.append(ids)
    pw, T, B = max(map(len, prompts)), max(map(len, resps)), len(rows)
    pad = tok.pad_token_id
    seq = torch.full((B, pw + T), pad, dtype=torch.long)
    attn = torch.zeros((B, pw + T), dtype=torch.long)
    rid = torch.full((B, T), pad, dtype=torch.long)
    mask = torch.zeros((B, T))
    old, ref, val = torch.zeros((B, T)), torch.zeros((B, T)), torch.zeros((B, T))
    for b, (p, rsp, r) in enumerate(zip(prompts, resps, rows)):
        n = len(rsp)
        seq[b, pw - len(p): pw] = torch.tensor(p)
        seq[b, pw: pw + n] = torch.tensor(rsp)
        attn[b, pw - len(p): pw + n] = 1
        rid[b, :n] = torch.tensor(rsp)
        mask[b, :n] = 1
        old[b, :n] = r["old_logprobs"].float()
        ref[b, :n] = r["ref_logprobs"].float()
        val[b, :n] = r["values"].float()
    reward = torch.tensor([float(r["effective_terminal_reward"]) for r in rows])
    return {"seq": seq, "attn": attn, "pw": pw, "rid": rid, "mask": mask, "old": old, "ref": ref,
            "values": val, "reward": reward}


def batch_logprobs(policy, bt, dev, grad=False):
    outs = []
    for s in range(0, bt["seq"].shape[0], MICRO):
        sl = slice(s, s + MICRO)
        with torch.set_grad_enabled(grad):
            lp, _ = token_logprobs_and_entropy(policy, bt["seq"][sl].to(dev), bt["attn"][sl].to(dev), bt["pw"],
                                               bt["rid"][sl].to(dev), with_entropy=False)
        outs.append(lp)
    return torch.cat(outs, 0)


def ratio_stats(new_lp, bt, adv, eps_list):
    mask = bt["mask"].bool()
    ratio = torch.exp(new_lp.cpu() - bt["old"])
    r = ratio[mask]
    out = {"ratio_q01": float(r.quantile(0.01)), "ratio_q50": float(r.quantile(0.5)), "ratio_q99": float(r.quantile(0.99)),
           "ratio_max": float(r.max()), "ratio_min": float(r.min()),
           "mean_abs_logratio": float((new_lp.cpu() - bt["old"])[mask].abs().mean())}
    a = adv[mask]
    for e in eps_list:
        outside = (r < 1 - e) | (r > 1 + e)
        affected = ((r > 1 + e) & (a > 0)) | ((r < 1 - e) & (a < 0))
        unclipped = (r * a).mean()
        clipped = torch.minimum(r * a, r.clamp(1 - e, 1 + e) * a).mean()
        out[f"eps{e:g}"] = {"clip_frac": float(outside.float().mean()), "affected_frac": float(affected.float().mean()),
                            "surrogate_unclipped": float(unclipped), "surrogate_clipped": float(clipped)}
    return out


def cached_study(config_path):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    eps_list = [float(e) for e in cfg["clip_values"]]
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    bt = build_cached_batch(cfg, tok, rows)
    mask = bt["mask"]
    rewards = shaped_rewards(bt["reward"], bt["old"], bt["ref"], mask, float(cfg["kl_beta"]))
    adv, ret = compute_gae(rewards, bt["values"], mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
    adv_n = normalize_advantages(adv, mask)

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    dev = device_of(policy)
    init_state = {k: v.detach().clone() for k, v in policy.state_dict().items() if "lora_" in k}
    params = trainable_parameters(policy)
    n_tok = float(mask.sum())

    result = {"n_rollouts": len(rows), "n_valid_tokens": int(n_tok), "kl_beta": float(cfg["kl_beta"]),
              "mean_effective_reward": float(bt["reward"].mean()),
              "advantage_raw_std": float(adv[mask.bool()].std()), "runs": {}}
    flat = []
    for eps in eps_list:
        policy.load_state_dict(init_state, strict=False)
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        passes = []
        with torch.no_grad():
            lp0 = batch_logprobs(policy, bt, dev)
        passes.append({"pass": 0, **ratio_stats(lp0, bt, adv_n, eps_list)})
        for e in range(int(cfg["ppo_epochs"])):
            # One optimizer step per pass over the full cached batch (token-weighted micro-batches).
            for s in range(0, bt["seq"].shape[0], MICRO):
                sl = slice(s, s + MICRO)
                lp, _ = token_logprobs_and_entropy(policy, bt["seq"][sl].to(dev), bt["attn"][sl].to(dev), bt["pw"],
                                                   bt["rid"][sl].to(dev), with_entropy=False)
                m = mask[sl].to(dev)
                loss, _, _ = ppo_policy_loss(lp, bt["old"][sl].to(dev), adv_n[sl].to(dev), m, eps)
                scaler.scale(loss * (float(m.sum()) / n_tok)).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                lpe = batch_logprobs(policy, bt, dev)
            passes.append({"pass": e + 1, "grad_norm": gn, **ratio_stats(lpe, bt, adv_n, eps_list)})
        result["runs"][f"eps{eps:g}"] = passes
        for p in passes:
            for e2 in eps_list:
                flat.append({"trained_with_eps": eps, "pass": p["pass"], "measured_at_eps": e2,
                             "mean_abs_logratio": p["mean_abs_logratio"], "ratio_q01": p["ratio_q01"],
                             "ratio_q99": p["ratio_q99"], **p[f"eps{e2:g}"]})
        own = passes[-1][f"eps{eps:g}"]
        print(f"  eps={eps:g}: pass0 |logratio|={passes[0]['mean_abs_logratio']:.2e} | after {cfg['ppo_epochs']} passes: "
              f"clip_frac={own['clip_frac']:.3f} affected={own['affected_frac']:.3f} "
              f"surrogate unclipped={own['surrogate_unclipped']:+.4f} clipped={own['surrogate_clipped']:+.4f}")
    res = repo_path(cfg["results_dir"])
    save_json(res / "clipping_cached.json", result)
    pd.DataFrame(flat).to_csv(res / "clipping_cached.csv", index=False)
    del policy
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=64)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    if not args.skip_cached:
        print("[clipping] Part A: cached-rollout study")
        cached_study(args.config)
    if not args.skip_forks:
        print("[clipping] Part B: matched short forks")
        beta = float(cfg["kl_beta"])
        rows = []
        for eps in cfg["clip_values"]:
            name = run_fork(args.config, eps, beta, n_prompts=args.n_prompts)
            rows.append(fork_row(cfg, name, clip_epsilon=float(eps), kl_beta=beta))
        df = pd.DataFrame(rows)
        df.to_csv(res / "clipping_forks.csv", index=False)
        pd.set_option("display.width", 250)
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
