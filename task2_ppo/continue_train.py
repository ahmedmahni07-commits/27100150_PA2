"""Task 2 Step 1: PPO continuation from the supplied midpoint (also used for the short forks).

One update =
  1. take the next `prompts_per_update` prompts from the fixed, seeded prompt order;
  2. sample a response with the current policy (course decoding, cap = max_response_length);
  3. score it with the frozen reward model; subtract `missing_eos_penalty` if it never ended;
  4. compute old-policy, reference (adapter off) and critic values on the sampled tokens;
  5. KL-shaped token rewards (task2_ppo.ppo.shaped_rewards) -> GAE advantages and returns;
  6. `ppo_epochs` passes of the clipped policy loss and the critic MSE loss, each optimizer
     stepped separately with grad-norm clipping.
Every update is logged to results/task2_ppo/train/<run>/update_log.jsonl and every rollout to
rollouts.jsonl. Dropout is disabled so the importance ratio is exactly 1 at the first pass.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, repo_path
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl import (
    critic_values,
    device_of,
    disable_dropout,
    explained_variance,
    generate,
    load_prompt_pool,
    masked_mean,
    peak_vram_gb,
    prompts_for_update,
    score_responses,
    token_logprobs_and_entropy,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"],
                                   train_mode=cfg.get("value_train_mode", "head_only"))
    # Trainable critic weights in fp32 (the copied score head arrives in fp16); optimizer state
    # is created lazily, so changing .data here keeps the same Parameter objects.
    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    disable_dropout(policy)
    disable_dropout(value_model)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts, dropped = load_prompt_pool(cfg["paths"]["rl_prompt_train"], tokenizer,
                                        cfg["max_prompt_length"], shuffle_seed=cfg["seed"])

    policy_optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    value_optimizer = AdamW(
        value_parameter_groups(value_model, lora_lr=float(cfg["value_lora_learning_rate"]),
                               head_lr=float(cfg["value_head_learning_rate"])),
        weight_decay=0.0,
    )
    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy, "value_model": value_model,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts, "dropped_prompt_ids": dropped,
        "policy_optimizer": policy_optimizer, "value_optimizer": value_optimizer,
    }


def _clip_step(params, optimizer, scaler, max_norm):
    scaler.unscale_(optimizer)
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(gn)


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None,
            kl_beta: float | None = None, run_name: str = "standard"):
    t_load = time.perf_counter()
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"]) / "train" / run_name
    res.mkdir(parents=True, exist_ok=True)
    for f in ("update_log.jsonl", "rollouts.jsonl"):
        if (res / f).exists():
            (res / f).unlink()

    tok, policy, vm = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rtok = bundle["reward_model"], bundle["reward_tokenizer"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    pparams, vparams = trainable_parameters(policy), trainable_parameters(vm)
    dev = device_of(policy)
    use_scaler = dev.type == "cuda"
    pscaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    vscaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    n_updates, per_update, n_epochs = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["ppo_epochs"])
    max_norm = float(cfg["max_grad_norm"])
    meta = {
        "run_name": run_name, "updates": n_updates, "prompts_per_update": per_update, "ppo_epochs": n_epochs,
        "clip_epsilon": eps, "kl_beta": beta, "gamma": cfg["gamma"], "gae_lambda": cfg["gae_lambda"],
        "value_coef": cfg["value_coef"], "missing_eos_penalty": cfg["missing_eos_penalty"],
        "policy_learning_rate": cfg["policy_learning_rate"], "value_lora_learning_rate": cfg["value_lora_learning_rate"],
        "value_head_learning_rate": cfg["value_head_learning_rate"], "max_response_length": cfg["max_response_length"],
        "max_prompt_length": cfg["max_prompt_length"], "seed": cfg["seed"],
        "start_policy": cfg["paths"]["ppo_midpoint_policy"], "start_value": cfg["paths"]["ppo_midpoint_value"],
        "prompt_pool_size": len(bundle["prompt_rows"]),
        "dropped_prompt_too_long": bundle["dropped_prompt_ids"],
        "prompt_ids_used": [str(r["prompt_id"]) for u in range(n_updates)
                            for r in prompts_for_update(bundle["prompt_rows"], u, per_update)],
        "output": str(out), "load_seconds": time.perf_counter() - t_load,
    }
    save_json(res / "run_config.json", meta)
    print(f"[PPO {run_name}] updates={n_updates} eps={eps} kl_beta={beta} pool={len(bundle['prompt_rows'])} "
          f"(dropped {len(bundle['dropped_prompt_ids'])} over-long prompts)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gen_tokens = 0

    for u in range(n_updates):
        rows = prompts_for_update(bundle["prompt_rows"], u, per_update)
        msgs = [prompt_messages(r) for r in rows]

        # ---- rollout
        g = generate(policy, tok, msgs, cfg, cfg["max_response_length"], cfg["max_prompt_length"])
        seq, attn, pw = g["sequences"], g["attention_mask"], g["prompt_width"]
        rid, mask = g["response_ids"], g["response_mask"].float()
        gen_tokens += int(mask.sum())
        with torch.no_grad():
            old_lp, ent = token_logprobs_and_entropy(policy, seq, attn, pw, rid)
            with reference_mode(policy):
                ref_lp, _ = token_logprobs_and_entropy(policy, seq, attn, pw, rid, with_entropy=False)
            values = critic_values(vm, seq, attn, pw, rid.shape[1]) * mask
        raw, eff = score_responses(rm, rtok, msgs, g["responses"], g["terminated_with_eos"],
                                   cfg["missing_eos_penalty"], cfg.get("reward_max_length", 1280))
        rewards = shaped_rewards(eff.to(dev), old_lp, ref_lp, mask, beta)
        adv, ret = compute_gae(rewards, values, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
        adv_n = normalize_advantages(adv, mask)
        ev_before = explained_variance(values, ret, mask)

        # ---- optimisation
        ep = []
        for e in range(n_epochs):
            new_lp, new_ent = token_logprobs_and_entropy(policy, seq, attn, pw, rid, with_entropy=False)
            ploss, ratio, clip_frac = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            pscaler.scale(ploss).backward()
            pgn = _clip_step(pparams, popt, pscaler, max_norm)

            v_new = critic_values(vm, seq, attn, pw, rid.shape[1])
            vloss = value_mse_loss(v_new, ret, mask)
            vscaler.scale(float(cfg["value_coef"]) * vloss).backward()
            vgn = _clip_step(vparams, vopt, vscaler, max_norm)

            with torch.no_grad():
                affected = (((ratio > 1 + eps) & (adv_n > 0)) | ((ratio < 1 - eps) & (adv_n < 0))).float()
                lr_ = torch.log(ratio.clamp_min(1e-12))
                ep.append({
                    "policy_loss": float(ploss), "value_loss": float(vloss),
                    "clip_frac": float(clip_frac), "affected_frac": float(masked_mean(affected, mask)),
                    "ratio_max": float((ratio * mask).max()), "ratio_min": float(torch.where(mask > 0, ratio, torch.ones_like(ratio)).min()),
                    "approx_kl_old_new": float(masked_mean(-lr_, mask)),
                    "grad_norm_policy": pgn, "grad_norm_value": vgn,
                })

        with torch.no_grad():
            after_lp, _ = token_logprobs_and_entropy(policy, seq, attn, pw, rid, with_entropy=False)
            v_after = critic_values(vm, seq, attn, pw, rid.shape[1])
        kl_tok = float(masked_mean(old_lp - ref_lp, mask))
        rec = {
            "update": u + 1,
            "prompt_ids": [str(r["prompt_id"]) for r in rows],
            "reward_raw": float(raw.mean()), "reward_effective": float(eff.mean()),
            "kl_token_mean": kl_tok, "kl_seq_sum": float(((old_lp - ref_lp) * mask).sum(-1).mean()),
            "entropy": float(masked_mean(ent, mask)),
            "response_len": float(mask.sum(-1).mean()),
            "truncated_frac": float(np.mean(g["truncated"])),
            "value_mean": float(masked_mean(values, mask)), "return_mean": float(masked_mean(ret, mask)),
            "value_first_token": float(values[:, 0].mean()), "return_first_token": float(ret[:, 0].mean()),
            "explained_var_before": ev_before, "explained_var_after": explained_variance(v_after, ret, mask),
            "adv_raw_std": float(adv[mask.bool()].std(unbiased=False)) if mask.sum() > 1 else 0.0,
            "policy_loss": float(np.mean([x["policy_loss"] for x in ep])),
            "value_loss": float(np.mean([x["value_loss"] for x in ep])),
            "clip_frac": float(np.mean([x["clip_frac"] for x in ep])),
            "affected_frac": float(np.mean([x["affected_frac"] for x in ep])),
            "clip_frac_last_epoch": ep[-1]["clip_frac"],
            "grad_norm_policy": float(np.mean([x["grad_norm_policy"] for x in ep])),
            "grad_norm_policy_max": float(np.max([x["grad_norm_policy"] for x in ep])),
            "grad_norm_value": float(np.mean([x["grad_norm_value"] for x in ep])),
            "ratio_max": float(np.max([x["ratio_max"] for x in ep])),
            "ratio_min": float(np.min([x["ratio_min"] for x in ep])),
            "approx_kl_update": float(masked_mean(old_lp - after_lp, mask)),
            "epochs": ep,
            "generated_tokens_total": gen_tokens,
            "elapsed_s": time.perf_counter() - t0, "peak_vram_gb": peak_vram_gb(),
        }
        append_jsonl(res / "update_log.jsonl", rec)
        for j, r in enumerate(rows):
            append_jsonl(res / "rollouts.jsonl", {
                "update": u + 1, "prompt_id": str(r["prompt_id"]), "prompt": msgs[j][-1]["content"],
                "response": g["responses"][j], "response_tokens": int(g["response_lengths"][j]),
                "terminated_with_eos": bool(g["terminated_with_eos"][j]), "truncated": bool(g["truncated"][j]),
                "reward_raw": float(raw[j]), "reward_effective": float(eff[j]),
            })
        print(f"  upd {u+1:3d} | R {rec['reward_effective']:+.3f} | KL {kl_tok:+.4f} | H {rec['entropy']:.3f} | "
              f"Lp {rec['policy_loss']:+.4f} | Lv {rec['value_loss']:.3f} | clip {rec['clip_frac']:.3f} | "
              f"gn {rec['grad_norm_policy']:.2f} | len {rec['response_len']:.0f} | {rec['elapsed_s']/60:.1f} min")

    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    value_out = out.parent / f"{out.name}_value"
    vm.save_pretrained(str(value_out))
    summary = {k: v for k, v in meta.items() if k not in ("prompt_ids_used", "dropped_prompt_too_long")}
    summary.update({
        "n_dropped_prompts": len(meta["dropped_prompt_too_long"]),
        "wall_clock_s": time.perf_counter() - t0, "peak_vram_gb": peak_vram_gb(),
        "generated_tokens_total": gen_tokens, "value_output": str(value_out),
    })
    save_json(res / "train_summary.json", summary)
    print(f"[PPO {run_name}] done in {summary['wall_clock_s']/60:.1f} min, peak VRAM {summary['peak_vram_gb']:.1f} GB -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
