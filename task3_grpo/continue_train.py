"""Task 3 Step 1: GRPO continuation from the supplied midpoint (also used for the normalization forks).

One update =
  1. take the next `prompts_per_update` prompts from the fixed, seeded prompt order (same helper and
     order as PPO);
  2. sample K = num_generations completions per prompt with the current policy (cap
     max_completion_length);
  3. score every completion with the frozen reward model;
  4. group-relative advantages per prompt (task3_grpo.grpo.group_relative_advantages);
  5. completions that hit the cap without EOS are masked out of the loss (mask_truncated_completions)
     but still count in their group's reward statistics;
  6. `policy_epochs` passes of the corrected clipped GRPO loss + beta * KL(ref) with `loss_type`
     'grpo' (1/T_k per sequence) or 'dr_grpo' (constant 1/max_completion_length).
The loss is back-propagated one completion at a time (the per-completion terms add up exactly to
the batch loss of grpo_policy_loss), which lets us log the gradient norm contributed by each
completion together with its length: the length-conditioned statistic used in Step 3.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, repo_path
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.rl import (
    device_of,
    disable_dropout,
    generate,
    load_prompt_pool,
    masked_mean,
    peak_vram_gb,
    prompts_for_update,
    score_responses,
    token_logprobs_and_entropy,
)
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

STD_TOL = 1e-6  # same tolerance as group_relative_advantages' eps: std <= tol => uninformative group


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts, dropped = load_prompt_pool(cfg["paths"]["rl_prompt_train"], tokenizer,
                                        cfg["max_prompt_length"], shuffle_seed=cfg["seed"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts, "dropped_prompt_ids": dropped, "optimizer": optimizer,
    }


def _grad_vector(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().float() for p in params])


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"]) / "train" / run_name
    res.mkdir(parents=True, exist_ok=True)
    for f in ("update_log.jsonl", "rollouts.jsonl", "sequence_log.jsonl"):
        if (res / f).exists():
            (res / f).unlink()

    tok, policy, opt = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rtok = bundle["reward_model"], bundle["reward_tokenizer"]
    params = trainable_parameters(policy)
    dev = device_of(policy)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    K, per_update = int(cfg["num_generations"]), int(cfg["prompts_per_update"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    Lmax = int(cfg["max_completion_length"])
    n_updates, n_epochs = int(cfg["updates"]), int(cfg.get("policy_epochs", 1))
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))

    meta = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "prompts_per_update": per_update,
        "num_generations": K, "policy_epochs": n_epochs, "learning_rate": cfg["learning_rate"],
        "clip_epsilon": eps, "kl_beta": beta, "max_completion_length": Lmax, "max_prompt_length": cfg["max_prompt_length"],
        "mask_truncated_completions": mask_trunc, "max_grad_norm": cfg["max_grad_norm"], "seed": cfg["seed"],
        "start_policy": cfg["paths"]["grpo_midpoint_policy"], "prompt_pool_size": len(bundle["prompt_rows"]),
        "dropped_prompt_too_long": bundle["dropped_prompt_ids"],
        "prompt_ids_used": [str(r["prompt_id"]) for u in range(n_updates)
                            for r in prompts_for_update(bundle["prompt_rows"], u, per_update)],
        "uninformative_std_tolerance": STD_TOL, "output": str(out),
    }
    save_json(res / "run_config.json", meta)
    print(f"[GRPO {run_name}] loss_type={loss_type} updates={n_updates} K={K} eps={eps} beta={beta} "
          f"pool={len(bundle['prompt_rows'])} (dropped {len(bundle['dropped_prompt_ids'])})")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gen_tokens = 0

    for u in range(n_updates):
        rows = prompts_for_update(bundle["prompt_rows"], u, per_update)
        msgs = [prompt_messages(r) for r in rows for _ in range(K)]
        gids = torch.tensor([i for i in range(len(rows)) for _ in range(K)])

        g = generate(policy, tok, msgs, cfg, Lmax, cfg["max_prompt_length"])
        seq, attn, pw, rid = g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"]
        mask = g["response_mask"].float()
        gen_tokens += int(mask.sum())
        loss_mask = mask_truncated_sequences(mask, g["truncated"]) if mask_trunc else mask
        # One completion at a time: K x (prompt + 512) x vocab-sized fp32 logits would not fit on a T4.
        olds, ents, refs = [], [], []
        with torch.no_grad():
            for k in range(seq.shape[0]):
                sl = slice(k, k + 1)
                lp_k, ent_k = token_logprobs_and_entropy(policy, seq[sl], attn[sl], pw, rid[sl])
                olds.append(lp_k)
                ents.append(ent_k)
            with reference_mode(policy):
                for k in range(seq.shape[0]):
                    sl = slice(k, k + 1)
                    refs.append(token_logprobs_and_entropy(policy, seq[sl], attn[sl], pw, rid[sl], with_entropy=False)[0])
        old_lp, ent, ref_lp = torch.cat(olds), torch.cat(ents), torch.cat(refs)
        raw, _ = score_responses(rm, rtok, msgs, g["responses"], g["terminated_with_eos"], 0.0,
                                 cfg.get("reward_max_length", 1280))
        adv = group_relative_advantages(raw, gids).to(dev)

        group_std = [float(raw[gids == i].std(unbiased=False)) for i in range(len(rows))]
        n_seq = seq.shape[0]
        T_total = float(loss_mask.sum().clamp_min(1.0))
        lengths = mask.sum(-1)

        seq_recs, ep_stats = [], []
        for e in range(n_epochs):
            prev = torch.zeros(sum(p.numel() for p in params), device=dev)
            any_backward = False
            loss_total, pol_total, kl_total, clip_num, ratio_num = 0.0, 0.0, 0.0, 0.0, 0.0
            for k in range(n_seq):
                sl = slice(k, k + 1)
                new_lp, _ = token_logprobs_and_entropy(policy, seq[sl], attn[sl], pw, rid[sl], with_entropy=False)
                # Policy part with gradient (beta=0 isolates it) + diagnostics from the same helper.
                pol_k, diag = grpo_policy_loss(new_lp, old_lp[sl], adv[sl], loss_mask[sl], ref_lp[sl], eps, 0.0,
                                               loss_type=loss_type, max_completion_length=Lmax)
                lr_ = ref_lp[sl] - new_lp
                kl_k = masked_mean(torch.exp(lr_) - lr_ - 1.0, loss_mask[sl])  # same k3 estimator as grpo_policy_loss
                Tk = float(loss_mask[k].sum())
                # Exact share of this completion in the batch loss of grpo_policy_loss: the policy term is
                # averaged over the n_seq sequences, the KL term is a token mean over the whole batch.
                term = pol_k / n_seq + beta * kl_k * (Tk / T_total)
                diag["sampled_kl"] = kl_k.detach()
                if Tk > 0:
                    any_backward = True
                    scaler.scale(term).backward()
                    cur = _grad_vector(params) / scaler.get_scale()
                    gk = float((cur - prev).norm())
                    prev = cur
                else:
                    gk = 0.0
                loss_total += float(term.detach())
                pol_total += float(diag["policy_term"]) / n_seq
                kl_total += float(diag["sampled_kl"]) * Tk / T_total
                clip_num += float(diag["clip_fraction"]) * Tk
                ratio_num += float(diag["ratio_mean"]) * Tk
                if e == 0:
                    seq_recs.append({"update": u + 1, "group": int(gids[k]), "k": k, "length": int(lengths[k]),
                                     "truncated": bool(g["truncated"][k]), "in_loss": Tk > 0,
                                     "reward": float(raw[k]), "advantage": float(adv[k]), "grad_norm": gk,
                                     "token_weight": (abs(float(adv[k])) / (Tk if loss_type == "grpo" else Lmax)) if Tk > 0 else 0.0})
            if any_backward:
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                scaler.step(opt)
                scaler.update()
            else:
                # Every completion hit the cap and was masked out (mask_truncated_completions):
                # there is no loss term, so this update makes no parameter change.
                gn = 0.0
            opt.zero_grad(set_to_none=True)
            ep_stats.append({"skipped_step": not any_backward,"loss": loss_total, "policy_loss": pol_total, "kl_term": kl_total,
                             "clip_frac": clip_num / max(T_total, 1.0), "ratio_mean": ratio_num / max(T_total, 1.0),
                             "grad_norm": gn})

        rec = {
            "update": u + 1, "prompt_ids": [str(r["prompt_id"]) for r in rows],
            "reward_mean": float(raw.mean()), "reward_std_all": float(raw.std(unbiased=False)),
            "group_reward_std_mean": float(np.mean(group_std)),
            "uninformative_frac": float(np.mean([s <= STD_TOL for s in group_std])),
            "kl_token_mean": float(masked_mean(old_lp - ref_lp, mask)),
            "entropy": float(masked_mean(ent, mask)),
            "response_len": float(lengths.mean()), "response_len_std": float(lengths.std(unbiased=False)),
            "truncated_frac": float(np.mean(g["truncated"])), "masked_completions": int((loss_mask.sum(-1) == 0).sum()),
            "policy_loss": float(np.mean([x["policy_loss"] for x in ep_stats])),
            "loss": float(np.mean([x["loss"] for x in ep_stats])),
            "kl_term": float(np.mean([x["kl_term"] for x in ep_stats])),
            "clip_frac": float(np.mean([x["clip_frac"] for x in ep_stats])),
            "grad_norm": float(np.mean([x["grad_norm"] for x in ep_stats])),
            "skipped_step": bool(all(x["skipped_step"] for x in ep_stats)),
            "generated_tokens_total": gen_tokens,
            "elapsed_s": time.perf_counter() - t0, "peak_vram_gb": peak_vram_gb(),
        }
        append_jsonl(res / "update_log.jsonl", rec)
        for s in seq_recs:
            append_jsonl(res / "sequence_log.jsonl", s)
        for j in range(n_seq):
            append_jsonl(res / "rollouts.jsonl", {
                "update": u + 1, "prompt_id": str(rows[int(gids[j])]["prompt_id"]), "k": j,
                "response": g["responses"][j], "response_tokens": int(g["response_lengths"][j]),
                "truncated": bool(g["truncated"][j]), "reward": float(raw[j]), "advantage": float(adv[j]),
            })
        print(f"  upd {u+1:3d} | R {rec['reward_mean']:+.3f} | grp-std {rec['group_reward_std_mean']:.3f} | "
              f"uninf {rec['uninformative_frac']:.2f} | KL {rec['kl_token_mean']:+.4f} | H {rec['entropy']:.3f} | "
              f"L {rec['policy_loss']:+.4f} | gn {rec['grad_norm']:.2f} | len {rec['response_len']:.0f} | "
              f"masked {rec['masked_completions']}{' (step skipped: all masked)' if rec['skipped_step'] else ''} | {rec['elapsed_s']/60:.1f} min")

    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    summary = {k: v for k, v in meta.items() if k not in ("prompt_ids_used", "dropped_prompt_too_long")}
    summary.update({"n_dropped_prompts": len(meta["dropped_prompt_too_long"]),
                    "wall_clock_s": time.perf_counter() - t0, "peak_vram_gb": peak_vram_gb(),
                    "generated_tokens_total": gen_tokens})
    save_json(res / "train_summary.json", summary)
    print(f"[GRPO {run_name}] done in {summary['wall_clock_s']/60:.1f} min, peak VRAM {summary['peak_vram_gb']:.1f} GB -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
