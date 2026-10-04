from __future__ import annotations

import argparse
import json
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


# ---------------------------------------------------------------------------------------------
# Shared DPO helpers (also imported by evaluate.py / ablate_beta.py / analyze_length.py)
# ---------------------------------------------------------------------------------------------

def filter_fitting_rows(rows, tokenizer, max_length):
    """Drop pairs whose *prompt alone* does not fit in max_length.

    common.data.encode_prompt_response (course update of 3 Oct) keeps the prompt intact and
    raises if it cannot fit. We apply the same rule up-front, deterministically, and return the
    excluded prompt_ids so they can be logged. Responses that are too long are right-truncated by
    encode_prompt_response itself.
    """
    kept, dropped = [], []
    for row in rows:
        n = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        (kept if n < max_length else dropped).append(row)
    return kept, [str(r.get("prompt_id", r.get("source_index"))) for r in dropped]


def make_collate(tokenizer, max_length):
    """Return ONE padded batch of 2B sequences: rows [0, B) are chosen, rows [B, 2B) rejected.

    Running chosen and rejected through the model together halves the number of forward passes.
    """
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        batch = pad_batch(tokenizer, chosen + rejected)
        batch["n_pairs"] = len(rows)
        batch["prompt_ids"] = [str(r.get("prompt_id")) for r in rows]
        return batch
    return collate


def model_device(model):
    return next(model.parameters()).device


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def pair_logps(model, batch):
    """Summed response-token log-probs for the policy and the frozen reference.

    The reference is the same network with the LoRA adapter disabled (common.models.reference_mode),
    so no second copy of the 1.5B model is needed. Returns tensors of shape [B] each:
    policy_chosen, policy_rejected, ref_chosen, ref_rejected (ref ones carry no gradient).
    """
    b = batch["n_pairs"]
    with torch.no_grad(), reference_mode(model):
        ref_logp, _, _ = response_sequence_logprobs(model, batch)
    pol_logp, _, _ = response_sequence_logprobs(model, batch)
    return pol_logp[:b], pol_logp[b:], ref_logp[:b].detach(), ref_logp[b:].detach()


def peak_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


# ---------------------------------------------------------------------------------------------

def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        # Short-run forks use the first N rows of the fixed file (same rows for every beta).
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    rows, dropped = filter_fitting_rows(rows, tokenizer, max_len)

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    generator = torch.Generator().manual_seed(seed)  # fixes the shuffle order across runs
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer, max_len),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_prompt_ids": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"]) / "train" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    device = model_device(model)
    params = trainable_parameters(model)
    # Base weights are fp16; LoRA weights are kept in fp32 by PEFT. A GradScaler protects the
    # fp16 backward pass through the frozen layers from gradient underflow.
    use_scaler = device.type == "cuda" and next(model.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    run_meta = {
        "run_name": run_name,
        "dataset_path": bundle["dataset_path"],
        "n_train_pairs": len(bundle["rows"]),
        "dropped_prompt_too_long": bundle["dropped_prompt_ids"],
        "train_prompt_ids": [str(r.get("prompt_id")) for r in bundle["rows"]],
        "beta": beta,
        "max_examples": max_examples,
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "effective_batch": int(cfg["batch_size"]) * accum,
        "learning_rate": float(cfg["learning_rate"]),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "lora": cfg["lora"],
        "seed": int(cfg["seed"]),
        "base_model": cfg["base_model"],
        "output": str(output),
    }
    save_json(results_dir / "run_config.json", run_meta)
    print(f"[DPO {run_name}] beta={beta} pairs={len(bundle['rows'])} dropped={len(bundle['dropped_prompt_ids'])} "
          f"micro-batches/epoch={len(loader)} optimizer-steps/epoch~{-(-len(loader)//accum)}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window = []  # per-micro-batch stats accumulated over one optimizer step
    opt_step = 0
    n_micro = len(loader)

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            pc, pr, rc, rr = pair_logps(model, batch)
            loss, diag = dpo_loss(pc, pr, rc, rr, beta)
            scaler.scale(loss / accum).backward()

            with torch.no_grad():
                margin = (pc - rc) - (pr - rr)
                window.append({
                    "loss": float(loss),
                    "acc": float((margin > 0).float().mean()),
                    "margin": float(margin.mean()),
                    "reward_chosen": float(beta * (pc - rc).mean()),
                    "reward_rejected": float(beta * (pr - rr).mean()),
                    "policy_chosen_logp": float(pc.mean()),
                    "policy_rejected_logp": float(pr.mean()),
                })

            last = (i + 1) == n_micro
            if (i + 1) % accum == 0 or last:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                rec = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
                rec.update({
                    "epoch": epoch,
                    "opt_step": opt_step,
                    "micro_step": i + 1,
                    "pairs_seen": (epoch * n_micro + i + 1) * int(cfg["batch_size"]),
                    "grad_norm": float(grad_norm),
                    "elapsed_s": time.perf_counter() - t0,
                    "peak_vram_gb": peak_vram_gb(),
                })
                append_jsonl(log_path, rec)
                window = []
                if opt_step % 5 == 0 or last:
                    print(f"  step {opt_step:4d} | loss {rec['loss']:.4f} | acc {rec['acc']:.3f} | "
                          f"margin {rec['margin']:+.3f} | gnorm {rec['grad_norm']:.3f} | {rec['elapsed_s']/60:.1f} min")

    model.save_pretrained(str(output))
    bundle["tokenizer"].save_pretrained(str(output))
    summary = dict(run_meta)
    summary.pop("train_prompt_ids")
    summary.update({
        "optimizer_steps": opt_step,
        "wall_clock_s": time.perf_counter() - t0,
        "peak_vram_gb": peak_vram_gb(),
    })
    save_json(results_dir / "train_summary.json", summary)
    save_json(output / "dpo_run_config.json", summary)
    print(f"[DPO {run_name}] saved adapter -> {output} ({summary['wall_clock_s']/60:.1f} min)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
