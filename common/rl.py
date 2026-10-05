"""Shared helpers for the online RL tasks (Task 2 PPO, Task 3 GRPO).

Kept deliberately small: prompt-pool handling, rollout generation, reward scoring, token-level
statistics and the critic forward pass. The PPO / GRPO objectives themselves stay in
task2_ppo/ppo.py and task3_grpo/grpo.py.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from common.data import chat_prompt_ids, prompt_messages, read_jsonl
from common.generation import batch_generate, score_reward_pairs


# ------------------------------------------------------------------ prompts

def load_prompt_pool(path, tokenizer, max_prompt_length, shuffle_seed=None):
    """Rows whose rendered chat prompt fits in max_prompt_length, in a fixed order.

    batch_generate truncates over-long prompts from the right (cutting the assistant header),
    so such prompts are excluded instead. With shuffle_seed the kept rows are shuffled once with
    a private RNG, giving every run/fork the identical prompt sequence.
    Returns (kept_rows, dropped_prompt_ids).
    """
    rows = read_jsonl(path)
    kept, dropped = [], []
    for r in rows:
        n = len(chat_prompt_ids(tokenizer, prompt_messages(r)))
        (kept if n <= int(max_prompt_length) else dropped).append(r)
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed)).shuffle(kept)
    return kept, [str(r.get("prompt_id")) for r in dropped]


def prompts_for_update(pool, update_idx, per_update):
    start = update_idx * per_update
    return [pool[(start + j) % len(pool)] for j in range(per_update)]


# ------------------------------------------------------------------ generation

def generate(model, tokenizer, messages_list, cfg, max_new_tokens, max_prompt_length, do_sample=None):
    """Sample responses with the course decoding settings; KV cache forced on for speed."""
    g = cfg["generation"]
    old_cfg = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    try:
        out = batch_generate(
            model, tokenizer, messages_list,
            max_prompt_length=int(max_prompt_length),
            max_new_tokens=int(max_new_tokens),
            temperature=float(g["temperature"]), top_p=float(g["top_p"]),
            do_sample=bool(g["do_sample"] if do_sample is None else do_sample),
        )
    finally:
        model.config.use_cache = old_cfg
    # batch_generate runs under torch.inference_mode(); clone so autograd may use these tensors.
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        out[k] = out[k].clone()
    return out


def disable_dropout(model):
    """PPO/GRPO ratios must equal 1 before the first update; LoRA dropout would break that."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


# ------------------------------------------------------------------ token statistics

def token_logprobs_and_entropy(model, sequences, attention_mask, prompt_width, response_ids, with_entropy=True):
    """Log-prob of each sampled response token and the full-distribution entropy at that step."""
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[:, prompt_width - 1: -1, :][:, : response_ids.shape[1], :].float()
    logp_all = F.log_softmax(logits, dim=-1)
    logp = torch.gather(logp_all, -1, response_ids.unsqueeze(-1)).squeeze(-1)
    ent = None
    if with_entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)
    return logp, ent


def masked_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def per_seq_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


# ------------------------------------------------------------------ rewards

@torch.no_grad()
def score_responses(rm, rtok, messages_list, responses, terminated, missing_eos_penalty=0.0, max_length=1280):
    """Raw RM score and effective terminal reward (raw - penalty when the response never ended)."""
    rtok.truncation_side = "left"  # keep the response if the pair is too long
    raw = score_reward_pairs(rm, rtok, messages_list, responses, max_length=int(max_length)).float().cpu()
    eff = raw - float(missing_eos_penalty) * (~torch.as_tensor(terminated, dtype=torch.bool)).float()
    return raw, eff


# ------------------------------------------------------------------ critic

def critic_values(value_model, sequences, attention_mask, prompt_width, n_resp):
    """V(s_t) for each response position t (state = prompt + response tokens before t).

    Uses the backbone's last hidden state and the scalar head; hidden states are cast to the
    head's dtype so an fp32 trainable head works on an fp16 backbone.
    """
    inner = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    backbone = getattr(inner, inner.base_model_prefix)
    hidden = backbone(input_ids=sequences, attention_mask=attention_mask, use_cache=False,
                      return_dict=True).last_hidden_state
    head = inner.score if hasattr(inner, "score") else inner.classifier
    params = list(head.parameters())
    w = next((p for p in params if p.requires_grad), params[0])  # the active (trainable) head copy
    vals = head(hidden.to(w.dtype)).squeeze(-1).float()
    return vals[:, prompt_width - 1: prompt_width - 1 + n_resp]


def explained_variance(pred, target, mask):
    m = mask.bool()
    if m.sum() < 2:
        return float("nan")
    p, t = pred[m].float(), target[m].float()
    var_t = t.var(unbiased=False)
    if var_t <= 1e-12:
        return float("nan")
    return float(1 - (t - p).var(unbiased=False) / var_t)


def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def device_of(model):
    return next(model.parameters()).device
