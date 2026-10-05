"""Numerical checks for the three corrected objectives (run: python -m tests.test_objectives).

Each check compares the starter helper against a direct transcription of the equation in the
assignment manual on small hand-constructed tensors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import ppo_policy_loss, compute_gae
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss


def check_dpo():
    pc = torch.tensor([-10.0, -20.0, -5.0])
    pr = torch.tensor([-12.0, -18.0, -5.0])
    rc = torch.tensor([-11.0, -19.0, -6.0])
    rr = torch.tensor([-11.0, -19.0, -4.0])
    beta = 0.1
    m = (pc - rc) - (pr - rr)
    expected = -F.logsigmoid(beta * m).mean()
    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    assert torch.allclose(loss, expected), (loss, expected)
    assert torch.allclose(diag["preference_accuracy"], (m > 0).float().mean())
    # Policy == reference must give loss log(2) regardless of the data.
    loss0, _ = dpo_loss(rc, rr, rc, rr, beta)
    assert torch.allclose(loss0, torch.tensor(0.6931472), atol=1e-6), loss0
    print("DPO ok: loss", float(loss), "| policy==ref ->", float(loss0))


def check_ppo():
    old = torch.zeros(1, 4)
    new = torch.log(torch.tensor([[0.5, 1.0, 1.5, 1.5]]))  # ratios 0.5, 1, 1.5, 1.5
    adv = torch.tensor([[1.0, 1.0, 1.0, -1.0]])
    mask = torch.ones(1, 4)
    eps = 0.2
    loss, ratio, clip_frac = ppo_policy_loss(new, old, adv, mask, eps)
    r = torch.exp(new - old)
    expected = -torch.minimum(r * adv, r.clamp(1 - eps, 1 + eps) * adv).mean()
    assert torch.allclose(loss, expected), (loss, expected)
    # Hand values: min(0.5,0.8)=0.5, 1, min(1.5,1.2)=1.2, min(-1.5,-1.2)=-1.5 -> mean 0.3
    assert torch.allclose(loss, torch.tensor(-0.3)), loss
    assert torch.allclose(clip_frac, torch.tensor(0.75)), clip_frac
    # GAE sanity: gamma=lam=1, zero values -> advantage = reward-to-go
    rew = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    a, ret = compute_gae(rew, torch.zeros_like(rew), m, gamma=1.0, lam=1.0)
    assert torch.allclose(a, torch.tensor([[1.0, 1.0, 1.0, 0.0]])), a
    print("PPO ok: loss", float(loss), "clip_frac", float(clip_frac))


def check_grpo():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    gids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    adv = group_relative_advantages(rewards, gids)
    assert torch.allclose(adv[:4], torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-4), adv
    assert torch.all(adv[4:] == 0), "zero-variance group must give zero advantage"
    # Shifting one group's rewards must not change any advantage (invariance to prompt difficulty)
    adv2 = group_relative_advantages(rewards + torch.tensor([0, 0, 0, 0, 100, 100, 100, 100.0]), gids)
    assert torch.allclose(adv, adv2)
    # Loss: with ratio=1 and beta=0 the grpo loss is -mean_k A_k
    T = 3
    lp = torch.zeros(8, T)
    mask = torch.ones(8, T)
    loss, _ = grpo_policy_loss(lp, lp, adv, mask, lp, eps=0.2, beta=0.0)
    assert torch.allclose(loss, -adv.mean(), atol=1e-6)
    print("GRPO ok: advantages", adv.tolist())


def check_versions():
    import transformers, peft
    assert transformers.__version__ == "4.57.1", (
        f"transformers {transformers.__version__} installed; the course pins 4.57.1 "
        "(5.x changes apply_chat_template outputs used by the supplied judges). Re-run the pip install cell.")
    print("versions ok: transformers", transformers.__version__, "peft", peft.__version__)


if __name__ == "__main__":
    check_versions()
    check_dpo()
    check_ppo()
    check_grpo()
    print("All objective checks passed.")
