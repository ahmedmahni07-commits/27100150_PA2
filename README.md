# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

---
# Student implementation notes (27100150)

Run everything from the repository root. GPU runs were done with `notebooks/run_on_gpu.ipynb`
(a thin driver that only calls the modules below; `outputs/` and `results/` are kept on persistent storage).

## Corrected objective defects (Tasks 1–3)

| File | Starter behaviour | Correction |
|---|---|---|
| `task1_dpo/dpo.py` | logit `beta * (policy_margin + ref_margin)` | `beta * (policy_margin - ref_margin)` (log-ratio margin relative to the reference) |
| `task2_ppo/ppo.py` | `torch.maximum(surr1, surr2)` | `torch.minimum(surr1, surr2)` (pessimistic clipped surrogate) |
| `task3_grpo/grpo.py` | mean/std over the whole batch, `group_ids` ignored | mean/std computed within each prompt group; zero-variance groups get zero advantage |

`python -m tests.test_objectives` checks each corrected helper against a direct transcription of
the manual's equation on hand-built tensors (the checks fail on the starter versions).

## Task 1 – DPO: exact commands

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard          # Step 1 training (1 epoch)
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft  # SFT reference point
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml                         # Step 2 (beta forks + table)
python -m task1_dpo.analyze_length --config configs/dpo.yaml                      # Step 3 (length study)
python -m task1_dpo.qualitative --config configs/dpo.yaml                         # candidate examples
```

Implementation choices (identical for every Task 1 condition):

* **Over-long prompts.** The course `encode_prompt_response` (3 Oct update) keeps prompts intact and raises
  if a prompt alone does not fit in `max_sequence_length=768`. Such pairs are excluded up-front by
  `task1_dpo.train.filter_fitting_rows`; the excluded `prompt_id`s are saved in each run's
  `results/task1_dpo/train/<run>/run_config.json` and each eval `summary.json`.
* **Training.** Fresh LoRA (configs/base.yaml) on Qwen2.5-1.5B-Instruct; the frozen reference is the same
  network with the adapter disabled. Shuffle order fixed with `torch.Generator().manual_seed(seed)`.
  Batch 2 x grad-accum 8, AdamW, grad-norm clip 1.0, fp16 base with fp32 LoRA weights and a GradScaler.
  β forks: first `short_ablation_examples=600` rows of the standard file, otherwise identical.
* **Held-out preference metrics.** DPO margin with summed response-token log-probs, accuracy = mean(m > 0),
  held-out DPO loss at the condition's β.
* **Generation metrics.** Fixed prompt set = first 100 held-out prompts whose rendered prompt is ≤ 512 tokens
  (IDs saved in `summary.json`) + the 10 word-limit prompts; sampling with configs/base.yaml settings,
  course seed, `max_generation_tokens=256`. KL = `common.metrics.sampled_kl` pooled over all valid sampled
  tokens (token-averaged); reward = course reward model (`score_reward_pairs`, prompt truncated from the left
  if needed); length = generated response tokens (mean, std, IQR).

Outputs: `results/task1_dpo/{train,eval}/...`, `beta_summary.csv`, `length_strata.csv`,
`length_generation.csv`, `length_dataset_stats.json`, `qualitative_candidates.md`.

## Task 2 – PPO: exact commands

```bash
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard     # Step 1 (20 updates)
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml                      # Step 2 (cached batch + eps forks)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml                             # Step 3 (beta_KL forks)
python -m task2_ppo.summarize --config configs/ppo.yaml                             # tables + qualitative candidates
```

Implementation choices (shared helpers in `common/rl.py`):

* **Prompts.** RL prompt pools are filtered to prompts whose rendered chat prompt is <= `max_prompt_length` (256)
  tokens, because `batch_generate` would otherwise cut the prompt from the right (256 of 1,200 training prompts
  dropped; IDs in `run_config.json`). The kept training prompts are shuffled once with the course seed; update u
  uses prompts `[u*k, (u+1)*k)`, so every run and fork sees the identical prompt sequence.
* **Rollout.** Course decoding (T=0.7, top-p 0.9), cap `max_response_length=512`; reward = course RM score, minus
  `missing_eos_penalty=1.0` when the response never emitted EOS.
* **Update.** KL-shaped token rewards (`shaped_rewards`, beta_KL), GAE (gamma=1, lambda=0.95) with the critic's
  values, advantages whitened over valid tokens, `ppo_epochs=2` passes of the corrected clipped loss and the critic
  MSE (`value_coef=0.5`), separate AdamW optimizers, grad-norm clip 1.0. Dropout is disabled in policy and critic so
  the ratio is exactly 1 on the first pass. Critic = supplied midpoint value model + fresh LoRA, trainable head in fp32.
* **Logged per update:** raw/effective reward, sampled KL (token mean and sequence sum), full-distribution entropy,
  policy/value loss, clip fraction, affected-token fraction (clipped branch active), policy/value grad norms, ratio
  range, KL(old||new) after the update, critic explained variance before/after, response length, truncation,
  elapsed time, peak VRAM.
* **Evaluation.** First 64 eval-pool prompts that fit the prompt cap, one sample each, cap
  `eval_max_response_length=768`, same seed and decoding for every condition.
* **Stability statistics for the forks** (`task2_ppo/forks.py`): max per-update KL(old||new), max/min ratio, max
  policy grad norm, std of the per-update policy loss, mean clip / affected fractions.
* **Cached clipping study.** The 32 cached rollouts are re-tokenised (text + EOS if terminated; lengths checked
  against `response_tokens`), advantages come from the cached rewards/values/log-probs, and for each epsilon the
  midpoint policy takes `ppo_epochs` full-batch steps on that fixed batch; clip/affected fractions and surrogates
  are measured after every pass, including pass 0 (cache-consistency check).

## Attribution

Code was written with assistance from an LLM coding assistant (Claude); I reviewed and am responsible for
every line. No external code was copied. The report text is my own.


## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
