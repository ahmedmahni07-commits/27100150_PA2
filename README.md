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
python -m task1_dpo.truncation_stats --config configs/dpo.yaml                    # length-handling counts (CPU)
```

Implementation choices (identical for every Task 1 condition):

* **Over-length pairs (prompt + response > 768 tokens).** Strategy: *prompt preservation + response truncation,
  with filtering* — the course patch of 3 Oct (`common.data.encode_prompt_response`) keeps the prompt whole and
  right-truncates the response (keeping EOS); pairs whose prompt alone is >= 768 tokens cannot be kept that way and
  are excluded by `task1_dpo.train.filter_fitting_rows`. The same rule is applied to every DPO training and
  evaluation run. Excluded IDs: `results/task1_dpo/train/<run>/run_config.json` and each eval `summary.json`;
  counts of excluded / truncated pairs per file and stratum: `results/task1_dpo/truncation_stats.json`.
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

### Task 1 supplementary: multi-epoch DPO (not part of the fixed budget)

`configs/dpo_long.yaml` is identical to `configs/dpo.yaml` except `epochs: 5` and `save_every_epoch: true`.
It writes to `outputs/task1_dpo_long/` and `results/task1_dpo_long/`, so the graded runs are untouched.
An interrupted run resumes from the last finished epoch (adapter, optimizer, grad-scaler and shuffle state).

```bash
python -m task1_dpo.train --config configs/dpo_long.yaml --run-name standard
for k in 1 3 5; do
  python -m task1_dpo.evaluate --config configs/dpo_long.yaml --adapter outputs/task1_dpo_long/standard/epoch_$k --name epoch_$k
done
python -m task1_dpo.summarize_epochs --config configs/dpo_long.yaml   # -> results/task1_dpo_long/epoch_summary.csv
```

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

### Task 2 supplementary stress test (configuration fixed before running)

The released fork budget never activates clipping (max |ratio - 1| ~ 0.03 on every update) and keeps KL drift at
~1e-5, so the epsilon and beta_KL comparisons are uninformative under it. `configs/ppo_supp.yaml` is identical except
policy lr x10 (3e-5), 4 PPO epochs, 2 prompts per update, and 128 evaluation prompts, so that each update is large
enough for clipping and KL pressure to act. Its results (`results/task2_ppo_supp/`) are a supplement to, not a
replacement for, the required results.

```bash
python -m task2_ppo.evaluate --config configs/ppo_supp.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint --n-prompts 128
python -m task2_ppo.analyze_clipping --config configs/ppo_supp.yaml --skip-cached --n-prompts 128
python -m task2_ppo.ablate_kl --config configs/ppo_supp.yaml --n-prompts 128
python -m task2_ppo.summarize --config configs/ppo_supp.yaml
```

## Task 3 – GRPO: exact commands

```bash
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard          # Step 1 (20 updates)
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml                         # Step 2 (CPU, cache only)
python -m task3_grpo.compare_normalization --config configs/grpo.yaml                      # Step 3 (grpo vs dr_grpo forks)
python -m task3_grpo.summarize --config configs/grpo.yaml
```

* **Loop.** Same seeded prompt order and prompt filter as PPO; K=4 completions per prompt (cap 512); RM score as reward;
  per-prompt group-relative advantages (corrected helper); completions that hit the cap without EOS are masked out of
  the loss but kept in their group's statistics; one pass of the corrected clipped loss + beta*KL (k3 estimator of
  the helper). Uninformative group = reward std <= 1e-6 (the helper's eps).
* **Per-completion backward.** Each completion's exact share of `grpo_policy_loss` is back-propagated separately (sum
  verified equal to the batch loss and gradient in `tests/test_objectives.py`), so the gradient norm contributed by
  every completion is logged with its length (`sequence_log.jsonl`) - the length-conditioned statistic of Step 3.
* **Evaluation.** The PPO protocol (`task2_ppo.evaluate`): same 64 held-out prompts, same seed/decoding, 768-token cap.
* **Group-size study.** The 24 x 8 cache is split into disjoint groups of K within each prompt (all 192 generations used
  for every K); metrics are averaged over 200 seeded random within-prompt partitions. Difficulty bins = terciles of a
  prompt's mean cached reward (hard = lowest).

## Task 4 – Safety: exact commands

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # greedy, 256 tokens, SFT/DPO/PPO/GRPO standard
python -m task4_safety.judge_responses --config configs/feedback.yaml      # fixed 3B judge, resumable
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # fixed 60 IDs + blind labelling sheet
# fill results/task4_safety/manual_audit_sheet.csv (manual_label column) by hand, then:
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

* Policies are fixed by `configs/feedback.yaml` (no ablation checkpoints). One greedy response per prompt, CSV order.
* Judge: the supplied prompt and parser; responses are judged in left-padded batches of 8 (`--batch-size 1` uses the
  supplied `judge_one` exactly).
* Audit: the course's fixed 60 IDs (30 SAFE + 30 UNSAFE, seed 6304). Identical responses from different policies are
  labelled once and the label applied to each; the sheet is shuffled and contains no AI labels.
* Rates as in `manifests/source_v2_task4_safety_judge_spec.json`, plus ambiguous rate and class-mismatch rate.

## Task 5 – RLVR vs RLAIF: exact commands

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm        # Step 1
python -m task5_feedback.score_perturbations --config configs/feedback.yaml                # Step 2
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer   # Step 3
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

* One greedy response per problem (cap 512) for SFT, RLVR and RLAIF; exact accuracy and format compliance from the
  supplied verifier; pairwise win rate vs SFT with the supplied judge (win 1 / tie 0.5 / loss 0; responses identical to
  SFT counted as ties without a judge call and reported separately).
* Diagnostics: four controlled pairs vs `clean_correct` (reasoning, outcome, filler, distractor); the judge is asked in
  both presentation orders (0.5 each) and its order-consistency is reported; S_reason / S_outcome as in the manual.
  The direct-RLAIF group reward of each variant (`group_rewards` over the 5 variants) is also reported.

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
