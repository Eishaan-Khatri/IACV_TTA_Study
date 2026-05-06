# Experiment Plan

## Project Title

**Uncertainty-Gated Test-Time Adaptation Under Corruption Shift**

Short title for CV/GitHub:

**Uncertainty-Gated TTA on CIFAR-10-C**

## Core Hypothesis

Tent-style entropy minimization improves adaptation under mild corruption, but under severe/noisy corruption it can adapt on unreliable pseudo-labels. Gating adaptation updates by predictive uncertainty should reduce harmful updates and improve stability, especially at high corruption severities.

## Why This Is The Right Option B

It gives direct evidence for:

- test-time adaptation
- domain/corruption shift
- robustness
- uncertainty-aware ML
- vision dataset experimentation

It also bridges naturally from the existing Uncertainty Study.

## Dataset Choice

### Primary: CIFAR-10-C

Reasons:

- standard corruption benchmark
- fast enough for a short project
- widely used in robustness and TTA papers
- easy to report results by corruption severity

### First-Pass Corruptions

Use this smaller set first:

- gaussian_noise
- shot_noise
- motion_blur
- brightness
- fog

Why:

- covers noise, blur, illumination, and weather-like corruption
- enough to show meaningful shift without running the full benchmark first

### Full-Pass Corruptions

If time allows, run all CIFAR-10-C corruptions:

- gaussian_noise
- shot_noise
- impulse_noise
- defocus_blur
- glass_blur
- motion_blur
- zoom_blur
- snow
- frost
- fog
- brightness
- contrast
- elastic_transform
- pixelate
- jpeg_compression

## Baselines

### 1. Source Only

Train on clean CIFAR-10 and evaluate on CIFAR-10-C without adaptation.

Purpose:

- absolute baseline
- shows corruption degradation

### 2. BN Adapt

Run the model in training mode on target batches so batch normalization statistics adapt, but do not update weights.

Purpose:

- simple adaptation baseline
- often surprisingly strong under covariate shift

### 3. Tent

Minimize prediction entropy at test time. Update only normalization affine parameters where possible.

Purpose:

- canonical TTA baseline
- directly relevant to test-time adaptation literature

### 4. UG-Tent

Uncertainty-gated Tent:

- compute predictive entropy for each sample
- select only samples below entropy threshold or above confidence threshold
- minimize entropy only on selected samples
- record update ratio

Purpose:

- small extension using your uncertainty/failure-analysis story
- easy to understand and defend
- not overclaimed as a new method

## UG-Tent Variants

Start simple:

- fixed confidence threshold: max probability >= 0.7
- fixed entropy threshold alternative: entropy <= 1.0

If time allows:

- dynamic threshold: keep lowest-entropy 50 percent of a batch
- severity-aware analysis: compare mild severity 1-2 vs severe 4-5

## Metrics

Required:

- top-1 accuracy
- mean accuracy across corruptions
- accuracy by severity
- average entropy
- update ratio for UG-Tent

Optional:

- expected calibration error
- negative transfer count: number of corruption/severity pairs where method is worse than source-only
- runtime per batch

## Tables To Produce

### Table 1: Mean Accuracy

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Source Only | | | | | | |
| BN Adapt | | | | | | |
| Tent | | | | | | |
| UG-Tent | | | | | | |

### Table 2: Failure Cases

| Corruption | Severity | Source | Tent | UG-Tent | Comment |
|---|---:|---:|---:|---:|---|

### Table 3: UG-Tent Update Ratio

| Corruption | Severity 1 | Severity 3 | Severity 5 |
|---|---:|---:|---:|

## Plots To Produce

- accuracy vs severity
- entropy vs severity
- UG-Tent update ratio vs severity

## Success Criteria

Minimum success:

- source-only, BN Adapt, Tent, and UG-Tent run on 5 corruptions x 5 severities
- readable result table
- honest limitations

Strong success:

- UG-Tent improves over Tent in at least some severe corruptions or reduces negative transfer cases
- even if not better, analysis explains when gating helps and when it fails

Failure is still useful if documented:

- If UG-Tent does not improve, frame it as "uncertainty gating reduced harmful updates in X cases but was too conservative overall" or "confidence was poorly calibrated under severe corruptions."

## CV Bullet If Results Are Good

Implemented a CIFAR-10-C test-time adaptation study comparing source-only, BN adaptation, Tent-style entropy minimization, and uncertainty-gated adaptation across corruption severities; analyzed negative transfer and update reliability under severe shift.

## Extended Variant Suite

The first-pass run showed that BN adaptation and Tent recover large accuracy under severe noise/blur/fog shifts, but adaptation can hurt under mild brightness/fog. The extended suite therefore tests variants aimed at reducing negative transfer.

Run:

```bash
python scripts/run_study.py \
  --config configs/colab.yaml \
  --data-root /content/drive/MyDrive/iacv_tta_data \
  --epochs 15 \
  --results results/extended_results.csv \
  --variant-suite extended
python scripts/analyze_variants.py --results results/extended_results.csv
```

Variants:

- `ug_tent_conf_080`, `ug_tent_conf_090`: stricter confidence-gated Tent. Tests whether the original 0.70 threshold was too permissive.
- `margin_tent_040`, `margin_tent_060`: updates only when the top-1/top-2 probability margin is large. This is stricter than max-softmax confidence.
- `anchor_tent_010`, `anchor_tent_030`: adds a KL anchor to the frozen source model's predictions. Tests whether source anchoring reduces damage on mild shifts.
- `anchor_ug_tent_020_c080`: combines source anchoring with confidence gating.
- `source_mix_tent_050`, `source_mix_tent_075`: adapts like Tent, but predicts with a source/adapted logit ensemble. Tests whether ensembling protects source behavior while preserving adaptation gains.

Primary analysis:

- mean accuracy
- negative-transfer count vs source-only
- delta vs Tent
- update ratio
- severity-wise behavior

Do not claim novelty from these variants without literature review. The safe claim is that they are exploratory stabilization variants inspired by the failure modes observed in the first-pass run.

## Adaptive Source-Mix Extension

The extended run found that fixed `source_mix_tent_075` improved mean accuracy and reduced negative-transfer cases, but it still underperformed Tent on some severe noise settings. This suggests that a fixed mix weight is too rigid:

- mild brightness/fog: source model is already reliable, so use more source logits
- severe noise/fog: source model is unreliable, so use more adapted logits

The adaptive suite tests this idea directly.

Run:

```bash
python scripts/run_study.py \
  --config configs/colab.yaml \
  --data-root /content/drive/MyDrive/iacv_tta_data \
  --epochs 15 \
  --results results/adaptive_results.csv \
  --variant-suite adaptive
python scripts/analyze_variants.py --results results/adaptive_results.csv
```

Adaptive variants:

- `eg_mix_batch`: batch-level entropy-gap adaptive mixing.
- `eg_mix_sample`: per-sample entropy-gap adaptive mixing.
- `disagree_aware_mix`: if source/adapted predictions agree, mostly trust adapted logits; if they disagree, choose mix weight from entropy gap.
- `probe_commit_tent`: makes a tentative Tent update, commits only if unlabeled safety checks pass, otherwise rolls back the update and predicts with the frozen source model.
- `probe_commit_tent_loose`: same idea with looser thresholds.
- `source_mix_tent_075`: fixed-mix reference from the previous run.

Entropy-gap rule:

```text
gap = normalized_entropy(source_logits) - normalized_entropy(adapted_logits)
```

If `gap` is large and positive, the adapted model is more confident than the source model, so the method increases adapted-logit weight. If `gap` is negative, the source model is more confident, so it falls back toward source logits.

This should be described as a student-proposed exploratory extension, not as a guaranteed novel method. Related literature already contains sample selection, confidence filtering, anti-forgetting, teacher-student averaging, and source anchoring ideas.

Probe-commit rule:

```text
1. snapshot adapted BN affine parameters
2. perform one tentative Tent update
3. compare frozen-source and post-update predictions
4. commit only if:
   - post-update normalized entropy is lower than source entropy by a margin
   - few high-confidence source predictions are flipped
   - predictions do not collapse into one dominant class
5. otherwise restore the snapshot and use source logits
```

The motivation is negative-transfer control. Unlike fixed source/adapted ensembling, probe-commit changes the adaptation dynamics: unsafe updates are not allowed to accumulate.
