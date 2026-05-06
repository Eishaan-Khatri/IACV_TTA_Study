# Results Writeup Template

Use this after `results/first_pass_results.csv` is generated.

## Project Title

Uncertainty-Gated Test-Time Adaptation Under Corruption Shift

## Short Abstract

This study evaluates test-time adaptation under image corruption shift using CIFAR-10-C. A source model trained on clean CIFAR-10 is evaluated under five corruption types and five severity levels. I compare source-only inference, batch-normalization adaptation, Tent-style entropy minimization, and a simple uncertainty-gated Tent variant that updates only on confident target samples. The goal is to study when adaptation helps, when it causes negative transfer, and whether confidence gating improves update reliability under severe corruptions.

## Experimental Setup

- Source dataset: CIFAR-10 train split.
- Target benchmark: CIFAR-10-C.
- Corruptions used in first pass: gaussian_noise, shot_noise, motion_blur, brightness, fog.
- Severities: 1 to 5.
- Model: CIFAR-adapted ResNet-18.
- Training: clean CIFAR-10 supervised training.
- Adaptation setting: no target labels used during adaptation.

## Methods Compared

### Source Only

No adaptation. The trained CIFAR-10 model is evaluated directly on corrupted target data.

### BN Adapt

The model runs on target batches with batch-normalization statistics updated, without gradient-based parameter updates.

### Tent

Entropy minimization at test time. Batch-normalization affine parameters are updated using target batches.

### UG-Tent

Uncertainty-gated Tent. The entropy-minimization update is applied only to samples whose maximum softmax confidence is above a fixed threshold. The intent is to avoid updating on highly uncertain samples under severe shift.

## Results To Fill

### Clean Source Accuracy

`clean_acc = ____`

If clean accuracy is below 70 percent, note that source-model weakness limits the strength of the conclusions.

### Mean Accuracy Across First-Pass Corruptions

| Method | Mean Accuracy | Mean Entropy | Mean Update Ratio |
| --- | ---: | ---: | ---: |
| Source Only |  |  |  |
| BN Adapt |  |  |  |
| Tent |  |  |  |
| UG-Tent |  |  |  |

### Accuracy By Severity

| Severity | Source Only | BN Adapt | Tent | UG-Tent |
| ---: | ---: | ---: | ---: | ---: |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

## Negative Transfer Analysis

Report cases where an adaptation method performs worse than source-only.

| Corruption | Severity | Method | Accuracy Drop vs Source Only |
| --- | ---: | --- | ---: |
|  |  |  |  |

Questions to answer:

- Does Tent help mostly at mild severities, severe severities, or neither?
- Does UG-Tent reduce the number of negative-transfer cases?
- Does the update ratio fall as corruption severity increases?
- Does lower update ratio correspond to better stability or just missed adaptation opportunities?

## Interpretation

Use cautious language. Good conclusions look like:

- "In this first-pass CIFAR-10-C study, UG-Tent reduced harmful updates on some severe corruptions, but performance remained dependent on the corruption type."
- "The results suggest confidence gating is a simple diagnostic tool for studying when entropy minimization is reliable under shift."
- "This is not a state-of-the-art claim; the study is intended as a reproducible empirical probe of TTA behavior."

Avoid:

- "Proposed a novel SOTA TTA method."
- "Solved domain adaptation."
- "Outperforms all baselines" unless the table actually proves it.

## CV Bullet Candidates

Use only after results are available.

Conservative:

- Implemented a CIFAR-10-C test-time adaptation study comparing source-only, BN adaptation, Tent-style entropy minimization, and uncertainty-gated adaptation across five corruption types and five severity levels.

Stronger if UG-Tent helps:

- Built an uncertainty-gated Tent variant for CIFAR-10-C corruption shift and analyzed negative-transfer cases across 25 corruption/severity settings, showing when confidence-filtered updates improve adaptation stability.

Strongest only if supported by results:

- Reduced negative-transfer cases in Tent-style test-time adaptation on CIFAR-10-C by gating entropy-minimization updates using predictive confidence, with analysis across 25 corruption/severity settings.

## Limitations

- Small-scale benchmark.
- CIFAR-10-C is useful but not equivalent to real medical, remote-sensing, or surveillance shifts.
- Source model is trained from scratch and may be weaker than standard robust baselines.
- Hyperparameters were not exhaustively tuned.
- UG-Tent is a simple modification, not a full new method.

