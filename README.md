# Uncertainty-Gated Test-Time Adaptation Under Corruption Shift

Small empirical study for the IISc IACV RA application.

## One-Line Goal

Evaluate whether uncertainty-gated test-time adaptation can reduce harmful entropy-minimization updates under image corruption shift.

## Research Question

Entropy-minimization based test-time adaptation can improve target-domain performance, but it can also reinforce wrong pseudo-labels under severe corruptions. Can a simple uncertainty/confidence gate make adaptation more stable by updating only on target samples that the model is sufficiently confident about?

## Why This Fits IACV

This project directly touches themes from Prof. Soma Biswas's IACV Lab:

- test-time adaptation
- domain/corruption shift
- limited target supervision
- robust visual recognition
- failure analysis under real deployment constraints

It is not claiming to reproduce a full IACV paper. It is a focused empirical study inspired by the same problem family.

## Minimal Study Design

Dataset:

- Source: CIFAR-10 train/test
- Target shift: CIFAR-10-C corruptions at severities 1-5
- Fast subset for first pass: 5 corruptions x 5 severities
- Full pass if time allows: all 15 CIFAR-10-C corruptions x 5 severities

Model:

- Compact CIFAR ResNet or small CNN trained on clean CIFAR-10
- Optional later: use a stronger public pretrained CIFAR model if setup time permits

Methods:

1. **Source Only**: no adaptation on target data.
2. **BN Adapt**: update batch normalization statistics on target batches only.
3. **Tent**: entropy minimization during test time, updating normalization affine parameters.
4. **UG-Tent (ours)**: uncertainty-gated Tent; entropy minimization updates only on samples with low predictive entropy / high confidence.

Main metrics:

- accuracy by corruption and severity
- mean accuracy across corruption groups
- update ratio: percentage of target samples used for adaptation
- entropy before/after adaptation
- failure cases where adaptation hurts source-only performance

## Claim Boundaries

Safe claim if completed:

> Implemented a small test-time adaptation study on CIFAR-10-C comparing source-only, BN adaptation, Tent-style entropy minimization, and uncertainty-gated adaptation under corruption shift.

Do not claim:

- state-of-the-art
- full pSTarC/SANTA reproduction
- novel method
- publication-level benchmark
- ImageNet-scale validation

## Primary References

- Tent: Fully Test-Time Adaptation by Entropy Minimization, Wang et al., ICLR 2021.
- CIFAR-10-C / CIFAR-100-C: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations, Hendrycks and Dietterich, ICLR 2019.
- SANTA: Source Anchoring Network and Target Alignment for Continual Test Time Adaptation, TMLR 2023.
- pSTarC: Pseudo Source Guided Target Clustering for Fully Test-Time Adaptation, WACV 2024.

## Final Deliverables

- clean GitHub repo
- reproducible experiment commands
- results table
- 2-3 plots
- short writeup with limitations
- 1 CV bullet if results are meaningful

## Colab Quick Start

Colab is the recommended runtime. See `COLAB_GUIDE.md`.

```bash
!pip install -q -r requirements.txt
!python scripts/run_study.py \
  --config configs/default.yaml \
  --data-root /content/drive/MyDrive/iacv_tta_data \
  --epochs 15 \
  --results results/first_pass_results.csv
!python scripts/summarize_results.py --results results/first_pass_results.csv
```

Use 30 epochs if the clean CIFAR-10 source model is too weak after the 15-epoch smoke run.
