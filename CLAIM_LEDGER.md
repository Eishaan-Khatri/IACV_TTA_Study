# Claim Ledger

This file protects the project from overclaiming.

## Safe Claims Before Running Experiments

- This project is a planned empirical study of test-time adaptation under corruption shift.
- It compares source-only, BN adaptation, Tent-style entropy minimization, and uncertainty-gated adaptation.
- It is designed for CIFAR-10-C, a standard corruption benchmark.
- It is inspired by test-time adaptation and domain-shift literature, not a full reproduction of SANTA or pSTarC.

## Safe Claims After Minimum Completion

Only use these after the experiments actually run:

- Implemented and evaluated the specified baselines.
- Reported accuracy by corruption and severity.
- Logged update ratios for the uncertainty-gated method.
- Analyzed where adaptation helped or hurt.

## Unsafe Claims

Do not claim:

- state-of-the-art
- novel algorithm
- full IACV paper reproduction
- publication-ready result
- ImageNet-scale robustness
- medical/remote-sensing validation

## How To Discuss If Asked

Good answer:

> I built this as a small empirical bridge into IACV's test-time adaptation themes. The point was not to beat SOTA, but to understand when entropy-minimization adaptation helps or hurts under corruption shift, and whether uncertainty gating can reduce harmful updates.

Bad answer:

> I developed a new state-of-the-art TTA method.

