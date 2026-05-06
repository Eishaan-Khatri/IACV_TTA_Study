# Interview Notes: TTA Study

## One-Minute Explanation

I wanted one project that directly tests the kind of deployment shift problem IACV works on. I trained a source model on clean CIFAR-10 and evaluated it on CIFAR-10-C corruptions. I compared source-only inference, BN adaptation, Tent-style entropy minimization, and a simple uncertainty-gated Tent variant. The question was not whether I could claim a new state-of-the-art method, but whether I could set up a controlled adaptation benchmark, reproduce a known TTA baseline, and analyze failure cases under increasing corruption severity.

## Why Test-Time Adaptation?

In deployment, the target distribution often changes after training: camera noise, blur, lighting, weather, sensor differences, or acquisition artifacts. Test-time adaptation studies whether a model can adjust at inference time without target labels and often without source data.

## Why Tent?

Tent is a clean baseline because it adapts by minimizing prediction entropy on target batches while updating only normalization parameters. It is simple enough to reproduce quickly, but still captures the central risk of TTA: if predictions are wrong but confident, adaptation can reinforce errors.

## What Is The Modification?

UG-Tent adds a confidence gate. Instead of using every target sample for entropy minimization, it updates only on samples whose maximum softmax probability crosses a threshold. The motivation is to reduce updates from highly uncertain samples under severe corruption.

## Why This Is Not Overclaiming

This is a small empirical study, not a full paper. The contribution is implementation, comparison, controlled analysis, and a simple extension. The defensible claim is that I can read a TTA method, implement the core idea, run a benchmark, and reason about negative transfer.

## Likely Questions

### Why not implement SANTA or pSTarC directly?

Those are more complex protocols. I started with Tent because it is the standard minimal TTA baseline and gives a controlled setting for understanding entropy minimization. A next step would be adding source anchoring or pseudo-source clustering inspired by SANTA/pSTarC.

### What can go wrong with confidence gating?

It may reject too many samples under severe shift, causing little adaptation. It can also select confidently wrong samples, especially under systematic corruptions. That is why update ratio and negative-transfer cases need to be reported alongside accuracy.

### Why CIFAR-10-C?

It is a standard robustness benchmark with controlled corruption types and severity levels. It is small enough to run on Colab but still tests domain/corruption shift in a reproducible way.

### How does this connect to IACV?

The connection is test-time adaptation and robust visual recognition under distribution shift. IACV has worked on continual TTA, fully test-time adaptation, and VLM adaptation; this project is a small-scale preparation step toward that research direction.

### What would you improve next?

- Run on CIFAR-100-C or DomainNet.
- Compare additional TTA baselines.
- Add source anchoring to reduce drift.
- Evaluate with a stronger pretrained model.
- Study class-wise failures and calibration metrics.

