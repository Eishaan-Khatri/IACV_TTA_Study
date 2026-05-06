# Beginner Guide To The Three Papers Behind This Project

This document explains the three papers most related to the current project:

1. **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**  
   Dan Hendrycks and Thomas Dietterich, ICLR 2019.

2. **Tent: Fully Test-Time Adaptation by Entropy Minimization**  
   Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell, ICLR 2021.

3. **SANTA: Source Anchoring Network and Target Alignment for Continual Test Time Adaptation**  
   Goirik Chakrabarty, Manogna Sreenivas, Soma Biswas, TMLR 2023.

The aim is not to copy the papers line by line. The aim is to explain every important idea we are using in this project, in beginner-friendly terms.

Primary links:

- CIFAR-C / ImageNet-C paper: https://openreview.net/forum?id=HJz6tiCqYm
- Tent paper: https://openreview.net/forum?id=uXl3bZLkr3c
- SANTA paper: https://openreview.net/forum?id=V7guVYzvE4

---

# 0. Basic Concepts Before The Papers

## Statement

In image classification, a model takes an image and predicts a class.

## Explanation

For example, the input may be an image of a dog. The model processes the image and outputs probabilities for each possible class:

```text
airplane: 0.01
car:      0.02
bird:     0.03
cat:      0.10
dog:      0.80
horse:    0.04
```

The model predicts the class with the highest probability. Here, the prediction is `dog` because `0.80` is the highest value.

In code, this usually looks like:

```python
logits = model(image)
probabilities = softmax(logits)
prediction = argmax(probabilities)
```

`logits` are raw model outputs before probabilities. `softmax` converts them into probabilities that sum to 1.

---

## Statement

A model is usually trained on a source dataset and tested on a target dataset.

## Explanation

The **source dataset** is the data used during training.

The **target dataset** is the data encountered during testing or deployment.

In our project:

```text
source dataset = clean CIFAR-10
target dataset = corrupted CIFAR-10-C
```

The model learns from clean images but is evaluated on corrupted images.

This is important because real-world deployment rarely looks exactly like training data. A surveillance camera may have blur, fog, low light, compression artifacts, or unusual viewpoints. A medical model may see images from a different scanner. A satellite model may see images from a different region or season.

---

## Statement

Domain shift means the training distribution and test distribution are different.

## Explanation

In machine learning, a distribution describes what kind of data the model sees.

If the model trains on clean images and tests on foggy images, then the input distribution has changed.

Mathematically:

```text
P_source(x, y) != P_target(x, y)
```

This means:

```text
the probability pattern of training images and labels
is not the same as
the probability pattern of test images and labels
```

Here:

- `x` means the image.
- `y` means the label.
- `P_source` means the training/source distribution.
- `P_target` means the testing/target distribution.

In our project:

```text
P_source = clean CIFAR-10
P_target = CIFAR-10-C with noise, blur, fog, brightness shift, etc.
```

---

## Statement

Test-time adaptation means modifying the model during testing using unlabeled test data.

## Explanation

Normal testing does not change the model:

```text
train model -> freeze model -> test model
```

Test-time adaptation changes this:

```text
train model -> test model -> update model during testing
```

The hard part is that test-time adaptation usually does **not** have labels.

So during testing, the model sees:

```text
image only
```

not:

```text
image + correct label
```

This makes the problem difficult. We cannot simply calculate normal supervised loss because we do not know the true answer.

---

## Statement

Entropy measures uncertainty in the model's prediction.

## Explanation

Entropy is high when the model is confused.

Example of high uncertainty:

```text
cat: 0.34
dog: 0.33
deer: 0.33
```

The model has no clear preference.

Entropy is low when the model is confident.

Example of low uncertainty:

```text
dog: 0.95
cat: 0.03
deer: 0.02
```

The model strongly believes the image is a dog.

The entropy formula is:

```text
H(p) = - sum_i p_i log(p_i)
```

Where:

- `p_i` is the probability of class `i`.
- `log` is the logarithm.
- `sum_i` means add the value over all classes.

You do not need to manually calculate it in the project. PyTorch handles it. But you should understand what it means:

```text
high entropy = uncertain prediction
low entropy = confident prediction
```

---

# 1. Paper One: CIFAR-C / ImageNet-C Robustness Benchmark

## Statement

The paper argues that neural networks should be tested on common real-world corruptions, not only clean images.

## Explanation

Many models perform very well on clean benchmark images. But clean benchmark images are often easier than real deployment data.

Real images may contain:

- blur from camera movement
- noise from low light
- fog or weather effects
- compression artifacts
- brightness changes
- pixelation

The paper says that if a model only works on clean data, it is not robust enough.

This is especially relevant to computer vision because images are highly sensitive to visual quality. A human may still recognize a foggy car, but a neural network may fail.

---

## Statement

The paper introduces standardized corruption benchmarks such as ImageNet-C and CIFAR-10-C.

## Explanation

A benchmark is a standard test setup used by many researchers.

Without a standard benchmark, every researcher might test on different corruptions, different severity levels, and different datasets. That makes results hard to compare.

This paper creates standardized corrupted datasets:

```text
ImageNet-C
CIFAR-10-C
CIFAR-100-C
```

The `-C` means **corrupted**.

For our project, we use:

```text
CIFAR-10-C
```

This gives our project a recognized benchmark instead of a random homemade noise experiment.

---

## Statement

CIFAR-10-C is created by applying corruption functions to the CIFAR-10 test set.

## Explanation

CIFAR-10 has 10 classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

CIFAR-10-C takes the test images from CIFAR-10 and corrupts them.

For example:

```text
clean dog image -> gaussian_noise dog image
clean dog image -> motion_blur dog image
clean dog image -> fog dog image
```

The label remains the same.

If the original image is a dog, the corrupted image is still labeled dog.

This lets us measure how much the model's accuracy drops because of the corruption.

---

## Statement

Each corruption in CIFAR-10-C has five severity levels.

## Explanation

Severity controls how strong the corruption is.

```text
severity 1 = mild corruption
severity 2 = slightly stronger
severity 3 = moderate
severity 4 = strong
severity 5 = severe
```

Example:

```text
gaussian_noise severity 1 = image is slightly noisy
gaussian_noise severity 5 = image is heavily noisy
```

This is useful because robustness is not a yes/no property. A model may handle mild fog but fail under severe fog.

In our project, this lets us ask:

```text
Does adaptation help at severity 1?
Does it still help at severity 5?
Does it become harmful when corruption becomes too strong?
```

---

## Statement

The paper distinguishes common corruption robustness from adversarial robustness.

## Explanation

Adversarial robustness studies tiny, carefully designed changes that are meant to fool the model.

Example:

```text
small invisible perturbation added to an image
model changes prediction from dog to airplane
```

Common corruption robustness studies naturally occurring visual problems:

```text
fog
blur
snow
noise
compression
brightness shift
```

The paper's point is that both matter, but common corruptions are extremely important for deployment.

Our project focuses on common corruption robustness, not adversarial attacks.

---

## Statement

The paper uses average performance across corruptions and severities to measure robustness.

## Explanation

A model should not be judged only on one corruption.

Example:

```text
good on brightness
bad on blur
bad on fog
```

That model is not generally robust.

So the benchmark averages performance across many corruption types and severity levels.

In our project, we use a simpler first-pass version:

```text
5 corruption types x 5 severity levels = 25 settings
```

Then we calculate:

```text
mean accuracy across all 25 settings
accuracy by severity
accuracy by corruption type
```

This gives a more honest view than a single number.

---

## Statement

The key lesson of the paper is that clean accuracy is not the same as robustness.

## Explanation

A model may have high clean accuracy:

```text
clean CIFAR-10 accuracy = 85%
```

But under corruption:

```text
gaussian_noise severity 5 accuracy = 20%
fog severity 5 accuracy = 35%
motion_blur severity 5 accuracy = 45%
```

This tells us the model learned patterns that work on clean images but fail when image quality changes.

This is the motivation for our project. We train on clean CIFAR-10, then evaluate and adapt on CIFAR-10-C.

---

# 2. Paper Two: Tent

## Statement

Tent stands for test entropy minimization.

## Explanation

Tent is a method for test-time adaptation.

Its central idea is:

```text
During testing, make the model's predictions more confident by minimizing entropy.
```

If a model is uncertain on target images, Tent updates part of the model so predictions become lower entropy.

The method is simple, which is why it is a strong baseline for our project.

---

## Statement

Tent works in the fully test-time adaptation setting.

## Explanation

Fully test-time adaptation is restrictive.

At test time, the method has:

```text
trained model
unlabeled target test data
```

It does not have:

```text
target labels
source training data
extra supervised training
```

This is realistic because in deployment you often cannot access training data again, and new target labels are unavailable.

For example, a deployed medical image model may receive new hospital scans, but nobody labels them instantly.

---

## Statement

Tent minimizes prediction entropy on target data.

## Explanation

Suppose the model outputs probabilities:

```text
cat: 0.40
dog: 0.35
deer: 0.25
```

The prediction is uncertain.

Tent calculates entropy:

```text
H(p) = - sum_i p_i log(p_i)
```

Then it updates model parameters to reduce this entropy.

After adaptation, the prediction might become:

```text
cat: 0.80
dog: 0.15
deer: 0.05
```

The prediction is now more confident.

The hope is that confidence corresponds to correctness.

---

## Statement

Tent does not need labels because entropy is computed from the model's own predictions.

## Explanation

Normal supervised training uses:

```text
loss = cross_entropy(prediction, true_label)
```

But at test time, we do not have `true_label`.

Tent instead uses:

```text
loss = entropy(prediction)
```

This loss only needs the predicted probabilities.

That is why Tent can adapt without labels.

The danger is that the model is learning from itself. If its own predictions are wrong, the adaptation signal can also be wrong.

---

## Statement

Tent updates batch normalization affine parameters.

## Explanation

Batch Normalization, or BatchNorm, is a layer used in many neural networks.

It normalizes internal activations:

```text
normalized_activation = (activation - mean) / sqrt(variance + epsilon)
```

Then it applies learnable scale and shift:

```text
output = gamma * normalized_activation + beta
```

Here:

- `gamma` controls scale.
- `beta` controls shift.
- `mean` and `variance` come from the batch or stored running statistics.

Tent updates mainly:

```text
gamma and beta
```

It does not update all model weights.

This matters because full model updates at test time can be unstable, expensive, and prone to overfitting.

---

## Statement

Tent also uses target batch statistics in BatchNorm.

## Explanation

BatchNorm behaves differently during training and evaluation.

During training:

```text
BatchNorm uses current batch mean and variance.
```

During normal evaluation:

```text
BatchNorm uses stored running mean and variance from training.
```

But if the target domain is shifted, old source statistics may not fit target data.

Example:

```text
source images = bright clean CIFAR
target images = foggy low-contrast CIFAR-C
```

The internal activation distributions change.

Using target batch statistics helps the model normalize target images better.

---

## Statement

Tent performs online adaptation batch by batch.

## Explanation

Online means the model adapts as data arrives.

Pseudo-code:

```python
for target_batch in target_loader:
    logits = model(target_batch)
    probabilities = softmax(logits)
    loss = entropy(probabilities)
    loss.backward()
    optimizer.step()
```

The model changes during evaluation.

This is different from normal testing, where the model remains frozen.

---

## Statement

Tent can improve robustness under corruption shift.

## Explanation

If target images are corrupted but still contain useful structure, entropy minimization may help the model adjust internal normalization.

Example:

```text
source-only on fog severity 3: 55%
Tent on fog severity 3: 60%
```

This means adaptation helped.

Tent is popular because it is simple, efficient, and often effective.

---

## Statement

Tent can also fail through negative transfer.

## Explanation

Negative transfer means adaptation makes performance worse.

Example:

```text
source-only accuracy: 45%
Tent accuracy: 38%
```

This can happen because Tent trusts the model's own predictions.

If the model predicts the wrong class and entropy minimization makes it more confident, the model becomes confidently wrong.

Example:

```text
true label: dog
model before adaptation: cat 0.65, dog 0.20
model after Tent: cat 0.90, dog 0.05
```

The entropy decreased, but the prediction became worse.

This is exactly why our project studies negative transfer.

---

## Statement

Our UG-Tent modification adds confidence gating.

## Explanation

UG-Tent means:

```text
Uncertainty-Gated Tent
```

Instead of adapting on all target samples, we adapt only on samples where the model is confident enough.

For each sample:

```text
confidence = maximum softmax probability
```

Example:

```text
cat: 0.20
dog: 0.75
truck: 0.05
```

Confidence is:

```text
0.75
```

If the threshold is:

```text
0.70
```

then this sample is used for adaptation.

If confidence is:

```text
0.45
```

then the sample is skipped.

---

## Statement

The UG-Tent loss uses only selected samples.

## Explanation

Standard Tent loss:

```text
L = average entropy over all samples
```

UG-Tent loss:

```text
L = average entropy over confident samples only
```

Mathematically:

```text
m_i = 1 if max(p_i) >= threshold else 0
```

Then:

```text
L = sum_i m_i H(p_i) / sum_i m_i
```

Here:

- `m_i` is a mask.
- `m_i = 1` means use the sample.
- `m_i = 0` means skip the sample.
- `H(p_i)` is entropy for sample `i`.

This is a simple modification, but it creates a meaningful experimental question:

```text
Does skipping uncertain samples reduce harmful adaptation?
```

---

## Statement

UG-Tent may help, but it is not guaranteed to help.

## Explanation

Confidence is imperfect.

A model can be confidently wrong.

Example:

```text
true label: dog
model prediction: cat 0.92
```

The confidence is high, but the prediction is wrong.

UG-Tent would still use this sample because it passes the confidence threshold.

Another problem:

```text
under severe corruption, almost no samples pass the confidence gate
```

Then the model barely adapts.

That is why we track:

```text
update ratio
```

Update ratio tells us what fraction of samples were used for adaptation.

---

# 3. Paper Three: SANTA

## Statement

SANTA studies continual test-time adaptation.

## Explanation

Standard test-time adaptation may assume one target shift.

Example:

```text
clean -> fog
```

Continual test-time adaptation is harder because the target distribution keeps changing.

Example:

```text
clean -> fog -> blur -> noise -> brightness shift -> snow
```

The model must keep adapting over time.

This is closer to real deployment because environments are not fixed.

---

## Statement

Continual adaptation can cause model drift.

## Explanation

Model drift means the model gradually moves away from useful original behavior.

Suppose the model adapts to fog, then blur, then noise.

Each update changes the model.

If the updates are based on unreliable predictions, the model may become worse over time.

This can cause:

- forgetting source-domain knowledge
- forgetting earlier target domains
- becoming overconfident on wrong predictions
- unstable behavior across batches

This is a central risk in continual TTA.

---

## Statement

SANTA uses source anchoring to reduce forgetting.

## Explanation

Source anchoring means keeping the adapted model connected to the original source-trained model.

The original model contains useful knowledge from source training.

If adaptation moves too far away from that knowledge, performance may collapse.

SANTA tries to adapt to target data while preserving source behavior.

Conceptually:

```text
adapt to target
but do not forget source
```

This is stronger than only minimizing entropy.

---

## Statement

SANTA uses self-distillation.

## Explanation

Distillation means one model teaches another model.

Usually:

```text
teacher model -> student model
```

The teacher gives soft predictions, not just hard labels.

Example:

```text
teacher prediction:
dog: 0.70
cat: 0.20
deer: 0.05
horse: 0.05
```

The student learns to match this distribution.

Self-distillation means the teacher is derived from the model itself, such as a source-anchored or stable version of the model.

In SANTA, this helps preserve useful knowledge while adapting.

---

## Statement

SANTA modifies BatchNorm affine parameters.

## Explanation

Like Tent, SANTA focuses on changing a limited part of the model.

The important BatchNorm affine parameters are:

```text
gamma
beta
```

These scale and shift internal normalized activations.

Updating only these parameters is useful because:

- it is faster than updating the full network
- it is less likely to destroy learned representations
- it requires fewer trainable parameters
- it is practical during online deployment

This is why BatchNorm adaptation appears repeatedly in TTA research.

---

## Statement

SANTA cares about small batch sizes.

## Explanation

Many test-time adaptation methods rely on batch statistics.

Large batch:

```text
statistics are more stable
```

Small batch:

```text
statistics are noisy
```

In real deployment, we may not get 128 images at once. We may get:

```text
one image
four images
small streaming batches
```

A method that works only with large batches may not be practical.

SANTA explicitly considers this kind of deployment constraint.

---

## Statement

SANTA is closer to IACV's research direction than our small project.

## Explanation

Our project is a first step:

```text
clean CIFAR-10 -> fixed CIFAR-10-C corruptions
```

SANTA is more advanced:

```text
continually changing target environments
source anchoring
self-distillation
small-batch behavior
reduced forgetting
```

This means our project should not claim to reproduce SANTA.

The correct relationship is:

```text
Our project studies a basic TTA baseline and a simple gating modification.
SANTA motivates what stronger stabilization methods look like.
```

---

# 4. How The Three Papers Connect To Our Project

## Statement

The CIFAR-C paper gives us the benchmark.

## Explanation

Without CIFAR-10-C, we would only be testing on manually corrupted images.

Using CIFAR-10-C makes the project more credible because it is a known robustness benchmark.

Our project uses it to measure performance under controlled corruption shift.

---

## Statement

Tent gives us the main adaptation baseline.

## Explanation

Tent is the simplest serious TTA method to implement and explain.

It gives us a baseline stronger than source-only evaluation.

Our comparison includes:

```text
source-only
BN adaptation
Tent
UG-Tent
```

This makes the project more than just model training.

It becomes a small adaptation study.

---

## Statement

SANTA gives us the IACV connection and next-step research direction.

## Explanation

Prof. Soma Biswas's lab works on test-time adaptation, continual adaptation, and robust vision under distribution shift.

SANTA is directly from that lab and addresses the limitations of simpler TTA methods.

Our project can honestly say:

```text
I started with a controlled Tent-style TTA study.
The natural next step is source anchoring, as studied in SANTA.
```

This shows that you understand where your project sits in the research landscape.

---

# 5. Exact Interview-Level Explanation Of Our Project

## Statement

I trained a source model on clean CIFAR-10.

## Explanation

This gives us the source-trained classifier.

The model learns from clean images only.

This is similar to many real settings where the training data is cleaner or more controlled than deployment data.

---

## Statement

I evaluated the source model on CIFAR-10-C.

## Explanation

This tests how the model behaves under corruption shift.

If accuracy drops badly, that shows the model is not robust to common corruptions.

This creates the need for adaptation.

---

## Statement

I compared source-only inference, BN adaptation, Tent, and UG-Tent.

## Explanation

Each method answers a different question.

`Source-only`:

```text
What happens if we do no adaptation?
```

`BN adaptation`:

```text
Can target batch statistics alone help?
```

`Tent`:

```text
Can entropy minimization improve target performance?
```

`UG-Tent`:

```text
Can confidence-gated entropy minimization reduce harmful updates?
```

---

## Statement

The main analysis is not only mean accuracy but also negative transfer.

## Explanation

Mean accuracy tells us which method is best on average.

But negative transfer tells us when adaptation is dangerous.

Example:

```text
source-only: 50%
Tent: 42%
```

Tent is worse than doing nothing.

This matters because a practical TTA method should not improve one condition while collapsing in another.

---

## Statement

Update ratio is important for UG-Tent.

## Explanation

UG-Tent may skip uncertain samples.

Update ratio tells us:

```text
what fraction of samples were actually used for adaptation
```

If update ratio is high:

```text
many samples passed the confidence gate
```

If update ratio is low:

```text
few samples passed the confidence gate
```

This helps interpret results.

For example:

```text
UG-Tent performs better and update ratio is moderate
```

This may mean gating helped filter bad updates.

But:

```text
UG-Tent performs same as source-only and update ratio is near zero
```

This may mean it barely adapted.

---

# 6. Safe Claims And Unsafe Claims

## Statement

Safe claim: this is a small empirical TTA study.

## Explanation

This is accurate because we are running a controlled benchmark and comparing methods.

Good wording:

```text
Implemented a CIFAR-10-C test-time adaptation study comparing source-only, BN adaptation, Tent, and uncertainty-gated Tent across corruption severities.
```

---

## Statement

Unsafe claim: this is a new state-of-the-art method.

## Explanation

We are not running a full benchmark suite.

We are not comparing against many modern TTA methods.

We are not testing ImageNet-C, DomainNet, medical imaging, or remote sensing.

So we should not claim:

```text
state-of-the-art
novel method
solves TTA
outperforms all baselines
```

The correct framing is:

```text
reproducible implementation + baseline comparison + small modification + failure analysis
```

That is still useful for an RA application.

---

# 7. What To Study Next

## Statement

First, understand entropy minimization deeply.

## Explanation

Entropy minimization is the center of Tent and many TTA methods.

You should be able to explain:

- what entropy means
- why minimizing entropy may help
- why it may fail
- how it works without labels

---

## Statement

Second, understand BatchNorm.

## Explanation

BatchNorm is central because Tent and SANTA both adapt normalization-related parameters.

You should know:

- what mean and variance do
- what gamma and beta do
- why source statistics may fail on target data
- why adapting BatchNorm is cheaper than adapting the whole model

---

## Statement

Third, understand negative transfer.

## Explanation

Negative transfer is the practical danger of adaptation.

If adaptation can make the model worse, then a real deployment system needs safeguards.

Our UG-Tent modification is a simple safeguard.

SANTA's source anchoring is a stronger safeguard.

---

## Statement

Fourth, understand where this project is limited.

## Explanation

Limitations make your explanation more credible.

You should say:

```text
This is a small CIFAR-10-C study. It is not a full reproduction of SANTA or pSTarC. I used it to understand the TTA pipeline, compare baselines, and study negative transfer. The next step would be source anchoring or a continual corruption sequence.
```

That is technically honest and research-aware.

