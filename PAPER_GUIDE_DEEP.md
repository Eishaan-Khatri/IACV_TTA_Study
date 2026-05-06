# Deep Guide To The Three Papers Behind The TTA Project

This is the detailed study document. It is written for someone new to computer vision, test-time adaptation, and robustness.

It covers three papers:

1. **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**  
   Dan Hendrycks and Thomas Dietterich, ICLR 2019.  
   Main use for us: benchmark design, CIFAR-10-C / ImageNet-C, corruption robustness metrics.

2. **Tent: Fully Test-Time Adaptation by Entropy Minimization**  
   Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell, ICLR 2021.  
   Main use for us: source-free test-time adaptation baseline.

3. **SANTA: Source Anchoring Network and Target Alignment for Continual Test Time Adaptation**  
   Goirik Chakrabarty, Manogna Sreenivas, Soma Biswas, TMLR 2023.  
   Main use for us: IACV connection, continual TTA, source anchoring, target alignment.

Primary sources:

- CIFAR-C / ImageNet-C: https://openreview.net/forum?id=HJz6tiCqYm
- Tent: https://openreview.net/forum?id=uXl3bZLkr3c
- SANTA: https://openreview.net/forum?id=V7guVYzvE4

Important constraint: this document does not copy the papers line by line. It explains the technical content section by section, equation by equation, and experiment by experiment in original words.

---

# 0. Vocabulary You Need Before Reading The Papers

## 0.1 Image classification

### Paper-level idea

A classifier receives an image and predicts one class label.

### What it means

The model is a function:

```text
f_theta(x) -> y_hat
```

Where:

- `x` is the input image.
- `theta` means the model parameters.
- `y_hat` means the predicted label.

In modern deep learning, the model usually outputs logits first:

```text
z = f_theta(x)
```

Then logits are converted into probabilities:

```text
p = softmax(z)
```

If the class probabilities are:

```text
airplane: 0.01
car:      0.02
bird:     0.01
cat:      0.08
dog:      0.84
truck:    0.04
```

Then the model predicts `dog`.

### Why it matters for this project

All three papers are about what happens when this basic classifier faces data that does not look like its training data.

---

## 0.2 Source domain and target domain

### Paper-level idea

The source domain is where the model is trained. The target domain is where the model is tested or deployed.

### What it means

If we train on clean CIFAR-10:

```text
source domain = clean CIFAR-10
```

If we test on corrupted CIFAR-10-C:

```text
target domain = CIFAR-10-C
```

The model learns source patterns, but target images may look different.

### Why it matters for this project

Test-time adaptation exists because source performance does not guarantee target performance.

---

## 0.3 Dataset shift / domain shift

### Paper-level idea

Dataset shift means source and target data come from different distributions.

### What it means

Written mathematically:

```text
P_source(x, y) != P_target(x, y)
```

Where:

- `x` is the image.
- `y` is the label.
- `P_source` is the source distribution.
- `P_target` is the target distribution.

If source images are clean and target images are noisy, the image distribution has changed.

### Concrete example

Source:

```text
clear daylight road images
```

Target:

```text
night images with rain and motion blur
```

Even if the classes are the same, the pixels and features are different.

### Why it matters for this project

CIFAR-10-C deliberately creates a target shift from clean images to corrupted images.

---

## 0.4 Robustness

### Paper-level idea

Robustness means a model keeps working when the input changes in realistic ways.

### What it means

A non-robust model might do this:

```text
clean image accuracy: 90%
foggy image accuracy: 35%
```

A more robust model might do this:

```text
clean image accuracy: 88%
foggy image accuracy: 70%
```

The second model loses less performance under shift.

### Why it matters for IACV

IACV works on data-efficient and robust learning under constraints. Medical images, satellite images, surveillance images, and low-quality images all create robustness problems.

---

## 0.5 Test-time adaptation

### Paper-level idea

Test-time adaptation updates the model while testing, using target data.

### What it means

Standard evaluation:

```text
train model
freeze model
test model
```

Test-time adaptation:

```text
train model
receive test batch
update model using test batch
predict
continue
```

The difficult part:

```text
target labels are not available
```

So the model must adapt using unsupervised signals.

### Why it matters for this project

Our project compares:

```text
source-only
BN adaptation
Tent
uncertainty-gated Tent
```

Only the last three adapt during testing.

---

## 0.6 Entropy

### Paper-level idea

Entropy measures prediction uncertainty.

### What it means

Formula:

```text
H(p) = - sum over classes c of p_c * log(p_c)
```

Low entropy:

```text
dog: 0.97
cat: 0.01
truck: 0.02
```

The model is confident.

High entropy:

```text
dog: 0.35
cat: 0.33
truck: 0.32
```

The model is confused.

### Why it matters for Tent

Tent adapts by minimizing entropy. It tries to make predictions more confident.

---

## 0.7 Batch Normalization

### Paper-level idea

BatchNorm normalizes internal features using mean and variance, then applies a learnable scale and shift.

### What it means

For an activation `a`, BatchNorm roughly does:

```text
a_norm = (a - mean) / sqrt(variance + epsilon)
output = gamma * a_norm + beta
```

Where:

- `mean` and `variance` describe feature statistics.
- `gamma` is a learnable scale.
- `beta` is a learnable shift.
- `epsilon` prevents division by zero.

### Why it matters for TTA

When the target distribution changes, feature statistics change too.

Example:

```text
source features: clean images
target features: foggy images
```

Updating BatchNorm statistics or affine parameters is a cheap way to adapt to target data.

---

# 1. Paper One: Benchmarking Neural Network Robustness To Common Corruptions And Perturbations

## 1.1 What this paper is trying to solve

### Paper-level idea

The paper argues that computer vision models should be tested on common corruptions, not only clean images or adversarial attacks.

### What it means

Before this paper, a lot of robustness discussion centered on adversarial examples. Those are carefully designed small changes intended to fool a classifier.

But real images often suffer from ordinary degradation:

```text
noise
blur
fog
snow
frost
brightness shifts
contrast shifts
compression artifacts
pixelation
```

The paper says these corruptions are important because they occur naturally in deployed systems.

### Why it matters

A model can be strong on clean benchmark data but unreliable in deployment. This paper creates a standardized way to measure that problem.

---

## 1.2 Human vision vs machine vision

### Paper-level idea

The introduction contrasts human robustness with model fragility.

### What it means

Humans can usually recognize an object even if the image is noisy, blurry, snowy, foggy, or compressed.

Many neural networks fail sharply under those same corruptions.

### Why it matters

This motivates the benchmark. If models are intended for real-world applications, clean test accuracy is insufficient.

---

## 1.3 Adversarial robustness vs corruption robustness

### Paper-level idea

The paper distinguishes common corruption robustness from adversarial robustness.

### What it means

Adversarial robustness asks:

```text
Can the model survive worst-case small perturbations chosen by an attacker?
```

Corruption robustness asks:

```text
Can the model survive common image quality changes that happen naturally?
```

Adversarial example:

```text
tiny optimized pixel changes designed to fool ResNet
```

Common corruption:

```text
blur caused by camera motion
```

### Why it matters

Our project is about common corruption robustness, not adversarial robustness.

---

## 1.4 Formal corruption robustness idea

### Paper-level idea

The paper defines corruption robustness as average performance under corruption functions.

### What it means

Suppose:

```text
f = classifier
x = image
y = true label
c = corruption function
```

A corruption function might do:

```text
c(x) = foggy version of x
```

Clean accuracy asks:

```text
Does f(x) = y?
```

Corruption robustness asks:

```text
Does f(c(x)) = y?
```

And not just for one corruption, but averaged over many corruptions.

### Why it matters

This is exactly our setup:

```text
source model trained on clean x
evaluated on corrupted c(x)
```

---

## 1.5 ImageNet-C

### Paper-level idea

ImageNet-C is a corrupted version of the ImageNet validation set.

### What it means

ImageNet is a large image classification dataset. ImageNet-C applies 15 corruption types to ImageNet validation images.

The 15 corruption types are grouped into four broad categories:

```text
noise
blur
weather
digital
```

The paper uses five severity levels for each corruption.

So:

```text
15 corruption types x 5 severities = 75 corruption settings
```

### Why it matters

This became a standard robustness benchmark. CIFAR-10-C follows the same idea at smaller scale.

---

## 1.6 CIFAR-10-C

### Paper-level idea

CIFAR-10-C is a smaller corrupted benchmark derived from CIFAR-10.

### What it means

CIFAR-10 contains 10 classes:

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

CIFAR-10-C corrupts the CIFAR-10 test images.

For each corruption:

```text
severity 1: mild
severity 2: low-medium
severity 3: medium
severity 4: high
severity 5: severe
```

Labels do not change.

If a clean image is labeled `dog`, its corrupted version is still `dog`.

### Why it matters

We use CIFAR-10-C because it is small enough for Colab and standard enough to be credible.

---

## 1.7 Corruption types explained

### Gaussian noise

Noise added from a Gaussian distribution. Think random pixel-level grain.

Why it occurs:

```text
low light
sensor noise
thermal noise
poor camera conditions
```

Why models fail:

Noise disrupts local edges, textures, and color patterns.

---

### Shot noise

Noise related to the discrete nature of light and sensor measurement.

Why it occurs:

```text
photon counting noise
low-light imaging
sensor limitations
```

Why models fail:

It creates random intensity variations that can corrupt fine details.

---

### Impulse noise

Also called salt-and-pepper noise.

Why it occurs:

```text
bit errors
dead pixels
transmission problems
```

Why models fail:

It inserts extreme pixel values and can destroy local patterns.

---

### Defocus blur

Blur caused by the camera being out of focus.

Why it occurs:

```text
wrong focal plane
cheap camera
fast capture
bad autofocus
```

Why models fail:

Edges become weak. Object boundaries and textures become unclear.

---

### Motion blur

Blur caused by camera or object movement.

Why it occurs:

```text
moving vehicle
walking person
shaky camera
low shutter speed
```

Why models fail:

Important visual structures smear across pixels.

---

### Fog

Fog reduces contrast and hides distant structure.

Why it occurs:

```text
weather
smoke
atmospheric haze
low contrast conditions
```

Why models fail:

Objects lose edges and contrast, and background/foreground separation becomes harder.

---

### Brightness

Brightness corruption changes light intensity.

Why it occurs:

```text
daylight variation
overexposure
underexposure
camera exposure settings
```

Why models fail:

Feature distributions shift even if object shapes remain visible.

---

### Contrast

Contrast changes how separated dark and bright areas are.

Why it occurs:

```text
lighting
weather
camera settings
image preprocessing
```

Why models fail:

Low contrast makes edges and textures less discriminative.

---

### Pixelate

Pixelation simulates low-resolution image artifacts.

Why it occurs:

```text
resizing
low-resolution cameras
compression
remote surveillance
```

Why models fail:

Fine details disappear.

---

### JPEG compression

JPEG introduces blocky compression artifacts.

Why it occurs:

```text
image storage
web compression
video compression
bandwidth limits
```

Why models fail:

Artificial block patterns and loss of detail shift the visual distribution.

---

## 1.8 ImageNet-P

### Paper-level idea

ImageNet-P tests prediction stability under small perturbation sequences.

### What it means

ImageNet-C asks:

```text
Is the prediction correct under corruption?
```

ImageNet-P asks:

```text
Does the prediction remain stable under small frame-to-frame perturbations?
```

Example sequence:

```text
same image rotated slightly over time
same image translated by one pixel at a time
same image with gradually changing brightness
```

### Why it matters

A model should not change predictions chaotically when the image changes slightly.

### Why it matters less for our current project

We are using CIFAR-10-C, not CIFAR-10-P. Our focus is corruption accuracy, not prediction stability sequences.

---

## 1.9 Corruption Error, mCE, Relative mCE

### Paper-level idea

The paper measures error across corruptions and severities, then normalizes by a baseline model.

### What it means

For a model `f`:

```text
E_f_s,c = error of model f at severity s for corruption c
```

For each corruption `c`, sum over five severities:

```text
sum_s E_f_s,c
```

Then divide by baseline error, originally AlexNet in ImageNet-C:

```text
CE_f_c = sum_s E_f_s,c / sum_s E_AlexNet_s,c
```

Then average across corruption types:

```text
mCE = average of CE values over all corruptions
```

### Why normalize?

Some corruptions are naturally harder than others.

Example:

```text
fog may obscure the object heavily
brightness may preserve shape
```

Normalization tries to account for different corruption difficulty.

### What Relative mCE means

Relative mCE tries to measure performance drop relative to clean performance.

A model with excellent clean accuracy but huge corruption drop may not be truly robust.

### What we use instead

Our project uses simpler metrics:

```text
accuracy by corruption
accuracy by severity
mean accuracy
negative transfer cases
```

We do not need official ImageNet mCE for a Colab-scale project.

---

## 1.10 Main experimental finding

### Paper-level idea

Clean accuracy improved across architectures, but corruption robustness did not automatically improve in the same proportion.

### What it means

Newer architectures may improve clean ImageNet error, but still remain fragile under corruptions.

The paper's broader message:

```text
better clean benchmark performance does not guarantee robust deployment performance
```

### Why it matters for us

If our source model gets good clean CIFAR-10 accuracy, that alone does not prove it is robust. We need CIFAR-10-C evaluation.

---

## 1.11 How this paper maps to our code

### Our dataset loading

In `src/data.py`, we download CIFAR-10-C and load corruption files.

Each corruption file contains all five severities.

The code slices:

```text
severity 1: rows 0 to 9999
severity 2: rows 10000 to 19999
severity 3: rows 20000 to 29999
severity 4: rows 30000 to 39999
severity 5: rows 40000 to 49999
```

### Our first-pass corruptions

We use:

```text
gaussian_noise
shot_noise
motion_blur
brightness
fog
```

This gives:

```text
5 corruptions x 5 severities = 25 evaluations
```

### Our project claim

Safe claim:

```text
I evaluated test-time adaptation methods on a standard corruption robustness benchmark, CIFAR-10-C.
```

Unsafe claim:

```text
I fully reproduced ImageNet-C robustness evaluation.
```

We are not doing full ImageNet-C.

---

# 2. Paper Two: Tent

## 2.1 What problem Tent solves

### Paper-level idea

Tent adapts a trained model during testing using only unlabeled target data and the model's own parameters.

### What it means

Tent assumes:

```text
you already trained a model
you now receive target test images
target labels are unavailable
source training data is unavailable
the model must keep making predictions
```

This is stricter than ordinary domain adaptation.

### Why it matters

In many deployments, you cannot collect labels before adapting. You also may not have source data due to privacy, storage, or deployment constraints.

---

## 2.2 Why fully test-time adaptation is different from fine-tuning

### Paper-level idea

Fine-tuning needs labeled target data.

### What it means

Fine-tuning:

```text
needs target images + target labels
```

Fully test-time adaptation:

```text
needs only target images
```

Fine-tuning loss:

```text
cross_entropy(prediction, true_label)
```

Tent loss:

```text
entropy(prediction)
```

### Why it matters

Tent is useful when target labels are unavailable.

---

## 2.3 Why fully test-time adaptation is different from unsupervised domain adaptation

### Paper-level idea

Unsupervised domain adaptation usually has source data and target data together during adaptation.

### What it means

Unsupervised domain adaptation often assumes:

```text
labeled source data
unlabeled target data
```

It can train losses that compare source and target domains.

Fully test-time adaptation assumes:

```text
no source data
unlabeled target data only
```

### Why it matters

Tent is more restrictive and more deployment-friendly.

---

## 2.4 Why fully test-time adaptation is different from test-time training

### Paper-level idea

Test-time training changes the training procedure by adding a self-supervised auxiliary task.

### What it means

Test-time training may train the model with:

```text
supervised classification loss
self-supervised rotation loss or other auxiliary loss
```

Then during testing, it updates the model using the auxiliary self-supervised task.

Tent does not require modifying source training.

### Why it matters

Tent can be applied to an already trained classifier without redesigning training.

---

## 2.5 Tent's central assumption

### Paper-level idea

Lower prediction entropy often correlates with lower error under corruption shift.

### What it means

If the model is more confident, it is often more likely to be correct.

This is not always true, but it is useful enough to become an adaptation signal.

Example:

```text
high entropy: model unsure between cat/dog/deer
low entropy: model strongly predicts dog
```

Tent uses this relationship as unsupervised supervision.

### Why it matters

No labels are needed. The model's own confidence becomes the learning signal.

---

## 2.6 Entropy objective

### Paper-level idea

Tent minimizes Shannon entropy of the predicted probability distribution.

### Formula

For prediction probabilities `p_c` over classes:

```text
H(p) = - sum_c p_c * log(p_c)
```

Tent minimizes:

```text
L_tent(x_t) = H(f_theta(x_t))
```

For a batch:

```text
L_tent(batch) = average entropy over target batch
```

### What each symbol means

- `x_t`: target image.
- `f_theta`: model with parameters theta.
- `p_c`: predicted probability for class c.
- `H`: entropy.

### Why entropy minimization can work

If target samples form clusters, entropy minimization pushes decision boundaries away from dense target regions.

This is related to a common semi-supervised learning assumption:

```text
decision boundaries should not cut through high-density clusters
```

---

## 2.7 Why optimizing one prediction alone is dangerous

### Paper-level idea

If you optimize a single prediction only for low entropy, the trivial solution is to put all probability on one class.

### What it means

For one sample, entropy is minimized by:

```text
class A: 1.0
all others: 0.0
```

But that does not mean class A is correct.

### How Tent avoids the worst version of this

Tent optimizes shared model parameters over batches, not an independent probability vector for each sample.

Changing shared parameters means each update affects many samples through the model structure.

Still, this does not eliminate the risk of wrong confident predictions.

---

## 2.8 Feature modulation

### Paper-level idea

Tent adapts by modulating features rather than updating the full model.

### What it means

Inside a neural network, each layer creates internal feature maps.

Tent changes how features are normalized and shifted/scaled.

This is done through BatchNorm.

### Why not update everything?

Updating the full network at test time can:

```text
overfit to target batches
destroy learned source features
increase compute
increase memory
cause instability
```

BatchNorm affine adaptation is a smaller controlled update.

---

## 2.9 BatchNorm in Tent

### Paper-level idea

Tent estimates target normalization statistics and optimizes channel-wise affine parameters.

### What it means

BatchNorm:

```text
x_norm = (x - mean) / std
x_out = gamma * x_norm + beta
```

Tent does two things:

1. Estimate `mean` and `std` from target data.
2. Optimize `gamma` and `beta` by entropy loss.

### Why this is efficient

`gamma` and `beta` are a tiny fraction of total model parameters.

Updating them is cheaper and less destructive than updating all convolution weights.

---

## 2.10 Tent algorithm

### Paper-level idea

For each target batch, calculate entropy, backpropagate, update selected parameters, and continue.

### Pseudo-code

```python
model = configure_for_tent(model)
optimizer = make_optimizer(batchnorm_affine_params)

for x_target in target_loader:
    logits = model(x_target)
    probs = softmax(logits)
    loss = entropy(probs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    prediction = probs.argmax(dim=1)
```

### What `configure_for_tent` does

It usually:

```text
sets model to train mode for BatchNorm behavior
freezes most parameters
enables gradients only for BatchNorm affine parameters
uses target batch statistics
```

### How this maps to our code

Our `src/tta.py` has:

```text
evaluate_tent(...)
configure_tent(...)
entropy(...)
```

The code follows the same core idea.

---

## 2.11 Baselines in Tent

### Source model

No adaptation.

Question answered:

```text
How bad is the domain shift if we do nothing?
```

### Test-time normalization / BN adaptation

Update only normalization statistics from target batches.

Question answered:

```text
How much improvement comes merely from target BatchNorm statistics?
```

### Pseudo-labeling

Use model predictions as labels, then train on those pseudo-labels.

Risk:

```text
wrong pseudo-labels reinforce wrong predictions
```

### Tent

Use entropy minimization instead of hard pseudo-labels.

Question answered:

```text
Can confidence itself be a useful unsupervised test-time objective?
```

---

## 2.12 Tent experiments

### Corrupted image classification

The paper evaluates on:

```text
ImageNet-C
CIFAR-10-C
CIFAR-100-C
```

Purpose:

```text
test robustness to common corruptions
```

### Digit domain adaptation

The paper evaluates shifts like:

```text
SVHN -> MNIST
SVHN -> MNIST-M
SVHN -> USPS
```

Purpose:

```text
test source-free domain adaptation beyond corruptions
```

### Semantic segmentation

The paper evaluates simulation-to-real segmentation:

```text
GTA -> Cityscapes
```

Purpose:

```text
show Tent can apply beyond classification
```

### Why this matters

Tent is not only a CIFAR trick. The authors test across multiple tasks and shifts.

---

## 2.13 Tent's main reported message

### Paper-level idea

Tent improves over source-only and test-time normalization on corruption benchmarks, with no source data and no target labels.

### What it means

The method can improve adaptation using only target data.

This is why it became a standard TTA baseline.

### Why it matters for our project

If we compare against Tent, reviewers understand the baseline.

---

## 2.14 Limitations of Tent

### Limitation 1: confirmation bias

If predictions are wrong, entropy minimization can reinforce wrong predictions.

Example:

```text
true class: dog
model predicts cat confidently
Tent makes cat even more confident
```

### Limitation 2: batch dependence

Tent often depends on batch statistics.

Small or biased batches can produce unstable updates.

### Limitation 3: continual drift

If the model keeps adapting across changing domains, it may forget source knowledge.

### Limitation 4: no explicit source anchor

Tent has no strong mechanism that says:

```text
do not move too far from useful source behavior
```

### Why these limitations matter

These limitations motivate later methods like CoTTA, EATA, SANTA, and others.

---

## 2.15 Our UG-Tent modification

### Idea

Tent adapts on all target samples. UG-Tent adapts only on confident samples.

### Formula

For sample `i`:

```text
confidence_i = max_c p_i,c
```

Mask:

```text
m_i = 1 if confidence_i >= threshold else 0
```

Loss:

```text
L_UG = sum_i m_i * H(p_i) / sum_i m_i
```

If too few samples pass the gate, skip the update.

### Why this is reasonable

Low-confidence samples are more likely to be unreliable adaptation signals.

UG-Tent tests whether filtering them reduces negative transfer.

### Why this may fail

Confidence is not correctness.

A model can be confidently wrong.

Also, if the threshold is too high, the model may barely adapt.

### What we measure

We need:

```text
accuracy
mean entropy
update ratio
negative transfer cases
```

Update ratio is crucial because it tells us how often UG-Tent actually updates.

---

# 3. Paper Three: SANTA

## 3.1 What problem SANTA solves

### Paper-level idea

SANTA addresses continual test-time adaptation under changing target environments.

### What it means

Instead of one fixed target domain:

```text
clean -> fog
```

SANTA handles sequences:

```text
clean -> fog -> blur -> noise -> brightness -> contrast -> snow
```

The model keeps adapting over time.

### Why it matters

Real deployments are dynamic:

```text
weather changes
lighting changes
camera quality changes
sensor source changes
hospital/scanner changes
satellite region/season changes
```

---

## 3.2 SANTA's practical requirements

### Paper-level idea

The method should work with small batch sizes, preserve source performance, and need minimal hyperparameters/storage.

### Requirement 1: small batch sizes

Real systems may not receive large batches.

A deployed system might process:

```text
1 image
10 images
25 images
```

If a TTA method only works with batch size 200, it may be impractical.

### Requirement 2: source performance should not collapse

If a model adapts to rainy weather, it should still work when clear weather returns.

This is source preservation.

### Requirement 3: minimal hyperparameters

At test time, there may be no validation set.

If a method needs extensive tuning, it is hard to deploy.

### Requirement 4: minimal storage

Storing multiple full copies of a model can be expensive.

SANTA tries to avoid unnecessary storage overhead.

---

## 3.3 Continual TTA problem setting

### Paper-level idea

The source model is trained on source data. During testing, source data is unavailable and target domains change over time.

### What it means

Source training:

```text
D_s = {(x_i, y_i)}
```

Target testing:

```text
D_t = {x_j}
```

No target labels:

```text
y_j unavailable
```

Changing target domains:

```text
P_t_1 != P_t_2 != ... != P_s
```

### Why it matters

This is more realistic than adapting to one fixed target distribution.

---

## 3.4 Main SANTA components

### Paper-level idea

SANTA has two major components:

```text
Source Anchorization Network
Target Alignment
```

### What it means

Source Anchorization:

```text
keep adapted model tied to useful source behavior
```

Target Alignment:

```text
make target features cluster in a semantically meaningful way
```

These two components address two different risks:

```text
forgetting source knowledge
poor target feature organization
```

---

## 3.5 Adapting model and source model

### Paper-level idea

SANTA keeps a source model and an adapting model.

### What it means

Source model:

```text
original trained model
```

Adapting model:

```text
model updated during test time
```

Initially:

```text
adapting model weights = source model weights
```

As target batches arrive, the adapting model changes.

### Why it matters

The source model acts like an anchor. It helps prevent the adapting model from drifting too far.

---

## 3.6 Target-corrected source model

### Paper-level idea

SANTA changes BN statistics of the source model using the current target batch, creating a target-corrected source response.

### What it means

The source model weights are not fully trained on target data, but its BatchNorm statistics can be adjusted to the current target batch.

This gives a model response that is:

```text
source-knowledge-based
but target-statistics-corrected
```

### Why it matters

If the pure source model uses old clean statistics, its predictions on target data may be poor.

Updating BN statistics to the target batch makes the source anchor more relevant to the current domain.

---

## 3.7 Source anchoring loss

### Paper-level idea

The adapting model is encouraged to match the target-corrected source model's predictions.

### Variables

For target sample `i` and class `j`:

```text
p_ij = prediction score from adapting model
a_ij = prediction score from target-corrected source model
```

Source anchoring loss has cross-entropy-like form:

```text
L_SA = - average over i,j of p_ij * log(a_ij)
```

The paper also includes augmented target images in the anchoring loss.

### What this means

The adapting model is not free to move anywhere. It is guided by the source model's target-corrected response.

### Why this matters

This reduces catastrophic forgetting.

---

## 3.8 Why augmentation is used

### Paper-level idea

SANTA uses augmented versions of target images to improve robustness.

### What it means

For an image `x`, create an augmented version:

```text
x_aug
```

This might involve transformations such as cropping, flipping, color changes, etc., depending on implementation.

The model is encouraged to behave consistently on original and augmented versions.

### Why it matters

Augmentation creates a self-supervised signal:

```text
same underlying object
different view
features/predictions should remain coherent
```

---

## 3.9 Target Alignment

### Paper-level idea

SANTA uses contrastive learning and source prototypes to make target features cluster meaningfully.

### What it means

Feature extractor:

```text
g_phi(x) -> feature vector
```

Classifier:

```text
h(feature) -> class prediction
```

The model is decomposed as:

```text
f_theta = h composed with g_phi
```

Target alignment works at the feature level, not just prediction level.

### Why it matters

Good features should group same-class samples together and separate different-class samples.

---

## 3.10 Contrastive learning

### Paper-level idea

Contrastive learning pulls related views together and pushes unrelated samples apart.

### What it means

For a target image:

```text
x_i = original image
x_i_aug = augmented view
```

Their features should be close.

For different images, features should generally be farther apart unless they share semantic structure.

Basic contrastive intuition:

```text
positive pair: original image and its augmentation
negative pairs: other images in the batch
```

### Temperature

The contrastive loss uses a temperature parameter.

Temperature controls sharpness of similarity comparisons.

Small temperature:

```text
stronger separation pressure
```

Large temperature:

```text
softer separation pressure
```

---

## 3.11 Source prototypes

### Paper-level idea

SANTA uses source prototypes to guide target feature alignment.

### What is a prototype?

A prototype is a representative feature vector for a class.

Example:

```text
dog prototype = average feature representation of source dog images
cat prototype = average feature representation of source cat images
```

### Why prototypes are useful

They preserve semantic structure learned from source data.

Without prototypes, target contrastive learning might form clusters that are visually consistent but not aligned with source classes.

### How SANTA uses them

For a target feature, find the nearest source prototype by cosine similarity.

Conceptually:

```text
nearest prototype = class anchor in source feature space
```

Then use that prototype as an additional positive view for alignment.

---

## 3.12 Source-prototype-driven contrastive alignment

### Paper-level idea

Target features are aligned not only with augmented views, but also with nearest source prototypes.

### What it means

For a target sample, positives include:

```text
augmented target view
nearest source prototype
```

This pushes the target feature toward semantically meaningful source structure.

### Why it matters

It prevents target clusters from drifting into arbitrary groupings.

Example:

Bad target adaptation:

```text
clusters form based on corruption type
```

Better target adaptation:

```text
clusters form based on object class
```

Source prototypes help with the second.

---

## 3.13 What SANTA updates

### Paper-level idea

SANTA updates only BN affine parameters in the adapting model.

### What it means

It updates:

```text
gamma
beta
```

It keeps most of the model fixed.

### Why it matters

This reduces overfitting and storage cost.

It is aligned with the Tent philosophy but adds better anchoring and alignment.

---

## 3.14 How SANTA differs from Tent

### Tent

```text
minimize entropy on target predictions
update BN affine parameters
no explicit source anchor
```

### SANTA

```text
use source anchoring self-distillation
use target alignment with source prototypes
update BN affine parameters
designed for continual test-time adaptation
```

### Core difference

Tent asks:

```text
Can we make target predictions more confident?
```

SANTA asks:

```text
Can we adapt to target shifts while preserving source knowledge and semantic feature structure?
```

---

## 3.15 SANTA baselines

### Source

No adaptation.

### BN Stats Adapt

Only adapt BatchNorm statistics.

### Tent-continual

Tent applied in continual adaptation setting.

### CoTTA

Continual Test-Time Adaptation method using teacher-student behavior, augmentation averaging, and stochastic restoration.

### RMT / PETAL and other methods

More recent continual TTA baselines with teacher-student or restoration mechanisms.

### Why baselines matter

SANTA is not compared only to weak source-only models. It is compared to methods designed for TTA/CTTA.

---

## 3.16 SANTA experiments

### Datasets

SANTA evaluates on:

```text
CIFAR-10-C
CIFAR-100-C
ImageNet-C
```

These are standard corruption benchmarks.

### Continual protocol

Instead of treating each corruption independently, continual TTA evaluates a sequence of changing corruptions.

This tests whether adaptation accumulates errors over time.

### Batch size study

SANTA specifically studies smaller batch sizes.

This is important because online deployment often cannot wait for huge batches.

### Source preservation

The method is also evaluated for whether it continues to perform well on source-like data.

---

## 3.17 SANTA's key result pattern

### Paper-level idea

SANTA is designed to remain strong under lower batch sizes and continual shifts while keeping storage/tuning practical.

### What it means

The method targets deployment constraints, not just best-case benchmark performance.

### Why it matters for IACV

This matches the lab's broader theme:

```text
robust learning under realistic constraints
limited data
domain shift
deployment practicality
```

---

## 3.18 How SANTA maps to our project

### What we do

We do:

```text
CIFAR-10-C
source-only
BN adaptation
Tent
UG-Tent
```

### What SANTA does beyond us

SANTA adds:

```text
continual sequence protocol
source anchoring
source prototypes
contrastive target alignment
small-batch analysis
source preservation
```

### Correct way to describe the relationship

Say:

```text
My project starts from the minimal Tent-style TTA setting. SANTA shows a stronger IACV-style direction: stabilizing continual adaptation using source anchoring and source-guided target alignment.
```

Do not say:

```text
I reproduced SANTA.
```

Unless we actually implement SANTA.

---

# 4. Exact Mapping From Papers To Our Implementation

## 4.1 CIFAR-C paper -> our benchmark

### Paper concept

Evaluate corruption robustness over corruption types and severity levels.

### Our implementation

Config:

```yaml
first_pass_corruptions:
  - gaussian_noise
  - shot_noise
  - motion_blur
  - brightness
  - fog
severities: [1, 2, 3, 4, 5]
```

### What this means

We evaluate 25 target conditions.

This is not the full benchmark, but it is enough for a focused first-pass study.

---

## 4.2 Tent paper -> our baseline

### Paper concept

Minimize entropy at test time and update BN affine parameters.

### Our implementation

In `src/tta.py`, Tent:

```text
sets model for adaptation
enables gradients for BN affine parameters
computes entropy loss
updates on target batches
records accuracy and entropy
```

### What this means

We are implementing the core Tent mechanism, not every experiment from the paper.

---

## 4.3 SANTA paper -> our next-step context

### Paper concept

Use source anchoring and target alignment to stabilize continual TTA.

### Our implementation

We do not implement SANTA yet.

But our UG-Tent tests one smaller stabilization idea:

```text
avoid updating on uncertain target samples
```

### What this means

Our project is a stepping stone toward SANTA-like thinking.

---

# 5. What You Must Be Able To Explain In An Interview

## 5.1 Why CIFAR-10-C?

Answer:

```text
CIFAR-10-C is a standard corruption robustness benchmark. It lets me evaluate how a clean-source model behaves under controlled target shifts across corruption type and severity.
```

Explain further:

```text
The classes stay the same, but the input distribution changes. That makes it useful for studying test-time adaptation.
```

---

## 5.2 What is Tent?

Answer:

```text
Tent is a fully test-time adaptation method that minimizes prediction entropy on unlabeled target batches, updating mainly BatchNorm affine parameters.
```

Explain further:

```text
It uses the model's own confidence as an unsupervised signal. It does not need target labels or source data.
```

---

## 5.3 Why can Tent fail?

Answer:

```text
Entropy minimization can reinforce wrong predictions. If the model is wrong but becomes more confident, adaptation can hurt performance.
```

Explain further:

```text
This is negative transfer. It is especially likely under severe corruptions where the model's predictions are unreliable.
```

---

## 5.4 What is UG-Tent?

Answer:

```text
UG-Tent is a simple confidence-gated version of Tent. It updates only on target samples whose maximum softmax confidence exceeds a threshold.
```

Explain further:

```text
The goal is to test whether skipping uncertain samples reduces harmful updates under severe corruption.
```

---

## 5.5 How is this connected to SANTA?

Answer:

```text
SANTA addresses the harder continual TTA setting using source anchoring and target alignment. My project starts with a smaller Tent-style study and tests a simple stabilization mechanism. SANTA is the natural next step for stronger anti-drift adaptation.
```

---

## 5.6 What is the limitation of your project?

Answer:

```text
It is a small CIFAR-10-C first-pass study. It does not claim state-of-the-art or full SANTA reproduction. The value is in implementing a known TTA baseline, comparing methods, adding a simple modification, and analyzing negative transfer.
```

---

# 6. What Would Make The Project Stronger After First Results

## 6.1 Add full CIFAR-10-C corruptions

Instead of 5 corruptions:

```text
all 15 corruptions
```

This gives:

```text
15 x 5 = 75 settings
```

---

## 6.2 Add CIFAR-100-C

CIFAR-100-C is harder because there are 100 classes.

This tests whether the method works when classification is more fine-grained.

---

## 6.3 Add continual corruption order

Instead of resetting for each corruption, evaluate a sequence:

```text
gaussian_noise -> shot_noise -> motion_blur -> brightness -> fog
```

This moves closer to SANTA.

---

## 6.4 Add source anchoring

A simple next step:

```text
keep a frozen source model
penalize adapted model if it moves too far from source predictions
```

This would be closer to SANTA but still manageable.

---

## 6.5 Add calibration metrics

Confidence gating depends on confidence quality.

So measure:

```text
expected calibration error
confidence vs accuracy
entropy vs correctness
```

This would make the UG-Tent analysis more rigorous.

---

# 7. Safe CV Language After Results

Use this only after we inspect actual results.

## Conservative bullet

```text
Implemented a CIFAR-10-C test-time adaptation study comparing source-only inference, BN adaptation, Tent-style entropy minimization, and uncertainty-gated Tent across corruption types and severity levels.
```

## Stronger bullet if UG-Tent helps

```text
Built and evaluated an uncertainty-gated Tent variant for CIFAR-10-C corruption shift, analyzing update ratio, entropy, and negative-transfer cases across 25 corruption/severity settings.
```

## Strongest bullet only if results clearly support it

```text
Reduced negative-transfer cases in Tent-style test-time adaptation on CIFAR-10-C by filtering entropy-minimization updates using predictive confidence, with analysis across 25 corruption/severity settings.
```

---

# 8. Unsafe Language To Avoid

Do not say:

```text
state-of-the-art
novel TTA method
full reproduction of SANTA
solved domain adaptation
proved confidence gating is generally better
```

Why:

```text
The project is small-scale and first-pass. Strong claims require broader benchmarks and comparisons.
```

Correct framing:

```text
controlled empirical study
baseline reproduction
simple extension
failure analysis
preparation for robust CV/TTA research
```

