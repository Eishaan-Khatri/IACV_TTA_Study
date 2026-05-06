from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvalResult:
    method: str
    corruption: str
    severity: int
    accuracy: float
    mean_entropy: float
    update_ratio: float = 1.0


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)


def normalized_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    entropy = entropy_from_logits(logits)
    num_classes = logits.shape[1]
    return entropy / math.log(num_classes)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def clone_model(model: nn.Module, device: torch.device) -> nn.Module:
    cloned = copy.deepcopy(model)
    cloned.to(device)
    return cloned


def configure_tent(model: nn.Module) -> list[nn.Parameter]:
    """Configure model for Tent-style adaptation.

    We update only BatchNorm affine parameters. BatchNorm layers remain in
    train mode so target batch statistics are used.
    """
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)

    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
                params.extend([module.weight, module.bias])
    return params


@torch.no_grad()
def evaluate_source(
    model: nn.Module,
    loader,
    device: torch.device,
    method_name: str,
    corruption: str,
    severity: int,
) -> EvalResult:
    model.eval()
    correct = 0
    total = 0
    entropies = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
        entropies.append(entropy_from_logits(logits).detach().cpu())
    entropy = torch.cat(entropies).mean().item()
    return EvalResult(method_name, corruption, severity, correct / total, entropy, 0.0)


@torch.no_grad()
def evaluate_bn_adapt(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
) -> EvalResult:
    adapted = clone_model(model, device)
    adapted.train()
    correct = 0
    total = 0
    entropies = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = adapted(images)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
        entropies.append(entropy_from_logits(logits).detach().cpu())
    entropy = torch.cat(entropies).mean().item()
    return EvalResult("bn_adapt", corruption, severity, correct / total, entropy, 1.0)


def evaluate_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    confidence_threshold: float | None = None,
    min_selected_fraction: float = 0.10,
    method_name: str | None = None,
) -> EvalResult:
    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)
    method = method_name or ("ug_tent" if confidence_threshold is not None else "tent")

    correct = 0
    total = 0
    entropies = []
    selected_total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = adapted(images)
        probs = F.softmax(logits, dim=1)
        entropy = entropy_from_logits(logits)

        if confidence_threshold is None:
            selected = torch.ones_like(entropy, dtype=torch.bool)
        else:
            confidence = probs.max(dim=1).values
            selected = confidence >= confidence_threshold
            min_selected = max(1, int(min_selected_fraction * images.size(0)))
            if selected.sum().item() < min_selected:
                selected = torch.zeros_like(selected)

        if selected.any():
            loss = entropy[selected].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_logits = adapted(images)
            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())
            selected_total += selected.sum().item()

    mean_entropy = torch.cat(entropies).mean().item()
    update_ratio = selected_total / total if total else 0.0
    return EvalResult(method, corruption, severity, correct / total, mean_entropy, update_ratio)


def evaluate_margin_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    margin_threshold: float = 0.40,
    min_selected_fraction: float = 0.10,
    method_name: str | None = None,
) -> EvalResult:
    """Tent with top-1/top-2 probability margin gating.

    Max-softmax confidence can be poorly calibrated. The margin between the
    highest and second-highest probabilities is a stricter signal: it asks
    whether the model clearly prefers one class over the nearest competitor.
    """
    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)
    method = method_name or f"margin_tent_{margin_threshold:.2f}"

    correct = 0
    total = 0
    entropies = []
    selected_total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = adapted(images)
        probs = F.softmax(logits, dim=1)
        entropy = entropy_from_logits(logits)
        top2 = probs.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        selected = margin >= margin_threshold
        min_selected = max(1, int(min_selected_fraction * images.size(0)))
        if selected.sum().item() < min_selected:
            selected = torch.zeros_like(selected)

        if selected.any():
            loss = entropy[selected].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_logits = adapted(images)
            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())
            selected_total += selected.sum().item()

    mean_entropy = torch.cat(entropies).mean().item()
    update_ratio = selected_total / total if total else 0.0
    return EvalResult(method, corruption, severity, correct / total, mean_entropy, update_ratio)


def evaluate_anchor_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    anchor_weight: float = 0.20,
    temperature: float = 2.0,
    confidence_threshold: float | None = None,
    min_selected_fraction: float = 0.10,
    method_name: str | None = None,
) -> EvalResult:
    """Tent with source-prediction anchoring.

    The adapted model minimizes target entropy, but a KL term discourages it
    from moving too far from the frozen source model's prediction distribution.
    This is a lightweight source-anchor idea, not a SANTA reproduction.
    """
    source = clone_model(model, device)
    source.eval()

    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)
    method = method_name or f"anchor_tent_{anchor_weight:.2f}"

    correct = 0
    total = 0
    entropies = []
    selected_total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            source_logits = source(images)
            source_probs = F.softmax(source_logits / temperature, dim=1)

        logits = adapted(images)
        probs = F.softmax(logits, dim=1)
        entropy = entropy_from_logits(logits)

        if confidence_threshold is None:
            selected = torch.ones_like(entropy, dtype=torch.bool)
        else:
            confidence = probs.max(dim=1).values
            selected = confidence >= confidence_threshold
            min_selected = max(1, int(min_selected_fraction * images.size(0)))
            if selected.sum().item() < min_selected:
                selected = torch.zeros_like(selected)

        if selected.any():
            anchor_loss = F.kl_div(
                F.log_softmax(logits[selected] / temperature, dim=1),
                source_probs[selected],
                reduction="batchmean",
            ) * (temperature**2)
            loss = entropy[selected].mean() + anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_logits = adapted(images)
            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())
            selected_total += selected.sum().item()

    mean_entropy = torch.cat(entropies).mean().item()
    update_ratio = selected_total / total if total else 0.0
    return EvalResult(method, corruption, severity, correct / total, mean_entropy, update_ratio)


def evaluate_source_mix_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    adapted_weight: float = 0.50,
    method_name: str | None = None,
) -> EvalResult:
    """Tent adaptation with source/adapted logit ensembling at prediction time.

    The model adapts like Tent, but predictions are made from a convex mixture
    of frozen source logits and adapted logits. This is meant to test whether
    source-model ensembling reduces negative transfer on mild shifts.
    """
    source = clone_model(model, device)
    source.eval()

    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)
    method = method_name or f"source_mix_tent_{adapted_weight:.2f}"

    correct = 0
    total = 0
    entropies = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = adapted(images)
        loss = entropy_from_logits(logits).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            adapted_logits = adapted(images)
            source_logits = source(images)
            final_logits = adapted_weight * adapted_logits + (1.0 - adapted_weight) * source_logits
            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())

    mean_entropy = torch.cat(entropies).mean().item()
    return EvalResult(method, corruption, severity, correct / total, mean_entropy, 1.0)


def evaluate_entropy_gap_mix_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    alpha: float = 12.0,
    margin: float = 0.02,
    min_adapt_weight: float = 0.10,
    max_adapt_weight: float = 0.95,
    mode: str = "batch",
    method_name: str | None = None,
) -> EvalResult:
    """Tent with entropy-gap adaptive source/adapted logit mixing.

    Fixed source/adapted mixing protects mild shifts but under-adapts severe
    shifts. This variant chooses the adapted-logit weight from the entropy gap:

        gap = normalized_entropy(source_logits) - normalized_entropy(adapted_logits)

    If the adapted model is much lower-entropy than the source model, the
    prediction trusts adapted logits more. If the frozen source model remains
    lower-entropy, the prediction falls back toward source logits. This changes
    prediction only; adaptation itself is still standard Tent.
    """
    if mode not in {"batch", "sample"}:
        raise ValueError("mode must be 'batch' or 'sample'")

    source = clone_model(model, device)
    source.eval()

    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)
    method = method_name or f"eg_mix_{mode}"

    correct = 0
    total = 0
    entropies = []
    weights = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = adapted(images)
        loss = entropy_from_logits(logits).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            adapted_logits = adapted(images)
            source_logits = source(images)
            source_entropy = normalized_entropy_from_logits(source_logits)
            adapted_entropy = normalized_entropy_from_logits(adapted_logits)
            gap = source_entropy - adapted_entropy

            if mode == "batch":
                gate = torch.sigmoid(alpha * (gap.mean() - margin))
                adapted_weight = min_adapt_weight + (max_adapt_weight - min_adapt_weight) * gate
                final_logits = adapted_weight * adapted_logits + (1.0 - adapted_weight) * source_logits
                weights.append(adapted_weight.detach().reshape(1).cpu())
            else:
                gate = torch.sigmoid(alpha * (gap - margin)).unsqueeze(1)
                adapted_weight = min_adapt_weight + (max_adapt_weight - min_adapt_weight) * gate
                final_logits = adapted_weight * adapted_logits + (1.0 - adapted_weight) * source_logits
                weights.append(adapted_weight.detach().flatten().cpu())

            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())

    mean_entropy = torch.cat(entropies).mean().item()
    mean_weight = torch.cat(weights).mean().item() if weights else 0.0
    return EvalResult(method, corruption, severity, correct / total, mean_entropy, mean_weight)


def evaluate_disagreement_aware_mix_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    agree_adapt_weight: float = 0.85,
    disagree_alpha: float = 14.0,
    disagree_margin: float = 0.02,
    min_adapt_weight: float = 0.05,
    max_adapt_weight: float = 0.95,
    method_name: str = "disagree_aware_mix",
) -> EvalResult:
    """Tent with disagreement-aware adaptive source/adapted logit mixing.

    When source and adapted models agree, prediction mostly trusts adapted
    logits. When they disagree, the adapted weight is decided by entropy gap.
    This targets exactly the negative-transfer setting where adaptation changes
    a source-correct prediction under mild shift.
    """
    source = clone_model(model, device)
    source.eval()

    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)

    correct = 0
    total = 0
    entropies = []
    weights = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = adapted(images)
        loss = entropy_from_logits(logits).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            adapted_logits = adapted(images)
            source_logits = source(images)
            source_pred = source_logits.argmax(dim=1)
            adapted_pred = adapted_logits.argmax(dim=1)
            agree = source_pred == adapted_pred

            source_entropy = normalized_entropy_from_logits(source_logits)
            adapted_entropy = normalized_entropy_from_logits(adapted_logits)
            gap = source_entropy - adapted_entropy
            disagree_gate = torch.sigmoid(disagree_alpha * (gap - disagree_margin))
            disagree_weight = min_adapt_weight + (
                max_adapt_weight - min_adapt_weight
            ) * disagree_gate

            adapted_weight = torch.where(
                agree,
                torch.full_like(disagree_weight, agree_adapt_weight),
                disagree_weight,
            ).unsqueeze(1)
            final_logits = adapted_weight * adapted_logits + (1.0 - adapted_weight) * source_logits

            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())
            weights.append(adapted_weight.detach().flatten().cpu())

    mean_entropy = torch.cat(entropies).mean().item()
    mean_weight = torch.cat(weights).mean().item() if weights else 0.0
    return EvalResult(method_name, corruption, severity, correct / total, mean_entropy, mean_weight)


def _snapshot_params(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _restore_params(params: list[nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def evaluate_probe_commit_tent(
    model: nn.Module,
    loader,
    device: torch.device,
    corruption: str,
    severity: int,
    lr: float = 2.5e-4,
    entropy_margin: float = 0.01,
    high_conf_threshold: float = 0.75,
    max_high_conf_flip_rate: float = 0.08,
    collapse_threshold: float = 0.65,
    method_name: str = "probe_commit_tent",
) -> EvalResult:
    """Tent with reversible probe-then-commit updates.

    For each target batch, this method:

    1. snapshots the current adapted BN affine parameters,
    2. makes a tentative Tent update,
    3. compares frozen-source vs post-update behavior using unlabeled signals,
    4. commits the update only if it appears useful and not risky,
    5. otherwise restores the pre-update parameters and predicts with source logits.

    Commit score is intentionally conservative. It commits when the tentative
    update reduces normalized entropy relative to the frozen source model, while
    avoiding high-confidence source prediction flips and one-class collapse.
    This is not a claim of novelty; it is an explicit negative-transfer guard.
    """
    source = clone_model(model, device)
    source.eval()

    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)

    correct = 0
    total = 0
    entropies = []
    committed_batches = 0
    total_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        total_batches += 1

        with torch.no_grad():
            source_logits = source(images)
            source_probs = F.softmax(source_logits, dim=1)
            source_conf, source_pred = source_probs.max(dim=1)
            source_entropy = normalized_entropy_from_logits(source_logits)

        param_snapshot = _snapshot_params(params)
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

        probe_logits = adapted(images)
        probe_loss = entropy_from_logits(probe_logits).mean()
        optimizer.zero_grad(set_to_none=True)
        probe_loss.backward()
        optimizer.step()

        with torch.no_grad():
            post_logits = adapted(images)
            post_entropy = normalized_entropy_from_logits(post_logits)
            post_pred = post_logits.argmax(dim=1)

            entropy_gain = source_entropy.mean() - post_entropy.mean()
            high_conf_flip = (
                (source_conf >= high_conf_threshold) & (source_pred != post_pred)
            ).float().mean()
            pred_counts = torch.bincount(post_pred, minlength=post_logits.shape[1]).float()
            collapse_rate = pred_counts.max() / pred_counts.sum().clamp_min(1.0)

            should_commit = (
                entropy_gain.item() >= entropy_margin
                and high_conf_flip.item() <= max_high_conf_flip_rate
                and collapse_rate.item() <= collapse_threshold
            )

            if should_commit:
                final_logits = post_logits
                committed_batches += 1
            else:
                _restore_params(params, param_snapshot)
                optimizer.load_state_dict(optimizer_snapshot)
                final_logits = source_logits

            correct += (final_logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            entropies.append(entropy_from_logits(final_logits).detach().cpu())

    mean_entropy = torch.cat(entropies).mean().item()
    commit_ratio = committed_batches / total_batches if total_batches else 0.0
    return EvalResult(method_name, corruption, severity, correct / total, mean_entropy, commit_ratio)
