from __future__ import annotations

import copy
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
) -> EvalResult:
    adapted = clone_model(model, device)
    params = configure_tent(adapted)
    optimizer = torch.optim.Adam(params, lr=lr)
    method = "ug_tent" if confidence_threshold is not None else "tent"

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

