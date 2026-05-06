from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_source_model(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: str | Path,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        return model

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
        scheduler.step()
        acc = evaluate_clean_accuracy(model, test_loader, device)
        loss_value = running_loss / len(train_loader.dataset)
        print(f"epoch={epoch:03d} train_loss={loss_value:.4f} clean_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "clean_acc": best_acc}, checkpoint_path)
    print(f"best_clean_acc={best_acc:.4f}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def evaluate_clean_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return correct / total

