from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import download_cifar10c, make_cifar10_loaders, make_cifar10c_loader
from src.models import count_parameters, make_cifar_resnet18
from src.train import train_source_model
from src.tta import evaluate_bn_adapt, evaluate_source, evaluate_tent


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def result_to_dict(result) -> dict:
    return {
        "method": result.method,
        "corruption": result.corruption,
        "severity": result.severity,
        "accuracy": result.accuracy,
        "mean_entropy": result.mean_entropy,
        "update_ratio": result.update_ratio,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results", default="results/first_pass_results.csv")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--first-pass-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    torch.manual_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    print(f"device={device}")

    data_root = Path(args.data_root or cfg["data"]["root"])
    data_root.mkdir(parents=True, exist_ok=True)

    train_loader, clean_test_loader = make_cifar10_loaders(
        data_root,
        batch_size=int(cfg["train"]["batch_size"]),
    )
    download_cifar10c(data_root)

    model = make_cifar_resnet18(num_classes=int(cfg["model"]["num_classes"]))
    print(f"trainable_parameters={count_parameters(model):,}")

    epochs = args.epochs if args.epochs is not None else int(cfg["train"]["epochs"])
    model = train_source_model(
        model,
        train_loader,
        clean_test_loader,
        device=device,
        epochs=epochs,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        checkpoint_path=ROOT / cfg["model"]["checkpoint"],
    )
    model.to(device)

    corruptions = cfg["data"]["first_pass_corruptions"]
    severities = cfg["data"]["severities"]
    rows: list[dict] = []
    for corruption in corruptions:
        for severity in severities:
            print(f"\ncorruption={corruption} severity={severity}")
            loader = make_cifar10c_loader(
                data_root,
                corruption=corruption,
                severity=int(severity),
                batch_size=int(cfg["tta"]["batch_size"]),
            )
            results = [
                evaluate_source(model, loader, device, "source_only", corruption, int(severity)),
                evaluate_bn_adapt(model, loader, device, corruption, int(severity)),
                evaluate_tent(
                    model,
                    loader,
                    device,
                    corruption,
                    int(severity),
                    lr=float(cfg["tta"]["lr"]),
                    confidence_threshold=None,
                ),
                evaluate_tent(
                    model,
                    loader,
                    device,
                    corruption,
                    int(severity),
                    lr=float(cfg["tta"]["lr"]),
                    confidence_threshold=float(cfg["tta"]["ug_tent"]["confidence_threshold"]),
                    min_selected_fraction=float(cfg["tta"]["ug_tent"]["min_selected_fraction"]),
                ),
            ]
            for result in results:
                row = result_to_dict(result)
                rows.append(row)
                print(json.dumps(row, indent=2))
            write_csv(ROOT / args.results, rows)

    print(f"\nwrote {ROOT / args.results}")


if __name__ == "__main__":
    main()

