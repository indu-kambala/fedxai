"""Federated server. Aggregates parameters only — never hospital pixels."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from ml.datasets import create_test_dataloader
from ml.federated.client import HospitalClient
from ml.federated.fedavg import fedavg_aggregate
from ml.federated.fedprox import copy_global_params
from ml.models.factory import create_model
from ml.training.trainer import collect_predictions, evaluate, save_checkpoint
from ml.evaluation.metrics import overall_metrics, roc_auc_ovr
from ml.training.repro import set_seed


DEFAULT_CLIENTS = [
    ("hospital_1", Path("data") / "hospital_1"),
    ("hospital_2", Path("data") / "hospital_2"),
    ("hospital_3", Path("data") / "hospital_3"),
    ("hospital_4", Path("data") / "hospital_4"),
    ("hospital_5", Path("data") / "hospital_5"),
]


def run_federated(
    *,
    algorithm: str = "fedavg",
    model_name: str = "densenet121",
    rounds: int = 2,
    local_epochs: int = 1,
    max_batches: int = 4,
    mu: float = 0.01,
    seed: int = 42,
    device: str | None = None,
    test_path: Path | None = None,
    dp_noise: float = 0.0,
    clip_norm: float | None = None,
) -> dict:
    set_seed(seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    global_model = create_model(model_name, frozen_backbone=True).to(torch_device)
    clients = [HospitalClient(hid, root) for hid, root in DEFAULT_CLIENTS]
    history = []
    started = time.perf_counter()

    for round_idx in range(rounds):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        global_params = copy_global_params(global_model) if algorithm == "fedprox" else None
        updates = []
        sizes = []
        client_rows = []
        for client in clients:
            local_model = create_model(model_name, frozen_backbone=True).to(torch_device)
            local_model.load_state_dict(global_model.state_dict())
            result = client.local_train(
                local_model,
                torch_device,
                local_epochs=local_epochs,
                max_batches=max_batches,
                proximal_mu=mu if algorithm == "fedprox" else 0.0,
                global_params=global_params,
            )
            client_rows.append(
                {
                    "hospital_id": result["hospital_id"],
                    "status": result["status"],
                    "train_loss": result.get("train_loss"),
                    "train_accuracy": result.get("train_accuracy"),
                    "n_samples": result.get("n_samples", 0),
                    "raw_images_sent": False,
                }
            )
            if result["status"] == "trained" and result.get("n_samples", 0) > 0:
                state = result["state_dict"]
                if clip_norm:
                    from ml.federated.privacy import clip_state_dict

                    state = clip_state_dict(state, global_state, clip_norm)
                if dp_noise:
                    from ml.federated.privacy import add_gaussian_noise

                    state = add_gaussian_noise(state, dp_noise)
                updates.append(state)
                sizes.append(int(result["n_samples"]))
        if updates:
            aggregated = fedavg_aggregate(updates, sizes)
            global_model.load_state_dict(aggregated)
        history.append(
            {
                "round": round_idx + 1,
                "algorithm": algorithm,
                "clients": client_rows,
                "participating_clients": len(updates),
                "raw_images_on_wire": False,
            }
        )

    test_metrics = {"status": "Not evaluated"}
    test_root = Path(test_path or (Path("data") / "global_test"))
    if test_root.exists() and any(test_root.rglob("*.png")) or (test_root.exists() and any(test_root.rglob("*.jpg"))):
        loader, _ = create_test_dataloader(test_root)
        loss, acc = evaluate(global_model, loader, torch_device, max_batches=8)
        y_true, y_pred, probs = collect_predictions(global_model, loader, torch_device, max_batches=8)
        test_metrics = {
            **overall_metrics(y_true, y_pred),
            "loss": loss,
            "accuracy_from_eval": acc,
            **roc_auc_ovr(y_true, probs),
        }

    elapsed = time.perf_counter() - started
    payload = {
        "algorithm": algorithm,
        "model_name": model_name,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "max_batches": max_batches,
        "fedprox_mu": mu if algorithm == "fedprox" else None,
        "seed": seed,
        "device": str(torch_device),
        "history": history,
        "global_test": test_metrics,
        "training_time_seconds": elapsed,
        "raw_images_on_wire": False,
        "disclaimer": "AI-assisted research prototype. Federated model output is not a diagnosis.",
        "state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
    }
    out_dir = Path("artifacts") / "experiments" / f"{algorithm}_{model_name}"
    save_checkpoint(out_dir / "global.pt", payload)
    metrics = {k: v for k, v in payload.items() if k != "state_dict"}
    (out_dir).mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics
