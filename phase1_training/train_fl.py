"""
Multi-round Federated Learning Orchestration Script for MNIST.
Coordinates client local updates, differential privacy clipping and noise,
and FedAvg aggregation with live telemetry tracking.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional, Callable
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.mnist_loader import load_mnist
from data.partitioned_clients import create_non_iid_partitions
from phase1_training.client import ClientNode
from phase1_training.server import FederatedServer


def run_federated_training(
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    dp_sigma: float = 0.05,
    clip_c: float = 1.0,
    use_oversampling: bool = True,
    skew_type: str = "structured",
    dirichlet_alpha: float = 0.5,
    save_path: Optional[str] = None,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Executes an end-to-end multi-round Federated Learning training session.

    Args:
        rounds: Number of federated communication rounds.
        local_epochs: Number of local training epochs per client per round.
        batch_size: Batch size for local SGD.
        lr: Learning rate.
        dp_sigma: Gaussian noise multiplier for Differential Privacy.
        clip_c: L2 gradient / update clipping threshold.
        use_oversampling: Whether to apply class-imbalance oversampling.
        skew_type: Non-IID partitioning strategy ('structured', 'dirichlet', 'extreme').
        dirichlet_alpha: Dirichlet concentration parameter (if skew_type is 'dirichlet').
        save_path: Path to save the final global model checkpoint.
        device: 'cpu' or 'cuda'.
        progress_callback: Optional callback invoked after each round with round metrics.

    Returns:
        Dictionary containing training history, final metrics, client status, and model path.
    """
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "phase1_training", "global_model.pt")

    # 1. Load MNIST dataset
    print("[FL Orchestrator] Loading MNIST dataset...")
    train_data, test_data = load_mnist()

    # 2. Partition across 3 non-IID clients
    print(f"[FL Orchestrator] Partitioning data across 3 clients (skew={skew_type})...")
    client_train, client_test, class_dist = create_non_iid_partitions(
        train_dataset=train_data,
        test_dataset=test_data,
        num_clients=3,
        skew_type=skew_type,
        dirichlet_alpha=dirichlet_alpha,
    )

    # 3. Instantiate Clients and Server
    clients = {
        cid: ClientNode(
            client_id=cid,
            train_dataset=client_train[cid],
            test_dataset=client_test[cid],
            batch_size=batch_size,
            lr=lr,
            device=device,
            use_oversampling=use_oversampling,
        )
        for cid in client_train.keys()
    }

    server = FederatedServer(test_dataset=test_data, device=device, use_smpc=True)

    history = {
        "rounds": [],
        "global_loss": [],
        "global_accuracy": [],
        "client_metrics": {cid: [] for cid in clients.keys()},
        "telemetry": [],
    }

    sample_counts = {cid: len(client_train[cid]) for cid in clients.keys()}

    print(f"\n{'='*70}")
    print(f" STARTING FEDERATED TRAINING (Rounds={rounds}, Local Epochs={local_epochs})")
    print(f" DP Noise Sigma={dp_sigma}, Clipping C={clip_c}, Oversampling={use_oversampling}")
    print(f"{'='*70}\n")

    for r in range(1, rounds + 1):
        print(f"\n--- Communication Round {r}/{rounds} ---")

        # Step A: Broadcast global weights to clients
        global_params = server.get_global_parameters()

        # Step B: Local Client Training with DP Noise and Gradient Clipping
        client_updates = {}
        round_telemetry = {}

        for cid, client in clients.items():
            updated_params, tele = client.train_epoch(
                global_parameters=global_params,
                local_epochs=local_epochs,
                clip_c=clip_c,
                dp_sigma=dp_sigma,
            )
            client_updates[cid] = updated_params
            round_telemetry[cid] = tele
            print(f"  [{cid}] Train Loss: {tele['train_loss']:.4f} | L2 Norm: {tele['post_clip_norm']:.3f} | DP Noise Std: {tele['noise_std']:.4f}")

        # Step C: Central FedAvg Secure Aggregation
        server.aggregate_fedavg(client_updates, sample_counts, round_num=r)

        # Step D: Evaluate Global Model on Global Test Set
        eval_global = server.evaluate_global()
        print(f"  >> [Server] Global Test Loss: {eval_global['loss']:.4f} | Accuracy: {eval_global['accuracy']*100:.2f}% | AUC: {eval_global['auc']:.4f}")

        # Step E: Evaluate on Local Client Test Partitions
        current_global_params = server.get_global_parameters()
        round_client_eval = {}
        for cid, client in clients.items():
            c_eval = client.evaluate_local(parameters=current_global_params)
            round_client_eval[cid] = c_eval
            history["client_metrics"][cid].append({
                "round": r,
                "loss": c_eval["loss"],
                "accuracy": c_eval["accuracy"],
                "f1_macro": c_eval["f1_macro"],
            })

        history["rounds"].append(r)
        history["global_loss"].append(eval_global["loss"])
        history["global_accuracy"].append(eval_global["accuracy"])
        history["telemetry"].append(round_telemetry)

        round_summary = {
            "round": r,
            "global_loss": eval_global["loss"],
            "global_accuracy": eval_global["accuracy"],
            "client_eval": round_client_eval,
            "telemetry": round_telemetry,
        }

        if progress_callback is not None:
            progress_callback(r, round_summary)

    # Save final model checkpoint
    server.save_checkpoint(save_path)

    final_results = {
        "history": history,
        "final_global_accuracy": history["global_accuracy"][-1],
        "final_global_loss": history["global_loss"][-1],
        "model_checkpoint_path": save_path,
        "class_distributions": class_dist,
        "sample_counts": sample_counts,
    }

    print(f"\n[FL Orchestrator] Federated training completed successfully. Model saved to '{save_path}'.\n")
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Multi-round Federated Learning on MNIST with DP and FedAvg")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL communication rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.05, help="DP Gaussian noise multiplier")
    parser.add_argument("--clip-c", type=float, default=1.0, help="Gradient clipping bound C")
    parser.add_argument("--no-oversample", action="store_true", help="Disable class imbalance oversampling")
    parser.add_argument("--skew", type=str, default="structured", choices=["structured", "dirichlet", "extreme"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Dirichlet skew alpha")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save global checkpoint")

    args = parser.parse_args()

    run_federated_training(
        rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dp_sigma=args.sigma,
        clip_c=args.clip_c,
        use_oversampling=not args.no_oversample,
        skew_type=args.skew,
        dirichlet_alpha=args.dirichlet_alpha,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
