"""
Main Fairness Evaluation Script for Federated MNIST.
Deploys the trained global model to clients, executes local non-parametric bootstrap resampling,
and runs the FED4FED statistical framework (Algorithms 1, 2, and 3) against significance level alpha = 0.05.
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.mnist_loader import load_mnist
from data.partitioned_clients import create_non_iid_partitions
from phase1_training.model import MNISTCNN
from phase1_training.client import ClientNode
from phase2_evaluation.bootstrap import run_client_bootstrap_evaluation
from phase2_evaluation.fed4fed_algorithms import (
    compute_algorithm1_chisquare,
    compute_algorithm2_welch_f,
    compute_algorithm3_confidence_intervals,
)


def run_fairness_evaluation(
    model_path: Optional[str] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    primary_metric: str = "accuracy",
    skew_type: str = "structured",
    dirichlet_alpha: float = 0.5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes an end-to-end FED4FED statistical fairness evaluation.

    Args:
        model_path: Path to the trained PyTorch global model checkpoint.
        alpha: Statistical significance level (default 0.05).
        n_bootstrap: Number of bootstrap resampling iterations.
        primary_metric: Metric to audit ('accuracy', 'f1_macro', etc.).
        skew_type: Skew partition type to evaluate on.
        dirichlet_alpha: Dirichlet concentration parameter.
        device: 'cpu' or 'cuda'.

    Returns:
        Structured fairness audit dictionary including test statistics, p-values, and verdicts.
    """
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "phase1_training", "global_model.pt")

    # 1. Load trained global model
    global_model = MNISTCNN().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        global_model.load_state_dict(state_dict)
        print(f"[Fairness Audit] Loaded model checkpoint from '{model_path}'.")
    else:
        print(f"[Fairness Audit] Model checkpoint not found at '{model_path}'. Using initialized weights for audit demo.")

    # 2. Prepare test data partitions
    train_data, test_data = load_mnist()
    _, client_test, class_dist = create_non_iid_partitions(
        train_dataset=train_data,
        test_dataset=test_data,
        num_clients=3,
        skew_type=skew_type,
        dirichlet_alpha=dirichlet_alpha,
    )

    # 3. Local Client Evaluations & Non-Parametric Bootstrap Resampling
    print(f"\n[Fairness Audit] Running local bootstrap evaluations across clients (B={n_bootstrap}, Metric={primary_metric})...")
    client_bootstrap_summaries = {}
    client_raw_metrics = {}

    for cid, test_subset in client_test.items():
        node = ClientNode(
            client_id=cid,
            train_dataset=test_subset,  # dummy for train
            test_dataset=test_subset,
            device=device,
        )
        node.model.load_state_dict(global_model.state_dict())
        eval_res = node.evaluate_local()
        client_raw_metrics[cid] = eval_res

        # Extract predictions for bootstrap
        all_preds = []
        all_targets = []
        all_probs = []
        node.model.eval()
        with torch.no_grad():
            for images, labels in node.test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = node.model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        # Run non-parametric bootstrap resampling locally
        boot_stats = run_client_bootstrap_evaluation(
            y_true=all_targets,
            y_pred=all_preds,
            y_prob=all_probs,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            primary_metric=primary_metric,
        )
        client_bootstrap_summaries[cid] = boot_stats
        print(f"  [{cid}] Mean {primary_metric.capitalize()}: {boot_stats['mean']*100:.2f}% | 95% CI: [{boot_stats['ci_lower']*100:.2f}%, {boot_stats['ci_upper']*100:.2f}%] | SE: {boot_stats['std_error']:.4f}")

    # 4. Execute FED4FED Statistical Algorithms
    print("\n[Fairness Audit] Executing FED4FED Hypothesis Testing Suite...")

    # Algorithm 1: Chi-Square Homogeneity Test
    algo1_results = compute_algorithm1_chisquare(client_bootstrap_summaries, alpha=alpha)

    # Algorithm 2: Welch's F-test (Heteroscedastic ANOVA)
    algo2_results = compute_algorithm2_welch_f(client_bootstrap_summaries, alpha=alpha)

    # Algorithm 3: Federated Confidence Intervals & Pairwise Disparity
    algo3_results = compute_algorithm3_confidence_intervals(client_bootstrap_summaries, alpha=alpha)

    # Overall Fairness Decision:
    # A system is deemed Fair if neither Chi-Square nor Welch's F rejects H0
    is_fair = not (algo1_results["reject_null_hypothesis"] or algo2_results["reject_null_hypothesis"])
    overall_verdict = "PASS: SYSTEM IS FAIR & HOMOGENEOUS" if is_fair else "FAIL: SIGNIFICANT INTER-CLIENT DISPARITY DETECTED"

    audit_report = {
        "overall_fairness_pass": is_fair,
        "overall_verdict": overall_verdict,
        "significance_level": alpha,
        "primary_metric": primary_metric,
        "client_raw_metrics": client_raw_metrics,
        "client_bootstrap_summaries": client_bootstrap_summaries,
        "algorithm_1_chisquare": algo1_results,
        "algorithm_2_welch_f": algo2_results,
        "algorithm_3_confidence_intervals": algo3_results,
    }

    # Print Terminal Summary Table
    _print_terminal_report(audit_report)
    return audit_report


def _print_terminal_report(report: Dict[str, Any]):
    alpha = report["significance_level"]
    a1 = report["algorithm_1_chisquare"]
    a2 = report["algorithm_2_welch_f"]
    a3 = report["algorithm_3_confidence_intervals"]

    print("\n" + "=" * 76)
    print("           FED4FED FAIRNESS & DISPARITY AUDIT REPORT (alpha = 0.05)")
    print("=" * 76)
    print(f"Overall Status: {report['overall_verdict']}")
    print("-" * 76)
    print(" [Algorithm 1] Chi-Square Homogeneity Test:")
    print(f"   * Chi2 Statistic:       {a1['chi2_statistic']:.4f} (df={a1['degrees_of_freedom']}, crit={a1['critical_value']:.4f})")
    print(f"   * p-value:              {a1['p_value']:.5e} (alpha={alpha})")
    print(f"   * Graybill-Deal Mean:   {a1['graybill_deal_mean'] * 100:.2f}% (SE={a1['graybill_deal_std_error']:.4f})")
    print(f"   * Verdict:              {a1['verdict']}")
    print("-" * 76)
    print(" [Algorithm 2] Welch's F-test (Heteroscedastic ANOVA):")
    print(f"   * F-statistic:          {a2['f_statistic']:.4f} (df1={a2['df1']}, df2={a2['df2']:.2f}, crit={a2['critical_f']:.4f})")
    print(f"   * p-value:              {a2['p_value']:.5e} (alpha={alpha})")
    print(f"   * Weighted Grand Mean:  {a2['weighted_grand_mean'] * 100:.2f}%")
    print(f"   * Verdict:              {a2['verdict']}")
    print("-" * 76)
    print(" [Algorithm 3] Pairwise Disparity Inference (Bonferroni-adjusted):")
    for pair in a3["pairwise_tests"]:
        diff_pct = pair["mean_difference"] * 100
        ci_str = f"[{pair['ci_lower']*100:.2f}%, {pair['ci_upper']*100:.2f}%]"
        print(f"   * {pair['comparison']:<20}: Diff = {diff_pct:+.2f}% | 95% CI = {ci_str:<18} | p = {pair['p_value']:.4e} | {pair['verdict']}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FED4FED Statistical Fairness Evaluation on MNIST")
    parser.add_argument("--model-path", type=str, default=None, help="Path to global_model.pt")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations B")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
    parser.add_argument("--skew", type=str, default="structured", choices=["structured", "dirichlet", "extreme"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)

    args = parser.parse_args()

    run_fairness_evaluation(
        model_path=args.model_path,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        primary_metric=args.metric,
        skew_type=args.skew,
        dirichlet_alpha=args.dirichlet_alpha,
    )


if __name__ == "__main__":
    main()
