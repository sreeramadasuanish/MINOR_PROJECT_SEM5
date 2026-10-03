"""
Non-Parametric Bootstrap Resampling for Local Client Evaluations in Federated Learning.
Computes local client metric distributions and privacy-preserving summary statistics
(mean, variance, standard error, and percentile confidence intervals).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def run_client_bootstrap_evaluation(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    primary_metric: str = "accuracy",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes non-parametric bootstrap resampling on a client's local evaluation set.

    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        y_prob: Predicted class probabilities (optional)
        n_bootstrap: Number of bootstrap iterations B (e.g. 500)
        alpha: Significance level (default 0.05 for 95% CI)
        primary_metric: Target metric to resample ('accuracy', 'f1_macro', 'precision_macro', 'recall_macro')
        seed: Random seed for reproducibility

    Returns:
        Dictionary of privacy-preserving summary statistics and bootstrap sample distribution.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_samples = len(y_true)

    if n_samples < 5:
        raise ValueError(f"Sample size too small for non-parametric bootstrap (found {n_samples}).")

    correct_mask = (y_true == y_pred)
    correct_count = int(np.sum(correct_mask))
    incorrect_count = n_samples - correct_count

    bootstrap_scores = []
    
    # Run B bootstrap iterations
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        b_true = y_true[boot_idx]
        b_pred = y_pred[boot_idx]

        if primary_metric == "accuracy":
            score = accuracy_score(b_true, b_pred)
        elif primary_metric == "f1_macro":
            score = f1_score(b_true, b_pred, average="macro", zero_division=0)
        elif primary_metric == "precision_macro":
            score = precision_score(b_true, b_pred, average="macro", zero_division=0)
        elif primary_metric == "recall_macro":
            score = recall_score(b_true, b_pred, average="macro", zero_division=0)
        else:
            score = accuracy_score(b_true, b_pred)

        bootstrap_scores.append(float(score))

    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = float(np.mean(bootstrap_scores))
    var_score = float(np.var(bootstrap_scores, ddof=1)) if len(bootstrap_scores) > 1 else 1e-6
    se_score = float(np.sqrt(max(var_score, 1e-10)))

    ci_lower = float(np.percentile(bootstrap_scores, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1.0 - alpha / 2.0)))

    return {
        "metric_name": primary_metric,
        "n_samples": int(n_samples),
        "n_bootstrap": int(n_bootstrap),
        "mean": mean_score,
        "variance": max(var_score, 1e-8),
        "std_error": se_score,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "alpha": alpha,
        "bootstrap_samples": bootstrap_scores.tolist(),
    }
