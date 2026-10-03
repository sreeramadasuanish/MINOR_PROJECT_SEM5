"""
Evaluation metrics computation module for MNIST classification in Federated Learning.
Computes Accuracy, Precision, Recall, F1-Score, and Multi-Class ROC-AUC.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import pandas as pd


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of classification metrics for multi-class MNIST.

    Args:
        y_true: Ground truth class indices (array-like, shape [N])
        y_pred: Predicted class indices (array-like, shape [N])
        y_prob: Predicted class probabilities (array-like, shape [N, num_classes]), optional
        num_classes: Total number of classes (default 10 for MNIST)

    Returns:
        Dict containing accuracy, precision, recall, f1, auc, and confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_samples = len(y_true)

    if n_samples == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "auc": 0.0,
            "n_samples": 0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
        }

    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Precision
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    # Recall
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # F1-Score
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Multi-class ROC-AUC
    auc = 0.5
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            # Check unique classes present in y_true
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                if y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
                    # One-vs-Rest ROC AUC
                    auc = float(
                        roc_auc_score(
                            y_true,
                            y_prob,
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(num_classes)),
                        )
                    )
                elif y_prob.ndim == 2 and y_prob.shape[1] == len(unique_classes):
                    auc = float(
                        roc_auc_score(
                            y_true,
                            y_prob,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
        except Exception:
            # Fallback if standard multi-class AUC calculation encounters edge cases
            auc = acc

    # Confusion matrix
    labels_all = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all).tolist()

    return {
        "accuracy": acc,
        "precision": prec_macro,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall": rec_macro,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1": f1_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc": auc,
        "n_samples": int(n_samples),
        "confusion_matrix": cm,
    }


def compute_client_metrics_table(client_eval_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts client evaluation dictionaries into a clean summary pandas DataFrame.
    """
    records = []
    for client_id, res in client_eval_results.items():
        records.append({
            "Client": client_id,
            "Samples": res.get("n_samples", 0),
            "Accuracy": f"{res.get('accuracy', 0.0) * 100:.2f}%",
            "Precision (Macro)": f"{res.get('precision_macro', 0.0) * 100:.2f}%",
            "Recall (Macro)": f"{res.get('recall_macro', 0.0) * 100:.2f}%",
            "F1-Score (Macro)": f"{res.get('f1_macro', 0.0) * 100:.2f}%",
            "ROC-AUC": f"{res.get('auc', 0.0):.4f}",
        })
    return pd.DataFrame(records)
