"""
Evaluation metrics for homodimerization prediction.
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score
)
from scipy import stats
from typing import Dict, Tuple, List, Optional
import warnings


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        AUPRC score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Matthews Correlation Coefficient.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
    
    Returns:
        MCC score
    """
    return matthews_corrcoef(y_true, y_pred)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'mcc'
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        metric: Metric to optimize ('mcc', 'f1', 'balanced_acc')
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'balanced_acc':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def compute_f1_optimal(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute F1 score at optimal threshold (determined by MCC).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
    
    Returns:
        Tuple of (f1_score, optimal_threshold)
    """
    optimal_threshold, _ = find_optimal_threshold(y_true, y_scores, metric='mcc')
    y_pred = (y_scores >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    return f1, optimal_threshold


def compute_precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute precision at top k% predictions.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        k_values: List of k values (percentages)
    
    Returns:
        Dictionary with precision@k values
    """
    n_samples = len(y_true)
    precision_at_k = {}
    
    # Sort predictions by score
    sorted_indices = np.argsort(y_scores)[::-1]
    
    for k in k_values:
        n_top = max(1, int(n_samples * k / 100))
        top_indices = sorted_indices[:n_top]
        
        precision = np.mean(y_true[top_indices])
        precision_at_k[f'precision@{k}'] = precision
    
    return precision_at_k


def compute_calibration_error(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        Tuple of (ECE, calibration_data)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    calibration_data = {
        'bin_centers': [],
        'bin_accuracies': [],
        'bin_confidences': [],
        'bin_counts': []
    }
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_scores[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_data['bin_centers'].append((bin_lower + bin_upper) / 2)
            calibration_data['bin_accuracies'].append(accuracy_in_bin)
            calibration_data['bin_confidences'].append(avg_confidence_in_bin)
            calibration_data['bin_counts'].append(in_bin.sum())
    
    return ece, calibration_data


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        threshold: Classification threshold (auto-determined if None)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # AUPRC (primary metric)
    metrics['auprc'] = compute_auprc(y_true, y_scores)
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, best_mcc = find_optimal_threshold(y_true, y_scores, metric='mcc')
        metrics['optimal_threshold'] = threshold
        metrics['mcc_at_optimal'] = best_mcc
    else:
        y_pred = (y_scores >= threshold).astype(int)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Binary predictions at threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Basic metrics
    metrics['accuracy'] = np.mean(y_true == y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # F1 at optimal threshold
    f1_opt, _ = compute_f1_optimal(y_true, y_scores)
    metrics['f1_optimal'] = f1_opt
    
    # Precision at k
    precision_at_k = compute_precision_at_k(y_true, y_scores)
    metrics.update(precision_at_k)
    
    # Calibration error
    ece, _ = compute_calibration_error(y_true, y_scores)
    metrics['calibration_error'] = ece
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Class-specific metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    
    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        metric_func: Function to compute metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for interval
        random_state: Random seed
    
    Returns:
        Tuple of (metric_value, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Compute metric on original data
    metric_value = metric_func(y_true, y_scores)
    
    # Bootstrap
    bootstrap_values = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                value = metric_func(y_true_boot, y_scores_boot)
                bootstrap_values.append(value)
            except:
                continue
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_values, lower_percentile)
    upper_bound = np.percentile(bootstrap_values, upper_percentile)
    
    return metric_value, lower_bound, upper_bound


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self, metrics_to_track: List[str]):
        """
        Initialize metric tracker.
        
        Args:
            metrics_to_track: List of metric names to track
        """
        self.metrics_to_track = metrics_to_track
        self.history = {metric: [] for metric in metrics_to_track}
        self.best_values = {metric: -np.inf for metric in metrics_to_track}
        self.best_epochs = {metric: 0 for metric in metrics_to_track}
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """Update tracked metrics."""
        for metric in self.metrics_to_track:
            if metric in metrics:
                value = metrics[metric]
                self.history[metric].append(value)
                
                if value > self.best_values[metric]:
                    self.best_values[metric] = value
                    self.best_epochs[metric] = epoch
    
    def get_best(self, metric: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        return self.best_values[metric], self.best_epochs[metric]
    
    def is_best(self, metric: str) -> bool:
        """Check if current value is best."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return False
        return self.history[metric][-1] == self.best_values[metric]


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Generate imbalanced dummy data
    n_samples = 1000
    n_positive = int(n_samples * 0.1)  # 10% positive
    
    y_true = np.zeros(n_samples)
    y_true[:n_positive] = 1
    
    # Generate scores with some discrimination
    y_scores = np.random.beta(2, 5, n_samples)
    y_scores[:n_positive] += np.random.normal(0.3, 0.1, n_positive)
    y_scores = np.clip(y_scores, 0, 1)
    
    # Compute all metrics
    metrics = compute_all_metrics(y_true, y_scores)
    
    print("Evaluation Metrics:")
    print("-" * 40)
    for metric, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{metric:25s}: {value:.4f}")
        else:
            print(f"{metric:25s}: {value}")
    
    # Test bootstrap confidence intervals
    print("\nBootstrap Confidence Intervals (95%):")
    print("-" * 40)
    
    auprc_val, auprc_lower, auprc_upper = bootstrap_confidence_intervals(
        y_true, y_scores, compute_auprc, n_bootstrap=100
    )
    print(f"AUPRC: {auprc_val:.4f} [{auprc_lower:.4f}, {auprc_upper:.4f}]")