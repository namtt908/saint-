import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Union, Optional

def compute_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        AUC score
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return roc_auc_score(y_true, y_pred)

def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Accuracy score
    """
    y_true = y_true.cpu().numpy()
    y_pred = (y_pred.cpu().numpy() > threshold).astype(int)
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute precision score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Precision score
    """
    y_true = y_true.cpu().numpy()
    y_pred = (y_pred.cpu().numpy() > threshold).astype(int)
    return precision_score(y_true, y_pred)

def compute_recall(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute recall score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Recall score
    """
    y_true = y_true.cpu().numpy()
    y_pred = (y_pred.cpu().numpy() > threshold).astype(int)
    return recall_score(y_true, y_pred)

def compute_f1(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        F1 score
    """
    y_true = y_true.cpu().numpy()
    y_pred = (y_pred.cpu().numpy() > threshold).astype(int)
    return f1_score(y_true, y_pred)

def compute_metrics(y_true: torch.Tensor, 
                   y_pred: torch.Tensor, 
                   threshold: float = 0.5,
                   metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        metrics: List of metrics to compute. If None, compute all metrics.
        
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    
    metric_fns = {
        'auc': compute_auc,
        'accuracy': compute_accuracy,
        'precision': compute_precision,
        'recall': compute_recall,
        'f1': compute_f1
    }
    
    results = {}
    for metric in metrics:
        if metric in metric_fns:
            results[metric] = metric_fns[metric](y_true, y_pred, threshold)
    
    return results 