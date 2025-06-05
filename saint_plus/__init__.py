"""
SAINT+ Knowledge Tracing

A PyTorch implementation of SAINT+: Integrating Temporal Features 
for EdNet Correctness Prediction
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .model import SAINTPlus
from .trainer import SAINTPlusTrainer
from .data.dataset import KnowledgeTracingDataset
from .utils.metrics import compute_auc, compute_accuracy

__all__ = [
    "SAINTPlus",
    "SAINTPlusTrainer", 
    "KnowledgeTracingDataset",
    "compute_auc",
    "compute_accuracy"
] 