"""Particiones, métricas y política de umbral."""

from efd.evaluation.metrics import METRICS, compute_metrics, precision_at_k, register_metric
from efd.evaluation.splits import Fold, stratified_folds
from efd.evaluation.threshold import (
    THRESHOLD_POLICIES,
    ThresholdPolicy,
    binarize,
    create_threshold_policy,
)

__all__ = [
    "METRICS",
    "THRESHOLD_POLICIES",
    "Fold",
    "ThresholdPolicy",
    "binarize",
    "compute_metrics",
    "create_threshold_policy",
    "precision_at_k",
    "register_metric",
    "stratified_folds",
]
