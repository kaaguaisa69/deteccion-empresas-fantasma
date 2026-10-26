"""Orquestación de experimentos y registro de corridas."""

from efd.experiments.config import ExperimentConfig
from efd.experiments.run_record import RunRecorder
from efd.experiments.runner import ExperimentRunner

__all__ = ["ExperimentConfig", "ExperimentRunner", "RunRecorder"]
