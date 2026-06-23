"""Task provider adapters used by XQT orchestration."""

from .detection import DetectionPrediction, decode_detection_output
from .evaluation import (
    EvaluationJob,
    EvaluationReport,
    build_evaluation_provider,
    resolve_evaluation_provider,
    run_evaluation_job,
)
from .training import (
    TrainingJob,
    TrainingReport,
    build_training_provider,
    resolve_training_provider,
    run_training_job,
)
from .xdl import XDLTrainingProvider

__all__ = [
    "EvaluationJob",
    "EvaluationReport",
    "DetectionPrediction",
    "TrainingJob",
    "TrainingReport",
    "XDLTrainingProvider",
    "build_evaluation_provider",
    "build_training_provider",
    "decode_detection_output",
    "resolve_evaluation_provider",
    "resolve_training_provider",
    "run_evaluation_job",
    "run_training_job",
]
