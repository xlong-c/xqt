"""XQT data -- synthetic samples, calibration utilities, and dataset loaders."""

from __future__ import annotations

from .builders import build_data_split
from .calibration import (
    calibration_sample_count,
    calibration_sweep,
    extract_calibration_samples,
)
from .detection import (
    Coco8DetectionSpec,
    build_coco8_detection_loader,
    SyntheticDetectionSpec,
    build_image_boxes_transform,
    build_synthetic_detection_loader,
    build_xdl_detection_loader,
    load_image_tensor,
)
from .input_utils import (
    BatchSplit,
    DETECTION_TARGET_KEYS,
    extract_model_inputs,
    infer_model_input_count,
    split_batch,
)
from .hf_text import build_hf_text_classification_loader
from .prompts import (
    PromptBatch,
    build_prompt_list,
    build_prompt_list_from_file,
    prompt_summary,
)
from .samples import (
    SyntheticClassificationSpec,
    build_example_input,
    build_synthetic_classification_loader,
)
from .torchvision import (
    TorchvisionImageClassificationSpec,
    build_torchvision_image_classification_loader,
)

__all__ = [
    "BatchSplit",
    "Coco8DetectionSpec",
    "DETECTION_TARGET_KEYS",
    "PromptBatch",
    "SyntheticClassificationSpec",
    "SyntheticDetectionSpec",
    "TorchvisionImageClassificationSpec",
    "build_coco8_detection_loader",
    "build_data_split",
    "build_example_input",
    "build_hf_text_classification_loader",
    "build_image_boxes_transform",
    "build_prompt_list",
    "build_prompt_list_from_file",
    "build_synthetic_classification_loader",
    "build_synthetic_detection_loader",
    "build_torchvision_image_classification_loader",
    "build_xdl_detection_loader",
    "calibration_sample_count",
    "calibration_sweep",
    "extract_model_inputs",
    "extract_calibration_samples",
    "infer_model_input_count",
    "load_image_tensor",
    "prompt_summary",
    "split_batch",
]
