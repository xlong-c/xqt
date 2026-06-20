"""测试共享夹具。"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def dummy_batch():
    """标准分类 batch: (B=8, C=10) logits + (B,) labels。"""

    class Batch:
        batch_size = 8
        num_classes = 10
        pred = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))

    return Batch()


@pytest.fixture
def dummy_image_batch():
    """图像分割 batch: (B=2, C=3, H=32, W=32) pred + (B, H, W) target。"""

    class ImageBatch:
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32))
        binary_pred = torch.randn(2, 1, 32, 32)
        binary_target = torch.randint(0, 2, (2, 32, 32)).float()

    return ImageBatch()


@pytest.fixture
def tiny_model():
    """最小可训练模型。"""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 3),
    )
