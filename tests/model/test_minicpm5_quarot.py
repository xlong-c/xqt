from __future__ import annotations

import pytest
import torch

from xqt.model.minicpm5_quarot import (
    apply_quarot_minicpm5,
    build_quarot_rotation,
)


def test_quarot_randomized_hadamard_is_orthogonal() -> None:
    rotation = build_quarot_rotation(128, seed=7)

    assert torch.allclose(
        rotation @ rotation.T,
        torch.eye(128, dtype=rotation.dtype),
        atol=1e-12,
        rtol=1e-12,
    )


def test_quarot_online_sites_are_explicitly_rejected() -> None:
    with pytest.raises(NotImplementedError, match="online QuaRot"):
        apply_quarot_minicpm5(object(), rotate_ov_projections=True)  # type: ignore[arg-type]
