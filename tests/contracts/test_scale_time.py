"""T5: ScaleTime enum parsing."""

from __future__ import annotations

import pytest

from xqt.contracts.scale_time import ScaleTime, scale_time_payload


def test_scale_time_values() -> None:
    assert ScaleTime.WEIGHT_OFFLINE.value == "weight_offline"
    assert ScaleTime.ACTIVATION_STATIC.value == "activation_static"
    assert ScaleTime.ACTIVATION_DYNAMIC.value == "activation_dynamic"
    assert ScaleTime.KV_SCALE.value == "kv_scale"
    assert ScaleTime.WEIGHT_LOAD_TIME.value == "weight_load_time"


def test_scale_time_parse_and_payload() -> None:
    assert ScaleTime.parse("activation_static") is ScaleTime.ACTIVATION_STATIC
    payload = scale_time_payload(
        ScaleTime.ACTIVATION_DYNAMIC, activation_granularity="per_token"
    )
    assert payload["scale_time"] == "activation_dynamic"
    assert payload["activation_granularity"] == "per_token"
    with pytest.raises(ValueError, match="unknown scale_time"):
        ScaleTime.parse("not_a_time")
