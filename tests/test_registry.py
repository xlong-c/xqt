import pytest

from xqt.core.errors import XQTRegistryError
from xqt.core.registry import XQTRegistry


def test_xqt_registry_registers_and_builds_targets() -> None:
    registry = XQTRegistry("TEST")

    @registry.register("callable_target")
    def make_value(value: int = 1) -> dict[str, int]:
        return {"value": value}

    @registry.register()
    class TargetClass:
        def __init__(self, label: str) -> None:
            self.label = label

    assert registry.get("callable_target") is make_value
    assert registry.build("callable_target", value=3) == {"value": 3}
    assert registry.build("TargetClass", label="xqt").label == "xqt"
    assert registry.list_available() == ["callable_target", "TargetClass"]
    assert "TargetClass" in registry


def test_xqt_registry_rejects_duplicate_names() -> None:
    registry = XQTRegistry("TEST")
    registry.register("same")(object())

    with pytest.raises(XQTRegistryError, match="already registered"):
        registry.register("same")(object())


def test_xqt_registry_reports_missing_name_with_suggestion() -> None:
    registry = XQTRegistry("TEST")
    registry.register("quant")(object())

    with pytest.raises(XQTRegistryError, match="Did you mean"):
        registry.get("qunat")


def test_xqt_registry_rejects_params_for_non_callable_target() -> None:
    registry = XQTRegistry("TEST")
    registry.register("value")(object())

    with pytest.raises(XQTRegistryError, match="does not accept params"):
        registry.build("value", x=1)
