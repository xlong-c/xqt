"""Tests for the engine registration table and unified dispatch entry (C3)."""

from __future__ import annotations

from xqt.runtime.engine_resolve import (
    EngineRegistration,
    engine_resolve_from_compute_config,
    engines_providing_all,
    get_engine_registration,
    iter_engine_registrations,
    resolve_compute_engine,
    resolve_int8_mma_engine,
)


_EXPECTED_PROVIDES: dict[str, frozenset[str]] = {
    "cuda_sm89": frozenset({"int8_mma", "true_int8_mma"}),
    "ptx_sm89": frozenset({"int8_mma", "true_int8_mma"}),
    "native_sm89": frozenset({"int8_mma", "true_int8_mma"}),
    "tilelang": frozenset(
        {
            "int8_mma",
            "true_int8_mma",
            "fp4_mma",
            "dequant_gemm_epilogue",
            "w4_storage_int8_mma",
            "hadamard_groupwise",
        }
    ),
    "torch_int_mm": frozenset({"int8_mma", "true_int8_mma"}),
    "torch": frozenset({"generic", "fp16_mma", "fp8_mma"}),
    "triton": frozenset(
        {
            "generic",
            "int8_mma",
            "true_int8_mma",
            "fp16_mma",
            "fp4_mma",
            "dequant_gemm_epilogue",
            "hadamard_groupwise",
        }
    ),
    "cutlass": frozenset({"int8_mma", "fp4_mma", "int4_mma", "fp16_mma"}),
    "cute_dsl": frozenset({"int8_mma", "fp4_mma"}),
    "cutile": frozenset({"generic", "fp16_mma"}),
    "reference": frozenset({"int8_mma", "generic"}),
}


def test_registry_preserves_the_previous_capability_matrix() -> None:
    registrations = {item.name: item for item in iter_engine_registrations()}
    assert set(registrations) == set(_EXPECTED_PROVIDES)
    for name, provides in _EXPECTED_PROVIDES.items():
        assert registrations[name].provides == provides, name
    # Derived matrix view stays consistent for legacy consumers.
    assert engines_providing_all(["int8_mma"]) == [
        "cuda_sm89",
        "cute_dsl",
        "cutlass",
        "ptx_sm89",
        "reference",
        "tilelang",
        "torch_int_mm",
        "triton",
    ]


def test_unwired_engines_are_honestly_nondispatchable() -> None:
    for name in ("ptx_sm89", "cuda_sm89", "native_sm89", "cutlass", "cute_dsl", "cutile"):
        registration = get_engine_registration(name)
        assert registration is not None, name
        assert registration.dispatchable is False, name
    for name in ("triton", "tilelang", "torch", "torch_int_mm", "reference"):
        registration = get_engine_registration(name)
        assert registration is not None, name
        assert registration.dispatchable is True, name


def test_min_capability_and_maturity_labels() -> None:
    expected: dict[str, tuple[int | None, str]] = {
        "ptx_sm89": (89, "executable"),
        "cuda_sm89": (89, "executable"),
        "tilelang": (80, "executable"),
        "cutlass": (80, "metadata_only"),
        "cute_dsl": (90, "reference_guarded"),
        "cutile": (90, "reference_guarded"),
        "triton": (None, "executable"),
        "torch": (None, "executable"),
    }
    for name, (min_capability, maturity) in expected.items():
        registration = get_engine_registration(name)
        assert registration is not None, name
        assert registration.min_capability == min_capability, name
        assert registration.maturity == maturity, name


def test_iter_engine_registrations_orders_by_priority() -> None:
    priorities = [item.priority for item in iter_engine_registrations()]
    assert priorities == sorted(priorities)
    first = next(iter_engine_registrations())
    assert isinstance(first, EngineRegistration)
    # C10: tilelang is the auto-chain primary (lowest priority number among dispatchable head)
    assert first.name == "tilelang"
    assert get_engine_registration("tilelang").priority < get_engine_registration(
        "triton"
    ).priority


def test_alias_resolved_registration_lookup() -> None:
    # native_sm89 is an alias of ptx_sm89; lookup resolves through the alias.
    aliased = get_engine_registration("native_sm89")
    assert aliased is not None
    assert aliased.name == "ptx_sm89"
    assert get_engine_registration("ptx_sm89") is not None
    assert get_engine_registration("nonexistent") is None
    assert get_engine_registration(None) is None


def test_resolve_compute_engine_matches_legacy_entry() -> None:
    compute_config = {
        "modules": [
            {
                "name": "int8_gemm",
                "required_capabilities": ["int8_mma"],
                "preferred_engines": [],
            }
        ]
    }
    unified = resolve_compute_engine(compute_config)
    legacy = engine_resolve_from_compute_config(compute_config)
    assert unified.engine == legacy.engine
    assert unified.candidates == legacy.candidates
    registration = unified.metadata.get("engine_registration")
    assert registration is not None
    assert set(registration) == {
        "maturity",
        "dispatchable",
        "min_capability",
        "priority",
    }


def test_resolve_compute_engine_merges_explicit_preference() -> None:
    compute_config = {
        "modules": [
            {
                "name": "int8_gemm",
                "required_capabilities": ["int8_mma"],
                "preferred_engines": [],
            }
        ]
    }
    result = resolve_compute_engine(
        compute_config,
        preferred_engines=["tilelang"],
    )
    assert result.engine == "tilelang"
    assert result.reason == "preferred"


def test_resolve_compute_engine_defaults_match_int8_path() -> None:
    unified = resolve_compute_engine(None)
    int8_path = resolve_int8_mma_engine("auto")
    assert unified.engine == int8_path.engine
    assert unified.engine == "tilelang"


def test_primary_kernel_maps_onto_engine_registry() -> None:
    from xqt.runtime.engine_resolve import map_primary_kernel_to_engine

    assert map_primary_kernel_to_engine("tilelang_fp4") == "tilelang"
    assert map_primary_kernel_to_engine("w8a8_int8_mma") == "tilelang"
    assert map_primary_kernel_to_engine("dequant_fp16") == "torch"
    assert map_primary_kernel_to_engine("triton") == "triton"
    reg = get_engine_registration(map_primary_kernel_to_engine("w8a8_int8_mma"))
    assert reg is not None
    assert reg.name == "tilelang"
    assert "int8_mma" in reg.provides


def test_c10_cutile_cute_cutlass_not_pretend_executable_auto_head() -> None:
    """U12: cutile/cute_dsl/cutlass stay non-dispatchable; auto head is tilelang."""

    for name, maturity in (
        ("cutlass", "metadata_only"),
        ("cute_dsl", "reference_guarded"),
        ("cutile", "reference_guarded"),
    ):
        registration = get_engine_registration(name)
        assert registration is not None, name
        assert registration.maturity == maturity, name
        assert registration.dispatchable is False, name
    auto = resolve_compute_engine(None)
    assert auto.engine == "tilelang"
    assert auto.engine not in {"cutlass", "cute_dsl", "cutile"}


def test_hadamard_groupwise_provided_by_tilelang_and_triton() -> None:
    """V3: C6 online kernel capability is declared on fusion DSL + portable line."""

    from xqt.runtime.engine_resolve import engines_providing

    providers = engines_providing("hadamard_groupwise")
    assert "tilelang" in providers
    assert "triton" in providers
