import pytest

from xqt.kernels.wrappers import build_profiling_plan, recommend_precision_strategy


class TestPrecisionAdvisor:
    def test_linear_prefers_fp8_on_ada_or_newer_when_low_precision_is_requested(self) -> None:
        recommendation = recommend_precision_strategy(
            operator_family="linear",
            target_sm="sm_89",
            prefer_low_precision=True,
        )

        assert recommendation.recommended_precision == "fp8"
        assert recommendation.fallback_precision == "bf16"
        assert recommendation.engine == "triton"
        assert recommendation.hardware_native is True
        assert "fp8" in recommendation.alternatives

    def test_attention_prefers_bf16_for_supported_nvidia_targets(self) -> None:
        recommendation = recommend_precision_strategy(
            operator_family="attn",
            target_sm="sm_90",
        )

        assert recommendation.recommended_precision == "bf16"
        assert recommendation.fallback_precision == "fp16"
        assert recommendation.validation["tighten_runtime_validation"] is True

    def test_norm_prefers_fp16_for_current_builtin_runtime_path(self) -> None:
        recommendation = recommend_precision_strategy(
            operator_family="norm",
            target_sm="sm_90",
        )

        assert recommendation.recommended_precision == "fp16"
        assert recommendation.fallback_precision == "fp16"
        assert recommendation.engine == "triton"
        assert "bf16" in recommendation.alternatives
        assert recommendation.validation["tighten_runtime_validation"] is True

    def test_megakernel_avoids_low_precision_as_the_first_step(self) -> None:
        recommendation = recommend_precision_strategy(
            operator_family="megakernel",
            target_sm="sm_120",
            prefer_low_precision=False,
        )

        assert recommendation.recommended_precision == "fp16"
        assert "occupancy" in " ".join(recommendation.risks).lower()

    def test_conflicting_precision_preferences_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            recommend_precision_strategy(
                operator_family="linear",
                target_sm="sm_89",
                prefer_low_precision=True,
                prioritize_accuracy=True,
            )


class TestProfilingPlanAdvisor:
    def test_memory_bandwidth_plan_requests_ncu_memory_views(self) -> None:
        plan = build_profiling_plan(
            operator_family="linear",
            target_sm="sm_89",
            bottleneck="memory_bandwidth",
        )

        assert "benchmark" in plan.tools
        assert "ncu" in plan.tools
        assert "MemoryWorkloadAnalysis" in plan.report_names
        assert "Roofline" in plan.report_names
        assert any("coalescing" in action for action in plan.suggested_actions)

    def test_launch_overhead_plan_starts_from_nsys(self) -> None:
        plan = build_profiling_plan(
            operator_family="fusion",
            target_sm="sm_90",
            bottleneck="launch_overhead",
        )

        assert "nsys" in plan.tools
        assert "cuda_gpu_kern_sum" in plan.report_names
        assert any("fusion" in action.lower() for action in plan.suggested_actions)

    def test_attention_numeric_stability_plan_mentions_reference_comparison(self) -> None:
        plan = build_profiling_plan(
            operator_family="attn",
            target_sm="sm_90",
            bottleneck="numeric_stability",
        )

        assert "benchmark" in plan.tools
        assert any("reference" in area.lower() for area in plan.focus_areas)
        assert any("softmax" in action.lower() for action in plan.suggested_actions)
