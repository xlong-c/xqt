"""Flux2 Klein example API surface smoke test."""

from __future__ import annotations


def test_flux2_klein_model_api_imports() -> None:
    from examples.xqt_models.flux2_klein import (
        load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit,
        load_flux2_klein_bf16_pipeline,
        load_flux2_klein_bf16_transformer,
        quantize_flux2_klein_bf16_pipeline_to_convrot_4bit,
        quantize_flux2_klein_bf16_transformer_to_convrot_4bit,
        run_flux2_klein_bf16_convrot_4bit_inference,
    )

    assert load_and_quantize_flux2_klein_bf16_pipeline_to_convrot_4bit is not None
    assert load_flux2_klein_bf16_pipeline is not None
    assert load_flux2_klein_bf16_transformer is not None
    assert quantize_flux2_klein_bf16_pipeline_to_convrot_4bit is not None
    assert quantize_flux2_klein_bf16_transformer_to_convrot_4bit is not None
    assert run_flux2_klein_bf16_convrot_4bit_inference is not None
