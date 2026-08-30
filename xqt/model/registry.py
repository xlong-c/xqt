"""Registry for declarative model compatibility profiles."""

from __future__ import annotations

from xqt.core.base.errors import XQTConfigError
from xqt.core.imports import resolve_target

from .config import ModelProfile
from .adapter import ModelAdapter

_PROFILES: dict[str, ModelProfile] = {}


def _register_builtin_profiles() -> None:
    """Register profiles for model loaders shipped with the examples."""

    builtins = (
        ModelProfile(
            profile_id="hf.hunyuan-ocr",
            family="multimodal",
            loader_target="xqt.model.hunyuan_ocr.load_hunyuan_ocr",
            loader_params={"repo_id": "tencent/HunyuanOCR"},
            inference_adapter="tensor",
            metadata={"source": "huggingface", "repo_id": "tencent/HunyuanOCR"},
        ),
        ModelProfile(
            profile_id="hf.unlimited-ocr",
            family="multimodal",
            loader_target="xqt.model.unlimited_ocr.load_unlimited_ocr",
            loader_params={"repo_id": "baidu/Unlimited-OCR"},
            inference_adapter="tensor",
            metadata={"source": "huggingface", "repo_id": "baidu/Unlimited-OCR"},
        ),
        ModelProfile(
            profile_id="diffusers.wan21-vae",
            family="diffusion",
            loader_target="xqt.model.wan21.load.load_wan21_vae",
            loader_params={"repo_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
            inference_adapter="tensor",
            metadata={"source": "huggingface", "repo_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
        ),
        ModelProfile(
            profile_id="diffusers.flux2-klein",
            family="diffusion",
            loader_target="xqt.model.flux2_klein.load.load_flux2_klein_bf16_transformer",
            loader_params={"repo_id": "black-forest-labs/FLUX.2-klein-4b"},
            inference_adapter="tensor",
            metadata={"source": "huggingface", "repo_id": "black-forest-labs/FLUX.2-klein-4b"},
        ),
    )
    _PROFILES.update({profile.profile_id: profile for profile in builtins})


_register_builtin_profiles()


def register_model_profile(
    profile: ModelProfile,
    *,
    replace: bool = False,
) -> None:
    """Register a reusable model profile by its stable identifier."""

    profile_id = profile.profile_id.strip() if isinstance(profile, ModelProfile) else ""
    if not profile_id:
        raise XQTConfigError("model profile must be a ModelProfile")
    if profile_id in _PROFILES and not replace:
        raise XQTConfigError(f"model profile {profile_id!r} is already registered")
    _PROFILES[profile_id] = profile


def resolve_model_profile(profile_id: str) -> ModelProfile:
    """Resolve a registered profile or raise a configuration error."""

    if not isinstance(profile_id, str) or not profile_id.strip():
        raise XQTConfigError("model.profile must be a non-empty string")
    try:
        return _PROFILES[profile_id.strip()]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES)) or "<none>"
        raise XQTConfigError(
            f"unknown model profile {profile_id!r}; registered profiles: {supported}"
        ) from exc


def resolve_model_adapter(profile: ModelProfile) -> ModelAdapter | None:
    """Instantiate the concrete adapter selected by a model profile."""

    if not profile.adapter_target:
        return None
    target = resolve_target(profile.adapter_target)
    adapter = target() if isinstance(target, type) else target
    if not isinstance(adapter, ModelAdapter):
        raise XQTConfigError(
            f"model adapter {profile.adapter_target!r} must implement ModelAdapter"
        )
    return adapter


def model_profile_names() -> tuple[str, ...]:
    """Return registered profile identifiers in stable order."""

    return tuple(sorted(_PROFILES))


__all__ = [
    "model_profile_names",
    "resolve_model_adapter",
    "register_model_profile",
    "resolve_model_profile",
]
